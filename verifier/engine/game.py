"""
Binding game: per obligation, bind one sufficient method to real evidence.

Per obligation:

0. For an equivalent final obligation, a process-local cache may supply
   decisive typed methods whose sufficiency a Judger already accepted. Only
   methods are reused: an adapter runs against the current candidate and its
   new evidence envelope is mechanically authenticated. Holds and refuted
   audits occupy separate slots and each must have been independently
   accepted; no adapter property implies the missing direction. The first
   candidate observed in a missing direction runs the complete game below,
   then may seed that direction for later candidates.
1. The oracle enumerates SUFFICIENT methods only (sufficiency is the
   oracle's own guarantee, researched first when the domain is
   unfamiliar), each with a signed strength declaration — decisive
   (a decision procedure for the obligation as written), bounded (a
   narrower instrument plus declared bridging assumptions), analytic
   (a disciplined reading with a witness trace) — and grades each with
   an assurance level of its own devising, ascending within a tier.
2. Execution is strictness-descending, enforced mechanically: decisive
   methods first; a weaker tier runs only after every stronger listed
   tier has been audited structurally unavailable (>=1 unobtainable
   attempt — degradation is an audited event, never a choice), or was
   never listed (an omission the judger audits). The discharge executor
   carries each method out faithfully (a missing instrument is
   installed at discharge time; a truly unobtainable one settles the
   claim at partial). Nothing below decisive may refute: a negative
   result from a bounded/analytic instrument converts to failed with
   the finding kept as a diagnostic.
   - The judger examines EVERY substantive result — holds and refuted
     alike: is the candidate method sufficient for the full obligation,
     was it faithfully executed against the right target, and was the
     promised evidence genuinely delivered? It opens the artifact itself
     and reruns when reproducibility is in doubt.
     A refutation gets no shortcut: it is the single most consequential
     outcome in the system (a refuted node refutes its claim — for the
     final, the whole certificate), so a counterexample settles only
     after the judger confirms it was sufficiently and faithfully obtained.
     An insufficient method is rejected immediately. Unfaithful -> the
     judger's reason goes back to the executor for a redo and the redone
     result is re-judged BLIND. A loop of at most three rounds.
3. An exhausted batch goes back to the oracle for at most one repair batch;
   offline, this also covers an initially empty batch so the oracle must recheck
   frozen-input experiments and retained effects. Duplicate procedures never
   run again.
4. All candidates exhausted -> failed (or unobtainable when a missing
   instrument blocked the way), with every attempt on record.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..llm.agents import ENUMERATE_ORACLES_AGENT, JUDGER_AGENT
from ..execution.discharge import DischargeResult
from ..execution.adapters import (
    AdapterInvocation, AdapterNotApplicable, adapter_method_view,
)
from ..core.events import EventLog
from ..execution.instruments import (
    ArtifactStore, authoritative_task_context, world_evidence_context,
    world_map,
)
from ..core.model import (
    BASIS_RECORD, BASIS_WORLD, Obligation, basis_satisfies, content_hash,
    OUTCOME_HOLDS, OUTCOME_REFUTED, OUTCOME_FAILED, OUTCOME_UNOBTAINABLE,
    PROVENANCE_DELIVERED, ROLE_SUBJECT,
    STRENGTH_ANALYTIC, STRENGTH_DECISIVE, STRENGTHS, strength_rank,
    WITNESS_NONE, witness_admissible, circular_inputs,
)
from ..llm.schema import RoleError
from .method_cache import AcceptedMethod, build_method_cache_key


def _method_strength(method: Dict[str, str]) -> str:
    """The method's declared strength, failing closed to analytic when the
    declaration is absent or unrecognized (mirrors basis -> record: an
    undeclared method must not execute as if it were a decision
    procedure)."""
    strength = method.get("strength", STRENGTH_ANALYTIC)
    return strength if strength in STRENGTHS else STRENGTH_ANALYTIC


def _tier_order(method: Dict[str, str]) -> int:
    """Sort key for strictness-descending execution; sorted() is stable,
    so the oracle's cost-ascending order survives within a tier."""
    return -strength_rank(_method_strength(method))


@dataclass
class Attempt:
    method: Dict[str, str]
    rejected: Optional[str] = None          # reason, if the method was rejected
    discharge_artifact: Optional[str] = None
    discharge_outcome: Optional[str] = None
    audit: Optional[Dict[str, str]] = None  # judger's first verdict
    redos: List[Dict] = None                # one {reason, outcome, artifact,
                                            # audit} per redo round

    def __post_init__(self):
        if self.redos is None:
            self.redos = []


class BindingGame:
    def __init__(
        self,
        backend,
        artifacts: ArtifactStore,
        log: EventLog,
        discharger,
        oracles_role=ENUMERATE_ORACLES_AGENT,
        judger_role=JUDGER_AGENT,
        redo_rounds: int = 3,
        registry: str = "[]",
        memory=None,
        live_world: bool = True,
        judger_error_retries: int = 1,
        method_cache=None,
    ):
        self.backend = backend
        self.artifacts = artifacts
        self.log = log
        self.discharger = discharger
        self.oracles_role = oracles_role
        self.judger_role = judger_role
        self.redo_rounds = redo_rounds
        # AgentRole already gives one malformed generation a schema-fix
        # re-prompt.  If that whole role call still fails (transport, parse,
        # or schema), retry only the independent Judger against the exact same
        # executor artifact.  This budget is deliberately separate from
        # ``redo_rounds``: an infrastructure failure is never feedback that
        # authorizes re-executing the bound method.
        self.judger_error_retries = max(0, int(judger_error_retries))
        self.registry = registry
        self.memory = memory  # advisory; feeds the oracle's registry slot
        self.live_world = live_world
        # Optional because custom/test BindingGames historically have no
        # cross-candidate sharing.  The shipped online/offline kernel injects
        # the process-local cache explicitly.
        self.method_cache = method_cache

    # Entry point
    def run(self, row: Dict, obligation: Obligation) -> DischargeResult:
        if self.method_cache is None:
            return self._run_uncached(row, obligation)
        key = self._method_cache_key(row, obligation)
        if key is None:
            return self._run_uncached(row, obligation)

        lease = self.method_cache.acquire(key)
        if lease.entries:
            result, missing, reason = self._run_cached_entries(
                row, obligation, lease.entries,
                key=key, waited=lease.waited)
            if result is not None:
                return result
            self.log.append("method_cache_bypass", {
                "claim": row["id"], "obligation": obligation.id,
                "key": key[:16], "reason": reason[:500],
            })
            if missing is not None:
                method, fresh = missing
                direction_lease = self.method_cache.acquire(
                    key, direction=fresh.outcome)
                if direction_lease.entry is not None:
                    accepted = direction_lease.entry
                    if accepted.method == method:
                        reused, reuse_reason = self._settle_cached_result(
                            row, obligation, accepted, fresh, key=key,
                            waited=(lease.waited or direction_lease.waited))
                    else:
                        reused, reuse_reason = self._run_cached(
                            row, obligation, accepted, key=key,
                            waited=(lease.waited or direction_lease.waited))
                    if reused is not None:
                        return reused
                    self.log.append("method_cache_bypass", {
                        "claim": row["id"], "obligation": obligation.id,
                        "key": key[:16], "reason": reuse_reason[:500],
                    })
                return self._run_uncached_and_cache(
                    row, obligation, key, lease=direction_lease,
                    direction=fresh.outcome)
            return self._run_uncached_and_cache(row, obligation, key)
        if not lease.leader:
            # The concurrent leader found no eligible accepted method.  Do
            # not serialize another cold fill; proceed independently.  A
            # successful follower may still seed later candidates.
            return self._run_uncached_and_cache(row, obligation, key)

        return self._run_uncached_and_cache(
            row, obligation, key, lease=lease)

    def _run_uncached_and_cache(
        self, row, obligation, key, *, lease=None, direction=None,
    ):
        """Run the complete game and publish only its authenticated direction.

        ``direction`` scopes an opposite-outcome single flight. A full game
        that unexpectedly settles the other direction may still seed that
        actual direction, but it never fills or wakes the requested slot as
        if the missing certificate had been obtained.
        """
        accepted = fill = None
        try:
            result = self._run_uncached(row, obligation)
            # The authority files are normally immutable.  Re-key anyway so
            # a live edit during the cold fill can never publish under the
            # earlier task identity.
            if self._method_cache_key(row, obligation) == key:
                accepted = self._accepted_method(row, obligation, result)
                if accepted is not None:
                    if lease is not None and lease.leader and (
                            direction is None
                            or accepted.source_outcome == direction):
                        fill = accepted
                    else:
                        self.method_cache.publish_if_absent(key, accepted)
            return result
        finally:
            if lease is not None:
                lease.finish(fill)

    def _method_cache_key(self, row, obligation):
        state = os.path.dirname(os.path.abspath(self.log.path))
        catalog = getattr(self.discharger, "adapter_catalog", lambda: "[]")()
        return build_method_cache_key(
            state_dir=state,
            row=row,
            obligation=obligation,
            backend=self.backend,
            oracles_role=self.oracles_role,
            judger_role=self.judger_role,
            live_world=self.live_world,
            adapter_catalog=catalog,
        )

    def _run_cached_entries(
        self, row, obligation, entries, *, key, waited=False,
    ):
        """Try each distinct cached MethodIR without crossing signatures.

        Returns ``(settled, missing_observation, reason)``. The observation is
        ``(method, fresh_result)`` only when its substantive direction has no
        independently accepted slot. If two direction slots contain the same
        exact MethodIR, one fresh adapter execution can select the matching
        certificate without running that adapter twice.
        """
        directions = {entry.source_outcome for entry in entries}
        attempted_methods = []
        reasons = []
        missing = None
        for accepted in entries:
            if any(accepted.method == method for method in attempted_methods):
                continue
            attempted_methods.append(accepted.method)
            fresh, reason = self._execute_cached_method(
                row, obligation, accepted, key=key)
            if fresh is None:
                reasons.append(reason)
                continue
            matching = next((entry for entry in entries
                             if entry.source_outcome == fresh.outcome
                             and entry.method == accepted.method), None)
            if matching is not None:
                result, reason = self._settle_cached_result(
                    row, obligation, matching, fresh,
                    key=key, waited=waited)
                if result is not None:
                    return result, None, ""
                reasons.append(reason)
                continue
            reasons.append(
                "fresh adapter outcome has no independently accepted "
                f"signature for this MethodIR: fresh={fresh.outcome}")
            if fresh.outcome not in directions and missing is None:
                missing = (accepted.method, fresh)
        return None, missing, "; ".join(reasons) or "no reusable MethodIR"

    def _run_cached(self, row, obligation, accepted, *, key, waited=False):
        fresh, reason = self._execute_cached_method(
            row, obligation, accepted, key=key)
        if fresh is None:
            return None, reason
        return self._settle_cached_result(
            row, obligation, accepted, fresh, key=key, waited=waited)

    def _execute_cached_method(self, row, obligation, accepted, *, key):
        adapter_only = getattr(
            self.discharger, "discharge_adapter_only", None)
        if adapter_only is None:
            return None, "discharger has no authenticated adapter-only path"
        method = accepted.method
        if not getattr(self.discharger, "supports_method_reuse",
                       lambda _method: False)(method):
            return None, "adapter requires a fresh independent Judger audit"
        if not self._cacheable_signature(method, {
                "sufficient": "yes", "faithful": "yes",
                "degradation": "n/a"}):
            return None, "cached MethodIR no longer meets reuse policy"
        result, reason = adapter_only(row, obligation, method)
        if result is None:
            return None, reason
        if result.outcome not in (OUTCOME_HOLDS, OUTCOME_REFUTED):
            return None, "fresh adapter execution was not substantive"
        if accepted.source_outcome not in (OUTCOME_HOLDS, OUTCOME_REFUTED):
            return None, "cached method has no valid source outcome"
        if not basis_satisfies(method.get("basis", BASIS_RECORD),
                               obligation.required_basis):
            return None, "cached method is below the current evidence floor"
        structural = self._structural_defect(result, obligation)
        if structural:
            return None, structural
        if self._method_cache_key(row, obligation) != key:
            return None, "task authority changed during adapter execution"

        return result, ""

    def _settle_cached_result(
        self, row, obligation, accepted, result, *, key, waited=False,
    ):
        """Bind fresh evidence only to its exact method/direction signature."""
        method = accepted.method
        if accepted.source_outcome not in (OUTCOME_HOLDS, OUTCOME_REFUTED):
            return None, "cached method has no valid source outcome"
        if result.outcome != accepted.source_outcome:
            return None, (
                "fresh adapter outcome changed direction: "
                f"source={accepted.source_outcome}, fresh={result.outcome}"
            )
        if not getattr(self.discharger, "supports_method_reuse",
                       lambda _method: False)(method):
            return None, "adapter requires a fresh independent Judger audit"
        if not self._cacheable_signature(method, {
                "sufficient": "yes", "faithful": "yes",
                "degradation": "n/a"}):
            return None, "cached MethodIR no longer meets reuse policy"
        if not basis_satisfies(method.get("basis", BASIS_RECORD),
                               obligation.required_basis):
            return None, "cached method is below the current evidence floor"
        structural = self._structural_defect(result, obligation)
        if structural:
            return None, structural
        if self._method_cache_key(row, obligation) != key:
            return None, "task authority changed while awaiting direction"

        audit = {
            "sufficient": "yes",
            "faithful": "yes",
            "degradation": "n/a",
            "reason": (
                "reused an independent Judger sufficiency signature; the "
                "typed adapter was re-executed and its new evidence envelope "
                "was mechanically authenticated"
            ),
            "cache": {
                "schema": "gv-accepted-method-reuse-v1",
                "source_binding_artifact": accepted.source_binding_artifact,
                "source_discharge_artifact": accepted.source_discharge_artifact,
                "source_outcome": accepted.source_outcome,
                "source_audit": accepted.audit,
                "cache_key": key,
            },
        }
        attempt = Attempt(
            method=method,
            discharge_artifact=result.artifact,
            discharge_outcome=result.outcome,
            audit=audit,
        )
        self.log.append("method_cache_hit", {
            "claim": row["id"], "obligation": obligation.id,
            "procedure": method["procedure"][:80],
            "source_binding_artifact": accepted.source_binding_artifact,
            "fresh_discharge_artifact": result.artifact,
            "source_outcome": accepted.source_outcome,
            "outcome": result.outcome,
            "waited_for_single_flight": waited,
        })
        return self._settle(
            row, obligation, [attempt], method, result, audit=audit), ""

    @staticmethod
    def _cacheable_signature(method, audit) -> bool:
        required = {"procedure", "artifact", "level", "basis", "strength",
                    "bridges", "adapter", "adapter_version",
                    "adapter_arguments", "adapter_inputs"}
        if set(method) != required:
            return False
        if (method.get("basis") != BASIS_WORLD
                or _method_strength(method) != STRENGTH_DECISIVE
                or method.get("bridges") != []
                or audit.get("sufficient") != "yes"
                or audit.get("faithful") != "yes"
                or audit.get("degradation") != "n/a"):
            return False
        try:
            invocation = AdapterInvocation.from_method(method)
        except AdapterNotApplicable:
            return False
        return bool(invocation and invocation.inputs) and all(
            item.get("provenance") == PROVENANCE_DELIVERED
            and item.get("role") == ROLE_SUBJECT
            for item in invocation.inputs
        )

    def _accepted_method(self, row, obligation, settled):
        """Extract one authenticated, cache-eligible binding artifact."""
        if settled.outcome not in (OUTCOME_HOLDS, OUTCOME_REFUTED) \
                or settled.assumed_premises:
            return None
        try:
            binding = self.artifacts.get(settled.artifact)
            if content_hash(binding) != settled.artifact \
                    or binding.get("instrument") != "binding-game" \
                    or binding.get("obligation") != obligation.text:
                return None
            bound = binding.get("bound")
            if not isinstance(bound, dict):
                return None
            method, audit = bound.get("method"), bound.get("audit")
            if not isinstance(method, dict) or not isinstance(audit, dict) \
                    or not self._cacheable_signature(method, audit):
                return None
            if not getattr(self.discharger, "supports_method_reuse",
                           lambda _method: False)(method):
                return None
            discharge_digest = bound.get("discharge_artifact")
            evidence = self.artifacts.get(discharge_digest)
            execution = evidence.get("execution")
            if not isinstance(execution, dict):
                return None
            inner = DischargeResult(
                outcome=execution.get("outcome"),
                witness=execution.get("witness"),
                artifact=discharge_digest,
                instrument=evidence.get("instrument", ""),
                witness_kind=execution.get("witness_kind"),
                inputs=tuple(execution.get("declared_inputs") or ()),
            )
            validator = getattr(
                getattr(self.discharger, "registry", None),
                "validate_evidence", None)
            if validator is None:
                return None
            valid, _reason = validator(
                method, row, obligation, inner, self.artifacts)
            if not valid or inner.outcome != settled.outcome:
                return None
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return None
        return AcceptedMethod(
            method=method,
            source_binding_artifact=settled.artifact,
            source_discharge_artifact=discharge_digest,
            source_outcome=inner.outcome,
            audit=audit,
        )

    def _run_uncached(self, row: Dict, obligation: Obligation) -> DischargeResult:
        attempts: List[Attempt] = []
        try:
            methods = self._enumerate(row, obligation)
        except RoleError as err:
            # A crashed role or schema-invalid generation is not evidence that
            # no sufficient method exists.  Keep generation failure distinct
            # from the oracle's valid, explicit empty-list conclusion.
            self.log.append(
                "methods_enumerated",
                {"claim": row["id"], "obligation": obligation.id,
                 "error": str(err)},
            )
            return self._fail(
                row, obligation, attempts,
                f"oracle enumeration failed: {err}",
            )

        initial_empty = not methods
        if initial_empty and self.live_world:
            # A live world may change, so preserve the existing fail-closed,
            # retryable result.  Offline gets the same single bounded repair as
            # an exhausted non-empty batch: a second look may notice a retained
            # effect or a fresh experiment over the frozen captured subject.
            return self._fail(
                row, obligation, attempts,
                "oracle found no executable sufficient method at the "
                f"required basis {obligation.required_basis!r}",
            )

        # the obligation's evidence floor is mechanical, not advisory: a
        # method whose declared basis cannot satisfy it never executes —
        # a record method faithfully run against a world-floor obligation
        # would only prove the solver's reporting, which is exactly the
        # evidence this floor exists to reject
        if not initial_empty:
            below = [m for m in methods
                     if not basis_satisfies(m.get("basis", BASIS_RECORD),
                                            obligation.required_basis)]
            for m in below:
                self._reject(row, obligation, m,
                             "declared basis below the obligation's floor")
            methods = [m for m in methods if m not in below]
            if not methods:
                return self._fail(
                    row, obligation, attempts,
                    "oracle protocol error: every returned method declares a "
                    "basis below the obligation's required basis "
                    f"{obligation.required_basis!r}",
                )

        # strictness-descending execution: decisive first, then bounded,
        # then analytic — a weaker tier runs only once every stronger
        # LISTED method has been audited structurally unavailable (the
        # deferral below); an unlisted stronger tier does not block, the
        # omission instead goes into the degradation record the judger
        # audits.
        methods.sort(key=_tier_order)

        unavailable_reasons: List[str] = []
        audited_gap = False   # an EXECUTOR-audited unobtainable is on record
        # (the empty-repair boundary declaration also lands in
        # unavailable_reasons but is weaker: it only settles unobtainable
        # when nothing failed — an oracle saying "no further method" must
        # not promote a pile of retryable failures)
        structural_feedback: List[Dict[str, str]] = []
        seen_procedures = set()
        deferred: List[Dict[str, str]] = []
        tier_listed = {_method_strength(m) for m in methods}
        tier_unobtainable: set = set()
        degradation_disputed = False
        contrary_signal = False   # a non-decisive instrument reported a
        # negative result: inadmissible as a refutation, but a live
        # contrary indicator — the obligation must not read out as
        # "everything that ran held" (partial) while it stands
        failed_seen = False
        repair_used = False
        method_index = 0
        while (method_index < len(methods) or deferred
               or ((attempts or initial_empty) and not repair_used)):
            if method_index >= len(methods):
                ready = [m for m in deferred
                         if not self._blocking(m, tier_listed,
                                               tier_unobtainable)]
                if ready:
                    # a stronger tier resolved to structurally unavailable
                    # after these were parked: they are executable now
                    deferred = [m for m in deferred if m not in ready]
                    ready.sort(key=_tier_order)
                    methods.extend(ready)
                    continue
                if not ((attempts or initial_empty) and not repair_used):
                    break
                # Hand an exhausted batch — including an offline initial empty
                # batch — back to the Oracle exactly once.  A different method
                # may notice a frozen-input experiment, retained effect, or
                # avoid a structural/instrument/execution/fidelity gap.  The
                # replacement still executes and is re-judged.
                repair_used = True
                feedback = json.dumps(
                    {
                        "initial_empty": initial_empty,
                        "structural_rejections": structural_feedback,
                        "unavailable_boundaries": list(dict.fromkeys(
                            unavailable_reasons
                        )),
                        "exhausted_attempts": [
                            _attempt_diagnostic(a) for a in attempts
                        ],
                        # parked, not rejected: these wait on a stronger
                        # tier's structural unavailability, so the repair
                        # must not re-list them
                        "deferred_methods": [
                            {"procedure": m.get("procedure", "")[:80],
                             "strength": _method_strength(m),
                             "waiting_on": self._blocking(
                                 m, tier_listed, tier_unobtainable)}
                            for m in deferred
                        ],
                    },
                    ensure_ascii=False,
                )
                try:
                    repaired = self._enumerate(
                        row, obligation, method_feedback=feedback
                    )
                except RoleError as err:
                    failed_seen = True
                    self.log.append(
                        "methods_enumerated",
                        {"claim": row["id"],
                         "obligation": obligation.id,
                         "repair": True,
                         "error": str(err)},
                    )
                    break

                focused_fallback_attempted = False
                if not repaired:
                    # A final-world acceptance obligation is load-bearing for
                    # candidate selection.  If both the broad first pass and
                    # the ordinary exhausted-batch repair return no method,
                    # give the Oracle one last *narrow* chance to enumerate a
                    # directionally decisive local counterexample.  This is
                    # semantic rather than task-specific: optimality,
                    # conditional coverage, universals, and exact equalities
                    # can often be refuted by one witness even when proving the
                    # positive direction would require exhaustive recovery.
                    #
                    # The pass still only ENUMERATES.  Its method must execute
                    # and survive the normal Judger, so the diagnostic flag can
                    # neither become evidence nor manufacture a refutation.
                    if (not self.live_world
                            and row.get("kind") == "final"
                            and obligation.required_basis == BASIS_WORLD):
                        focused_fallback_attempted = True
                        focused_feedback_payload = json.loads(feedback)
                        focused_feedback_payload.update({
                            "focused_counterexample_fallback": True,
                            "ordinary_repair_empty": True,
                            "focus_reason": (
                                "both broad enumeration and the bounded "
                                "ordinary repair returned methods=[]"
                            ),
                        })
                        focused_feedback = json.dumps(
                            focused_feedback_payload, ensure_ascii=False)
                        try:
                            repaired = self._enumerate(
                                row,
                                obligation,
                                method_feedback=focused_feedback,
                            )
                        except RoleError as err:
                            # An infrastructure/schema failure is not an
                            # evidence-availability finding.
                            failed_seen = True
                            self.log.append(
                                "methods_enumerated",
                                {"claim": row["id"],
                                 "obligation": obligation.id,
                                 "repair": True,
                                 "focused_counterexample_fallback": True,
                                 "error": str(err)},
                            )
                            break

                if not repaired:
                    if self.live_world:
                        failed_seen = True
                    elif focused_fallback_attempted:
                        unavailable_reasons.append(
                            "focused counterexample fallback and ordinary "
                            "oracle repair found no materially new sufficient "
                            "method at the fixed evidence boundary"
                        )
                    else:
                        unavailable_reasons.append(
                            "oracle repair found no materially new sufficient "
                            "method at the fixed evidence boundary"
                        )
                    continue

                added = 0
                for candidate in repaired:
                    if not basis_satisfies(
                        candidate.get("basis", BASIS_RECORD),
                        obligation.required_basis,
                    ):
                        self._reject(
                            row, obligation, candidate,
                            "declared basis below the obligation's floor",
                        )
                    elif candidate["procedure"] in seen_procedures:
                        self._reject(
                            row, obligation, candidate,
                            "duplicate procedure returned on method repair",
                        )
                    else:
                        methods.append(candidate)
                        tier_listed.add(_method_strength(candidate))
                        added += 1
                if added:
                    # the unexecuted tail keeps the strictness-descending
                    # invariant the initial sort established
                    methods[method_index:] = sorted(
                        methods[method_index:], key=_tier_order)
                if not added:
                    # A non-empty repair batch made entirely of low-floor or
                    # already-tried procedures is a protocol failure, not an
                    # evidence-availability finding.
                    failed_seen = True
                continue

            method = methods[method_index]
            method_index += 1
            strength = _method_strength(method)
            blocking = self._blocking(method, tier_listed, tier_unobtainable)
            if blocking:
                # degradation is an audited event, never a choice: this
                # method parks until every stronger listed tier has been
                # audited structurally unavailable
                deferred.append(method)
                self.log.append(
                    "method_deferred",
                    {"claim": row["id"], "obligation": obligation.id,
                     "procedure": method.get("procedure", "")[:80],
                     "strength": strength, "waiting_on": blocking},
                )
                continue
            procedure = method["procedure"]
            if procedure in seen_procedures:
                self._reject(
                    row, obligation, method,
                    "duplicate procedure returned by oracle",
                )
                continue
            seen_procedures.add(procedure)
            attempt = Attempt(method=method)
            attempts.append(attempt)

            result = self.discharger.discharge(
                row, obligation, method=method, tag="bound"
            )
            attempt.discharge_artifact = result.artifact
            attempt.discharge_outcome = result.outcome

            if result.outcome == OUTCOME_UNOBTAINABLE:
                detail = str(result.witness or
                             "required instrument or evidence unavailable")
                unavailable_reasons.append(detail)
                audited_gap = True
                tier_unobtainable.add(strength)
                reason = f"unobtainable: {detail}"
                self._reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue

            if result.outcome == OUTCOME_FAILED:
                failed_seen = True
                detail = str(result.witness or "")
                reason = "execution failure at retry limit"
                if detail:
                    reason += f": {detail}"
                self._reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue

            if result.outcome == OUTCOME_REFUTED \
                    and strength != STRENGTH_DECISIVE:
                # nothing below decisive may refute: the instrument's own
                # scope gap is exactly what a negative verdict cannot carry
                # across (a failing unit test does not prove the system-
                # level bug, and a passing one does not disprove it). The
                # negative finding stays on record as a diagnostic for the
                # repair pass — a decisive method may yet refute honestly.
                failed_seen = True
                contrary_signal = True
                reason = (f"refutation below decisive strength is "
                          f"inadmissible ({strength} instrument reported: "
                          f"{result.witness})")
                structural_feedback.append({
                    "procedure": procedure,
                    "reason": reason,
                })
                self._reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue

            structural = self._structural_defect(result, obligation)
            if structural:
                # witness / independence axioms, enforced before the
                # judger spends a call: an absence-only refutation or a
                # holds grounded on a delivered artifact as authority is
                # a method defect the repair pass may fix, never evidence
                failed_seen = True
                structural_feedback.append({
                    "procedure": procedure, "reason": structural})
                self._reject(row, obligation, method, structural)
                attempt.rejected = structural
                continue

            # every substantive outcome — holds AND refuted — faces the
            # judger before it settles, for both method sufficiency and
            # execution fidelity. A refutation is the most
            # consequential verdict in the system (a refuted node refutes
            # its claim — for the final, the whole certificate), so an
            # unaudited one is exactly the evidence we cannot afford to
            # accept.
            degradation = self._degradation_record(
                strength, attempts, tier_listed)
            verdict = self._judge(obligation, method, result, attempt,
                                  degradation=degradation)
            if strength != STRENGTH_DECISIVE \
                    and verdict.get("degradation") == "unjustified":
                # the third signature failed: the judger disputes that the
                # stronger tiers were genuinely unavailable, so this weaker
                # evidence may not settle — and the obligation's own
                # unobtainable story is disputed with it (fail-closed to
                # failed at aggregation)
                degradation_disputed = True
                if repair_used:
                    failed_seen = True
                reason = ("judger degradation unjustified: "
                          f"{verdict['reason']}")
                structural_feedback.append({
                    "procedure": procedure,
                    "reason": reason,
                })
                self._reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue
            if verdict.get("sufficient") == "no":
                # First-batch insufficiency is exactly what the bounded
                # Oracle repair can correct. A repaired method that is still
                # insufficient is the terminal protocol failure.
                if repair_used:
                    failed_seen = True
                structural_feedback.append({
                    "procedure": procedure,
                    "reason": verdict["reason"],
                })
                reason = f"judger method insufficient: {verdict['reason']}"
                self._reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue
            if (verdict.get("sufficient") == "yes"
                    and verdict["faithful"] == "yes"):
                return self._settle(row, obligation, attempts,
                                    method, result, audit=verdict)
            failed_seen = True

            # judger-discharger loop: an unfaithful delivery hands the
            # judger's reason back to the executor for a redo, and the
            # redone evidence is re-judged BLIND. At most redo_rounds rounds.
            # (a judger infrastructure failure never awards a redo)
            reason = None
            rounds = 0
            while (verdict.get("sufficient") == "yes"
                   and verdict["faithful"] == "no"
                   and not verdict.get("_infra")
                   and rounds < self.redo_rounds):
                rounds += 1
                result = self._redo(row, obligation, method, verdict,
                                    attempt, rounds)
                if result.outcome == OUTCOME_UNOBTAINABLE:
                    detail = str(result.witness or
                                 "required instrument or evidence unavailable")
                    unavailable_reasons.append(detail)
                    audited_gap = True
                    tier_unobtainable.add(strength)
                    reason = f"unobtainable (redo {rounds}): {detail}"
                    break
                if result.outcome == OUTCOME_FAILED:
                    failed_seen = True
                    detail = str(result.witness or "")
                    reason = f"execution failure at retry limit (redo {rounds})"
                    if detail:
                        reason += f": {detail}"
                    break
                if result.outcome == OUTCOME_REFUTED \
                        and strength != STRENGTH_DECISIVE:
                    failed_seen = True
                    contrary_signal = True
                    reason = (f"refutation below decisive strength is "
                              f"inadmissible (redo {rounds}; {strength} "
                              f"instrument reported: {result.witness})")
                    structural_feedback.append({
                        "procedure": procedure,
                        "reason": reason,
                    })
                    break
                structural = self._structural_defect(result, obligation)
                if structural:
                    failed_seen = True
                    reason = f"{structural} (redo {rounds})"
                    structural_feedback.append({
                        "procedure": procedure, "reason": reason})
                    break
                verdict = self._judge(obligation, method, result, attempt,
                                      redo=rounds, degradation=degradation)
                if strength != STRENGTH_DECISIVE \
                        and verdict.get("degradation") == "unjustified":
                    degradation_disputed = True
                    reason = (f"judger degradation unjustified "
                              f"(redo {rounds}): {verdict['reason']}")
                    structural_feedback.append({
                        "procedure": procedure,
                        "reason": reason,
                    })
                    break
                if verdict.get("sufficient") == "no":
                    structural_feedback.append({
                        "procedure": procedure,
                        "reason": verdict["reason"],
                    })
                    reason = (f"judger method insufficient (redo {rounds}): "
                              f"{verdict['reason']}")
                    break
                if (verdict.get("sufficient") == "yes"
                        and verdict["faithful"] == "yes"):
                    return self._settle(row, obligation, attempts,
                                        method, result, audit=verdict)
            if reason is None:
                reason = (f"judger after {rounds} redo round(s): "
                          f"{verdict['reason']}"
                          if rounds else f"judger: {verdict['reason']}")
            self._reject(row, obligation, method, reason)
            attempt.rejected = reason

        # terminal aggregation, one rule shared with the tier-unlock gate:
        # at least one audited structural gap and zero settled verdicts
        # settles unobtainable — a construction failure BESIDE an audited
        # gap is noise around an established blocker, not counter-evidence
        # to it (pure failures, with no unobtainable in sight, still fail:
        # a false statement's honest construction attempts also all fail).
        # A judger-disputed degradation withdraws that trust, and so does
        # a live contrary signal (a non-decisive instrument's negative
        # result): the structural story is contested or contradicted, so
        # fail closed to failed.
        unavailable_only = (
            (audited_gap
             or (bool(unavailable_reasons) and not failed_seen))
            and not degradation_disputed
            and not contrary_signal)
        reason = (
            "all candidate methods structurally unavailable: "
            + "; ".join(dict.fromkeys(unavailable_reasons))
            if unavailable_only else "all candidate methods rejected"
        )
        if deferred:
            reason += (f"; {len(deferred)} weaker-tier method(s) stayed "
                       "deferred behind a stronger tier never audited "
                       "structurally unavailable")
        return self._fail(row, obligation, attempts, reason,
                          unobtainable=unavailable_only)

    @staticmethod
    def _structural_defect(result: DischargeResult,
                           obligation: Obligation) -> Optional[str]:
        """The two axioms checked on the executor's own declaration,
        before any semantic audit: a refutation needs an admissible
        witness kind (positive observation; absence only against a
        presence requirement), and a holds must not rest on a delivered
        artifact standing as authority (the circular certificate)."""
        if result.outcome == OUTCOME_REFUTED and not witness_admissible(
                result.witness_kind, obligation.refutable_by_absence):
            kind = result.witness_kind or WITNESS_NONE
            return (f"refutation witness of kind {kind!r} is inadmissible: "
                    f"absence, silence, and undeclared witnesses never "
                    f"refute unless the obligation is a presence "
                    f"requirement (reported: {result.witness})")
        if result.outcome == "holds":
            circular = circular_inputs(result.inputs)
            if circular:
                return (f"circular evidence: delivered artifact(s) "
                        f"{circular} declared as authority; a holds "
                        f"grounded on the solver's own output establishes "
                        f"consistency, not truth")
        return None

    @staticmethod
    def _blocking(method: Dict[str, str], tier_listed: set,
                  tier_unobtainable: set) -> List[str]:
        """The stronger tiers this method is parked behind: every LISTED
        stronger tier not yet audited structurally unavailable (>=1
        unobtainable attempt; failed attempts do not clear a tier — nor
        do they hold it closed once an audited gap is on record). Empty
        means executable. An unlisted stronger tier never blocks: the
        omission itself goes into the degradation record for the judger
        to audit."""
        strength = _method_strength(method)
        return [tier for tier in STRENGTHS
                if strength_rank(tier) > strength_rank(strength)
                and tier in tier_listed and tier not in tier_unobtainable]

    def _degradation_record(self, strength: str, attempts: List[Attempt],
                            tier_listed: set) -> str:
        """What the judger audits before countersigning a non-decisive
        binding: every stronger-tier attempt with its terminal state, plus
        stronger tiers the oracle never listed (an omission is itself a
        signed declaration that no such method exists in principle)."""
        if strength == STRENGTH_DECISIVE:
            return "(none — decisive method, no degradation to audit)"
        stronger = [a for a in attempts
                    if strength_rank(_method_strength(a.method))
                    > strength_rank(strength)]
        return json.dumps(
            {
                "stronger_attempts": [
                    {"tier": _method_strength(a.method),
                     "procedure": a.method.get("procedure", "")[:80],
                     "outcome": a.discharge_outcome,
                     "terminal_reason": a.rejected}
                    for a in stronger
                ],
                "omitted_tiers": [
                    tier for tier in STRENGTHS
                    if strength_rank(tier) > strength_rank(strength)
                    and tier not in tier_listed
                ],
            },
            ensure_ascii=False,
        )

    # enumeration and advisory review
    def _enumerate(
        self,
        row: Dict,
        obligation: Obligation,
        method_feedback: str = "(none)",
    ) -> List[Dict[str, str]]:
        registry = (self.memory.registry_summary()
                    if self.memory is not None else self.registry)
        catalog = getattr(self.discharger, "adapter_catalog", lambda: "[]")()
        state = os.path.dirname(os.path.abspath(self.log.path))
        world_mode, world, manifest = world_evidence_context(
            state, live_world=self.live_world)
        payload = self.oracles_role.call(
            self.backend,
            {
                "claim_text": row["text"],
                "obligation_text": obligation.text,
                "required_basis": obligation.required_basis,
                "severity": obligation.severity,
                "clause": obligation.clause,
                "refutable_by_absence": obligation.refutable_by_absence,
                "task_context": authoritative_task_context(state),
                "features": _features(row),
                "registry": registry,
                "adapter_catalog": catalog,
                "method_feedback": method_feedback,
                "ledger_path": os.path.abspath(self.log.path),
                "artifact_root": os.path.abspath(self.artifacts.root),
                "world_mode": world_mode,
                "world_path": world,
                "world_manifest": manifest,
                "world_map": world_map(state, live_world=self.live_world),
            },
        )
        methods = payload["methods"]
        event = {
            "claim": row["id"],
            "obligation": obligation.id,
            "repair": method_feedback != "(none)",
            "methods": [{"procedure": m["procedure"][:80],
                         "level": m.get("level", ""),
                         "basis": m.get("basis", ""),
                         "strength": _method_strength(m),
                         **({"adapter": m["adapter"],
                             "adapter_version": m.get("adapter_version")}
                            if "adapter" in m else {})}
                        for m in methods],
        }
        if method_feedback != "(none)":
            try:
                parsed_feedback = json.loads(method_feedback)
            except (json.JSONDecodeError, TypeError):
                parsed_feedback = {}
            if (isinstance(parsed_feedback, dict)
                    and parsed_feedback.get(
                        "focused_counterexample_fallback") is True):
                event["focused_counterexample_fallback"] = True
        self.log.append("methods_enumerated", event)
        return methods

    # the judger
    def _judge(self, obligation, method, result, attempt,
               redo: int = 0,
               degradation: str = "(none — decisive method, no degradation "
                                  "to audit)") -> Dict[str, str]:
        evidence = {
            "outcome": result.outcome,
            "witness": result.witness,
            # the executor's own classification of its refutation witness
            # and of every input it consumed — the judger audits both:
            # a mislabelled absence or a delivered artifact hidden under
            # another provenance class is unfaithful
            "witness_kind": result.witness_kind,
            "inputs": list(result.inputs),
            "artifact_digest": result.artifact,
            "artifact_path": os.path.join(
                self.artifacts.root, result.artifact + ".json"
            ),
            # premise claim ids whose content the executor declared it
            # ASSUMED (workspace borrowing without re-establishment) —
            # part of the fidelity audit: undeclared reliance on prior
            # artifacts or premise content is unfaithful
            "assumed_premises": list(result.assumed_premises),
        }
        # same state-dir derivation as the kernel: the judger audits
        # refutations against the delivered world and its capture
        # manifest, so it needs the same paths the executor had
        state = os.path.dirname(os.path.abspath(self.log.path))
        world_mode, world, manifest = world_evidence_context(
            state, live_world=self.live_world)
        variables = {
            "obligation_text": obligation.text,
            "severity": obligation.severity,
            "clause": obligation.clause,
            "refutable_by_absence": obligation.refutable_by_absence,
            "procedure": method["procedure"],
            "adapter_method": adapter_method_view(method),
            "promised_artifact": method.get("artifact", ""),
            # the oracle's declared evidence basis — the judger audits it as
            # part of fidelity, and it is what the settled verdict will carry
            # downstream
            "basis": method.get("basis", ""),
            # the strength ladder's three inputs: sufficiency is judged AT the
            # declared tier, bridges must close the gap, and a non-decisive
            # binding needs the third signature over the degradation record
            "declared_strength": _method_strength(method),
            "bridges": json.dumps(method.get("bridges", []),
                                  ensure_ascii=False),
            "degradation_record": degradation,
            "task_context": authoritative_task_context(state),
            "ledger_path": os.path.abspath(self.log.path),
            "artifact_root": os.path.abspath(self.artifacts.root),
            "world_mode": world_mode,
            "world_path": world,
            "world_manifest": manifest,
            "world_map": world_map(state, live_world=self.live_world),
            "evidence": json.dumps(evidence, ensure_ascii=False),
        }
        total_calls = self.judger_error_retries + 1
        for call_no in range(1, total_calls + 1):
            try:
                verdict = self.judger_role.call(self.backend, variables)
                break
            except RoleError as err:
                will_retry = call_no < total_calls
                # This is the error audit.  Backends retain ownership of usage
                # accounting (CodexBackend records exact usage before parsing
                # its final JSON), so do not synthesize or duplicate token
                # rows here.
                self.log.append(
                    "judger_call_failed",
                    {
                        "obligation": obligation.id,
                        "redo": redo,
                        "procedure": method["procedure"][:80],
                        "artifact_digest": result.artifact,
                        "call": call_no,
                        "max_calls": total_calls,
                        "will_retry": will_retry,
                        "error": str(err)[:500],
                    },
                )
                if will_retry:
                    continue
                # fail-closed; the _infra marker keeps a persistent Judger
                # infrastructure failure from awarding itself an executor
                # redo.  The original digest remains in Attempt and in the
                # final binding-game artifact for later audit/recovery.
                verdict = {
                    # Internal fail-closed sentinel; public role output permits
                    # only yes/no, so neither signature can accidentally pass.
                    "sufficient": "unknown",
                    "faithful": "no",
                    "degradation": "n/a",
                    "reason": f"judger failed: {err}",
                    "_infra": True,
                }
        if redo:
            attempt.redos[-1]["audit"] = verdict
        else:
            attempt.audit = verdict
        public = {k: v for k, v in verdict.items() if not k.startswith("_")}
        self.log.append(
            "judger_verdict",
            {"obligation": obligation.id, "redo": redo,
             "procedure": method["procedure"][:80], **public},
        )
        return verdict

    def _redo(self, row, obligation, method, verdict, attempt, round_no: int):
        """
        Re-execute the unchanged bound method with every judger reason supplied
        separately as untrusted diagnostic feedback (latest last); the result
        goes back for a blind re-judge.
        """
        
        self.log.append(
            "method_redo",
            {"claim": row["id"], "obligation": obligation.id,
             "procedure": method["procedure"][:80], "round": round_no,
             "reason": verdict["reason"][:200]},
        )
        attempt.redos.append({"reason": verdict["reason"],
                              "outcome": None, "artifact": None,
                              "audit": None})
        notes = "\n\n".join(
            f"Redo round {i + 1} diagnostic — the judger rejected the "
            f"previous attempt: {r['reason']}"
            for i, r in enumerate(attempt.redos)
        )
        method_fb = {**method, "redo_feedback": notes}
        result = self.discharger.discharge(
            row, obligation, method=method_fb, tag="bound-redo")
        attempt.redos[-1]["outcome"] = result.outcome
        attempt.redos[-1]["artifact"] = result.artifact
        return result

    # settlement
    def _reject(self, row, obligation, method, reason: str) -> None:
        self.log.append(
            "method_rejected",
            {"claim": row["id"], "obligation": obligation.id,
             "procedure": method["procedure"][:80], "reason": reason},
        )

    def _record(self, obligation, attempts, bound) -> str:
        return self.artifacts.put(
            {
                "instrument": "binding-game",
                "obligation": obligation.text,
                "attempts": [vars(a) for a in attempts],
                "bound": bound,
            }
        )

    def _settle(self, row, obligation, attempts, method, result,
                audit) -> DischargeResult:
        bound = {"method": method, "audit": audit,
                 "discharge_artifact": result.artifact}
        digest = self._record(obligation, attempts, bound)
        if self.memory is not None:
            self.memory.record_instrument(method["procedure"])
        self.log.append(
            "method_bound",
            {"claim": row["id"], "obligation": obligation.id,
             "procedure": method["procedure"][:80],
             "level": method.get("level", ""),
             "strength": _method_strength(method)},
        )
        return DischargeResult(
            outcome=result.outcome,
            witness=result.witness,
            artifact=digest,
            instrument=method["procedure"][:80],
            # the oracle's declared basis, already audited by the judger as
            # part of fidelity; absent (legacy fixtures) fails closed to record
            basis=method.get("basis", BASIS_RECORD),
            # the declared strength the judger countersigned (sufficiency at
            # tier, bridge completeness, degradation): what the settled
            # verdict is worth downstream — with the unverified bridges it
            # rests on, for certificate disclosure
            strength=_method_strength(method),
            bridges=tuple(method.get("bridges", ())),
            # the executor's declared premise assumptions ride through the
            # binding game onto the settled evidence — the kernel gates
            # them against the claim's dependency closure
            assumed_premises=result.assumed_premises,
            witness_kind=result.witness_kind,
            inputs=result.inputs,
        )

    def _fail(self, row, obligation, attempts, reason,
              unobtainable: bool = False) -> DischargeResult:
        digest = self._record(obligation, attempts, None)
        return DischargeResult(
            outcome=OUTCOME_UNOBTAINABLE if unobtainable else OUTCOME_FAILED,
            witness=reason,
            artifact=digest,
            instrument="binding-game",
        )


def _features(row: Dict) -> str:
    """Mechanical claim features"""
    text = row["text"]
    return json.dumps(
        {
            "kind": row.get("kind", ""),
            "length": len(text),
            "has_digits": any(c.isdigit() for c in text),
            "has_quantifier_words": any(
                w in text for w in ("all", "any", "every", "each", "exists")
            ),
        },
        ensure_ascii=False,
    )


def _attempt_diagnostic(attempt: Attempt) -> Dict:
    """Bounded method diagnostics for the Oracle's single repair pass.

    Artifact contents and digests are deliberately absent: this feedback says
    why an implementation was rejected, never whether the obligation is true.
    """
    method = {
        "procedure": attempt.method.get("procedure", ""),
        "promised_artifact": attempt.method.get("artifact", ""),
        "level": attempt.method.get("level", ""),
        "basis": attempt.method.get("basis", BASIS_RECORD),
        "strength": _method_strength(attempt.method),
        "bridges": attempt.method.get("bridges", []),
    }

    def public(audit):
        if audit is None:
            return None
        return {k: v for k, v in audit.items() if not k.startswith("_")}

    return {
        "method": method,
        "initial_outcome": attempt.discharge_outcome,
        "initial_audit": public(attempt.audit),
        "redos": [
            {
                "reason": redo.get("reason"),
                "outcome": redo.get("outcome"),
                "audit": public(redo.get("audit")),
            }
            for redo in attempt.redos
        ],
        "terminal_reason": attempt.rejected,
    }
