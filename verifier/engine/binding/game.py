"""
Binding game: per obligation, bind one sufficient method to real evidence.

Per obligation:

0. An explicitly injected experimental process-local cache may supply
   decisive typed methods whose sufficiency a Judger already accepted. Only
   methods are reused: an adapter runs against the current candidate and its
   new evidence envelope is mechanically authenticated. Holds and refuted
   audits occupy separate slots and each must have been independently
   accepted; no adapter property implies the missing direction. When a fresh
   adapter observation lands in a missing direction, that exact MethodIR and
   evidence face a new independent Judger audit. Only a complete yes/yes/n/a
   audit may seed the direction; otherwise the complete game below runs.
   The shipped online/offline kernels do not inject this cache: their normal
   architecture always enters the Oracle/executor/fresh-Judger game below.
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
from typing import Dict, List, Optional

from ...config import (
    APPEND_ONLY_DELTA_PROTOCOL,
    RunOptions,
    append_only_deltas_enabled,
)
from ...core.events import EventLog
from ...core.model import (
    BASIS_RECORD, Obligation, basis_satisfies, OUTCOME_REFUTED, OUTCOME_FAILED, OUTCOME_UNOBTAINABLE,
    STRENGTH_DECISIVE,
)
from ...execution.discharge import DischargeResult, ScriptRun
from ...execution.instruments import ArtifactStore
from ...llm.agents import (
    ENUMERATE_ORACLES_AGENT, JUDGER_AGENT, SCRIPT_JUDGER_AGENT,
)
from ...llm.model_spans import model_scope
from ...llm.runtime import capabilities_of
from ...llm.schema import RoleError
from ..gates import below_decisive_reason, preflight_defect
from .auditor import Auditor
from .method_cache import build_method_cache_key
from .methods import (
    MOVED_RETRIES, Attempt, attempt_diagnostic, blocking_tiers,
    degradation_record,
    method_strength, moved_reason, prior_attempt_briefing,
    tier_order,
)
from .oracle import MethodSource
from .reuse import MethodReuse


class BindingGame:
    def __init__(
        self,
        backend,
        artifacts: ArtifactStore,
        log: EventLog,
        discharger,
        oracles_role=ENUMERATE_ORACLES_AGENT,
        judger_role=JUDGER_AGENT,
        script_judger_role=SCRIPT_JUDGER_AGENT,
        redo_rounds: int = 3,
        registry: str = "[]",
        memory=None,
        live_world: bool = True,
        judger_error_retries: int = 1,
        method_cache=None,
        typed_adapter_judger_packet: bool = False,
        sufficiency_role=None,
        packet_method_reuse: bool = False,
        run_options: Optional[RunOptions] = None,
    ):
        if type(typed_adapter_judger_packet) is not bool:
            raise ValueError("typed_adapter_judger_packet must be a bool")
        if type(packet_method_reuse) is not bool:
            raise ValueError("packet_method_reuse must be a bool")
        self.backend = backend
        # the run's evidence-presentation options (default: the runtime's)
        self.run_options = (run_options if run_options is not None
                            else capabilities_of(backend).run_options)
        # An operator's explicit reuse request also covers the packet
        # transports (prefetch, typed packet).  Default off: the packet
        # experiments alone never acquire authority to skip a signer.
        self.packet_method_reuse = packet_method_reuse
        # Optional pre-execution sufficiency audit of agent-executed methods.
        # None keeps the established Oracle/executor/Judger sequence.
        self.sufficiency_role = sufficiency_role
        self.artifacts = artifacts
        self.log = log
        self.discharger = discharger
        self.oracles_role = oracles_role
        self.judger_role = judger_role
        # the reader of a method in EXECUTABLE FORM (a script the executor
        # ran verbatim): sufficiency from the script, outcome from its
        # output. None -> scripts are ignored and every method goes to
        # the agent executor.
        self.script_judger_role = script_judger_role
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
        # Match the Kernel's run-scoped compiler identity. Environment
        # mutation cannot move accepted methods between experiment modes.
        self.append_only_deltas = append_only_deltas_enabled()
        self.append_only_delta_protocol = (
            APPEND_ONLY_DELTA_PROTOCOL if self.append_only_deltas else None)
        # Optional experimental dependency.  The shipped online/offline
        # kernels deliberately pass ``None`` so every obligation follows the
        # Oracle/executor/fresh-Judger architecture below.
        self.method_cache = method_cache
        # Default-off transport experiment. It keeps the same fresh Judger
        # seat and schema, but cannot share accepted methods while enabled.
        self.typed_adapter_judger_packet = typed_adapter_judger_packet
        # the game's collaborators: the oracle seat's method source, the
        # judger seats, and — when a method cache is set — method reuse
        self.oracle = MethodSource(self)
        self.auditor = Auditor(self)
        self.reuse = MethodReuse(self)

    def prewarm(self, row: Dict, obligation: Obligation) -> None:
        """Enumerate an obligation's first method batch ahead of the commit
        that needs it (see MethodSource.prewarm)."""
        self.oracle.prewarm(row, obligation)

    # Entry point
    def run(self, row: Dict, obligation: Obligation) -> DischargeResult:
        with model_scope(
                claim_id=row.get("id", ""),
                claim_kind=row.get("kind", ""),
                locus=("final_task" if row.get("kind") == "final"
                       else "support"),
                obligation_id=obligation.id):
            if self.method_cache is None:
                return self.play(row, obligation)
            return self.reuse.run(row, obligation)

    def method_cache_key(self, row, obligation):
        # This transport protocol is deliberately fresh-review-only until its
        # semantic/result gates have passed.  It may reduce inspection turns;
        # it must not also acquire authority to skip either signer.
        if (not self.packet_method_reuse
                and (self.run_options.judger_prefetch
                     or self.typed_adapter_judger_packet)):
            return None
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
            append_only_deltas=self.append_only_deltas,
            append_only_delta_protocol=self.append_only_delta_protocol,
        )

    def play(self, row: Dict, obligation: Obligation) -> DischargeResult:
        attempts: List[Attempt] = []
        try:
            methods = self.oracle.enumerate(row, obligation, oracle_round=0)
        except RoleError as err:
            # A crashed role or schema-invalid generation is not evidence that
            # no sufficient method exists.  Keep generation failure distinct
            # from the oracle's valid, explicit empty-list conclusion.
            self.log.append(
                "methods_enumerated",
                {"claim": row["id"], "obligation": obligation.id,
                 "error": str(err)},
            )
            return self.fail(
                row, obligation, attempts,
                f"oracle enumeration failed: {err}",
            )

        initial_empty = not methods
        if initial_empty and self.live_world:
            # A live world may change, so preserve the existing fail-closed,
            # retryable result.  Offline gets the same single bounded repair as
            # an exhausted non-empty batch: a second look may notice a retained
            # effect or a fresh experiment over the frozen captured subject.
            return self.fail(
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
                self.reject(row, obligation, m,
                             "declared basis below the obligation's floor")
            methods = [m for m in methods if m not in below]
            if not methods:
                return self.fail(
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
        methods.sort(key=tier_order)

        unavailable_reasons: List[str] = []
        audited_gap = False   # an EXECUTOR-audited unobtainable is on record
        # (the empty-repair boundary declaration also lands in
        # unavailable_reasons but is weaker: it only settles unobtainable
        # when nothing failed — an oracle saying "no further method" must
        # not promote a pile of retryable failures)
        structural_feedback: List[Dict[str, str]] = []
        seen_procedures = set()
        deferred: List[Dict[str, str]] = []
        tier_listed = {method_strength(m) for m in methods}
        tier_unobtainable: set = set()
        degradation_disputed = False
        contrary_signal = False   # a non-decisive instrument reported a
        # negative result: inadmissible as a refutation, but a live
        # contrary indicator — the obligation must not read out as
        # "everything that ran held" (partial) while it stands
        failed_seen = False
        repair_used = False
        moved_runs: Dict[str, int] = {}
        method_index = 0

        def judged_attempts():
            return any(not a.moved for a in attempts)

        while (method_index < len(methods) or deferred
               or ((judged_attempts() or initial_empty)
                   and not repair_used)):
            if method_index >= len(methods):
                ready = [m for m in deferred
                         if not blocking_tiers(m, tier_listed,
                                               tier_unobtainable)]
                if ready:
                    # a stronger tier resolved to structurally unavailable
                    # after these were parked: they are executable now
                    deferred = [m for m in deferred if m not in ready]
                    ready.sort(key=tier_order)
                    methods.extend(ready)
                    continue
                if not ((judged_attempts() or initial_empty)
                        and not repair_used):
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
                            attempt_diagnostic(a) for a in attempts
                            if not a.moved
                        ],
                        # parked, not rejected: these wait on a stronger
                        # tier's structural unavailability, so the repair
                        # must not re-list them
                        "deferred_methods": [
                            {"procedure": m.get("procedure", "")[:80],
                             "strength": method_strength(m),
                             "waiting_on": blocking_tiers(
                                 m, tier_listed, tier_unobtainable)}
                            for m in deferred
                        ],
                    },
                    ensure_ascii=False,
                )
                try:
                    repaired = self.oracle.enumerate(
                        row, obligation, method_feedback=feedback,
                        oracle_round=1,
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

                if not repaired:
                    if self.live_world:
                        failed_seen = True
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
                        self.reject(
                            row, obligation, candidate,
                            "declared basis below the obligation's floor",
                        )
                    elif candidate["procedure"] in seen_procedures:
                        self.reject(
                            row, obligation, candidate,
                            "duplicate procedure returned on method repair",
                        )
                    else:
                        methods.append(candidate)
                        tier_listed.add(method_strength(candidate))
                        added += 1
                if added:
                    # the unexecuted tail keeps the strictness-descending
                    # invariant the initial sort established
                    methods[method_index:] = sorted(
                        methods[method_index:], key=tier_order)
                if not added:
                    # A non-empty repair batch made entirely of low-floor or
                    # already-tried procedures is a protocol failure, not an
                    # evidence-availability finding.
                    failed_seen = True
                continue

            method = methods[method_index]
            method_index += 1
            strength = method_strength(method)
            blocking = blocking_tiers(method, tier_listed, tier_unobtainable)
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
                self.reject(
                    row, obligation, method,
                    "duplicate procedure returned by oracle",
                )
                continue
            seen_procedures.add(procedure)
            attempt = Attempt(method=method)
            attempts.append(attempt)

            if method.get("script") and self.script_judger_role is not None \
                    and hasattr(self.discharger, "run_script"):
                # EXECUTABLE FORM: the script runs verbatim (no agent, so
                # fidelity holds by construction and there is no redo
                # loop); the script judger signs sufficiency and reads
                # the outcome off the retained output. The same axioms
                # and floors gate the result as on the agent path.
                settled = self.run_script_method(
                    row, obligation, method, attempt, attempts, tier_listed)
                if settled is not None:
                    return settled
                if attempt.moved:
                    self.requeue_moved(row, obligation, method, methods,
                                        method_index, seen_procedures,
                                        moved_runs, reason=attempt.rejected)
                    continue
                outcome = attempt.discharge_outcome
                detail = str(attempt.rejected or "")
                audit = attempt.audit or {}
                if outcome == OUTCOME_UNOBTAINABLE:
                    unavailable_reasons.append(
                        detail[len("unobtainable: "):]
                        if detail.startswith("unobtainable: ") else detail)
                    audited_gap = True
                    tier_unobtainable.add(strength)
                    continue
                failed_seen = True
                if audit.get("_contrary"):
                    contrary_signal = True
                disputed = (audit.get("degradation") == "unjustified"
                            and strength != STRENGTH_DECISIVE
                            and audit.get("sufficient") == "yes")
                if disputed:
                    degradation_disputed = True
                if audit and (audit.get("sufficient") != "yes" or disputed
                              or audit.get("_contrary")
                              or audit.get("_structural")):
                    # a method defect the oracle's repair pass may fix
                    # (an execution failure is not one: it stays in the
                    # exhausted-attempts diagnostic only)
                    structural_feedback.append(
                        {"procedure": procedure, "reason": detail})
                continue

            method_attempt = len(attempts)
            preaudit = self.auditor.preaudit(row, obligation, method, method_attempt)
            if preaudit is not None and preaudit.get("sufficient") == "no":
                # The Judger's first question, asked before the executor
                # spends its budget.  Same repair semantics as a post-hoc
                # sufficient=no: the Oracle may replace the method once.
                if repair_used:
                    failed_seen = True
                structural_feedback.append({
                    "procedure": procedure,
                    "reason": preaudit["reason"],
                })
                reason = f"sufficiency pre-audit: {preaudit['reason']}"
                self.reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue
            # Typed adapters execute a declared signature, not a prompt: an
            # extra key would only make the router treat the method as an
            # agent method.  Only an agent executor is briefed.
            briefing = ("" if "adapter" in method
                        else prior_attempt_briefing(attempts[:-1]))
            bound_method = method
            if briefing:
                attempt.briefing = briefing
                bound_method = {**method, "redo_feedback": briefing}
                self.log.append("executor_briefed", {
                    "claim": row["id"], "obligation": obligation.id,
                    "procedure": procedure[:80],
                    "method_attempt": method_attempt,
                    "prior_attempts": len(attempts) - 1})
            with model_scope(
                    stage="discharge", method_attempt=method_attempt, redo=0):
                result = self.discharger.discharge(
                    row, obligation, method=bound_method, tag="bound"
                )
            attempt.discharge_artifact = result.artifact
            attempt.discharge_outcome = result.outcome
            self.oracle.note_reached(row, result)

            if result.world_moved and not result.world_changes:
                # the solver changed the delivered world while this check
                # ran: whatever it reported is about no single state, and
                # nothing about the method follows (a change the record
                # charges to the check itself stays a structural defect,
                # judged below)
                attempt.moved = True
                attempt.rejected = moved_reason(result.world_moved)
                self.requeue_moved(row, obligation, method, methods,
                                    method_index, seen_procedures,
                                    moved_runs, reason=attempt.rejected)
                continue

            if result.outcome == OUTCOME_UNOBTAINABLE:
                detail = str(result.witness or
                             "required instrument or evidence unavailable")
                unavailable_reasons.append(detail)
                self.oracle.note_boundary(row, obligation, detail)
                audited_gap = True
                tier_unobtainable.add(strength)
                reason = f"unobtainable: {detail}"
                self.reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue

            if result.outcome == OUTCOME_FAILED:
                failed_seen = True
                detail = str(result.witness or "")
                reason = "execution failure at retry limit"
                if detail:
                    reason += f": {detail}"
                self.reject(row, obligation, method, reason)
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
                reason = below_decisive_reason(strength, result.witness)
                structural_feedback.append({
                    "procedure": procedure,
                    "reason": reason,
                })
                self.reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue

            structural = self.structural_defect(result, obligation)
            if structural:
                # witness / independence axioms, enforced before the
                # judger spends a call: an absence-only refutation or a
                # holds grounded on a delivered artifact as authority is
                # a method defect the repair pass may fix, never evidence
                failed_seen = True
                structural_feedback.append({
                    "procedure": procedure, "reason": structural})
                self.reject(row, obligation, method, structural)
                attempt.rejected = structural
                continue

            # every substantive outcome — holds AND refuted — faces the
            # judger before it settles, for both method sufficiency and
            # execution fidelity. A refutation is the most
            # consequential verdict in the system (a refuted node refutes
            # its claim — for the final, the whole certificate), so an
            # unaudited one is exactly the evidence we cannot afford to
            # accept.
            degradation = degradation_record(
                strength, attempts, tier_listed)
            verdict = self.auditor.judge(
                obligation, method, result, attempt,
                degradation=degradation, method_attempt=method_attempt)
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
                self.reject(row, obligation, method, reason)
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
                self.reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue
            if (verdict.get("sufficient") == "yes"
                    and verdict["faithful"] == "yes"):
                return self.settle(row, obligation, attempts,
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
                result = self.redo(
                    row, obligation, method, verdict, attempt, rounds,
                    method_attempt=method_attempt)
                if result.world_moved and not result.world_changes:
                    reason = (f"{moved_reason(result.world_moved)} "
                              f"(redo {rounds})")
                    break
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
                    reason = below_decisive_reason(
                        strength, result.witness, redo=rounds)
                    structural_feedback.append({
                        "procedure": procedure,
                        "reason": reason,
                    })
                    break
                structural = self.structural_defect(result, obligation)
                if structural:
                    failed_seen = True
                    reason = f"{structural} (redo {rounds})"
                    structural_feedback.append({
                        "procedure": procedure, "reason": reason})
                    break
                verdict = self.auditor.judge(
                    obligation, method, result, attempt,
                    redo=rounds, degradation=degradation,
                    method_attempt=method_attempt)
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
                    return self.settle(row, obligation, attempts,
                                        method, result, audit=verdict)
            if reason is None:
                reason = (f"judger after {rounds} redo round(s): "
                          f"{verdict['reason']}"
                          if rounds else f"judger: {verdict['reason']}")
            self.reject(row, obligation, method, reason)
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
        if attempts and not judged_attempts() and not unavailable_only:
            reason = ("the delivered world changed during every check of "
                      "this obligation while the solver was running tools "
                      "that change it; nothing settled")
        if deferred:
            reason += (f"; {len(deferred)} weaker-tier method(s) stayed "
                       "deferred behind a stronger tier never audited "
                       "structurally unavailable")
        return self.fail(row, obligation, attempts, reason,
                          unobtainable=unavailable_only)

    def run_script_method(self, row, obligation, method, attempt,
                           attempts, tier_listed) -> Optional[DischargeResult]:
        """Run a method's EXECUTABLE FORM and adjudicate it. Returns the
        settled DischargeResult when the method binds; otherwise records
        the terminal reason on the attempt (attempt.rejected, with
        attempt.discharge_outcome and attempt.audit for the caller's
        aggregation) and returns None."""
        strength = method_strength(method)
        procedure = method["procedure"]
        run: ScriptRun = self.discharger.run_script(
            row, obligation, method=method, tag="bound")
        attempt.discharge_artifact = run.artifact
        self.log.append(
            "discharge",
            {"claim": row["id"], "obligation": obligation.id,
             "instrument": "script", "outcome": "ran",
             "exit_code": run.exit_code, "timed_out": run.timed_out,
             "seconds": run.seconds, "artifact": run.artifact})
        if run.launch_error or run.timed_out:
            reason = ("script execution failure: "
                      + (run.launch_error or f"timed out after {run.seconds}s"))
            attempt.discharge_outcome = OUTCOME_FAILED
            self.reject(row, obligation, method, reason)
            attempt.rejected = reason
            return None
        if run.world_moved and not run.world_changes:
            attempt.moved = True
            attempt.discharge_outcome = OUTCOME_FAILED
            attempt.rejected = moved_reason(run.world_moved)
            return None
        if run.world_changes:
            reason = self.structural_defect(
                DischargeResult(outcome=OUTCOME_FAILED,
                                world_changes=run.world_changes), obligation)
            attempt.discharge_outcome = OUTCOME_FAILED
            attempt.audit = {"_structural": True, "reason": reason}
            self.reject(row, obligation, method, reason)
            attempt.rejected = reason
            return None

        degradation = degradation_record(strength, attempts, tier_listed)
        verdict = self.auditor.judge_script(obligation, method, run, attempt,
                                     degradation)
        outcome = verdict.get("outcome", OUTCOME_FAILED)
        attempt.discharge_outcome = outcome
        if verdict.get("sufficient") != "yes":
            reason = f"judger method insufficient: {verdict['reason']}"
            if verdict.get("_infra"):
                reason = verdict["reason"]
            self.reject(row, obligation, method, reason)
            attempt.rejected = reason
            return None
        if strength != STRENGTH_DECISIVE \
                and verdict.get("degradation") == "unjustified":
            reason = f"judger degradation unjustified: {verdict['reason']}"
            self.reject(row, obligation, method, reason)
            attempt.rejected = reason
            return None
        result = DischargeResult(
            outcome=outcome,
            witness=verdict.get("witness") or None,
            artifact=run.artifact,
            instrument="script",
            basis=method.get("basis", BASIS_RECORD),
            strength=strength,
            witness_kind=verdict.get("witness_kind"),
            inputs=tuple(verdict.get("inputs") or ()),
        )
        if outcome == OUTCOME_UNOBTAINABLE:
            detail = str(result.witness
                         or "required instrument or evidence unavailable")
            self.oracle.note_boundary(row, obligation, detail)
            reason = f"unobtainable: {detail}"
            self.reject(row, obligation, method, reason)
            attempt.rejected = reason
            return None
        if outcome == OUTCOME_FAILED:
            reason = "script did not decide the obligation"
            if result.witness:
                reason += f": {result.witness}"
            self.reject(row, obligation, method, reason)
            attempt.rejected = reason
            return None
        if outcome == OUTCOME_REFUTED and strength != STRENGTH_DECISIVE:
            reason = below_decisive_reason(strength, result.witness)
            attempt.audit["_contrary"] = True
            self.reject(row, obligation, method, reason)
            attempt.rejected = reason
            return None
        structural = self.structural_defect(result, obligation)
        if structural:
            attempt.audit["_structural"] = True
            self.reject(row, obligation, method, structural)
            attempt.rejected = structural
            return None
        if outcome == OUTCOME_REFUTED and self.judger_role is not None:
            # REFUTATION COUNTERSIGN: a refutation is the one verdict that
            # can send the solver to break a right delivery, and on this
            # path a single no-tools reader produced it. The independent
            # judger, with the artifact and the world in reach, must
            # countersign both signatures before it settles; a disputed
            # refutation is a failed attempt with the dispute on record.
            reading = attempt.audit
            counter = self.auditor.judge(obligation, method, result, attempt,
                                  degradation=degradation,
                                  source="refutation countersign")
            attempt.audit = reading
            if not (counter.get("sufficient") == "yes"
                    and counter.get("faithful") == "yes"):
                reason = ("refutation not countersigned: "
                          f"{counter.get('reason', '')}")
                attempt.audit = dict(attempt.audit, countersign=counter)
                if counter.get("sufficient") == "no":
                    attempt.audit["_structural"] = True
                self.reject(row, obligation, method, reason)
                attempt.rejected = reason
                return None
            attempt.audit = dict(attempt.audit, countersign=counter)
        return self.settle(row, obligation, attempts, method, result,
                            audit=attempt.audit)

    @staticmethod
    def structural_defect(result: DischargeResult,
                           obligation: Obligation) -> Optional[str]:
        """The axioms checked on the executor's own declaration, before
        any semantic audit (see engine.gates PREFLIGHT): the check left
        the delivered world as it found it, a refutation carries an
        admissible witness, and a holds rests on no delivered artifact
        standing as authority."""
        return preflight_defect(result, obligation)

    # redo, rejection, and settlement
    def redo(self, row, obligation, method, verdict, attempt, round_no: int,
              *, method_attempt: int = 0):
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
        # Select before appending: each redo sees its immediate predecessor,
        # never another obligation/method or an arbitrary model-chosen path.
        previous_artifact = (attempt.redos[-1].get("artifact")
                             if attempt.redos else attempt.discharge_artifact)
        attempt.redos.append({"reason": verdict["reason"],
                              "outcome": None, "artifact": None,
                              "audit": None})
        notes = "\n\n".join(
            f"Redo round {i + 1} diagnostic — the judger rejected the "
            f"previous attempt: {r['reason']}"
            for i, r in enumerate(attempt.redos)
        )
        if attempt.briefing:
            notes = attempt.briefing + "\n\n" + notes
        method_fb = {**method, "redo_feedback": notes}
        with model_scope(
                stage="discharge", method_attempt=method_attempt,
                redo=round_no):
            repair = getattr(self.discharger, "discharge_with_repair", None)
            if callable(repair):
                result = repair(
                    row, obligation, method=method_fb, tag="bound-redo",
                    previous_artifact=previous_artifact)
            else:
                result = self.discharger.discharge(
                    row, obligation, method=method_fb, tag="bound-redo")
        attempt.redos[-1]["outcome"] = result.outcome
        attempt.redos[-1]["artifact"] = result.artifact
        return result

    # settlement
    def reject(self, row, obligation, method, reason: str) -> None:
        self.log.append(
            "method_rejected",
            {"claim": row["id"], "obligation": obligation.id,
             "procedure": method["procedure"][:80], "reason": reason},
        )

    def requeue_moved(self, row, obligation, method, methods,
                       method_index, seen_procedures, moved_runs,
                       reason: str = "") -> None:
        """A check the solver's own work moved the world under: logged
        apart from rejections, and the method runs again (once) after the
        rest of its tier, keeping the tail's strictness order."""
        procedure = method["procedure"]
        runs = moved_runs.get(procedure, 0) + 1
        moved_runs[procedure] = runs
        requeued = runs <= MOVED_RETRIES
        if requeued:
            seen_procedures.discard(procedure)
            methods.append(method)
            methods[method_index:] = sorted(methods[method_index:],
                                            key=tier_order)
        self.log.append(
            "method_moved",
            {"claim": row["id"], "obligation": obligation.id,
             "procedure": procedure[:80], "run": runs,
             "requeued": requeued, "reason": reason[:600]},
        )

    def record(self, obligation, attempts, bound) -> str:
        return self.artifacts.put(
            {
                "instrument": "binding-game",
                "obligation": obligation.text,
                "attempts": [vars(a) for a in attempts],
                "bound": bound,
            }
        )

    def settle(self, row, obligation, attempts, method, result,
                audit) -> DischargeResult:
        bound = {"method": method, "audit": audit,
                 "discharge_artifact": result.artifact}
        digest = self.record(obligation, attempts, bound)
        if self.memory is not None:
            self.memory.record_instrument(method["procedure"])
        self.log.append(
            "method_bound",
            {"claim": row["id"], "obligation": obligation.id,
             "procedure": method["procedure"][:80],
             "level": method.get("level", ""),
             "strength": method_strength(method)},
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
            strength=method_strength(method),
            bridges=tuple(method.get("bridges", ())),
            # the executor's declared premise assumptions ride through the
            # binding game onto the settled evidence — the kernel gates
            # them against the claim's dependency closure
            assumed_premises=result.assumed_premises,
            witness_kind=result.witness_kind,
            inputs=result.inputs,
        )

    def fail(self, row, obligation, attempts, reason,
              unobtainable: bool = False) -> DischargeResult:
        digest = self.record(obligation, attempts, None)
        return DischargeResult(
            outcome=OUTCOME_UNOBTAINABLE if unobtainable else OUTCOME_FAILED,
            witness=reason,
            artifact=digest,
            instrument="binding-game",
        )
