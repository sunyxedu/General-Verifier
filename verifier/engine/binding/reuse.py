"""Method reuse across candidates: the experimental accepted-method cache
path of the binding game.

Only methods are reused, never verdicts: a cached MethodIR whose
sufficiency a Judger accepted in one direction (holds or refuted) is run
again as a typed adapter against the current candidate, and its new
evidence envelope is mechanically authenticated; a fresh result in a
direction without an accepted signature faces a complete independent
Judger audit; anything else plays the complete game.
"""
from __future__ import annotations

import json
from typing import Dict

from ...core.model import (
    BASIS_RECORD, BASIS_WORLD, OUTCOME_HOLDS, OUTCOME_REFUTED,
    Obligation, PROVENANCE_DELIVERED, ROLE_SUBJECT, STRENGTH_DECISIVE, basis_satisfies,
    content_hash,
)
from ...execution.adapters import AdapterInvocation, AdapterNotApplicable
from ...execution.discharge import DischargeResult
from .method_cache import AcceptedMethod
from .methods import Attempt, degradation_record, method_strength


class MethodReuse:
    """The accepted-method cache path of one binding game (the ``host``,
    whose cache, executor, judger and settlement it uses)."""

    def __init__(self, host):
        self.host = host

    def run(self, row: Dict, obligation: Obligation) -> DischargeResult:
        if self.host.method_cache is None:
            return self.host.play(row, obligation)
        key = self.host.method_cache_key(row, obligation)
        if key is None:
            self.host.log.append("method_cache_bypass", {
                "claim": row["id"], "obligation": obligation.id,
                "key": None, "reason": "no exact reuse key for this "
                "claim, obligation, authority, world, or backend"})
            return self.host.play(row, obligation)

        lease = self.host.method_cache.acquire(key)
        if lease.entries:
            result, missing, reason = self.run_cached_entries(
                row, obligation, lease.entries,
                key=key, waited=lease.waited)
            if result is not None:
                return result
            self.host.log.append("method_cache_bypass", {
                "claim": row["id"], "obligation": obligation.id,
                "key": key[:16], "reason": reason[:500],
            })
            if missing is not None:
                source_accepted, fresh = missing
                method = source_accepted.method
                direction_lease = self.host.method_cache.acquire(
                    key, direction=fresh.outcome)
                if direction_lease.entry is not None:
                    accepted = direction_lease.entry
                    if accepted.method == method:
                        reused, reuse_reason = self.settle_cached_result(
                            row, obligation, accepted, fresh, key=key,
                            waited=(lease.waited or direction_lease.waited))
                    else:
                        reused, reuse_reason = self.run_cached(
                            row, obligation, accepted, key=key,
                            waited=(lease.waited or direction_lease.waited))
                    if reused is not None:
                        return reused
                    self.host.log.append("method_cache_bypass", {
                        "claim": row["id"], "obligation": obligation.id,
                        "key": key[:16], "reason": reuse_reason[:500],
                    })
                elif direction_lease.leader:
                    try:
                        attempt, audit_passed, audit_reason = \
                            self.audit_missing_direction(
                            obligation, method, fresh)
                        audit = attempt.audit or {}
                        judger_accepted = audit_passed
                        authority_stable = (
                            self.host.method_cache_key(row, obligation) == key)
                        if audit_passed and not authority_stable:
                            audit_passed = False
                            audit_reason = (
                                "task authority changed during direction "
                                "audit")
                        self.host.log.append("method_cache_direction_audit", {
                            "schema": "gv-method-direction-audit-v1",
                            "claim": row["id"],
                            "obligation": obligation.id,
                            "key": key[:16],
                            "method_digest": content_hash(method),
                            "procedure": method["procedure"][:80],
                            "fresh_discharge_artifact": fresh.artifact,
                            "fresh_outcome": fresh.outcome,
                            "source_binding_artifact":
                                source_accepted.source_binding_artifact,
                            "source_discharge_artifact":
                                source_accepted.source_discharge_artifact,
                            "source_outcome":
                                source_accepted.source_outcome,
                            "judger_accepted": judger_accepted,
                            "authority_stable_after_judger": authority_stable,
                            "direction_eligible": audit_passed,
                            "sufficient": audit.get("sufficient"),
                            "faithful": audit.get("faithful"),
                            "degradation": audit.get("degradation"),
                            "reason": str(audit.get(
                                "reason", audit_reason))[:500],
                        })
                        if audit_passed:
                            audited = self.host.settle(
                                row, obligation, [attempt], method, fresh,
                                audit=attempt.audit)
                            accepted = self.accepted_method(
                                row, obligation, audited)
                            if self.host.method_cache_key(row, obligation) != key:
                                audit_reason = (
                                    "task authority changed while publishing "
                                    "direction audit")
                            elif (accepted is None
                                  or accepted.source_outcome != fresh.outcome):
                                audit_reason = (
                                    "independent direction audit could not be "
                                    "authenticated for cache publication")
                            else:
                                direction_lease.finish(accepted)
                                return audited
                        self.host.log.append("method_cache_bypass", {
                            "claim": row["id"],
                            "obligation": obligation.id,
                            "key": key[:16],
                            "reason": audit_reason[:500],
                            # Keep the rejected optimization attempt
                            # auditable without importing it into the
                            # complete Oracle-led game's evidence set.
                            "direction_audit": {
                                "schema": "gv-method-direction-audit-v1",
                                "method_digest": content_hash(method),
                                "procedure": method["procedure"][:80],
                                "fresh_discharge_artifact": fresh.artifact,
                                "fresh_outcome": fresh.outcome,
                                "sufficient": audit.get("sufficient"),
                                "faithful": audit.get("faithful"),
                                "degradation": audit.get("degradation"),
                                "reason": str(audit.get(
                                    "reason", audit_reason))[:500],
                            },
                        })
                        return self.run_uncached_and_cache(
                            row, obligation, key, lease=direction_lease,
                            direction=fresh.outcome)
                    finally:
                        # This owns the direction flight until either a
                        # certified entry is published or the complete-game
                        # fallback takes it over. Every exceptional exit,
                        # including logging/artifact failures, wakes waiters.
                        direction_lease.finish()
                return self.run_uncached_and_cache(
                    row, obligation, key, lease=direction_lease,
                    direction=fresh.outcome)
            return self.run_uncached_and_cache(row, obligation, key)
        if not lease.leader:
            # The concurrent leader found no eligible accepted method.  Do
            # not serialize another cold fill; proceed independently.  A
            # successful follower may still seed later candidates.
            return self.run_uncached_and_cache(row, obligation, key)

        return self.run_uncached_and_cache(
            row, obligation, key, lease=lease)

    def run_uncached_and_cache(
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
            result = self.host.play(row, obligation)
            # The authority files are normally immutable.  Re-key anyway so
            # a live edit during the cold fill can never publish under the
            # earlier task identity.
            if self.host.method_cache_key(row, obligation) == key:
                accepted, why = self.accepted_method_or_reason(
                    row, obligation, result)
                if accepted is not None:
                    if lease is not None and lease.leader and (
                            direction is None
                            or accepted.source_outcome == direction):
                        fill = accepted
                    else:
                        self.host.method_cache.publish_if_absent(key, accepted)
                    self.host.log.append("method_cache_published", {
                        "claim": row["id"], "obligation": obligation.id,
                        "key": key[:16], "direction": accepted.source_outcome,
                        "adapter": str(accepted.method.get("adapter", ""))[:80]})
                else:
                    self.host.log.append("method_cache_bypass", {
                        "claim": row["id"], "obligation": obligation.id,
                        "key": key[:16],
                        "reason": "settled result not cacheable: " + why})
            else:
                self.host.log.append("method_cache_bypass", {
                    "claim": row["id"], "obligation": obligation.id,
                    "key": key[:16],
                    "reason": "task authority changed during the game; "
                              "the result is not published under the "
                              "earlier reuse key"})
            return result
        finally:
            if lease is not None:
                lease.finish(fill)

    def run_cached_entries(
        self, row, obligation, entries, *, key, waited=False,
    ):
        """Try each distinct cached MethodIR without crossing signatures.

        Returns ``(settled, missing_observation, reason)``. The observation is
        ``(source_accepted_method, fresh_result)`` only when its substantive
        direction has no independently accepted slot. If two direction slots
        contain the same exact MethodIR, one fresh adapter execution can
        select the matching certificate without running that adapter twice.
        """
        directions = {entry.source_outcome for entry in entries}
        attempted_methods = []
        reasons = []
        missing = None
        for accepted in entries:
            if any(accepted.method == method for method in attempted_methods):
                continue
            attempted_methods.append(accepted.method)
            fresh, reason = self.execute_cached_method(
                row, obligation, accepted, key=key)
            if fresh is None:
                reasons.append(reason)
                continue
            matching = next((entry for entry in entries
                             if entry.source_outcome == fresh.outcome
                             and entry.method == accepted.method), None)
            if matching is not None:
                result, reason = self.settle_cached_result(
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
                missing = (accepted, fresh)
        return None, missing, "; ".join(reasons) or "no reusable MethodIR"

    def run_cached(self, row, obligation, accepted, *, key, waited=False):
        fresh, reason = self.execute_cached_method(
            row, obligation, accepted, key=key)
        if fresh is None:
            return None, reason
        return self.settle_cached_result(
            row, obligation, accepted, fresh, key=key, waited=waited)

    def execute_cached_method(self, row, obligation, accepted, *, key):
        adapter_only = getattr(
            self.host.discharger, "discharge_adapter_only", None)
        if adapter_only is None:
            return None, "discharger has no authenticated adapter-only path"
        method = accepted.method
        if not getattr(self.host.discharger, "supports_method_reuse",
                       lambda _method: False)(method):
            return None, "adapter requires a fresh independent Judger audit"
        if not self.cacheable_signature(method, {
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
        structural = self.host.structural_defect(result, obligation)
        if structural:
            return None, structural
        if self.host.method_cache_key(row, obligation) != key:
            return None, "task authority changed during adapter execution"

        return result, ""

    def settle_cached_result(
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
        if not getattr(self.host.discharger, "supports_method_reuse",
                       lambda _method: False)(method):
            return None, "adapter requires a fresh independent Judger audit"
        if not self.cacheable_signature(method, {
                "sufficient": "yes", "faithful": "yes",
                "degradation": "n/a"}):
            return None, "cached MethodIR no longer meets reuse policy"
        if not basis_satisfies(method.get("basis", BASIS_RECORD),
                               obligation.required_basis):
            return None, "cached method is below the current evidence floor"
        structural = self.host.structural_defect(result, obligation)
        if structural:
            return None, structural
        if self.host.method_cache_key(row, obligation) != key:
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
        self.host.log.append("method_cache_hit", {
            "claim": row["id"], "obligation": obligation.id,
            "procedure": method["procedure"][:80],
            "source_binding_artifact": accepted.source_binding_artifact,
            "fresh_discharge_artifact": result.artifact,
            "source_outcome": accepted.source_outcome,
            "outcome": result.outcome,
            "waited_for_single_flight": waited,
        })
        return self.host.settle(
            row, obligation, [attempt], method, result, audit=audit), ""

    def audit_missing_direction(
        self, obligation, method, result,
    ):
        """Independently audit one fresh adapter result in a new direction.

        The opposite direction's signature supplies only the already accepted
        exact MethodIR. It never supplies a verdict. The current candidate's
        fresh evidence is structurally checked by ``_execute_cached_method``
        before this function, then receives the same full Judger prompt as an
        ordinary decisive attempt. Any non-complete audit falls through to
        the unchanged Oracle-led game.
        """
        attempt = Attempt(
            method=method,
            discharge_artifact=result.artifact,
            discharge_outcome=result.outcome,
        )
        degradation = degradation_record(
            method_strength(method), [attempt], {method_strength(method)})
        verdict = self.host.auditor.judge(
            obligation, method, result, attempt, degradation=degradation,
            method_attempt=1, independent_judger=True)
        if (verdict.get("sufficient") != "yes"
                or verdict.get("faithful") != "yes"
                or verdict.get("degradation") != "n/a"):
            return attempt, False, (
                "fresh opposite direction did not pass an independent "
                "complete Judger audit"
            )
        return attempt, True, ""

    @staticmethod
    def cacheable_signature(method, audit) -> bool:
        required = {"procedure", "artifact", "level", "basis", "strength",
                    "bridges", "adapter", "adapter_version",
                    "adapter_arguments", "adapter_inputs"}
        if set(method) != required:
            return False
        if (method.get("basis") != BASIS_WORLD
                or method_strength(method) != STRENGTH_DECISIVE
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

    def accepted_method(self, row, obligation, settled):
        """Extract one authenticated, cache-eligible binding artifact."""
        return self.accepted_method_or_reason(row, obligation, settled)[0]

    def accepted_method_or_reason(self, row, obligation, settled):
        """As :meth:`_accepted_method`, plus the reason when ineligible."""
        if settled.outcome not in (OUTCOME_HOLDS, OUTCOME_REFUTED):
            return None, f"outcome {settled.outcome!r} is not a direction"
        if settled.assumed_premises:
            return None, "result assumed premises"
        try:
            binding = self.host.artifacts.get(settled.artifact)
            if content_hash(binding) != settled.artifact \
                    or binding.get("instrument") != "binding-game" \
                    or binding.get("obligation") != obligation.text:
                return None, "binding artifact is not this obligation's game"
            bound = binding.get("bound")
            if not isinstance(bound, dict):
                return None, "no bound method"
            method, audit = bound.get("method"), bound.get("audit")
            if not isinstance(method, dict) or not isinstance(audit, dict):
                return None, "bound method or audit missing"
            if not self.cacheable_signature(method, audit):
                return None, "method signature is not reusable (needs a typed " \
                    "world-basis decisive adapter with delivered subjects and " \
                    "a yes/yes/n/a audit)"
            if not getattr(self.host.discharger, "supports_method_reuse",
                           lambda _method: False)(method):
                return None, "adapter does not accept method reuse"
            discharge_digest = bound.get("discharge_artifact")
            evidence = self.host.artifacts.get(discharge_digest)
            execution = evidence.get("execution")
            if not isinstance(execution, dict):
                return None, "discharge evidence has no execution record"
            inner = DischargeResult(
                outcome=execution.get("outcome"),
                witness=execution.get("witness"),
                artifact=discharge_digest,
                instrument=evidence.get("instrument", ""),
                witness_kind=execution.get("witness_kind"),
                inputs=tuple(execution.get("declared_inputs") or ()),
            )
            validator = getattr(
                getattr(self.host.discharger, "registry", None),
                "validate_evidence", None)
            if validator is None:
                return None, "discharger has no evidence validator"
            valid, reason = validator(
                method, row, obligation, inner, self.host.artifacts)
            if not valid:
                return None, f"evidence not authenticated: {reason}"
            if inner.outcome != settled.outcome:
                return None, "discharge outcome differs from settled outcome"
        except (OSError, TypeError, ValueError, KeyError,
                json.JSONDecodeError) as error:
            return None, f"eligibility check failed: {type(error).__name__}"
        return AcceptedMethod(
            method=method,
            source_binding_artifact=settled.artifact,
            source_discharge_artifact=discharge_digest,
            source_outcome=inner.outcome,
            audit=audit,
        ), ""
