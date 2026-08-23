"""
Binding game: per obligation, bind one sufficient method to real evidence.

Per obligation:

1. The oracle enumerates SUFFICIENT methods only (sufficiency is the
   oracle's own guarantee, researched first when the domain is
   unfamiliar) and grades each with an assurance level of its own
   devising, listing them ascending. Simplest sufficient first, most
   assured last. Execution follows the oracle's order exactly.
2. For each method in order the discharge executor carries it out
   faithfully (a missing instrument is installed at discharge time; a
   truly unobtainable one settles the claim at partial).
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
3. All candidates exhausted -> failed (or unobtainable when a missing
   instrument blocked the way), with every attempt on record.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..llm.agents import ENUMERATE_ORACLES_AGENT, JUDGER_AGENT
from ..execution.discharge import DischargeResult
from ..core.events import EventLog
from ..execution.instruments import (
    ArtifactStore, authoritative_task_context, world_evidence_context,
)
from ..core.model import (
    BASIS_RECORD, Obligation, basis_satisfies,
    OUTCOME_REFUTED, OUTCOME_FAILED, OUTCOME_UNOBTAINABLE,
)
from ..llm.schema import RoleError


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
    ):
        self.backend = backend
        self.artifacts = artifacts
        self.log = log
        self.discharger = discharger
        self.oracles_role = oracles_role
        self.judger_role = judger_role
        self.redo_rounds = redo_rounds
        self.registry = registry
        self.memory = memory  # advisory; feeds the oracle's registry slot
        self.live_world = live_world

    # Entry point
    def run(self, row: Dict, obligation: Obligation) -> DischargeResult:
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

        if not methods:
            # In offline replay the evidence boundary is frozen: a valid empty
            # list means the oracle found no executable sufficient method at
            # the required basis, so the honest result is unavailable rather
            # than a retryable execution failure.  A live world may change, so
            # preserve its fail-closed/retryable behaviour.
            return self._fail(
                row, obligation, attempts,
                "oracle found no executable sufficient method at the "
                f"required basis {obligation.required_basis!r}",
                unobtainable=not self.live_world,
            )

        # the obligation's evidence floor is mechanical, not advisory: a
        # method whose declared basis cannot satisfy it never executes —
        # a record method faithfully run against a world-floor obligation
        # would only prove the solver's reporting, which is exactly the
        # evidence this floor exists to reject
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

        unavailable_reasons: List[str] = []
        failed_seen = False
        for method in methods:
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
                reason = f"unobtainable: {detail}"
                self._reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue

            if result.outcome == OUTCOME_FAILED:
                failed_seen = True
                reason = "execution failure at retry limit"
                self._reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue

            # every substantive outcome — holds AND refuted — faces the
            # judger before it settles, for both method sufficiency and
            # execution fidelity. A refutation is the most
            # consequential verdict in the system (a refuted node refutes
            # its claim — for the final, the whole certificate), so an
            # unaudited one is exactly the evidence we cannot afford to
            # accept.
            verdict = self._judge(obligation, method, result, attempt)
            if verdict.get("sufficient") == "no":
                failed_seen = True
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
                    reason = f"unobtainable (redo {rounds}): {detail}"
                    break
                if result.outcome == OUTCOME_FAILED:
                    failed_seen = True
                    reason = f"execution failure at retry limit (redo {rounds})"
                    break
                verdict = self._judge(obligation, method, result, attempt,
                                      redo=rounds)
                if verdict.get("sufficient") == "no":
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

        unavailable_only = bool(unavailable_reasons) and not failed_seen
        reason = (
            "all candidate methods structurally unavailable: "
            + "; ".join(dict.fromkeys(unavailable_reasons))
            if unavailable_only else "all candidate methods rejected"
        )
        return self._fail(row, obligation, attempts, reason,
                          unobtainable=unavailable_only)

    # enumeration and advisory review
    def _enumerate(self, row: Dict,
                   obligation: Obligation) -> List[Dict[str, str]]:
        registry = (self.memory.registry_summary()
                    if self.memory is not None else self.registry)
        state = os.path.dirname(os.path.abspath(self.log.path))
        world_mode, world, manifest = world_evidence_context(
            state, live_world=self.live_world)
        payload = self.oracles_role.call(
            self.backend,
            {
                "claim_text": row["text"],
                "obligation_text": obligation.text,
                "required_basis": obligation.required_basis,
                "task_context": authoritative_task_context(state),
                "features": _features(row),
                "registry": registry,
                "ledger_path": os.path.abspath(self.log.path),
                "world_mode": world_mode,
                "world_path": world,
                "world_manifest": manifest,
            },
        )
        methods = payload["methods"]
        self.log.append(
            "methods_enumerated",
            {"claim": row["id"], "obligation": obligation.id,
             "methods": [{"procedure": m["procedure"][:80],
                          "level": m.get("level", ""),
                          "basis": m.get("basis", "")}
                         for m in methods]},
        )
        return methods

    # the judger
    def _judge(self, obligation, method, result, attempt,
               redo: int = 0) -> Dict[str, str]:
        evidence = {
            "outcome": result.outcome,
            "witness": result.witness,
            "artifact_digest": result.artifact,
            "artifact_path": os.path.join(
                self.artifacts.root, result.artifact + ".json"
            ),
        }
        # same state-dir derivation as the kernel: the judger audits
        # refutations against the delivered world and its capture
        # manifest, so it needs the same paths the executor had
        state = os.path.dirname(os.path.abspath(self.log.path))
        world_mode, world, manifest = world_evidence_context(
            state, live_world=self.live_world)
        try:
            verdict = self.judger_role.call(
                self.backend,
                {
                    "obligation_text": obligation.text,
                    "procedure": method["procedure"],
                    "promised_artifact": method.get("artifact", ""),
                    # the oracle's declared evidence basis — the judger
                    # audits it as part of fidelity, and it is what the
                    # settled verdict will carry downstream
                    "basis": method.get("basis", ""),
                    "task_context": authoritative_task_context(state),
                    "ledger_path": os.path.abspath(self.log.path),
                    "world_mode": world_mode,
                    "world_path": world,
                    "world_manifest": manifest,
                    "evidence": json.dumps(evidence, ensure_ascii=False),
                },
            )
        except RoleError as err:
            # fail-closed; the _infra marker keeps a judger infrastructure
            # failure from awarding itself a redo
            verdict = {
                # Internal fail-closed sentinel; public role output permits
                # only yes/no, so neither signature can accidentally pass.
                "sufficient": "unknown",
                "faithful": "no",
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
        One arm of the judger-discharger loop: re-execute with every
        judger reason so far attached (latest last); the result goes back
        for a blind re-judge.
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
        notes = "".join(
            f"\n\n[Redo round {i + 1} — the judger rejected the previous "
            f"attempt; fix it and redo. Judger reason: {r['reason']}]"
            for i, r in enumerate(attempt.redos)
        )
        method_fb = {**method, "procedure": method["procedure"] + notes}
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
             "level": method.get("level", "")},
        )
        return DischargeResult(
            outcome=result.outcome,
            witness=result.witness,
            artifact=digest,
            instrument=method["procedure"][:80],
            # the oracle's declared basis, already audited by the judger as
            # part of fidelity; absent (legacy fixtures) fails closed to record
            basis=method.get("basis", BASIS_RECORD),
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
