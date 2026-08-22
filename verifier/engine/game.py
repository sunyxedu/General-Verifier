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
     alike: was the bound method faithfully executed against the right
     target, and the promised evidence genuinely delivered? It opens
     the artifact itself and reruns when reproducibility is in doubt.
     A refutation gets no shortcut: it is the single most consequential
     outcome in the system (a refuted node refutes its claim — for the
     final, the whole certificate), so a counterexample settles only
     after the judger confirms it was faithfully obtained. Unfaithful -> the judger's
     reason goes back to the executor for a redo and the redone result
     is re-judged BLIND. A loop of at most three rounds.
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
from ..execution.instruments import ArtifactStore
from ..core.model import (
    BASIS_RECORD, Obligation,
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

    # Entry point
    def run(self, row: Dict, obligation: Obligation) -> DischargeResult:
        attempts: List[Attempt] = []
        methods = self._enumerate(row, obligation)
        if not methods:
            return self._fail(row, obligation, attempts,
                              "no candidate methods")

        unobtainable_seen = False   # a missing, uninstallable instrument
                                    # blocked a candidate -> settles partial
        for method in methods:
            attempt = Attempt(method=method)
            attempts.append(attempt)

            result = self.discharger.discharge(
                row, obligation, method=method, tag="bound"
            )
            attempt.discharge_artifact = result.artifact
            attempt.discharge_outcome = result.outcome

            if result.outcome == OUTCOME_UNOBTAINABLE:
                unobtainable_seen = True
                reason = "instrument unobtainable"
                self._reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue

            if result.outcome == OUTCOME_FAILED:
                reason = "execution failure at retry limit"
                self._reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue

            # every substantive outcome — holds AND refuted — faces the
            # judger before it settles. A refutation is the most
            # consequential verdict in the system (a refuted node refutes
            # its claim — for the final, the whole certificate), so an
            # unaudited one is exactly the evidence we cannot afford to
            # accept.
            verdict = self._judge(obligation, method, result, attempt)
            if verdict["faithful"] == "yes":
                return self._settle(row, obligation, attempts,
                                    method, result, audit=verdict)

            # judger-discharger loop: an unfaithful delivery hands the
            # judger's reason back to the executor for a redo, and the
            # redone evidence is re-judged BLIND. At most redo_rounds rounds.
            # (a judger infrastructure failure never awards a redo)
            reason = None
            rounds = 0
            while (verdict["faithful"] == "no"
                   and not verdict.get("_infra")
                   and rounds < self.redo_rounds):
                rounds += 1
                result = self._redo(row, obligation, method, verdict,
                                    attempt, rounds)
                if result.outcome == OUTCOME_UNOBTAINABLE:
                    unobtainable_seen = True
                    reason = f"instrument unobtainable (redo {rounds})"
                    break
                if result.outcome == OUTCOME_FAILED:
                    reason = f"execution failure at retry limit (redo {rounds})"
                    break
                verdict = self._judge(obligation, method, result, attempt,
                                      redo=rounds)
                if verdict["faithful"] == "yes":
                    return self._settle(row, obligation, attempts,
                                        method, result, audit=verdict)
            if reason is None:
                reason = (f"judger after {rounds} redo round(s): "
                          f"{verdict['reason']}"
                          if rounds else f"judger: {verdict['reason']}")
            self._reject(row, obligation, method, reason)
            attempt.rejected = reason

        return self._fail(row, obligation, attempts,
                          "all candidate methods rejected",
                          unobtainable=unobtainable_seen)

    # enumeration and advisory review
    def _enumerate(self, row: Dict,
                   obligation: Obligation) -> List[Dict[str, str]]:
        registry = (self.memory.registry_summary()
                    if self.memory is not None else self.registry)
        try:
            payload = self.oracles_role.call(
                self.backend,
                {
                    "claim_text": row["text"],
                    "obligation_text": obligation.text,
                    "features": _features(row),
                    "registry": registry,
                },
            )
        except RoleError as err:
            self.log.append("methods_enumerated",
                            {"claim": row["id"], "obligation": obligation.id,
                             "error": str(err)})
            return []
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
        try:
            verdict = self.judger_role.call(
                self.backend,
                {
                    "obligation_text": obligation.text,
                    "procedure": method["procedure"],
                    "promised_artifact": method.get("artifact", ""),
                    "evidence": json.dumps(evidence, ensure_ascii=False),
                },
            )
        except RoleError as err:
            # fail-closed; the _infra marker keeps a judger infrastructure
            # failure from awarding itself a redo
            verdict = {"faithful": "no", "reason": f"judger failed: {err}",
                       "_infra": True}
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
