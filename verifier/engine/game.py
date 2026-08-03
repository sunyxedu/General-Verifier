"""Binding game: per obligation, bind one sufficient method to real evidence.

Per obligation:

1. The oracle enumerates SUFFICIENT methods only (sufficiency is the
   oracle's own guarantee, researched first when the domain is
   unfamiliar) and grades each with an assurance level L1 (simplest
   sufficient) -> L3 (machine-checked). Execution proceeds by level
   ascending: simplest first.
2. For each method in order the discharge executor carries it out
   faithfully (a missing instrument is installed at discharge time; a
   truly unobtainable one settles the claim at partial).
   - refuted short-circuits: a concrete counterexample is asymmetric (a
     wrong refutation costs a redo, a wrong pass costs correctness) and
     goes on record immediately with its witness.
   - The judger examines the RESULT: was the bound method faithfully
     executed and the promised artifact delivered? It opens the artifact
     itself and reruns when reproducibility is in doubt. Unfaithful ->
     one redo with the judger's reason attached, then a blind re-judge.
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
    Obligation,
    OUTCOME_REFUTED, OUTCOME_FAILED, OUTCOME_UNOBTAINABLE,
)
from ..llm.roles import RoleError


@dataclass
class Attempt:
    method: Dict[str, str]
    rejected: Optional[str] = None          # reason, if the method was rejected
    discharge_artifact: Optional[str] = None
    discharge_outcome: Optional[str] = None
    audit: Optional[Dict[str, str]] = None       # judger's first verdict
    redo_outcome: Optional[str] = None           # one bounded redo, if granted
    redo_artifact: Optional[str] = None
    redo_audit: Optional[Dict[str, str]] = None  # blind re-judge verdict


class BindingGame:
    def __init__(
        self,
        backend,
        artifacts: ArtifactStore,
        log: EventLog,
        discharger,
        oracles_role=ENUMERATE_ORACLES_AGENT,
        judger_role=JUDGER_AGENT,
        redo_rounds: int = 1,
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

    # ---- entry ---------------------------------------------------------------

    def run(self, row: Dict, obligation: Obligation) -> DischargeResult:
        attempts: List[Attempt] = []
        methods = self._enumerate(row, obligation)
        if not methods:
            return self._fail(row, obligation, attempts,
                              "no candidate methods")
        methods = self._order_by_level(methods)

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

            if result.outcome == OUTCOME_REFUTED:
                return self._settle(row, obligation, attempts,
                                    method, result, audit=None)

            verdict = self._judge(obligation, method, result, attempt)
            if verdict["faithful"] == "yes":
                return self._settle(row, obligation, attempts,
                                    method, result, audit=verdict)

            # judger loop: an unfaithful delivery earns exactly one redo
            # with the judger's reason attached; the redone evidence is
            # then re-judged BLIND. (a judger infrastructure failure never
            # awards a redo)
            if (not verdict.get("_infra")
                    and self.redo_rounds > 0
                    and attempt.redo_outcome is None):
                result2 = self._redo(row, obligation, method, verdict, attempt)
                if result2.outcome == OUTCOME_REFUTED:
                    return self._settle(row, obligation, attempts,
                                        method, result2, audit=None)
                if result2.outcome == OUTCOME_UNOBTAINABLE:
                    unobtainable_seen = True
                    reason = "instrument unobtainable (redo)"
                elif result2.outcome == OUTCOME_FAILED:
                    reason = "execution failure at retry limit (redo)"
                else:
                    verdict2 = self._judge(obligation, method, result2,
                                           attempt, redo=True)
                    if verdict2["faithful"] == "yes":
                        return self._settle(row, obligation, attempts,
                                            method, result2, audit=verdict2)
                    reason = f"judger after redo: {verdict2['reason']}"
            else:
                reason = f"judger: {verdict['reason']}"
            self._reject(row, obligation, method, reason)
            attempt.rejected = reason

        return self._fail(row, obligation, attempts,
                          "all candidate methods rejected",
                          unobtainable=unobtainable_seen)

    # ---- enumeration and advisory review -------------------------------------

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
                          "level": m.get("level", "")}
                         for m in methods]},
        )
        return methods

    _LEVEL_RANK = {"L1": 0, "L2": 1, "L3": 2}   # simplest sufficient first

    def _order_by_level(self, methods):
        """Mechanical order from the oracle's own level declarations."""
        return sorted(methods,
                      key=lambda m: self._LEVEL_RANK.get(m.get("level"), 1))

    # ---- the judger ----------------------------------------------------------

    def _judge(self, obligation, method, result, attempt,
               redo: bool = False) -> Dict[str, str]:
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
            attempt.redo_audit = verdict
        else:
            attempt.audit = verdict
        public = {k: v for k, v in verdict.items() if not k.startswith("_")}
        self.log.append(
            "judger_verdict",
            {"obligation": obligation.id, "redo": redo,
             "procedure": method["procedure"][:80], **public},
        )
        return verdict

    def _redo(self, row, obligation, method, verdict, attempt):
        """The judger loop's arm: re-execute once with the judger's reason
        attached; the result goes back for a blind re-judge."""
        self.log.append(
            "method_redo",
            {"claim": row["id"], "obligation": obligation.id,
             "procedure": method["procedure"][:80],
             "reason": verdict["reason"][:200]},
        )
        method_fb = {
            **method,
            "procedure": method["procedure"]
            + "\n\n[Previous attempt was rejected by the judger; fix it and "
              "redo. Judger reason: " + verdict["reason"] + "]",
        }
        result2 = self.discharger.discharge(
            row, obligation, method=method_fb, tag="bound-redo")
        attempt.redo_outcome = result2.outcome
        attempt.redo_artifact = result2.artifact
        return result2

    # ---- settlement ----------------------------------------------------------

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
    """Mechanical claim features — no interpretation, just surface signals."""
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
