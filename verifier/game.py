"""Binding game (M1, report §7/§8): strength-first method binding, two gates.

Per obligation:

1. Enumerate candidate methods (oracles agent probes the environment; a
   not-installed instrument is a one-time cost, not a boundary). Methods are
   mechanically re-sorted strongest evidence kind first — the agent's own
   ordering is advisory only.
2. Generate the exam: wrong variants of the claim, each self-confirmed false
   by the generator (with a disproof on record). No exam -> nothing can bind
   (fail-closed).
3. For each method, strongest reachable first:
   - Gate 1 (discrimination): run the method's discharge against every false
     variant. It must refute ALL of them; a single "holds" is a false pass
     and rejects the method; a "failed" means it could not sit the exam.
   - Discharge the real obligation. Execution failure after role-level
     retries == unreachable -> downgrade to the next method.
   - refuted short-circuits: a concrete counterexample is asymmetric (a wrong
     refutation costs a redo, a wrong pass costs correctness) and goes on
     record immediately with its witness.
   - Gate 2 (entailment audit): an independent agent opens the artifacts and
     judges whether the evidence logically discharges the obligation.
     entails=no -> downgrade.
4. All candidates exhausted -> failed, with every attempt on record.

Grades are mechanical: evidence_kind gives a ceiling (formal->proven,
execute/rederive->demonstrated, search/text->corroborated); the final grade is
the weaker of the auditor's grade and that ceiling. Generation influences
which method binds (efficiency); whether it binds is decided by gates.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .agents import (
    ENTAILMENT_AUDIT_AGENT, ENUMERATE_ORACLES_AGENT, GENERATE_VARIANTS_AGENT,
    VALIDATE_VARIANT_AGENT,
)
from .discharge import DischargeResult
from .events import EventLog
from .instruments import ArtifactStore
from .model import (
    Obligation, weakest_grade,
    OUTCOME_HOLDS, OUTCOME_REFUTED, OUTCOME_FAILED,
)
from .roles import RoleError

# strength order, strongest first (report §7: 强度优先)
KIND_RANK = {"formal": 0, "execute": 1, "rederive": 2, "search": 3, "text": 4}
GRADE_CEILING = {
    "formal": "proven",
    "execute": "demonstrated",
    "rederive": "demonstrated",
    "search": "corroborated",
    "text": "corroborated",
}


@dataclass
class Attempt:
    method: Dict[str, str]
    discrimination: List[Dict[str, Any]]
    rejected: Optional[str] = None          # reason, if the method was rejected
    discharge_artifact: Optional[str] = None
    discharge_outcome: Optional[str] = None
    audit: Optional[Dict[str, str]] = None


class BindingGame:
    def __init__(
        self,
        backend,
        artifacts: ArtifactStore,
        log: EventLog,
        discharger,
        oracles_role=ENUMERATE_ORACLES_AGENT,
        variants_role=GENERATE_VARIANTS_AGENT,
        entailment_role=ENTAILMENT_AUDIT_AGENT,
        validate_variant_role=VALIDATE_VARIANT_AGENT,
        n_variants: int = 2,
        max_methods: int = 3,
        registry: str = "[]",
        memory_hits: str = "[]",
        memory=None,
    ):
        self.backend = backend
        self.artifacts = artifacts
        self.log = log
        self.discharger = discharger
        self.oracles_role = oracles_role
        self.variants_role = variants_role
        self.entailment_role = entailment_role
        self.validate_variant_role = validate_variant_role
        self.n_variants = n_variants
        self.max_methods = max_methods
        self.registry = registry
        self.memory_hits = memory_hits
        self.memory = memory  # M4: advisory; feeds registry + attack hits

    # ---- entry ---------------------------------------------------------------

    def run(self, row: Dict, obligation: Obligation) -> DischargeResult:
        attempts: List[Attempt] = []
        methods = self._enumerate(row, obligation)
        variants = self._exam(row, obligation)
        self._variant_verdicts: Dict[str, bool] = {}   # exam quality memo

        if not methods:
            return self._fail(row, obligation, attempts, [],
                              "no candidate methods")
        if not variants:
            return self._fail(row, obligation, attempts, [],
                              "no discrimination exam")

        for method in methods:
            if not [v for v in variants if self._variant_usable(v)]:
                return self._fail(row, obligation, attempts, variants,
                                  "no usable discrimination exam")
            attempt = Attempt(method=method, discrimination=[])
            attempts.append(attempt)

            reason = self._gate1(row, obligation, method, variants, attempt)
            if reason:
                self._reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue

            result = self.discharger.discharge(
                row, obligation, method=method, tag="bound"
            )
            attempt.discharge_artifact = result.artifact
            attempt.discharge_outcome = result.outcome

            if result.outcome == OUTCOME_FAILED:
                reason = "execution failure at retry limit"
                self._reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue

            if result.outcome == OUTCOME_REFUTED:
                # asymmetric: counterexamples go on record immediately
                return self._settle(row, obligation, attempts, variants,
                                    method, result, audit=None,
                                    grade=GRADE_CEILING[method["evidence_kind"]])

            verdict = self._gate2(obligation, method, result, attempt)
            if verdict["entails"] != "yes":
                reason = f"entailment audit: {verdict['reason']}"
                self._reject(row, obligation, method, reason)
                attempt.rejected = reason
                continue

            grade = weakest_grade(
                [GRADE_CEILING[method["evidence_kind"]], verdict["grade"]]
            )
            return self._settle(row, obligation, attempts, variants,
                                method, result, audit=verdict, grade=grade)

        if not [v for v in variants if self._variant_usable(v)]:
            return self._fail(row, obligation, attempts, variants,
                              "no usable discrimination exam")
        return self._fail(row, obligation, attempts, variants,
                          "all candidate methods rejected")

    # ---- stages --------------------------------------------------------------

    def _enumerate(self, row: Dict, obligation: Obligation) -> List[Dict[str, str]]:
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
        methods = [
            m for m in payload["methods"] if m["evidence_kind"] in KIND_RANK
        ]
        methods.sort(key=lambda m: KIND_RANK[m["evidence_kind"]])
        methods = methods[: self.max_methods]
        self.log.append(
            "methods_enumerated",
            {"claim": row["id"], "obligation": obligation.id,
             "methods": [{"instrument_id": m["instrument_id"],
                          "evidence_kind": m["evidence_kind"]} for m in methods]},
        )
        return methods

    def _exam(self, row: Dict, obligation: Obligation) -> List[Dict[str, str]]:
        memory_hits = (self.memory.hits_for_attacks(row["text"])
                       if self.memory is not None else self.memory_hits)
        try:
            payload = self.variants_role.call(
                self.backend,
                {
                    "claim_text": row["text"],
                    "obligation_text": obligation.text,
                    "memory_hits": memory_hits,
                    "n": self.n_variants,
                },
            )
        except RoleError as err:
            self.log.append("variants_generated",
                            {"claim": row["id"], "obligation": obligation.id,
                             "error": str(err)})
            return []
        variants = [v for v in payload["variants"] if v["disproof"].strip()]
        self.log.append(
            "variants_generated",
            {"claim": row["id"], "obligation": obligation.id,
             "variants": [v["text"] for v in variants],
             "dropped_no_disproof": len(payload["variants"]) - len(variants)},
        )
        return variants

    def _gate1(self, row, obligation, method, variants, attempt) -> Optional[str]:
        """Discrimination: the method must refute every USABLE false variant.

        When a method holds on a variant, there are two suspects: the method
        (blind spot) or the exam question itself (the generator misconfirmed
        its falsity, or made it false in a way this obligation does not
        watch). Before rejecting the method, the question is audited ONCE by
        an independent agent that re-runs the disproof; only a question that
        fails its audit is disqualified. Fail-closed direction is preserved:
        an unclear audit keeps the variant and the method is rejected.
        """
        # An obligation is a FUNCTION of the claim, not a constant: "3 | n^5-n"
        # compiled for the original claim is a fact of arithmetic and holds no
        # matter what variant it is checked against. The exam must therefore
        # transport the obligation PATTERN onto the variant — check the
        # analogous instance, not the original text verbatim.
        exam_obligation = Obligation(
            obligation.claim_id,
            "【考题模式移植】下面是原 claim 的一条义务模式；当前 claim 是原 claim 的一个"
            "变体。把这条义务的同构实例应用到当前 claim 上执行检查（检查的是当前 claim"
            "的对应性质，不是原义务的字面内容）：\n" + obligation.text,
            obligation.failure_looks_like,
        )
        sat = 0
        for i, variant in enumerate(variants):
            if not self._variant_usable(variant):
                continue
            variant_row = {"id": row["id"], "text": variant["text"]}
            result = self.discharger.discharge(
                variant_row, exam_obligation, method=method, tag=f"exam{i}"
            )
            attempt.discrimination.append(
                {"variant": variant["text"], "outcome": result.outcome,
                 "artifact": result.artifact}
            )
            self.log.append(
                "discrimination",
                {"claim": row["id"], "obligation": obligation.id,
                 "instrument_id": method["instrument_id"],
                 "variant": variant["text"], "outcome": result.outcome},
            )
            if result.outcome == OUTCOME_HOLDS:
                if not self._audit_variant(row, obligation, variant):
                    continue   # question disqualified, not the method's fault
                if self.memory is not None:
                    # yesterday's escape becomes tomorrow's exam question
                    self.memory.record_attack(
                        row["text"], variant["text"],
                        f"false-passed by {method['instrument_id']}",
                        kind="false_pass_caught")
                return "false pass on a known-false variant"
            if result.outcome == OUTCOME_FAILED:
                return "could not sit the discrimination exam"
            sat += 1
        if sat == 0:
            # every question evaporated mid-exam: no exam == nothing binds
            return "no usable discrimination exam"
        return None

    def _variant_usable(self, variant) -> bool:
        return self._variant_verdicts.get(variant["text"], True)

    def _audit_variant(self, row, obligation, variant) -> bool:
        """-> True if the exam question stands (variant really is false)."""
        text = variant["text"]
        if text in self._variant_verdicts:
            return self._variant_verdicts[text]
        try:
            verdict = self.validate_variant_role.call(
                self.backend,
                {
                    "claim_text": row["text"],
                    "obligation_text": obligation.text,
                    "variant_text": text,
                    "disproof": variant["disproof"],
                },
            )
            stands = verdict["actually_false"] == "yes"
            reason = verdict["reason"]
        except RoleError as err:
            stands, reason = True, f"audit failed, question kept: {err}"
        self._variant_verdicts[text] = stands
        if not stands:
            self.log.append(
                "variant_disqualified",
                {"claim": row["id"], "obligation": obligation.id,
                 "variant": text, "reason": reason},
            )
        return stands

    def _gate2(self, obligation, method, result, attempt) -> Dict[str, str]:
        evidence = {
            "outcome": result.outcome,
            "witness": result.witness,
            "artifact_digest": result.artifact,
            "artifact_path": os.path.join(
                self.artifacts.root, result.artifact + ".json"
            ),
        }
        try:
            verdict = self.entailment_role.call(
                self.backend,
                {
                    "obligation_text": obligation.text,
                    "failure_looks_like": obligation.failure_looks_like,
                    "evidence": json.dumps(evidence, ensure_ascii=False),
                },
            )
        except RoleError as err:
            verdict = {"entails": "no", "grade": "",
                       "reason": f"audit failed: {err}"}  # fail-closed
        attempt.audit = verdict
        self.log.append(
            "entailment_audit",
            {"obligation": obligation.id,
             "instrument_id": method["instrument_id"], **verdict},
        )
        return verdict

    # ---- settlement ----------------------------------------------------------

    def _reject(self, row, obligation, method, reason: str) -> None:
        self.log.append(
            "method_rejected",
            {"claim": row["id"], "obligation": obligation.id,
             "instrument_id": method["instrument_id"], "reason": reason},
        )

    def _record(self, obligation, attempts, variants, bound) -> str:
        return self.artifacts.put(
            {
                "instrument": "binding-game",
                "obligation": obligation.text,
                "variants": variants,
                "attempts": [vars(a) for a in attempts],
                "bound": bound,
            }
        )

    def _settle(self, row, obligation, attempts, variants, method, result,
                audit, grade) -> DischargeResult:
        bound = {"method": method, "audit": audit,
                 "discharge_artifact": result.artifact}
        digest = self._record(obligation, attempts, variants, bound)
        if self.memory is not None:
            self.memory.record_instrument(
                method["instrument_id"], method["evidence_kind"],
                method["procedure"])
        self.log.append(
            "method_bound",
            {"claim": row["id"], "obligation": obligation.id,
             "instrument_id": method["instrument_id"],
             "evidence_kind": method["evidence_kind"], "grade": grade},
        )
        return DischargeResult(
            outcome=result.outcome,
            witness=result.witness,
            artifact=digest,
            instrument=method["instrument_id"],
            grade=grade,
        )

    def _fail(self, row, obligation, attempts, variants, reason) -> DischargeResult:
        digest = self._record(obligation, attempts, variants, None)
        return DischargeResult(
            outcome=OUTCOME_FAILED,
            witness=reason,
            artifact=digest,
            instrument="binding-game",
            grade="corroborated",
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
                w in text for w in ("所有", "任意", "每个", "存在", "all", "every", "exists")
            ),
        },
        ensure_ascii=False,
    )
