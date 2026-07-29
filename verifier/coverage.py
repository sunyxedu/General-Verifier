"""Coverage audit (验漏, loop 3 of report §8): exam the OBLIGATION SET.

Loop 1 asks "does this method discriminate for this obligation"; this loop
asks "do the obligations as a whole cover the claim's failure modes". Both
share the variant generator (the report's honest "两层半" note): here the
variants target the claim as a whole rather than one obligation. For each
self-confirmed-false variant, every current obligation gets a discharge run
against it; if NO obligation refutes a false variant, that is a coverage
hole, and a patch role writes the missing obligation that would have caught
it (gated by the same falsifiability filter as everything else).

Coverage failure is fail-OPEN with a logged trace: this component only ever
adds checks, so its absence degrades coverage, not soundness — unlike a
discharge, it fabricates no evidence.
"""
from __future__ import annotations

from typing import Dict, List

from .agents import GENERATE_VARIANTS_AGENT
from .events import EventLog
from .model import Obligation, OUTCOME_REFUTED
from .roles import COVERAGE_PATCH, RoleError

WHOLE_CLAIM_EXAM = "（本轮考察义务集的整体覆盖，不针对单条义务：生成整体上似是而非的错误变体）"


class CoverageAudit:
    def __init__(self, backend, log: EventLog,
                 variants_role=GENERATE_VARIANTS_AGENT,
                 patch_role=COVERAGE_PATCH,
                 n_variants: int = 2, memory_hits: str = "[]"):
        self.backend = backend
        self.log = log
        self.variants_role = variants_role
        self.patch_role = patch_role
        self.n_variants = n_variants
        self.memory_hits = memory_hits

    def audit(self, row: Dict, obligations: List[Obligation],
              discharger) -> List[Obligation]:
        try:
            payload = self.variants_role.call(
                self.backend,
                {"claim_text": row["text"], "obligation_text": WHOLE_CLAIM_EXAM,
                 "memory_hits": self.memory_hits, "n": self.n_variants},
            )
        except RoleError as err:
            self.log.append("coverage_exam",
                            {"claim": row["id"], "error": str(err)})
            return []

        added: List[Obligation] = []
        existing = {o.text for o in obligations}
        for variant in payload["variants"]:
            if not variant["disproof"].strip():
                continue
            covered_by = None
            for obligation in obligations:
                result = discharger.discharge(
                    {"id": row["id"], "text": variant["text"]},
                    obligation, tag="coverage",
                )
                if result.outcome == OUTCOME_REFUTED:
                    covered_by = obligation.text
                    break
            self.log.append(
                "coverage_exam",
                {"claim": row["id"], "variant": variant["text"],
                 "covered": covered_by is not None, "covered_by": covered_by},
            )
            if covered_by is not None:
                continue

            patch = self._patch(row, variant)
            if patch is None or not patch.falsifiable or patch.text in existing:
                continue
            existing.add(patch.text)
            added.append(patch)
            self.log.append(
                "obligations_added",
                {"claim": row["id"], "obligation": patch.text,
                 "hole_variant": variant["text"]},
            )
        return added

    def _patch(self, row: Dict, variant: Dict[str, str]):
        try:
            payload = self.patch_role.call(
                self.backend,
                {"claim_text": row["text"], "variant_text": variant["text"],
                 "disproof": variant["disproof"]},
            )
        except RoleError:
            return None
        return Obligation(row["id"], payload["text"], payload["failure_looks_like"])
