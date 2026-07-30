"""M0 kernel (components 14, 15, 20 + mechanical slice of 21).

Scope per report §16: obligations -> falsifiability filter -> discharge via a
single instrument -> mechanical audit bits -> status. The discrimination game,
variant generation and the two full audits arrive in M1; every seam they will
plug into is already an explicit step here.

All kernel activity is appended to the same event stream the solver's activity
came from — the verifier is auditable through its own ledger.
"""
from __future__ import annotations

from typing import Dict, List

from .discharge import SandboxDischarger
from .events import EventLog
from .instruments import ArtifactStore, LedgerInstrument, PythonSandbox
from .model import (
    Claim, Evidence, Obligation, weakest_grade,
    KIND_EFFECT, UNVERIFIED, STALE, REFUTED,
    OUTCOME_HOLDS, OUTCOME_REFUTED, OUTCOME_FAILED,
)
from .roles import COMPILE_OBLIGATIONS
from .store import ClaimStore


class Kernel:
    """Role seats (obligations compiler, discharger) are injectable: the same
    kernel runs on fixture-backed text roles (offline, deterministic) or on
    Claude Agent SDK agents (see verifier.agents / verifier.discharge) without
    knowing which is seated."""

    def __init__(
        self,
        store: ClaimStore,
        log: EventLog,
        backend,
        artifacts: ArtifactStore,
        sandbox: PythonSandbox = None,
        memory_hits: str = "[]",
        obligations_role=None,
        discharger=None,
        binder=None,
        coverage=None,
        memory=None,
        max_obligations: int = 16,
    ):
        self.store = store
        self.log = log
        self.backend = backend
        self.artifacts = artifacts
        self.sandbox = sandbox or PythonSandbox()
        self.ledger = LedgerInstrument(log)
        self.memory_hits = memory_hits
        self.obligations_role = obligations_role or COMPILE_OBLIGATIONS
        self.discharger = discharger or SandboxDischarger(
            backend, self.sandbox, artifacts
        )
        self.binder = binder      # M1 binding game; None -> direct discharge
        self.coverage = coverage  # M1 coverage audit; None -> no exam
        self.memory = memory      # M4 advisory stores; None -> static hints
        # termination controller (component 27): every loop in the kernel is
        # bounded by construction; this is the one global cap — a pathological
        # compilation cannot stall verification, it gets an honest
        # unverified(bounds) instead. Strength-first is untouched: the cap is
        # on breadth (obligation count), never on instrument strength.
        self.max_obligations = max_obligations
        store.on_trigger(self.verify)  # the single rule wires straight in

    # ---- entry ---------------------------------------------------------------

    def verify(self, claim_id: str) -> str:
        """Verify one claim; returns its resulting status. Idempotent."""
        row = self.store.get(claim_id)
        if row["status"] not in (UNVERIFIED, STALE):
            return row["status"]
        if row["kind"] != KIND_EFFECT and self._dedup(row):
            return self.store.status(claim_id)
        if row["kind"] == KIND_EFFECT:
            self._verify_effect(row)
        else:
            self._verify_assertion(row)
        return self.store.status(claim_id)

    def _dedup(self, row: Dict) -> bool:
        """Cross-branch content-addressed reuse: identical (kind, text, deps)
        already settled on another branch -> transfer its verdict + evidence
        instead of re-spending the instruments."""
        other = self.store.find_equivalent(row["id"])
        if other is None:
            return False
        other_row = self.store.get(other)
        self.store.record_evidence(
            row["id"],
            {"outcome": OUTCOME_HOLDS if other_row["status"] == "verified"
             else OUTCOME_REFUTED,
             "instrument": "dedup", "source": other,
             "grade": other_row["grade"]},
        )
        if other_row["status"] == "verified":
            self.store.set_verified(row["id"], other_row["grade"])
        else:
            self.store.set_refuted(row["id"], reason=f"dedup of {other[:12]}")
            self._prune(row["id"])
        self.log.append("dedup_hit", {"claim": row["id"], "source": other,
                                      "status": other_row["status"]})
        return True

    # ---- effect: ledger check (component 25) ---------------------------------

    def _verify_effect(self, row: Dict) -> None:
        result = self.ledger.check(row["text"])
        digest = self.artifacts.put(result.artifacts)
        self.store.record_evidence(
            row["id"],
            {
                "obligation": "ledger:event-happened-as-recorded",
                "instrument": self.ledger.name,
                "outcome": result.outcome,
                "artifact": digest,
                "witness": result.witness,
            },
        )
        self.log.append(
            "discharge",
            {"claim": row["id"], "instrument": self.ledger.name, "outcome": result.outcome},
        )
        self._settle(row["id"], [result.outcome], grade="demonstrated")

    # ---- assertion / final ---------------------------------------------------

    def _verify_assertion(self, row: Dict) -> None:
        refuted_deps = [
            p for p in self.store.parents(row["id"])
            if self.store.status(p) == REFUTED
        ]
        if refuted_deps:
            self.store.record_evidence(
                row["id"],
                {"outcome": OUTCOME_FAILED, "error": "depends on refuted claim",
                 "refuted_deps": refuted_deps},
            )
            self.log.append(
                "status_change",
                {"claim": row["id"], "status": UNVERIFIED,
                 "reason": "depends on refuted claim"},
            )
            return

        obligations, dropped = self._compile_obligations(row)
        if not obligations:
            # fail-closed: nothing checkable means nothing verified
            self.store.record_evidence(
                row["id"],
                {"outcome": OUTCOME_FAILED, "error": "no falsifiable obligations"},
            )
            self.log.append(
                "status_change",
                {"claim": row["id"], "status": UNVERIFIED,
                 "reason": "no falsifiable obligations"},
            )
            return

        if self.coverage is not None and obligations:
            exam_discharger = (
                self.binder.discharger if self.binder else self.discharger
            )
            obligations.extend(
                self.coverage.audit(row, obligations, exam_discharger)
            )

        if len(obligations) > self.max_obligations:
            kept_n = self.max_obligations
            self.store.record_evidence(
                row["id"],
                {"outcome": OUTCOME_FAILED,
                 "error": f"bounds: {len(obligations)} obligations exceed "
                          f"cap {kept_n}; verification not attempted"},
            )
            self.log.append(
                "status_change",
                {"claim": row["id"], "status": UNVERIFIED,
                 "reason": "bounds: obligations cap"},
            )
            return

        outcomes: List[str] = []
        grades: List[str] = []
        refuting_witness = None
        for obligation in obligations:
            outcome, grade = self._discharge(row, obligation)
            outcomes.append(outcome)
            grades.append(grade)
            if outcome != OUTCOME_HOLDS:
                if outcome == OUTCOME_REFUTED:
                    ev = self.store.evidence_for(row["id"])
                    refuting_witness = ev[-1].get("witness") if ev else None
                break  # refuted or failed: no point burning further checks
        self._settle(row["id"], outcomes, grade=weakest_grade(grades))

        if self.memory is not None:
            status = self.store.status(row["id"])
            self.memory.record_obligations(
                row["text"], [o.text for o in obligations], status)
            if status == REFUTED:
                self.memory.record_attack(
                    row["text"], row["text"],
                    str(refuting_witness), kind="refutation")
            self.log.append("memory_recorded", {"claim": row["id"]})

    def _compile_obligations(self, row: Dict):
        memory_hits = (
            self.memory.hits_for_obligations(row["text"])
            if self.memory is not None else self.memory_hits
        )
        payload = self.obligations_role.call(
            self.backend, {"claim_text": row["text"], "memory_hits": memory_hits}
        )
        all_obligations = [
            Obligation(row["id"], o["text"], o["failure_looks_like"])
            for o in payload["obligations"]
        ]
        kept = [o for o in all_obligations if o.falsifiable]
        dropped = [o.text for o in all_obligations if not o.falsifiable]
        self.log.append(
            "obligations_compiled",
            {"claim": row["id"], "kept": [o.text for o in kept], "dropped": dropped},
        )
        return kept, dropped

    def _discharge(self, row: Dict, obligation: Obligation):
        if self.binder is not None:
            result = self.binder.run(row, obligation)
        else:
            result = self.discharger.discharge(row, obligation)
        evidence = Evidence(
            obligation_id=obligation.id,
            instrument=result.instrument,
            outcome=result.outcome,
            artifacts={"artifact": result.artifact},
            witness=result.witness,
        )
        self.store.record_evidence(
            row["id"],
            {
                "obligation": obligation.text,
                "obligation_id": obligation.id,
                "instrument": result.instrument,
                "outcome": result.outcome,
                "artifact": result.artifact,
                "witness": result.witness,
                "grade": result.grade,
                "evidence_id": evidence.id,
            },
        )
        self.log.append(
            "discharge",
            {"claim": row["id"], "obligation": obligation.id,
             "instrument": result.instrument, "outcome": result.outcome},
        )
        return result.outcome, result.grade

    # ---- settlement ----------------------------------------------------------

    def _settle(self, claim_id: str, outcomes: List[str], grade: str) -> None:
        if any(o == OUTCOME_REFUTED for o in outcomes):
            self.store.set_refuted(claim_id, reason="counterexample on record")
            new_status = "refuted"
        elif all(o == OUTCOME_HOLDS for o in outcomes) and outcomes:
            self.store.set_verified(claim_id, grade)
            new_status = "verified"
        else:
            new_status = UNVERIFIED  # failed discharge: stays unverified, on record
        self.log.append("status_change", {"claim": claim_id, "status": new_status})
        if new_status == "refuted":
            self._prune(claim_id)

    def _prune(self, claim_id: str) -> None:
        """Asymmetric pruning (M3): a refutation hard-kills its downstream
        cone for the SEARCHER. Advisory: statuses are untouched (the taint
        guard already refuses to build on refuted claims); this event just
        tells the search loop where to stop spending."""
        cone = self.store.downstream(claim_id)
        branches = sorted(
            {self.store.get(c)["branch"] for c in [claim_id, *cone]}
            - {None}
        )
        if cone or branches:
            self.log.append(
                "branch_pruned",
                {"refuted": claim_id, "dead_cone": cone, "branches": branches},
            )
