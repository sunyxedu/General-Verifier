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
        self.binder = binder  # M1 binding game; None -> direct discharge
        store.on_trigger(self.verify)  # the single rule wires straight in

    # ---- entry ---------------------------------------------------------------

    def verify(self, claim_id: str) -> str:
        """Verify one claim; returns its resulting status. Idempotent."""
        row = self.store.get(claim_id)
        if row["status"] not in (UNVERIFIED, STALE):
            return row["status"]
        if row["kind"] == KIND_EFFECT:
            self._verify_effect(row)
        else:
            self._verify_assertion(row)
        return self.store.status(claim_id)

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

        outcomes: List[str] = []
        grades: List[str] = []
        for obligation in obligations:
            outcome, grade = self._discharge(row, obligation)
            outcomes.append(outcome)
            grades.append(grade)
            if outcome != OUTCOME_HOLDS:
                break  # refuted or failed: no point burning further checks
        self._settle(row["id"], outcomes, grade=weakest_grade(grades))

    def _compile_obligations(self, row: Dict):
        payload = self.obligations_role.call(
            self.backend, {"claim_text": row["text"], "memory_hits": self.memory_hits}
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
