"""Kernel.

Compile obligations (graph-aware, answer-blind) -> materialize each as an
obligation NODE in the claim graph (identical text anywhere = same node, so
settled verdicts transfer for free) -> discharge each node through the
binding game (oracle levels -> execute -> judger) -> settle the claim as a
fold over its obligation nodes.

All kernel activity is appended to the same event stream the solver's activity
came from — the verifier is auditable through its own ledger.
"""
from __future__ import annotations

import json
from typing import Dict, List

from ..execution.discharge import SandboxDischarger
from ..core.events import EventLog
from ..execution.instruments import ArtifactStore, LedgerInstrument, PythonSandbox
from ..core.model import (
    Claim, Evidence, Obligation,
    KIND_EFFECT, UNVERIFIED, VERIFIED, STALE, REFUTED, PARTIAL,
    OUTCOME_HOLDS, OUTCOME_REFUTED, OUTCOME_FAILED, OUTCOME_UNOBTAINABLE,
)
from ..llm.roles import COMPILE_OBLIGATIONS
from ..core.store import ClaimStore


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
        memory=None,
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
        self.binder = binder      # binding game; None -> direct discharge
        self.memory = memory      # advisory stores; None -> static hints
        store.on_trigger(self.verify)  # the single rule wires straight in

    # ---- entry ---------------------------------------------------------------

    def verify(self, claim_id: str) -> str:
        """Verify one claim; returns its resulting status. Idempotent."""
        row = self.store.get(claim_id)
        if row["status"] not in (UNVERIFIED, STALE, PARTIAL):
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
             "instrument": "dedup", "source": other},
        )
        if other_row["status"] == "verified":
            self.store.set_verified(row["id"])
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
        self._settle(row["id"], [result.outcome])

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

        obligations = self._compile_obligations(row)
        if not obligations:
            # fail-closed: nothing checkable means nothing verified
            self.store.record_evidence(
                row["id"],
                {"outcome": OUTCOME_FAILED, "error": "no obligations"},
            )
            self.log.append(
                "status_change",
                {"claim": row["id"], "status": UNVERIFIED,
                 "reason": "no obligations"},
            )
            return

        outcomes: List[str] = []
        refuting_witness = None
        for obligation in obligations:
            oid = self.store.add_obligation(row["id"], obligation)
            known = self.store.status(oid)
            if known in (VERIFIED, REFUTED, PARTIAL):
                # content-addressed reuse: the same world-condition was
                # already settled under another claim — the verdict and its
                # evidence transfer without re-spending instruments
                outcome = {VERIFIED: OUTCOME_HOLDS,
                           REFUTED: OUTCOME_REFUTED,
                           PARTIAL: OUTCOME_UNOBTAINABLE}[known]
                self.log.append("dedup_hit",
                                {"claim": row["id"], "obligation": oid,
                                 "status": known})
                outcomes.append(outcome)
                if outcome == OUTCOME_REFUTED:
                    ev = self.store.evidence_for(oid)
                    refuting_witness = ev[-1].get("witness") if ev else None
                    break
                continue
            src = self.store.settled_statement(obligation.text,
                                               exclude=row["id"])
            if src is not None:
                # the graph already settled this exact statement as a CLAIM
                # (different identity space, so node dedup cannot see it) —
                # the verdict and its evidence transfer, no instrument spent
                outcome, witness = self._inherit(row, obligation, src)
                outcomes.append(outcome)
                if outcome == OUTCOME_REFUTED:
                    refuting_witness = witness
                    break
                continue
            outcome, witness = self._discharge(row, obligation)
            outcomes.append(outcome)
            if outcome == OUTCOME_REFUTED:
                refuting_witness = witness
                break
        self._settle(row["id"], outcomes)

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
        # graph context: verified premises are system-established facts the
        # compiler may build residual obligations on; everything else is
        # context only. The solver's argument stays invisible either way.
        verified_deps, pending_deps = [], []
        for pid in self.store.parents(row["id"]):
            prow = self.store.get(pid)
            label = ("[a verified tool-execution record]"
                     if prow["kind"] == KIND_EFFECT else prow["text"])
            if prow["status"] == VERIFIED:
                verified_deps.append(label)
            else:
                pending_deps.append(f"{label} (status: {prow['status']})")
        payload = self.obligations_role.call(
            self.backend,
            {"claim_text": row["text"],
             "verified_deps": json.dumps(verified_deps, ensure_ascii=False),
             "pending_deps": json.dumps(pending_deps, ensure_ascii=False),
             "graph": json.dumps(
                 self.store.graph_snapshot(exclude=row["id"]),
                 ensure_ascii=False),
             "memory_hits": memory_hits},
        )
        obligations = [
            Obligation(row["id"], o["text"]) for o in payload["obligations"]
        ]
        self.log.append(
            "obligations_compiled",
            {"claim": row["id"],
             "obligations": [o.text for o in obligations]},
        )
        return obligations

    def _inherit(self, row: Dict, obligation: Obligation, src: Dict):
        """Transfer a same-text settled CLAIM's verdict onto an obligation
        node. The node is settled too, so every later claim sharing it gets
        the plain node-dedup path. Returns (outcome, witness)."""
        oid = obligation.id
        ev = self.store.evidence_for(src["id"])
        witness = ev[-1].get("witness") if ev else None
        if witness is None and src["status"] == REFUTED:
            # a claim's concrete counterexample lives on its refuting
            # obligation node, one hop down — carry it over
            for kid in self.store.obligation_children(src["id"]):
                if kid["status"] == REFUTED and kid["reason"]:
                    witness = kid["reason"]
                    break
        if witness is None:
            witness = self.store.get(src["id"])["reason"]
        outcome = (OUTCOME_HOLDS if src["status"] == VERIFIED
                   else OUTCOME_REFUTED)
        self.store.record_evidence(
            oid,
            {"obligation": obligation.text, "instrument": "graph",
             "outcome": outcome, "source": src["id"], "witness": witness},
        )
        if outcome == OUTCOME_HOLDS:
            self.store.set_verified(oid)
        else:
            self.store.set_refuted(oid, reason=str(witness)[:200])
        self.log.append(
            "dedup_hit",
            {"claim": row["id"], "obligation": oid, "source": src["id"],
             "status": src["status"], "space": "claim"},
        )
        return outcome, witness

    def _discharge(self, row: Dict, obligation: Obligation):
        """Discharge one obligation NODE: evidence lands on the node, and
        the node's status is settled so any other claim sharing this
        obligation reuses the verdict. Returns (outcome, witness)."""
        oid = obligation.id
        if self.binder is not None:
            result = self.binder.run(row, obligation)
        else:
            result = self.discharger.discharge(row, obligation)
        evidence = Evidence(
            obligation_id=oid,
            instrument=result.instrument,
            outcome=result.outcome,
            artifacts={"artifact": result.artifact},
            witness=result.witness,
        )
        self.store.record_evidence(
            oid,
            {
                "obligation": obligation.text,
                "instrument": result.instrument,
                "outcome": result.outcome,
                "artifact": result.artifact,
                "witness": result.witness,
                "evidence_id": evidence.id,
            },
        )
        if result.outcome == OUTCOME_HOLDS:
            self.store.set_verified(oid)
        elif result.outcome == OUTCOME_REFUTED:
            self.store.set_refuted(oid, reason=str(result.witness)[:200])
        elif result.outcome == OUTCOME_UNOBTAINABLE:
            self.store.set_partial(oid)   # well-defined check, env can't host
        # failed -> the node stays unverified; a later claim may retry it
        self.log.append(
            "discharge",
            {"claim": row["id"], "obligation": oid,
             "instrument": result.instrument, "outcome": result.outcome},
        )
        return result.outcome, result.witness

    # ---- settlement ----------------------------------------------------------

    def _settle(self, claim_id: str, outcomes: List[str]) -> None:
        if any(o == OUTCOME_REFUTED for o in outcomes):
            self.store.set_refuted(claim_id, reason="counterexample on record")
            new_status = "refuted"
        elif all(o == OUTCOME_HOLDS for o in outcomes) and outcomes:
            self.store.set_verified(claim_id)
            new_status = "verified"
        elif any(o == OUTCOME_HOLDS for o in outcomes):
            self.store.set_partial(claim_id)
            new_status = "partial"
        elif any(o == OUTCOME_UNOBTAINABLE for o in outcomes):
            # the required check is well-defined but its instrument cannot be
            # obtained here and no equivalent method bound: honest partial
            self.store.set_partial(claim_id)
            new_status = "partial"
        else:
            new_status = UNVERIFIED
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
