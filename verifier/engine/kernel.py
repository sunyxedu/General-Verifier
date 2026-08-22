"""
Kernel.

Compile obligations (graph-aware, answer-blind) -> materialize each as an
obligation NODE in the claim graph (identical text anywhere = same node, so
settled verdicts transfer for free) -> discharge each node through the
binding game (oracle levels -> execute -> judger) -> settle the claim as a
fold over its obligation nodes.

All kernel activity is appended to the same event stream the solver's activity
came from. The verifier is auditable through its own ledger.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from ..execution.discharge import AgentDischarger, DischargeResult
from ..core.events import EventLog
from ..execution.instruments import ArtifactStore, LedgerInstrument
from ..core.model import (
    Claim, Evidence, Obligation, basis_satisfies,
    BASIS_RECORD, BASIS_WORLD,
    CLAIM_EFFECT, CLAIM_FINAL,
    STATUS_UNVERIFIED, STATUS_VERIFIED, STATUS_STALE, STATUS_REFUTED, STATUS_PARTIAL,
    OUTCOME_HOLDS, OUTCOME_REFUTED, OUTCOME_FAILED, OUTCOME_UNOBTAINABLE,
)
from ..llm import prompts
from ..llm.agents import COMPILE_OBLIGATIONS_AGENT, CRITIC_AGENT
from ..core.store import ClaimStore

# attack rounds on a decomposition before the claim fails closed: each round
# is one critic attempt; an `insufficient` verdict sends the decomposition
# back for one repair, and a counterexample still standing after the last
# round leaves the claim unverified
MAX_CRITIC_ROUNDS = 2


class Kernel:
    """
    Role seats (obligations compiler, discharger, binder) are injectable:
    the same kernel runs on fixture-backed seats (tests, deterministic) or on
    the Claude Agent SDK agents (the shipped configuration) without knowing
    which is seated.
    """

    def __init__(
        self,
        store: ClaimStore,
        log: EventLog,
        backend,
        artifacts: ArtifactStore,
        memory_hits: str = "[]",
        obligations_role=None,
        critic_role=CRITIC_AGENT,
        discharger=None,
        binder=None,
        memory=None,
    ):
        self.store = store
        self.log = log
        self.backend = backend
        self.artifacts = artifacts
        self.ledger = LedgerInstrument(log)
        self.memory_hits = memory_hits
        self.obligations_role = obligations_role or COMPILE_OBLIGATIONS_AGENT
        self.critic_role = critic_role   # None -> no adversarial review
        self.discharger = discharger or AgentDischarger(
            backend, artifacts,
            state_dir=os.path.dirname(os.path.abspath(log.path)))
        self.binder = binder      # binding game; None -> direct discharge (tests)
        self.memory = memory      # advisory stores; None -> static hints
        store.on_trigger(self.verify)  # the single rule wires straight in

    # entry
    def verify(self, claim_id: str) -> str:
        """Verify one claim; returns its resulting status. Idempotent."""
        row = self.store.get(claim_id)
        if row["status"] not in (STATUS_UNVERIFIED, STATUS_STALE, STATUS_PARTIAL):
            return row["status"]
        if row["kind"] == CLAIM_EFFECT:
            self._verify_effect(row)
        else:
            self._verify_assertion(row)
        return self.store.status(claim_id)

    # effect: ledger check
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
        # an effect's evidence IS the ledger: record basis by construction
        self._settle(row["id"], [result.outcome], [BASIS_RECORD])

    # assertion / final
    def _verify_assertion(self, row: Dict) -> None:
        # a FINAL is never spared adjudication by a collapsed premise: its
        # obligations are the task contract, compiled and established
        # independently of the argument's health — a refuted intermediate
        # eliminates that line of support, not the deliverable's verdict
        refuted_deps = [] if row["kind"] == CLAIM_FINAL else [
            p for p in self.store.parents(row["id"])
            if self.store.status(p) == STATUS_REFUTED
        ]
        if refuted_deps:
            self.store.record_evidence(
                row["id"],
                {"outcome": OUTCOME_FAILED, "error": "depends on refuted claim",
                 "refuted_deps": refuted_deps},
            )
            self.log.append(
                "status_change",
                {"claim": row["id"], "status": STATUS_UNVERIFIED,
                 "reason": "depends on refuted claim"},
            )
            return

        obligations = self._compile_obligations(row)
        if obligations is None:
            return  # critic fail-closed: evidence and status already recorded
        if not obligations:
            # fail-closed: nothing checkable means nothing verified
            self.store.record_evidence(
                row["id"],
                {"outcome": OUTCOME_FAILED, "error": "no obligations"},
            )
            self.log.append(
                "status_change",
                {"claim": row["id"], "status": STATUS_UNVERIFIED,
                 "reason": "no obligations"},
            )
            return

        outcomes: List[str] = []
        bases: List[str] = []     # per-obligation evidence basis, for the
        refuting_witness = None   # claim-level provenance fold at settle
        for obligation in obligations:
            oid = self.store.add_obligation(row["id"], obligation)
            known = self.store.status(oid)
            if known in (STATUS_VERIFIED, STATUS_REFUTED, STATUS_PARTIAL):
                # content-addressed reuse: the same world-condition was
                # already settled under another claim. The verdict and its
                # evidence transfer without re-spending instruments
                outcome = {STATUS_VERIFIED: OUTCOME_HOLDS,
                           STATUS_REFUTED: OUTCOME_REFUTED,
                           STATUS_PARTIAL: OUTCOME_UNOBTAINABLE}[known]
                self.log.append("dedup_hit",
                                {"claim": row["id"], "obligation": oid,
                                 "status": known})
                outcomes.append(outcome)
                bases.append(self.store.get(oid)["basis"] or BASIS_RECORD)
                if outcome == OUTCOME_REFUTED:
                    ev = self.store.evidence_for(oid)
                    refuting_witness = ev[-1].get("witness") if ev else None
                    break
                continue
            src = self.store.settled_statement(obligation.text,
                                               exclude=row["id"])
            if src is not None and src["status"] == STATUS_VERIFIED \
                    and not basis_satisfies(
                        self.store.get(src["id"])["basis"],
                        obligation.required_basis):
                # a record-established claim cannot settle a world-floor
                # obligation by inheritance — fall through and discharge
                src = None
            if src is not None:
                # the graph already settled this exact statement as a CLAIM
                # (different identity space, so node dedup cannot see it).
                # the verdict and its evidence transfer, no instrument spent
                outcome, witness = self._inherit(row, obligation, src)
                outcomes.append(outcome)
                bases.append(self.store.get(oid)["basis"] or BASIS_RECORD)
                if outcome == OUTCOME_REFUTED:
                    refuting_witness = witness
                    break
                continue
            outcome, witness = self._discharge(row, obligation)
            outcomes.append(outcome)
            bases.append(self.store.get(oid)["basis"] or BASIS_RECORD)
            if outcome == OUTCOME_REFUTED:
                refuting_witness = witness
                break
        self._settle(row["id"], outcomes, bases)

        if self.memory is not None:
            status = self.store.status(row["id"])
            self.memory.record_obligations(
                row["text"], [o.text for o in obligations], status)
            if status == STATUS_REFUTED:
                self.memory.record_attack(
                    row["text"], row["text"],
                    str(refuting_witness), kind="refutation")
            self.log.append("memory_recorded", {"claim": row["id"]})

    def _compile_obligations(self, row: Dict):
        memory_hits = (
            self.memory.hits_for_obligations(row["text"])
            if self.memory is not None else self.memory_hits
        )
        # graph context: verified premises reach the compiler tagged with
        # their evidence basis — world-level premises are facts residual
        # obligations may build on; record-level ones only prove the solver
        # reported/did as recorded. The solver's argument stays invisible
        # either way.
        verified_deps, pending_deps = [], []
        for pid in self.store.parents(row["id"]):
            prow = self.store.get(pid)
            if prow["kind"] == CLAIM_EFFECT:
                label = "[a verified tool-execution record]"
                if prow["status"] == STATUS_VERIFIED:
                    verified_deps.append(label)
                else:
                    pending_deps.append(f"{label} (status: {prow['status']})")
                continue
            if prow["status"] == STATUS_VERIFIED:
                basis = prow["basis"] or BASIS_RECORD
                verified_deps.append(f"[{basis}-level] {prow['text']}")
            else:
                pending_deps.append(
                    f"{prow['text']} (status: {prow['status']})")
        # a run scored after the fact (offline ORM) has no live world; the
        # harness may instead have captured the delivered tree next to the
        # ledger. Absolute, like ledger_path: the checker runs elsewhere.
        world = os.path.join(os.path.dirname(os.path.abspath(self.log.path)),
                             "world")
        variables = {
            "claim_text": row["text"],
            "claim_kind": row["kind"],
            "ledger_path": os.path.abspath(self.log.path),
            "world_path": world if os.path.isdir(world) else "",
            "verified_deps": json.dumps(verified_deps, ensure_ascii=False),
            "pending_deps": json.dumps(pending_deps, ensure_ascii=False),
            "graph": json.dumps(
                self.store.graph_snapshot(exclude=row["id"]),
                ensure_ascii=False),
            "memory_hits": memory_hits,
            "critic_feedback": "",
        }
        payload = self.obligations_role.call(self.backend, variables)
        payload = self._critique(row, payload, variables)
        if payload is None:
            return None
        obligations = [
            Obligation(row["id"], o["text"],
                       o.get("required_basis", BASIS_RECORD))
            for o in payload["obligations"]
        ]
        self.log.append(
            "obligations_compiled",
            {"claim": row["id"],
             "obligations": [{"text": o.text,
                              "required_basis": o.required_basis}
                             for o in obligations]},
        )
        return obligations

    def _critique(self, row: Dict, payload: Dict, variables: Dict):
        """
        Adversarial second signature on the decomposition: the critic
        tries to construct a world where every obligation and every verified
        premise holds and the claim is still false. It never sees the
        compiler's reasoning. Only the claim, the obligations, and the
        verified premises. An attack that stands sends the decomposition
        back for repair; one still standing after MAX_CRITIC_ROUNDS fails
        closed (returns None, claim stays unverified).
        """
        if self.critic_role is None:
            return payload
        # the critic judges completion claims against the harness's
        # authoritative task statement, not the claim's own paraphrase —
        # same toggle as the compiler's addendum: the file's presence
        task = os.path.join(os.path.dirname(os.path.abspath(self.log.path)),
                            "task.md")
        task_context = (f"Read the file at {os.path.abspath(task)}"
                        if os.path.exists(task) else "(none supplied)")
        verdict = None
        for round_no in range(MAX_CRITIC_ROUNDS):
            verdict = self.critic_role.call(
                self.backend,
                {
                    "claim_text": row["text"],
                    "claim_kind": row["kind"],
                    "task_context": task_context,
                    "obligations_json": json.dumps(
                        [o["text"] for o in payload["obligations"]],
                        ensure_ascii=False
                    ),
                    "verified_deps": variables["verified_deps"],
                },
            )
            self.log.append(
                "decomposition_attacked",
                {"claim": row["id"], "round": round_no,
                 "verdict": verdict["verdict"],
                 "counterexample_world": verdict["counterexample_world"],
                 "missing_condition": verdict["missing_condition"]},
            )
            if verdict["verdict"] == "pass":
                return payload
            if round_no + 1 == MAX_CRITIC_ROUNDS:
                break  # last attack stands; a repair would go unreviewed
            variables = dict(
                variables,
                critic_feedback=prompts.CRITIC_FEEDBACK_TEMPLATE.format(
                    counterexample_world=verdict["counterexample_world"],
                    missing_condition=verdict["missing_condition"]),
            )
            payload = self.obligations_role.call(self.backend, variables)
        self.store.record_evidence(
            row["id"],
            {
                "outcome": OUTCOME_FAILED,
                "error": "decomposition failed critic",
                "counterexample_world": verdict["counterexample_world"],
                "missing_condition": verdict["missing_condition"],
            },
        )
        self.log.append(
            "status_change",
            {
                "claim": row["id"], 
                "status": STATUS_UNVERIFIED,
                "reason": "decomposition failed critic",
            },
        )
        return None

    def _inherit(self, row: Dict, obligation: Obligation, src: Dict):
        """
        Transfer a same-text settled CLAIM's verdict onto an obligation
        node. The node is settled too, so every later claim sharing it gets
        the plain node-dedup path. Returns (outcome, witness).
        """

        oid = obligation.id
        ev = self.store.evidence_for(src["id"])
        witness = ev[-1].get("witness") if ev else None
        if witness is None and src["status"] == STATUS_REFUTED:
            # a claim's concrete counterexample lives on its refuting
            # obligation node, one hop down. Carry it over.
            for kid in self.store.obligation_children(src["id"]):
                if kid["status"] == STATUS_REFUTED and kid["reason"]:
                    witness = kid["reason"]
                    break
        if witness is None:
            witness = self.store.get(src["id"])["reason"]
        outcome = (OUTCOME_HOLDS if src["status"] == STATUS_VERIFIED
                   else OUTCOME_REFUTED)
        self.store.record_evidence(
            oid,
            {
                "obligation": obligation.text,
                "instrument": "graph",
                "outcome": outcome,
                "source": src["id"],
                "witness": witness,
            },
        )
        if outcome == OUTCOME_HOLDS:
            # inheritance transfers the verdict WITH its basis: a verdict
            # settled on the solver's record does not become independent
            # by changing identity space
            src_basis = self.store.get(src["id"])["basis"] or BASIS_RECORD
            self.store.set_verified(oid, basis=src_basis)
        else:
            self.store.set_refuted(oid, reason=str(witness)[:200])
        self.log.append(
            "dedup_hit",
            {"claim": row["id"], "obligation": oid, "source": src["id"],
             "status": src["status"], "space": "claim"},
        )
        return outcome, witness

    def _discharge(self, row: Dict, obligation: Obligation):
        """
        Discharge one obligation NODE: evidence lands on the node, and
        the node's status is settled so any other claim sharing this
        obligation reuses the verdict. Returns (outcome, witness).
        """

        oid = obligation.id
        if self.binder is not None:
            result = self.binder.run(row, obligation)
        else:
            result = self.discharger.discharge(row, obligation)
        if result.outcome == OUTCOME_HOLDS \
                and not basis_satisfies(result.basis,
                                        obligation.required_basis):
            # backstop behind the binding game's method filter (direct
            # discharge has no filter; a mislabel can slip an audit):
            # evidence below the obligation's floor never settles it —
            # fails closed, never converted into a pass
            result = DischargeResult(
                outcome=OUTCOME_FAILED,
                witness=(f"evidence basis {result.basis!r} below the "
                         f"obligation's floor {obligation.required_basis!r}"),
                artifact=result.artifact,
                instrument=result.instrument,
                basis=result.basis,
            )
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
                "required_basis": obligation.required_basis,
                "instrument": result.instrument,
                "outcome": result.outcome,
                "artifact": result.artifact,
                "witness": result.witness,
                "evidence_id": evidence.id,
            },
        )
        if result.outcome == OUTCOME_HOLDS:
            self.store.set_verified(oid, basis=result.basis)
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

    # settlement
    def _settle(self, claim_id: str, outcomes: List[str],
                bases: List[str] = ()) -> None:
        if any(o == OUTCOME_REFUTED for o in outcomes):
            self.store.set_refuted(claim_id, reason="counterexample on record")
            new_status = "refuted"
        elif all(o == OUTCOME_HOLDS for o in outcomes) and outcomes \
                and self.store.get(claim_id)["kind"] == CLAIM_FINAL \
                and BASIS_WORLD not in bases:
            # a completion verdict must rest on at least one world-
            # established obligation: an all-record final is the
            # laundering signature (every check passed, none of them
            # ever left the solver's own account of events)
            self.store.record_evidence(
                claim_id,
                {"outcome": OUTCOME_FAILED,
                 "error": "no world-basis evidence for a completion claim"},
            )
            new_status = STATUS_UNVERIFIED
        elif all(o == OUTCOME_HOLDS for o in outcomes) and outcomes:
            # provenance taint fold: world only when every obligation was
            # world-based AND every verified assertion premise it may have
            # relied on is world-level. Effect premises are exempt — they
            # reach the compiler as opaque labels, so they cannot have
            # contributed semantic content. Any record component -> record:
            # a chain of reliance must bottom out in world checks or say so.
            basis = (BASIS_WORLD
                     if bases and all(b == BASIS_WORLD for b in bases)
                     and self._premises_world(claim_id)
                     else BASIS_RECORD)
            self.store.set_verified(claim_id, basis=basis)
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
            new_status = STATUS_UNVERIFIED
        self.log.append("status_change", {"claim": claim_id, "status": new_status})

    def _premises_world(self, claim_id: str) -> bool:
        """True when no verified assertion premise is record-level. The
        compiler may have built residual obligations on any verified
        premise it was shown, so a record-level one taints the claim."""
        for pid in self.store.parents(claim_id):
            prow = self.store.get(pid)
            if prow["kind"] == CLAIM_EFFECT:
                continue
            if prow["status"] == STATUS_VERIFIED \
                    and (prow["basis"] or BASIS_RECORD) != BASIS_WORLD:
                return False
        return True
