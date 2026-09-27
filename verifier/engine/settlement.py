"""Settling a claim's obligations: reuse where it is sound, a fresh check
where it is not.

An obligation node is content-addressed, so the same condition compiled
under another claim is the same node. Before a check is spent on it, its
settled verdict is reused only past every reuse gate — the world has not
moved since a world-basis verdict was won, the verdict was won for this
candidate response, the premises it assumed are in force here — or an
identical settled claim's effective verdict is inherited under the same
floor and witness rules. Otherwise the node is discharged (the binding
game, or a direct discharge) and the evidence axioms (engine.gates) stand
behind the result before it settles the node.
"""
from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import FIRST_COMPLETED, wait
from typing import Callable, Dict, List, Optional

from ..core.model import (
    BASIS_RECORD, BASIS_WORLD, CLAIM_FINAL, Evidence, Obligation,
    OUTCOME_FAILED, OUTCOME_HOLDS, OUTCOME_REFUTED, OUTCOME_UNOBTAINABLE,
    SEVERITY_MUST, STATUS_PARTIAL, STATUS_REFUTED, STATUS_STALE,
    STATUS_UNVERIFIED, STATUS_VERIFIED, basis_satisfies,
)
from ..core.world import world_epoch
from ..llm.model_spans import model_scope, submit_with_context
from .gates import GateScope, apply_backstop, transferable_refutation


class ObligationSettler:
    """Settles obligation nodes for the claims of one store and ledger.

    ``binder`` runs the binding game for a node (None: the ``discharger``
    discharges it directly); ``pool()`` is the shared obligation pool (None:
    sequential); ``lock_for(table, key)`` hands out the per-node locks, so
    two claims sharing a node never discharge it twice."""

    def __init__(self, *, store, log, binder, discharger, live_world: bool,
                 pool: Callable, lock_for: Callable):
        self.store = store
        self.log = log
        self.binder = binder
        self.discharger = discharger
        self.live_world = live_world
        self.pool = pool
        self.lock_for = lock_for
        self.node_locks: Dict[str, threading.Lock] = {}
        # obligation node -> seq of its newest seat failure (see
        # seat_failure), noted as the log appends; None when the log
        # takes no listeners
        subscribe = getattr(log, "subscribe", None)
        self.seat_failures: Optional[Dict[str, int]] = (
            {} if callable(subscribe) else None)
        self.seat_failures_lock = threading.Lock()
        if callable(subscribe):
            subscribe(self.note_seat_failure)

    def settle_obligations(self, row: Dict, obligations: List[Obligation]):
        """Settle every obligation of a claim; returns
        [(obligation, outcome, witness, basis, strength, infrastructure)]
        in compile order for the obligations actually settled
        (infrastructure: a failed outcome owed to a seat error, see
        _seat_failure).

        Sequential (parallel_obligations == 1): one after another,
        stopping at the first MUST refutation — the claim is decided and
        the remaining checks would only spend instruments.

        Parallel: every obligation is submitted at once (each is an
        independent check of a content-addressed node), results are
        collected as they land, and the first MUST refutation cancels the
        obligations that have not started yet; those already running
        finish and are recorded. Latency becomes the slowest check
        instead of the sum, at the price of some speculative work on a
        refuted claim."""
        pool = self.pool()
        if pool is None or len(obligations) <= 1:
            out = []
            for obligation in obligations:
                result = self.settle_one(row, obligation)
                out.append((obligation, *result))
                if result[0] == OUTCOME_REFUTED \
                        and obligation.severity == SEVERITY_MUST:
                    break
            return out
        futures = {submit_with_context(pool, self.settle_one, row, o): o
                   for o in obligations}
        pending = set(futures)
        decided = False
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            if decided:
                continue
            for fut in done:
                if fut.cancelled():
                    continue
                outcome = fut.result()[0]
                if outcome == OUTCOME_REFUTED \
                        and futures[fut].severity == SEVERITY_MUST:
                    decided = True
                    for other in pending:
                        other.cancel()   # not-yet-started checks only
        out = []
        for fut, obligation in futures.items():
            if fut.cancelled():
                self.log.append(
                    "status_change",
                    {"claim": row["id"], "obligation": obligation.id,
                     "status": "skipped",
                     "reason": "claim decided by a must refutation before "
                               "this check started"})
                continue
            out.append((obligation, *fut.result()))
        return out

    def current_epoch(self) -> int:
        """The world epoch now: how many world-mutating solver actions the
        ledger holds. Offline the world is fixed (epoch 0)."""
        if not self.live_world:
            return 0
        return world_epoch(self.log.replay())

    def node_epoch_current(self, oid: str, epoch: int) -> bool:
        """Whether a settled node's verdict belongs to the current world:
        record-basis verdicts always do; a world-basis verdict only when
        its evidence was recorded in this epoch."""
        node = self.store.get(oid)
        if node["basis"] != BASIS_WORLD:
            return True
        ev = self.store.evidence_for(oid)
        recorded = (ev[-1].get("world_epoch") if ev else None)
        return recorded is not None and int(recorded) == epoch

    def settle_one(self, row: Dict, obligation: Obligation):
        """Settle one obligation node for this claim — by dedup, by
        inheritance from a same-text claim, or by fresh discharge — under
        the node's lock, so two claims sharing a node never discharge it
        twice: the second waits, then takes the dedup path. Returns
        (outcome, witness, basis, strength, infrastructure)."""
        oid = self.store.add_obligation(row["id"], obligation)
        with self.lock_for(self.node_locks, oid):
            # the epoch is read once the node is ours: a wait on another
            # claim's check of it may outlast world-mutating solver
            # actions, and reuse must be judged against the world as it
            # stands when the verdict is taken, not when the wait began
            epoch = self.current_epoch()
            known = self.store.status(oid)
            if known in (STATUS_VERIFIED, STATUS_REFUTED, STATUS_PARTIAL) \
                    and not self.node_epoch_current(oid, epoch):
                # WORLD EPOCH: the world moved since this world-basis
                # verdict was won; reusing it would certify the new world
                # on the old world's evidence. Reopen and check afresh.
                self.log.append(
                    "dedup_hit",
                    {"claim": row["id"], "obligation": oid,
                     "status": known, "refused": "world moved",
                     "world_epoch": epoch})
                self.store.set_stale(oid, reason="world moved since the "
                                                 "verdict was won")
                known = self.store.status(oid)
            candidate = self.candidate_digest(row)
            if known in (STATUS_VERIFIED, STATUS_REFUTED, STATUS_PARTIAL):
                ev = self.store.evidence_for(oid)
                won_for = ev[-1].get("candidate_digest") if ev else None
                if candidate is not None \
                        and won_for not in (None, candidate):
                    # CANDIDATE IDENTITY: the verdict was won while another
                    # final's text was the candidate response the seats
                    # read (one shared file); an answer-bearing check says
                    # nothing about this candidate. Reopen and check afresh.
                    # Only a final presents a candidate: a support claim
                    # sharing the node reuses what any final settled.
                    self.log.append(
                        "dedup_hit",
                        {"claim": row["id"], "obligation": oid,
                         "status": known,
                         "refused": "another candidate response"})
                    self.store.set_stale(oid, reason="verdict won for "
                                                     "another candidate")
                    known = self.store.status(oid)
            if known in (STATUS_VERIFIED, STATUS_REFUTED, STATUS_PARTIAL):
                # foreign-assumption gate: a node settled under another
                # claim may have ASSUMED that claim's premises (declared
                # assumed_premises on its evidence). The verdict is
                # conditional on those; it transfers only where the same
                # premises are in force — i.e. inside THIS claim's own
                # dependency closure, which the conditional fold already
                # charges. Otherwise discharge fresh.
                ev = self.store.evidence_for(oid)
                assumed = (ev[-1].get("assumed_premises")
                           if ev else None) or []
                if assumed and not set(assumed) \
                        <= self.premises_in_force(row):
                    self.log.append(
                        "dedup_hit",
                        {"claim": row["id"], "obligation": oid,
                         "status": known, "refused": "foreign assumptions",
                         "assumed_premises": list(assumed)})
                    # Never append a retry to a settled node: pairing its
                    # old status with new evidence would erase assumptions.
                    return (OUTCOME_FAILED,
                            "settled evidence has foreign assumptions",
                            None, None, False)
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
                node = self.store.get(oid)
                witness = None
                if outcome == OUTCOME_REFUTED:
                    ev = self.store.evidence_for(oid)
                    witness = ev[-1].get("witness") if ev else None
                return (outcome, witness, node["basis"] or BASIS_RECORD,
                        node["strength"], False)
            # the node is unsettled from here on and ours: whatever settles
            # it below does so only while no revision invalidates it
            generation = self.store.generation(oid)
            src_epoch = epoch
            src = self.store.settled_statement(obligation.text,
                                               exclude=row["id"])
            if src is not None:
                # inheritance transfers only EFFECTIVE verdicts: a local
                # verdict is conditional on its premise chain, and letting
                # a conditionally-verified claim settle another claim's
                # obligation would launder the open assumptions out of the
                # account (the final's contract most of all). A verified
                # source must be effectively verified (its whole chain
                # closed); a refuted source must be refuted on its own
                # content with no open premise (a premise-poisoned or
                # conditional refutation collapses that line of support —
                # it does not refute the statement unconditionally).
                # Anything else falls through to fresh discharge, which
                # can still establish the obligation directly.
                src_eff = self.store.effective(src["id"])
                transferable = (
                    (src["status"] == STATUS_VERIFIED
                     and src_eff["effective_status"] == STATUS_VERIFIED)
                    or (src["status"] == STATUS_REFUTED
                        and src_eff["effective_status"] == STATUS_REFUTED
                        and src_eff["premise_refuted"] is None
                        and not src_eff["open_premises"])
                )
                src_ev = self.store.evidence_for(src["id"])
                stamp = next((e.get("world_epoch") for e in
                              reversed(src_ev)
                              if e.get("world_epoch") is not None), None)
                if stamp is not None:
                    src_epoch = int(stamp)
                if transferable and src_eff["effective_basis"] == BASIS_WORLD:
                    if stamp is None or int(stamp) != epoch:
                        transferable = False   # the world moved since
                if transferable and not basis_satisfies(
                        src_eff["effective_basis"],
                        obligation.required_basis):
                    # evidence below the floor transfers in NEITHER
                    # direction: a record-established verdict — verified
                    # or refuted — cannot settle a world-floor obligation
                    # by inheritance (a legacy row with no basis fails
                    # closed the same way).
                    transferable = False
                if transferable and src["status"] == STATUS_REFUTED:
                    evidence = self.store.evidence_for(src["id"])
                    witness_kind = (evidence[-1].get("witness_kind")
                                    if evidence else None)
                    transferable = transferable_refutation(
                        witness_kind, src_eff["effective_strength"],
                        obligation)
                if not transferable:
                    src = None
            if src is not None:
                outcome, witness, basis, strength = self.inherit(
                    row, obligation, src, epoch=src_epoch,
                    generation=generation)
                return (outcome, witness, basis or BASIS_RECORD, strength,
                        False)
            since = self.log.last_seq()
            outcome, witness, basis, strength = self.discharge(
                row, obligation, generation=generation)
        # outside the node lock: other claims waiting on this node do not
        # wait on a look at the ledger
        infrastructure = (outcome == OUTCOME_FAILED
                          and self.seat_failure(oid, since))
        return (outcome, witness, basis or BASIS_RECORD, strength,
                infrastructure)

    # a seat call that failed for good (after its own retries): the
    # ledger events through which the binding game reports them
    @staticmethod
    def is_seat_failure(event: Dict) -> bool:
        payload = event.get("payload") or {}
        if event.get("type") == "judger_call_failed":
            return not payload.get("will_retry")
        if event.get("type") == "methods_enumerated":
            return bool(payload.get("error")) and not payload.get("prewarm")
        return False

    def note_seat_failure(self, event: Dict) -> None:
        """Log listener: the newest seat-failure seq per obligation node."""
        if not self.is_seat_failure(event):
            return
        oid = (event.get("payload") or {}).get("obligation")
        if isinstance(oid, str):
            with self.seat_failures_lock:
                self.seat_failures[oid] = max(
                    self.seat_failures.get(oid, 0), event["seq"])

    def seat_failure(self, oid: str, since: int) -> bool:
        """Whether this node's discharge, which began after ledger seq
        ``since``, lost a seat to an infrastructure error (an Oracle or
        Judger call that raised through its retries — a transport error,
        an HTTP 403, a schema-invalid generation). A failed outcome owed
        to one is no finding about the world. Read from what the log
        listener noted; a log without listeners is scanned."""
        if self.seat_failures is not None:
            with self.seat_failures_lock:
                return self.seat_failures.get(oid, 0) > since
        for event in self.log.replay():
            if event["seq"] > since \
                    and (event.get("payload") or {}).get("obligation") == oid \
                    and self.is_seat_failure(event):
                return True
        return False

    def premises_in_force(self, row: Dict) -> set:
        """The premise ids whose content evidence for this claim may
        assume: the dependency closure for an ordinary claim (the
        conditional fold charges them to every consumer), the EMPTY set
        for a final — its contract is compiled independently and it is
        exempt from the premise fold, so an assumption there would
        never be charged to anyone."""
        if row["kind"] == CLAIM_FINAL:
            return set()
        return set(self.store.backward_slice(row["id"])[1:])

    def inherit(self, row: Dict, obligation: Obligation, src: Dict,
                 epoch: Optional[int] = None,
                 generation: Optional[int] = None):
        """
        Transfer a same-text settled CLAIM's verdict onto an obligation
        node. The node is settled too (when still settleable), so every
        later claim sharing it gets the plain node-dedup path. Returns
        (outcome, witness, basis, strength).

        ``epoch`` is the world epoch of the source's evidence: the
        inherited verdict is about the world the source was checked on,
        not the one of the day. ``generation`` is the node's, read when
        this settlement began.
        """

        oid = obligation.id
        # inheritance transfers the verdict WITH its EFFECTIVE basis and
        # strength: a verdict settled on the solver's record — or resting
        # anywhere in its premise chain on one — does not become
        # independent by changing identity space. The caller has already
        # gated on the effective verdict being settled and chain-closed.
        src_eff = self.store.effective(src["id"])
        src_basis = src_eff["effective_basis"] or BASIS_RECORD
        src_strength = src_eff["effective_strength"]
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
                "required_basis": obligation.required_basis,
                "instrument": "graph",
                "outcome": outcome,
                "basis": src_basis,
                "strength": src_strength,
                "source": src["id"],
                "witness": witness,
                "witness_kind": ev[-1].get("witness_kind") if ev else None,
                "artifact": ev[-1].get("artifact", "") if ev else "",
                "assumed_premises": [],
                "world_epoch": (self.current_epoch() if epoch is None
                                else epoch),
                **self.candidate_stamp(row),
            },
        )
        # The node lock and scoped identity keep its verdict paired with
        # this evidence. Only previously unsettled nodes reach inheritance.
        if self.store.status(oid) in (STATUS_UNVERIFIED, STATUS_STALE):
            if outcome == OUTCOME_HOLDS:
                self.store.set_verified(oid, basis=src_basis,
                                        strength=src_strength,
                                        generation=generation)
            else:
                self.store.set_refuted(oid, reason=str(witness)[:200],
                                       basis=src_basis,
                                       strength=src_strength,
                                       generation=generation)
        self.log.append(
            "dedup_hit",
            {"claim": row["id"], "obligation": oid, "source": src["id"],
             "status": src["status"], "space": "claim"},
        )
        return outcome, witness, src_basis, src_strength

    def candidate_stamp(self, row: Dict) -> Dict:
        digest = self.candidate_digest(row)
        return {} if digest is None else {"candidate_digest": digest}

    def candidate_digest(self, row: Dict) -> Optional[str]:
        """Identity of the candidate response a final's checks read (the
        solver's reply once written, else the final's own text — see
        publish_candidate_response); None for any other claim. Evidence
        won under a final is about THAT candidate: it is reused only by
        a final presenting the same one."""
        if row.get("kind") != CLAIM_FINAL:
            return None
        state = os.path.dirname(os.path.abspath(self.log.path))
        try:
            with open(os.path.join(state, "final.txt"), "rb") as handle:
                return hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            return hashlib.sha256(row["text"].encode("utf-8")).hexdigest()

    def discharge(self, row: Dict, obligation: Obligation,
                   generation: Optional[int] = None):
        """
        Discharge one obligation NODE: evidence lands on the node, and
        the node's status is settled (when still settleable, and while
        the node's ``generation`` is the one read when this discharge
        began) so any other claim sharing this obligation reuses the
        verdict. Returns (outcome, witness, basis, strength).

        A discharge whose node a revision invalidated meanwhile (e.g. a
        superseded final's check still running, or a premise revised)
        records its evidence for audit and settles nothing: the node
        stays stale and the next claim relying on it checks it afresh.
        """

        oid = obligation.id
        # the epoch the check began in: world-basis evidence is about the
        # world as the check found it, so a verdict won while the solver's
        # actions were being recorded must not read as current for the
        # later epoch (reuse compares this stamp to the epoch of the day)
        started_epoch = self.current_epoch()
        with model_scope(obligation_id=oid):
            if self.binder is not None:
                result = self.binder.run(row, obligation)
            else:
                with model_scope(stage="discharge", method_attempt=1, redo=0):
                    result = self.discharger.discharge(row, obligation)
        # the axioms behind every discharge (a custom binder could
        # mislabel, direct discharge has no game, an audit can be slipped):
        # each violated one converts the result to failed, its finding
        # kept in the witness for audit (see engine.gates)
        result = apply_backstop(result, obligation, GateScope(
            premises_in_force=frozenset(self.premises_in_force(row)),
            completion_claim=row["kind"] == CLAIM_FINAL))
        evidence = Evidence(
            obligation_id=oid,
            instrument=result.instrument,
            outcome=result.outcome,
            artifacts={"artifact": result.artifact},
            witness=result.witness,
        )
        evidence_payload = {
            "obligation": obligation.text,
            "required_basis": obligation.required_basis,
            "instrument": result.instrument,
            "outcome": result.outcome,
            "basis": result.basis,
            "world_epoch": started_epoch,
            "strength": result.strength,
            "bridges": list(result.bridges),
            "assumed_premises": list(result.assumed_premises),
            "artifact": result.artifact,
            "witness": result.witness,
            "witness_kind": result.witness_kind,
            "inputs": list(result.inputs),
            "severity": obligation.severity,
            "clause": obligation.clause,
            "evidence_id": evidence.id,
            **self.candidate_stamp(row),
        }
        self.store.record_evidence(oid, evidence_payload)
        # Retries append only while the node is unsettled. A settled node
        # and its evidence history are reused together within its scope.
        if self.store.status(oid) in (STATUS_UNVERIFIED, STATUS_STALE):
            if result.outcome == OUTCOME_HOLDS:
                self.store.set_verified(oid, basis=result.basis,
                                        strength=result.strength,
                                        generation=generation)
            elif result.outcome == OUTCOME_REFUTED:
                self.store.set_refuted(oid,
                                       reason=str(result.witness)[:200],
                                       basis=result.basis,
                                       strength=result.strength,
                                       generation=generation)
            elif result.outcome == OUTCOME_UNOBTAINABLE:
                # well-defined, env can't host
                self.store.set_partial(oid, generation=generation)
        # failed -> the node stays unverified; a later claim may retry it
        self.log.append(
            "discharge",
            {"claim": row["id"], "obligation": oid,
             "instrument": result.instrument, "outcome": result.outcome,
             "required_basis": obligation.required_basis,
             "basis": result.basis, "strength": result.strength,
             "assumed_premises": list(result.assumed_premises),
             "artifact": result.artifact,
             "evidence_id": evidence.id},
        )
        return result.outcome, result.witness, result.basis, result.strength
