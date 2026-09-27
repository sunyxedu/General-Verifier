"""
Kernel.

Compile obligations (graph-aware, answer-blind) -> materialize each as an
obligation NODE in the claim graph (same statement, evidence floor, absence
rule, and premise scope share a verdict) -> discharge each node through the
binding game (oracle levels -> execute -> judger) -> settle the claim as a
fold over its obligation nodes.

All kernel activity is appended to the same event stream the solver's activity
came from. The verifier is auditable through its own ledger.
"""
from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Dict, List, Optional

from ..execution.discharge import AgentDischarger
from ..config import (
    APPEND_ONLY_DELTA_PROTOCOL,
    RunOptions,
    append_only_deltas_enabled,
)
from ..core.events import EventLog
from ..execution.instruments import (
    ArtifactStore, LedgerInstrument,
)
from ..core.model import (
    BASIS_RECORD, CLAIM_EFFECT, CLAIM_FINAL, OBLIGATION_KINDS,
    STATUS_UNVERIFIED, STATUS_VERIFIED, STATUS_STALE, STATUS_REFUTED, STATUS_PARTIAL,
    OUTCOME_REFUTED, OUTCOME_FAILED, STRENGTH_DECISIVE, SEVERITY_MUST,
)
from ..llm import prompts
from ..llm.agents import AgentRole, COMPILE_OBLIGATIONS_AGENT, CRITIC_AGENT
from ..llm.model_spans import model_scope, submit_with_context
from ..llm.runtime import capabilities_of
from ..core.store import ClaimStore
from .contract_compiler import (
    MAX_CRITIC_ROUNDS, ContractCompiler,
)
from .fold import fold
from .screening import Screener
from .settlement import ObligationSettler

def compile_roles_for_run(run_options, obligations_role, critic_role):
    """Treatment-only prompt specialization of the contract seats.

    Clone (never mutate) the compiler and Critic seats, retaining their
    schemas, tools, names, call/retry behavior, and the kernel's critic-round
    budget.  Unknown injected role implementations are left untouched rather
    than being wrapped with assumptions about their prompt protocol.  The
    concurrent final-contract precompile applies the identical rule so its
    contract key can match the kernel's.
    """
    if run_options.compact_inspection is True:
        if (type(obligations_role) is AgentRole
                and obligations_role.name == "obligations"):
            obligations_role = replace(
                obligations_role,
                system=(obligations_role.system
                        + prompts.COMPACT_COMPLETENESS_COMPILER_ADDENDUM),
            )
        if (type(critic_role) is AgentRole
                and critic_role.name == "critic"):
            critic_role = replace(
                critic_role,
                system=(critic_role.system
                        + prompts.COMPACT_COMPLETENESS_CRITIC_ADDENDUM),
            )
    return obligations_role, critic_role


class _ContractSetting:
    """A contract setting the kernel exposes and its compiler owns: reading
    or tuning it on the kernel reads or tunes the compiler's."""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return getattr(instance.contracts, self.name)

    def __set__(self, instance, value):
        setattr(instance.contracts, self.name, value)


class Kernel:
    """
    The verifier's orchestrator: it schedules a claim's check, screens it,
    has its contract compiled (ContractCompiler), settles the obligations
    (ObligationSettler), and folds them into the claim's verdict.

    Role seats (obligations compiler, discharger, binder) are injectable:
    the same kernel runs on fixture-backed seats (tests, deterministic) or on
    the Claude Agent SDK agents (the shipped configuration) without knowing
    which is seated.
    """

    obligations_role = _ContractSetting()
    critic_role = _ContractSetting()
    critic_rounds = _ContractSetting()
    critic_rounds_nonfinal = _ContractSetting()
    unify_final = _ContractSetting()
    contract_cache = _ContractSetting()
    contract_replay = _ContractSetting()
    memory = _ContractSetting()
    memory_hits = _ContractSetting()
    append_only_deltas = _ContractSetting()
    append_only_delta_protocol = _ContractSetting()

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
        live_world: bool = True,
        critic_rounds: Optional[int] = None,
        parallel_obligations: int = 1,
        parallel_claims: int = 1,
        contract_cache=None,
        contract_replay=None,
        record_boundary=None,
        screener_role=None,
        critic_rounds_nonfinal: Optional[int] = None,
        speculative_discharge: bool = False,
        unify_final: bool = False,
        run_options: Optional[RunOptions] = None,
    ):
        # parallelism is an execution strategy, not a judgment: the same
        # obligations get the same evidence, only the schedule changes.
        # parallel_obligations bounds concurrent discharges within one
        # claim; parallel_claims is read by request_certificate to verify
        # independent claims of a slice concurrently (offline only — an
        # online commit still verifies one claim as it arrives).
        self.parallel_obligations = max(1, int(parallel_obligations or 1))
        self.parallel_claims = max(1, int(parallel_claims or 1))
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_guard = threading.Lock()
        self._claim_locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        # unsettled claims a verify request left alone because their last
        # complete check ran on the world as it stands (see verify)
        self.not_rechecked: set = set()
        self.store = store
        self.log = log
        self.backend = backend
        # The run's evidence-presentation options; a kernel built without
        # them runs with the options its runtime declares.
        self.run_options = (run_options if run_options is not None
                            else capabilities_of(backend).run_options)
        self.artifacts = artifacts
        self.ledger = LedgerInstrument(log)
        # claim id -> ledger seq at which the claim was made (the Builder
        # knows; a bare kernel does not).  Without it the record boundary is
        # the ledger's end when verification of the claim begins.
        self._record_boundary = (record_boundary
                                 if callable(record_boundary) else None)
        self.live_world = live_world
        # Resolve the experimental repair representation once per Kernel.
        # Runtime environment mutation cannot change compiler semantics
        # halfway through a verification game.
        append_only_deltas = append_only_deltas_enabled()
        compiler_role, critic_role = compile_roles_for_run(
            self.run_options, obligations_role or COMPILE_OBLIGATIONS_AGENT,
            critic_role)   # critic None -> no adversarial review
        # UNIFIED FINAL: the final takes the path of any claim — screened
        # (a high-confidence screener verdict settles it like any claim)
        # and reviewed on the non-final critic budget — while its contract,
        # when it is compiled, still comes from the authoritative task
        # statement.
        self.contracts = ContractCompiler(
            self, obligations_role=compiler_role, critic_role=critic_role,
            critic_rounds=(MAX_CRITIC_ROUNDS if critic_rounds is None
                           else critic_rounds),
            critic_rounds_nonfinal=critic_rounds_nonfinal,
            unify_final=unify_final, contract_cache=contract_cache,
            memory=memory, memory_hits=memory_hits,
            append_only_deltas=append_only_deltas,
            append_only_delta_protocol=(
                APPEND_ONLY_DELTA_PROTOCOL if append_only_deltas else None),
            conclude_unverified=self._conclude_unverified,
            pool=self._obligation_pool,
            premises_in_force=lambda row: self.settler.premises_in_force(row))
        # SPECULATIVE DISCHARGE: start settling the compiler's obligations
        # while the critic reviews them. Obligation nodes are
        # content-addressed, so every obligation the repaired
        # decomposition keeps is already settled when the review ends;
        # only added ones are settled afterwards, and dropped ones lose
        # their edge to this claim (their nodes and evidence remain facts
        # about their statements). The claim's verdict is folded only
        # over the reviewed decomposition — no soundness change, the
        # critic's wall time simply overlaps the checks. Off by default
        # (offline scoring and tests keep the sequential order).
        self.speculative_discharge = speculative_discharge
        # screener: reads a claim's verdict off the retained record before
        # any instrument is spent; a high-confidence verified / supported /
        # refuted settles the claim and the formal checks are skipped.
        # None -> every claim is checked.
        self.screener_role = screener_role
        self.discharger = discharger or AgentDischarger(
            backend, artifacts,
            state_dir=os.path.dirname(os.path.abspath(log.path)),
            live_world=live_world, store=store, log=log,
            run_options=self.run_options)
        self.binder = binder      # binding game; None -> direct discharge (tests)
        if binder is not None:
            binder_delta = getattr(
                binder, "append_only_deltas", self.append_only_deltas)
            binder_protocol = getattr(
                binder, "append_only_delta_protocol",
                self.append_only_delta_protocol)
            if (binder_delta != self.append_only_deltas
                    or binder_protocol != self.append_only_delta_protocol):
                raise ValueError(
                    "binding game and kernel use different append-repair "
                    "protocols")
        self.settler = ObligationSettler(
            store=store, log=log, binder=binder, discharger=self.discharger,
            live_world=live_world, pool=self._obligation_pool,
            lock_for=self._lock_for)
        self.screener = None if screener_role is None else Screener(
            screener_role, backend, store, log,
            report_status=self._report_status,
            current_epoch=self.settler.current_epoch,
            candidate_stamp=self.settler.candidate_stamp)
        # Experimental adapter-isolation diagnostic only.  Ordinary online
        # and offline verification never inject this provider; it freezes a
        # Critic-approved contract, not any method, evidence, or verdict.
        if contract_replay is not None:
            if live_world:
                raise ValueError(
                    "sealed contract replay is an offline diagnostic only")
            contract_replay.require_replayable()
            validate_kernel = getattr(contract_replay, "validate_kernel", None)
            if not callable(validate_kernel):
                raise ValueError(
                    "sealed contract replay must be bound to this kernel")
            validate_kernel(self)
        self.contract_replay = contract_replay
        # claim id -> the store generation its running check began at (see
        # ClaimStore.generation); one entry per claim, checks being
        # serialized per claim
        self._check_generation: Dict[str, int] = {}
        store.on_trigger(self.verify)  # the single rule wires straight in
        store.on_trigger_error(self._check_raised)
        store.on_stale(self._log_staled)

    # locks and pools
    def _lock_for(self, table: Dict[str, threading.Lock],
                  key: str) -> threading.Lock:
        with self._locks_guard:
            lock = table.get(key)
            if lock is None:
                lock = table[key] = threading.Lock()
            return lock

    def _obligation_pool(self) -> Optional[ThreadPoolExecutor]:
        if self.parallel_obligations <= 1:
            return None
        with self._pool_guard:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.parallel_obligations,
                    thread_name_prefix="gv-obligation")
            return self._pool

    # entry
    def verify(self, claim_id: str, force: bool = False) -> str:
        """Verify one claim; returns its resulting status. Idempotent, and
        serialized per claim: two callers verifying the same claim
        concurrently (parallel certificate slices) do the work once.

        A verdict the screener settled stands like any other until the
        claim goes stale, except that an explicit request (force: the
        solver's `verify` verb) reopens it for the full formal check."""
        with self._lock_for(self._claim_locks, claim_id):
            row = self.store.get(claim_id)
            if force and row.get("confident") \
                    and row["status"] in (STATUS_VERIFIED, STATUS_REFUTED) \
                    and not self.store.superseded(claim_id):
                reason = ("explicit verify: the screener's verdict is "
                          "reopened for the full check")
                self.store.set_stale(claim_id, reason=reason)
                self.log.append("status_change",
                                {"claim": claim_id, "status": STATUS_STALE,
                                 "reason": reason})
            # read before the row: an invalidation landing in between
            # makes the check's settlement refused, never silently wrong
            generation = self.store.generation(claim_id)
            row = self.store.get(claim_id)
            if row["status"] not in (STATUS_UNVERIFIED, STATUS_STALE):
                return row["status"]
            if self.store.superseded(claim_id):
                # replaced by its revision: a check would settle nothing
                return row["status"]
            if (row["status"] == STATUS_UNVERIFIED and not force
                    and self._checked_on_this_world(claim_id)):
                # the whole check already ran on this very world and could
                # not settle the claim: running it again (every queued
                # certificate request did, 30-100 minutes each) spends the
                # same seats on the same evidence. gv.verify forces it.
                self.not_rechecked.add(claim_id)
                self.log.append("verify_skipped",
                                {"claim": claim_id,
                                 "reason": "already checked on this world"})
                return row["status"]
            self.not_rechecked.discard(claim_id)
            self._check_generation[claim_id] = generation
            try:
                with model_scope(
                        claim_id=row["id"], claim_kind=row["kind"],
                        locus=("final_task" if row["kind"] == CLAIM_FINAL
                               else "support")):
                    if row["kind"] == CLAIM_EFFECT:
                        self._verify_effect(row)
                    else:
                        if row["kind"] == CLAIM_FINAL:
                            self._publish_candidate_response(row)
                        self._freeze_record(row)
                        self._verify_assertion(row, force=force)
            finally:
                self._check_generation.pop(claim_id, None)
            return self.store.status(claim_id)

    def _check_raised(self, claim_id: str, error: Exception) -> None:
        """A triggered check raised (a seat outage, a compile error): the
        claim keeps its status, the failure is on its evidence and in the
        ledger, and the rest of the triggered batch still runs (see
        ClaimStore.fire)."""
        message = f"check raised {type(error).__name__}: {error}"[:500]
        self.store.record_evidence(
            claim_id, {"outcome": OUTCOME_FAILED, "error": message})
        self.log.append("status_change",
                        {"claim": claim_id,
                         "status": self.store.status(claim_id),
                         "reason": "check raised", "error": message})

    def _log_staled(self, staled, revised: str) -> None:
        """A revision's cone marked these stale: the ledger records each
        status change (obligation nodes under their own key, so readers
        that follow claim verdicts are not misled)."""
        for cid, kind in staled:
            key = "obligation" if kind == OBLIGATION_KINDS else "claim"
            self.log.append("status_change",
                            {key: cid, "status": STATUS_STALE,
                             "reason": "cone", "revised": revised})

    def _report_status(self, claim_id: str, applied: bool, proposed: str,
                       reason: Optional[str] = None) -> None:
        """Log the status the store actually holds after a settlement: the
        proposed one when applied, otherwise the held one with the reason
        the settlement was refused (never a verdict the store lacks)."""
        payload = {"claim": claim_id}
        if applied:
            payload["status"] = proposed
        else:
            payload["status"] = self.store.status(claim_id)
            payload["settle_refused"] = (
                self.store.settle_refusal(
                    claim_id, self._check_generation.get(claim_id))
                or "refused")
            payload["proposed"] = proposed
        if reason is not None:
            payload["reason"] = reason
        self.log.append("status_change", payload)

    def _conclude_unverified(self, claim_id: str, reason: str) -> bool:
        """A check that ended without settling the claim (premise chain
        refuted, nothing checkable, critic fail-closed) leaves it
        unverified in the store too — the ledger and the store agree."""
        if claim_id == ContractCompiler.PRECOMPILE_ID \
                or getattr(self.store, "set_unverified", None) is None:
            # a precompile's pseudo-claim has no store row (and the
            # concurrent precompile critiques against an evidence buffer)
            self.log.append("status_change",
                            {"claim": claim_id, "status": STATUS_UNVERIFIED,
                             "reason": reason})
            return False
        applied = self.store.set_unverified(
            claim_id, reason=reason,
            generation=self._check_generation.get(claim_id))
        self._report_status(claim_id, applied, STATUS_UNVERIFIED,
                            reason=reason)
        return applied

    # the verifier's own state (rewritten on every solver act) and bytecode
    # caches (rewritten by merely running the code) are not the world
    FINGERPRINT_SKIP = frozenset({".gv", "__pycache__"})
    FINGERPRINT_MAX_ENTRIES = 200_000

    def checking(self, claim_id: str) -> bool:
        """Whether a check of this claim is running right now."""
        lock = self._claim_locks.get(claim_id)
        return bool(lock is not None and lock.locked())

    def _world_fingerprint(self) -> Optional[str]:
        """A cheap identity of the delivered world as it stands: every entry
        under the recorded root, and each file's mode, size and times (the
        VCS store included: a commit changes the world). None offline,
        without a baseline root, or when the tree is too large to walk
        cheaply. State outside the files (a running service, an installed
        package) is not in it: gv.verify reruns a check after such a change."""
        if not self.live_world:
            return None
        from ..baseline import delivered_root
        state = os.path.dirname(os.path.abspath(self.log.path))
        root = delivered_root(state)
        if not root:
            return None
        digest = hashlib.sha256()
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in self.FINGERPRINT_SKIP)
            for name in dirnames + sorted(filenames):
                path = os.path.join(dirpath, name)
                try:
                    st = os.lstat(path)
                except OSError:
                    continue
                # a directory's own times move with its entries, which are
                # walked anyway; a file's mode and change time catch chmod
                # and chown, which leave its content alone
                stamp = ("dir" if name in dirnames else
                         f"{st.st_mode}\0{st.st_size}\0{st.st_mtime_ns}"
                         f"\0{st.st_ctime_ns}")
                digest.update(f"{os.path.relpath(path, root)}\0{stamp}\n"
                              .encode("utf-8", "surrogateescape"))
                count += 1
                if count > self.FINGERPRINT_MAX_ENTRIES:
                    return None
        return digest.hexdigest()

    def _checked_on_this_world(self, claim_id: str) -> bool:
        """Whether the claim's last complete check ran on the world as it
        stands now (same fingerprint), on the graph as it stands now (same
        generation: a check from before the claim was invalidated judged
        premises that were revised since). A check some of whose
        obligations failed on an infrastructure error carries no
        fingerprint: it is rerun once the seats are back."""
        epoch = next((e for e in reversed(self.store.evidence_for(claim_id))
                      if e.get("outcome") == "epoch"), None)
        if not epoch or not epoch.get("world_fingerprint"):
            return False
        if "generation" in epoch \
                and epoch["generation"] != self.store.generation(claim_id):
            return False
        return epoch["world_fingerprint"] == self._world_fingerprint()

    def _publish_candidate_response(self, row: Dict) -> None:
        """The final claim under check is the candidate response until the
        solver writes its reply; the seats read it at a fixed path in the
        state (see authoritative_task_context)."""
        from ..execution.instruments import publish_candidate_response
        state = os.path.dirname(os.path.abspath(self.log.path))
        try:
            path = publish_candidate_response(state, row["text"])
        except OSError as error:
            self.log.append("candidate_response_published",
                            {"claim": row["id"], "error": str(error)[:200]})
            return
        if path:
            self.log.append("candidate_response_published",
                            {"claim": row["id"],
                             "file": os.path.basename(path)})

    def _freeze_record(self, row: Dict) -> None:
        """Bind an immutable ledger prefix to this claim's verification.

        The retained record a claim is judged against must not grow while
        the seats read it: the executor's complete scan and the Judger's
        independent recount have to see the same bounded file.  The
        boundary is the claim's own commit seq when known, otherwise the
        ledger's end now.  A publication failure keeps the live ledger and
        is recorded; it never changes what is verified.
        """
        boundary = None
        if self._record_boundary is not None:
            try:
                boundary = self._record_boundary(row["id"])
            except Exception:
                boundary = None
        if (isinstance(boundary, bool) or not isinstance(boundary, int)
                or boundary < 0):
            boundary = self.log.last_seq()
        payload = {"claim": row["id"], "through_seq": boundary}
        try:
            path = self.log.freeze_record(boundary)
        except (OSError, ValueError) as error:
            payload["error"] = str(error)[:300]
        else:
            self.log.bind_record(row["id"], path)
            payload["record"] = os.path.basename(path)
        self.log.append("record_frozen", payload)

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
        # an effect's evidence IS the ledger: record basis by construction;
        # the digest check itself is a decision procedure for "the event
        # is recorded as claimed", so its strength is decisive
        self._settle(row["id"], [result.outcome], [BASIS_RECORD],
                     [STRENGTH_DECISIVE], [SEVERITY_MUST])

    # screening
    # assertion / final
    def _verify_assertion(self, row: Dict, force: bool = False) -> None:
        # the epoch this verification began in; the claim's verdict is
        # stamped with it (see _discharge), never with a later one — and the
        # world's fingerprint then, so an unsettled check is not rerun on the
        # same world (verify)
        claim_epoch = self.settler.current_epoch()
        claim_world = self._world_fingerprint()
        # Local verdicts are CONDITIONAL: a claim is verified/refuted
        # assuming its premises (deps) are true, and the read-time
        # effective fold (store.effective) charges the premise chain's
        # actual state to every consumer. One premise state is not worth
        # verifying under: a refuted chain — the effective verdict is
        # refuted regardless of what local adjudication would say, so the
        # instrument spend is pure waste. A FINAL is exempt: its
        # obligations are the task contract, compiled and established
        # independently of the argument's health — a refuted intermediate
        # eliminates that line of support, not the deliverable's verdict.
        refuted_deps = [] if row["kind"] == CLAIM_FINAL else [
            p for p in self.store.parents(row["id"])
            if self.store.effective(p)["effective_status"] == STATUS_REFUTED
        ]
        if refuted_deps:
            self.store.record_evidence(
                row["id"],
                {"outcome": OUTCOME_FAILED,
                 "error": "premise chain refuted; conditional verification "
                          "skipped (effective verdict is refuted)",
                 "refuted_deps": refuted_deps},
            )
            self._conclude_unverified(row["id"], "premise chain refuted")
            return

        # screening: a claim whose verdict the retained record already
        # settles takes that verdict, unchecked; an explicit request
        # (force) always checks. A final is screened only when finals are
        # handled like any claim (unify_final).
        screen = (self.screener_role is not None and not force
                  and (row["kind"] != CLAIM_FINAL
                       or self.unify_final)
                  and row["status"] == STATUS_UNVERIFIED)
        compiled = None
        if screen and self.speculative_discharge:
            # the compiler starts alongside the screener: on a screen that
            # settles nothing its answer is ready when the screener
            # returns; on a settled verdict the call is simply
            # dropped (a spent seat, no waiting solver)
            from concurrent.futures import ThreadPoolExecutor
            variables = self.contracts.compile_inputs(row)
            pool = ThreadPoolExecutor(max_workers=1,
                                      thread_name_prefix="gv-compile")
            compiled = (variables,
                        submit_with_context(pool, self.obligations_role.call,
                                            self.backend, variables))
            pool.shutdown(wait=False)
            self.log.append("speculative_compile", {"claim": row["id"]})
        if screen and self.screener.screen(
                row, self._check_generation.get(row["id"])):
            return

        speculated: Dict[str, tuple] = {}
        speculate = None
        if self.speculative_discharge and self.binder is not None:
            def speculate(unreviewed):
                for entry in self.settler.settle_obligations(row, unreviewed):
                    speculated[entry[0].id] = entry
        obligations = self.contracts.compile_obligations(row, speculate=speculate,
                                                compiled=compiled)
        if obligations is None:
            return  # critic fail-closed: evidence and status already recorded
        if not obligations:
            # fail-closed: nothing checkable means nothing verified
            for oid in speculated:
                self.store.remove_obligation_edge(row["id"], oid)
            self.store.record_evidence(
                row["id"],
                {"outcome": OUTCOME_FAILED, "error": "no obligations"},
            )
            self._conclude_unverified(row["id"], "no obligations")
            return

        if speculated:
            kept = {o.id for o in obligations}
            for oid in list(speculated):
                if oid not in kept:
                    self.store.remove_obligation_edge(row["id"], oid)
                    del speculated[oid]
            fresh = [o for o in obligations if o.id not in speculated]
            settled_fresh = self.settler.settle_obligations(row, fresh) if fresh \
                else []
            by_id = dict(speculated)
            by_id.update({r[0].id: r for r in settled_fresh})
            # compile order, over the reviewed decomposition only
            settled = [by_id[o.id] for o in obligations if o.id in by_id]
        else:
            settled = self.settler.settle_obligations(row, obligations)
        outcomes = [r[1] for r in settled]
        bases = [r[3] or BASIS_RECORD for r in settled]
        strengths = [r[4] for r in settled]
        severities = [r[0].severity for r in settled]
        refuting_witness = next(
            (r[2] for r in settled if r[1] == OUTCOME_REFUTED
             and r[0].severity == SEVERITY_MUST), None)
        # the epoch this verdict is about (a world-basis claim's verdict
        # is about the world as it was now); recorded before the fold so
        # the fold's own rows stay the last word on the claim. A check
        # some of whose obligations failed on a seat error is not a
        # complete check of this world: it gets no fingerprint, so it is
        # not taken as the last word once the seats recover (verify)
        infrastructure = sum(1 for r in settled if r[5])
        epoch_evidence = {"outcome": "epoch",
                          "world_epoch": claim_epoch,
                          "world_fingerprint": (None if infrastructure
                                                else claim_world)}
        if row["id"] in self._check_generation:
            epoch_evidence["generation"] = self._check_generation[row["id"]]
        if infrastructure:
            epoch_evidence["infrastructure_failures"] = infrastructure
        self.store.record_evidence(row["id"], epoch_evidence)
        applied = self._settle(row["id"], outcomes, bases, strengths,
                               severities)
        refutation = next((r for r in settled if r[1] == OUTCOME_REFUTED
                           and r[0].severity == SEVERITY_MUST), None)
        if refutation is not None and applied:
            evidence = self.store.evidence_for(refutation[0].id)[-1]
            self.store.record_evidence(row["id"], {
                "outcome": OUTCOME_REFUTED,
                "refuting_obligation": refutation[0].id,
                "witness": evidence.get("witness"),
                "witness_kind": evidence.get("witness_kind"),
                "artifact": evidence.get("artifact", ""),
            })

        if self.memory is not None:
            status = self.store.status(row["id"])
            self.memory.record_obligations(
                row["text"], [o.text for o in obligations], status)
            if status == STATUS_REFUTED:
                self.memory.record_attack(
                    row["text"], row["text"],
                    str(refuting_witness), kind="refutation")
            self.log.append("memory_recorded", {"claim": row["id"]})

    # settlement
    # the final-contract precompiles, started and adopted by the entry
    # points (see ContractCompiler)
    def start_final_precompile(self) -> bool:
        return self.contracts.start_final_precompile()

    def adopt_precompiled_final_contract(self, handle) -> None:
        self.contracts.adopt_precompiled_final_contract(handle)

    def _settle(self, claim_id: str, outcomes: List[str],
                bases: List[str] = (),
                strengths: List[Optional[str]] = (),
                severities: List[str] = ()) -> bool:
        """Fold the obligation outcomes into the claim's local verdict
        (engine.fold) and apply it. Returns whether the store took it: a
        claim revised, or invalidated by a revision in its cone, while its
        check ran keeps the status it holds, and the ledger says so
        (settle_refused)."""
        generation = self._check_generation.get(claim_id)
        verdict = fold(outcomes, bases, strengths, severities,
                       completion_claim=(self.store.get(claim_id)["kind"]
                                         == CLAIM_FINAL))
        if verdict.evidence is not None:
            self.store.record_evidence(claim_id, verdict.evidence)
        if verdict.status == STATUS_REFUTED:
            applied = self.store.set_refuted(
                claim_id, reason=verdict.reason, basis=verdict.basis,
                strength=verdict.strength, generation=generation)
        elif verdict.status == STATUS_VERIFIED:
            applied = self.store.set_verified(
                claim_id, basis=verdict.basis, strength=verdict.strength,
                generation=generation)
        elif verdict.status == STATUS_PARTIAL:
            applied = self.store.set_partial(claim_id, generation=generation)
        else:
            applied = self.store.set_unverified(
                claim_id, reason=verdict.reason, generation=generation)
        self._report_status(claim_id, applied, verdict.status)
        return applied
