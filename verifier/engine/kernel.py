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
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Dict, List, Optional

from ..execution.discharge import AgentDischarger
from ..execution.resource_share import warmup_launches
from ..config import (
    APPEND_ONLY_DELTA_PROTOCOL,
    RunOptions,
    append_only_deltas_enabled,
)
from ..core.events import EventLog
from ..execution.instruments import (
    ArtifactStore, LedgerInstrument, authoritative_task_context,
    world_evidence_context, world_map,
)
from ..core.model import (
    Obligation, BASIS_RECORD, CLAIM_EFFECT, CLAIM_FINAL, OBLIGATION_KINDS,
    STATUS_UNVERIFIED, STATUS_VERIFIED, STATUS_STALE, STATUS_REFUTED, STATUS_PARTIAL,
    OUTCOME_REFUTED, OUTCOME_FAILED, STRENGTH_DECISIVE, SEVERITY_MUST, SEVERITIES,
    TASK_BEARING_REQUIRED, TASK_BEARING_UNCLEAR,
)
from ..llm import prompts
from ..llm.agents import AgentRole, COMPILE_OBLIGATIONS_AGENT, CRITIC_AGENT
from ..llm.model_spans import model_scope, submit_with_context
from ..llm.runtime import capabilities_of
from ..core.store import ClaimStore
from .authority_view import (
    AuthorityViewUnavailable, build_authority_compilation_view,
)
from .contract_cache import (
    CompiledFinalContract, contract_cache_key,
)
from .source_evidence import bind_source_evidence_variables
from .quoted_identities import recorded_identities, restore
from .contract_critic import CriticGame
from .referent import dependency_referent_context
from .fold import fold
from .screening import Screener
from .settlement import ObligationSettler

# attack rounds on a decomposition before the claim fails closed: each round
# is one critic attempt; an `insufficient` verdict sends the decomposition
# back for one repair, and a counterexample still standing after the last
# round leaves the claim unverified. Default only — the kernel takes
# critic_rounds per run (part of the scoring config identity).
MAX_CRITIC_ROUNDS = 4


# How long a final waits for a final-contract precompile still under
# review before compiling its contract inline instead. While it waits it
# settles the compiler's draft (speculation), so the wait is rarely idle;
# a review still running after this long is stuck in critic rounds or on
# a hung seat, and an inline compile — minutes — is no slower than
# waiting on. The precompile keeps running and still serves later finals.
PRECOMPILE_WAIT_SECONDS = 600


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
        self.memory_hits = memory_hits
        self.live_world = live_world
        self.critic_rounds = (MAX_CRITIC_ROUNDS if critic_rounds is None
                              else critic_rounds)
        # Resolve the experimental repair representation once per Kernel.
        # Runtime environment mutation cannot change compiler semantics
        # halfway through a verification game.
        self.append_only_deltas = append_only_deltas_enabled()
        self.append_only_delta_protocol = (
            APPEND_ONLY_DELTA_PROTOCOL if self.append_only_deltas else None)
        self.obligations_role, self.critic_role = compile_roles_for_run(
            self.run_options, obligations_role or COMPILE_OBLIGATIONS_AGENT,
            critic_role)   # critic None -> no adversarial review
        # A final contract compiled concurrently with extraction; adopted
        # only under an identical contract key (see engine.precompile).
        self._precompiled_final = None
        # ADVISORY review for non-final claims: whole-task acceptance is
        # adjudicated only at the final (contract locus), so a non-final
        # decomposition defect costs the solver a wrong intermediate
        # verdict, never certificate soundness. With a budget set, the
        # critic reviews a non-final decomposition that many times and
        # the last repair stands unreviewed instead of failing closed;
        # 0 seats no critic for non-finals. None -> the full game.
        self.critic_rounds_nonfinal = critic_rounds_nonfinal
        # UNIFIED FINAL: the final takes the path of any claim — screened
        # (a high-confidence screener verdict settles it like any claim)
        # and reviewed on the non-final critic budget — while its contract,
        # when it is compiled, still comes from the authoritative task
        # statement.
        self.unify_final = unify_final
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
        self.memory = memory      # advisory stores; None -> static hints
        self.contract_cache = contract_cache
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
        if claim_id == self.PRECOMPILE_ID \
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
                       or getattr(self, "unify_final", False))
                  and row["status"] == STATUS_UNVERIFIED)
        compiled = None
        if screen and self.speculative_discharge:
            # the compiler starts alongside the screener: on a screen that
            # settles nothing its answer is ready when the screener
            # returns; on a settled verdict the call is simply
            # dropped (a spent seat, no waiting solver)
            from concurrent.futures import ThreadPoolExecutor
            variables = self._compile_inputs(row)
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
        obligations = self._compile_obligations(row, speculate=speculate,
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

    # final-contract precompile: the task contract does not depend on the
    # solver's final wording (it is compiled from the authoritative task
    # statement with solver premises withheld), so it can be compiled —
    # and its methods enumerated — while the solver is still working,
    # off the critical path. Keyed by the task statement's content.
    PRECOMPILE_ID = "precompile:final"
    PRECOMPILE_TEXT = (
        "The task stated in the authoritative task statement has been "
        "completed as required. (The solver's own final wording is not "
        "available yet: compile the contract from the task statement "
        "alone; where the task's answer would be carried by the solver's "
        "response rather than by a named artifact, phrase that condition "
        "on the final claim's stated answer.)")

    def start_final_precompile(self) -> bool:
        """Compile the final's task contract (compiler + critic) and
        enumerate a method batch per obligation in a background thread.
        Returns False when there is nothing to precompile (no task
        statement, no live world, no binder, already started)."""
        state = os.path.dirname(os.path.abspath(self.log.path))
        if not self.live_world or self.binder is None \
                or getattr(self, "_precompile_thread", None) is not None:
            return False
        task = os.path.join(state, "task.md")
        if not os.path.isfile(task):
            return False
        self._final_contract: Optional[Dict] = None
        self._draft_contract: Optional[Dict] = None
        self._contract_ready = threading.Event()
        self._precompile_thread = threading.Thread(
            target=self._precompile_final, name="gv-precompile", daemon=True)
        self._precompile_thread.start()
        return True

    @staticmethod
    def _task_hash(state: str) -> Optional[str]:
        task = os.path.join(state, "task.md")
        if not os.path.isfile(task):
            return None
        with open(task, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _precompile_final(self) -> None:
        state = os.path.dirname(os.path.abspath(self.log.path))
        task_hash = self._task_hash(state)
        row = {"id": self.PRECOMPILE_ID, "kind": CLAIM_FINAL,
               "text": self.PRECOMPILE_TEXT}
        started = time.time()
        enumerate = getattr(self.binder, "prewarm", None)
        pool = self._obligation_pool()
        draft_futures = []

        def on_draft(draft):
            # the compiler's draft: usable for speculation by a final that
            # arrives before the review ends, and warm enough to start
            # enumerating now — content-addressed nodes mean every
            # obligation the review keeps is already served
            try:
                obligations = self._materialize_obligations(
                    row, draft, record=False) or []
            except (KeyError, TypeError, ValueError):
                return   # a malformed draft: the review decides alone
            self._draft_contract = {"task_hash": task_hash,
                                    "payload": draft}
            if enumerate is not None and pool is not None:
                draft_futures.extend(
                    submit_with_context(pool, warm, row, o)
                    for o in obligations)

        def warm(row, obligation):
            # a warm-up: while the review runs, its launches wait behind
            # the review's (and every check's), which the envelope's room
            # must serve first; the final then waits on no queued draft
            with warmup_launches(until=self._contract_ready.is_set):
                enumerate(row, obligation)

        try:
            with model_scope(claim_id=row["id"], claim_kind=CLAIM_FINAL,
                             locus="final_task"):
                payload = self._compile_obligation_payload(
                    row, on_draft=on_draft)
        except Exception as err:  # noqa: BLE001 - background, never fatal
            self.log.append("contract_precompiled",
                            {"task_hash": task_hash, "error": str(err)[:300],
                             "seconds": round(time.time() - started, 1)})
            self._contract_ready.set()
            return
        if payload is None:
            self.log.append("contract_precompiled",
                            {"task_hash": task_hash,
                             "error": "decomposition failed critic",
                             "seconds": round(time.time() - started, 1)})
            self._contract_ready.set()
            return
        self._final_contract = {"task_hash": task_hash, "payload": payload}
        self.log.append(
            "contract_precompiled",
            {"task_hash": task_hash,
             "obligations": [o["text"] for o in payload["obligations"]],
             "seconds": round(time.time() - started, 1)})
        # the contract is usable from here on; a commit arriving during
        # the prewarm below waits only for the batches it needs
        self._contract_ready.set()
        # method batches: the oracle only reads, so its enumeration is
        # as valid now as at commit time; the scripts run against the
        # world as it is then. Draft batches already served are skipped.
        obligations = self._materialize_obligations(row, payload,
                                                    record=False) or []
        if enumerate is None:
            return
        for fut in draft_futures:
            try:
                fut.result()
            except Exception:  # noqa: BLE001 - a warm-up only
                pass
        if pool is None:
            for obligation in obligations:
                enumerate(row, obligation)
        else:
            list(pool.map(lambda o: enumerate(row, o), obligations))

    def _precompiled_contract(self, row: Dict, speculate=None) -> Optional[Dict]:
        """The precompiled final contract for the current task statement,
        waiting for a precompile still under review. While waiting, a
        caller that can speculate settles the compiler's draft: whatever
        the review keeps is then already settled (content-addressed
        nodes); the caller reconciles against the reviewed contract."""
        thread = getattr(self, "_precompile_thread", None)
        if thread is None or row["kind"] != CLAIM_FINAL \
                or row["id"] == self.PRECOMPILE_ID \
                or threading.current_thread() is thread:
            return None
        state = os.path.dirname(os.path.abspath(self.log.path))
        task_hash = self._task_hash(state)
        waited_from = time.time()
        if not self._contract_ready.is_set() and speculate is not None:
            draft = getattr(self, "_draft_contract", None)
            if draft and draft["task_hash"] == task_hash:
                unreviewed = self._materialize_obligations(
                    row, draft["payload"], record=False) or []
                self.log.append(
                    "speculative_discharge",
                    {"claim": row["id"], "source": "precompile draft",
                     "obligations": [o.id for o in unreviewed]})
                speculate(unreviewed)
        remaining = PRECOMPILE_WAIT_SECONDS - (time.time() - waited_from)
        if not self._contract_ready.wait(max(0.0, remaining)):
            # bounded: the final compiles its own contract (the normal
            # path) rather than idle on a review that does not end
            self.log.append("contract_precompiled",
                            {"claim": row["id"], "adopted": False,
                             "reason": "precompile still under review after "
                                       f"{PRECOMPILE_WAIT_SECONDS}s; "
                                       "compiled inline"})
            return None
        contract = getattr(self, "_final_contract", None)
        if not contract or contract["task_hash"] != task_hash:
            return None
        return contract["payload"]

    def _compile_obligations(self, row: Dict, speculate=None,
                             compiled=None):
        """Compile the claim's obligations. With ``speculate`` (a callable
        taking the compiler's unreviewed obligations and settling them),
        the critic's review runs concurrently with those checks; the
        reviewed decomposition is what this returns either way."""
        if self.contract_replay is not None:
            if self.live_world:
                raise ValueError(
                    "sealed contract replay is an offline diagnostic only")
            self.contract_replay.require_replayable()
            validate_kernel = getattr(
                self.contract_replay, "validate_kernel", None)
            if not callable(validate_kernel):
                raise ValueError(
                    "sealed contract replay must be bound to this kernel")
            validate_kernel(self)
            # Lookup is deliberately the first operation.  The bundle's
            # compile-input identities were checked once against a complete,
            # fresh pre-execution graph; recomputing them after dependencies
            # begin to settle would make a valid replay schedule-dependent.
            entry = self.contract_replay.lookup(row["id"])
            # a sealed replay reproduces the recorded texts exactly
            obligations = self._materialize_obligations(
                row, entry.payload, record=False, repair=False)
            self.contract_replay.validate_materialized(entry, obligations)
            self._record_compiled_obligations(row, obligations)
            return obligations

        payload = self._compile_obligation_payload(
            row, speculate=speculate, compiled=compiled)
        return self._materialize_obligations(row, payload)

    def _compile_inputs(self, row: Dict) -> Dict:
        """The Compiler's bound inputs for a non-final claim (the same
        values the inline compile uses), built without invoking a role."""
        return bind_source_evidence_variables(
            self.log,
            self._obligation_compile_variables(row),
            enabled=self.run_options.compact_inspection,
        )

    def _compile_obligation_payload(self, row: Dict, speculate=None,
                                    compiled=None, on_draft=None):
        """Run the unchanged Compiler/Critic path and return its raw payload.

        This extraction is a seam for the explicit offline isolation setup;
        default verification still calls it only through
        :meth:`_compile_obligations` and materializes exactly once afterwards.
        ``on_draft`` receives the compiler's payload before the critic
        reviews it (the precompile warms and publishes the draft).
        """
        state = os.path.dirname(os.path.abspath(self.log.path))
        cached = self._precompiled_contract(row, speculate=speculate)
        if cached is not None:
            self.log.append("contract_cache_hit", {"claim": row["id"]})
            return cached
        handled, payload = self._authority_final_contract(
            row, state, on_draft=on_draft)
        if handled:
            return payload
        if compiled is not None:
            variables, future = compiled
            payload = future.result()
        else:
            variables = self._compile_inputs(row)
            with model_scope(stage="contract_compile", round=0):
                payload = self.obligations_role.call(self.backend, variables)
        if on_draft is not None:
            on_draft(payload)
        if speculate is not None and self.critic_role is not None:
            # SPECULATIVE DISCHARGE: the critic reviews while the
            # compiler's obligations are already being settled; nodes
            # are content-addressed, so the reviewed decomposition finds
            # its kept obligations settled and only the added ones wait
            from concurrent.futures import ThreadPoolExecutor
            unreviewed = self._materialize_obligations(
                row, payload, record=False) or []
            self.log.append(
                "speculative_discharge",
                {"claim": row["id"],
                 "obligations": [o.id for o in unreviewed]})
            with ThreadPoolExecutor(max_workers=1,
                                    thread_name_prefix="gv-critic") as pool:
                review = submit_with_context(
                    pool, self._critique, row, payload, variables)
                speculate(unreviewed)
                payload = review.result()
            if payload is None:
                for o in unreviewed:
                    self.store.remove_obligation_edge(row["id"], o.id)
            return payload
        return self._critique(row, payload, variables)

    def _obligation_compile_variables(self, row: Dict) -> Dict:
        """Build the legacy Compiler inputs without invoking any role.

        The returned values are byte-for-byte the values used by the normal
        Compiler/Critic path.  Isolation setup may hash this one fresh,
        complete pre-execution snapshot with the diagnostic coordinator's
        exact host-slot normalizer; production scheduling and role inputs are
        otherwise unchanged.
        """
        state = os.path.dirname(os.path.abspath(self.log.path))
        memory_hits = (
            self.memory.hits_for_obligations(row["text"])
            if self.memory is not None else self.memory_hits
        )
        # Every final is independent of its premises: the effective fold
        # exempts finals even when no separate task statement was supplied.
        # In that case the final's own wording is the available contract.
        # Solver-derived graph nodes may later supply evidence for the
        # obligations it emits, but they must not become axioms that let the
        # compiler residualize an authoritative task requirement out of the
        # contract.  Exact obligation dedup/inheritance still happens after
        # compilation, so sound reuse remains available.
        task_context = authoritative_task_context(state)
        independent_final = row["kind"] == CLAIM_FINAL
        # graph context: EVERY premise (dep) is assumable — the claim's
        # verdict is conditional on its premises by definition, and the
        # read-time effective fold charges each premise's actual state
        # (still-open, record-level, refuted) to every consumer. The
        # compiler therefore compiles the RESIDUAL: what must hold beyond
        # the premises. Verified premises carry their settled provenance
        # tag; unsettled ones are assumed premises whose eventual state
        # the fold accounts for. The solver's argument stays invisible
        # either way.
        verified_deps, pending_deps = [], []
        if not independent_final:
            for pid in self.store.parents(row["id"]):
                prow = self.store.get(pid)
                eff = self.store.effective(pid)
                if prow["kind"] == CLAIM_EFFECT:
                    label = "[a verified tool-execution record]"
                    if eff["effective_status"] == STATUS_VERIFIED:
                        verified_deps.append(label)
                    else:
                        pending_deps.append(
                            "[a tool-execution record, assumed — status: "
                            f"{eff['effective_status']}]")
                    continue
                if eff["effective_status"] == STATUS_VERIFIED:
                    basis = eff["effective_basis"] or BASIS_RECORD
                    tag = (", settled by the screener on the record"
                           if prow.get("confident") else "")
                    verified_deps.append(
                        f"[{basis}-level{tag}] {prow['text']}")
                else:
                    pending_deps.append(
                        f"[assumed premise — status: "
                        f"{eff['effective_status']}] {prow['text']}")
        # a run scored after the fact (offline ORM) has no live world; the
        # harness may instead have captured the delivered tree next to the
        # ledger. Absolute, like ledger_path: the checker runs elsewhere.
        world_mode, world, _ = world_evidence_context(
            state, live_world=self.live_world)
        graph = (
            [{"note": "withheld: harness final contract is compiled "
                      "independently from authoritative task context"}]
            if independent_final
            else self.store.graph_snapshot(exclude=row["id"])
        )
        referent_context = (
            {"withheld": "harness final contract is independent of the "
                         "solver-derived dependency graph"}
            if independent_final
            else dependency_referent_context(self.store, self.log, row["id"])
        )
        return {
            "claim_text": row["text"],
            "claim_kind": row["kind"],
            # division-of-labor fact, shared verbatim with the critic: for a
            # non-final claim the task contract is owned by the final node,
            # so neither seat treats this claim as the last line of defense
            "contract_locus": (
                prompts.CONTRACT_LOCUS_FINAL if row["kind"] == CLAIM_FINAL
                else prompts.CONTRACT_LOCUS_NONFINAL),
            "task_context": task_context,
            "ledger_path": self.log.record_path_for(row["id"]),
            "world_mode": world_mode,
            "world_path": world,
            "world_map": world_map(state, live_world=self.live_world),
            "verified_deps": json.dumps(verified_deps, ensure_ascii=False),
            "pending_deps": json.dumps(pending_deps, ensure_ascii=False),
            "dependency_referent_context": json.dumps(
                referent_context, ensure_ascii=False, separators=(",", ":")),
            "graph": json.dumps(graph, ensure_ascii=False),
            "memory_hits": memory_hits,
            "critic_feedback": "",
        }

    def _authority_final_contract(self, row: Dict, state: str,
                                  on_draft=None):
        """Compile a final through the candidate-independent authority boundary.

        Cache availability controls reuse only.  A fresh miss and a cache-off
        run have the same answer-blind Compiler/Critic inputs and tool boundary;
        otherwise disabling the cache would leak the final response back into
        contract semantics.  Unsupported inputs use the legacy path.  The
        Oracle and all candidate verification run afterwards.
        """
        if row["kind"] != CLAIM_FINAL:
            return False, None
        if self.critic_role is None or self.critic_rounds < 1:
            self.log.append("contract_cache_bypass", {
                "claim": row["id"], "reason": "independent Critic required"})
            return False, None
        from ..llm.authority_backend import (
            AuthorityBackendUnavailable, authority_compilation_backend,
        )
        from ..llm.schema import validate

        if self._precompiled_final is not None and self.contract_cache is None:
            adopted, payload = self._adopt_precompiled_final(row, state)
            if adopted:
                return True, payload

        try:
            view = build_authority_compilation_view(state)
            with authority_compilation_backend(
                    self.backend, self.obligations_role,
                    self.critic_role) as session:
                variables = self._authority_contract_variables(view)
                key = contract_cache_key(
                    view=view, variables=variables,
                    backend_identity=session.identity,
                    obligations_role=session.obligations_role,
                    critic_role=session.critic_role,
                    critic_rounds=self.critic_rounds,
                    append_only_deltas=self.append_only_deltas,
                    append_only_delta_protocol=
                        self.append_only_delta_protocol)
                if self.contract_cache is None:
                    with model_scope(stage="contract_compile", round=0):
                        payload = session.obligations_role.call(
                            session.backend, variables)
                    if on_draft is not None:
                        on_draft(payload)
                    payload = self._critique(
                        row, payload, variables, backend=session.backend,
                        obligations_role=session.obligations_role,
                        critic_role=session.critic_role)
                    view.revalidate(state)
                    session.revalidate()
                    if payload is not None:
                        validate(session.obligations_role.schema, payload)
                    return True, payload
                lease = self.contract_cache.acquire(key)
                try:
                    # The lookup/single-flight wait may outlive an online task
                    # revision. Never attach a previous contract to that task.
                    view.revalidate(state)
                    session.revalidate()
                    if lease.entry is not None:
                        validate(session.obligations_role.schema,
                                 lease.entry.payload)
                        self.log.append("contract_cache_hit", {
                            "claim": row["id"], "key": key,
                            "authority": view.identity,
                            "source_digest": lease.entry.source_digest,
                            "waited": lease.waited})
                        return True, lease.entry.payload
                    self.log.append("contract_cache_miss", {
                        "claim": row["id"], "key": key,
                        "authority": view.identity,
                        "leader": lease.leader, "waited": lease.waited})
                    # A miss is where the concurrent authority-only
                    # compilation pays off: adopt it under the identical
                    # key and publish it for the task's other candidates.
                    adopted = False
                    if self._precompiled_final is not None:
                        adopted, payload = self._adopt_precompiled_final(
                            row, state)
                    if not adopted:
                        with model_scope(stage="contract_compile", round=0):
                            payload = session.obligations_role.call(
                                session.backend, variables)
                        if on_draft is not None:
                            on_draft(payload)
                        payload = self._critique(
                            row, payload, variables,
                            backend=session.backend,
                            obligations_role=session.obligations_role,
                            critic_role=session.critic_role)
                    view.revalidate(state)
                    session.revalidate()
                    if payload is None:
                        return True, None
                    validate(session.obligations_role.schema, payload)
                    # Empty decompositions never certify a contract.
                    if not payload["obligations"]:
                        return True, payload
                    entry = CompiledFinalContract.from_payload(payload)
                    published = True
                    if lease.leader:
                        lease.finish(entry)
                    else:
                        entry, published = self.contract_cache.publish_if_absent(key, entry)
                        payload = entry.payload
                    self.log.append("contract_cache_published" if published
                                    else "contract_cache_hit", {
                        "claim": row["id"], "key": key,
                        "authority": view.identity,
                        "waited": lease.waited,
                        "after_duplicate_fill": not published,
                        "source_digest": entry.source_digest})
                    return True, payload
                finally:
                    lease.finish()
        except (AuthorityViewUnavailable, AuthorityBackendUnavailable) as error:
            self.log.append("contract_cache_bypass", {
                "claim": row["id"], "reason": str(error)[:300]})
            return False, None

    def adopt_precompiled_final_contract(self, handle) -> None:
        """Register a concurrent authority-only compilation for the final."""
        self._precompiled_final = handle

    def _adopt_precompiled_final(self, row: Dict, state: str):
        """Use the concurrent compilation only under an identical key.

        The key covers the authority view, the withheld compile variables,
        the backend and role identities and the critic budget.  A mismatch,
        an authority bypass, or a failed thread all fall back to the
        unchanged inline compile; a failed thread's work is not reused.
        """
        from ..llm.authority_backend import (
            AuthorityBackendUnavailable, authority_compilation_backend,
        )

        handle = self._precompiled_final
        self._precompiled_final = None
        outcome = handle.wait()
        event = {"claim": row["id"],
                 "precompiled_claim": handle.pseudo_claim_id,
                 "elapsed_seconds": round(outcome.elapsed_seconds, 3),
                 "adopted": False}
        if outcome.bypass_reason is not None:
            event["reason"] = outcome.bypass_reason
            self.log.append("contract_precompiled", event)
            return False, None
        if outcome.error is not None:
            event["reason"] = (
                f"precompile failed: {type(outcome.error).__name__}")
            self.log.append("contract_precompiled", event)
            return False, None
        try:
            view = build_authority_compilation_view(state)
            with authority_compilation_backend(
                    self.backend, self.obligations_role,
                    self.critic_role) as session:
                expected = contract_cache_key(
                    view=view,
                    variables=self._authority_contract_variables(view),
                    backend_identity=session.identity,
                    obligations_role=session.obligations_role,
                    critic_role=session.critic_role,
                    critic_rounds=self.critic_rounds,
                    append_only_deltas=self.append_only_deltas,
                    append_only_delta_protocol=self.append_only_delta_protocol)
        except (AuthorityViewUnavailable, AuthorityBackendUnavailable) as error:
            event["reason"] = str(error)[:300]
            self.log.append("contract_precompiled", event)
            return False, None
        if expected != outcome.key:
            event["reason"] = "contract key differs from this kernel's"
            self.log.append("contract_precompiled", event)
            return False, None
        for kind, payload in outcome.events:
            self.log.append(kind, {**payload, "claim": row["id"]})
        for evidence in outcome.evidence:
            self.store.record_evidence(row["id"], evidence)
        event["adopted"] = True
        event["critic_rounds_used"] = sum(
            1 for kind, _payload in outcome.events
            if kind == "decomposition_attacked")
        self.log.append("contract_precompiled", event)
        return True, outcome.payload

    @staticmethod
    def _authority_contract_variables(view):
        withheld = "withheld: authority-only contract compilation"
        return {
            "claim_text": (
                "The fixed current candidate satisfies the complete "
                "authoritative task contract. Its response, delivered "
                "artifacts, and execution history are opaque subjects, not "
                "available facts. Compile requirements about that candidate "
                "without substituting another candidate or inventing values."),
            "claim_kind": CLAIM_FINAL,
            "contract_locus": prompts.CONTRACT_LOCUS_FINAL,
            "task_context": view.context_json,
            "ledger_path": withheld,
            "world_mode": "authority-only",
            "world_path": withheld,
            "world_map": withheld,
            "verified_deps": "[]", "pending_deps": "[]",
            "dependency_referent_context": json.dumps({"withheld": withheld}),
            "graph": "[]", "memory_hits": "[]", "critic_feedback": "",
        }

    def _recorded_identities(self):
        ids = getattr(self, "_recorded_ids", None)
        if ids is None:
            state = os.path.dirname(os.path.abspath(self.log.path))
            ids = recorded_identities(state)
            self._recorded_ids = ids
        return ids

    def _restore_quoted_identities(self, row: Dict, payload, record: bool):
        """Near copies of task-start identities in the compiled texts are
        restored to the recorded values (quoted_identities)."""
        identities = self._recorded_identities()
        if not identities:
            return payload
        items = []
        repairs = []
        for o in payload["obligations"]:
            text, restored = restore(o["text"], identities)
            if restored:
                repairs.append({"before": o["text"][:400],
                                "restored": [{"copied": c, "recorded": r}
                                             for c, r in restored]})
                o = dict(o, text=text)
            items.append(o)
        if not repairs:
            return payload
        if record:
            self.log.append("obligation_identity_restored",
                            {"claim": row["id"], "repairs": repairs})
        return dict(payload, obligations=items)

    def _materialize_obligations(self, row: Dict, payload, *, record=True,
                                 repair=True):
        if payload is None:
            return None
        if repair:
            payload = self._restore_quoted_identities(row, payload, record)
        independent_final = row["kind"] == CLAIM_FINAL
        obligations = [
            Obligation(row["id"], o["text"],
                       o.get("required_basis", BASIS_RECORD),
                       # edge attributes, each defaulting to the
                       # fail-closed reading when the compiler omits it
                       severity=(o.get("severity")
                                 if o.get("severity") in SEVERITIES
                                 else SEVERITY_MUST),
                       clause=str(o.get("clause") or ""),
                       refutable_by_absence=bool(
                           o.get("refutable_by_absence", False)),
                       task_bearing=(TASK_BEARING_REQUIRED
                                     if row["kind"] == CLAIM_FINAL
                                     else str(o.get("task_bearing")
                                              or TASK_BEARING_UNCLEAR)),
                       premise_scope=tuple(self.settler.premises_in_force(row)))
            for o in payload["obligations"]
        ]
        if not independent_final:
            # mechanical residual filter: an obligation that restates a
            # premise VERBATIM is a re-demand — the premise edge already
            # carries that condition and the read-time fold charges it to
            # this claim, so verifying it again here is duplicated work
            # (and a spurious-failure risk for an honest claim). Dropped
            # only on exact text identity; kept whenever dropping would
            # empty the decomposition (an empty compile fails closed, and
            # a claim that IS the conjunction of its premises still needs
            # the explicit list until an empty-residual mechanism exists).
            premise_texts = set()
            for pid in self.store.parents(row["id"]):
                prow = self.store.get(pid)
                if prow["kind"] != CLAIM_EFFECT:
                    premise_texts.add(prow["text"])
            residual = [o for o in obligations
                        if o.text not in premise_texts]
            dropped = [o.text for o in obligations
                       if o.text in premise_texts]
            if dropped and residual:
                if record:
                    self.log.append(
                        "obligations_residualized",
                        {"claim": row["id"], "dropped": dropped},
                    )
                obligations = residual
        if record:
            self._record_compiled_obligations(
                row, obligations, excluded=payload.get("excluded"))
        return obligations

    def _record_compiled_obligations(self, row: Dict,
                                     obligations: List[Obligation],
                                     excluded=None) -> None:
        event = {"claim": row["id"],
                 "obligations": [{"id": o.id, "text": o.text,
                                  "premise_scope": list(o.premise_scope),
                                  "required_basis": o.required_basis,
                                  "severity": o.severity,
                                  "clause": o.clause,
                                  "refutable_by_absence":
                                      o.refutable_by_absence,
                                  "task_bearing": o.task_bearing}
                                 for o in obligations]}
        if excluded:
            # The compiler's exclusion ledger, as the Critic reviewed it:
            # which candidate-facing clauses generated no obligation and why.
            event["excluded"] = [
                {"clause": str(item.get("clause", ""))[:300],
                 "reason": str(item.get("reason", ""))[:300]}
                for item in excluded if isinstance(item, dict)]
        self.log.append("obligations_compiled", event)

    def _critique(self, row: Dict, payload: Dict, variables: Dict, *,
                  backend=None, obligations_role=None, critic_role=None,
                  record: bool = True):
        """Review a decomposition in the critic game (engine.contract_critic):
        the reviewed payload, or None when it failed the critic."""
        return CriticGame(
            log=self.log, store=self.store, backend=self.backend,
            obligations_role=self.obligations_role,
            critic_role=self.critic_role, critic_rounds=self.critic_rounds,
            run_options=self.run_options,
            critic_rounds_nonfinal=self.critic_rounds_nonfinal,
            unify_final=self.unify_final,
            append_only_deltas=self.append_only_deltas,
            conclude_unverified=self._conclude_unverified,
        ).review(row, payload, variables, backend=backend,
                 obligations_role=obligations_role, critic_role=critic_role,
                 record=record)

    # settlement
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
