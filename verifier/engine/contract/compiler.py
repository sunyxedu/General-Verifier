"""Compiling a claim's contract: the obligations its verdict folds over.

The compiler seat decomposes a claim into obligations — answer-blind, and
for a final from the authoritative task statement alone, through the
authority-only boundary — and the critic game reviews the decomposition
before it stands. A final's contract may instead come from a source that
already holds it under the identical key: the online precompile (compiled
from the task statement while the solver works), the offline concurrent
precompile, the process-local contract cache, or a sealed replay bundle
(an offline diagnostic). Every source yields the same payload the inline
compile would; the materialized obligation nodes are then the kernel's to
settle.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Protocol

from ...core.model import (
    BASIS_RECORD, CLAIM_EFFECT, CLAIM_FINAL, Obligation, SEVERITIES,
    SEVERITY_MUST, STATUS_VERIFIED, TASK_BEARING_REQUIRED,
    TASK_BEARING_UNCLEAR,
)
from ...execution.instruments import (
    authoritative_task_context, world_evidence_context, world_map,
)
from ...execution.resource_share import warmup_launches
from ...llm import prompts
from ...llm.model_spans import model_scope, submit_with_context
from .authority_view import (
    AuthorityViewUnavailable, build_authority_compilation_view,
)
from .cache import CompiledFinalContract, contract_cache_key
from .critic import CriticGame
from .quoted_identities import recorded_identities, restore
from .referent import dependency_referent_context
from ..source_evidence import bind_source_evidence_variables

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


class CompilerHost(Protocol):
    """What the compiler reads from the verifier it compiles for: the claim
    store and ledger, the runtime its seats run on, whether the world is
    live, the run's options, and the binding game (whose method batches a
    precompile warms)."""
    store: object
    log: object
    backend: object
    live_world: bool
    run_options: object
    binder: object


class ContractCompiler:
    """Compiles, reviews, and materializes claim contracts for one host.
    The contract settings (seats, critic budgets, caches, memory hints) are
    the compiler's own; the host exposes them too, so a caller that tunes
    one on the kernel tunes it here."""

    def __init__(self, host: CompilerHost, *, obligations_role, critic_role,
                 critic_rounds: int, critic_rounds_nonfinal: Optional[int],
                 unify_final: bool, contract_cache, memory, memory_hits: str,
                 append_only_deltas: bool,
                 append_only_delta_protocol: Optional[str],
                 conclude_unverified: Callable, pool: Callable,
                 premises_in_force: Callable):
        self.host = host
        self.obligations_role = obligations_role
        self.critic_role = critic_role      # None -> no adversarial review
        self.critic_rounds = critic_rounds
        # ADVISORY review for non-final claims: whole-task acceptance is
        # adjudicated only at the final (contract locus), so a non-final
        # decomposition defect costs the solver a wrong intermediate
        # verdict, never certificate soundness. With a budget set, the
        # critic reviews a non-final decomposition that many times and
        # the last repair stands unreviewed instead of failing closed;
        # 0 seats no critic for non-finals. None -> the full game.
        self.critic_rounds_nonfinal = critic_rounds_nonfinal
        self.unify_final = unify_final
        self.contract_cache = contract_cache
        # Experimental adapter-isolation diagnostic only (attached after
        # construction, see verifier.contract_replay): it freezes a
        # Critic-approved contract, not any method, evidence, or verdict.
        self.contract_replay = None
        self.memory = memory
        self.memory_hits = memory_hits
        self.append_only_deltas = append_only_deltas
        self.append_only_delta_protocol = append_only_delta_protocol
        self.conclude_unverified = conclude_unverified
        self.pool = pool
        self.premises_in_force = premises_in_force
        # A final contract compiled concurrently with extraction; adopted
        # only under an identical contract key (see engine.precompile).
        self.precompiled_final = None
        # the online precompile's state (see start_final_precompile)
        self.precompile_thread: Optional[threading.Thread] = None
        self.final_contract: Optional[Dict] = None
        self.draft_contract: Optional[Dict] = None
        self.contract_ready: Optional[threading.Event] = None
        self.recorded_ids = None

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
        state = os.path.dirname(os.path.abspath(self.host.log.path))
        if not self.host.live_world or self.host.binder is None \
                or self.precompile_thread is not None:
            return False
        task = os.path.join(state, "task.md")
        if not os.path.isfile(task):
            return False
        self.final_contract: Optional[Dict] = None
        self.draft_contract: Optional[Dict] = None
        self.contract_ready = threading.Event()
        self.precompile_thread = threading.Thread(
            target=self.precompile_final, name="gv-precompile", daemon=True)
        self.precompile_thread.start()
        return True

    @staticmethod
    def task_hash(state: str) -> Optional[str]:
        task = os.path.join(state, "task.md")
        if not os.path.isfile(task):
            return None
        with open(task, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def precompile_final(self) -> None:
        state = os.path.dirname(os.path.abspath(self.host.log.path))
        task_hash = self.task_hash(state)
        row = {"id": self.PRECOMPILE_ID, "kind": CLAIM_FINAL,
               "text": self.PRECOMPILE_TEXT}
        started = time.time()
        enumerate = getattr(self.host.binder, "prewarm", None)
        pool = self.pool()
        draft_futures = []

        def on_draft(draft):
            # the compiler's draft: usable for speculation by a final that
            # arrives before the review ends, and warm enough to start
            # enumerating now — content-addressed nodes mean every
            # obligation the review keeps is already served
            try:
                obligations = self.materialize_obligations(
                    row, draft, record=False) or []
            except (KeyError, TypeError, ValueError):
                return   # a malformed draft: the review decides alone
            self.draft_contract = {"task_hash": task_hash,
                                    "payload": draft}
            if enumerate is not None and pool is not None:
                draft_futures.extend(
                    submit_with_context(pool, warm, row, o)
                    for o in obligations)

        def warm(row, obligation):
            # a warm-up: while the review runs, its launches wait behind
            # the review's (and every check's), which the envelope's room
            # must serve first; the final then waits on no queued draft
            with warmup_launches(until=self.contract_ready.is_set):
                enumerate(row, obligation)

        try:
            with model_scope(claim_id=row["id"], claim_kind=CLAIM_FINAL,
                             locus="final_task"):
                payload = self.compile_payload(
                    row, on_draft=on_draft)
        except Exception as err:  # noqa: BLE001 - background, never fatal
            self.host.log.append("contract_precompiled",
                            {"task_hash": task_hash, "error": str(err)[:300],
                             "seconds": round(time.time() - started, 1)})
            self.contract_ready.set()
            return
        if payload is None:
            self.host.log.append("contract_precompiled",
                            {"task_hash": task_hash,
                             "error": "decomposition failed critic",
                             "seconds": round(time.time() - started, 1)})
            self.contract_ready.set()
            return
        self.final_contract = {"task_hash": task_hash, "payload": payload}
        self.host.log.append(
            "contract_precompiled",
            {"task_hash": task_hash,
             "obligations": [o["text"] for o in payload["obligations"]],
             "seconds": round(time.time() - started, 1)})
        # the contract is usable from here on; a commit arriving during
        # the prewarm below waits only for the batches it needs
        self.contract_ready.set()
        # method batches: the oracle only reads, so its enumeration is
        # as valid now as at commit time; the scripts run against the
        # world as it is then. Draft batches already served are skipped.
        obligations = self.materialize_obligations(row, payload,
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

    def precompiled_contract(self, row: Dict, speculate=None) -> Optional[Dict]:
        """The precompiled final contract for the current task statement,
        waiting for a precompile still under review. While waiting, a
        caller that can speculate settles the compiler's draft: whatever
        the review keeps is then already settled (content-addressed
        nodes); the caller reconciles against the reviewed contract."""
        thread = self.precompile_thread
        if thread is None or row["kind"] != CLAIM_FINAL \
                or row["id"] == self.PRECOMPILE_ID \
                or threading.current_thread() is thread:
            return None
        state = os.path.dirname(os.path.abspath(self.host.log.path))
        task_hash = self.task_hash(state)
        waited_from = time.time()
        if not self.contract_ready.is_set() and speculate is not None:
            draft = self.draft_contract
            if draft and draft["task_hash"] == task_hash:
                unreviewed = self.materialize_obligations(
                    row, draft["payload"], record=False) or []
                self.host.log.append(
                    "speculative_discharge",
                    {"claim": row["id"], "source": "precompile draft",
                     "obligations": [o.id for o in unreviewed]})
                speculate(unreviewed)
        remaining = PRECOMPILE_WAIT_SECONDS - (time.time() - waited_from)
        if not self.contract_ready.wait(max(0.0, remaining)):
            # bounded: the final compiles its own contract (the normal
            # path) rather than idle on a review that does not end
            self.host.log.append("contract_precompiled",
                            {"claim": row["id"], "adopted": False,
                             "reason": "precompile still under review after "
                                       f"{PRECOMPILE_WAIT_SECONDS}s; "
                                       "compiled inline"})
            return None
        contract = self.final_contract
        if not contract or contract["task_hash"] != task_hash:
            return None
        return contract["payload"]

    def compile_obligations(self, row: Dict, speculate=None,
                             compiled=None):
        """Compile the claim's obligations. With ``speculate`` (a callable
        taking the compiler's unreviewed obligations and settling them),
        the critic's review runs concurrently with those checks; the
        reviewed decomposition is what this returns either way."""
        if self.contract_replay is not None:
            if self.host.live_world:
                raise ValueError(
                    "sealed contract replay is an offline diagnostic only")
            self.contract_replay.require_replayable()
            validate_kernel = getattr(
                self.contract_replay, "validate_kernel", None)
            if not callable(validate_kernel):
                raise ValueError(
                    "sealed contract replay must be bound to this kernel")
            validate_kernel(self.host)
            # Lookup is deliberately the first operation.  The bundle's
            # compile-input identities were checked once against a complete,
            # fresh pre-execution graph; recomputing them after dependencies
            # begin to settle would make a valid replay schedule-dependent.
            entry = self.contract_replay.lookup(row["id"])
            # a sealed replay reproduces the recorded texts exactly
            obligations = self.materialize_obligations(
                row, entry.payload, record=False, repair=False)
            self.contract_replay.validate_materialized(entry, obligations)
            self.record_compiled_obligations(row, obligations)
            return obligations

        payload = self.compile_payload(
            row, speculate=speculate, compiled=compiled)
        return self.materialize_obligations(row, payload)

    def compile_inputs(self, row: Dict) -> Dict:
        """The Compiler's bound inputs for a non-final claim (the same
        values the inline compile uses), built without invoking a role."""
        return bind_source_evidence_variables(
            self.host.log,
            self.compile_variables(row),
            enabled=self.host.run_options.compact_inspection,
        )

    def compile_payload(self, row: Dict, speculate=None,
                                    compiled=None, on_draft=None):
        """Run the unchanged Compiler/Critic path and return its raw payload.

        This extraction is a seam for the explicit offline isolation setup;
        default verification still calls it only through
        :meth:`_compile_obligations` and materializes exactly once afterwards.
        ``on_draft`` receives the compiler's payload before the critic
        reviews it (the precompile warms and publishes the draft).
        """
        state = os.path.dirname(os.path.abspath(self.host.log.path))
        cached = self.precompiled_contract(row, speculate=speculate)
        if cached is not None:
            self.host.log.append("contract_cache_hit", {"claim": row["id"]})
            return cached
        handled, payload = self.authority_final_contract(
            row, state, on_draft=on_draft)
        if handled:
            return payload
        if compiled is not None:
            variables, future = compiled
            payload = future.result()
        else:
            variables = self.compile_inputs(row)
            with model_scope(stage="contract_compile", round=0):
                payload = self.obligations_role.call(self.host.backend, variables)
        if on_draft is not None:
            on_draft(payload)
        if speculate is not None and self.critic_role is not None:
            # SPECULATIVE DISCHARGE: the critic reviews while the
            # compiler's obligations are already being settled; nodes
            # are content-addressed, so the reviewed decomposition finds
            # its kept obligations settled and only the added ones wait
            from concurrent.futures import ThreadPoolExecutor
            unreviewed = self.materialize_obligations(
                row, payload, record=False) or []
            self.host.log.append(
                "speculative_discharge",
                {"claim": row["id"],
                 "obligations": [o.id for o in unreviewed]})
            with ThreadPoolExecutor(max_workers=1,
                                    thread_name_prefix="gv-critic") as pool:
                review = submit_with_context(
                    pool, self.critique, row, payload, variables)
                speculate(unreviewed)
                payload = review.result()
            if payload is None:
                for o in unreviewed:
                    self.host.store.remove_obligation_edge(row["id"], o.id)
            return payload
        return self.critique(row, payload, variables)

    def compile_variables(self, row: Dict) -> Dict:
        """Build the legacy Compiler inputs without invoking any role.

        The returned values are byte-for-byte the values used by the normal
        Compiler/Critic path.  Isolation setup may hash this one fresh,
        complete pre-execution snapshot with the diagnostic coordinator's
        exact host-slot normalizer; production scheduling and role inputs are
        otherwise unchanged.
        """
        state = os.path.dirname(os.path.abspath(self.host.log.path))
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
            for pid in self.host.store.parents(row["id"]):
                prow = self.host.store.get(pid)
                eff = self.host.store.effective(pid)
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
            state, live_world=self.host.live_world)
        graph = (
            [{"note": "withheld: harness final contract is compiled "
                      "independently from authoritative task context"}]
            if independent_final
            else self.host.store.graph_snapshot(exclude=row["id"])
        )
        referent_context = (
            {"withheld": "harness final contract is independent of the "
                         "solver-derived dependency graph"}
            if independent_final
            else dependency_referent_context(self.host.store, self.host.log, row["id"])
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
            "ledger_path": self.host.log.record_path_for(row["id"]),
            "world_mode": world_mode,
            "world_path": world,
            "world_map": world_map(state, live_world=self.host.live_world),
            "verified_deps": json.dumps(verified_deps, ensure_ascii=False),
            "pending_deps": json.dumps(pending_deps, ensure_ascii=False),
            "dependency_referent_context": json.dumps(
                referent_context, ensure_ascii=False, separators=(",", ":")),
            "graph": json.dumps(graph, ensure_ascii=False),
            "memory_hits": memory_hits,
            "critic_feedback": "",
        }

    def authority_final_contract(self, row: Dict, state: str,
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
            self.host.log.append("contract_cache_bypass", {
                "claim": row["id"], "reason": "independent Critic required"})
            return False, None
        from ...llm.authority_backend import (
            AuthorityBackendUnavailable, authority_compilation_backend,
        )
        from ...llm.schema import validate

        if self.precompiled_final is not None and self.contract_cache is None:
            adopted, payload = self.adopt_precompiled_final(row, state)
            if adopted:
                return True, payload

        try:
            view = build_authority_compilation_view(state)
            with authority_compilation_backend(
                    self.host.backend, self.obligations_role,
                    self.critic_role) as session:
                variables = self.authority_contract_variables(view)
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
                    payload = self.critique(
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
                        self.host.log.append("contract_cache_hit", {
                            "claim": row["id"], "key": key,
                            "authority": view.identity,
                            "source_digest": lease.entry.source_digest,
                            "waited": lease.waited})
                        return True, lease.entry.payload
                    self.host.log.append("contract_cache_miss", {
                        "claim": row["id"], "key": key,
                        "authority": view.identity,
                        "leader": lease.leader, "waited": lease.waited})
                    # A miss is where the concurrent authority-only
                    # compilation pays off: adopt it under the identical
                    # key and publish it for the task's other candidates.
                    adopted = False
                    if self.precompiled_final is not None:
                        adopted, payload = self.adopt_precompiled_final(
                            row, state)
                    if not adopted:
                        with model_scope(stage="contract_compile", round=0):
                            payload = session.obligations_role.call(
                                session.backend, variables)
                        if on_draft is not None:
                            on_draft(payload)
                        payload = self.critique(
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
                    self.host.log.append("contract_cache_published" if published
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
            self.host.log.append("contract_cache_bypass", {
                "claim": row["id"], "reason": str(error)[:300]})
            return False, None

    def adopt_precompiled_final_contract(self, handle) -> None:
        """Register a concurrent authority-only compilation for the final."""
        self.precompiled_final = handle

    def adopt_precompiled_final(self, row: Dict, state: str):
        """Use the concurrent compilation only under an identical key.

        The key covers the authority view, the withheld compile variables,
        the backend and role identities and the critic budget.  A mismatch,
        an authority bypass, or a failed thread all fall back to the
        unchanged inline compile; a failed thread's work is not reused.
        """
        from ...llm.authority_backend import (
            AuthorityBackendUnavailable, authority_compilation_backend,
        )

        handle = self.precompiled_final
        self.precompiled_final = None
        outcome = handle.wait()
        event = {"claim": row["id"],
                 "precompiled_claim": handle.pseudo_claim_id,
                 "elapsed_seconds": round(outcome.elapsed_seconds, 3),
                 "adopted": False}
        if outcome.bypass_reason is not None:
            event["reason"] = outcome.bypass_reason
            self.host.log.append("contract_precompiled", event)
            return False, None
        if outcome.error is not None:
            event["reason"] = (
                f"precompile failed: {type(outcome.error).__name__}")
            self.host.log.append("contract_precompiled", event)
            return False, None
        try:
            view = build_authority_compilation_view(state)
            with authority_compilation_backend(
                    self.host.backend, self.obligations_role,
                    self.critic_role) as session:
                expected = contract_cache_key(
                    view=view,
                    variables=self.authority_contract_variables(view),
                    backend_identity=session.identity,
                    obligations_role=session.obligations_role,
                    critic_role=session.critic_role,
                    critic_rounds=self.critic_rounds,
                    append_only_deltas=self.append_only_deltas,
                    append_only_delta_protocol=self.append_only_delta_protocol)
        except (AuthorityViewUnavailable, AuthorityBackendUnavailable) as error:
            event["reason"] = str(error)[:300]
            self.host.log.append("contract_precompiled", event)
            return False, None
        if expected != outcome.key:
            event["reason"] = "contract key differs from this kernel's"
            self.host.log.append("contract_precompiled", event)
            return False, None
        for kind, payload in outcome.events:
            self.host.log.append(kind, {**payload, "claim": row["id"]})
        for evidence in outcome.evidence:
            self.host.store.record_evidence(row["id"], evidence)
        event["adopted"] = True
        event["critic_rounds_used"] = sum(
            1 for kind, _payload in outcome.events
            if kind == "decomposition_attacked")
        self.host.log.append("contract_precompiled", event)
        return True, outcome.payload

    @staticmethod
    def authority_contract_variables(view):
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

    def recorded_identities(self):
        ids = self.recorded_ids
        if ids is None:
            state = os.path.dirname(os.path.abspath(self.host.log.path))
            ids = recorded_identities(state)
            self.recorded_ids = ids
        return ids

    def restore_quoted_identities(self, row: Dict, payload, record: bool):
        """Near copies of task-start identities in the compiled texts are
        restored to the recorded values (quoted_identities)."""
        identities = self.recorded_identities()
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
            self.host.log.append("obligation_identity_restored",
                            {"claim": row["id"], "repairs": repairs})
        return dict(payload, obligations=items)

    def materialize_obligations(self, row: Dict, payload, *, record=True,
                                 repair=True):
        if payload is None:
            return None
        if repair:
            payload = self.restore_quoted_identities(row, payload, record)
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
                       premise_scope=tuple(self.premises_in_force(row)))
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
            for pid in self.host.store.parents(row["id"]):
                prow = self.host.store.get(pid)
                if prow["kind"] != CLAIM_EFFECT:
                    premise_texts.add(prow["text"])
            residual = [o for o in obligations
                        if o.text not in premise_texts]
            dropped = [o.text for o in obligations
                       if o.text in premise_texts]
            if dropped and residual:
                if record:
                    self.host.log.append(
                        "obligations_residualized",
                        {"claim": row["id"], "dropped": dropped},
                    )
                obligations = residual
        if record:
            self.record_compiled_obligations(
                row, obligations, excluded=payload.get("excluded"))
        return obligations

    def record_compiled_obligations(self, row: Dict,
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
        self.host.log.append("obligations_compiled", event)

    def critique(self, row: Dict, payload: Dict, variables: Dict, *,
                  backend=None, obligations_role=None, critic_role=None,
                  record: bool = True):
        """Review a decomposition in the critic game (engine.contract_critic):
        the reviewed payload, or None when it failed the critic."""
        return CriticGame(
            log=self.host.log, store=self.host.store, backend=self.host.backend,
            obligations_role=self.obligations_role,
            critic_role=self.critic_role, critic_rounds=self.critic_rounds,
            run_options=self.host.run_options,
            critic_rounds_nonfinal=self.critic_rounds_nonfinal,
            unify_final=self.unify_final,
            append_only_deltas=self.append_only_deltas,
            conclude_unverified=self.conclude_unverified,
        ).review(row, payload, variables, backend=backend,
                 obligations_role=obligations_role, critic_role=critic_role,
                 record=record)
