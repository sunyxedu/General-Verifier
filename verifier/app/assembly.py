"""The composition root: the one place a verifier is assembled from its
parts. The MCP server and offline scoring both build their kernel here —
seat backend, the discharger chain (agent executor, typed-adapter router),
the binding game, the process-local caches, and the kernel with its online
economies — so the two entry points cannot drift apart.

Settings are read from ``verifier.config`` at the moment each piece is built,
as the entry points always did: a harness (or a test) that sets the
environment before building a kernel gets what it set.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Protocol

from ..engine.kernel import Kernel
from ..config import ExtensionConfig, VerifierConfig


class KernelHost(Protocol):
    """What the composition root needs from the service it builds a
    kernel for (the MCP server's Service, or a test's stand-in):
    ``state_dir``, ``store``, ``log``, ``artifacts``, ``memory``,
    ``live_world``, ``toolchain_dir``, and ``builder.commit_seq``.
    ``shared_cache_dir``, ``parallel_obligations`` and ``parallel_claims``
    are optional and read with a default."""
    state_dir: str
    live_world: bool


def harness_roles(state_dir: str) -> Dict[str, Any]:
    """The role registry a state's kernel seats: the mounting harness's
    ground truth (task.md / baseline/) is the toggle; when present
    (experiment mounts write it; the interactive mount does not), the
    obligation compiler is told to compile against it."""
    from dataclasses import replace

    from ..llm import prompts
    from ..llm.agents import ALL_AGENT_ROLES, COMPILE_OBLIGATIONS_AGENT

    obligations = COMPILE_OBLIGATIONS_AGENT
    if os.path.exists(os.path.join(state_dir, "task.md")) or \
            os.path.isdir(os.path.join(state_dir, "baseline")):
        obligations = replace(
            obligations, system=obligations.system + "\n"
            + prompts.OBLIGATIONS_HARNESS_ADDENDUM)
    return dict(ALL_AGENT_ROLES, obligations=obligations)


def final_contract_cache(extensions: ExtensionConfig):
    """The process-wide compiled final-contract cache when the experiment
    is on, else None."""
    if not extensions.contract_cache:
        return None
    from ..engine.contract_cache import default_compiled_final_contract_cache
    return default_compiled_final_contract_cache()


def build_kernel(sc: KernelHost, model, backend=None,
                 critic_rounds: Optional[int] = None,
                 parallel_obligations: Optional[int] = None,
                 parallel_claims: Optional[int] = None,
                 unify_final: Optional[bool] = None) -> Kernel:
    """The default kernel for a service: ``backend`` defaults to the agent
    backend on the configured provider (tool-free roles seated on direct
    model calls when configured); an injected backend (Codex seats, a
    fixture) is given the same harness role registry."""
    from ..llm.agents import SCREENER_AGENT
    from ..execution.discharge import AgentDischarger
    from ..engine.game import BindingGame

    # The backend resolves roles by name, so the harness-swapped role must
    # go into its registry too, not just into the Kernel.
    roles = harness_roles(sc.state_dir)
    obligations = roles["obligations"]
    if backend is None:
        # the agent backend, with the tool-free roles seated on a direct
        # model call when the configuration carries an API key
        from ..llm.direct_backend import build_backend
        backend = build_backend(roles, cwd=sc.state_dir, model=model)
    else:
        # an injected backend (e.g. CodexBackend) still needs the
        # harness-swapped role registry — same double-registration rule
        backend.roles = roles
    validate_tool_scope = getattr(
        backend, "validate_offline_tool_scope", None)
    if not sc.live_world and validate_tool_scope is not None:
        validate_tool_scope(
            sc.state_dir, sc.toolchain_dir,
            getattr(sc, "shared_cache_dir", None))
    agent_discharger = AgentDischarger(
        backend, sc.artifacts,
        workdir_base=os.path.join(sc.state_dir, "work"),
        cleanup=True,   # snapshots are the evidence; workdirs are scratch
        toolchain_dir=sc.toolchain_dir,
        shared_cache_dir=getattr(sc, "shared_cache_dir", None),
        state_dir=sc.state_dir,
        live_world=sc.live_world,
        store=sc.store,   # evidence-workspace index over the claim graph
        log=sc.log)
    config = VerifierConfig.from_env()
    extensions = config.extensions
    # Process-local, explicit opt-in reuse across the candidates of one task
    # scored in this process.  Off by default: every candidate then follows
    # the fresh Compiler/Critic/Oracle/executor/Judger sequence.
    method_cache = None
    if extensions.method_cache:
        from ..engine.method_cache import default_accepted_method_cache
        method_cache = default_accepted_method_cache()
    contract_cache = final_contract_cache(extensions)
    extensions.require_consistent()
    if extensions.typed_adapters:
        from ..execution.adapters import (
            AdapterRoutingDischarger, default_adapter_registry,
        )

        discharger = AdapterRoutingDischarger(
            agent_discharger, default_adapter_registry(), sc.artifacts, sc.log,
            sc.state_dir, live_world=sc.live_world)
    else:
        discharger = agent_discharger
    binder = BindingGame(backend, sc.artifacts, sc.log, discharger,
                         memory=sc.memory, live_world=sc.live_world,
                         method_cache=method_cache,
                         typed_adapter_judger_packet=(
                             extensions.typed_judger_packet),
                         sufficiency_role=(
                             roles["sufficiency"]
                             if extensions.sufficiency_preaudit else None),
                         packet_method_reuse=method_cache is not None)
    if parallel_obligations is None:
        parallel_obligations = getattr(sc, "parallel_obligations", None)
    if parallel_claims is None:
        parallel_claims = getattr(sc, "parallel_claims", None)
    if parallel_obligations is None:
        # a compound claim can compile to dozens of obligations; with the
        # executable-form path most of them are a short script plus one
        # no-tools reading, so eight in flight is well inside the seat gate
        parallel_obligations = config.parallelism.parallel_obligations
    if parallel_claims is None:
        # online (live world): one claim at a time as commits arrive
        parallel_claims = (1 if sc.live_world
                           else config.parallelism.parallel_claims)
    online = config.online
    # the admission screener is an ONLINE economy: with a live world the
    # solver waits on every commit, and a claim the record already settles
    # is assumed rather than checked. Offline scoring seats none — every
    # claim in a scored slice gets its check. GV_SCREENER=0 disables it.
    screener = (SCREENER_AGENT if sc.live_world and online.screener
                else None)
    # online, the task contract lives at the final, which keeps the full
    # adversarial game; a non-final decomposition is the compiler's own by
    # default (GV_CRITIC_ROUNDS_NONFINAL=0). A positive budget seats an
    # ADVISORY review: that many rounds, the last repair standing. Offline
    # scoring keeps the full game everywhere. (An advisory repair was
    # observed to replace a concrete obligation with a vaguer, costlier
    # one; the solver's intermediate feedback is better served by the
    # compiler's decomposition, gated by the same evidence axioms.)
    nonfinal_rounds = online.critic_rounds_nonfinal if sc.live_world else None
    return Kernel(sc.store, sc.log, backend, sc.artifacts,
                  obligations_role=obligations,
                  discharger=discharger, binder=binder,
                  memory=sc.memory, live_world=sc.live_world,
                  record_boundary=sc.builder.commit_seq.get,
                  critic_rounds=critic_rounds,
                  contract_cache=contract_cache,
                  parallel_obligations=parallel_obligations,
                  parallel_claims=parallel_claims,
                  screener_role=screener,
                  critic_rounds_nonfinal=nonfinal_rounds,
                  # online, the final takes the path of any claim (screened,
                  # non-final critic budget) with its contract still
                  # compiled from the task statement; GV_UNIFY_FINAL=0
                  # restores the separate final path
                  unify_final=(
                      unify_final if unify_final is not None else (
                          sc.live_world and online.unify_final)),
                  # online: overlap the critic's review with the checks
                  # (GV_SPECULATIVE_DISCHARGE=0 disables)
                  speculative_discharge=(
                      sc.live_world and online.speculative_discharge))


def build_online_codex_backend(state_dir: str, model: Optional[str],
                               effort: str, service_tier: str,
                               seat_search: bool):
    """The MCP server's Codex seats: the Codex CLI for tool-using roles,
    with the tool-free roles on a direct model call when configured."""
    from ..llm.codex_backend import CodexBackend
    from ..llm.direct_backend import build_backend
    codex = CodexBackend(cwd=state_dir, model=model, effort=effort,
                         service_tier=service_tier, seat_search=seat_search)
    return build_backend(harness_roles(state_dir), cwd=state_dir, model=model,
                         agent=codex)


def build_offline_codex_backend(state_dir: str, model: Optional[str],
                                fields: Dict[str, Any],
                                toolchain_dir: Optional[str],
                                cache_dir: Optional[str],
                                restricted_judger_cwd: Optional[str]):
    """An offline scoring run's Codex seats, bound to the run's resolved
    config fields (effort, service tier, evidence presentation)."""
    from ..llm.codex_backend import CodexBackend
    return CodexBackend(cwd=state_dir, model=model,
                        effort=fields["effort"],
                        service_tier=fields["service_tier"],
                        compact_inspection=fields["compact_inspection"],
                        judger_prefetch=fields["judger_prefetch"],
                        offline_toolchain_dir=toolchain_dir,
                        offline_shared_cache_dir=cache_dir,
                        restricted_judger_cwd=restricted_judger_cwd)


def start_offline_final_precompile(state_dir: str, kernel_backend,
                                   critic_rounds: int):
    """Start compiling the candidate-blind final contract on the kernel's
    own backend, side by side with extraction; the kernel adopts it only
    under an identical contract key. With the process-local contract cache
    the thread compiles only for the task's first candidate (a key nobody
    has published or started); later candidates' kernels hit the published
    entry."""
    from ..engine.precompile import PrecompiledFinalContract
    from ..engine.kernel import compile_roles_for_backend
    from ..config import APPEND_ONLY_DELTA_PROTOCOL
    roles = harness_roles(state_dir)
    compile_role, critic_role = compile_roles_for_backend(
        kernel_backend, roles["obligations"], roles["critic"])
    extensions = ExtensionConfig.from_env()
    deltas = extensions.append_only_deltas
    return PrecompiledFinalContract.start(
        state_dir, kernel_backend, compile_role, critic_role,
        critic_rounds, deltas,
        APPEND_ONLY_DELTA_PROTOCOL if deltas else None,
        contract_cache=final_contract_cache(extensions))
