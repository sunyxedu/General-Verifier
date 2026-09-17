"""MCP server: the verifier as a separate process.

The solver stays a vanilla Claude Code session; it mounts this server over
stdio and gets four verbs — commit, verify, certificate, status. Every
call is an event in the server's own append-only log, verification runs
inside the server (through the same kernel as offline replay, zero changes),
and the certificate returns to the solver over MCP; only CI bypasses it via
`python -m verifier.cli`.

`commit` is where verification happens (`--trigger on-commit`, the default):
the call blocks until the claim it just recorded has been checked, and returns
the failing obligation with its counterexample. The solver therefore cannot
write a second claim on top of an unchecked first one.

Hook events (tool_call/tool_result written by the solver's PreToolUse /
PostToolUse hooks) arrive on a separate raw feed file: hooks are concurrent
short-lived processes and cannot share seq discipline, so the server — the
canonical log's single writer — ingests the feed in arrival order and assigns
seq itself. The feed is drained before every MCP verb so effect claims are
referenceable (`effect:<seq>`) at commit time.

Run:
  uv run python -m verifier.mcp_server --state-dir .gv [--model m]
      [--memory-dir d] [--toolchain-dir d]
      [--trigger on-commit|on-first-dependent]

Claude Code config (.mcp.json):
  {"mcpServers": {"gv": {"type": "stdio", "command": "uv",
                          "args": ["run", "python", "-m", "verifier.mcp_server", "--state-dir", ".gv"]}}}
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from verifier.core.model import CLAIM_KINDS
from .stream.builder import Builder, TraceError
from .engine.certificate import request_certificate
from .core.events import EventLog
from .execution.instruments import ArtifactStore
from .engine.kernel import Kernel
from .llm.schema import RoleError
from .core.store import (
    ClaimStore, UnknownDependency, TRIGGER_ON_COMMIT, TRIGGERS,
)
from .stream.tail import Tail

VERB_ERRORS = (TraceError, UnknownDependency, RoleError, ValueError)


def _witness(obligation: dict) -> str:
    """The counterexample recorded on an obligation node, if any."""
    for ev in reversed(obligation.get("evidence") or []):
        if ev.get("witness"):
            return str(ev["witness"])[:300]
    return ""


def _obligation_summary(store, claim_id: str) -> list:
    """Obligation-level detail straight from the graph nodes."""
    out = []
    for o in store.obligation_children(claim_id):
        if o["status"] == "verified":
            continue
        entry = {"text": o["text"], "status": o["status"]}
        reason = o.get("reason")
        if not reason:
            ev = store.evidence_for(o["id"])
            reason = (str(ev[-1].get("witness"))[:300]
                      if ev and ev[-1].get("witness") else None)
        if reason:
            entry["reason"] = reason
        out.append(entry)
    return out


def _residue(store, claim_id: str) -> Dict[str, Any]:
    """What a non-verified claim owes the solver: the obligations that did not
    hold with their counterexamples — or, when no obligation was ever compiled,
    the fail-closed reason (a refuted premise, say), which is otherwise
    invisible to the solver."""
    obligations = _obligation_summary(store, claim_id)
    if obligations:
        return {"obligations": obligations}
    for ev in reversed(store.evidence_for(claim_id)):
        if ev.get("error"):
            out = {"error": ev["error"]}
            if ev.get("refuted_deps"):
                out["refuted_deps"] = [d[:12] for d in ev["refuted_deps"]]
            return out
    return {"obligations": []}


HOOK_FEED = "hook_feed.jsonl"
HOOK_EVENT_TYPES = ("tool_call", "tool_result")


class Service:
    def __init__(self, state_dir: str, kernel_factory=None,
                 model: Optional[str] = None,
                 memory_dir: Optional[str] = None,
                 toolchain_dir: Optional[str] = None,
                 trigger: str = TRIGGER_ON_COMMIT,
                 live_world: bool = True,
                 critic_rounds: Optional[int] = None,
                 parallel_obligations: Optional[int] = None,
                 parallel_claims: Optional[int] = None,
                 shared_cache_dir: Optional[str] = None):
        self.toolchain_dir = toolchain_dir
        # a verifier-owned cache root shared across trials (install caches
        # for the executor's instruments); None keeps every cache
        # state-local as before
        self.shared_cache_dir = shared_cache_dir
        self.parallel_obligations = parallel_obligations
        self.parallel_claims = parallel_claims
        self.live_world = live_world
        os.makedirs(state_dir, exist_ok=True)
        self.state_dir = state_dir
        self.memory = None
        if memory_dir:
            from .memory import Memory
            self.memory = Memory(memory_dir)
        self.log = EventLog(os.path.join(state_dir, "events.jsonl"))
        self.store = ClaimStore(os.path.join(state_dir, "claims.sqlite"),
                                trigger=trigger)
        self.artifacts = ArtifactStore(os.path.join(state_dir, "artifacts"))
        self.builder = Builder(self.log, self.store)

        if kernel_factory is None:
            kernel_factory = lambda sc: _default_kernel(  # noqa: E731
                sc, model=model, critic_rounds=critic_rounds,
                parallel_obligations=parallel_obligations,
                parallel_claims=parallel_claims)
        self.kernel = kernel_factory(self)  # factory receives the service

        self._feed = Tail(os.path.join(state_dir, HOOK_FEED),
                          self._ingest_hook_event, raw=True)
        # a restarted server rebuilds its view from its own ledger
        for event in self.log.replay():
            self.builder.feed(event)
        # online: compile the final's task contract and its method
        # batches while the solver works (GV_FINAL_PRECOMPILE=0 disables)
        if live_world and os.environ.get("GV_FINAL_PRECOMPILE", "1") != "0":
            start = getattr(self.kernel, "start_final_precompile", None)
            if start is not None:
                start()

    # hook feed
    def _ingest_hook_event(self, raw_event: Dict[str, Any]) -> None:
        etype = raw_event.get("type")
        if etype not in HOOK_EVENT_TYPES:
            return  # unknown feed line: ignored, the feed is untrusted input
        event = self.log.append(etype, raw_event.get("payload", {}))
        self.builder.feed(event)

    def drain_hooks(self) -> int:
        return self._feed.poll()

    # MCP verbs
    def commit(self, label: str, kind: str, text: str,
               deps: List[str],
               revises: Optional[str] = None) -> Dict[str, Any]:
        self.drain_hooks()
        payload = {"label": label, "kind": kind, "text": text, "deps": deps}
        if revises:
            payload["revises"] = revises
        failure = None
        try:
            event = self.log.append("commit", payload)
            # under the on-commit rule this is where verification happens: the
            # call returns only once the claim has been checked
            self.builder.feed(event)
        except VERB_ERRORS as err:
            failure = str(err)
        if label not in self.builder.labels:
            return {"error": failure or "commit failed"}
        claim_id = self.builder.labels[label]
        row = self.store.get(claim_id)
        status = row["status"]
        result = {"claim_id": claim_id, "status": status}
        if failure is not None:
            # the claim is recorded; its check did not finish — fail-closed, so
            # the status stands as unverified and the solver is told why
            result["error"] = failure
        if row.get("confident"):
            # admitted on the record: assumed by every consumer, no check
            # run, nothing owed — the solver may build on it (and may ask
            # for the full check with `verify` if it wants one)
            result["confident"] = True
            result["confidence_reason"] = row.get("confidence_reason")
            return result
        if kind != "effect" and status != "verified":
            # the verdict lands on the commit call now, so its residue must too
            result.update(_residue(self.store, claim_id))
        return result

    def verify(self, label: str) -> Dict[str, Any]:
        self.drain_hooks()
        try:
            claim_id = self.builder.resolve(label)
            self.log.append("verify_request", {"target": claim_id, "cause": "mcp"})
            # an explicit request is never answered by the screener: a
            # confident claim gets its full check here
            status = self.kernel.verify(claim_id, force=True)
        except VERB_ERRORS as err:
            return {"error": str(err)}
        result = {"claim_id": claim_id, "status": status}
        if status != "verified":
            # every non-verified status owes the solver its residue: which
            # obligation failed and the counterexample witness. Withholding
            # it on `refuted` left the solver with nothing to act on but the
            # claim's wording — the mechanism behind claim-shopping.
            result.update(_residue(self.store, claim_id))
        return result

    def certificate(self, final_label: str) -> Dict[str, Any]:
        self.drain_hooks()
        try:
            final_id = self.builder.resolve(final_label)
            cert = request_certificate(self.store, self.kernel, self.log,
                                       self.artifacts, final_id)
        except VERB_ERRORS as err:
            return {"error": str(err)}
        slice_summary = []
        for e in cert["slice"]:
            entry = {"text": e["text"], "status": e["status"]}
            if e.get("confident"):
                entry["confident"] = True
                slice_summary.append(entry)
                continue
            if e["status"] != "verified":
                obligations = [
                    {"text": o["text"], "status": o["status"],
                     **({"reason": o["reason"] or _witness(o)}
                        if (o["reason"] or _witness(o)) else {})}
                    for o in e.get("obligations", [])
                    if o["status"] != "verified"
                ]
                if obligations:
                    entry["obligations"] = obligations
                else:
                    for ev in reversed(self.store.evidence_for(e["id"])):
                        if ev.get("error"):
                            entry["error"] = ev["error"]
                            break
            slice_summary.append(entry)
        return {
            "verdict": cert["verdict"],
            "certificate_id": cert["certificate_id"],
            "slice": slice_summary,
        }

    def status(self, label: Optional[str] = None) -> Dict[str, Any]:
        self.drain_hooks()
        if label is not None:
            try:
                claim_id = self.builder.resolve(label)
            except TraceError as err:
                return {"error": str(err)}
            row = self.store.get(claim_id)
            result = {k: row[k] for k in ("id", "kind", "text", "status")}
            if row.get("confident"):
                result["confident"] = True
                result["confidence_reason"] = row.get("confidence_reason")
            elif row["status"] != "verified":
                result.update(_residue(self.store, claim_id))
            return result
        out = {}
        for label, cid in self.builder.labels.items():
            row = self.store.get(cid)
            out[label] = (f"{row['status']} (confident)"
                          if row.get("confident") else row["status"])
        return out

def _parallel_default(name: str, fallback: int) -> int:
    """Per-trial parallelism defaults: GV_PARALLEL_OBLIGATIONS (discharges
    concurrent within one claim) and GV_PARALLEL_CLAIMS (claims of a
    certificate slice verified concurrently; offline only)."""
    try:
        return max(1, int(os.environ.get(name, fallback)))
    except ValueError:
        return fallback


def harness_roles(state_dir: str) -> Dict[str, Any]:
    """The role registry a state's kernel seats: the mounting harness's
    ground truth (task.md / baseline/) is the toggle; when present
    (experiment mounts write it; the interactive mount does not), the
    obligation compiler is told to compile against it."""
    from dataclasses import replace

    from .llm import prompts
    from .llm.agents import ALL_AGENT_ROLES, COMPILE_OBLIGATIONS_AGENT

    obligations = COMPILE_OBLIGATIONS_AGENT
    if os.path.exists(os.path.join(state_dir, "task.md")) or \
            os.path.isdir(os.path.join(state_dir, "baseline")):
        obligations = replace(
            obligations, system=obligations.system + "\n"
            + prompts.OBLIGATIONS_HARNESS_ADDENDUM)
    return dict(ALL_AGENT_ROLES, obligations=obligations)


def _default_kernel(sc: Service, model, backend=None,
                    critic_rounds: Optional[int] = None,
                    parallel_obligations: Optional[int] = None,
                    parallel_claims: Optional[int] = None) -> Kernel:
    from .llm.agents import AgentBackend, SCREENER_AGENT
    from .execution.config import (
        contract_cache_enabled,
        method_cache_enabled,
        sufficiency_preaudit_enabled,
        typed_adapters_enabled,
        typed_judger_packet_enabled,
    )
    from .execution.discharge import AgentDischarger
    from .engine.game import BindingGame

    # The backend resolves roles by name, so the harness-swapped role must
    # go into its registry too, not just into the Kernel.
    roles = harness_roles(sc.state_dir)
    obligations = roles["obligations"]
    if backend is None:
        backend = AgentBackend(roles=roles, cwd=sc.state_dir, model=model)
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
    # Process-local, explicit opt-in reuse across the candidates of one task
    # scored in this process.  Off by default: every candidate then follows
    # the fresh Compiler/Critic/Oracle/executor/Judger sequence.
    method_cache = None
    if method_cache_enabled():
        from .engine.method_cache import default_accepted_method_cache
        method_cache = default_accepted_method_cache()
    contract_cache = None
    if contract_cache_enabled():
        from .engine.contract_cache import (
            default_compiled_final_contract_cache,
        )
        contract_cache = default_compiled_final_contract_cache()
    typed_adapters = typed_adapters_enabled()
    typed_judger_packet = typed_judger_packet_enabled()
    if typed_judger_packet and not typed_adapters:
        raise ValueError("typed-adapter Judger packet requires typed adapters")
    if typed_adapters:
        from .execution.adapters import (
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
                         typed_adapter_judger_packet=typed_judger_packet,
                         sufficiency_role=(
                             roles["sufficiency"]
                             if sufficiency_preaudit_enabled() else None),
                         packet_method_reuse=method_cache is not None)
    if parallel_obligations is None:
        parallel_obligations = getattr(sc, "parallel_obligations", None)
    if parallel_claims is None:
        parallel_claims = getattr(sc, "parallel_claims", None)
    if parallel_obligations is None:
        parallel_obligations = _parallel_default("GV_PARALLEL_OBLIGATIONS", 8)
    if parallel_claims is None:
        # online (live world): one claim at a time as commits arrive
        parallel_claims = (1 if sc.live_world
                           else _parallel_default("GV_PARALLEL_CLAIMS", 4))
    # the admission screener is an ONLINE economy: with a live world the
    # solver waits on every commit, and a claim the record already settles
    # is assumed rather than checked. Offline scoring seats none — every
    # claim in a scored slice gets its check. GV_SCREENER=0 disables it.
    screener = (SCREENER_AGENT
                if sc.live_world and os.environ.get("GV_SCREENER", "1") != "0"
                else None)
    return Kernel(sc.store, sc.log, backend, sc.artifacts,
                  obligations_role=obligations,
                  discharger=discharger, binder=binder,
                  memory=sc.memory, live_world=sc.live_world,
                  record_boundary=sc.builder.commit_seq.get,
                  critic_rounds=critic_rounds,
                  contract_cache=contract_cache,
                  parallel_obligations=parallel_obligations,
                  parallel_claims=parallel_claims,
                  screener_role=screener)


# MCP wiring

_PROTOCOL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "protocol", "solver-protocol.md")


def _solver_protocol() -> str:
    """The solver-side protocol, served as MCP server instructions: every
    client that mounts gv gets it injected into the solver's context, so
    there is no CLAUDE.md copy to keep in sync. Advisory only — soundness
    does not depend on the solver following it; ignoring it forfeits the
    certificate, never corrupts one."""
    with open(_PROTOCOL_PATH, encoding="utf-8") as f:
        return f.read()


def build_server(service: Service):
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("gv", instructions=_solver_protocol())

    @server.tool()
    def commit(label: str, kind: CLAIM_KINDS, text: str, deps: List[str],
               revises: str = "") -> str:
        """Commit a claim (kind: assertion|effect|final). deps is a list of
        labels of already-committed claims (including effect:<seq>). The claim
        is VERIFIED BEFORE THIS RETURNS, so the call blocks for as long as its
        checks take; the result carries the resulting status, and on anything
        but `verified` the obligation that failed plus its counterexample."""
        return json.dumps(
            service.commit(label, kind, text, deps,
                           revises or None),
            ensure_ascii=False)

    @server.tool()
    def verify(label: str) -> str:
        """Run the full independent check on a committed claim now (also
        for a claim that commit admitted as `confident`); returns its
        status."""
        return json.dumps(service.verify(label), ensure_ascii=False)

    @server.tool()
    def certificate(final_label: str) -> str:
        """Certify a final claim: settle the dependency closure, freeze the
        certificate, and return the verdict plus a slice summary."""
        return json.dumps(service.certificate(final_label), ensure_ascii=False)

    @server.tool()
    def status(label: str = "") -> str:
        """Query the status of one label or of all labels."""
        return json.dumps(service.status(label or None), ensure_ascii=False)

    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--memory-dir", default=None)
    ap.add_argument("--toolchain-dir", default=None)
    ap.add_argument("--trigger", default=TRIGGER_ON_COMMIT, choices=list(TRIGGERS),
                    help="when a claim is verified (default: at commit)")
    args = ap.parse_args()
    service = Service(args.state_dir, model=args.model,
                      memory_dir=args.memory_dir,
                      toolchain_dir=args.toolchain_dir,
                      trigger=args.trigger)
    build_server(service).run()   # stdio


if __name__ == "__main__":
    main()
