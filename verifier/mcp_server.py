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
  uv run python -m verifier.mcp_server --state-dir .gv [--model m] [--seats agent|sandbox]
      [--game/--no-game] [--trigger on-commit|on-first-dependent]

Claude Code config (.mcp.json):
  {"mcpServers": {"gv": {"type": "stdio", "command": "uv",
                          "args": ["run", "python", "-m", "verifier.mcp_server", "--state-dir", ".gv"]}}}
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from .stream.builder import Builder, TraceError
from .engine.certificate import request_certificate
from .core.events import EventLog
from .execution.instruments import ArtifactStore
from .engine.kernel import Kernel
from .llm.roles import RoleError
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
        entry = {"text": o["text"], "status": o["status"]}
        if o["status"] != "verified":
            reason = o.get("reason")
            if not reason:
                ev = store.evidence_for(o["id"])
                reason = str(ev[-1].get("witness"))[:300] if ev and ev[-1].get("witness") else None
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
                 model: Optional[str] = None, seats: str = "agent",
                 game: bool = True,
                 memory_dir: Optional[str] = None,
                 toolchain_dir: Optional[str] = None,
                 trigger: str = TRIGGER_ON_COMMIT):
        self.toolchain_dir = toolchain_dir
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
                sc, model=model, seats=seats, game=game
            )
        self.kernel = kernel_factory(self)  # factory receives the service

        self._feed = Tail(os.path.join(state_dir, HOOK_FEED),
                          self._ingest_hook_event, raw=True)
        # a restarted server rebuilds its view from its own ledger
        for event in self.log.replay():
            self.builder.feed(event)

    # ---- hook feed -----------------------------------------------------------

    def _ingest_hook_event(self, raw_event: Dict[str, Any]) -> None:
        etype = raw_event.get("type")
        if etype not in HOOK_EVENT_TYPES:
            return  # unknown feed line: ignored, the feed is untrusted input
        event = self.log.append(etype, raw_event.get("payload", {}))
        self.builder.feed(event)

    def drain_hooks(self) -> int:
        return self._feed.poll()

    # ---- MCP verbs -----------------------------------------------------------

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
        status = self.store.status(claim_id)
        result = {"claim_id": claim_id, "status": status}
        if failure is not None:
            # the claim is recorded; its check did not finish — fail-closed, so
            # the status stands as unverified and the solver is told why
            result["error"] = failure
        if kind != "effect" and status != "verified":
            # the verdict lands on the commit call now, so its residue must too
            result.update(_residue(self.store, claim_id))
        return result

    def verify(self, label: str) -> Dict[str, Any]:
        self.drain_hooks()
        try:
            claim_id = self.builder.resolve(label)
            self.log.append("verify_request", {"target": claim_id, "cause": "mcp"})
            status = self.kernel.verify(claim_id)
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
            if row["status"] != "verified":
                result.update(_residue(self.store, claim_id))
            return result
        return {
            label: self.store.status(cid)
            for label, cid in self.builder.labels.items()
        }

def _default_kernel(sc: Service, model, seats, game) -> Kernel:
    if seats == "agent":
        from .llm.agents import (
            ALL_AGENT_ROLES, AgentBackend, COMPILE_OBLIGATIONS_AGENT,
        )
        from .execution.discharge import AgentDischarger
        from .engine.game import BindingGame

        backend = AgentBackend(roles=ALL_AGENT_ROLES, cwd=sc.state_dir,
                               model=model)
        discharger = AgentDischarger(
            backend, sc.artifacts,
            workdir_base=os.path.join(sc.state_dir, "work"),
            cleanup=True,   # snapshots are the evidence; workdirs are scratch
            toolchain_dir=sc.toolchain_dir)
        binder = BindingGame(backend, sc.artifacts, sc.log, discharger,
                             memory=sc.memory) if game else None
        return Kernel(sc.store, sc.log, backend, sc.artifacts,
                      obligations_role=COMPILE_OBLIGATIONS_AGENT,
                      discharger=discharger, binder=binder,
                      memory=sc.memory)
    if seats == "sandbox":
        from .llm import provider
        from .llm.roles import OpenAIBackend

        backend = OpenAIBackend(model=model or provider.MODEL,
                                base_url=provider.OPENAI_BASE_URL,
                                api_key=provider.API_KEY)
        return Kernel(sc.store, sc.log, backend, sc.artifacts)
    raise ValueError(f"unknown seats: {seats}")


# ---- MCP wiring --------------------------------------------------------------

def build_server(service: Service):
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("gv")

    @server.tool()
    def commit(label: str, kind: str, text: str, deps: List[str],
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
        """Verify a committed claim now; returns its status."""
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
    ap.add_argument("--seats", default="agent", choices=["agent", "sandbox"])
    ap.add_argument("--game", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--memory-dir", default=None)
    ap.add_argument("--toolchain-dir", default=None)
    ap.add_argument("--trigger", default=TRIGGER_ON_COMMIT, choices=list(TRIGGERS),
                    help="when a claim is verified (default: at commit)")
    args = ap.parse_args()
    service = Service(args.state_dir, model=args.model, seats=args.seats,
                      game=args.game,
                      memory_dir=args.memory_dir,
                      toolchain_dir=args.toolchain_dir,
                      trigger=args.trigger)
    build_server(service).run()   # stdio


if __name__ == "__main__":
    main()
