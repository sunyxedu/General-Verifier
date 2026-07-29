"""Sidecar MCP service (report §11): the verifier as a separate process.

The solver stays a vanilla Claude Code session; it mounts this server over
stdio and gets three verbs — commit, verify, certificate — plus status. Every
call is an event in the sidecar's own append-only log, verification runs
inside the sidecar (through the same kernel as offline replay, zero changes),
and the certificate returns to the solver over MCP; only CI bypasses it via
`python -m verifier.cli`.

Hook events (tool_call/tool_result written by the solver's PreToolUse /
PostToolUse hooks) arrive on a separate raw feed file: hooks are concurrent
short-lived processes and cannot share seq discipline, so the sidecar — the
canonical log's single writer — ingests the feed in arrival order and assigns
seq itself. The feed is drained before every MCP verb so effect claims are
referenceable (`effect:<seq>`) at commit time.

Run:
  python -m verifier.sidecar --state-dir .gv [--model m] [--seats agent|sandbox]
      [--game/--no-game] [--coverage/--no-coverage]

Claude Code config (.mcp.json):
  {"mcpServers": {"gv": {"type": "stdio", "command": ".venv/bin/python",
                          "args": ["-m", "verifier.sidecar", "--state-dir", ".gv"]}}}
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from .builder import Builder, TraceError
from .certificate import request_certificate
from .events import EventLog
from .instruments import ArtifactStore
from .kernel import Kernel
from .store import ClaimStore, UnknownDependency
from .tail import Tail

HOOK_FEED = "hook_feed.jsonl"
HOOK_EVENT_TYPES = ("tool_call", "tool_result")


class Sidecar:
    def __init__(self, state_dir: str, kernel_factory=None,
                 model: Optional[str] = None, seats: str = "agent",
                 game: bool = True, coverage: bool = False):
        os.makedirs(state_dir, exist_ok=True)
        self.state_dir = state_dir
        self.log = EventLog(os.path.join(state_dir, "events.jsonl"))
        self.store = ClaimStore(os.path.join(state_dir, "claims.sqlite"))
        self.artifacts = ArtifactStore(os.path.join(state_dir, "artifacts"))
        self.builder = Builder(self.log, self.store)

        if kernel_factory is None:
            kernel_factory = lambda sc: _default_kernel(  # noqa: E731
                sc, model=model, seats=seats, game=game, coverage=coverage
            )
        self.kernel = kernel_factory(self)  # factory receives the sidecar

        self._feed = Tail(os.path.join(state_dir, HOOK_FEED),
                          self._ingest_hook_event, raw=True)
        # a restarted sidecar rebuilds its view from its own ledger
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
               deps: List[str], branch: Optional[str] = None,
               revises: Optional[str] = None) -> Dict[str, Any]:
        self.drain_hooks()
        payload = {"label": label, "kind": kind, "text": text, "deps": deps}
        if branch:
            payload["branch"] = branch
        if revises:
            payload["revises"] = revises
        try:
            event = self.log.append("commit", payload)
            self.builder.feed(event)   # trigger rule may verify deps inline
        except (TraceError, UnknownDependency, ValueError) as err:
            return {"error": str(err)}
        claim_id = self.builder.labels[label]
        return {"claim_id": claim_id, "status": self.store.status(claim_id)}

    def verify(self, label: str) -> Dict[str, Any]:
        self.drain_hooks()
        try:
            claim_id = self.builder.resolve(label)
        except TraceError as err:
            return {"error": str(err)}
        self.log.append("verify_request", {"target": claim_id, "cause": "mcp"})
        status = self.kernel.verify(claim_id)
        return {"claim_id": claim_id, "status": status,
                "grade": self.store.grade(claim_id)}

    def certificate(self, final_label: str) -> Dict[str, Any]:
        self.drain_hooks()
        try:
            final_id = self.builder.resolve(final_label)
        except TraceError as err:
            return {"error": str(err)}
        cert = request_certificate(self.store, self.kernel, self.log,
                                   self.artifacts, final_id)
        return {
            "verdict": cert["verdict"],
            "overall_grade": cert["overall_grade"],
            "certificate_id": cert["certificate_id"],
            "slice": [
                {"text": e["text"], "status": e["status"], "grade": e["grade"]}
                for e in cert["slice"]
            ],
        }

    def status(self, label: Optional[str] = None) -> Dict[str, Any]:
        self.drain_hooks()
        if label is not None:
            try:
                claim_id = self.builder.resolve(label)
            except TraceError as err:
                return {"error": str(err)}
            row = self.store.get(claim_id)
            return {k: row[k] for k in ("id", "kind", "text", "status", "grade")}
        return {
            label: self.store.status(cid)
            for label, cid in self.builder.labels.items()
        }


def _default_kernel(sc: Sidecar, model, seats, game, coverage) -> Kernel:
    if seats == "agent":
        from .agents import (
            ALL_AGENT_ROLES, AgentBackend, COMPILE_OBLIGATIONS_AGENT,
        )
        from .coverage import CoverageAudit
        from .discharge import AgentDischarger
        from .game import BindingGame

        backend = AgentBackend(roles=ALL_AGENT_ROLES, cwd=sc.state_dir,
                               model=model)
        discharger = AgentDischarger(
            backend, sc.artifacts,
            workdir_base=os.path.join(sc.state_dir, "work"))
        binder = BindingGame(backend, sc.artifacts, sc.log,
                             discharger) if game else None
        audit = CoverageAudit(backend, sc.log) if coverage else None
        return Kernel(sc.store, sc.log, backend, sc.artifacts,
                      obligations_role=COMPILE_OBLIGATIONS_AGENT,
                      discharger=discharger, binder=binder, coverage=audit)
    if seats == "sandbox":
        from .roles import OpenAIBackend

        backend = OpenAIBackend(model=model or "gpt-4o")
        return Kernel(sc.store, sc.log, backend, sc.artifacts)
    raise ValueError(f"unknown seats: {seats}")


# ---- MCP wiring --------------------------------------------------------------

def build_server(sidecar: Sidecar):
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("gv")

    @server.tool()
    def commit(label: str, kind: str, text: str, deps: List[str],
               branch: str = "", revises: str = "") -> str:
        """提交一个 claim（kind: assertion|effect|final）。deps 是已提交 claim 的
        label 列表（含 effect:<seq>）。返回 claim_id 与当前 status。"""
        return json.dumps(
            sidecar.commit(label, kind, text, deps,
                           branch or None, revises or None),
            ensure_ascii=False)

    @server.tool()
    def verify(label: str) -> str:
        """立即验证一个已提交的 claim，返回 status 与 grade。"""
        return json.dumps(sidecar.verify(label), ensure_ascii=False)

    @server.tool()
    def certificate(final_label: str) -> str:
        """对 final claim 出证：结算依赖闭包、冻结证书，返回 verdict 与切片摘要。"""
        return json.dumps(sidecar.certificate(final_label), ensure_ascii=False)

    @server.tool()
    def status(label: str = "") -> str:
        """查询单个 label 或全部 label 的状态。"""
        return json.dumps(sidecar.status(label or None), ensure_ascii=False)

    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seats", default="agent", choices=["agent", "sandbox"])
    ap.add_argument("--game", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--coverage", action=argparse.BooleanOptionalAction,
                    default=False)
    args = ap.parse_args()
    sidecar = Sidecar(args.state_dir, model=args.model, seats=args.seats,
                      game=args.game, coverage=args.coverage)
    build_server(sidecar).run()   # stdio


if __name__ == "__main__":
    main()
