"""Mechanical parser + file replay.

Deterministic, zero-LLM: tool events become effect claims whose text is the
canonical ledger reference; commit events become assertion/final claims. Trace
authors refer to claims by human labels; the builder resolves labels to
content-addressed ids ("effect:<seq>" refers to a tool_result's effect claim).
"""
from __future__ import annotations

from typing import Dict, Optional

from ..core.events import EventLog
from ..core.model import Claim, canonical, CLAIM_EFFECT
from ..core.store import ClaimStore


class TraceError(ValueError):
    pass


def effect_text(event: dict, tool: Optional[str]) -> str:
    return canonical(
        {
            "event_seq": event["seq"],
            "tool": tool,
            "digest": EventLog.event_digest(event),
        }
    )


class Builder:
    """Incremental: feed(event) is the one entry point; replay() is just
    feeding the whole file. Online mode is a
    Tail pushing the same events into the same feed — the kernel, wired to the
    store's trigger, does not know or care which mode it is in."""

    def __init__(self, log: EventLog, store: ClaimStore):
        self.log = log
        self.store = store
        self.labels: Dict[str, str] = {}   # label -> claim id
        self.finals: list = []
        # tool_call/tool_result pairing. Keyed by tool_use_id when the hook
        # recorded one — concurrent tool calls interleave in the feed, so
        # adjacency lies about which call a result belongs to. The single
        # slot survives only as the fallback for ledgers written before
        # tool_use_id existed, preserving their replay byte-for-byte.
        self._pending_by_id: Dict[str, Optional[str]] = {}
        self._pending_tool: Optional[str] = None

    def resolve(self, ref: str) -> str:
        if ref not in self.labels:
            raise TraceError(f"unknown claim reference: {ref}")
        return self.labels[ref]

    def feed(self, event: dict) -> None:
        etype = event["type"]
        if etype == "tool_call":
            call_id = event["payload"].get("tool_use_id")
            if call_id:
                self._pending_by_id[call_id] = event["payload"].get("tool")
            else:
                self._pending_tool = event["payload"].get("tool")
        elif etype == "tool_result":
            call_id = event["payload"].get("tool_use_id")
            if call_id:
                tool = self._pending_by_id.pop(call_id, None)
            else:
                tool, self._pending_tool = self._pending_tool, None
            claim = Claim(CLAIM_EFFECT, effect_text(event, tool))
            self.store.add_claim(claim)
            self.labels[f"effect:{event['seq']}"] = claim.id
        elif etype == "commit":
            payload = event["payload"]
            deps = tuple(self.resolve(r) for r in payload.get("deps", []))
            revises = payload.get("revises")
            claim = Claim(
                kind=payload["kind"],
                text=payload["text"],
                deps=deps,
                revises=self.resolve(revises) if revises else None,
            )
            # register before inserting: the claim id is its content hash, and
            # under the on-commit rule add_claim runs the whole verification
            # inline — if a check blows up, the claim is still in the store and
            # its label must still resolve for gv.status / gv.verify
            if "label" in payload:
                self.labels[payload["label"]] = claim.id
            if payload["kind"] == "final":
                self.finals.append(claim.id)
            self.store.add_claim(claim)
        # session_start / message / stop / verifier-side events: no graph action

    def replay(self) -> "Builder":
        for event in self.log.replay():
            self.feed(event)
        return self
