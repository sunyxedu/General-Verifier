"""
A EventLog to ClaimStore builder.

The EventLog flows into the builder by `feed(event)`, and map to different nodes in the ClaimStore.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..core.events import EventLog
from ..core.model import Claim, canonical, CLAIM_EFFECT
from ..core.store import ClaimStore


class TraceError(ValueError):
    pass


def effect_text(event: dict, tool: Optional[str]) -> str:
    return canonical({
        "event_seq": event["seq"],
        "tool": tool,
        "digest": EventLog.event_digest(event),
    })


class Builder:
    """
    Incremental: feed(event) is the one entry point; replay() is just feeding the whole file. 
    Online mode is a Tail pushing the same events into the same feed.
    The kernel, wired to the store's trigger, does not know or care which mode it is in.
    """

    def __init__(self, log: EventLog, store: ClaimStore):
        self.log = log
        self.store = store
        self.labels: Dict[str, str] = {}   # label -> claim id
        self.finals: list = []
        self._pending_by_id: Dict[str, Optional[str]] = {}
        self._pending_tool: Optional[str] = None

    def resolve(self, ref: str) -> str:
        if ref not in self.labels:
            raise TraceError(f"unknown claim reference: {ref}")
        return self.labels[ref]

    def claim_from_commit(self, payload: dict) -> Claim:
        return Claim(payload["kind"], payload["text"],
                     tuple(self.resolve(r) for r in payload.get("deps", [])),
                     self.resolve(payload["revises"]) if payload.get("revises") else None)

    def require_available_label(self, label: str, claim_id: str) -> None:
        if label in self.labels and self.labels[label] != claim_id:
            raise TraceError(f"claim label {label!r} is already bound")

    # session_start / message / stop / most verifier-side events: no graph action
    # Obligation will be in the graph, but not handled here. It's handled by the kernel.
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
            self.require_available_label(f"effect:{event['seq']}", claim.id)
            self.store.add_claim(claim)
            self.labels[f"effect:{event['seq']}"] = claim.id
        elif etype == "commit":
            payload = event["payload"]
            claim = self.claim_from_commit(payload)
            if "label" in payload:
                self.require_available_label(payload["label"], claim.id)
                self.labels[payload["label"]] = claim.id
            if payload["kind"] == "final":
                self.finals.append(claim.id)
            self.store.add_claim(claim)

    def replay(self) -> "Builder":
        for event in self.log.replay():
            self.feed(event)
        return self
