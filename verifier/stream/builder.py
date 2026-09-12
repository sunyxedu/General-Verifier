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
        # claim id -> seq of the event that first introduced it.  A claim
        # speaks about the record as it stood when the claim was made; the
        # kernel freezes exactly that ledger prefix for its verification.
        self.commit_seq: Dict[str, int] = {}
        self._pending_by_id: Dict[str, Optional[str]] = {}
        self._pending_tool: Optional[str] = None

    def resolve(self, ref: str) -> str:
        if ref not in self.labels:
            raise TraceError(f"unknown claim reference: {ref}")
        return self.labels[ref]

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
            # before add_claim: an on-commit trigger verifies inside it
            self.commit_seq.setdefault(claim.id, event["seq"])
            self.store.add_claim(claim)
            self.labels[f"effect:{event['seq']}"] = claim.id
        elif etype in ("commit", "extracted_commit"):
            payload = event["payload"]
            deps = tuple(self.resolve(r) for r in payload.get("deps", []))
            revises = payload.get("revises")
            claim = Claim(
                kind=payload["kind"],
                text=payload["text"],
                deps=deps,
                revises=self.resolve(revises) if revises else None,
            )
            if "label" in payload:
                self.labels[payload["label"]] = claim.id
            if payload["kind"] == "final":
                self.finals.append(claim.id)
            self.commit_seq.setdefault(claim.id, event["seq"])
            self.store.add_claim(claim)

    def replay(self) -> "Builder":
        for event in self.log.replay():
            self.feed(event)
        return self
