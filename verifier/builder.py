"""Mechanical parser (component 6) + file replay (component 4).

Deterministic, zero-LLM: tool events become effect claims whose text is the
canonical ledger reference; commit events become assertion/final claims. Trace
authors refer to claims by human labels; the builder resolves labels to
content-addressed ids ("effect:<seq>" refers to a tool_result's effect claim).
"""
from __future__ import annotations

from typing import Dict, Optional

from .events import EventLog
from .model import Claim, canonical, KIND_EFFECT
from .store import ClaimStore


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
    def __init__(self, log: EventLog, store: ClaimStore):
        self.log = log
        self.store = store
        self.labels: Dict[str, str] = {}   # label -> claim id
        self.finals: list = []

    def resolve(self, ref: str) -> str:
        if ref not in self.labels:
            raise TraceError(f"unknown claim reference: {ref}")
        return self.labels[ref]

    def replay(self) -> "Builder":
        pending_tool: Optional[str] = None
        for event in self.log.replay():
            etype = event["type"]
            if etype == "tool_call":
                pending_tool = event["payload"].get("tool")
            elif etype == "tool_result":
                claim = Claim(KIND_EFFECT, effect_text(event, pending_tool))
                self.store.add_claim(claim)
                self.labels[f"effect:{event['seq']}"] = claim.id
                pending_tool = None
            elif etype == "commit":
                payload = event["payload"]
                deps = tuple(self.resolve(r) for r in payload.get("deps", []))
                revises = payload.get("revises")
                claim = Claim(
                    kind=payload["kind"],
                    text=payload["text"],
                    deps=deps,
                    branch=payload.get("branch"),
                    revises=self.resolve(revises) if revises else None,
                )
                self.store.add_claim(claim)
                if "label" in payload:
                    self.labels[payload["label"]] = claim.id
                if payload["kind"] == "final":
                    self.finals.append(claim.id)
            # session_start / message / stop / verifier-side events: no graph action
        return self
