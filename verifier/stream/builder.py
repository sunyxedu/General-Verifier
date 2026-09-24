"""
A EventLog to ClaimStore builder.

The EventLog flows into the builder by `feed(event)`, and map to different nodes in the ClaimStore.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..core.events import EventLog
from ..core.model import Claim, canonical, CLAIM_EFFECT
from ..core.store import ClaimStore, UnknownDependency


class TraceError(ValueError):
    pass


class UnresolvedCommit(TraceError):
    """A commit that names something the record does not hold (an unknown
    reference, a dependency missing from the store): no claim was recorded,
    and replaying the commit fails the same way."""


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
        # An effect is one call/result pair. Its label is the RESULT's
        # sequence number (the event that carries the output), but a
        # reader of the record sees two numbered lines per effect and may
        # cite the call's. Either number names the same effect, so the
        # call's sequence resolves to the pair's claim as well.
        self.call_alias: Dict[str, str] = {}   # effect:<call seq> -> label
        self._pending_seq_by_id: Dict[str, int] = {}
        self._pending_seq: Optional[int] = None

    def resolve(self, ref: str) -> str:
        if ref in self.labels:
            return self.labels[ref]
        if ref in self.call_alias:
            return self.labels[self.call_alias[ref]]
        raise TraceError(f"unknown claim reference: {ref}")

    # session_start / message / stop / most verifier-side events: no graph action
    # Obligation will be in the graph, but not handled here. It's handled by the kernel.
    def feed(self, event: dict) -> None:
        etype = event["type"]
        if etype == "tool_call":
            call_id = event["payload"].get("tool_use_id")
            if call_id:
                self._pending_by_id[call_id] = event["payload"].get("tool")
                self._pending_seq_by_id[call_id] = event["seq"]
            else:
                self._pending_tool = event["payload"].get("tool")
                self._pending_seq = event["seq"]
        elif etype == "tool_result":
            call_id = event["payload"].get("tool_use_id")
            if call_id:
                tool = self._pending_by_id.pop(call_id, None)
                call_seq = self._pending_seq_by_id.pop(call_id, None)
            else:
                tool, self._pending_tool = self._pending_tool, None
                call_seq, self._pending_seq = self._pending_seq, None
            claim = Claim(CLAIM_EFFECT, effect_text(event, tool))
            # before add_claim: an on-commit trigger verifies inside it
            self.commit_seq.setdefault(claim.id, event["seq"])
            self.store.add_claim(claim)
            label = f"effect:{event['seq']}"
            self.labels[label] = claim.id
            if call_seq is not None:
                self.call_alias[f"effect:{call_seq}"] = label
        elif etype in ("commit", "extracted_commit"):
            payload = event["payload"]
            try:
                deps = tuple(self.resolve(r) for r in payload.get("deps", []))
                revises = payload.get("revises")
                claim = Claim(
                    kind=payload["kind"],
                    text=payload["text"],
                    deps=deps,
                    revises=self.resolve(revises) if revises else None,
                )
            except (ValueError, KeyError, TypeError) as err:
                raise UnresolvedCommit(
                    str(err) if isinstance(err, TraceError)
                    else f"malformed commit: {err!r}") from None
            label = payload.get("label")
            before = self.labels.get(label) if label is not None else None
            if label is not None:
                self.labels[label] = claim.id
            if payload["kind"] == "final":
                self.finals.append(claim.id)
            new_seq = claim.id not in self.commit_seq
            self.commit_seq.setdefault(claim.id, event["seq"])
            try:
                self.store.add_claim(claim)
            except UnknownDependency as err:
                if self.store._exists(claim.id):
                    raise
                # nothing was recorded: undo the wiring above
                if label is not None:
                    if before is None:
                        self.labels.pop(label, None)
                    else:
                        self.labels[label] = before
                if payload["kind"] == "final":
                    self.finals.remove(claim.id)
                if new_seq:
                    self.commit_seq.pop(claim.id, None)
                raise UnresolvedCommit(str(err)) from None

    def replay(self) -> "Builder":
        for event in self.log.replay():
            self.feed(event)
        return self
