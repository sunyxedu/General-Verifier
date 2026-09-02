"""
Unified event stream: append-only JSONL, the system's only input and its only log. 
Replay = reading the same file.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Iterator, Optional

from .model import content_hash

EVENT_TYPES = (
    # solver-side behaviors
    "session_start",
    "tool_call",
    "tool_result",
    "message",
    "commit",
    "stop",

    # scheduling
    "verify_request",
    "status_change",
    "certificate",

    # obligation & critic agent
    "obligations_compiled",
    "obligations_residualized",
    "decomposition_attacked",
    
    # binding game
    "methods_enumerated",
    "judger_call_failed",
    "judger_verdict",
    "method_rejected",
    "method_deferred",
    "method_redo",
    "method_bound",
    "discharge",

    # search support
    "dedup_hit",
    
    # memory (advisory)
    "memory_recorded",
)


class EventLog:
    """
    Append-only JSONL. Every event gets a monotonically increasing seq.

    The file is only ever opened in append mode;
    replay validates monotonicity so silent truncation/reordering is detected rather than absorbed.
    """

    def __init__(self, path: str):
        self.path = path
        # one writer at a time: seq allocation and the write are atomic
        # together, so concurrent seats (parallel obligations/claims) can
        # never interleave into a gap or a duplicate that replay rejects
        self._lock = threading.Lock()
        self._seq = self._last_seq()

    def _last_seq(self) -> int:
        if not os.path.exists(self.path):
            return 0
        last = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line)["seq"]
        return last

    def append(self, type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if type not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {type}")
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "ts": time.time(), "type": type,
                     "payload": payload}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False,
                                   sort_keys=True) + "\n")
                f.flush()
        return event

    def replay(self) -> Iterator[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return
        expected = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                expected += 1
                if event["seq"] != expected:
                    raise ValueError(
                        f"event log corrupted: expected seq {expected}, got {event['seq']}"
                    )
                yield event

    def get(self, seq: int) -> Optional[Dict[str, Any]]:
        for event in self.replay():
            if event["seq"] == seq:
                return event
        return None

    @staticmethod
    def event_digest(event: Dict[str, Any]) -> str:
        """Digest over the immutable identity of an event (ts excluded so the
        ledger check does not depend on wall-clock noise)."""
        return content_hash({
            "seq": event["seq"], 
            "type": event["type"], 
            "payload": event["payload"]
        })
