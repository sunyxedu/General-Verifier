"""
Instruments: the mechanical components of the verifier.

Contract (report, fail-closed iron rule): any crash / timeout / unparseable output is a FAILED discharge. Never converted into a pass.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..core.events import EventLog
from ..core.model import content_hash, OUTCOME_HOLDS, OUTCOME_REFUTED, OUTCOME_FAILED


@dataclass(frozen=True)
class RunResult:
    outcome: str                 # holds | refuted | failed
    artifacts: Dict[str, Any]
    witness: Optional[Any] = None


class ArtifactStore:
    """Content-addressed artifact directory."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def put(self, payload: Dict[str, Any]) -> str:
        digest = content_hash(payload)
        path = os.path.join(self.root, digest + ".json")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=1)
        return digest

    def get(self, digest: str) -> Dict[str, Any]:
        with open(os.path.join(self.root, digest + ".json"), "r", encoding="utf-8") as f:
            return json.load(f)


class LedgerInstrument:
    """
    Verifies effect claims against the event stream.

    An effect claim's text is the canonical JSON produced by the mechanical
    parser: {"event_seq": n, "tool": ..., "digest": <event_digest>}. The check
    recomputes the digest from the log. The ledger, not the narration, is the
    source of truth for "this happened".
    """

    name = "ledger"

    def __init__(self, log: EventLog):
        self.log = log

    def check(self, effect_text: str) -> RunResult:
        base = {"instrument": self.name}
        try:
            ref = json.loads(effect_text)
            seq, claimed_digest = ref["event_seq"], ref["digest"]
        except Exception:
            return RunResult(OUTCOME_FAILED, {**base, "error": "malformed effect text"})

        event = self.log.get(seq)
        if event is None:
            return RunResult(
                OUTCOME_REFUTED, {**base, "event_seq": seq},
                witness={"reason": "no such event in ledger"},
            )
        actual = EventLog.event_digest(event)
        if actual != claimed_digest:
            return RunResult(
                OUTCOME_REFUTED,
                {**base, "event_seq": seq, "actual_digest": actual},
                witness={"reason": "ledger digest mismatch"},
            )
        return RunResult(OUTCOME_HOLDS, {**base, "event_seq": seq, "digest": actual})
