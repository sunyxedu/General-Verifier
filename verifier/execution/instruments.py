"""
Instruments: the mechanical components of the verifier.

Contract (report, fail-closed iron rule): any crash / timeout / unparseable output is a FAILED discharge. Never converted into a pass.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
            # write-then-rename: a concurrent reader (the judger opening an
            # artifact while another seat stores one) never sees a partial
            # file, and two writers of the same digest are idempotent
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, sort_keys=True,
                          indent=1)
            os.replace(tmp, path)
        return digest

    def get(self, digest: str) -> Dict[str, Any]:
        with open(os.path.join(self.root, digest + ".json"), "r", encoding="utf-8") as f:
            return json.load(f)


def world_evidence_context(
    state_dir: Optional[str], live_world: bool = True,
) -> Tuple[str, str, str]:
    """Return ``(mode, world_path, manifest_path)`` for a verifier seat.

    An empty captured-world path is ambiguous on its own: during an online
    run it means the original world is still live, while in offline replay it
    means that world is gone and was not captured.  Keep that distinction
    explicit at every semantic seat instead of asking an LLM to infer it from
    a missing path.
    """
    if state_dir:
        state = os.path.abspath(state_dir)
        world = os.path.join(state, "world")
        if os.path.isdir(world):
            manifest = os.path.join(state, "world_manifest.json")
            return (
                "captured",
                world,
                manifest if os.path.exists(manifest) else "",
            )
    return ("live", "", "") if live_world else ("unavailable", "", "")


def world_map(state_dir: Optional[str], live_world: bool = True) -> str:
    """The scheme-agnostic world description every seat receives (see
    core.world.WorldView): the task's own addressing scheme, which
    addresses were captured, and where they live on this host."""
    from ..core.world import load_world_view
    return load_world_view(state_dir, live_world=live_world).render()


def authoritative_task_context(state_dir: Optional[str]) -> str:
    """Inline harness task truth so semantic seats cannot skip a Read."""
    if not state_dir:
        return "(none supplied)"
    task = os.path.join(os.path.abspath(state_dir), "task.md")
    if not os.path.isfile(task):
        return "(none supplied)"
    with open(task, encoding="utf-8") as handle:
        text = handle.read()
    return json.dumps(
        {"path": os.path.abspath(task), "text": text},
        ensure_ascii=False,
    )


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
