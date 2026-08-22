"""
Codex CLI rollout log -> General Verifier hook feed.

The recording hooks only exist for Claude Code. Codex writes its own append-only rollout JSONL (~/.codex/sessions/<Y>/<M>/<D>/rollout-*.jsonl)。
So, it carries the same substance: one `function_call` per tool invocation and one `function_call_output` per result, paired by `call_id`. 
This module rewrites those into the exact feed lines the hooks would have produced, so a Codex trace scores through the same offline pipeline as a Claude Code trace.

Usage:
  python -m verifier.stream.codex ROLLOUT.jsonl STATE_DIR
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional

from ..core.model import canonical

EXCERPT_CHARS = 30000       # same bound as hooks/post_tool_use.py

CALL_TYPES = ("function_call", "custom_tool_call")
OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")

# Harness-injected wrapper blocks in user messages: environment banners and
# plugin advertisements, not task content. A content item opening with one
# of these tags is dropped when hunting for the task instruction.
NOISE_TAGS = ("<recommended_plugins>", "<environment_context>",
              "<user_instructions>", "<turn_context>", "<permissions")


def _payload(event: Dict[str, Any]) -> Dict[str, Any]:
    return event.get("payload") or {}


def _tool_input(p: Dict[str, Any]) -> Any:
    """
    Codex ships call arguments as a JSON string; 
    decode when possible so the recorded input matches the hooks' structured tool_input.
    """
    
    raw = p.get("arguments") if "arguments" in p else p.get("input")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw

def _tool_output(p: Dict[str, Any]) -> Any:
    out = p.get("output")
    if isinstance(out, str):
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return out
    return out


def feed_lines(rollout_path: str) -> Iterator[Dict[str, Any]]:
    """Rollout events -> hook-feed lines, in file order."""

    session: Optional[str] = None
    tool_of: Dict[str, Any] = {}   # call_id -> tool name, to backfill results
    with open(rollout_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = _payload(event)
            if event.get("type") == "session_meta":
                session = p.get("session_id") or p.get("id") or session
                continue
            if event.get("type") != "response_item":
                continue
            kind = p.get("type")
            if kind in CALL_TYPES:
                tool_of[p.get("call_id")] = p.get("name")
                yield {
                    "type": "tool_call",
                    "payload": {
                        "tool": p.get("name"),
                        "tool_use_id": p.get("call_id"),
                        "input": _tool_input(p),
                        "session": session,
                        "agent": None,
                    }
                }
            elif kind in OUTPUT_TYPES:
                response = _tool_output(p)
                blob = canonical(response)
                yield {
                    "type": "tool_result",
                    "payload": {
                        # the hooks record the tool name on results too;
                        # codex omits it, so pair it back by call_id
                        "tool": tool_of.get(p.get("call_id")),
                        "tool_use_id": p.get("call_id"),
                        "response_sha256": hashlib.sha256(
                            blob.encode("utf-8")).hexdigest(),
                        "response_chars": len(blob),
                        "excerpt_truncated": len(blob) > EXCERPT_CHARS,
                        "excerpt": blob[:EXCERPT_CHARS],
                        "session": session,
                        "agent": None,
                    }
                }


def task_instruction(rollout_path: str) -> str:
    """The task statement the run was given: the first substantive user
    message in the rollout. Harness wrapper blocks (plugin ads, environment
    banners) are dropped; what remains of the earliest non-empty user
    message is the instruction. Empty string when none is found."""
    with open(rollout_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = _payload(event)
            if event.get("type") != "response_item" \
                    or p.get("type") != "message" or p.get("role") != "user":
                continue
            texts = []
            for block in p.get("content") or []:
                if not isinstance(block, dict):
                    continue
                text = (block.get("text") or "").strip()
                if not text or text.startswith(NOISE_TAGS):
                    continue
                texts.append(text)
            joined = "\n".join(texts).strip()
            if joined:
                return joined
    return ""


def final_text(rollout_path: str) -> str:
    """The solver's last assistant message (final output)."""

    last = ""
    with open(rollout_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = _payload(event)
            if event.get("type") == "event_msg" and p.get("type") == "agent_message":
                last = p.get("message") or last
            elif event.get("type") == "response_item" and p.get("type") == "message" \
                    and p.get("role") == "assistant":
                texts = [b.get("text", "") for b in (p.get("content") or [])
                         if isinstance(b, dict)]
                joined = "\n".join(t for t in texts if t)
                last = joined or last
    return last


def convert(rollout_path: str, state_dir: str) -> Dict[str, int]:
    """Write hook_feed.jsonl + final.txt into `state_dir` (the layout verifier.offline expects). Returns counts."""
    os.makedirs(state_dir, exist_ok=True)
    lines: List[Dict[str, Any]] = list(feed_lines(rollout_path))
    feed = os.path.join(state_dir, "hook_feed.jsonl")
    with open(feed, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    with open(os.path.join(state_dir, "final.txt"), "w", encoding="utf-8") as f:
        f.write(final_text(rollout_path))
    calls = sum(1 for ln in lines if ln["type"] == "tool_call")
    return {"events": len(lines), "tool_calls": calls}


def newest_rollout(sessions_dir: str, since: Optional[float] = None) -> Optional[str]:
    """Find the most recently modified rollout-*.jsonl under a codex sessions tree (optionally only those touched after `since`)."""
    best, best_mtime = None, -1.0
    for root, _dirs, files in os.walk(sessions_dir):
        for name in files:
            if not (name.startswith("rollout-") and name.endswith(".jsonl")):
                continue
            path = os.path.join(root, name)
            mtime = os.path.getmtime(path)
            if since is not None and mtime < since:
                continue
            if mtime > best_mtime:
                best, best_mtime = path, mtime
    return best


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    counts = convert(sys.argv[1], sys.argv[2])
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
