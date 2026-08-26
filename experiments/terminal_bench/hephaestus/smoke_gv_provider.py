#!/usr/bin/env python3
"""End-to-end Claude Agent SDK smoke test against a custom GV provider.

Run with GV_PROVIDER=custom and the same GV_MODEL,
GV_ANTHROPIC_BASE_URL, GV_API_KEY, and GV_TEXT_ONLY environment used by the
real offline scorer.  Unlike a direct HTTP probe, this verifies the exact
SDK/CLI streaming, tool-use, and structured-output path used by every GV seat.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verifier.llm.agents import AgentBackend, AgentRole

ROLE = AgentRole(
    name="provider_smoke",
    system=(
        "This is a protocol smoke test. Follow the requested tool step, then "
        "return the structured result exactly."
    ),
    template=(
        "Use the Bash tool to evaluate the integer expression 6*7 with "
        "Python. Return that integer as answer and set used_tool to true."
    ),
    schema={"answer": int, "used_tool": bool},
    tools=["Bash"],
    max_turns=4,
)


def main() -> int:
    backend = AgentBackend(roles={ROLE.name: ROLE})
    payload = ROLE.call(backend, {})
    if payload != {"answer": 42, "used_tool": True}:
        raise RuntimeError(f"unexpected structured result: {payload!r}")
    summary = {
        "result": payload,
        "transcript": backend.transcripts[-1] if backend.transcripts else None,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise
