#!/usr/bin/env python3
"""
PreToolUse hook: append a raw tool_call line to the MCP server's hook feed.

Hooks are concurrent short-lived processes, so they never touch the canonical event log. 
They append bare JSON lines to a feed file (GV_HOOK_FEED); the server, the log's single writer, ingests them in arrival order.  

Settings (.claude/settings.json):
  {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": "python3 hooks/pre_tool_use.py"}]}]}}
"""

import json
import os
import sys
import time


def main() -> None:
    data = json.load(sys.stdin)
    feed = os.environ.get("GV_HOOK_FEED", os.path.join(".gv", "hook_feed.jsonl"))
    directory = os.path.dirname(feed)
    if directory:
        os.makedirs(directory, exist_ok=True)
    line = json.dumps(
        {
            "type": "tool_call",
            "payload": {
                "tool": data.get("tool_name"),
                "tool_use_id": data.get("tool_use_id"),
                "input": data.get("tool_input"),
                "session": data.get("session_id"),
                "agent": data.get("agent_id"),
                # wall-clock time of the call (pre) or its completion
                # (post): which world changes fell inside a solver tool
                # run, and which inside nothing the solver was running
                "ts": time.time(),
            },
        },
        ensure_ascii=False,
    )
    # a lone surrogate (undecodable bytes in a tool's input) cannot be
    # encoded as UTF-8 and used to raise here, dropping the call from the
    # record: it is written as "?", so the line is always written and valid
    with open(feed, "a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
