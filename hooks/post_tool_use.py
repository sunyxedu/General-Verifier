#!/usr/bin/env python3
"""PostToolUse hook: append a raw tool_result line to the MCP server's hook feed.

The response is digested (sha256 over its canonical JSON) plus a bounded excerpt. 
The excerpt is generous: the ledger is first-class verification evidence. 
Claims about past events/states are discharged against what the hooks recorded here, so the recorded output must carry the substance, not just the identity.

Settings (.claude/settings.json):
  {"hooks": {"PostToolUse": [{"matcher": "*", "hooks": [ {"type": "command", "command": "python3 hooks/post_tool_use.py"}]}]}}
"""

import hashlib
import json
import os
import sys


def main() -> None:
    data = json.load(sys.stdin)
    feed = os.environ.get("GV_HOOK_FEED", os.path.join(".gv", "hook_feed.jsonl"))
    directory = os.path.dirname(feed)
    if directory:
        os.makedirs(directory, exist_ok=True)
    response = data.get("tool_response")
    canonical = json.dumps(response, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), default=str)
    line = json.dumps(
        {
            "type": "tool_result",
            "payload": {
                "tool": data.get("tool_name"),
                "tool_use_id": data.get("tool_use_id"),
                "response_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "excerpt": canonical[:30000],
                "session": data.get("session_id"),
                "agent": data.get("agent_id"),
            },
        },
        ensure_ascii=False,
    )
    with open(feed, "a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
