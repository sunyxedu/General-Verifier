"""Live M2 acceptance: a vanilla Claude Code session (SDK harness) mounts the
sidecar over stdio MCP, commits claims, and walks away with a certificate.
Opt-in via GV_LIVE=1."""
import json
import os
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GV_LIVE") != "1", reason="live SDK test; set GV_LIVE=1"
)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(PROJECT, ".venv", "bin", "python")


def test_live_solver_mounts_sidecar_over_stdio(tmp_path):
    import asyncio

    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    state = str(tmp_path / "gv-state")
    options = ClaudeAgentOptions(
        model="claude-haiku-4-5",
        mcp_servers={
            "gv": {
                "type": "stdio",
                "command": PYTHON,
                "args": ["-m", "verifier.sidecar", "--state-dir", state,
                         "--seats", "agent", "--no-game",
                         "--model", "claude-haiku-4-5"],
                "env": {"PYTHONPATH": PROJECT,
                        "PATH": os.environ.get("PATH", "")},
            }
        },
        allowed_tools=["mcp__gv__commit", "mcp__gv__certificate",
                       "mcp__gv__status"],
        permission_mode="bypassPermissions",
        max_turns=12,
        cwd=str(tmp_path),
        setting_sources=[],
        output_format={
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    # closed enum: a fabricated verdict cannot validate
                    "verdict": {"type": "string",
                                "enum": ["verified", "unverified",
                                         "refuted", "error"]},
                    "certificate_id": {"type": "string"},
                },
                "required": ["verdict", "certificate_id"],
                "additionalProperties": False,
            },
        },
        env={"MCP_TOOL_TIMEOUT": "900000", "MCP_TIMEOUT": "120000"},
        max_buffer_size=64 * 1024 * 1024,
    )
    prompt = (
        "依次做三件事，全部通过 gv 工具完成：\n"
        "1) commit(label='A', kind='assertion', text='1 到 10 的整数之和等于 55', deps=[])\n"
        "2) commit(label='F', kind='final', text='结论：求和结果正确', deps=['A'])\n"
        "3) certificate(final_label='F')\n"
        "结构化输出中的 verdict 与 certificate_id 必须逐字取自 certificate 工具"
        "返回的 JSON（字段 verdict 与 certificate_id），不得改写或自创；"
        "工具报错则 verdict 填 error。"
    )

    async def run():
        result = None
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                result = message
        return result

    result = asyncio.run(run())
    assert result is not None and result.subtype == "success"
    out = result.structured_output
    assert out["verdict"] == "verified", out

    # the sidecar's own ledger corroborates what the solver reported
    events = [json.loads(line)
              for line in open(os.path.join(state, "events.jsonl"))]
    certs = [e for e in events if e["type"] == "certificate"]
    assert certs and certs[-1]["payload"]["verdict"] == "verified"
    assert certs[-1]["payload"]["digest"] == out["certificate_id"]
