"""Live SDK round-trips — opt-in (GV_LIVE=1), spends real tokens.

Two smokes: a pure-text role (plumbing: query -> output_format ->
structured_output -> our validate), and a Bash-palette role (grounding: the
agent must actually execute something and report what it saw).
"""
import os

import pytest

from verifier.agents import AgentBackend, AgentRole

pytestmark = pytest.mark.skipif(
    os.environ.get("GV_LIVE") != "1", reason="live SDK test; set GV_LIVE=1"
)

MODEL = "claude-haiku-4-5"   # plumbing smoke only — real roles inherit stronger defaults


def test_live_pure_text_role(tmp_path):
    role = AgentRole(
        name="echo",
        template="计算 {a} * {b}，把结果数字作为字符串填入 answer。",
        schema={"answer": str},
        tools=[],
        model=MODEL,
        max_turns=4,
    )
    backend = AgentBackend(roles={"echo": role}, cwd=str(tmp_path))
    payload = role.call(backend, {"a": 12, "b": 9})
    assert payload["answer"].strip() == "108"
    t = backend.transcripts[-1]
    assert t["subtype"] == "success" and t["session_id"]


def test_live_bash_palette_role(tmp_path):
    role = AgentRole(
        name="prober",
        template=(
            "用 Bash 运行 `echo $((6*7))`，把它 stdout 里的数字作为字符串填入 "
            "answer；把你实际运行的命令原样填入 command。"
        ),
        schema={"answer": str, "command": str},
        tools=["Bash"],
        model=MODEL,
        max_turns=6,
    )
    backend = AgentBackend(roles={"prober": role}, cwd=str(tmp_path))
    payload = role.call(backend, {})
    assert payload["answer"].strip() == "42"
    assert "echo" in payload["command"]
