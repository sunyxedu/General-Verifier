"""Opt-in benign live smoke; never part of the default unit suite."""
import json
import os

import pytest

from verifier.llm.agents import AgentRole
from verifier.llm.authority_backend import authority_compilation_backend
from verifier.llm.codex_backend import CodexRuntime, codex_usage_ledger


@pytest.mark.skipif(os.environ.get("GV_TEST_LIVE_CODEX") != "1",
                    reason="requires explicit paid GPT-5.6-Luna smoke opt-in")
def test_live_authority_only_luna_has_complete_usage_and_no_tool_activity(tmp_path):
    role = AgentRole(
        name="obligations", template='Return exactly one JSON object: {{"ok":true}}.',
        schema={"ok": bool}, tools=[],
        system="This is a benign protocol smoke test. Follow the requested JSON schema.")
    critic = AgentRole(name="critic", template="unused", schema={"ok": bool})
    backend = CodexRuntime(cwd=str(tmp_path), model="gpt-5.6-luna", timeout=180)
    usage_path = tmp_path / "usage.jsonl"
    with codex_usage_ledger(str(usage_path), {str(tmp_path): {"smoke": "authority"}}):
        with authority_compilation_backend(backend, role, critic) as session:
            assert session.obligations_role.call(session.backend, {}) == {"ok": True}
    rows = [json.loads(line) for line in usage_path.read_text().splitlines()]
    assert rows
    assert all(row["usage_complete"] for row in rows)
    assert rows[-1]["attempt_status"] == "succeeded"
    assert all(row["model"] == "gpt-5.6-luna" for row in rows)
    assert all(row["state_dir"] == str(tmp_path.resolve()) for row in rows)
