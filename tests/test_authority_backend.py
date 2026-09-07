"""Authority-only backend gates are hard capabilities, not prompt wishes."""
import json
import tomllib
from dataclasses import replace

import pytest

from verifier.llm.agents import COMPILE_OBLIGATIONS_AGENT, CRITIC_AGENT
from verifier.llm.authority_backend import (
    AUTHORITY_ONLY_NOTE, AuthorityBackendUnavailable,
    _DISABLED_CODE_MODE_NOTICE,
    authority_compilation_backend, authority_policy_args,
    validate_authority_stream,
)
from verifier.llm.schema import FixtureBackend


def stream(*middle):
    return "\n".join(json.dumps(event) for event in (
        {"type": "thread.started", "thread_id": "synthetic"},
        {"type": "turn.started"}, *middle,
        {"type": "turn.completed", "usage": {}},
    ))


def test_authority_profile_has_no_world_read_write_or_network(tmp_path):
    args = authority_policy_args(str(tmp_path))
    assert {"--strict-config", "--ignore-user-config", "--ignore-rules",
            "--ephemeral"} <= set(args)
    assert "--sandbox" not in args
    assert "--dangerously-bypass-approvals-and-sandbox" not in args
    values = [args[i + 1] for i, arg in enumerate(args) if arg == "-c"]
    config = tomllib.loads("\n".join(values))
    policy = config["permissions"]["gv-authority"]
    assert policy["filesystem"][":root"] == "deny"
    assert set(policy["filesystem"].values()) <= {"read", "deny"}
    assert policy["network"]["enabled"] is False
    assert config["web_search"] == "disabled"
    assert config["tools"] == {"web_search": False}
    assert config["mcp_servers"] == {}
    assert config["project_doc_max_bytes"] == 0
    for name in ("shell_tool", "unified_exec", "apps", "plugins", "multi_agent",
                 "memories", "browser_use", "computer_use", "workspace_dependencies"):
        assert config["features"][name] is False
    assert config["features"]["skip_host_skill_discovery"] is True


def test_valid_tool_free_stream():
    validate_authority_stream(stream(
        {"type": "item.completed", "item": {"type": "reasoning", "text": "..."}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
    ))


def test_exact_capability_disabled_startup_notice_is_not_a_tool_result():
    raw = stream().splitlines()
    notice = json.dumps({"type": "item.completed", "item": {
        "id": "notice", "type": "error", "message": _DISABLED_CODE_MODE_NOTICE}})
    validate_authority_stream("\n".join([raw[0], notice, *raw[1:]]))
    with pytest.raises(AuthorityBackendUnavailable):
        validate_authority_stream("\n".join([raw[0], raw[1], notice, raw[2]]))
    with pytest.raises(AuthorityBackendUnavailable):
        validate_authority_stream("\n".join([
            raw[0], notice.replace("is unavailable", "is available"), *raw[1:]]))


@pytest.mark.parametrize("item_type", [
    "command_execution", "mcp_tool_call", "web_search", "file_change",
    "image_generation", "collab_tool_call", "future_tool_type", None,
])
def test_any_tool_or_unknown_item_invalidates_generation(item_type):
    with pytest.raises(AuthorityBackendUnavailable):
        validate_authority_stream(stream(
            {"type": "item.completed", "item": {"type": item_type}}))


@pytest.mark.parametrize("extra", [
    {"type": "thread.resumed"}, {"type": "turn.failed"},
    {"type": "context_compacted"}, {"type": "memory_retrieved"},
    {"type": "error"}, {"type": "turn.completed"},
    {"type": "thread.started"}, {}, [],
])
def test_unknown_incomplete_or_reused_context_is_rejected(extra):
    with pytest.raises(AuthorityBackendUnavailable):
        validate_authority_stream(stream(extra))


@pytest.mark.parametrize("raw", ["", "not JSON", '{"type":"turn.started"}'])
def test_non_complete_stream_is_rejected(raw):
    with pytest.raises(AuthorityBackendUnavailable):
        validate_authority_stream(raw)


def test_role_registry_detached_and_fixture_not_duck_typed():
    fixture = FixtureBackend({})
    original = list(COMPILE_OBLIGATIONS_AGENT.tools)
    with authority_compilation_backend(
            fixture, COMPILE_OBLIGATIONS_AGENT, CRITIC_AGENT) as session:
        assert session.backend is fixture
        assert session.obligations_role.tools == []
        assert session.critic_role.tools == []
        assert AUTHORITY_ONLY_NOTE in session.obligations_role.system
    assert COMPILE_OBLIGATIONS_AGENT.tools == original
    class PretendFixture(FixtureBackend):
        pass
    with pytest.raises(AuthorityBackendUnavailable):
        with authority_compilation_backend(
                PretendFixture({}), COMPILE_OBLIGATIONS_AGENT, CRITIC_AGENT):
            pytest.fail("a subclass is not a no-tool boundary")
