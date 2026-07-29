"""Agent-role plumbing tests — everything around the SDK call, no API spend.

The live path (real query() round-trips) lives in tests/test_agents_live.py,
opt-in via GV_LIVE=1.
"""
import dataclasses

import pytest

import verifier.agents as agents
from verifier.agents import (
    ALL_AGENT_ROLES, AgentBackend, AgentRole, SDK_AVAILABLE, to_json_schema,
    COMPILE_OBLIGATIONS_AGENT, DISCHARGE_AGENT, ENTAILMENT_AUDIT_AGENT,
)
from verifier.roles import RoleError, validate


def test_sdk_importable():
    assert SDK_AVAILABLE, "claude-agent-sdk must be installed in this venv"


# ---- schema DSL --------------------------------------------------------------

def test_to_json_schema_scalars_enums_lists_objects():
    spec = {
        "outcome": ("holds", "refuted", "failed"),
        "items": [{"text": str, "count": int}],
        "ok": bool,
    }
    js = to_json_schema(spec)
    assert js["type"] == "object"
    assert js["required"] == ["outcome", "items", "ok"]
    assert js["additionalProperties"] is False
    assert js["properties"]["outcome"] == {
        "type": "string", "enum": ["holds", "refuted", "failed"]
    }
    inner = js["properties"]["items"]["items"]
    assert inner["properties"]["text"] == {"type": "string"}
    assert inner["properties"]["count"] == {"type": "integer"}
    assert js["properties"]["ok"] == {"type": "boolean"}


def test_to_json_schema_rejects_unknown_spec():
    with pytest.raises(ValueError):
        to_json_schema({"x": object})


def test_validate_enforces_enum():
    validate({"outcome": ("holds", "refuted")}, {"outcome": "holds"})
    with pytest.raises(RoleError, match="not in"):
        validate({"outcome": ("holds", "refuted")}, {"outcome": "maybe"})


def test_every_role_schema_converts_and_validates_itself():
    for name, role in ALL_AGENT_ROLES.items():
        js = to_json_schema(role.schema)   # raises if the DSL is malformed
        assert js["type"] == "object", name


# ---- role definitions --------------------------------------------------------

def test_templates_carry_no_domain_knowledge():
    """The whole design rests on templates being domain-free."""
    banned = ["lean", "mathlib", "pytest", "√2", "素数", "阶梯电价", "同比"]
    for name, role in ALL_AGENT_ROLES.items():
        blob = (role.template + role.system).lower()
        for word in banned:
            assert word.lower() not in blob, f"{name} leaks domain knowledge: {word}"


def test_capability_boundaries():
    """The palette is part of a role's identity."""
    # obligation compiler is answer-blind AND execution-blind
    assert "Bash" not in COMPILE_OBLIGATIONS_AGENT.tools
    assert "Write" not in COMPILE_OBLIGATIONS_AGENT.tools
    # discharge must be able to act; auditor must be able to open artifacts
    assert "Bash" in DISCHARGE_AGENT.tools
    assert "Read" in ENTAILMENT_AUDIT_AGENT.tools


def test_registry_names_are_consistent():
    for name, role in ALL_AGENT_ROLES.items():
        assert role.name == name


# ---- backend dispatch --------------------------------------------------------

TRIVIAL = AgentRole(
    name="trivial", template="q: {q}", schema={"answer": str}
)


def test_complete_looks_up_role_by_name(monkeypatch):
    backend = AgentBackend(roles={"trivial": TRIVIAL})

    async def fake_run(self, role, prompt):
        assert role is TRIVIAL and prompt == "q: hi"
        return {"answer": "ok"}

    monkeypatch.setattr(AgentBackend, "run", fake_run)
    assert backend.complete("trivial", "q: hi", {}) == {"answer": "ok"}


def test_complete_rejects_unknown_role():
    backend = AgentBackend(roles={})
    with pytest.raises(RoleError, match="unknown agent role"):
        backend.complete("nope", "p", {})


def test_role_call_retries_once_then_raises(monkeypatch):
    calls = []

    class Flaky:
        def complete(self, role_name, prompt, variables):
            calls.append(prompt)
            raise RoleError("boom")

    with pytest.raises(RoleError, match="failed twice"):
        TRIVIAL.call(Flaky(), {"q": "x"})
    assert len(calls) == 2
    assert "修正后重来" in calls[1]      # the second prompt carries the error


def test_role_call_recovers_on_second_attempt():
    attempts = []

    class FlakyThenGood:
        def complete(self, role_name, prompt, variables):
            attempts.append(1)
            if len(attempts) == 1:
                raise RoleError("boom")
            return {"answer": "ok"}

    assert TRIVIAL.call(FlakyThenGood(), {"q": "x"}) == {"answer": "ok"}


# ---- run(): fail-closed result handling --------------------------------------

def _mk_result(**over):
    """Build a real ResultMessage, filling required fields generically so the
    test survives SDK field additions."""
    from claude_agent_sdk import ResultMessage

    kwargs = {}
    for f in dataclasses.fields(ResultMessage):
        if (f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING):
            kwargs[f.name] = {
                "subtype": "success", "duration_ms": 1, "duration_api_ms": 1,
                "is_error": False, "num_turns": 1, "session_id": "s",
                "total_cost_usd": 0.0,
            }.get(f.name, None)
    kwargs.update(over)
    return ResultMessage(**kwargs)


def _fake_query(result):
    async def fake(prompt, options):
        yield _mk_result(**result)
    return fake


def _run(backend, role):
    import asyncio
    return asyncio.run(backend.run(role, "p"))


def test_run_returns_validated_structured_output(monkeypatch):
    backend = AgentBackend(roles={})
    monkeypatch.setattr(
        agents, "query", _fake_query({"structured_output": {"answer": "42"}})
    )
    assert _run(backend, TRIVIAL) == {"answer": "42"}
    t = backend.transcripts[-1]
    assert t["role"] == "trivial" and t["subtype"] == "success"


def test_run_fails_closed_on_retry_exhaustion(monkeypatch):
    backend = AgentBackend(roles={})
    monkeypatch.setattr(agents, "query", _fake_query(
        {"subtype": "error_max_structured_output_retries", "is_error": True}
    ))
    with pytest.raises(RoleError, match="no structured output"):
        _run(backend, TRIVIAL)


def test_run_fails_closed_on_success_without_output(monkeypatch):
    backend = AgentBackend(roles={})
    monkeypatch.setattr(agents, "query", _fake_query({}))  # success, no output
    with pytest.raises(RoleError, match="no structured output"):
        _run(backend, TRIVIAL)


def test_run_applies_our_own_gate_on_top_of_sdk(monkeypatch):
    backend = AgentBackend(roles={})
    monkeypatch.setattr(
        agents, "query", _fake_query({"structured_output": {"wrong_key": 1}})
    )
    with pytest.raises(RoleError, match="missing"):
        _run(backend, TRIVIAL)


def test_run_wraps_sdk_exceptions(monkeypatch):
    backend = AgentBackend(roles={})

    async def exploding(prompt, options):
        raise RuntimeError("cli died")
        yield  # pragma: no cover

    monkeypatch.setattr(agents, "query", exploding)
    with pytest.raises(RoleError, match="sdk failure"):
        _run(backend, TRIVIAL)


def test_transient_status_backs_off_then_succeeds(monkeypatch):
    backend = AgentBackend(roles={})
    backend.transient_delays = (0, 0, 0)
    calls = {"n": 0}

    async def flaky(prompt, options):
        calls["n"] += 1
        if calls["n"] < 3:
            yield _mk_result(subtype="error_during_execution", is_error=True,
                             api_error_status=429)
        else:
            yield _mk_result(structured_output={"answer": "ok"})

    monkeypatch.setattr(agents, "query", flaky)
    assert _run(backend, TRIVIAL) == {"answer": "ok"}
    assert calls["n"] == 3


def test_transient_exhaustion_raises_role_error(monkeypatch):
    backend = AgentBackend(roles={})
    backend.transient_delays = (0, 0)
    monkeypatch.setattr(agents, "query", _fake_query(
        {"subtype": "error_during_execution", "is_error": True,
         "api_error_status": 529}
    ))
    with pytest.raises(RoleError, match="transient sdk failure persisted"):
        _run(backend, TRIVIAL)


def test_opaque_cli_placeholder_is_treated_transient(monkeypatch):
    backend = AgentBackend(roles={})
    backend.transient_delays = (0, 0)
    calls = {"n": 0}

    async def opaque_then_good(prompt, options):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Claude Code returned an error result: success")
        yield _mk_result(structured_output={"answer": "ok"})

    monkeypatch.setattr(agents, "query", opaque_then_good)
    assert _run(backend, TRIVIAL) == {"answer": "ok"}


def test_non_transient_error_fails_fast(monkeypatch):
    backend = AgentBackend(roles={})
    backend.transient_delays = (0, 0, 0)
    calls = {"n": 0}

    async def hard_fail(prompt, options):
        calls["n"] += 1
        raise RuntimeError("invalid api key")
        yield  # pragma: no cover

    monkeypatch.setattr(agents, "query", hard_fail)
    with pytest.raises(RoleError, match="sdk failure"):
        _run(backend, TRIVIAL)
    assert calls["n"] == 1                       # no pointless retries


# ---- options construction ----------------------------------------------------

def test_build_options_enforce_palette_and_official_output_format():
    backend = AgentBackend(roles={}, cwd="/tmp/x", model="m")
    role = AgentRole(
        name="r", template="t", schema={"answer": str},
        tools=["Bash", "Read"], system="附加规则", max_turns=7,
    )
    opts = backend._build_options(role, lambda line: None)
    assert opts.tools == ["Bash", "Read"]           # hard palette
    assert opts.allowed_tools == ["Bash", "Read"]   # pre-approved within it
    assert opts.output_format == {
        "type": "json_schema", "schema": to_json_schema(role.schema)
    }
    assert opts.system_prompt == {
        "type": "preset", "preset": "claude_code", "append": "附加规则"
    }
    assert opts.setting_sources == []
    assert opts.max_turns == 7 and opts.cwd == "/tmp/x" and opts.model == "m"
    assert opts.max_buffer_size == 64 * 1024 * 1024


def test_build_options_empty_system_omits_append():
    backend = AgentBackend(roles={})
    role = AgentRole(name="r", template="t", schema={"a": str})
    opts = backend._build_options(role, lambda line: None)
    assert opts.system_prompt == {"type": "preset", "preset": "claude_code"}
    assert opts.tools == []                          # no palette at all
