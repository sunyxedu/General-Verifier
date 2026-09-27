"""Tool policies are declared once and translated by each runtime; runtime
capabilities are declared members with defaults, boundaries not inherited."""
import json
from dataclasses import replace

import pytest

from verifier.config import RunOptions
from verifier.llm import authority_backend as ab
from verifier.llm import codex_backend as cb
from verifier.llm import typed_judger_backend as tjb
from verifier.llm.agents import ALL_AGENT_ROLES, ClaudeCodeRuntime
from verifier.llm.codex_backend import CodexRuntime, no_tools_policy_args
from verifier.llm.direct_backend import DirectRuntime, RoutingRuntime
from verifier.llm.policy import NO_TOOLS, NoToolsSession, ToolPolicy
from verifier.llm.runtime import (
    RuntimeCapabilities, capabilities_of, runtime_identity_type,
)
from verifier.llm.schema import FixtureBackend


class _Endpoint:
    name = "fake@x"
    model = "m"


def _session(tmp_path, policy=NO_TOOLS):
    return NoToolsSession(policy=policy, scratch=lambda: str(tmp_path),
                          validate_stream=lambda stdout: None,
                          recover_response=lambda stdout: None)


def test_policy_values():
    assert NO_TOOLS.no_tools and not NO_TOOLS.web_search
    assert not ToolPolicy().no_tools and ToolPolicy().web_search
    assert not replace(NO_TOOLS, network=True).no_tools
    with pytest.raises(ValueError, match="palette"):
        ToolPolicy(tools="everything")
    with pytest.raises(ValueError, match="no-tools policy"):
        NoToolsSession(policy=ToolPolicy(), scratch=lambda: "/",
                       validate_stream=print, recover_response=print)


def test_codex_translation_appends_denied_integrations(tmp_path):
    base = no_tools_policy_args(str(tmp_path), NO_TOOLS)
    assert base == ab.authority_policy_args(str(tmp_path))
    hardened = replace(NO_TOOLS, denied_integrations=("alpha", "beta"))
    assert no_tools_policy_args(str(tmp_path), hardened) == [
        *base, "-c", "features.alpha=false", "-c", "features.beta=false"]
    assert tjb.typed_judger_policy_args(str(tmp_path)) == no_tools_policy_args(
        str(tmp_path), tjb.TYPED_JUDGER_POLICY)
    with pytest.raises(ValueError):
        no_tools_policy_args(str(tmp_path), ToolPolicy())


def test_which_runtime_honours_which_policy():
    audited = tjb.TYPED_JUDGER_POLICY
    claude = ClaudeCodeRuntime()
    direct = DirectRuntime(_Endpoint())
    codex = CodexRuntime()
    assert claude.supports_policy(NO_TOOLS) and not claude.supports_policy(audited)
    assert direct.supports_policy(NO_TOOLS) and not direct.supports_policy(audited)
    assert codex.supports_policy(NO_TOOLS) and codex.supports_policy(audited)
    for runtime in (claude, direct, codex):
        assert not runtime.supports_policy(ToolPolicy())
    routing = RoutingRuntime(direct, codex, {"critic"})
    assert not routing.supports_policy(NO_TOOLS)
    assert routing.seat_runtime(["critic"]) is direct
    assert routing.seat_runtime(["critic", "judger"]) is codex
    assert not FixtureBackend({}).supports_policy(NO_TOOLS)


def test_boundary_capabilities_are_not_inherited():
    class Child(CodexRuntime):
        pass

    class PretendFixture(FixtureBackend):
        pass

    assert not Child().supports_policy(NO_TOOLS)
    assert Child().no_tools_identity(NO_TOOLS, "/x") == {}
    assert FixtureBackend({}).replays_fixtures
    assert not PretendFixture({}).replays_fixtures
    # ordinary members are inherited as usual
    assert Child().allows_compact_source_evidence("judger", {}) is False


def test_claude_translation_is_an_empty_palette(tmp_path):
    runtime = ClaudeCodeRuntime(model="m", permission_mode="default")
    isolated = runtime.no_tools_runtime(dict(ALL_AGENT_ROLES),
                                        _session(tmp_path))
    assert type(isolated) is ClaudeCodeRuntime
    assert isolated.cwd == str(tmp_path)
    assert (isolated.model, isolated.permission_mode) == ("m", "default")
    assert all(role.tools == [] for role in isolated.roles.values())
    assert ALL_AGENT_ROLES["discharge"].tools   # the shared roles unchanged


def test_codex_session_runs_under_its_policy(tmp_path):
    runtime = CodexRuntime(cwd=str(tmp_path / "state"), model="a;judger=b",
                           effort="high", judger_prefetch=True)
    isolated = runtime.no_tools_runtime(
        runtime.roles, replace(_session(tmp_path), evidence_prefetch=False))
    assert type(isolated) is CodexRuntime and isolated.session is not None
    assert (isolated.model, isolated.model_overrides, isolated.effort) == (
        "a", {"judger": "b"}, "high")
    assert isolated.cwd == runtime.cwd
    assert isolated._execution_cwd("judger", {}) == str(tmp_path)
    assert isolated._execution_policy_args("judger", {}) == \
        no_tools_policy_args(str(tmp_path), NO_TOOLS)
    assert not isolated._search_enabled("critic")
    assert isolated._requires_json() and not runtime._requires_json()
    assert isolated.captured_images("judger", {
        "world_mode": "captured", "basis": "world"}) == ()
    assert runtime.judger_prefetch and not isolated.judger_prefetch


def test_a_replaced_ordinary_policy_never_reaches_a_session(
        tmp_path, monkeypatch):
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return type("Done", (), {"returncode": 0, "stderr": "", "stdout": ""})

    role = ALL_AGENT_ROLES["critic"]
    runtime = CodexRuntime(
        roles={"critic": replace(role, schema={"ok": bool})},
        session=_session(tmp_path))
    monkeypatch.setattr(cb, "_run_codex_process", run)
    monkeypatch.setattr(CodexRuntime, "_execution_policy_args",
                        lambda self, role_name, variables: ["overlay"])
    runtime.complete("critic", "prompt", {})
    assert "overlay" not in commands[0]
    assert "default_permissions=\"gv-authority\"" in commands[0]


def test_duck_typed_backends_read_with_defaults():
    class Bare:
        def complete(self, role_name, prompt, variables):
            return {}

    class Declaring(Bare):
        uses_role_wire_schema = True
        compact_inspection = True

        def captured_images(self, role_name, variables):
            return ("image",)

    bare, declaring = capabilities_of(Bare()), capabilities_of(Declaring())
    assert isinstance(bare, RuntimeCapabilities)
    assert bare.run_options == RunOptions()
    assert not bare.uses_role_wire_schema
    assert bare.captured_images("judger", {}) == ()
    assert bare.allows_compact_source_evidence("judger", {}) is None
    assert not bare.allows_lazy_oracle_catalog("oracles", {})
    assert not bare.supports_policy(NO_TOOLS) and not bare.replays_fixtures
    assert declaring.run_options == RunOptions(compact_inspection=True)
    assert declaring.uses_role_wire_schema
    assert declaring.captured_images("judger", {}) == ("image",)
    runtime = CodexRuntime()
    assert capabilities_of(runtime) is runtime


def test_run_options_are_strict_bools():
    with pytest.raises(ValueError, match="compact_inspection must be a bool"):
        RunOptions(compact_inspection=1)
    with pytest.raises(ValueError, match="judger_prefetch must be a bool"):
        RunOptions(judger_prefetch="yes")
    assert CodexRuntime(compact_inspection=True).run_options == RunOptions(
        compact_inspection=True)
    routing = RoutingRuntime(DirectRuntime(_Endpoint()),
                             CodexRuntime(judger_prefetch=True), set())
    assert routing.run_options == RunOptions(judger_prefetch=True)


def test_reuse_key_names_survive_renames():
    assert runtime_identity_type(CodexRuntime()) == \
        "verifier.llm.codex_backend.CodexBackend"
    assert runtime_identity_type(ClaudeCodeRuntime()) == \
        "verifier.llm.agents.AgentBackend"
    direct = DirectRuntime(_Endpoint())
    assert runtime_identity_type(direct) == \
        "verifier.llm.direct_backend.DirectBackend"
    assert runtime_identity_type(RoutingRuntime(direct, CodexRuntime(), ())) \
        == "verifier.llm.direct_backend.RoutingBackend"
    assert runtime_identity_type(FixtureBackend({})) == \
        "verifier.llm.schema.FixtureBackend"

    class Child(CodexRuntime):
        pass
    # a subclass is a different runtime in every reuse key
    assert runtime_identity_type(Child()).endswith(".Child")
