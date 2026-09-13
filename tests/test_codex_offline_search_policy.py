"""Offline web-search policy is explicit in argv, without real model calls."""
import json
import tomllib
from types import SimpleNamespace

import pytest

from verifier.llm.codex_backend import CodexBackend
from verifier.llm import codex_backend
from verifier.profiles.cyber.backend import CyberCodexBackend
from experiments.terminal_bench.eval_offline_policy import (
    OfflinePolicyBinding, eval_offline_policy, eval_online_policy,
)


@pytest.fixture(autouse=True)
def forbid_real_processes(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("real subprocess/model invocation is forbidden")

    monkeypatch.setattr("verifier.llm.codex_backend.subprocess.Popen", forbidden)


def config_values(argv):
    return [argv[index + 1] for index, value in enumerate(argv[:-1])
            if value == "-c"]


def complete_argv(backend, role, monkeypatch, variables):
    observed = []

    def fake_run(argv, **_kwargs):
        observed.append(list(argv))
        with open(argv[argv.index("-o") + 1], "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("verifier.llm.codex_backend._run_codex_process", fake_run)
    assert backend.complete(role, "unchanged generic instruction", variables) == {"ok": True}
    assert len(observed) == 1
    return observed[0]


@pytest.mark.parametrize("role", ["discharge", "judger"])
@pytest.mark.parametrize("world_mode", ["captured", "unavailable"])
def test_disabled_offline_plain_roles_have_exactly_one_top_level_override(
    tmp_path, monkeypatch, role, world_mode,
):
    backend = CodexBackend(cwd=str(tmp_path), roles={
        role: SimpleNamespace(system="", schema={"ok": bool}),
    })
    argv = complete_argv(backend, role, monkeypatch, {"world_mode": world_mode})
    values = config_values(argv)
    assert "--ignore-user-config" in argv
    assert values.count('web_search="disabled"') == 1
    assert values.count("tools.web_search=false") == 1
    assert "tools.web_search=true" not in values
    config = tomllib.loads("\n".join(value for value in values
                                    if value.startswith(("web_search=", "tools.web_search="))))
    assert config["web_search"] == "disabled"
    assert config["tools"]["web_search"] is False


@pytest.mark.parametrize("role", ["obligations", "critic", "oracles", "extract"])
def test_enabled_plain_roles_preserve_previous_complete_argv(
    tmp_path, monkeypatch, role,
):
    backend = CodexBackend(cwd=str(tmp_path), roles={
        role: SimpleNamespace(system="", schema={"ok": bool}),
    })
    variables = {"world_mode": "captured"}
    assert backend._search_enabled(role)
    policy = backend._execution_policy_args(role, variables)
    assert not any(value.startswith("web_search=") for value in config_values(policy))
    argv = complete_argv(backend, role, monkeypatch, variables)
    # Existing enabled argv sets the legacy flag before the unchanged policy.
    assert argv[5:7] == ["-c", "tools.web_search=true"]
    policy_index = argv.index("--ignore-user-config")
    assert argv[policy_index:policy_index + len(policy)] == policy
    assert config_values(argv).count("tools.web_search=true") == 2
    assert "tools.web_search=false" not in config_values(argv)
    assert not any(value.startswith("web_search=") for value in config_values(argv))


@pytest.mark.parametrize("role", [
    "extract", "obligations", "critic", "oracles", "sufficiency", "discharge", "judger",
])
def test_every_cyber_role_disables_web_in_complete_argv(tmp_path, monkeypatch, role):
    backend = CyberCodexBackend(cwd=str(tmp_path), roles={
        role: SimpleNamespace(system="", schema={"ok": bool}),
    })
    assert not backend.search_roles
    argv = complete_argv(backend, role, monkeypatch, {"world_mode": "live"})
    values = config_values(argv)
    assert "--ignore-user-config" in argv
    assert values.count('web_search="disabled"') == 1
    assert values.count("tools.web_search=false") == 1
    assert "tools.web_search=true" not in values
    assert "WEB SEARCH (knowledge channel only)" not in argv[-1]


@pytest.mark.parametrize("role", ["discharge", "judger", "obligations"])
def test_plain_live_policy_is_unchanged(tmp_path, role):
    assert CodexBackend(cwd=str(tmp_path))._execution_policy_args(
        role, {"world_mode": "live"},
    ) == ["--dangerously-bypass-approvals-and-sandbox"]


@pytest.mark.parametrize("overlay,world_mode", [
    (eval_offline_policy, "captured"), (eval_online_policy, "live"),
])
@pytest.mark.parametrize("role", [
    "extract", "obligations", "critic", "oracles", "discharge", "judger",
])
def test_eval_adapter_complete_argv_matches_native_offline_search_policy(
    tmp_path, monkeypatch, overlay, world_mode, role,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    backend = CodexBackend(cwd=str(tmp_path), roles={
        role: SimpleNamespace(system="", schema={"ok": bool}),
    })
    expected = backend._execution_policy_args(role, {"world_mode": "captured"})
    with overlay(codex_backend, bindings=[OfflinePolicyBinding(state_dir=str(tmp_path))]):
        assert backend._execution_policy_args(role, {"world_mode": world_mode}) == expected
        argv = complete_argv(backend, role, monkeypatch, {"world_mode": world_mode})
    values = config_values(argv)
    if backend._search_enabled(role):
        assert not any(value.startswith("web_search=") for value in values)
        assert values.count("tools.web_search=true") == 2
        assert "tools.web_search=false" not in values
    else:
        assert values.count('web_search="disabled"') == 1
        assert values.count("tools.web_search=false") == 1
        assert "tools.web_search=true" not in values
