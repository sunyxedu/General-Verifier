"""Tests for the source-comparison offline permission overlay."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tomllib
import uuid

import pytest

from experiments.terminal_bench.eval_offline_policy import (
    ONLINE_POLICY_IDENTITY_SCHEMA,
    ONLINE_POLICY_PROTOCOL,
    POLICY_IDENTITY_SCHEMA,
    POLICY_PROTOCOL,
    EvalOfflinePolicyError,
    EvalOnlinePolicyPlan,
    OfflinePolicyBinding,
    apply_eval_offline_policy,
    apply_eval_online_policy,
    eval_offline_policy,
    eval_online_policy,
    plan_eval_offline_policy,
    plan_eval_online_policy,
)
from verifier.llm import codex_backend
from verifier.llm.codex_backend import CodexRuntime
from verifier.llm.schema import RoleError


def _config(arguments):
    values = [
        arguments[index + 1]
        for index, value in enumerate(arguments)
        if value == "-c"
    ]
    return tomllib.loads("\n".join(values))


def _legacy_module(tmp_path):
    name = "legacy_codex_backend_" + uuid.uuid4().hex
    path = tmp_path / f"{name}.py"
    path.write_text(
        '''\
import os

_OFFLINE_WORLD_MODES = frozenset({"captured", "unavailable"})
_SEARCH_ROLES = frozenset({"extract", "obligations", "critic", "oracles"})

class RoleError(RuntimeError):
    pass

class CodexBackend:
    def __init__(self, cwd=None):
        self.cwd = cwd

    def _is_offline_execution(self, role_name, variables):
        return bool(
            variables.get("world_mode") in _OFFLINE_WORLD_MODES
            or (self.cwd is not None and role_name == "extract")
        )

    def _execution_policy_args(self, role_name, variables):
        if not self._is_offline_execution(role_name, variables):
            return ["--dangerously-bypass-approvals-and-sandbox"]
        return ["legacy-policy"]
''',
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _binding(state, **kwargs):
    return OfflinePolicyBinding(state_dir=os.fspath(state), **kwargs)


def test_plan_is_read_only_and_records_source_method_and_binding_identity(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv("GV_OFFLINE_RUNTIME_READ_ROOTS", raising=False)
    state = tmp_path / "state"
    state.mkdir()
    binding = _binding(state)

    plan = plan_eval_offline_policy(codex_backend, bindings=[binding])

    assert not (state / "work").exists()
    identity = plan.identity
    assert identity["schema"] == POLICY_IDENTITY_SCHEMA
    assert identity["protocol"] == POLICY_PROTOCOL
    assert identity["backend_module"] == codex_backend.__name__
    assert identity["scope"] == "exact-class-offline-only"
    assert identity["binding_count"] == 1
    for key in (
        "backend_source_sha256",
        "original_method_source_sha256",
        "original_method_code_sha256",
        "overlay_source_sha256",
        "policy_method_source_sha256",
        "policy_method_code_sha256",
        "bindings_sha256",
    ):
        assert len(identity[key]) == 64
        int(identity[key], 16)
    identity["protocol"] = "caller mutation"
    assert plan.identity["protocol"] == POLICY_PROTOCOL
    assert not (state / "work").exists()


@pytest.mark.parametrize(
    ("role", "variables"),
    [
        ("extract", {}),
        ("obligations", {"world_mode": "captured"}),
        ("critic", {"world_mode": "unavailable"}),
        ("oracles", {"world_mode": "captured"}),
        ("judger", {"world_mode": "captured"}),
        ("discharge", {"world_mode": "captured"}),
    ],
)
def test_current_backend_cold_policy_has_byte_for_byte_parity(
    tmp_path, monkeypatch, role, variables,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    state.mkdir()
    backend = CodexRuntime(cwd=os.fspath(state))
    expected = backend._execution_policy_args(role, dict(variables))
    plan = plan_eval_offline_policy(
        codex_backend, bindings=[_binding(state)])

    with apply_eval_offline_policy(plan):
        actual = backend._execution_policy_args(role, dict(variables))

    assert actual == expected


def test_current_external_roots_workspace_and_dynamic_workdir_have_parity(
    tmp_path, monkeypatch,
):
    state = tmp_path / "state"
    toolchain = tmp_path / "shared" / "toolchain"
    shared_cache = tmp_path / "shared" / "cache"
    runtime_bin = tmp_path / "runtime" / "bin"
    state.mkdir()
    toolchain.mkdir(parents=True)
    shared_cache.mkdir(parents=True)
    runtime_bin.mkdir(parents=True)
    monkeypatch.setenv(
        "GV_OFFLINE_RUNTIME_READ_ROOTS", json.dumps([os.fspath(runtime_bin)]))
    backend = CodexRuntime(
        cwd=os.fspath(state),
        offline_toolchain_dir=os.fspath(toolchain),
        offline_shared_cache_dir=os.fspath(shared_cache),
    )
    variables = {
        "world_mode": "captured",
        "workdir": os.fspath(state / "work" / "one"),
        "workspace_dir": os.fspath(state / "workspace"),
        "toolchain_dir": os.fspath(toolchain),
        "shared_cache_dir": os.fspath(shared_cache),
    }
    expected = backend._execution_policy_args("discharge", variables)
    binding = _binding(
        state,
        toolchain_dir=os.fspath(toolchain),
        shared_cache_dir=os.fspath(shared_cache),
        runtime_read_roots=(os.fspath(runtime_bin),),
    )

    with eval_offline_policy(codex_backend, bindings=[binding]):
        actual = backend._execution_policy_args("discharge", variables)

    assert actual == expected


@pytest.mark.parametrize("role", ["extract", "oracles", "discharge", "judger"])
def test_legacy_and_current_modules_receive_the_same_overlay_policy(
    tmp_path, monkeypatch, role,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    state.mkdir()
    variables = {} if role == "extract" else {"world_mode": "captured"}
    binding = _binding(state)
    current = CodexRuntime(cwd=os.fspath(state))
    with eval_offline_policy(codex_backend, bindings=[binding]):
        current_args = current._execution_policy_args(role, variables)

    legacy = _legacy_module(tmp_path)
    try:
        old_backend = legacy.CodexBackend(cwd=os.fspath(state))
        with eval_offline_policy(legacy, bindings=[binding]):
            legacy_args = old_backend._execution_policy_args(role, variables)
    finally:
        sys.modules.pop(legacy.__name__, None)

    assert legacy_args == current_args


def test_role_permissions_match_current_cold_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    state.mkdir()
    backend = CodexRuntime(cwd=os.fspath(state))
    toolchain = os.fspath(state / "work" / "_toolchains")

    with eval_offline_policy(codex_backend, bindings=[_binding(state)]):
        discharge = _config(backend._execution_policy_args(
            "discharge", {"world_mode": "captured"}))
        judger = _config(backend._execution_policy_args(
            "judger", {"world_mode": "captured"}))
        oracle = _config(backend._execution_policy_args(
            "oracles", {"world_mode": "captured"}))

    discharge_fs = discharge["permissions"]["gv-offline"]["filesystem"]
    judger_fs = judger["permissions"]["gv-offline"]["filesystem"]
    oracle_fs = oracle["permissions"]["gv-offline"]["filesystem"]
    assert discharge_fs[":root"] == "deny"
    assert discharge_fs[os.fspath(state)] == "read"
    assert discharge_fs[os.fspath(state / "work")] == "write"
    assert toolchain not in discharge_fs
    assert judger_fs[toolchain] == "read"
    assert toolchain not in oracle_fs
    assert discharge["shell_environment_policy"]["set"]["PATH"].split(
        os.pathsep)[0] == os.path.join(toolchain, "bin")
    assert judger["shell_environment_policy"]["set"]["PATH"].split(
        os.pathsep)[0] == os.path.join(toolchain, "bin")
    assert oracle["shell_environment_policy"]["set"]["PATH"].split(
        os.pathsep) == ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]


def test_external_cache_is_discharge_only_and_runtime_is_read_only(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    toolchain = tmp_path / "shared" / "toolchain"
    cache = tmp_path / "shared" / "cache"
    runtime_bin = tmp_path / "runtime" / "bin"
    for path in (state, toolchain, cache, runtime_bin):
        path.mkdir(parents=True)
    binding = _binding(
        state,
        toolchain_dir=os.fspath(toolchain),
        shared_cache_dir=os.fspath(cache),
        runtime_read_roots=(os.fspath(runtime_bin),),
    )
    backend = CodexRuntime(
        cwd=os.fspath(state),
        offline_toolchain_dir=os.fspath(toolchain),
        offline_shared_cache_dir=os.fspath(cache),
    )

    with eval_offline_policy(codex_backend, bindings=[binding]):
        policies = {
            role: _config(backend._execution_policy_args(
                role, {"world_mode": "captured"}))
            for role in ("discharge", "judger", "oracles")
        }

    filesystems = {
        role: value["permissions"]["gv-offline"]["filesystem"]
        for role, value in policies.items()
    }
    assert filesystems["discharge"][os.fspath(toolchain)] == "write"
    assert filesystems["discharge"][os.fspath(cache)] == "write"
    assert filesystems["judger"][os.fspath(toolchain)] == "read"
    assert os.fspath(cache) not in filesystems["judger"]
    assert os.fspath(toolchain) not in filesystems["oracles"]
    assert os.fspath(cache) not in filesystems["oracles"]
    for filesystem in filesystems.values():
        assert filesystem[os.fspath(runtime_bin)] == "read"
    discharge_env = policies["discharge"]["shell_environment_policy"]["set"]
    assert discharge_env["UV_CACHE_DIR"] == os.fspath(cache / "uv")
    assert "UV_CACHE_DIR" not in policies["judger"][
        "shell_environment_policy"]["set"]


@pytest.mark.parametrize("key", ["toolchain_dir", "shared_cache_dir"])
def test_candidate_cannot_replace_operator_roots(tmp_path, monkeypatch, key):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    replacement = tmp_path / "replacement"
    state.mkdir()
    replacement.mkdir()
    backend = CodexRuntime(cwd=os.fspath(state))

    with eval_offline_policy(codex_backend, bindings=[_binding(state)]):
        with pytest.raises(RoleError, match="operator binding"):
            backend._execution_policy_args("discharge", {
                "world_mode": "captured", key: os.fspath(replacement),
            })


def test_candidate_workspace_must_be_exact_but_workdir_may_vary_under_work(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    state.mkdir()
    backend = CodexRuntime(cwd=os.fspath(state))
    workdir = state / "work" / "claim" / "attempt"
    workspace = state / "workspace"

    with eval_offline_policy(codex_backend, bindings=[_binding(state)]):
        config = _config(backend._execution_policy_args("discharge", {
            "world_mode": "captured",
            "workdir": os.fspath(workdir),
            "workspace_dir": os.fspath(workspace),
        }))
        with pytest.raises(RoleError, match="does not match state/workspace"):
            backend._execution_policy_args("discharge", {
                "world_mode": "captured",
                "workspace_dir": os.fspath(workspace / "candidate-choice"),
            })
        with pytest.raises(RoleError, match="escapes state/work"):
            backend._execution_policy_args("discharge", {
                "world_mode": "captured",
                "workdir": os.fspath(state / "other"),
            })

    filesystem = config["permissions"]["gv-offline"]["filesystem"]
    assert filesystem[os.fspath(workspace)] == "write"
    assert workdir.is_dir()


def test_live_unknown_state_cwdless_and_subclass_fail_before_policy_output(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    other = tmp_path / "other"
    state.mkdir()
    other.mkdir()
    plan = plan_eval_offline_policy(
        codex_backend, bindings=[_binding(state)])

    class Child(CodexRuntime):
        pass

    with apply_eval_offline_policy(plan):
        with pytest.raises(RoleError, match="live or unclassified"):
            CodexRuntime(cwd=os.fspath(state))._execution_policy_args(
                "critic", {"world_mode": "live"})
        with pytest.raises(RoleError, match="no eval offline policy binding"):
            CodexRuntime(cwd=os.fspath(other))._execution_policy_args(
                "critic", {"world_mode": "captured"})
        with pytest.raises(RoleError, match="requires an operator-bound"):
            CodexRuntime()._execution_policy_args(
                "critic", {"world_mode": "captured"})
        with pytest.raises(RoleError, match="exact CodexBackend class"):
            Child(cwd=os.fspath(state))._execution_policy_args(
                "critic", {"world_mode": "captured"})


def test_current_constructor_roots_must_match_overlay_binding(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    external = tmp_path / "external" / "toolchain"
    state.mkdir()
    external.mkdir(parents=True)
    backend = CodexRuntime(
        cwd=os.fspath(state), offline_toolchain_dir=os.fspath(external))

    with eval_offline_policy(codex_backend, bindings=[_binding(state)]):
        with pytest.raises(RoleError, match="backend toolchain disagrees"):
            backend._execution_policy_args(
                "discharge", {"world_mode": "captured"})


def test_overlay_restores_original_and_refuses_nested_install(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    state.mkdir()
    plan = plan_eval_offline_policy(
        codex_backend, bindings=[_binding(state)])
    original = CodexRuntime.__dict__["_execution_policy_args"]

    with apply_eval_offline_policy(plan):
        assert CodexRuntime.__dict__["_execution_policy_args"] is not original
        with pytest.raises(EvalOfflinePolicyError, match="already installed"):
            with apply_eval_offline_policy(plan):
                pass

    assert CodexRuntime.__dict__["_execution_policy_args"] is original
    assert CodexRuntime(cwd=os.fspath(state))._execution_policy_args(
        "critic", {"world_mode": "live"},
    ) == ["--dangerously-bypass-approvals-and-sandbox"]


def test_body_exception_is_preserved_and_method_is_restored(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    state.mkdir()
    original = CodexRuntime.__dict__["_execution_policy_args"]

    with pytest.raises(KeyboardInterrupt, match="operator stop"):
        with eval_offline_policy(codex_backend, bindings=[_binding(state)]):
            raise KeyboardInterrupt("operator stop")

    assert CodexRuntime.__dict__["_execution_policy_args"] is original


def test_plan_rejects_method_changed_before_install(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    state.mkdir()
    plan = plan_eval_offline_policy(
        codex_backend, bindings=[_binding(state)])
    original = CodexRuntime.__dict__["_execution_policy_args"]

    def replacement(self, role_name, variables):
        return []

    monkeypatch.setattr(CodexRuntime, "_execution_policy_args", replacement)
    with pytest.raises(EvalOfflinePolicyError, match="changed after"):
        with apply_eval_offline_policy(plan):
            pass
    monkeypatch.setattr(CodexRuntime, "_execution_policy_args", original)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state_dir": "relative"},
        {"runtime_read_roots": ("relative",)},
        {"runtime_read_roots": (os.path.sep,)},
    ],
)
def test_binding_rejects_ambiguous_or_broad_paths(tmp_path, kwargs):
    state = tmp_path / "state"
    state.mkdir()
    values = {"state_dir": os.fspath(state), **kwargs}
    with pytest.raises(EvalOfflinePolicyError):
        OfflinePolicyBinding(**values)


def test_binding_rejects_symlinked_state_without_creating_work(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(state, target_is_directory=True)

    with pytest.raises(EvalOfflinePolicyError, match="symlink"):
        _binding(alias)
    assert not (state / "work").exists()


def test_binding_rejects_invalid_external_authority_before_creating_work(
    tmp_path,
):
    state = tmp_path / "state"
    state.mkdir()

    with pytest.raises(EvalOfflinePolicyError, match="existing directory"):
        _binding(state, toolchain_dir=os.fspath(tmp_path / "missing"))
    with pytest.raises(EvalOfflinePolicyError, match="outside state/work"):
        _binding(state, toolchain_dir=os.fspath(state / "workspace"))
    with pytest.raises(EvalOfflinePolicyError, match="overlap state_dir"):
        _binding(
            state,
            runtime_read_roots=(os.fspath(state),),
        )
    assert not (state / "work").exists()


def test_online_plan_is_explicit_read_only_and_has_distinct_identity(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    state.mkdir()

    plan = plan_eval_online_policy(
        codex_backend, bindings=[_binding(state)])

    assert isinstance(plan, EvalOnlinePolicyPlan)
    assert not (state / "work").exists()
    assert plan.identity["schema"] == ONLINE_POLICY_IDENTITY_SCHEMA
    assert plan.identity["protocol"] == ONLINE_POLICY_PROTOCOL
    assert plan.identity["scope"] == "exact-class-live-only"
    assert plan.identity["binding_count"] == 1
    for key in (
        "backend_source_sha256",
        "original_method_source_sha256",
        "original_method_code_sha256",
        "overlay_source_sha256",
        "policy_method_source_sha256",
        "policy_method_code_sha256",
        "bindings_sha256",
    ):
        assert len(plan.identity[key]) == 64
        int(plan.identity[key], 16)


@pytest.mark.parametrize(
    "role", ["extract", "obligations", "critic", "oracles", "discharge", "judger"],
)
def test_online_and_offline_entry_points_render_the_same_isolation_bytes(
    tmp_path, monkeypatch, role,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    state.mkdir()
    binding = _binding(state)
    backend = CodexRuntime(cwd=os.fspath(state))

    with eval_offline_policy(codex_backend, bindings=[binding]) as offline_plan:
        offline = backend._execution_policy_args(
            role, {"world_mode": "captured"})
    with eval_online_policy(codex_backend, bindings=[binding]) as online_plan:
        online = backend._execution_policy_args(role, {"world_mode": "live"})

    assert offline == online
    assert offline_plan.identity["schema"] == POLICY_IDENTITY_SCHEMA
    assert offline_plan.identity["protocol"] == POLICY_PROTOCOL
    assert offline_plan.identity["scope"] == "exact-class-offline-only"
    assert online_plan.identity["schema"] == ONLINE_POLICY_IDENTITY_SCHEMA


@pytest.mark.parametrize(
    "variables",
    [
        {},
        {"world_mode": "captured"},
        {"world_mode": "unavailable"},
        {"world_mode": "LIVE"},
        {"world_mode": None},
    ],
)
def test_online_api_requires_exact_live_world_mode(
    tmp_path, monkeypatch, variables,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    state.mkdir()
    backend = CodexRuntime(cwd=os.fspath(state))

    with eval_online_policy(codex_backend, bindings=[_binding(state)]):
        with pytest.raises(RoleError, match="requires world_mode='live'"):
            backend._execution_policy_args("critic", variables)


def test_online_api_rejects_string_subclass_world_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    state.mkdir()
    backend = CodexRuntime(cwd=os.fspath(state))

    class Live(str):
        pass

    with eval_online_policy(codex_backend, bindings=[_binding(state)]):
        with pytest.raises(RoleError, match="requires world_mode='live'"):
            backend._execution_policy_args(
                "critic", {"world_mode": Live("live")})


def test_online_api_requires_bound_exact_backend_and_cwd(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    other = tmp_path / "other"
    state.mkdir()
    other.mkdir()
    plan = plan_eval_online_policy(
        codex_backend, bindings=[_binding(state)])

    class Child(CodexRuntime):
        pass

    with apply_eval_online_policy(plan):
        with pytest.raises(RoleError, match="no eval offline policy binding"):
            CodexRuntime(cwd=os.fspath(other))._execution_policy_args(
                "critic", {"world_mode": "live"})
        with pytest.raises(RoleError, match="operator-bound backend cwd"):
            CodexRuntime()._execution_policy_args(
                "critic", {"world_mode": "live"})
        with pytest.raises(RoleError, match="exact CodexBackend class"):
            Child(cwd=os.fspath(state))._execution_policy_args(
                "critic", {"world_mode": "live"})


@pytest.mark.parametrize("key", ["toolchain_dir", "shared_cache_dir"])
def test_online_candidate_cannot_replace_operator_roots(
    tmp_path, monkeypatch, key,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    replacement = tmp_path / "replacement"
    state.mkdir()
    replacement.mkdir()
    backend = CodexRuntime(cwd=os.fspath(state))

    with eval_online_policy(codex_backend, bindings=[_binding(state)]):
        with pytest.raises(RoleError, match="operator binding"):
            backend._execution_policy_args("discharge", {
                "world_mode": "live", key: os.fspath(replacement),
            })
        with pytest.raises(RoleError, match="escapes state/work"):
            backend._execution_policy_args("discharge", {
                "world_mode": "live",
                "workdir": os.fspath(state.parent / "candidate-work"),
            })


def test_offline_api_still_rejects_live_after_online_api_is_added(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    state.mkdir()
    backend = CodexRuntime(cwd=os.fspath(state))
    original = CodexRuntime.__dict__["_execution_policy_args"]

    with eval_offline_policy(codex_backend, bindings=[_binding(state)]):
        with pytest.raises(RoleError, match="refuses live"):
            backend._execution_policy_args("critic", {"world_mode": "live"})

    assert CodexRuntime.__dict__["_execution_policy_args"] is original


def test_online_overlay_restores_after_body_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_OFFLINE_RUNTIME_READ_ROOTS", "[]")
    state = tmp_path / "state"
    state.mkdir()
    original = CodexRuntime.__dict__["_execution_policy_args"]

    with pytest.raises(KeyboardInterrupt, match="operator stop"):
        with eval_online_policy(codex_backend, bindings=[_binding(state)]):
            assert CodexRuntime.__dict__["_execution_policy_args"] is not original
            raise KeyboardInterrupt("operator stop")

    assert CodexRuntime.__dict__["_execution_policy_args"] is original
    assert CodexRuntime(cwd=os.fspath(state))._execution_policy_args(
        "critic", {"world_mode": "live"},
    ) == ["--dangerously-bypass-approvals-and-sandbox"]
