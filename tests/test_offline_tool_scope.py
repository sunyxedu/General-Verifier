"""Operator-owned toolchain/cache scope across offline Codex roles."""
from __future__ import annotations

import os
import tomllib

import pytest

from verifier.execution.adapters import AdapterRoutingDischarger
from verifier.execution.discharge import AgentDischarger
from verifier.llm.authority_backend import _AuthorityCodexBackend
from verifier.llm.codex_backend import CodexBackend
from verifier.llm.schema import FixtureBackend, RoleError
from verifier.mcp_server import Service, _default_kernel
from verifier import offline


def _config(args):
    values = [args[index + 1] for index, item in enumerate(args)
              if item == "-c"]
    return tomllib.loads("\n".join(values))


def _policy(backend, role, **variables):
    return _config(backend._execution_policy_args(
        role, {"world_mode": "captured", **variables}))


def _filesystem(config):
    return config["permissions"]["gv-offline"]["filesystem"]


def _path(config):
    return config["shell_environment_policy"]["set"]["PATH"].split(
        os.pathsep)


def test_cold_toolchain_is_derived_with_judger_read_override_and_role_path(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv("GV_OFFLINE_RUNTIME_READ_ROOTS", raising=False)
    state = tmp_path / "state"
    state.mkdir()
    backend = CodexBackend(cwd=os.fspath(state))
    toolchain = os.fspath(state / "work" / "_toolchains")

    discharge = _policy(backend, "discharge")
    judger = _policy(backend, "judger")
    oracle = _policy(backend, "oracles")

    # The parent work grant already makes the cold toolchain writable for the
    # executor. The Judger receives a more-specific read-only override.
    assert toolchain not in _filesystem(discharge)
    assert _filesystem(judger)[toolchain] == "read"
    # Oracle receives no toolchain-specific grant or PATH entry. Because a
    # cold toolchain is nested below the pre-existing state/work write root,
    # that is not a claim that the directory is hidden from this role.
    assert toolchain not in _filesystem(oracle)
    assert _path(discharge)[0] == os.path.join(toolchain, "bin")
    assert _path(judger)[0] == os.path.join(toolchain, "bin")
    assert _path(oracle) == ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    for config in (discharge, judger, oracle):
        env = config["shell_environment_policy"]["set"]
        assert "UV_CACHE_DIR" not in env
        assert "PIP_CACHE_DIR" not in env


def test_shared_scope_is_write_for_discharge_read_for_judger_and_hidden_from_oracle(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv("GV_OFFLINE_RUNTIME_READ_ROOTS", raising=False)
    state = tmp_path / "state"
    toolchain = tmp_path / "shared" / "toolchain"
    cache = tmp_path / "shared" / "cache"
    state.mkdir()
    toolchain.mkdir(parents=True)
    cache.mkdir(parents=True)
    backend = CodexBackend(
        cwd=os.fspath(state),
        offline_toolchain_dir=os.fspath(toolchain),
        offline_shared_cache_dir=os.fspath(cache),
    )

    discharge = _policy(
        backend, "discharge", toolchain_dir=os.fspath(toolchain),
        shared_cache_dir=os.fspath(cache))
    judger = _policy(backend, "judger")
    oracle = _policy(backend, "oracles")

    assert _filesystem(discharge)[os.fspath(toolchain)] == "write"
    assert _filesystem(discharge)[os.fspath(cache)] == "write"
    assert _filesystem(judger)[os.fspath(toolchain)] == "read"
    assert os.fspath(cache) not in _filesystem(judger)
    assert os.fspath(toolchain) not in _filesystem(oracle)
    assert os.fspath(cache) not in _filesystem(oracle)
    assert _path(discharge)[0] == os.fspath(toolchain / "bin")
    assert _path(judger)[0] == os.fspath(toolchain / "bin")
    assert os.fspath(toolchain / "bin") not in _path(oracle)
    discharge_env = discharge["shell_environment_policy"]["set"]
    assert discharge_env["UV_CACHE_DIR"] == os.fspath(cache / "uv")
    assert discharge_env["PIP_CACHE_DIR"] == os.fspath(cache / "pip")
    judger_env = judger["shell_environment_policy"]["set"]
    assert "UV_CACHE_DIR" not in judger_env
    assert "PIP_CACHE_DIR" not in judger_env


@pytest.mark.parametrize("field", ["toolchain_dir", "shared_cache_dir"])
def test_role_variables_cannot_replace_operator_scope(
    tmp_path, monkeypatch, field,
):
    monkeypatch.delenv("GV_OFFLINE_RUNTIME_READ_ROOTS", raising=False)
    state = tmp_path / "state"
    toolchain = tmp_path / "shared" / "toolchain"
    cache = tmp_path / "shared" / "cache"
    replacement = tmp_path / "replacement"
    for path in (state, toolchain, cache, replacement):
        path.mkdir(parents=True)
    backend = CodexBackend(
        cwd=os.fspath(state),
        offline_toolchain_dir=os.fspath(toolchain),
        offline_shared_cache_dir=os.fspath(cache),
    )

    with pytest.raises(RoleError, match="operator-owned scope"):
        _policy(backend, "discharge", **{field: os.fspath(replacement)})


@pytest.mark.parametrize("cache_relative", [".", "nested/cache", "../"])
def test_toolchain_and_cache_must_not_overlap(
    tmp_path, monkeypatch, cache_relative,
):
    monkeypatch.delenv("GV_OFFLINE_RUNTIME_READ_ROOTS", raising=False)
    state = tmp_path / "state"
    toolchain = tmp_path / "shared" / "toolchain"
    cache = toolchain / cache_relative
    state.mkdir()
    toolchain.mkdir(parents=True)
    backend = CodexBackend(
        cwd=os.fspath(state),
        offline_toolchain_dir=os.fspath(toolchain),
        offline_shared_cache_dir=os.fspath(cache),
    )

    with pytest.raises(RoleError, match="must be disjoint"):
        _policy(backend, "discharge")


def test_offline_kernel_rejects_service_backend_scope_mismatch(tmp_path):
    state = tmp_path / "state"
    service_toolchain = tmp_path / "service" / "toolchain"
    backend_toolchain = tmp_path / "backend" / "toolchain"
    for path in (service_toolchain, backend_toolchain):
        path.mkdir(parents=True)
    service = Service(
        os.fspath(state), live_world=False,
        toolchain_dir=os.fspath(service_toolchain),
        kernel_factory=lambda _service: object(),
    )
    backend = CodexBackend(
        cwd=os.fspath(state),
        offline_toolchain_dir=os.fspath(backend_toolchain),
    )

    with pytest.raises(RoleError, match="Service toolchain does not match"):
        _default_kernel(service, model=None, backend=backend, critic_rounds=0)


def test_offline_kernel_rejects_service_backend_state_mismatch(tmp_path):
    service_state = tmp_path / "service-state"
    backend_state = tmp_path / "backend-state"
    service = Service(
        os.fspath(service_state), live_world=False,
        kernel_factory=lambda _service: object(),
    )
    backend = CodexBackend(cwd=os.fspath(backend_state))

    with pytest.raises(RoleError, match="state does not match"):
        _default_kernel(service, model=None, backend=backend, critic_rounds=0)


def test_offline_score_rejects_unbound_shared_scope_before_model_call(
    tmp_path,
):
    state = tmp_path / "state"
    shared = tmp_path / "shared"
    different = tmp_path / "different" / "toolchain"
    state.mkdir()
    different.mkdir(parents=True)
    backend = CodexBackend(
        cwd=os.fspath(state),
        offline_toolchain_dir=os.fspath(different),
    )

    with pytest.raises(RoleError, match="operator-owned scope"):
        offline.score(
            os.fspath(state), seats="codex", shared_dir=os.fspath(shared),
            backend=backend,
        )


def test_offline_kernel_gives_discharger_the_exact_bound_scope(tmp_path):
    state = tmp_path / "state"
    toolchain = tmp_path / "shared" / "toolchain"
    cache = tmp_path / "shared" / "cache"
    toolchain.mkdir(parents=True)
    cache.mkdir(parents=True)
    service = Service(
        os.fspath(state), live_world=False,
        toolchain_dir=os.fspath(toolchain),
        shared_cache_dir=os.fspath(cache),
        kernel_factory=lambda _service: object(),
    )
    backend = CodexBackend(
        cwd=os.fspath(state),
        offline_toolchain_dir=os.fspath(toolchain),
        offline_shared_cache_dir=os.fspath(cache),
    )

    kernel = _default_kernel(
        service, model=None, backend=backend, critic_rounds=0)
    discharger = kernel.discharger
    if isinstance(discharger, AdapterRoutingDischarger):
        discharger = discharger.fallback

    assert isinstance(discharger, AgentDischarger)
    assert discharger.toolchain_dir == os.fspath(toolchain)
    assert discharger.shared_cache_dir == os.fspath(cache)


def test_custom_backend_keeps_existing_service_scope_contract(tmp_path):
    toolchain = tmp_path / "shared" / "toolchain"
    cache = tmp_path / "shared" / "cache"
    toolchain.mkdir(parents=True)
    cache.mkdir(parents=True)
    service = Service(
        os.fspath(tmp_path / "state"), live_world=False,
        toolchain_dir=os.fspath(toolchain),
        shared_cache_dir=os.fspath(cache),
        kernel_factory=lambda _service: object(),
    )

    kernel = _default_kernel(
        service, model=None, backend=FixtureBackend({}), critic_rounds=0)
    discharger = kernel.discharger
    if isinstance(discharger, AdapterRoutingDischarger):
        discharger = discharger.fallback

    assert discharger.toolchain_dir == os.fspath(toolchain)
    assert discharger.shared_cache_dir == os.fspath(cache)


def test_live_policy_ignores_offline_operator_scope_and_role_variables(tmp_path):
    toolchain = tmp_path / "shared" / "toolchain"
    cache = tmp_path / "shared" / "cache"
    replacement = tmp_path / "replacement"
    for path in (toolchain, cache, replacement):
        path.mkdir(parents=True)
    backend = CodexBackend(
        cwd=os.fspath(tmp_path),
        offline_toolchain_dir=os.fspath(toolchain),
        offline_shared_cache_dir=os.fspath(cache),
    )

    assert backend._execution_policy_args("judger", {
        "world_mode": "live", "toolchain_dir": os.fspath(replacement),
        "shared_cache_dir": os.fspath(replacement),
    }) == ["--dangerously-bypass-approvals-and-sandbox"]


def test_authority_backend_never_inherits_operator_tool_scope(tmp_path):
    toolchain = tmp_path / "shared" / "toolchain"
    cache = tmp_path / "shared" / "cache"
    scratch = tmp_path / "authority"
    toolchain.mkdir(parents=True)
    cache.mkdir(parents=True)
    scratch.mkdir()
    backend = _AuthorityCodexBackend(
        cwd=os.fspath(tmp_path),
        offline_toolchain_dir=os.fspath(toolchain),
        offline_shared_cache_dir=os.fspath(cache),
    )
    backend._authority_scratch = os.fspath(scratch)

    config = _config(backend._execution_policy_args("obligations", {}))
    filesystem = config["permissions"]["gv-authority"]["filesystem"]

    assert os.fspath(toolchain) not in filesystem
    assert os.fspath(cache) not in filesystem
    assert filesystem[os.fspath(scratch)] == "read"
