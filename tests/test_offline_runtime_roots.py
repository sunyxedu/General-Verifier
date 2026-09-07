"""Operator-owned offline runtime grants; no model calls."""
from __future__ import annotations

import json
import os
import tomllib

import pytest

from verifier.execution.config import (
    OFFLINE_RUNTIME_READ_ROOTS_ENV,
    offline_runtime_read_roots,
)
from verifier.llm.authority_backend import _AuthorityCodexBackend
from verifier.llm.codex_backend import CodexBackend
from verifier.llm.schema import RoleError


def _cli_config(args):
    values = [args[index + 1] for index, arg in enumerate(args)
              if arg == "-c"]
    return tomllib.loads("\n".join(values))


def _set_roots(monkeypatch, *roots):
    monkeypatch.setenv(
        OFFLINE_RUNTIME_READ_ROOTS_ENV,
        json.dumps([os.fspath(root) for root in roots]),
    )


def test_runtime_roots_default_to_empty():
    assert offline_runtime_read_roots({}) == ()


def test_runtime_roots_preserve_canonical_operator_order(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    roots = offline_runtime_read_roots({
        OFFLINE_RUNTIME_READ_ROOTS_ENV: json.dumps([
            os.fspath(second), os.fspath(first),
        ]),
    }, home=os.path.join(os.sep, "unrelated-home"))

    assert roots == (os.fspath(second), os.fspath(first))


@pytest.mark.parametrize("value", [
    "", "null", "{}", '"/tmp"', "[1]", '[""]', "not-json",
])
def test_runtime_roots_reject_malformed_configuration(value):
    with pytest.raises(ValueError, match=OFFLINE_RUNTIME_READ_ROOTS_ENV):
        offline_runtime_read_roots(
            {OFFLINE_RUNTIME_READ_ROOTS_ENV: value},
            home=os.path.join(os.sep, "unrelated-home"),
        )


def test_runtime_roots_reject_root_relative_missing_file_and_duplicate(
    tmp_path,
):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    regular = tmp_path / "file"
    regular.write_text("not a directory", encoding="utf-8")
    cases = [
        [os.path.sep],
        ["relative"],
        [os.fspath(tmp_path / "missing")],
        [os.fspath(regular)],
        [os.fspath(runtime), os.fspath(runtime)],
    ]
    for roots in cases:
        with pytest.raises(ValueError):
            offline_runtime_read_roots({
                OFFLINE_RUNTIME_READ_ROOTS_ENV: json.dumps(roots),
            }, home=os.path.join(os.sep, "unrelated-home"))


def test_runtime_roots_reject_symlink_components(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="canonical.*symlinks"):
        offline_runtime_read_roots({
            OFFLINE_RUNTIME_READ_ROOTS_ENV: json.dumps([os.fspath(alias)]),
        }, home=os.path.join(os.sep, "unrelated-home"))


@pytest.mark.parametrize("which", ["home", "ancestor", "descendant"])
def test_runtime_roots_reject_any_user_home_overlap(tmp_path, which):
    home = tmp_path / "users" / "operator"
    descendant = home / "runtime"
    descendant.mkdir(parents=True)
    paths = {
        "home": home,
        "ancestor": tmp_path / "users",
        "descendant": descendant,
    }

    with pytest.raises(ValueError, match="user home"):
        offline_runtime_read_roots({
            OFFLINE_RUNTIME_READ_ROOTS_ENV: json.dumps([
                os.fspath(paths[which]),
            ]),
        }, home=os.fspath(home))


def test_offline_policy_grants_only_configured_roots_and_pins_path(
    tmp_path, monkeypatch,
):
    state = tmp_path / "state"
    state.mkdir()
    runtime = tmp_path / "public-runtime"
    runtime.mkdir()
    runtime_bin = tmp_path / "runtime-bin" / "bin"
    runtime_bin.mkdir(parents=True)
    toolchain = tmp_path / "shared" / "toolchain"
    toolchain.mkdir(parents=True)
    _set_roots(monkeypatch, runtime, runtime_bin)

    args = CodexBackend(cwd=os.fspath(state))._execution_policy_args(
        "discharge", {
            "world_mode": "captured",
            "toolchain_dir": os.fspath(toolchain),
            # A candidate-facing field with a plausible name is inert.
            "runtime_read_roots": [os.fspath(tmp_path)],
        })
    config = _cli_config(args)
    filesystem = config["permissions"]["gv-offline"]["filesystem"]

    assert filesystem[os.fspath(runtime)] == "read"
    assert filesystem[os.fspath(runtime_bin)] == "read"
    assert filesystem[os.fspath(toolchain)] == "write"
    assert os.fspath(tmp_path) not in filesystem
    assert config["shell_environment_policy"]["set"]["PATH"] == (
        f"{toolchain}/bin:{runtime_bin}:/usr/bin:/bin:/usr/sbin:/sbin"
    )


def test_offline_policy_rejects_runtime_overlap_with_state(
    tmp_path, monkeypatch,
):
    state = tmp_path / "state"
    state.mkdir()
    _set_roots(monkeypatch, tmp_path)

    with pytest.raises(RoleError, match="overlaps the isolated state"):
        CodexBackend(cwd=os.fspath(state))._execution_policy_args(
            "critic", {"world_mode": "captured"})


def test_state_local_toolchain_stays_first_on_path(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    runtime_bin = tmp_path / "runtime" / "bin"
    runtime_bin.mkdir(parents=True)
    toolchain = state / "work" / "toolchain"
    _set_roots(monkeypatch, runtime_bin)

    args = CodexBackend(cwd=os.fspath(state))._execution_policy_args(
        "discharge", {
            "world_mode": "captured",
            "toolchain_dir": os.fspath(toolchain),
        })
    path = _cli_config(args)["shell_environment_policy"]["set"]["PATH"]

    assert path.split(os.pathsep)[:2] == [
        os.fspath(toolchain / "bin"), os.fspath(runtime_bin),
    ]


def test_offline_policy_rejects_runtime_overlap_with_writable_toolchain(
    tmp_path, monkeypatch,
):
    state = tmp_path / "state"
    state.mkdir()
    runtime = tmp_path / "runtime"
    toolchain = runtime / "toolchain"
    toolchain.mkdir(parents=True)
    _set_roots(monkeypatch, runtime)

    with pytest.raises(RoleError, match="overlaps a writable root"):
        CodexBackend(cwd=os.fspath(state))._execution_policy_args(
            "critic", {
                "world_mode": "captured",
                "toolchain_dir": os.fspath(toolchain),
            })


def test_live_policy_does_not_consume_offline_runtime_configuration(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv(OFFLINE_RUNTIME_READ_ROOTS_ENV, "not-json")

    assert CodexBackend(cwd=os.fspath(tmp_path))._execution_policy_args(
        "critic", {"world_mode": "live"},
    ) == ["--dangerously-bypass-approvals-and-sandbox"]


def test_authority_policy_never_inherits_offline_runtime_roots(
    tmp_path, monkeypatch,
):
    runtime = tmp_path / "runtime"
    scratch = tmp_path / "scratch"
    runtime.mkdir()
    scratch.mkdir()
    _set_roots(monkeypatch, runtime)
    backend = _AuthorityCodexBackend(cwd=os.fspath(tmp_path))
    backend._authority_scratch = os.fspath(scratch)

    config = _cli_config(backend._execution_policy_args("obligations", {}))
    filesystem = config["permissions"]["gv-authority"]["filesystem"]

    assert filesystem == {
        ":root": "deny", ":minimal": "read", os.fspath(scratch): "read",
    }
    assert os.fspath(runtime) not in filesystem
