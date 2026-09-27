"""Configuration tests for optional typed execution adapters."""
from __future__ import annotations

import pytest

from verifier.core.model import Obligation
from verifier.config import (
    typed_adapters_enabled,
    typed_judger_packet_enabled,
)
from verifier.execution.discharge import AgentDischarger
from verifier.llm.codex.runtime import CodexRuntime
from verifier.llm.schema import FixtureBackend
from verifier.app.assembly import build_kernel
from verifier.mcp_server import Service
from verifier import offline


def test_typed_adapters_are_enabled_by_default():
    assert typed_adapters_enabled({}) is True


def test_typed_judger_packet_is_disabled_by_default():
    assert typed_judger_packet_enabled({}) is False


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
def test_typed_adapters_accept_only_explicit_binary_values(value, expected):
    assert typed_adapters_enabled({"GV_TYPED_ADAPTERS": value}) is expected


@pytest.mark.parametrize("value", ["", "false", "true", " 0", "1 ", "2"])
def test_typed_adapters_reject_ambiguous_values(value):
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        typed_adapters_enabled({"GV_TYPED_ADAPTERS": value})


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
def test_typed_judger_packet_accepts_only_explicit_binary_values(value, expected):
    assert typed_judger_packet_enabled(
        {"GV_TYPED_JUDGER_PACKET": value}) is expected


@pytest.mark.parametrize("value", ["", "false", "true", " 0", "1 ", "2"])
def test_typed_judger_packet_rejects_ambiguous_values(value):
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        typed_judger_packet_enabled({"GV_TYPED_JUDGER_PACKET": value})


def test_disabled_kernel_keeps_agent_discharge_and_removes_method_cache(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "0")
    service = Service(
        str(tmp_path / "gv"), kernel_factory=lambda _service: object())

    kernel = build_kernel(
        service, model=None, backend=FixtureBackend({}), critic_rounds=0)

    assert isinstance(kernel.discharger, AgentDischarger)
    assert kernel.binder.discharger is kernel.discharger
    assert kernel.binder.method_cache is None
    assert kernel.binder.registry == "[]"


def test_enabled_kernel_packet_seam_disables_method_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "1")
    service = Service(
        str(tmp_path / "gv"), kernel_factory=lambda _service: object())

    kernel = build_kernel(
        service, model=None, backend=FixtureBackend({}), critic_rounds=0)

    assert kernel.binder.typed_adapter_judger_packet is True
    assert kernel.binder._method_cache_key(
        {"id": "claim"},
        Obligation("claim", "fixture"),
    ) is None


def test_packet_mode_is_stamped_and_cannot_run_without_typed_adapters(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "1")
    fields = offline.config_fields(seats="codex")
    assert fields["typed_adapter_judger_packet"] is True
    assert fields["typed_adapter_judger_packet_protocol"].startswith(
        "gv-typed-adapter-judger-packet-v6;")
    assert "transport=no-tools-v1" in fields[
        "typed_adapter_judger_packet_protocol"]

    monkeypatch.setenv("GV_TYPED_ADAPTERS", "0")
    with pytest.raises(ValueError, match="requires typed adapters"):
        offline.config_fields(seats="codex")
    service = Service(
        str(tmp_path / "gv"), kernel_factory=lambda _service: object())
    with pytest.raises(ValueError, match="requires typed adapters"):
        build_kernel(
            service, model=None, backend=FixtureBackend({}), critic_rounds=0)


def test_restricted_judger_cwd_cannot_be_silently_ignored(
        tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o500)
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "0")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "0")

    with pytest.raises(ValueError, match="requires the built-in typed Codex"):
        offline.score(
            str(tmp_path / "state"), seats="codex",
            restricted_judger_cwd=str(scratch))


def test_restricted_judger_cwd_rejects_custom_kernel_factory(
        tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o500)
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "1")

    with pytest.raises(ValueError, match="built-in typed Codex"):
        offline.score(
            str(tmp_path / "state"), seats="codex",
            restricted_judger_cwd=str(scratch),
            kernel_factory=lambda _service: object())


def test_restricted_judger_cwd_rejects_unbound_injected_backend(
        tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o500)
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "1")
    backend = CodexRuntime(cwd=str(tmp_path))

    with pytest.raises(ValueError, match="exact restricted Judger cwd"):
        offline.score(
            str(tmp_path / "state"), seats="codex", backend=backend,
            judger_prefetch=False,
            restricted_judger_cwd=str(scratch))


def test_injected_backend_cannot_hide_restricted_judger_cwd(
        tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o500)
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "1")
    backend = CodexRuntime(
        cwd=str(state), restricted_judger_cwd=str(scratch))

    with pytest.raises(ValueError, match="exact restricted Judger cwd"):
        offline.score(
            str(state), seats="codex", backend=backend,
            judger_prefetch=False)
