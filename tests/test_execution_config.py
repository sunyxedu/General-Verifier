"""Configuration tests for optional typed execution adapters."""
from __future__ import annotations

import pytest

from verifier.core.model import Obligation
from verifier.execution.config import (
    typed_adapters_enabled,
    typed_judger_packet_enabled,
)
from verifier.execution.discharge import AgentDischarger
from verifier.llm.schema import FixtureBackend
from verifier.mcp_server import Service, _default_kernel
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

    kernel = _default_kernel(
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

    kernel = _default_kernel(
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
        "gv-typed-adapter-judger-packet-v2;")
    assert "transport=no-tools-v1" in fields[
        "typed_adapter_judger_packet_protocol"]

    monkeypatch.setenv("GV_TYPED_ADAPTERS", "0")
    with pytest.raises(ValueError, match="requires typed adapters"):
        offline.config_fields(seats="codex")
    service = Service(
        str(tmp_path / "gv"), kernel_factory=lambda _service: object())
    with pytest.raises(ValueError, match="requires typed adapters"):
        _default_kernel(
            service, model=None, backend=FixtureBackend({}), critic_rounds=0)
