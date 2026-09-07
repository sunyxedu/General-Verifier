"""Configuration tests for optional typed execution adapters."""
from __future__ import annotations

import pytest

from verifier.execution.config import typed_adapters_enabled
from verifier.execution.discharge import AgentDischarger
from verifier.llm.schema import FixtureBackend
from verifier.mcp_server import Service, _default_kernel


def test_typed_adapters_are_enabled_by_default():
    assert typed_adapters_enabled({}) is True


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
def test_typed_adapters_accept_only_explicit_binary_values(value, expected):
    assert typed_adapters_enabled({"GV_TYPED_ADAPTERS": value}) is expected


@pytest.mark.parametrize("value", ["", "false", "true", " 0", "1 ", "2"])
def test_typed_adapters_reject_ambiguous_values(value):
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        typed_adapters_enabled({"GV_TYPED_ADAPTERS": value})


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
