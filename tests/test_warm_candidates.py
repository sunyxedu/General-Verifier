"""Candidates of one task scored in one process share audited work."""
from __future__ import annotations

import copy
import json

import pytest

from verifier import offline
from verifier.core.store import TRIGGER_LAZY
from verifier.engine.certificate import CERTIFICATE_SCOPE_FINAL, request_certificate
from verifier.engine import contract_cache as contract_cache_module
from verifier.engine import method_cache as method_cache_module
from verifier.llm.schema import FixtureBackend
from verifier.app.assembly import build_kernel
from verifier.mcp_server import Service


PASS = {"verdict": "pass", "counterexample_world": "n/a",
        "missing_condition": "n/a"}
JUDGE = {"sufficient": "yes", "faithful": "yes", "degradation": "n/a",
         "reason": "the registered adapter decides the exact obligation"}


def _fixtures():
    return {
        "obligations": {"obligations": [{
            "text": "The delivered file /app/answer.txt exists.",
            "required_basis": "world", "severity": "must",
            "clause": "Write the answer file.", "refutable_by_absence": True,
        }]},
        "critic": copy.deepcopy(PASS),
        "oracles": {"methods": [{
            "procedure": "check that the delivered file exists via the adapter",
            "artifact": "typed evidence", "level": "exact", "basis": "world",
            "strength": "decisive", "bridges": [],
            "adapter": "filesystem.exists", "adapter_version": 1,
            "adapter_arguments": json.dumps({"address": "/app/answer.txt"}),
            "adapter_inputs": [{"ref": "/app/answer.txt",
                                "provenance": "delivered", "role": "subject"}],
        }]},
        "judger": copy.deepcopy(JUDGE),
    }


def _candidate(root, name, backend):
    state = root / name
    state.mkdir()
    (state / "task.md").write_text("Write the answer file.\n", encoding="utf-8")
    world = state / "world" / "app"
    world.mkdir(parents=True)
    (world / "answer.txt").write_text(name + "\n", encoding="utf-8")
    (state / "world_manifest.json").write_text(json.dumps({
        "captured_roots": ["/app"], "world_mode": "captured",
        "scheme": "filesystem"}), encoding="utf-8")
    with open(state / "hook_feed.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "tool_call", "payload": {
            "tool": "bash", "input": {"command": "true"},
            "tool_use_id": "call-1"}}) + "\n")
        handle.write(json.dumps({"type": "tool_result", "payload": {
            "tool": "bash", "tool_use_id": "call-1", "excerpt": ""}}) + "\n")
    service = Service(
        str(state), live_world=False, trigger=TRIGGER_LAZY,
        kernel_factory=lambda sc: build_kernel(sc, None, backend=backend),
    )
    service.drain_hooks()
    service.commit("f", "final", f"{name} finished the task", ["effect:2"])
    certificate = request_certificate(
        service.store, service.kernel, service.log, service.artifacts,
        service.builder.finals[-1], scope=CERTIFICATE_SCOPE_FINAL)
    events = [e["type"] for e in service.log.replay()]
    return certificate, events


def _fresh_caches(monkeypatch):
    monkeypatch.setattr(
        contract_cache_module, "_DEFAULT_COMPILED_FINAL_CONTRACT_CACHE",
        contract_cache_module.CompiledFinalContractCache())
    monkeypatch.setattr(
        method_cache_module, "_DEFAULT_ACCEPTED_METHOD_CACHE",
        method_cache_module.AcceptedMethodCache())


def test_second_candidate_reuses_contract_and_accepted_method(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "0")
    monkeypatch.setenv("GV_CONTRACT_CACHE", "1")
    monkeypatch.setenv("GV_METHOD_CACHE", "1")
    _fresh_caches(monkeypatch)
    backend = FixtureBackend(_fixtures())

    first, first_events = _candidate(tmp_path, "alpha", backend)
    calls_after_first = [c["role"] for c in backend.calls]
    second, second_events = _candidate(tmp_path, "beta", backend)
    calls_after_second = [c["role"] for c in backend.calls]

    assert first["verdict"] == second["verdict"] == "verified"
    assert "contract_cache_published" in first_events
    assert "method_cache_published" in first_events
    assert "contract_cache_hit" in second_events
    assert "method_cache_hit" in second_events
    # the second candidate spent no compiler, critic, oracle or judger call:
    # its adapter still executed against its own delivered world
    assert calls_after_second[len(calls_after_first):] == []
    assert "adapter_discharge" in second_events
    assert offline.config_fields(model="m", seats="codex")["method_cache"] is True


def test_caches_stay_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1")
    monkeypatch.delenv("GV_CONTRACT_CACHE", raising=False)
    monkeypatch.delenv("GV_METHOD_CACHE", raising=False)
    _fresh_caches(monkeypatch)
    backend = FixtureBackend(_fixtures())
    _candidate(tmp_path, "alpha", backend)
    before = len(backend.calls)
    _second, events = _candidate(tmp_path, "beta", backend)
    assert len(backend.calls) == 2 * before
    assert not any(kind.startswith(("contract_cache_hit", "method_cache_hit"))
                   for kind in events)
    fields = offline.config_fields(model="m", seats="codex")
    assert fields["contract_cache"] is False and fields["method_cache"] is False


def test_non_reusable_settlement_reports_why(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "0")
    monkeypatch.setenv("GV_CONTRACT_CACHE", "1")
    monkeypatch.setenv("GV_METHOD_CACHE", "1")
    _fresh_caches(monkeypatch)
    fixtures = _fixtures()
    # an agent method (no adapter binding) can never be reused
    fixtures["oracles"] = {"methods": [{
        "procedure": "read the delivered file and compare by hand",
        "artifact": "a retained read", "level": "exact", "basis": "world",
        "strength": "decisive", "bridges": []}]}
    fixtures["discharge"] = {
        "outcome": "holds", "witness": "the file exists", "artifact": "read",
        "assumed_premises": [], "witness_kind": "none",
        "inputs": [{"ref": "/app/answer.txt", "provenance": "delivered",
                    "role": "subject"}]}
    backend = FixtureBackend(fixtures)
    _certificate, events = _candidate(tmp_path, "alpha", backend)
    assert "method_cache_published" not in events
    assert "method_cache_bypass" in events
