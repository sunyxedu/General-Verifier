"""Mode wiring and freshness regressions for general typed adapters."""
from __future__ import annotations

import json

from verifier.core.events import EventLog
from verifier.core.model import BASIS_RECORD, OUTCOME_HOLDS, Obligation
from verifier.engine.game import BindingGame
from verifier.engine.method_cache import AcceptedMethodCache
from verifier.execution.adapters import (
    AdapterRoutingDischarger,
    default_adapter_registry,
)
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend
from verifier.app.assembly import build_kernel
from verifier.mcp_server import Service


JUDGE_YES = {
    "sufficient": "yes",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "the retained event field decides the exact obligation",
}


class _Fallback:
    def __init__(self, artifacts: ArtifactStore):
        self.artifacts = artifacts
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append((row, obligation, method, tag))
        digest = self.artifacts.put({"instrument": "unexpected-fallback"})
        return DischargeResult(
            outcome="failed", witness="unexpected fallback",
            artifact=digest, instrument="unexpected-fallback",
        )


def _ledger_method():
    return {
        "procedure": "read one exact string from the bounded ledger prefix",
        "artifact": "content-addressed typed evidence",
        "level": "deterministic",
        "basis": BASIS_RECORD,
        "strength": "decisive",
        "bridges": [],
        "adapter": "ledger.tool-call-string",
        "adapter_version": 1,
        "adapter_arguments": json.dumps({
            "selector": "any",
            "value_pointer": "/input/value",
            "through_seq": 1,
            "expected": "alpha",
        }, sort_keys=True),
        "adapter_inputs": [{
            "ref": "ledger:events",
            "provenance": "ledger",
            "role": "subject",
        }],
    }


def test_online_and_offline_default_kernels_publish_the_same_adapter_catalog(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1")
    # Process-local reuse is an explicit opt-in that both entry points honor
    # identically; it is part of the scoring identity, never ambient.
    monkeypatch.setenv("GV_CONTRACT_CACHE", "1")
    monkeypatch.delenv("GV_METHOD_CACHE", raising=False)
    for name in (
        "GV_BROWSER_OBSERVER",
        "GV_BROWSER_OBSERVER_SHA256",
        "GV_BROWSER_EXECUTABLE",
        "GV_BROWSER_EXECUTABLE_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)

    online = Service(
        str(tmp_path / "online"), live_world=True,
        kernel_factory=lambda _service: object())
    offline_state = tmp_path / "offline"
    (offline_state / "world").mkdir(parents=True)
    (offline_state / "world_manifest.json").write_text(json.dumps({
        "scheme": "filesystem", "captured_roots": ["/app"],
    }))
    offline = Service(
        str(offline_state), live_world=False,
        kernel_factory=lambda _service: object())

    online_kernel = build_kernel(
        online, model=None, backend=FixtureBackend({}), critic_rounds=0,
        parallel_obligations=1, parallel_claims=1)
    offline_kernel = build_kernel(
        offline, model=None, backend=FixtureBackend({}), critic_rounds=0,
        parallel_obligations=1, parallel_claims=1)

    assert online_kernel.discharger.adapter_catalog() \
        == offline_kernel.discharger.adapter_catalog()
    catalog = json.loads(online_kernel.discharger.adapter_catalog())
    ledger = [entry for entry in catalog
              if entry["name"].startswith("ledger.")]
    assert {entry["name"] for entry in ledger} == {
        "ledger.event", "ledger.sequence", "ledger.tool-call-string",
    }
    assert sum(entry["name"] == "record.query" and entry["version"] == 1
               for entry in catalog) == 1
    assert all(entry["accepted_method_reuse"] is False for entry in ledger)
    assert online_kernel.discharger.context.world.mode == "live"
    assert offline_kernel.discharger.context.world.mode == "captured"
    assert online_kernel.binder.method_cache is None
    assert offline_kernel.binder.method_cache is None
    assert online_kernel.contract_cache is offline_kernel.contract_cache
    assert online_kernel.contract_cache is not None
    monkeypatch.delenv("GV_CONTRACT_CACHE")
    plain = build_kernel(
        offline, model=None, backend=FixtureBackend({}), critic_rounds=0,
        parallel_obligations=1, parallel_claims=1)
    assert plain.contract_cache is None


def test_ledger_method_is_freshly_oracle_judged_for_each_state(tmp_path):
    cache = AcceptedMethodCache()
    backends = []
    logs = []
    fallbacks = []

    for index in range(2):
        state = tmp_path / f"candidate-{index}"
        state.mkdir()
        (state / "task.md").write_text("Retain the recorded alpha value")
        artifacts = ArtifactStore(str(state / "artifacts"))
        log = EventLog(str(state / "events.jsonl"))
        log.append("tool_call", {
            "tool": "benign-recorder", "input": {"value": "alpha"},
        })
        fallback = _Fallback(artifacts)
        routing = AdapterRoutingDischarger(
            fallback, default_adapter_registry(), artifacts, log, str(state),
            live_world=False)
        backend = FixtureBackend({
            "oracles": {"methods": [_ledger_method()]},
            "judger": JUDGE_YES,
        })
        game = BindingGame(
            backend, artifacts, log, routing, live_world=False,
            method_cache=cache)

        result = game.run(
            {"id": f"claim-{index}", "kind": "final", "text": "alpha kept"},
            Obligation(f"claim-{index}", "recorded value equals alpha",
                       BASIS_RECORD),
        )

        assert result.outcome == OUTCOME_HOLDS
        backends.append(backend)
        logs.append(log)
        fallbacks.append(fallback)

    assert [[call["role"] for call in backend.calls]
            for backend in backends] == [
        ["oracles", "judger"], ["oracles", "judger"],
    ]
    assert len(cache) == 0
    assert all(fallback.calls == [] for fallback in fallbacks)
    assert not any(
        event["type"] == "method_cache_hit"
        for log in logs for event in log.replay()
    )


def test_routing_discharger_keeps_the_executable_form_path(tmp_path):
    """BindingGame probes the discharger for `run_script` by name; the
    typed-adapter router must expose the wrapped executor's script
    runner, or every executable-form method silently falls back to an
    agent seat."""
    from verifier.execution.adapters import (
        AdapterRoutingDischarger, default_adapter_registry,
    )
    state = tmp_path / "state"
    state.mkdir()
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))

    class _ScriptFallback(_Fallback):
        def run_script(self, row, obligation, method, tag=""):
            self.calls.append(("script", method.get("script"), tag))
            return "ran"

    fallback = _ScriptFallback(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=False)
    assert hasattr(routing, "run_script")
    assert routing.run_script({}, object(), {"script": "echo hi"}, tag="t") \
        == "ran"
    assert fallback.calls == [("script", "echo hi", "t")]

