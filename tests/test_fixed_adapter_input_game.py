"""Fixed input lowering preserves the full execution/independent-review chain.

These fixture audits test wiring, not model-level sufficiency or paid speedup.
"""
import copy
import json

import pytest

from verifier.core.events import EventLog
from verifier.core.model import BASIS_RECORD, Obligation
from verifier.engine.game import BindingGame
from verifier.engine.method_cache import AcceptedMethodCache
from verifier.execution.adapters import (
    AdapterRoutingDischarger, default_adapter_registry,
)
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend


YES = {"sufficient": "yes", "faithful": "yes", "degradation": "n/a",
       "reason": "fixture audit of the supplied exact method"}
ROW = {"id": "c1", "kind": "final", "text": "recorded exact value"}


def _method(adapter):
    arguments = ({"event_seq": 1, "event_type": "tool_call",
                  "payload_contains": {"input": {"value": "alpha"}}}
                 if adapter == "ledger.event" else {
                     "selector": "first", "value_pointer": "/input/value",
                     "through_seq": 1, "predicate": "equals", "expected": "alpha"})
    return {"procedure": f"evaluate the declared exact retained record predicate with {adapter}",
            "artifact": "a content-addressed evidence envelope",
            "level": "deterministic", "basis": BASIS_RECORD,
            "strength": "decisive", "bridges": [], "adapter": adapter,
            "adapter_version": 1 if adapter == "ledger.event" else 2,
            "adapter_arguments": json.dumps(arguments),
            "adapter_inputs": [], "adapter_input_binding": "fixed-v1"}


class _Fallback:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append(copy.deepcopy(method))
        artifact = self.artifacts.put({"fixture": True, "method": method})
        return DischargeResult(outcome="holds", witness=None, artifact=artifact,
                               instrument="fixture-fallback", basis=BASIS_RECORD,
                               inputs=tuple(method["adapter_inputs"]))


def _harness(tmp_path, live, value="alpha"):
    (tmp_path / "task.md").write_text("The retained value must equal alpha.")
    artifacts = ArtifactStore(str(tmp_path / "artifacts"))
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("tool_call", {"tool": "instrument.arbitrary-v5",
                             "input": {"value": value}})
    fallback = _Fallback(artifacts)
    router = AdapterRoutingDischarger(fallback, default_adapter_registry(),
                                       artifacts, log, str(tmp_path), live_world=live)
    return artifacts, log, fallback, router


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("adapter", ["ledger.event", "ledger.tool-call-string"])
@pytest.mark.parametrize("value,outcome", [("alpha", "holds"), ("omega", "refuted")])
def test_exact_expanded_method_is_freshly_executed_and_judged(
        tmp_path, live, adapter, value, outcome):
    artifacts, log, fallback, router = _harness(tmp_path, live, value)
    original = _method(adapter)
    untouched = copy.deepcopy(original)
    expected = router.registry.bind_inputs(original)
    audits = []

    def judge(variables):
        view = json.loads(variables["adapter_method"])
        evidence = json.loads(variables["evidence"])
        receipt = artifacts.get(evidence["artifact_digest"])
        assert view["input_binding"] == "fixed-v1"
        assert view["inputs"] == expected["adapter_inputs"]
        assert evidence["inputs"] == view["inputs"]
        assert receipt["method"]["adapter"] == view
        assert receipt["execution"]["declared_inputs"] == view["inputs"]
        assert evidence["outcome"] == outcome
        audits.append(evidence)
        return YES

    backend = FixtureBackend({"oracles": {"methods": [original]}, "judger": judge})
    cache = AcceptedMethodCache()
    game = BindingGame(backend, artifacts, log, router, live_world=live,
                       method_cache=cache)
    obligation = Obligation("c1", "the retained value equals alpha", BASIS_RECORD)
    for _ in range(2):
        result = game.run(ROW, obligation)
        assert result.outcome == outcome
        bound = artifacts.get(result.artifact)["bound"]
        assert bound["method"] == expected
        assert list(result.inputs) == expected["adapter_inputs"]
    assert original == untouched
    assert [call["role"] for call in backend.calls] == ["oracles", "judger"] * 2
    assert len(audits) == 2 and len(cache) == 0 and not fallback.calls
    executions = [event for event in log.replay() if event["type"] == "adapter_discharge"]
    assert len(executions) == 2
    assert all(event["payload"]["adapter_input_binding"] == "fixed-v1" for event in executions)


@pytest.mark.parametrize("live", [False, True])
def test_independent_rejection_still_blocks_binding(tmp_path, live):
    artifacts, log, fallback, router = _harness(tmp_path, live)
    methods = iter([{"methods": [_method("ledger.event")]}, {"methods": []}])
    backend = FixtureBackend({"oracles": lambda _: next(methods),
                              "judger": {**YES, "sufficient": "no"}})
    cache = AcceptedMethodCache()
    result = BindingGame(backend, artifacts, log, router, live_world=live,
                         method_cache=cache).run(
                             ROW, Obligation("c1", "a stronger predicate", BASIS_RECORD))
    assert result.outcome not in {"holds", "refuted"}
    assert artifacts.get(result.artifact)["bound"] is None
    assert not fallback.calls and len(cache) == 0
    assert not any(event["type"] == "method_bound" for event in log.replay())


def test_judger_redo_preserves_the_same_materialized_method(tmp_path):
    artifacts, log, fallback, router = _harness(tmp_path, False)
    original = _method("ledger.event")
    expected = router.registry.bind_inputs(original)
    audits = iter([{**YES, "faithful": "no"}, YES])
    views = []

    def judge(variables):
        views.append(json.loads(variables["adapter_method"]))
        return next(audits)

    backend = FixtureBackend({"oracles": {"methods": [original]}, "judger": judge})
    result = BindingGame(backend, artifacts, log, router, live_world=False).run(
        ROW, Obligation("c1", "the exact retained value", BASIS_RECORD))
    assert result.outcome == "holds"
    assert len(fallback.calls) == 1 and views[0] == views[1]
    assert fallback.calls[0]["adapter_inputs"] == expected["adapter_inputs"]
    assert artifacts.get(result.artifact)["bound"]["method"] == expected
    assert [call["role"] for call in backend.calls] == ["oracles", "judger", "judger"]


@pytest.mark.parametrize("fixed", [False, True])
def test_execution_failure_preserves_original_or_expanded_inputs(tmp_path, fixed):
    artifacts, log, fallback, router = _harness(tmp_path, False)
    original = _method("ledger.tool-call-string")
    if not fixed:
        del original["adapter_input_binding"]
    args = json.loads(original["adapter_arguments"])
    args["predicate"] = "nonblank"  # expected is forbidden; never silently strip it.
    original["adapter_arguments"] = json.dumps(args)
    expected = router.registry.bind_inputs(original)
    views = []
    backend = FixtureBackend({"oracles": {"methods": [original]},
                              "judger": lambda v: (views.append(json.loads(
                                  v["adapter_method"])), YES)[1]})
    result = BindingGame(backend, artifacts, log, router, live_world=False).run(
        ROW, Obligation("c1", "the exact retained predicate", BASIS_RECORD))
    assert result.outcome == "holds"  # Only the fixture Judger grants this.
    assert len(fallback.calls) == 1
    assert fallback.calls[0]["adapter_inputs"] == expected["adapter_inputs"]
    assert fallback.calls[0]["adapter_arguments"] == original["adapter_arguments"]
    assert "expected is permitted only" in fallback.calls[0]["adapter_fallback_reason"]
    assert views[0]["inputs"] == expected["adapter_inputs"]
    assert artifacts.get(result.artifact)["bound"]["method"] == expected


@pytest.mark.parametrize("absence_allowed", [False, True])
def test_fixed_source_binding_does_not_bypass_absence_gate(tmp_path, absence_allowed):
    artifacts, log, fallback, router = _harness(tmp_path, False)
    original = _method("ledger.tool-call-string")
    args = json.loads(original["adapter_arguments"])
    args.update(selector="any", tool_names=["different.literal.instrument"])
    original["adapter_arguments"] = json.dumps(args)
    bound = router.registry.bind_inputs(original)
    obligation = Obligation("c1", "presence in the complete declared scope", BASIS_RECORD,
                            refutable_by_absence=absence_allowed)
    result = router.discharge(ROW, obligation, bound)
    if absence_allowed:
        assert result.outcome == "refuted" and result.witness_kind == "absence"
        assert not fallback.calls
    else:
        assert result.instrument == "fixture-fallback" and len(fallback.calls) == 1
    assert bound["adapter_arguments"] == original["adapter_arguments"]


def test_each_oracle_repair_is_lowered_before_independent_review(tmp_path):
    artifacts, log, fallback, router = _harness(tmp_path, False)
    first = _method("ledger.event")
    second = _method("ledger.tool-call-string")
    proposals = iter([{"methods": [first]}, {"methods": [second]}])
    verdicts = iter([{**YES, "sufficient": "no"}, YES])
    views = []

    def judge(variables):
        views.append(json.loads(variables["adapter_method"]))
        return next(verdicts)

    backend = FixtureBackend({"oracles": lambda _: next(proposals), "judger": judge})
    result = BindingGame(backend, artifacts, log, router, live_world=False).run(
        ROW, Obligation("c1", "an independently checked record predicate", BASIS_RECORD))
    assert result.outcome == "holds" and not fallback.calls
    assert [view["inputs"][0]["ref"] for view in views] == ["event:1", "ledger:events"]
    assert first["adapter_inputs"] == second["adapter_inputs"] == []
    assert artifacts.get(result.artifact)["bound"]["method"] == router.registry.bind_inputs(second)
    assert [call["role"] for call in backend.calls] == ["oracles", "judger"] * 2
