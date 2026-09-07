"""Typed adapter execution stays a fail-safe optimization of BindingGame."""
from __future__ import annotations

import json

import pytest

from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_RECORD,
    BASIS_WORLD,
    OUTCOME_HOLDS,
    OUTCOME_REFUTED,
    WITNESS_ABSENCE,
    WITNESS_NONE,
    WITNESS_RECORD_CONTRADICTION,
    WITNESS_WORLD_STATE,
    Obligation,
)
from verifier.engine.game import BindingGame
from verifier.execution.adapters import (
    EVIDENCE_SCHEMA,
    AdapterInvocation,
    AdapterNotApplicable,
    AdapterRoutingDischarger,
    default_adapter_registry,
)
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend
from verifier.mcp_server import Service, _default_kernel


ROW = {"id": "c1", "kind": "final", "text": "the task is complete"}
JUDGE_YES = {
    "sufficient": "yes",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "the adapter artifact matches the method",
}


def method(name, arguments, *, basis=BASIS_WORLD, inputs=None, version=1):
    return {
        "procedure": f"execute the exact registered method {name}",
        "artifact": "a content-addressed adapter evidence envelope",
        "level": "deterministic",
        "basis": basis,
        "strength": "decisive",
        "bridges": [],
        "adapter": name,
        "adapter_version": version,
        "adapter_arguments": json.dumps(arguments, sort_keys=True),
        "adapter_inputs": inputs or [],
    }


def subject(ref, provenance="delivered"):
    return {"ref": ref, "provenance": provenance, "role": "subject"}


class FallbackDischarger:
    def __init__(self, artifacts, outcome=OUTCOME_HOLDS):
        self.artifacts = artifacts
        self.outcome = outcome
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append({"method": method, "tag": tag})
        digest = self.artifacts.put({"instrument": "fallback", "tag": tag})
        return DischargeResult(
            outcome=self.outcome, witness=None, artifact=digest,
            instrument="fallback", witness_kind="execution")


def captured(tmp_path):
    state = tmp_path / "gv"
    world = state / "world" / "app"
    world.mkdir(parents=True)
    (state / "world_manifest.json").write_text(json.dumps({
        "scheme": "filesystem",
        "captured_roots": ["/app"],
    }))
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = FallbackDischarger(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=False,
    )
    return state, world, artifacts, log, fallback, routing


def test_method_ir_is_optional_but_never_partially_accepted():
    assert AdapterInvocation.from_method({"procedure": "legacy"}) is None
    with pytest.raises(AdapterNotApplicable, match="partial"):
        AdapterInvocation.from_method({"adapter": "filesystem.exists"})
    with pytest.raises(AdapterNotApplicable, match="duplicate JSON key"):
        AdapterInvocation.from_method(method(
            "filesystem.exists", {"address": "/app/x"},
            inputs=[subject("/app/x")],
        ) | {"adapter_arguments": '{"address":"/app/x","address":"/app/y"}'})


def test_catalog_is_versioned_narrow_and_has_no_command_executor():
    catalog = json.loads(default_adapter_registry().catalog())
    assert {(item["name"], item["version"]) for item in catalog} == {
        ("filesystem.exists", 1),
        ("filesystem.sha256", 1),
        ("document.json-value", 1),
        ("ledger.event", 1),
    }
    assert not any("command" in item["name"] for item in catalog)


def test_file_exists_adapter_emits_auditable_world_evidence(tmp_path):
    _, world, artifacts, log, fallback, routing = captured(tmp_path)
    target = world / "result.json"
    target.write_text('{"status":"completed"}')
    obligation = Obligation(
        "c1", "the result artifact exists", BASIS_WORLD,
        refutable_by_absence=True)
    result = routing.discharge(
        ROW, obligation,
        method("filesystem.exists", {"address": "/app/result.json",
                                      "kind": "file"},
               inputs=[subject("/app/result.json")]),
    )

    assert result.outcome == OUTCOME_HOLDS
    assert result.witness_kind == WITNESS_NONE
    assert fallback.calls == []
    evidence = artifacts.get(result.artifact)
    assert evidence["schema"] == EVIDENCE_SCHEMA
    assert evidence["instrument"] == "adapter:filesystem.exists@1"
    assert evidence["execution"]["observation"]["actual_kind"] == "file"
    events = list(log.replay())
    assert events[-1]["type"] == "adapter_discharge"


def test_absence_refutes_only_an_explicit_presence_requirement(tmp_path):
    _, _, _, _, fallback, routing = captured(tmp_path)
    typed = method(
        "filesystem.exists", {"address": "/app/missing", "kind": "file"},
        inputs=[subject("/app/missing")])

    refuted = routing.discharge(
        ROW, Obligation("c1", "the file exists", BASIS_WORLD,
                        refutable_by_absence=True), typed)
    fallback_result = routing.discharge(
        ROW, Obligation("c1", "the file has a property", BASIS_WORLD), typed)

    assert refuted.outcome == OUTCOME_REFUTED
    assert refuted.witness_kind == WITNESS_ABSENCE
    assert fallback_result.instrument == "fallback"
    assert len(fallback.calls) == 1


def test_captured_path_escape_and_wrong_basis_fall_back(tmp_path):
    _, _, _, log, fallback, routing = captured(tmp_path)
    escape = method(
        "filesystem.exists", {"address": "/app/../../etc/passwd"},
        inputs=[subject("/app/../../etc/passwd")])
    wrong_basis = method(
        "filesystem.exists", {"address": "/app/result.json"},
        basis=BASIS_RECORD, inputs=[subject("/app/result.json")])
    obligation = Obligation("c1", "a condition", BASIS_RECORD)

    assert routing.discharge(ROW, obligation, escape).instrument == "fallback"
    assert routing.discharge(ROW, obligation, wrong_basis).instrument == "fallback"
    assert len(fallback.calls) == 2
    fallback_events = [e for e in log.replay()
                       if e["type"] == "adapter_fallback"]
    assert "escapes" in fallback_events[0]["payload"]["reason"]
    assert "basis=world" in fallback_events[1]["payload"]["reason"]


def test_unknown_adapter_bad_version_and_missing_subject_fall_back(tmp_path):
    _, _, _, log, fallback, routing = captured(tmp_path)
    obligation = Obligation("c1", "the file exists", BASIS_WORLD,
                            refutable_by_absence=True)
    candidates = [
        method("unknown.adapter", {}, inputs=[subject("/app/x")]),
        method("filesystem.exists", {"address": "/app/x"}, version=2,
               inputs=[subject("/app/x")]),
        method("filesystem.exists", {"address": "/app/x"}),
    ]

    for candidate in candidates:
        assert routing.discharge(ROW, obligation, candidate).instrument \
            == "fallback"
    assert len(fallback.calls) == 3
    assert len([e for e in log.replay()
                if e["type"] == "adapter_fallback"]) == 3


def test_sha256_adapter_records_the_consumed_content_digest(tmp_path):
    _, world, artifacts, _, fallback, routing = captured(tmp_path)
    target = world / "answer.bin"
    target.write_bytes(b"answer\x00bytes")
    import hashlib
    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    obligation = Obligation("c1", "the artifact has the required digest",
                            BASIS_WORLD)

    holds = routing.discharge(
        ROW, obligation,
        method("filesystem.sha256", {
            "address": "/app/answer.bin", "expected_sha256": expected,
        }, inputs=[subject("/app/answer.bin")]))
    refuted = routing.discharge(
        ROW, obligation,
        method("filesystem.sha256", {
            "address": "/app/answer.bin", "expected_sha256": "0" * 64,
        }, inputs=[subject("/app/answer.bin")]))

    assert holds.outcome == OUTCOME_HOLDS
    evidence = artifacts.get(holds.artifact)
    assert evidence["execution"]["consumed"][0]["sha256"] == expected
    assert refuted.outcome == OUTCOME_REFUTED
    assert refuted.witness_kind == WITNESS_WORLD_STATE
    assert fallback.calls == []


def test_json_value_uses_strict_json_and_type_sensitive_equality(tmp_path):
    _, world, _, _, fallback, routing = captured(tmp_path)
    (world / "result.json").write_text(
        '{"status":"completed","flag":true}')
    obligation = Obligation("c1", "the exact JSON value matches", BASIS_WORLD)

    holds = routing.discharge(
        ROW, obligation,
        method("document.json-value", {
            "address": "/app/result.json", "pointer": "/status",
            "expected": "completed",
        }, inputs=[subject("/app/result.json")]))
    bool_is_not_one = routing.discharge(
        ROW, obligation,
        method("document.json-value", {
            "address": "/app/result.json", "pointer": "/flag",
            "expected": 1,
        }, inputs=[subject("/app/result.json")]))

    assert holds.outcome == OUTCOME_HOLDS
    assert bool_is_not_one.outcome == OUTCOME_REFUTED
    assert fallback.calls == []


def test_live_world_adapter_reads_the_original_absolute_address(tmp_path):
    state = tmp_path / "gv"
    state.mkdir()
    target = tmp_path / "delivery" / "result.json"
    target.parent.mkdir()
    target.write_text('{"status":"completed"}')
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = FallbackDischarger(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=True)
    typed = method(
        "document.json-value", {
            "address": str(target), "pointer": "/status",
            "expected": "completed",
        }, inputs=[subject(str(target))])

    result = routing.discharge(
        ROW, Obligation("c1", "status equals completed", BASIS_WORLD), typed)

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []


def test_ledger_adapter_matches_an_immutable_event_subset(tmp_path):
    state = tmp_path / "gv"
    state.mkdir()
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    event = log.append("tool_result", {
        "tool": "example", "result": {"exit": 0, "stdout": "ok"}})
    fallback = FallbackDischarger(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=False)
    obligation = Obligation("c1", "the retained action succeeded", BASIS_RECORD)
    declared = [subject(f"event:{event['seq']}", "ledger")]

    holds = routing.discharge(
        ROW, obligation,
        method("ledger.event", {
            "event_seq": event["seq"], "event_type": "tool_result",
            "payload_contains": {"result": {"exit": 0}},
        }, basis=BASIS_RECORD, inputs=declared))
    refuted = routing.discharge(
        ROW, obligation,
        method("ledger.event", {
            "event_seq": event["seq"],
            "payload_contains": {"result": {"exit": 1}},
        }, basis=BASIS_RECORD, inputs=declared))

    assert holds.outcome == OUTCOME_HOLDS
    assert refuted.outcome == OUTCOME_REFUTED
    assert refuted.witness_kind == WITNESS_RECORD_CONTRADICTION
    assert fallback.calls == []


def test_binding_game_keeps_oracle_and_judger_but_skips_agent_discharge(
        tmp_path):
    _, world, artifacts, log, fallback, routing = captured(tmp_path)
    (world / "result.json").write_text('{"status":"completed"}')
    oracle_method = method(
        "document.json-value", {
            "address": "/app/result.json", "pointer": "/status",
            "expected": "completed",
        }, inputs=[subject("/app/result.json")])
    backend = FixtureBackend({
        "oracles": {"methods": [oracle_method]},
        "judger": JUDGE_YES,
    })
    game = BindingGame(
        backend, artifacts, log, routing, live_world=False)

    result = game.run(
        ROW, Obligation("c1", "status equals completed", BASIS_WORLD))

    assert result.outcome == OUTCOME_HOLDS
    assert [call["role"] for call in backend.calls] == ["oracles", "judger"]
    assert fallback.calls == []
    oracle_vars = backend.calls[0]["variables"]
    assert "document.json-value" in oracle_vars["adapter_catalog"]
    judge_vars = backend.calls[1]["variables"]
    assert "document.json-value" in judge_vars["adapter_method"]
    evidence = json.loads(judge_vars["evidence"])
    assert artifacts.get(evidence["artifact_digest"])["schema"] \
        == EVIDENCE_SCHEMA


def test_judger_redo_routes_to_agent_instead_of_repeating_adapter(tmp_path):
    _, world, artifacts, log, fallback, routing = captured(tmp_path)
    (world / "result.json").write_text('{"status":"completed"}')
    typed = method(
        "document.json-value", {
            "address": "/app/result.json", "pointer": "/status",
            "expected": "completed",
        }, inputs=[subject("/app/result.json")])
    verdicts = iter([{
        "sufficient": "yes", "faithful": "no", "degradation": "n/a",
        "reason": "the execution needs an additional retained explanation",
    }, JUDGE_YES])
    backend = FixtureBackend({
        "oracles": {"methods": [typed]},
        "judger": lambda _: next(verdicts),
    })
    game = BindingGame(backend, artifacts, log, routing, live_world=False)

    result = game.run(
        ROW, Obligation("c1", "status equals completed", BASIS_WORLD))

    assert result.outcome == OUTCOME_HOLDS
    assert len(fallback.calls) == 1
    assert fallback.calls[0]["tag"] == "bound-redo"
    assert any(event["type"] == "adapter_fallback"
               for event in log.replay())


def test_legacy_oracle_method_is_byte_shape_compatible_and_uses_fallback(
        tmp_path):
    _, _, _, log, fallback, routing = captured(tmp_path)
    legacy = {
        "procedure": "perform the existing agent check",
        "artifact": "existing evidence", "level": "existing",
        "basis": BASIS_WORLD, "strength": "decisive", "bridges": [],
    }

    result = routing.discharge(
        ROW, Obligation("c1", "a semantic condition", BASIS_WORLD), legacy)

    assert result.instrument == "fallback"
    assert len(fallback.calls) == 1
    assert not any(event["type"] == "adapter_fallback"
                   for event in log.replay())


def test_default_online_and_offline_kernel_path_installs_adapter_router(
        tmp_path):
    service = Service(
        str(tmp_path / "gv"), kernel_factory=lambda _service: object())
    kernel = _default_kernel(
        service, model=None, backend=FixtureBackend({}), critic_rounds=0)

    assert isinstance(kernel.discharger, AdapterRoutingDischarger)
    assert kernel.binder.discharger is kernel.discharger
    assert "filesystem.exists" in kernel.discharger.adapter_catalog()
