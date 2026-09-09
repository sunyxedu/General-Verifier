"""Security, generality, and fallback tests for the pure evidence VM."""
from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

import verifier.execution.evidence_program as evidence_program
from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_WORLD,
    OUTCOME_FAILED,
    OUTCOME_HOLDS,
    Obligation,
)
from verifier.engine.game import BindingGame
from verifier.engine.typed_judger_packet import (
    TypedAdapterJudgerPacket,
    build_typed_adapter_judger_packet,
)
from verifier.execution.adapters import AdapterRoutingDischarger, default_adapter_registry
from verifier.execution.discharge import DischargeResult
from verifier.execution.evidence_program import (
    COMPLETE_AUDIT_OBSERVATION_SCHEMA,
    LANGUAGE,
    MAX_ASSERTIONS,
    MAX_AST_NODES,
    MAX_CALL_DEPTH,
    MAX_INTEGER_BITS,
    MAX_OBSERVATIONS,
    MAX_SOURCE_CHARS,
    WORLD_INPUT_SNAPSHOT_SCHEMA,
)
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend


ROW = {"id": "claim", "kind": "final", "text": "the delivery is complete"}


class _Fallback:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append({"method": method, "tag": tag})
        return DischargeResult(
            outcome=OUTCOME_HOLDS,
            witness=None,
            artifact=self.artifacts.put({"fallback": len(self.calls)}),
            instrument="agent:fallback",
            witness_kind="execution",
        )


def _environment(tmp_path):
    state = tmp_path / "state"
    world = state / "world" / "app"
    world.mkdir(parents=True)
    (state / "task.md").write_text("Validate the delivered result.", encoding="utf-8")
    (state / "world_manifest.json").write_text(json.dumps({
        "scheme": "filesystem", "captured_roots": ["/app"],
    }), encoding="utf-8")
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = _Fallback(artifacts)
    route = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=False,
    )
    obligation = Obligation(
        "claim", "the declared inputs satisfy the stated relation", BASIS_WORLD)
    return state, world, artifacts, fallback, route, obligation


def _input(ref, *, provenance="delivered", role="subject"):
    return {"ref": ref, "provenance": provenance, "role": role}


def _method(source, inputs, *, version=1):
    return {
        "procedure": "independently parse every declared input and decide the relation",
        "artifact": "a bounded pure-program execution receipt",
        "level": "deterministic pure inspection",
        "basis": BASIS_WORLD,
        "strength": "decisive",
        "bridges": [],
        "adapter": "evidence.program",
        "adapter_version": version,
        "adapter_arguments": json.dumps({"source": source}, sort_keys=True),
        "adapter_inputs": inputs,
    }


def test_catalog_describes_a_closed_pure_language():
    catalog = json.loads(default_adapter_registry().catalog())
    entry = next(item for item in catalog if item["name"] == "evidence.program")

    assert entry["version"] == 1
    assert entry["basis"] == BASIS_WORLD
    assert entry["accepted_method_reuse"] is False
    assert entry["language"]["identity"] == LANGUAGE
    assert "import" in entry["language"]["excluded"]
    assert "process" in entry["language"]["excluded"]
    assert entry["direction"].startswith("holds-only")
    assert entry["resource_limits"]["source_chars"] == MAX_SOURCE_CHARS
    assert "every require" in entry["typed_transport"]["requirement"]
    assert "input_text(0)" in entry["typed_transport"]["example"]
    assert "ordinary fresh toolful Judger" in entry["typed_transport"]["fallback"]
    assert "SQLite" not in json.dumps(entry)
    assert "recover.json" not in json.dumps(entry)


def test_complete_audit_catalog_is_method_failure_only_and_not_typed():
    catalog = json.loads(default_adapter_registry().catalog())
    entries = [item for item in catalog if item["name"] == "evidence.program"]

    assert [item["version"] for item in entries] == [1, 2]
    complete = entries[1]
    assert complete["basis"] == BASIS_WORLD
    assert complete["accepted_method_reuse"] is False
    assert "method failed" in complete["direction"]
    assert "never an obligation verdict" in complete[
        "complete_audit"]["false_requirement"]
    assert complete["complete_audit"]["directory_completeness"] is False
    assert complete["complete_audit"]["future_closure"] is False
    assert complete["typed_transport"]["eligible"] is False
    statements = " ".join(complete["language"]["statements"])
    assert "require(bool,label,observed)" in statements
    assert "require(bool,label,observed?)" not in statements


@pytest.mark.parametrize("refutable", [False, True])
def test_complete_audit_false_requirement_is_failed_never_refuted(
    tmp_path, refutable,
):
    _, world, artifacts, fallback, route, _obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")
    obligation = Obligation(
        "claim", "the declared input has the required value", BASIS_WORLD,
        refutable_by_absence=refutable)
    source = (
        'value = input_text(0)\n'
        'require(value == "beta", "exact value", value)'
    )
    method = _method(source, [_input("/app/value.txt")], version=2)

    result = route.discharge(ROW, obligation, method)

    assert result.instrument == "adapter:evidence.program@2"
    assert result.outcome == OUTCOME_FAILED
    assert result.witness_kind == "none"
    assert fallback.calls == []
    execution = artifacts.get(result.artifact)["execution"]
    observation = execution["observation"]
    assert observation["schema"] == COMPLETE_AUDIT_OBSERVATION_SCHEMA
    assert observation["complete_audit"] == {
        "audit_complete": True,
        "program_outcome": "require-false",
        "failed_assertion_index": 0,
        "require_is_terminal": True,
        "obligation_verdict": False,
        "proves_future_closure": False,
    }
    assert observation["assertions"][0]["holds"] is False
    assert observation["snapshot"]["schema"] == WORLD_INPUT_SNAPSHOT_SCHEMA
    assert observation["snapshot"]["complete_regular_files"] is True
    assert observation["snapshot"]["proves_directory_completeness"] is False
    assert build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result, "none") is None


def test_complete_audit_positive_uses_adapter_but_not_restricted_packet(
    tmp_path,
):
    _, world, artifacts, fallback, route, obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")
    method = _method(
        'value = input_text(0)\n'
        'require(value == "alpha", "exact value", value)',
        [_input("/app/value.txt")], version=2)

    result = route.discharge(ROW, obligation, method)

    assert result.instrument == "adapter:evidence.program@2"
    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    observation = artifacts.get(result.artifact)["execution"]["observation"]
    assert observation["complete_audit"]["program_outcome"] \
        == "all-requirements-held"
    assert build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result, "none") is None


@pytest.mark.parametrize("source,inputs", [
    (
        'value = input_text(0)\n'
        'require(value == "beta", "missing observed")',
        [_input("/app/value.txt")],
    ),
    (
        'value = input_text(0)\n'
        'require(False, "constant false", value)',
        [_input("/app/value.txt")],
    ),
    (
        'value = input_text(0)\n'
        'require(value == "beta", "unread second", value)',
        [_input("/app/value.txt"), _input("/app/other.txt", role="input")],
    ),
    (
        'value = input_text(0)\n'
        'require(value == "beta", "uncovered", "constant")',
        [_input("/app/value.txt")],
    ),
])
def test_complete_audit_invalid_or_incomplete_false_still_falls_back(
    tmp_path, source, inputs,
):
    _, world, _artifacts, fallback, route, obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")
    (world / "other.txt").write_text("other", encoding="utf-8")

    result = route.discharge(
        ROW, obligation, _method(source, inputs, version=2))

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


def test_complete_audit_binary_false_is_a_bounded_method_failure(tmp_path):
    _, world, artifacts, fallback, route, obligation = _environment(tmp_path)
    raw = bytes(range(256))
    (world / "data.bin").write_bytes(raw)
    source = (
        'raw = input_bytes(0)\n'
        'digest = sha256(raw)\n'
        f'require(digest == {"0" * 64!r}, "binary digest", digest)'
    )

    result = route.discharge(
        ROW, obligation,
        _method(source, [_input("/app/data.bin")], version=2))

    assert result.instrument == "adapter:evidence.program@2"
    assert result.outcome == OUTCOME_FAILED
    assert fallback.calls == []
    slot = artifacts.get(result.artifact)["execution"]["observation"][
        "input_slots"][0]
    assert slot["bytes"] == len(raw)
    assert len(slot["sha256"]) == 64


def test_complete_audit_two_file_snapshot_preserves_slot_order(tmp_path):
    _, world, artifacts, fallback, route, obligation = _environment(tmp_path)
    (world / "left.txt").write_text("left", encoding="utf-8")
    (world / "right.txt").write_text("right", encoding="utf-8")
    source = (
        'left = input_text(0)\n'
        'right = input_text(1)\n'
        'require(left == right, "same bytes", [left, right])'
    )

    result = route.discharge(
        ROW, obligation, _method(source, [
            _input("/app/left.txt"),
            _input("/app/right.txt", role="input"),
        ], version=2))

    assert result.outcome == OUTCOME_FAILED
    assert fallback.calls == []
    execution = artifacts.get(result.artifact)["execution"]
    assert [item["ref"] for item in execution["consumed"]] == [
        "/app/left.txt", "/app/right.txt"]
    assert execution["observation"]["used_input_slots"] == [0, 1]
    assert execution["observation"]["snapshot"]["input_count"] == 2


@pytest.mark.parametrize(("payload", "reader"), [
    (b"\xff", "input_text"),
    (b'{"duplicate":1,"duplicate":2}', "input_json"),
])
def test_complete_audit_decode_errors_remain_fallback(
    tmp_path, payload, reader,
):
    _, world, _artifacts, fallback, route, obligation = _environment(tmp_path)
    (world / "value.bin").write_bytes(payload)
    source = (
        f'value = {reader}(0)\n'
        'require(value == "different", "decoded value", value)'
    )

    result = route.discharge(
        ROW, obligation,
        _method(source, [_input("/app/value.bin")], version=2))

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("authorities", [
    [_input("task.md", provenance="delivered", role="authority")],
    [_input("task.md", provenance="task_given", role="authority")],
    [
        _input("task.md", provenance="task_given", role="authority"),
        _input("other.md", provenance="task_given", role="authority"),
    ],
])
def test_complete_audit_rejects_unbound_authority_shapes(
    tmp_path, authorities,
):
    _, world, _artifacts, fallback, route, obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")
    source = (
        'value = input_text(0)\n'
        'require(value == "beta", "exact value", value)'
    )

    result = route.discharge(
        ROW, obligation, _method(
            source, [_input("/app/value.txt"), *authorities], version=2))

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("unsafe_kind", [
    "leaf-symlink", "parent-symlink", "hardlink", "fifo",
])
def test_complete_audit_unsafe_files_fail_back_without_execution(
    tmp_path, unsafe_kind,
):
    _, world, _artifacts, fallback, route, obligation = _environment(tmp_path)
    ref = "/app/value.txt"
    target = world / "value.txt"
    if unsafe_kind == "leaf-symlink":
        real = world / "real.txt"
        real.write_text("alpha", encoding="utf-8")
        target.symlink_to(real)
    elif unsafe_kind == "parent-symlink":
        real = world / "real"
        real.mkdir()
        (real / "value.txt").write_text("alpha", encoding="utf-8")
        (world / "alias").symlink_to(real, target_is_directory=True)
        ref = "/app/alias/value.txt"
    elif unsafe_kind == "hardlink":
        real = world / "real.txt"
        real.write_text("alpha", encoding="utf-8")
        os.link(real, target)
    else:
        os.mkfifo(target)
    source = (
        'value = input_text(0)\n'
        'require(value == "beta", "exact value", value)'
    )

    result = route.discharge(
        ROW, obligation, _method(source, [_input(ref)], version=2))

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


def test_complete_audit_revalidates_all_files_after_the_program(
    tmp_path, monkeypatch,
):
    _, world, _artifacts, fallback, route, obligation = _environment(tmp_path)
    target = world / "value.txt"
    target.write_text("alpha", encoding="utf-8")
    original = evidence_program._revalidate_open_world_input
    calls = 0

    def mutate_before_final_revalidation(item):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.write_text("changed", encoding="utf-8")
        return original(item)

    monkeypatch.setattr(
        evidence_program, "_revalidate_open_world_input",
        mutate_before_final_revalidation)
    source = (
        'value = input_text(0)\n'
        'require(value == "beta", "exact value", value)'
    )

    result = route.discharge(
        ROW, obligation,
        _method(source, [_input("/app/value.txt")], version=2))

    assert result.instrument == "agent:fallback"
    assert calls == 2
    assert len(fallback.calls) == 1


def test_complete_audit_truncation_during_read_falls_back(
    tmp_path, monkeypatch,
):
    _, world, _artifacts, fallback, route, obligation = _environment(tmp_path)
    target = world / "value.txt"
    target.write_text("alpha", encoding="utf-8")
    original_read = evidence_program.os.read
    truncated = False

    def read_then_truncate(fd, size):
        nonlocal truncated
        chunk = original_read(fd, size)
        if not truncated:
            truncated = True
            target.write_bytes(b"a")
        return chunk

    monkeypatch.setattr(evidence_program.os, "read", read_then_truncate)
    source = (
        'value = input_text(0)\n'
        'require(value == "beta", "exact value", value)'
    )

    result = route.discharge(
        ROW, obligation,
        _method(source, [_input("/app/value.txt")], version=2))

    assert result.instrument == "agent:fallback"
    assert truncated is True
    assert len(fallback.calls) == 1


def test_complete_audit_revalidates_the_world_mapping(tmp_path, monkeypatch):
    _, world, _artifacts, fallback, route, obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")
    original = evidence_program.load_world_view
    calls = 0

    def changed_view(*args, **kwargs):
        nonlocal calls
        calls += 1
        view = original(*args, **kwargs)
        return replace(view, note="changed") if calls == 2 else view

    monkeypatch.setattr(evidence_program, "load_world_view", changed_view)
    source = (
        'value = input_text(0)\n'
        'require(value == "beta", "exact value", value)'
    )

    result = route.discharge(
        ROW, obligation,
        _method(source, [_input("/app/value.txt")], version=2))

    assert result.instrument == "agent:fallback"
    assert calls == 2
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("boundary", ["file", "total", "receipt"])
def test_complete_audit_boundaries_fail_back(tmp_path, monkeypatch, boundary):
    _, world, _artifacts, fallback, route, obligation = _environment(tmp_path)
    (world / "first.bin").write_bytes(b"abc")
    (world / "second.bin").write_bytes(b"def")
    inputs = [_input("/app/first.bin")]
    source = (
        'first = input_bytes(0)\n'
        'require(first == b"other", "bounded input", first)'
    )
    if boundary == "file":
        monkeypatch.setattr(evidence_program, "MAX_INPUT_BYTES", 2)
    elif boundary == "total":
        monkeypatch.setattr(evidence_program, "MAX_TOTAL_INPUT_BYTES", 5)
        inputs.append(_input("/app/second.bin", role="input"))
        source = (
            'first = input_bytes(0)\n'
            'second = input_bytes(1)\n'
            'require(first == second, "bounded inputs", [first, second])'
        )
    else:
        monkeypatch.setattr(evidence_program, "MAX_RECEIPT_BYTES", 1)

    result = route.discharge(
        ROW, obligation, _method(source, inputs, version=2))

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


def test_complete_audit_accepts_exact_file_and_total_byte_boundaries(
    tmp_path, monkeypatch,
):
    _, world, _artifacts, fallback, route, obligation = _environment(tmp_path)
    (world / "first.bin").write_bytes(b"abc")
    (world / "second.bin").write_bytes(b"def")
    monkeypatch.setattr(evidence_program, "MAX_INPUT_BYTES", 3)
    monkeypatch.setattr(evidence_program, "MAX_TOTAL_INPUT_BYTES", 6)
    source = (
        'first = input_bytes(0)\n'
        'second = input_bytes(1)\n'
        'require(first == second, "exact boundaries", [first, second])'
    )

    result = route.discharge(
        ROW, obligation, _method(source, [
            _input("/app/first.bin"),
            _input("/app/second.bin", role="input"),
        ], version=2))

    assert result.instrument == "adapter:evidence.program@2"
    assert result.outcome == OUTCOME_FAILED
    assert fallback.calls == []


def test_complete_audit_uses_the_same_point_snapshot_in_live_mode(tmp_path):
    state = tmp_path / "live-state"
    state.mkdir()
    (state / "task.md").write_text("Check the live value.", encoding="utf-8")
    target = tmp_path / "live-value.txt"
    target.write_text("alpha", encoding="utf-8")
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = _Fallback(artifacts)
    route = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=True)
    obligation = Obligation(
        "claim", "the live input has the required value", BASIS_WORLD)
    source = (
        'value = input_text(0)\n'
        'require(value == "beta", "live exact value", value)'
    )

    result = route.discharge(
        ROW, obligation,
        _method(source, [_input(str(target))], version=2))

    assert result.instrument == "adapter:evidence.program@2"
    assert result.outcome == OUTCOME_FAILED
    assert fallback.calls == []


def test_complete_audit_failure_keeps_existing_oracle_repair_and_judger(
    tmp_path,
):
    _, world, artifacts, fallback, route, obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")
    failed = _method(
        'value = input_text(0)\n'
        'require(value == "beta", "wrong value", value)',
        [_input("/app/value.txt")], version=2)
    repaired = _method(
        'value = input_text(0)\n'
        'require(value == "alpha", "exact value", value)',
        [_input("/app/value.txt")], version=2)
    repaired["procedure"] = "inspect the stable file with the repaired predicate"
    calls = 0

    def oracle(_variables):
        nonlocal calls
        calls += 1
        return {"methods": [failed] if calls == 1 else [repaired]}

    backend = FixtureBackend({
        "oracles": oracle,
        "judger": {
            "sufficient": "yes", "faithful": "yes", "degradation": "n/a",
            "reason": "the repaired positive program is sufficient",
        },
    })
    game = BindingGame(
        backend, artifacts, route.log, route, live_world=False)

    result = game.run(ROW, obligation)

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    assert [call["role"] for call in backend.calls] == [
        "oracles", "oracles", "judger"]
    assert [event["payload"]["outcome"] for event in route.log.replay()
            if event["type"] == "adapter_discharge"] == [
                OUTCOME_FAILED, OUTCOME_HOLDS]


def test_program_relates_two_declared_json_inputs_without_agent_execution(tmp_path):
    _, world, artifacts, fallback, route, obligation = _environment(tmp_path)
    (world / "source.json").write_text(
        '[{"id":"a","value":1},{"id":"b","value":2}]', encoding="utf-8")
    (world / "result.json").write_text(
        '[{"id":"a","value":1},{"id":"b","value":2}]', encoding="utf-8")
    source = """source = input_json(0)
result = input_json(1)
require(source == result, "result preserves every source item", {"source": source, "result": result})
observe("result_count", len(result))
"""
    method = _method(source, [
        _input("/app/source.json", provenance="task_given"),
        _input("/app/result.json"),
    ])

    result = route.discharge(ROW, obligation, method)

    assert result.instrument == "adapter:evidence.program@1"
    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    evidence = artifacts.get(result.artifact)
    execution = evidence["execution"]
    assert [item["ref"] for item in execution["consumed"]] == [
        "/app/source.json", "/app/result.json"]
    assert execution["observation"]["used_input_slots"] == [0, 1]
    assert execution["observation"]["assertions"][0]["holds"] is True
    assert execution["observation"]["assertions"][0]["subject_slots"] == [0, 1]
    assert execution["observation"]["observations"] == [
        {"label": "result_count", "value": 2}]
    packet = build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result,
        "(none — decisive method, no degradation to audit)",
    )
    assert isinstance(packet, TypedAdapterJudgerPacket)
    assert packet.revalidate() is True


def test_general_binary_primitives_can_validate_a_custom_format(tmp_path):
    _, world, artifacts, fallback, route, obligation = _environment(tmp_path)
    # Generic fixture format: count:u8 followed by repeated key_len:u8, key,
    # value:i16be.  There is no format-specific opcode in the VM.
    (world / "records.bin").write_bytes(
        bytes([2, 1]) + b"a" + bytes([0, 7, 1]) + b"b" + bytes([0, 9]))
    (world / "result.json").write_text(
        '[{"key":"a","value":7},{"key":"b","value":9}]', encoding="utf-8")
    source = """raw = input_bytes(0)
candidate = input_json(1)
count = raw[0]
offset = 1
decoded = []
for index in range(count):
    width = raw[offset]
    offset += 1
    key = utf8(raw[offset:offset + width])
    offset += width
    value = be_int(raw, offset, 2)
    offset += 2
    decoded = decoded + [{"key": key, "value": value}]
require(offset == len(raw), "the complete binary input was consumed", {"bytes": offset})
require(decoded == candidate, "candidate equals all decoded records", {"decoded": decoded, "candidate": candidate})
"""
    method = _method(source, [
        _input("/app/records.bin", provenance="task_given"),
        _input("/app/result.json"),
    ])

    result = route.discharge(ROW, obligation, method)

    assert result.instrument == "adapter:evidence.program@1"
    assert fallback.calls == []
    receipt = artifacts.get(result.artifact)["execution"]["observation"]
    assert [item["label"] for item in receipt["assertions"]] == [
        "the complete binary input was consumed",
        "candidate equals all decoded records",
    ]
    packet = build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result,
        "(none — decisive method, no degradation to audit)",
    )
    assert isinstance(packet, TypedAdapterJudgerPacket)
    assert packet.revalidate() is True


@pytest.mark.parametrize("source", [
    'value = input_text(0)\nrequire(value == "different", "must match")',
    'import os\nrequire(True, "not reachable")',
    'value = input_text(0)\nvalue.upper()\nrequire(True, "not reachable")',
    'require(True, "hard coded")',
    'value = input_text(0)\nrequire(True, "dummy read is not dependency")',
    'value = input_text(0)\nrequire(1, "not boolean")',
    'value = input_text(0)\nfor item in range(100001):\n    pass\nrequire(True, "too much")',
    'value = input_text(0)\ny = (1 << 4095) / 1\n'
    'require(value == "alpha", "overflow", value)',
    'value = input_text(0)\ny = (1 << 4095) + 1.0\n'
    'require(value == "alpha", "overflow", value)',
    'value = input_text(0)\ny = (1 << 4095) // 1.0\n'
    'require(value == "alpha", "overflow", value)',
    'value = input_text(0)\ny = (1 << 4095) % 1.0\n'
    'require(value == "alpha", "overflow", value)',
    'value = input_text(0)\nrequire(value is not value, "identity", value)',
])
def test_false_unsafe_unused_or_ill_typed_programs_fallback(tmp_path, source):
    _, world, _, fallback, route, obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")

    result = route.discharge(
        ROW, obligation, _method(source, [_input("/app/value.txt")]))

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1
    events = list(route.log.replay())
    assert any(event["type"] == "adapter_fallback" for event in events)


def test_duplicate_or_unresolved_runtime_inputs_fallback(tmp_path):
    _, world, _, fallback, route, obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")
    source = 'value = input_text(0)\nrequire(value == "alpha", "matches")'
    duplicated = [_input("/app/value.txt"), _input("/app/value.txt", role="input")]

    assert route.discharge(ROW, obligation, _method(source, duplicated)).instrument \
        == "agent:fallback"
    assert route.discharge(
        ROW, obligation,
        _method(source, [_input("/app/missing.txt")]),
    ).instrument == "agent:fallback"
    assert len(fallback.calls) == 2


def test_receipt_replays_exactly_and_input_mutation_changes_it(tmp_path):
    _, world, artifacts, fallback, route, obligation = _environment(tmp_path)
    target = world / "value.txt"
    target.write_text("alpha", encoding="utf-8")
    method = _method(
        'value = input_text(0)\nrequire(value == "alpha", "matches", value)',
        [_input("/app/value.txt")],
    )
    first = route.discharge(ROW, obligation, method)
    second = route.discharge(ROW, obligation, method)

    assert first.artifact == second.artifact
    assert fallback.calls == []
    valid, reason = route.registry.validate_evidence(
        method, ROW, obligation, first, artifacts)
    assert valid is True, reason

    target.write_text("beta", encoding="utf-8")
    third = route.discharge(ROW, obligation, method)
    assert third.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


def test_complete_small_program_is_eligible_for_fresh_typed_judger_packet(tmp_path):
    _, world, artifacts, fallback, route, obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")
    method = _method(
        'value = input_text(0)\nrequire(value == "alpha", "exact value", value)',
        [_input("/app/value.txt")],
    )
    result = route.discharge(ROW, obligation, method)

    packet = build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result,
        "(none — decisive method, no degradation to audit)",
    )

    assert fallback.calls == []
    assert isinstance(packet, TypedAdapterJudgerPacket)
    document = json.loads(packet.render())
    assert document["adapter_contract"]["identity"] == "evidence.program@1"
    receipt = document["verifier_execution_receipt"]["observation"]
    assert receipt["language"] == LANGUAGE
    assert receipt["assertions"][0]["subject_slots"] == [0]
    assert packet.revalidate() is True


def test_typed_packet_requires_a_complete_subject_dependent_observation(tmp_path):
    _, world, artifacts, fallback, route, obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")
    method = _method(
        'value = input_text(0)\nrequire(value == "alpha", "exact value")',
        [_input("/app/value.txt")],
    )
    result = route.discharge(ROW, obligation, method)

    packet = build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result,
        "(none — decisive method, no degradation to audit)",
    )

    assert result.instrument == "adapter:evidence.program@1"
    assert fallback.calls == []
    assert packet is None


def test_runtime_input_selector_dependency_is_retained_in_receipt(tmp_path):
    _, world, artifacts, fallback, route, obligation = _environment(tmp_path)
    (world / "selector.txt").write_text("first", encoding="utf-8")
    (world / "first.txt").write_text("alpha", encoding="utf-8")
    (world / "second.txt").write_text("beta", encoding="utf-8")
    source = """choice = input_text(0)
slot = 1 if choice == "first" else 2
selected = input_text(slot)
other_slot = 2 if slot == 1 else 1
other = input_text(other_slot)
require(selected == "alpha", "selected value", selected)
"""
    method = _method(source, [
        _input("/app/selector.txt"),
        _input("/app/first.txt"),
        _input("/app/second.txt", role="input"),
    ])

    result = route.discharge(ROW, obligation, method)

    assert result.instrument == "adapter:evidence.program@1"
    assert fallback.calls == []
    assertion = artifacts.get(result.artifact)["execution"]["observation"][
        "assertions"][0]
    assert assertion["input_slots"] == [0, 1]
    assert assertion["subject_slots"] == [0, 1]
    assert assertion["observed_input_slots"] == [0, 1]


def test_source_and_input_caps_fail_closed(tmp_path):
    _, world, _, fallback, route, obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")
    oversized = "x" * (MAX_SOURCE_CHARS + 1)

    result = route.discharge(
        ROW, obligation, _method(oversized, [_input("/app/value.txt")]))

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("hidden", [
    "if False:\n    import os",
    "if False:\n    while True:\n        pass",
    "if False:\n    value.upper()",
    "if False:\n    evil()",
    "if False:\n    len = 3",
    "def hidden():\n    import os",
    "def hidden(value: object):\n    return value",
    "def hidden():\n    return lambda: 1",
])
def test_forbidden_syntax_is_rejected_even_when_unreachable(tmp_path, hidden):
    _, world, _, fallback, route, obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")
    source = (
        f"{hidden}\n"
        'value = input_text(0)\n'
        'require(value == "alpha", "exact", value)'
    )

    result = route.discharge(
        ROW, obligation, _method(source, [_input("/app/value.txt")]))

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("source", [
    'value = input_text(0)\nrequire(True, value, value)',
    'value = input_text(0)\nrequire(True, "constant", value)',
    'def always(value):\n    return True\n'
    'value = input_text(0)\nrequire(always(value), "constant", value)',
    'value = input_text(0)\nrequire(len([]) == 0, "fresh constant", value)',
])
def test_subject_data_cannot_launder_a_constant_requirement(tmp_path, source):
    _, world, _, fallback, route, obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")

    result = route.discharge(
        ROW, obligation, _method(source, [_input("/app/value.txt")]))

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("source", [
    'value = input_text(0)\nrequire(value == "alpha", "direct", value)',
    'def exact(value):\n    return value == "alpha"\n'
    'value = input_text(0)\nrequire(exact(value), "function", value)',
    'value = input_text(0)\nresult = False\n'
    'if value == "alpha":\n    result = True\n'
    'require(result, "control", value)',
])
def test_real_subject_dependencies_survive_assignments_functions_and_control(
        tmp_path, source):
    _, world, _, fallback, route, obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")

    result = route.discharge(
        ROW, obligation, _method(source, [_input("/app/value.txt")]))

    assert result.instrument == "adapter:evidence.program@1"
    assert fallback.calls == []


@pytest.mark.parametrize(("document", "predicate"), [
    ("[1]", "value == [True]"),
    ('{"x":1}', 'value == {"x":1.0}'),
    ("[1]", "True in value"),
    ("[1]", "value[True] == 1"),
    ("[[1]]", "value < [[True]]"),
])
def test_nested_values_use_exact_types_and_boolean_is_not_an_index(
        tmp_path, document, predicate):
    _, world, _, fallback, route, obligation = _environment(tmp_path)
    (world / "value.json").write_text(document, encoding="utf-8")
    source = (
        'value = input_json(0)\n'
        f'require({predicate}, "strict relation", value)'
    )

    result = route.discharge(
        ROW, obligation, _method(source, [_input("/app/value.json")]))

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


def test_json_object_equality_is_order_independent_but_type_strict(tmp_path):
    _, world, _, fallback, route, obligation = _environment(tmp_path)
    (world / "value.json").write_text('{"a":1,"b":true}', encoding="utf-8")
    source = (
        'value = input_json(0)\n'
        'require(value == {"b": True, "a": 1}, "same object", value)'
    )

    result = route.discharge(
        ROW, obligation, _method(source, [_input("/app/value.json")]))

    assert result.instrument == "adapter:evidence.program@1"
    assert fallback.calls == []


@pytest.mark.parametrize("source", [
    "padding = [" + ",".join("0" for _ in range(MAX_AST_NODES)) + "]",
    'value = input_text(0)\nlarge = 1 << '
    f'{MAX_INTEGER_BITS}\nrequire(value == "alpha", "large", value)',
    'def recurse(value):\n    return recurse(value)\n'
    'value = input_text(0)\nrequire(recurse(value), "depth", value)',
    'value = input_text(0)\ncount = 0\n'
    'for index in range(100000):\n    count += index\n'
    'require(value == "alpha", "fuel", count)',
    'value = input_text(0)\n'
    + "\n".join(
        f'require(value == "alpha", "assertion {index}", value)'
        for index in range(MAX_ASSERTIONS + 1)),
    'value = input_text(0)\n'
    + "\n".join(
        f'observe("observation {index}", value)'
        for index in range(MAX_OBSERVATIONS + 1))
    + '\nrequire(value == "alpha", "exact", value)',
])
def test_computation_and_receipt_resource_caps_fail_closed(tmp_path, source):
    _, world, _, fallback, route, obligation = _environment(tmp_path)
    (world / "value.txt").write_text("alpha", encoding="utf-8")

    result = route.discharge(
        ROW, obligation, _method(source, [_input("/app/value.txt")]))

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


def test_node_rich_json_is_rejected_before_host_parser_allocation(
        tmp_path, monkeypatch):
    _, world, _, fallback, route, obligation = _environment(tmp_path)
    (world / "value.json").write_bytes(
        b"[" + b"{}," * 100_001 + b"{}]")
    parser_called = False

    def unexpected_parser(_data):
        nonlocal parser_called
        parser_called = True
        raise AssertionError("bounded preflight must run before json.loads")

    monkeypatch.setattr(
        evidence_program, "_strict_json_document", unexpected_parser)
    source = (
        'value = input_json(0)\n'
        'require(len(value) > 0, "nonempty", len(value))'
    )

    result = route.discharge(
        ROW, obligation, _method(source, [_input("/app/value.json")]))

    assert result.instrument == "agent:fallback"
    assert parser_called is False
    assert len(fallback.calls) == 1
