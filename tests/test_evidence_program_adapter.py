"""Security, generality, and fallback tests for the pure evidence VM."""
from __future__ import annotations

import json

import pytest

import verifier.execution.evidence_program as evidence_program
from verifier.core.events import EventLog
from verifier.core.model import BASIS_WORLD, OUTCOME_HOLDS, Obligation
from verifier.engine.typed_judger_packet import (
    TypedAdapterJudgerPacket,
    build_typed_adapter_judger_packet,
)
from verifier.execution.adapters import AdapterRoutingDischarger, default_adapter_registry
from verifier.execution.discharge import DischargeResult
from verifier.execution.evidence_program import (
    LANGUAGE,
    MAX_ASSERTIONS,
    MAX_AST_NODES,
    MAX_CALL_DEPTH,
    MAX_INTEGER_BITS,
    MAX_OBSERVATIONS,
    MAX_SOURCE_CHARS,
)
from verifier.execution.instruments import ArtifactStore


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


def _method(source, inputs):
    return {
        "procedure": "independently parse every declared input and decide the relation",
        "artifact": "a bounded pure-program execution receipt",
        "level": "deterministic pure inspection",
        "basis": BASIS_WORLD,
        "strength": "decisive",
        "bridges": [],
        "adapter": "evidence.program",
        "adapter_version": 1,
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
