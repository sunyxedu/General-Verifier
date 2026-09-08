"""General, fail-closed checks for the declarative JSON predicate adapter."""
from __future__ import annotations

import json
import random

import pytest

from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_WORLD,
    OUTCOME_HOLDS,
    OUTCOME_REFUTED,
    WITNESS_ABSENCE,
    WITNESS_NONE,
    WITNESS_WORLD_STATE,
    Obligation,
    canonical,
)
from verifier.engine.game import BindingGame
from verifier.engine.typed_judger_packet import (
    TypedAdapterJudgerPacket,
    build_typed_adapter_judger_packet,
)
from verifier.execution.adapters import (
    AdapterRoutingDischarger,
    default_adapter_registry,
)
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend


ROW = {"id": "claim", "kind": "final", "text": "the delivery is complete"}
YES = {
    "sufficient": "yes",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "fresh independent review accepted the evidence",
}


class Fallback:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append({"method": method, "tag": tag})
        return DischargeResult(
            outcome=OUTCOME_HOLDS,
            witness=None,
            artifact=self.artifacts.put({
                "instrument": "json-predicate-fallback",
                "call": len(self.calls),
            }),
            instrument="json-predicate-fallback",
            witness_kind="execution",
        )


def environment(tmp_path):
    state = tmp_path / "state"
    world = state / "world" / "app"
    world.mkdir(parents=True)
    (state / "task.md").write_text(
        "Validate the delivered JSON result.", encoding="utf-8")
    (state / "world_manifest.json").write_text(json.dumps({
        "scheme": "filesystem", "captured_roots": ["/app"],
    }), encoding="utf-8")
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = Fallback(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=False)
    return state, world, artifacts, log, fallback, routing


def method(predicates, *, address="/app/result.json", raw=None):
    return {
        "procedure": "apply the declared closed JSON predicates",
        "artifact": "a bounded content-addressed predicate receipt",
        "level": "deterministic",
        "basis": BASIS_WORLD,
        "strength": "decisive",
        "bridges": [],
        "adapter": "document.json-predicate",
        "adapter_version": 1,
        "adapter_arguments": (json.dumps({
            "address": address, "predicates": predicates,
        }, ensure_ascii=True, sort_keys=True) if raw is None else raw),
        "adapter_inputs": [{
            "ref": address, "provenance": "delivered", "role": "subject",
        }],
    }


def obligation(*, absence=False):
    return Obligation(
        "claim", "the delivered JSON satisfies the declared structure",
        BASIS_WORLD, refutable_by_absence=absence)


def test_catalog_exposes_a_closed_machine_readable_predicate_union():
    catalog = json.loads(default_adapter_registry().catalog())
    entry = next(item for item in catalog
                 if item["name"] == "document.json-predicate")

    assert entry["accepted_method_reuse"] is False
    assert entry["read_only"] is True
    assert {variant["kind"] for variant in entry["predicate_variants"]} == {
        "valid", "equals", "array.contains", "array.length",
        "array.items-object",
    }
    assert entry["resource_limits"]["predicates"] == 16
    assert "command" not in json.dumps(entry).lower()
    assert default_adapter_registry().supports_method_reuse(
        method([{"kind": "valid"}])) is False


def test_conjoined_predicates_emit_replayable_bounded_evidence(tmp_path):
    _, world, artifacts, _, fallback, routing = environment(tmp_path)
    document = {
        "summary": {"count": 2, "ok": True},
        "items": [
            {"id": 1, "name": "alpha", "active": True},
            {"id": 2, "name": "beta", "active": False},
        ],
    }
    (world / "result.json").write_text(
        json.dumps(document), encoding="utf-8")
    predicates = [
        {"kind": "valid"},
        {"kind": "equals", "pointer": "/summary",
         "expected": {"ok": True, "count": 2}},
        {"kind": "array.contains", "pointer": "/items",
         "expected": {"name": "beta", "active": False, "id": 2}},
        {"kind": "array.length", "pointer": "/items",
         "operator": "eq", "expected": 2},
        {"kind": "array.items-object", "pointer": "/items",
         "required": ["id", "name", "active"],
         "properties": {"id": "integer", "name": "string",
                        "active": "boolean"},
         "additional_properties": False},
    ]
    typed = method(predicates)

    result = routing.discharge(ROW, obligation(), typed)

    assert result.outcome == OUTCOME_HOLDS
    assert result.witness_kind == WITNESS_NONE
    assert fallback.calls == []
    evidence = artifacts.get(result.artifact)
    observation = evidence["execution"]["observation"]
    assert observation["strict_json_valid"] is True
    assert observation["predicate_count"] == len(predicates)
    assert all(item["matched"] for item in observation["results"])
    valid, reason = routing.registry.validate_evidence(
        typed, ROW, obligation(), result, artifacts)
    assert valid, reason


@pytest.mark.parametrize(("actual", "different"), [
    (True, 1),
    (1, 1.0),
    (None, "null"),
    ([], {}),
    ([1, 2], [2, 1]),
    ({"nested": [False, 0, "0"]}, {"nested": [False, 0, 0]}),
])
def test_equals_is_type_sensitive_and_structural(
        tmp_path, actual, different):
    _, world, _, _, fallback, routing = environment(tmp_path)
    (world / "result.json").write_text(
        json.dumps({"value": actual}), encoding="utf-8")

    holds = routing.discharge(
        ROW, obligation(), method([{
            "kind": "equals", "pointer": "/value", "expected": actual,
        }]))
    refuted = routing.discharge(
        ROW, obligation(), method([{
            "kind": "equals", "pointer": "/value", "expected": different,
        }]))

    assert holds.outcome == OUTCOME_HOLDS
    assert refuted.outcome == OUTCOME_REFUTED
    assert refuted.witness_kind == WITNESS_WORLD_STATE
    assert fallback.calls == []


def test_array_contains_requires_one_exact_complete_value(tmp_path):
    _, world, artifacts, _, fallback, routing = environment(tmp_path)
    (world / "result.json").write_text(json.dumps({
        "values": [True, 1, {"id": 7, "label": "complete"}],
    }), encoding="utf-8")
    exact = method([{
        "kind": "array.contains", "pointer": "/values",
        "expected": {"label": "complete", "id": 7},
    }])
    subset = method([{
        "kind": "array.contains", "pointer": "/values",
        "expected": {"id": 7},
    }])
    bool_is_not_integer = method([{
        "kind": "array.contains", "pointer": "/values", "expected": 1.0,
    }])

    holds = routing.discharge(ROW, obligation(absence=True), exact)
    absent_subset = routing.discharge(ROW, obligation(absence=True), subset)
    absent_float = routing.discharge(
        ROW, obligation(absence=True), bool_is_not_integer)

    assert holds.outcome == OUTCOME_HOLDS
    assert absent_subset.outcome == absent_float.outcome == OUTCOME_REFUTED
    assert absent_subset.witness_kind == absent_float.witness_kind \
        == WITNESS_ABSENCE
    observation = artifacts.get(holds.artifact)["execution"]["observation"]
    assert observation["results"][0]["first_match_index"] == 2
    assert fallback.calls == []


def test_array_length_preserves_absence_vs_positive_contradiction(tmp_path):
    _, world, _, _, fallback, routing = environment(tmp_path)
    (world / "result.json").write_text(
        json.dumps({"items": [1, 2, 3]}), encoding="utf-8")
    too_short = method([{
        "kind": "array.length", "pointer": "/items",
        "operator": "gte", "expected": 4,
    }])
    too_long = method([{
        "kind": "array.length", "pointer": "/items",
        "operator": "lte", "expected": 2,
    }])

    assert routing.discharge(
        ROW, obligation(), too_short).instrument == "json-predicate-fallback"
    absent = routing.discharge(ROW, obligation(absence=True), too_short)
    positive = routing.discharge(ROW, obligation(), too_long)

    assert absent.outcome == positive.outcome == OUTCOME_REFUTED
    assert absent.witness_kind == WITNESS_ABSENCE
    assert positive.witness_kind == WITNESS_WORLD_STATE
    assert len(fallback.calls) == 1


def test_missing_pointer_and_file_follow_explicit_absence_policy(tmp_path):
    _, world, _, _, fallback, routing = environment(tmp_path)
    (world / "result.json").write_text('{"present":1}', encoding="utf-8")
    missing_pointer = method([{
        "kind": "equals", "pointer": "/missing", "expected": 1,
    }])

    assert routing.discharge(
        ROW, obligation(), missing_pointer).instrument \
        == "json-predicate-fallback"
    pointer_refutation = routing.discharge(
        ROW, obligation(absence=True), missing_pointer)
    file_refutation = routing.discharge(
        ROW, obligation(absence=True), method(
            [{"kind": "valid"}], address="/app/missing.json"))

    assert pointer_refutation.outcome == file_refutation.outcome \
        == OUTCOME_REFUTED
    assert pointer_refutation.witness_kind == file_refutation.witness_kind \
        == WITNESS_ABSENCE
    assert len(fallback.calls) == 1


def test_array_item_shape_types_and_empty_array_semantics(tmp_path):
    _, world, artifacts, _, fallback, routing = environment(tmp_path)
    shape = {
        "kind": "array.items-object", "pointer": "/items",
        "required": ["id", "score"],
        "properties": {"id": "integer", "score": "number",
                       "note": "string"},
        "additional_properties": False,
    }
    (world / "result.json").write_text(json.dumps({
        "items": [{"id": 1, "score": 1},
                  {"id": 2, "score": 1.5, "note": "ok"}],
    }), encoding="utf-8")
    assert routing.discharge(
        ROW, obligation(), method([shape])).outcome == OUTCOME_HOLDS

    (world / "result.json").write_text(json.dumps({
        "items": [{"id": True, "score": 1, "extra": None}],
    }), encoding="utf-8")
    wrong = routing.discharge(ROW, obligation(), method([shape]))
    wrong_observation = artifacts.get(
        wrong.artifact)["execution"]["observation"]["results"][0]
    assert wrong.outcome == OUTCOME_REFUTED
    assert wrong.witness_kind == WITNESS_WORLD_STATE
    assert wrong_observation["failure_count"] == 2

    (world / "result.json").write_text(
        json.dumps({"items": [{}]}), encoding="utf-8")
    assert routing.discharge(
        ROW, obligation(), method([shape])).instrument == "json-predicate-fallback"
    missing = routing.discharge(ROW, obligation(absence=True), method([shape]))
    assert missing.outcome == OUTCOME_REFUTED
    assert missing.witness_kind == WITNESS_ABSENCE

    (world / "result.json").write_text(
        json.dumps({"items": []}), encoding="utf-8")
    empty = routing.discharge(ROW, obligation(), method([shape]))
    assert empty.outcome == OUTCOME_HOLDS
    assert fallback.calls and len(fallback.calls) == 1


def test_array_item_failure_receipt_is_bounded_but_counts_complete_scan(
        tmp_path):
    _, world, artifacts, _, fallback, routing = environment(tmp_path)
    (world / "result.json").write_text(
        json.dumps({"items": list(range(100))}), encoding="utf-8")
    shape = {
        "kind": "array.items-object", "pointer": "/items",
        "required": [], "properties": {}, "additional_properties": True,
    }

    result = routing.discharge(ROW, obligation(), method([shape]))

    receipt = artifacts.get(
        result.artifact)["execution"]["observation"]["results"][0]
    assert result.outcome == OUTCOME_REFUTED
    assert receipt["items_examined"] == receipt["failure_count"] == 100
    assert len(receipt["failures"]) == 64
    assert receipt["failures_truncated"] == 36
    assert fallback.calls == []


@pytest.mark.parametrize("raw", [
    b'{"value":1,"value":2}',
    b'{"value":NaN}',
    b'{"value":Infinity}',
    b'{"value":1e999}',
    b'{"value":"\xff"}',
    b'\xef\xbb\xbf{"value":1}',
    b'{"value":1} trailing',
    b'{"value":',
    b'{"value":"\\ud800"}',
])
def test_valid_predicate_rejects_non_strict_json(tmp_path, raw):
    _, world, artifacts, _, fallback, routing = environment(tmp_path)
    (world / "result.json").write_bytes(raw)

    result = routing.discharge(
        ROW, obligation(), method([{"kind": "valid"}]))

    assert result.outcome == OUTCOME_REFUTED
    assert result.witness_kind == WITNESS_WORLD_STATE
    observation = artifacts.get(result.artifact)["execution"]["observation"]
    assert observation["strict_json_valid"] is False
    assert fallback.calls == []


def test_valid_but_overdeep_document_falls_back_as_resource_limit(tmp_path):
    _, world, _, _, fallback, routing = environment(tmp_path)
    value = None
    for _index in range(140):
        value = [value]
    (world / "result.json").write_text(json.dumps(value), encoding="utf-8")

    result = routing.discharge(
        ROW, obligation(), method([{"kind": "valid"}]))

    assert result.instrument == "json-predicate-fallback"
    assert len(fallback.calls) == 1


def test_valid_but_oversized_integer_falls_back_as_resource_limit(tmp_path):
    _, world, _, _, fallback, routing = environment(tmp_path)
    (world / "result.json").write_text(
        '{"value":' + ("1" * 4_100) + '}', encoding="utf-8")

    result = routing.discharge(
        ROW, obligation(), method([{"kind": "valid"}]))

    assert result.instrument == "json-predicate-fallback"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("predicates", [
    [],
    [{"kind": "unknown"}],
    [{"kind": "valid", "pointer": ""}],
    [{"kind": "equals", "pointer": "/value"}],
    [{"kind": "equals", "pointer": "not-a-pointer", "expected": 1}],
    [{"kind": "array.length", "pointer": "/items",
      "operator": "eq", "expected": True}],
    [{"kind": "array.length", "pointer": "/items",
      "operator": "ne", "expected": 1}],
    [{"kind": "array.items-object", "pointer": "/items",
      "required": ["id", "id"], "properties": {"id": "integer"},
      "additional_properties": False}],
    [{"kind": "array.items-object", "pointer": "/items",
      "required": ["id"], "properties": {},
      "additional_properties": False}],
    [{"kind": "array.items-object", "pointer": "/items",
      "required": [], "properties": {"id": "callable"},
      "additional_properties": False}],
])
def test_malformed_or_open_ended_predicates_fall_back(tmp_path, predicates):
    _, world, _, _, fallback, routing = environment(tmp_path)
    (world / "result.json").write_text(
        json.dumps({"value": 1, "items": []}), encoding="utf-8")

    result = routing.discharge(ROW, obligation(), method(predicates))

    assert result.instrument == "json-predicate-fallback"
    assert len(fallback.calls) == 1


def test_path_escape_symlink_wrong_basis_and_missing_subject_fall_back(tmp_path):
    _, world, _, _, fallback, routing = environment(tmp_path)
    (world / "real.json").write_text('{"value":1}', encoding="utf-8")
    (world / "linked.json").symlink_to(world / "real.json")
    candidates = [
        method([{"kind": "valid"}], address="/app/../../etc/passwd"),
        method([{"kind": "valid"}], address="/app/linked.json"),
        {**method([{"kind": "valid"}]), "basis": "record"},
        {**method([{"kind": "valid"}]), "adapter_inputs": []},
    ]

    results = [routing.discharge(ROW, obligation(), candidate)
               for candidate in candidates]

    assert all(result.instrument == "json-predicate-fallback"
               for result in results)
    assert len(fallback.calls) == len(candidates)


def _random_json(rng, depth=0):
    scalars = [None, False, True, rng.randrange(-5, 6),
               rng.randrange(-10, 11) / 2, "s" + str(rng.randrange(5))]
    if depth >= 2:
        return rng.choice(scalars)
    choice = rng.randrange(4)
    if choice == 0:
        return rng.choice(scalars)
    if choice == 1:
        return [_random_json(rng, depth + 1) for _ in range(rng.randrange(4))]
    return {f"k{index}": _random_json(rng, depth + 1)
            for index in range(rng.randrange(4))}


def test_seeded_predicate_properties_match_independent_reference(tmp_path):
    _, world, artifacts, _, fallback, routing = environment(tmp_path)
    rng = random.Random(20260908)
    for _case in range(100):
        items = [_random_json(rng) for _ in range(rng.randrange(7))]
        expected = (_random_json(rng) if not items or rng.randrange(2)
                    else rng.choice(items))
        document = {"items": items, "value": _random_json(rng)}
        (world / "result.json").write_text(
            json.dumps(document, allow_nan=False), encoding="utf-8")
        predicates = [{
            "kind": "array.contains", "pointer": "/items",
            "expected": expected,
        }, {
            "kind": "array.length", "pointer": "/items",
            "operator": "eq", "expected": len(items),
        }, {
            "kind": "equals", "pointer": "/value",
            "expected": document["value"],
        }]
        typed = method(predicates)
        result = routing.discharge(ROW, obligation(absence=True), typed)
        reference_contains = any(
            canonical(item) == canonical(expected) for item in items)
        assert result.outcome == (
            OUTCOME_HOLDS if reference_contains else OUTCOME_REFUTED)
        valid, reason = routing.registry.validate_evidence(
            typed, ROW, obligation(absence=True), result, artifacts)
        assert valid, reason
    assert fallback.calls == []


def test_typed_packet_replays_but_never_replaces_fresh_judger(tmp_path):
    _, world, artifacts, log, fallback, routing = environment(tmp_path)
    target = world / "result.json"
    target.write_text(json.dumps({"items": [1]}), encoding="utf-8")
    typed = method([{
        "kind": "array.length", "pointer": "/items",
        "operator": "eq", "expected": 1,
    }])
    check = obligation()
    result = routing.discharge(ROW, check, typed)
    packet = build_typed_adapter_judger_packet(
        routing, artifacts, check, typed, result, "decisive")

    assert type(packet) is TypedAdapterJudgerPacket
    assert packet.revalidate() is True
    assert json.loads(packet.render())["adapter_contract"]["identity"] \
        == "document.json-predicate@1"
    target.write_text(json.dumps({"items": []}), encoding="utf-8")
    assert packet.revalidate() is False

    # The ordinary game still calls Oracle and one independent Judger; only
    # its executor seat is replaced by the adapter.
    target.write_text(json.dumps({"items": [1]}), encoding="utf-8")
    backend = FixtureBackend({"oracles": {"methods": [typed]}, "judger": YES})
    game = BindingGame(
        backend, artifacts, log, routing, live_world=False,
        typed_adapter_judger_packet=True)
    settled = game.run(ROW, check)
    assert settled.outcome == OUTCOME_HOLDS
    assert [call["role"] for call in backend.calls] == ["oracles", "judger"]
    assert type(backend.calls[1]["variables"][
        "_typed_adapter_judger_packet"]) is TypedAdapterJudgerPacket
    assert fallback.calls == []


def test_non_finite_method_arguments_fall_back_before_execution(tmp_path):
    _, world, _, _, fallback, routing = environment(tmp_path)
    (world / "result.json").write_text('{"value":1}', encoding="utf-8")
    raw = ('{"address":"/app/result.json","predicates":['
           '{"kind":"equals","pointer":"/value","expected":1e999}]}')

    result = routing.discharge(
        ROW, obligation(), method([], raw=raw))

    assert result.instrument == "json-predicate-fallback"
    assert len(fallback.calls) == 1
