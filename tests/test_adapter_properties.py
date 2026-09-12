"""Broader behavioral and concurrency checks for typed adapters."""
from __future__ import annotations

import hashlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_WORLD,
    OUTCOME_HOLDS,
    OUTCOME_REFUTED,
    Obligation,
)
from verifier.execution.adapters import (
    AdapterRoutingDischarger,
    default_adapter_registry,
)
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore


ROW = {"id": "claim", "kind": "final", "text": "delivery is complete"}


class Fallback:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append(method)
        return DischargeResult(
            outcome=OUTCOME_HOLDS,
            witness=None,
            artifact=self.artifacts.put({
                "instrument": "property-fallback", "call": len(self.calls)}),
            instrument="property-fallback",
            witness_kind="execution",
        )


def environment(tmp_path):
    state = tmp_path / "state"
    world = state / "world" / "app"
    world.mkdir(parents=True)
    (state / "world_manifest.json").write_text(json.dumps({
        "scheme": "filesystem", "captured_roots": ["/app"]}))
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = Fallback(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=False)
    return state, world, artifacts, log, fallback, routing


def method(name, arguments, address, *, arguments_raw=None):
    return {
        "procedure": f"execute {name}",
        "artifact": "typed evidence",
        "level": "deterministic",
        "basis": BASIS_WORLD,
        "strength": "decisive",
        "bridges": [],
        "adapter": name,
        "adapter_version": 1,
        "adapter_arguments": (json.dumps(arguments, sort_keys=True)
                              if arguments_raw is None else arguments_raw),
        "adapter_inputs": [{
            "ref": address, "provenance": "delivered", "role": "subject"}],
    }


@pytest.mark.parametrize(("actual", "different"), [
    (True, 1),
    (1, 1.0),
    (None, "null"),
    ("", 0),
    ([], {}),
    ([1], ["1"]),
    ({"nested": [False, 0, "0"]}, {"nested": [False, 0, 0]}),
])
def test_json_value_exact_equality_is_type_sensitive(
        tmp_path, actual, different):
    _, world, _, _, fallback, routing = environment(tmp_path)
    (world / "value.json").write_text(json.dumps({"value": actual}))
    obligation = Obligation("claim", "the exact JSON value matches", BASIS_WORLD)
    address = "/app/value.json"

    holds = routing.discharge(
        ROW, obligation,
        method("document.json-value", {
            "address": address, "pointer": "/value", "expected": actual,
        }, address))
    refuted = routing.discharge(
        ROW, obligation,
        method("document.json-value", {
            "address": address, "pointer": "/value", "expected": different,
        }, address))

    assert holds.outcome == OUTCOME_HOLDS
    assert refuted.outcome == OUTCOME_REFUTED
    assert fallback.calls == []


def test_json_pointer_escaping_and_absence_policy(tmp_path):
    _, world, _, _, fallback, routing = environment(tmp_path)
    (world / "pointer.json").write_text(json.dumps({
        "a/b": {"~key": [{"": "found"}]}}))
    address = "/app/pointer.json"
    exact = method("document.json-value", {
        "address": address, "pointer": "/a~1b/~0key/0/",
        "expected": "found",
    }, address)
    missing = method("document.json-value", {
        "address": address, "pointer": "/a~1b/~0key/1",
        "expected": "found",
    }, address)

    assert routing.discharge(
        ROW, Obligation("claim", "escaped member matches", BASIS_WORLD),
        exact).outcome == OUTCOME_HOLDS
    assert routing.discharge(
        ROW, Obligation("claim", "member has a property", BASIS_WORLD),
        missing).instrument == "property-fallback"
    absent = routing.discharge(
        ROW, Obligation("claim", "member exists", BASIS_WORLD,
                        refutable_by_absence=True), missing)
    assert absent.outcome == OUTCOME_REFUTED
    assert absent.witness_kind == "absence"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("size", [0, 1, 4095, 65_536, 1_048_576])
def test_sha256_adapter_covers_file_size_boundaries(tmp_path, size):
    _, world, artifacts, _, fallback, routing = environment(tmp_path)
    rng = random.Random(size)
    data = rng.randbytes(size)
    (world / "blob.bin").write_bytes(data)
    address = "/app/blob.bin"
    expected = hashlib.sha256(data).hexdigest()
    obligation = Obligation("claim", "the exact bytes match", BASIS_WORLD)
    typed = method("filesystem.sha256", {
        "address": address, "expected_sha256": expected,
    }, address)

    holds = routing.discharge(ROW, obligation, typed)
    (world / "blob.bin").write_bytes(data + b"x")
    refuted = routing.discharge(ROW, obligation, typed)

    assert holds.outcome == OUTCOME_HOLDS
    assert refuted.outcome == OUTCOME_REFUTED
    assert holds.artifact != refuted.artifact
    valid, reason = routing.registry.validate_evidence(
        typed, ROW, obligation, refuted, artifacts)
    assert valid, reason
    assert fallback.calls == []


def test_symlinks_and_non_strict_documents_never_get_silent_fast_pass(
        tmp_path):
    _, world, _, _, fallback, routing = environment(tmp_path)
    (world / "real.json").write_text('{"value":1}')
    os.symlink(world / "real.json", world / "linked.json")
    address = "/app/linked.json"
    obligation = Obligation("claim", "document value matches", BASIS_WORLD)
    linked = method("document.json-value", {
        "address": address, "pointer": "/value", "expected": 1,
    }, address)
    assert routing.discharge(ROW, obligation, linked).instrument \
        == "property-fallback"

    cases = {
        "duplicate.json": b'{"value":1,"value":1}',
        "nan.json": b'{"value":NaN}',
        "utf8.json": b'{"value":"\xff"}',
    }
    for filename, data in cases.items():
        (world / filename).write_bytes(data)
        case_address = f"/app/{filename}"
        result = routing.discharge(
            ROW, obligation,
            method("document.json-value", {
                "address": case_address, "pointer": "/value", "expected": 1,
            }, case_address))
        assert result.outcome == OUTCOME_REFUTED
        assert result.witness_kind == "world_state"
    assert len(fallback.calls) == 1


def test_concurrent_identical_adapter_writes_are_content_addressed(tmp_path):
    _, world, artifacts, log, fallback, routing = environment(tmp_path)
    (world / "result.json").write_text('{"status":"completed"}')
    address = "/app/result.json"
    typed = method("document.json-value", {
        "address": address, "pointer": "/status", "expected": "completed",
    }, address)
    obligation = Obligation("claim", "status equals completed", BASIS_WORLD)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(
            lambda _index: routing.discharge(ROW, obligation, typed),
            range(128)))

    assert {result.outcome for result in results} == {OUTCOME_HOLDS}
    assert len({result.artifact for result in results}) == 1
    for result in results:
        valid, reason = routing.registry.validate_evidence(
            typed, ROW, obligation, result, artifacts)
        assert valid, reason
    events = list(log.replay())
    assert [event["seq"] for event in events] == list(range(1, 129))
    assert fallback.calls == []


def test_malformed_methodir_fuzz_falls_back_without_raising(tmp_path):
    _, world, _, _, fallback, routing = environment(tmp_path)
    (world / "result.json").write_text('{"status":"completed"}')
    address = "/app/result.json"
    obligation = Obligation("claim", "status equals completed", BASIS_WORLD)
    base = method("document.json-value", {
        "address": address, "pointer": "/status", "expected": "completed",
    }, address)
    rng = random.Random(73)
    fields = ["adapter", "adapter_version", "adapter_arguments",
              "adapter_inputs"]
    malformed = []
    for _index in range(250):
        value = dict(base)
        action = rng.randrange(5)
        field = rng.choice(fields)
        if action == 0:
            value.pop(field)
        elif action == 1:
            value[field] = None
        elif action == 2:
            value[field] = rng.choice([True, -1, [], {}, "not-json"])
        elif action == 3:
            value["adapter"] = "unknown." + str(rng.randrange(100))
        else:
            value["adapter_arguments"] = rng.choice([
                "{", "[]", '{"address":"/app/result.json","x":NaN}',
                '{"address":"a","address":"b"}',
            ])
        malformed.append(value)

    results = [routing.discharge(ROW, obligation, value)
               for value in malformed]

    assert all(result.instrument == "property-fallback" for result in results)
    assert len(fallback.calls) == len(malformed)
