"""General predicate semantics for ``ledger.tool-call-string@2``.

The adapter selects retained strings only.  These fixtures are inert ledger
records; no selected value is parsed as code or executed.
"""
from __future__ import annotations

import json

import pytest

from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_RECORD,
    OUTCOME_HOLDS,
    OUTCOME_REFUTED,
    WITNESS_ABSENCE,
    WITNESS_RECORD_CONTRADICTION,
    Obligation,
    content_hash,
)
from verifier.engine.game import BindingGame
from verifier.engine.method_cache import AcceptedMethodCache
from verifier.execution.adapters import (
    AdapterRegistry,
    AdapterRoutingDischarger,
    LedgerToolCallStringAdapter,
    LedgerToolCallStringAdapterV2,
    default_adapter_registry,
)
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend


ROW = {"id": "claim-1", "kind": "final", "text": "retained string claim"}
JUDGE_YES = {
    "sufficient": "yes",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "the exact bounded ledger predicate decides the obligation",
}


class FallbackDischarger:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append({"method": method, "tag": tag})
        digest = self.artifacts.put({"instrument": "fallback"})
        return DischargeResult(
            outcome=OUTCOME_HOLDS,
            witness=None,
            artifact=digest,
            instrument="fallback",
            witness_kind="execution",
        )


def adapter_method(arguments, *, version=2):
    return {
        "procedure": "evaluate one exact predicate over retained strings",
        "artifact": "a content-addressed bounded ledger predicate result",
        "level": "deterministic",
        "basis": BASIS_RECORD,
        "strength": "decisive",
        "bridges": [],
        "adapter": "ledger.tool-call-string",
        "adapter_version": version,
        "adapter_arguments": json.dumps(
            arguments, ensure_ascii=False, sort_keys=True),
        "adapter_inputs": [{
            "ref": "ledger:events",
            "provenance": "ledger",
            "role": "subject",
        }],
    }


def harness(tmp_path, *, default_registry=False, live_world=False):
    state = tmp_path / "gv"
    state.mkdir()
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = FallbackDischarger(artifacts)
    registry = (default_adapter_registry() if default_registry else
                AdapterRegistry((
                    LedgerToolCallStringAdapter(),
                    LedgerToolCallStringAdapterV2(),
                )))
    routing = AdapterRoutingDischarger(
        fallback, registry, artifacts, log, str(state),
        live_world=live_world)
    return state, artifacts, log, fallback, routing


def discharge(routing, arguments, *, refutable_by_absence=False, version=2):
    obligation = Obligation(
        "obligation-1",
        "the retained string predicate holds",
        BASIS_RECORD,
        refutable_by_absence=refutable_by_absence,
    )
    return routing.discharge(
        ROW, obligation, adapter_method(arguments, version=version))


def execution(artifacts, result):
    return artifacts.get(result.artifact)["execution"]


def append_value(log, value, *, tool="runner", pointer="value", **extra):
    return log.append(
        "tool_call", {"tool": tool, "input": {pointer: value}, **extra})


@pytest.mark.parametrize(("value", "predicate", "outcome"), [
    ("", "nonempty", OUTCOME_REFUTED),
    ("", "nonblank", OUTCOME_REFUTED),
    (" ", "nonempty", OUTCOME_HOLDS),
    (" \t\n", "nonblank", OUTCOME_REFUTED),
    ("\u2003\u3000\xa0", "nonblank", OUTCOME_REFUTED),
    ("\u200b", "nonblank", OUTCOME_HOLDS),
    (" text ", "nonblank", OUTCOME_HOLDS),
])
def test_nonempty_and_nonblank_use_exact_python_string_semantics(
        tmp_path, value, predicate, outcome):
    _, artifacts, log, fallback, routing = harness(tmp_path)
    event = append_value(log, value)

    result = discharge(routing, {
        "value_pointer": "/input/value",
        "tool_names": ["runner"],
        "selector": "first",
        "through_seq": event["seq"],
        "predicate": predicate,
    })

    assert result.outcome == outcome
    assert result.instrument == "adapter:ledger.tool-call-string@2"
    assert fallback.calls == []
    assert execution(artifacts, result)["observation"]["predicate"] == predicate
    if outcome == OUTCOME_REFUTED:
        assert result.witness_kind == WITNESS_RECORD_CONTRADICTION


def test_exists_means_a_string_record_exists_even_when_its_value_is_empty(
        tmp_path):
    _, artifacts, log, fallback, routing = harness(tmp_path)
    event = append_value(log, "")

    result = discharge(routing, {
        "value_pointer": "/input/value", "tool_names": ["runner"],
        "selector": "any", "through_seq": event["seq"],
        "predicate": "exists",
    })

    assert result.outcome == OUTCOME_HOLDS
    assert execution(artifacts, result)["observation"]["predicate"] == "exists"
    assert fallback.calls == []


@pytest.mark.parametrize(("selector", "expected", "outcome"), [
    ("first", "alpha", OUTCOME_HOLDS),
    ("first", "omega", OUTCOME_REFUTED),
    ("last", "omega", OUTCOME_HOLDS),
    ("last", "alpha", OUTCOME_REFUTED),
    ("any", "middle", OUTCOME_HOLDS),
    ("all", "alpha", OUTCOME_REFUTED),
])
def test_equals_obeys_first_last_any_and_all(
        tmp_path, selector, expected, outcome):
    _, artifacts, log, fallback, routing = harness(tmp_path)
    append_value(log, "alpha")
    append_value(log, "middle")
    last = append_value(log, "omega")

    result = discharge(routing, {
        "value_pointer": "/input/value", "tool_names": ["runner"],
        "selector": selector, "through_seq": last["seq"],
        "predicate": "equals", "expected": expected,
    }, refutable_by_absence=True)

    assert result.outcome == outcome
    assert fallback.calls == []
    observed = execution(artifacts, result)["observation"]
    assert observed["predicate"] == "equals"
    if outcome == OUTCOME_REFUTED and selector != "any":
        assert result.witness_kind == WITNESS_RECORD_CONTRADICTION


@pytest.mark.parametrize(("predicate", "values", "any_outcome", "all_outcome"), [
    ("nonempty", ["", "x"], OUTCOME_HOLDS, OUTCOME_REFUTED),
    ("nonblank", [" \t", "x"], OUTCOME_HOLDS, OUTCOME_REFUTED),
])
def test_unary_predicates_obey_any_and_all(
        tmp_path, predicate, values, any_outcome, all_outcome):
    _, _, log, fallback, routing = harness(tmp_path)
    for value in values:
        last = append_value(log, value)
    base = {
        "value_pointer": "/input/value", "tool_names": ["runner"],
        "through_seq": last["seq"], "predicate": predicate,
    }

    any_result = discharge(
        routing, {**base, "selector": "any"},
        refutable_by_absence=True)
    all_result = discharge(
        routing, {**base, "selector": "all"},
        refutable_by_absence=True)

    assert any_result.outcome == any_outcome
    assert all_result.outcome == all_outcome
    assert all_result.witness_kind == WITNESS_RECORD_CONTRADICTION
    assert fallback.calls == []


@pytest.mark.parametrize(("predicate", "extra"), [
    ("equals", {"expected": "missing"}),
    ("nonempty", {}),
    ("nonblank", {}),
])
def test_failed_any_predicate_obeys_absence_authorization(
        tmp_path, predicate, extra):
    _, _, log, fallback, routing = harness(tmp_path)
    event = append_value(log, "" if predicate != "equals" else "different")
    arguments = {
        "value_pointer": "/input/value", "tool_names": ["runner"],
        "selector": "any", "through_seq": event["seq"],
        "predicate": predicate, **extra,
    }

    deferred = discharge(routing, arguments)
    refuted = discharge(
        routing, arguments, refutable_by_absence=True)

    assert deferred.instrument == "fallback"
    assert refuted.outcome == OUTCOME_REFUTED
    assert refuted.witness_kind == WITNESS_ABSENCE
    assert len(fallback.calls) == 1


@pytest.mark.parametrize(("selector", "predicate", "extra"), [
    ("any", "exists", {}),
    ("first", "equals", {"expected": "x"}),
    ("last", "nonempty", {}),
    ("all", "nonblank", {}),
])
def test_empty_selection_obeys_absence_authorization(
        tmp_path, selector, predicate, extra):
    _, _, _, fallback, routing = harness(tmp_path)
    arguments = {
        "value_pointer": "/input/value", "tool_names": ["runner"],
        "selector": selector, "through_seq": 0,
        "predicate": predicate, **extra,
    }

    deferred = discharge(routing, arguments)
    refuted = discharge(
        routing, arguments, refutable_by_absence=True)

    assert deferred.instrument == "fallback"
    assert refuted.outcome == OUTCOME_REFUTED
    assert refuted.witness_kind == WITNESS_ABSENCE
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("arguments", [
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "predicate": "equals"},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "predicate": "equals", "expected": 7},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "predicate": "exists", "expected": "x"},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "predicate": "nonempty", "expected": "x"},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "predicate": "nonblank", "expected": "x"},
    {"value_pointer": "/input/value", "selector": "first",
     "through_seq": 0, "predicate": "exists"},
    {"value_pointer": "/input/value", "selector": "last",
     "through_seq": 0, "predicate": "exists"},
    {"value_pointer": "/input/value", "selector": "all",
     "through_seq": 0, "predicate": "exists"},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "predicate": "contains"},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "predicate": "exists", "unknown": True},
])
def test_predicate_and_expected_argument_contract_is_strict(tmp_path, arguments):
    _, _, _, fallback, routing = harness(tmp_path)

    result = discharge(
        routing, arguments, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert len(fallback.calls) == 1
    assert "adapter_fallback_reason" in fallback.calls[0]["method"]


@pytest.mark.parametrize("arguments", [
    {"value_pointer": "/input/~2bad", "selector": "any",
     "through_seq": 0, "predicate": "exists"},
    {"value_pointer": "input/value", "selector": "any",
     "through_seq": 0, "predicate": "exists"},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "predicate": "exists", "tool_names": []},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "predicate": "exists", "tool_names": [7]},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "predicate": "exists", "tool_names": [""]},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "predicate": "exists",
     "tool_names": ["runner", "runner"]},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "predicate": "exists", "payload_equals": []},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "predicate": "exists",
     "payload_equals": {"/missing/~2": "x"}},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": True, "predicate": "exists"},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": -1, "predicate": "exists"},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "after_seq": True, "predicate": "exists"},
    {"value_pointer": "/input/value", "selector": "any",
     "through_seq": 0, "after_seq": 1, "predicate": "exists"},
])
def test_scope_pointer_and_watermark_arguments_fail_closed(tmp_path, arguments):
    _, _, _, fallback, routing = harness(tmp_path)

    result = discharge(
        routing, arguments, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize(("payload", "reason_fragment"), [
    ({"tool": "runner", "input": {}}, "lacks"),
    ({"tool": "runner", "input": {"value": None}}, "not a string"),
    ({"tool": "runner", "input": {"value": 7}}, "not a string"),
    ({"tool": "runner", "input": {"value": {"nested": "x"}}},
     "not a string"),
])
def test_in_scope_missing_or_wrong_type_value_fails_closed(
        tmp_path, payload, reason_fragment):
    _, _, log, fallback, routing = harness(tmp_path)
    event = log.append("tool_call", payload)

    result = discharge(routing, {
        "value_pointer": "/input/value", "tool_names": ["runner"],
        "selector": "any", "through_seq": event["seq"],
        "predicate": "exists",
    }, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert reason_fragment in fallback.calls[0]["method"][
        "adapter_fallback_reason"]


def test_wrong_tool_malformed_value_is_out_of_scope(tmp_path):
    _, _, log, fallback, routing = harness(tmp_path)
    log.append("tool_call", {"tool": "other", "input": {}})
    wanted = append_value(log, "kept", tool="runner")

    result = discharge(routing, {
        "value_pointer": "/input/value", "tool_names": ["runner"],
        "selector": "any", "through_seq": wanted["seq"],
        "predicate": "nonblank",
    })

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []


def test_after_seq_tool_names_and_payload_equals_define_exact_scope(tmp_path):
    _, artifacts, log, fallback, routing = harness(tmp_path)
    lower = log.append("tool_call", {
        "tool": "runner", "session": "wanted", "agent": "worker",
        "input": {"value": ""},
    })
    log.append("tool_call", {
        "tool": "runner", "session": "other", "agent": "worker",
        "input": {},
    })
    log.append("tool_call", {
        "tool": "other", "session": "wanted", "agent": "worker",
        "input": {},
    })
    wanted = log.append("tool_call", {
        "tool": "runner", "session": "wanted", "agent": "worker",
        "input": {"value": "retained"},
    })

    result = discharge(routing, {
        "selector": "all",
        "value_pointer": "/input/value",
        "after_seq": lower["seq"],
        "through_seq": wanted["seq"],
        "tool_names": ["runner"],
        "payload_equals": {"/session": "wanted", "/agent": "worker"},
        "predicate": "nonblank",
    })

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    observation = execution(artifacts, result)["observation"]
    assert observation["tool_names"] == ["runner"]
    assert observation["payload_equals"] == {
        "/session": "wanted", "/agent": "worker"}
    assert observation["snapshot"]["candidate_string_count"] == 1
    assert observation["witness"] == {
        "predicate": "nonblank",
        "selection": "first_match",
        "items": [{
            "seq": wanted["seq"],
            "tool": "runner",
            "value": "retained",
            "event_digest": EventLog.event_digest(wanted),
        }],
        "candidate_count": 1,
        "selector_domain_count": 1,
        "predicate_match_count": 1,
    }


def test_compact_v2_receipts_cover_large_selections_without_retaining_them(
        tmp_path):
    large = "x" * 4_096

    many_root = tmp_path / "many"
    many_root.mkdir()
    _, many_artifacts, many_log, _, many_routing = harness(many_root)
    many_events = []
    counterexample = None
    for index in range(80):
        event = append_value(
            many_log, " " * 4_096 if index == 70 else large)
        many_events.append(event)
        if index == 70:
            counterexample = event
    many = discharge(many_routing, {
        "selector": "any",
        "value_pointer": "/input/value",
        "through_seq": many_events[-1]["seq"],
        "tool_names": ["runner"],
        "predicate": "nonempty",
    })
    many_payload = many_artifacts.get(many.artifact)
    many_observation = many_payload["execution"]["observation"]
    many_witness = many_observation["witness"]
    snapshot = many_observation["snapshot"]
    event_digests = [EventLog.event_digest(event) for event in many_events]

    assert many.outcome == OUTCOME_HOLDS
    assert many_witness["candidate_count"] == 80
    assert many_witness["predicate_match_count"] == 80
    assert len(many_witness["items"]) == 1
    assert snapshot["prefix_event_count"] == 80
    assert snapshot["range_event_count"] == 80
    assert snapshot["candidate_event_count"] == 80
    assert snapshot["prefix_events_digest"] == content_hash(event_digests)
    assert snapshot["range_events_digest"] == content_hash(event_digests)
    assert snapshot["candidate_events_digest"] == content_hash(event_digests)

    one_root = tmp_path / "one"
    one_root.mkdir()
    _, one_artifacts, one_log, _, one_routing = harness(one_root)
    one_event = append_value(one_log, large)
    one = discharge(one_routing, {
        "selector": "any",
        "value_pointer": "/input/value",
        "through_seq": one_event["seq"],
        "tool_names": ["runner"],
        "predicate": "nonempty",
    })
    one_payload = one_artifacts.get(one.artifact)
    many_bytes = len(json.dumps(
        many_payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8"))
    one_bytes = len(json.dumps(
        one_payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8"))
    assert many_bytes <= one_bytes + 1_024
    assert many_bytes < len(large.encode("utf-8")) * 5

    all_result = discharge(many_routing, {
        "selector": "all",
        "value_pointer": "/input/value",
        "through_seq": many_events[-1]["seq"],
        "tool_names": ["runner"],
        "predicate": "nonblank",
    })
    all_payload = many_artifacts.get(all_result.artifact)
    all_witness = all_payload["execution"]["observation"]["witness"]
    all_bytes = len(json.dumps(
        all_payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8"))

    assert all_result.outcome == OUTCOME_REFUTED
    assert all_witness["candidate_count"] == 80
    assert all_witness["predicate_match_count"] == 79
    assert len(all_witness["items"]) == 1
    assert all_witness["items"][0]["seq"] == counterexample["seq"]
    assert all_witness["items"][0]["value"] == " " * 4_096
    assert all_bytes < len(large.encode("utf-8")) * 6

    legacy = discharge(many_routing, {
        "selector": "any",
        "value_pointer": "/input/value",
        "through_seq": many_events[-1]["seq"],
        "tool_names": ["runner"],
    }, version=1)
    legacy_payload = many_artifacts.get(legacy.artifact)
    legacy_observation = legacy_payload["execution"]["observation"]

    assert legacy_observation["selected_count"] == 80
    assert len(legacy_observation["selected"]) == 64
    assert legacy_observation["selected_truncated"] is True


@pytest.mark.parametrize("live_world", [False, True])
def test_explicit_ledger_prefix_has_online_offline_parity_and_fresh_suffix(
        tmp_path, live_world):
    _, artifacts, log, fallback, routing = harness(
        tmp_path, live_world=live_world)
    prefix = append_value(
        log, "retained payload", tool="opaque-runtime.v7", pointer="result")
    arguments = {
        "selector": "all",
        "value_pointer": "/input/result",
        "through_seq": prefix["seq"],
        "tool_names": ["opaque-runtime.v7"],
        "predicate": "nonblank",
    }

    before = discharge(routing, arguments)
    before_snapshot = execution(artifacts, before)["observation"]["snapshot"]
    suffix = append_value(
        log, "\u2003\t", tool="opaque-runtime.v7", pointer="result")
    unchanged = discharge(routing, arguments)
    unchanged_snapshot = execution(
        artifacts, unchanged)["observation"]["snapshot"]
    extended = discharge(
        routing, {**arguments, "through_seq": suffix["seq"]})
    extended_snapshot = execution(
        artifacts, extended)["observation"]["snapshot"]

    assert before.outcome == OUTCOME_HOLDS
    assert unchanged.outcome == OUTCOME_HOLDS
    assert before.artifact == unchanged.artifact
    assert before_snapshot == unchanged_snapshot
    assert before_snapshot["candidate_string_count"] == 1
    assert extended.outcome == OUTCOME_REFUTED
    assert extended.witness_kind == WITNESS_RECORD_CONTRADICTION
    assert extended.artifact != before.artifact
    assert extended_snapshot["through_seq"] == suffix["seq"]
    assert extended_snapshot["candidate_string_count"] == 2
    assert extended_snapshot["prefix_events_digest"] \
        != before_snapshot["prefix_events_digest"]
    assert routing.supports_method_reuse(adapter_method(arguments)) is False
    assert fallback.calls == []


def test_through_seq_is_exact_and_corrupt_future_suffix_is_not_parsed(tmp_path):
    _, _, log, fallback, routing = harness(tmp_path)
    wanted = append_value(log, "kept")
    append_value(log, "", tool="runner")
    with open(log.path, "ab") as stream:
        stream.write(b"{not-json\n")

    result = discharge(routing, {
        "value_pointer": "/input/value", "tool_names": ["runner"],
        "selector": "all", "through_seq": wanted["seq"],
        "predicate": "nonempty",
    })

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []


def test_future_watermark_falls_back_instead_of_treating_suffix_as_absent(
        tmp_path):
    _, _, log, fallback, routing = harness(tmp_path)
    event = append_value(log, "kept")

    result = discharge(routing, {
        "value_pointer": "/input/value", "tool_names": ["runner"],
        "selector": "any", "through_seq": event["seq"] + 1,
        "predicate": "exists",
    }, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert "ends before through_seq" in fallback.calls[0]["method"][
        "adapter_fallback_reason"]


def test_v1_behavior_is_unchanged_and_v2_fields_are_not_v1_aliases(tmp_path):
    _, _, log, fallback, routing = harness(tmp_path)
    event = append_value(log, "alpha")

    old = discharge(routing, {
        "selector": "any", "value_pointer": "/input/value",
        "through_seq": event["seq"], "tool_names": ["runner"],
        "expected": "alpha",
    }, version=1)
    v2_as_v1 = discharge(routing, {
        "value_pointer": "/input/value", "tool_names": ["runner"],
        "selector": "any", "through_seq": event["seq"],
        "predicate": "equals", "expected": "alpha",
    }, version=1)

    assert old.outcome == OUTCOME_HOLDS
    assert old.instrument == "adapter:ledger.tool-call-string@1"
    assert v2_as_v1.instrument == "fallback"
    assert len(fallback.calls) == 1


def test_default_catalog_registers_both_versions_without_method_reuse():
    catalog = json.loads(default_adapter_registry().catalog())
    entries = [entry for entry in catalog
               if entry["name"] == "ledger.tool-call-string"]

    assert {entry["version"] for entry in entries} == {1, 2}
    assert all(entry["accepted_method_reuse"] is False for entry in entries)
    v2 = next(entry for entry in entries if entry["version"] == 2)
    assert set(v2["arguments"]) == {
        "selector", "value_pointer", "through_seq", "after_seq",
        "expected", "tool_names", "payload_equals", "predicate",
    }


def test_v2_adapter_still_requires_fresh_oracle_and_judger_each_time(tmp_path):
    state, _, log, fallback, routing = harness(
        tmp_path, default_registry=True)
    (state / "task.md").write_text("Retain the exact nonblank string")
    event = append_value(log, "alpha")
    typed = adapter_method({
        "value_pointer": "/input/value", "tool_names": ["runner"],
        "selector": "any", "through_seq": event["seq"],
        "predicate": "equals", "expected": "alpha",
    })
    cache = AcceptedMethodCache()
    backend = FixtureBackend({
        "oracles": {"methods": [typed]},
        "judger": JUDGE_YES,
    })
    game = BindingGame(
        backend, routing.artifacts, log, routing,
        live_world=False, method_cache=cache)
    obligation = Obligation(
        "obligation-1", "the retained value equals alpha", BASIS_RECORD)

    first = game.run(ROW, obligation)
    second = game.run(ROW, obligation)

    assert first.outcome == OUTCOME_HOLDS
    assert second.outcome == OUTCOME_HOLDS
    assert [call["role"] for call in backend.calls] == [
        "oracles", "judger", "oracles", "judger"]
    assert len(cache) == 0
    assert fallback.calls == []
    adapter_events = [event for event in log.replay()
                      if event["type"] == "adapter_discharge"]
    assert len(adapter_events) == 2
    assert all(event["payload"]["adapter"]
               == "ledger.tool-call-string@2" for event in adapter_events)


def test_invalid_v2_binding_uses_existing_fallback_and_audit_event(tmp_path):
    _, _, log, fallback, routing = harness(tmp_path)

    result = discharge(routing, {
        "value_pointer": "/input/value", "tool_names": ["runner"],
        "selector": "first", "through_seq": 0,
        "predicate": "exists",
    }, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert len(fallback.calls) == 1
    events = [event for event in log.replay()
              if event["type"] == "adapter_fallback"]
    assert len(events) == 1
    assert events[0]["payload"]["adapter"] == "ledger.tool-call-string"
