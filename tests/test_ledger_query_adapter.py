"""Domain-neutral and fail-closed contracts for ledger.tool-call-string."""
from __future__ import annotations

import json

import pytest

import verifier.execution.adapters as adapters_module
from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_RECORD,
    OUTCOME_HOLDS,
    OUTCOME_REFUTED,
    WITNESS_ABSENCE,
    WITNESS_RECORD_CONTRADICTION,
    Obligation,
)
from verifier.execution.adapters import (
    AdapterRegistry,
    AdapterRoutingDischarger,
    LedgerToolCallStringAdapter,
)
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore


ROW = {"id": "claim-1", "kind": "final", "text": "retained record claim"}


class FallbackDischarger:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append({"method": method, "tag": tag})
        artifact = self.artifacts.put({"instrument": "fallback"})
        return DischargeResult(
            outcome=OUTCOME_HOLDS,
            witness=None,
            artifact=artifact,
            instrument="fallback",
            witness_kind="execution",
        )


def method(arguments):
    return {
        "procedure": "select the declared retained tool-call string exactly",
        "artifact": "a content-addressed exact ledger query",
        "level": "deterministic",
        "basis": BASIS_RECORD,
        "strength": "decisive",
        "bridges": [],
        "adapter": "ledger.tool-call-string",
        "adapter_version": 1,
        "adapter_arguments": json.dumps(arguments, sort_keys=True),
        "adapter_inputs": [{
            "ref": "ledger:events",
            "provenance": "ledger",
            "role": "subject",
        }],
    }


def harness(tmp_path):
    state = tmp_path / "gv"
    state.mkdir()
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = FallbackDischarger(artifacts)
    registry = AdapterRegistry((LedgerToolCallStringAdapter(),))
    routing = AdapterRoutingDischarger(
        fallback, registry, artifacts, log, str(state), live_world=False)
    return state, artifacts, log, fallback, routing


def discharge(routing, arguments, *, refutable_by_absence=False):
    obligation = Obligation(
        "o1", "the exact retained string relation holds", BASIS_RECORD,
        refutable_by_absence=refutable_by_absence,
    )
    return routing.discharge(ROW, obligation, method(arguments))


def evidence(artifacts, result):
    return artifacts.get(result.artifact)["execution"]


@pytest.mark.parametrize(("tool", "payload", "pointer", "expected"), [
    ("Bash", {"input": {"command": "alpha"}}, "/input/command", "alpha"),
    ("exec_command", {"input": {"cmd": "beta"}}, "/input/cmd", "beta"),
    ("bash", {"input": {"keystrokes": "gamma"}},
     "/input/keystrokes", "gamma"),
    ("unfamiliar-tool", {"request": {"nested": {"value": "delta"}}},
     "/request/nested/value", "delta"),
])
def test_exact_string_query_is_dialect_neutral(
        tmp_path, tool, payload, pointer, expected):
    _, _, log, fallback, routing = harness(tmp_path)
    event = log.append("tool_call", {"tool": tool, **payload})

    result = discharge(routing, {
        "selector": "any",
        "value_pointer": pointer,
        "through_seq": event["seq"],
        "tool_names": [tool],
        "expected": expected,
    })

    assert result.outcome == OUTCOME_HOLDS
    assert result.instrument == "adapter:ledger.tool-call-string@1"
    assert fallback.calls == []


def test_watermarks_and_exact_payload_scope_exclude_other_sessions(tmp_path):
    _, artifacts, log, fallback, routing = harness(tmp_path)
    log.append("tool_call", {
        "tool": "runner", "session": "old", "agent": "worker",
        "input": {"line": "wrong-before"},
    })
    first = log.append("tool_call", {
        "tool": "runner", "session": "wanted", "agent": "worker",
        "input": {"line": "first"},
    })
    last = log.append("tool_call", {
        "tool": "runner", "session": "wanted", "agent": "worker",
        "input": {"line": "last"},
    })
    log.append("tool_call", {
        "tool": "runner", "session": "wanted", "agent": "other",
        "input": {"line": "wrong-after"},
    })

    result = discharge(routing, {
        "selector": "last",
        "value_pointer": "/input/line",
        "after_seq": first["seq"] - 1,
        "through_seq": last["seq"],
        "tool_names": ["runner"],
        "payload_equals": {"/session": "wanted", "/agent": "worker"},
        "expected": "last",
    })

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    execution = evidence(artifacts, result)
    snapshot = execution["observation"]["snapshot"]
    assert snapshot["candidate_string_count"] == 2
    assert snapshot["range_event_count"] == 2
    assert execution["observation"]["selected"][0]["seq"] == last["seq"]


def test_through_seq_is_a_hard_prefix_and_does_not_parse_a_corrupt_suffix(
        tmp_path):
    _, _, log, fallback, routing = harness(tmp_path)
    event = log.append("tool_call", {
        "tool": "runner", "input": {"line": "kept"}})
    with open(log.path, "ab") as stream:
        stream.write(b"{not-json\n")

    result = discharge(routing, {
        "selector": "first", "value_pointer": "/input/line",
        "through_seq": event["seq"], "expected": "kept",
    })

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []


@pytest.mark.parametrize("arguments", [
    {
        "selector": "any", "value_pointer": "/missing/~2",
        "through_seq": 0,
    },
    {
        "selector": "any", "value_pointer": "/input/value",
        "through_seq": 0, "payload_equals": {"/missing/~2": "x"},
    },
])
def test_every_pointer_segment_is_validated_before_scanning(tmp_path, arguments):
    _, _, _, fallback, routing = harness(tmp_path)

    result = discharge(routing, arguments, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert "invalid RFC 6901 escape" in fallback.calls[0]["method"][
        "adapter_fallback_reason"]


@pytest.mark.parametrize("input_value", [None, 7, {"nested": "value"}])
def test_malformed_value_in_a_matching_record_fails_closed(
        tmp_path, input_value):
    _, _, log, fallback, routing = harness(tmp_path)
    event = log.append("tool_call", {
        "tool": "runner", "input": {"value": input_value}})

    result = discharge(routing, {
        "selector": "any", "value_pointer": "/input/value",
        "through_seq": event["seq"], "tool_names": ["runner"],
    }, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert "is not a string" in fallback.calls[0]["method"][
        "adapter_fallback_reason"]


def test_missing_value_in_a_matching_record_fails_closed(tmp_path):
    _, _, log, fallback, routing = harness(tmp_path)
    event = log.append("tool_call", {"tool": "runner", "input": {}})

    result = discharge(routing, {
        "selector": "any", "value_pointer": "/input/value",
        "through_seq": event["seq"], "tool_names": ["runner"],
    }, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert "lacks value_pointer" in fallback.calls[0]["method"][
        "adapter_fallback_reason"]


def test_malformed_value_outside_exact_scope_does_not_poison_query(tmp_path):
    _, _, log, fallback, routing = harness(tmp_path)
    log.append("tool_call", {"tool": "other", "input": {}})
    wanted = log.append("tool_call", {
        "tool": "runner", "session": "s", "input": {"value": "ok"}})

    result = discharge(routing, {
        "selector": "any", "value_pointer": "/input/value",
        "through_seq": wanted["seq"], "tool_names": ["runner"],
        "payload_equals": {"/session": "s"}, "expected": "ok",
    })

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []


def test_failed_existential_match_is_absence_not_a_positive_contradiction(
        tmp_path):
    _, _, log, fallback, routing = harness(tmp_path)
    event = log.append("tool_call", {
        "tool": "runner", "input": {"value": "different"}})
    arguments = {
        "selector": "any", "value_pointer": "/input/value",
        "through_seq": event["seq"], "expected": "wanted",
    }

    deferred = discharge(routing, arguments)
    refuted = discharge(routing, arguments, refutable_by_absence=True)

    assert deferred.instrument == "fallback"
    assert refuted.outcome == OUTCOME_REFUTED
    assert refuted.witness_kind == WITNESS_ABSENCE
    assert len(fallback.calls) == 1


@pytest.mark.parametrize(("selector", "expected"), [
    ("first", "not-first"),
    ("last", "not-last"),
    ("all", "same"),
])
def test_nonexistential_mismatch_is_a_retained_record_contradiction(
        tmp_path, selector, expected):
    _, _, log, fallback, routing = harness(tmp_path)
    first = log.append("tool_call", {
        "tool": "runner", "input": {"value": "first"}})
    last = log.append("tool_call", {
        "tool": "runner", "input": {"value": "last"}})

    result = discharge(routing, {
        "selector": selector, "value_pointer": "/input/value",
        "through_seq": last["seq"], "after_seq": first["seq"] - 1,
        "expected": expected,
    })

    assert result.outcome == OUTCOME_REFUTED
    assert result.witness_kind == WITNESS_RECORD_CONTRADICTION
    assert fallback.calls == []


def test_prefix_candidate_and_range_digests_cover_distinct_scopes(tmp_path):
    _, artifacts, log, _, routing = harness(tmp_path)
    log.append("message", {"text": "outside the query range"})
    log.append("tool_call", {
        "tool": "ignored", "input": {"value": "ignored"}})
    wanted = log.append("tool_call", {
        "tool": "runner", "input": {"value": "wanted"}})

    result = discharge(routing, {
        "selector": "any", "value_pointer": "/input/value",
        "after_seq": 1, "through_seq": wanted["seq"],
        "tool_names": ["runner"], "expected": "wanted",
    })

    snapshot = evidence(artifacts, result)["observation"]["snapshot"]
    assert snapshot["prefix_event_count"] == 3
    assert snapshot["range_event_count"] == 2
    assert snapshot["candidate_event_count"] == 1
    assert len({snapshot["prefix_events_digest"],
                snapshot["range_events_digest"],
                snapshot["candidate_events_digest"]}) == 3


def test_event_and_byte_limits_fall_back_before_unbounded_work(
        tmp_path, monkeypatch):
    _, _, log, fallback, routing = harness(tmp_path)
    event = log.append("tool_call", {
        "tool": "runner", "input": {"value": "x" * 400}})
    monkeypatch.setattr(adapters_module, "MAX_LEDGER_BYTES", 128)

    byte_limited = discharge(routing, {
        "selector": "any", "value_pointer": "/input/value",
        "through_seq": event["seq"],
    })
    monkeypatch.setattr(adapters_module, "MAX_LEDGER_EVENTS", 0)
    event_limited = discharge(routing, {
        "selector": "any", "value_pointer": "/input/value",
        "through_seq": event["seq"],
    })

    assert byte_limited.instrument == "fallback"
    assert "prefix exceeds" in fallback.calls[0]["method"][
        "adapter_fallback_reason"]
    assert event_limited.instrument == "fallback"
    assert "through_seq exceeds" in fallback.calls[1]["method"][
        "adapter_fallback_reason"]


def test_value_limits_are_per_item_and_cumulative(tmp_path, monkeypatch):
    _, _, log, fallback, routing = harness(tmp_path)
    first = log.append("tool_call", {
        "tool": "runner", "input": {"value": "1234"}})
    second = log.append("tool_call", {
        "tool": "runner", "input": {"value": "5678"}})
    arguments = {
        "selector": "any", "value_pointer": "/input/value",
        "through_seq": second["seq"],
    }
    monkeypatch.setattr(adapters_module, "MAX_LEDGER_VALUE_BYTES", 3)
    per_item = discharge(routing, arguments)
    monkeypatch.setattr(adapters_module, "MAX_LEDGER_VALUE_BYTES", 4)
    monkeypatch.setattr(adapters_module, "MAX_LEDGER_VALUE_BYTES_TOTAL", 7)
    cumulative = discharge(routing, arguments)

    assert per_item.instrument == "fallback"
    assert "string exceeds" in fallback.calls[0]["method"][
        "adapter_fallback_reason"]
    assert cumulative.instrument == "fallback"
    assert "cumulative byte limit" in fallback.calls[1]["method"][
        "adapter_fallback_reason"]


def test_decisive_value_after_retention_limit_is_still_retained(tmp_path):
    _, artifacts, log, fallback, routing = harness(tmp_path)
    for index in range(adapters_module.MAX_RETAINED_LEDGER_VALUES + 1):
        event = log.append("tool_call", {
            "tool": "runner", "input": {"value": f"other-{index}"}})
    witness = log.append("tool_call", {
        "tool": "runner", "input": {"value": "wanted"}})

    result = discharge(routing, {
        "selector": "any", "value_pointer": "/input/value",
        "through_seq": witness["seq"], "expected": "wanted",
    })

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    observation = evidence(artifacts, result)["observation"]
    assert observation["selected"][0]["seq"] == witness["seq"]
    assert observation["selected"][0]["value"] == "wanted"


def test_raw_wrapper_is_compared_as_a_string_and_never_parsed_or_executed(
        tmp_path):
    state, _, log, fallback, routing = harness(tmp_path)
    marker = state / "must-not-exist"
    wrapper = (
        "tools.exec_command({cmd: 'touch " + str(marker) + " && echo inner'})"
    )
    event = log.append("tool_call", {"tool": "exec", "input": wrapper})

    inner = discharge(routing, {
        "selector": "any", "value_pointer": "/input",
        "through_seq": event["seq"], "expected": "touch " + str(marker),
    })
    exact = discharge(routing, {
        "selector": "any", "value_pointer": "/input",
        "through_seq": event["seq"], "expected": wrapper,
    })

    assert inner.instrument == "fallback"
    assert exact.outcome == OUTCOME_HOLDS
    assert not marker.exists()
    assert len(fallback.calls) == 1


def test_ambiguous_existence_selector_and_future_watermark_fail_closed(tmp_path):
    _, _, _, fallback, routing = harness(tmp_path)

    ambiguous = discharge(routing, {
        "selector": "all", "value_pointer": "/input/value",
        "through_seq": 0,
    }, refutable_by_absence=True)
    future = discharge(routing, {
        "selector": "any", "value_pointer": "/input/value",
        "through_seq": 10,
    }, refutable_by_absence=True)

    assert ambiguous.instrument == "fallback"
    assert "existence queries require selector=any" in fallback.calls[0][
        "method"]["adapter_fallback_reason"]
    assert future.instrument == "fallback"
    assert "ends before through_seq=10" in fallback.calls[1]["method"][
        "adapter_fallback_reason"]
