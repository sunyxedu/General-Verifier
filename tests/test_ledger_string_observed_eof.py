"""Point-in-time EOF scope for ``ledger.tool-call-string@3``."""
from __future__ import annotations

import hashlib
import json
import os

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
    content_hash,
)
from verifier.engine.binding.game import BindingGame
from verifier.engine.binding.method_cache import AcceptedMethodCache
from verifier.execution.adapters import (
    AdapterNotApplicable,
    AdapterRegistry,
    AdapterRoutingDischarger,
    LedgerToolCallStringAdapter,
    LedgerToolCallStringAdapterV2,
    LedgerToolCallStringAdapterV3,
    default_adapter_registry,
)
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend


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


def _method(arguments, *, version=3, inputs=None):
    return {
        "procedure": "evaluate one declared retained string predicate",
        "artifact": "a content-addressed exact ledger observation",
        "level": "deterministic",
        "basis": BASIS_RECORD,
        "strength": "decisive",
        "bridges": [],
        "adapter": "ledger.tool-call-string",
        "adapter_version": version,
        "adapter_arguments": json.dumps(arguments, ensure_ascii=False,
                                        sort_keys=True),
        "adapter_inputs": ([{
            "ref": "ledger:events",
            "provenance": "ledger",
            "role": "subject",
        }] if inputs is None else inputs),
    }


def _harness(tmp_path, *, live_world=False, default_registry=False):
    state = tmp_path / "state"
    state.mkdir()
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = FallbackDischarger(artifacts)
    registry = (default_adapter_registry() if default_registry else
                AdapterRegistry((
                    LedgerToolCallStringAdapter(),
                    LedgerToolCallStringAdapterV2(),
                    LedgerToolCallStringAdapterV3(),
                )))
    routing = AdapterRoutingDischarger(
        fallback, registry, artifacts, log, str(state),
        live_world=live_world)
    return state, artifacts, log, fallback, routing


def _append(log, value, *, tool="runtime.v7", session="wanted"):
    return log.append("tool_call", {
        "tool": tool,
        "session": session,
        "input": {"value": value},
    })


def _discharge(
    routing,
    arguments,
    *,
    version=3,
    refutable_by_absence=False,
    inputs=None,
):
    obligation = Obligation(
        "o1",
        "the exact retained string relation holds",
        BASIS_RECORD,
        refutable_by_absence=refutable_by_absence,
    )
    return routing.discharge(
        ROW, obligation, _method(arguments, version=version, inputs=inputs))


def _execution(artifacts, result):
    return artifacts.get(result.artifact)["execution"]


@pytest.mark.parametrize(("selector", "predicate", "expected"), [
    ("any", "exists", None),
    ("first", "equals", ""),
    ("last", "equals", "beta"),
    ("any", "equals", "alpha"),
    ("all", "equals", "alpha"),
    ("first", "nonempty", None),
    ("last", "nonempty", None),
    ("any", "nonempty", None),
    ("all", "nonempty", None),
    ("first", "nonblank", None),
    ("last", "nonblank", None),
    ("any", "nonblank", None),
    ("all", "nonblank", None),
])
def test_prefix_scope_matches_v2_selector_predicate_results(
    tmp_path, selector, predicate, expected,
):
    _state, artifacts, log, fallback, routing = _harness(tmp_path)
    _append(log, "")
    _append(log, "\u2003")
    _append(log, "alpha")
    last = _append(log, "beta")
    base = {
        "selector": selector,
        "value_pointer": "/input/value",
        "through_seq": last["seq"],
        "tool_names": ["runtime.v7"],
        "payload_equals": {"/session": "wanted"},
        "predicate": predicate,
    }
    if predicate == "equals":
        base["expected"] = expected

    v2 = _discharge(
        routing, base, version=2, refutable_by_absence=True)
    v3 = _discharge(
        routing, {**base, "scope": "prefix"}, version=3,
        refutable_by_absence=True)

    assert (v3.outcome, v3.witness_kind) == (v2.outcome, v2.witness_kind)
    v2_execution = _execution(artifacts, v2)
    v3_execution = _execution(artifacts, v3)
    assert v3_execution["observation"]["matches"] \
        == v2_execution["observation"]["matches"]
    assert v3_execution["observation"]["witness"] \
        == v2_execution["observation"]["witness"]
    assert v3_execution["observation"]["snapshot"]["scope"] == "prefix"
    assert fallback.calls == []


@pytest.mark.parametrize("live_world", [False, True])
def test_observed_eof_receipt_binds_complete_point_in_time_bytes(
    tmp_path, live_world,
):
    state, artifacts, log, fallback, routing = _harness(
        tmp_path, live_world=live_world)
    log.append("message", {"text": "unrelated"})
    _append(log, "old", session="old")
    wanted = _append(log, "雪", session="wanted")
    raw_at_call = (state / "events.jsonl").read_bytes()
    parsed_at_call = [json.loads(line) for line in raw_at_call.splitlines()]
    event_digests = [EventLog.event_digest(event) for event in parsed_at_call]

    result = _discharge(routing, {
        "scope": "observed-eof",
        "selector": "last",
        "value_pointer": "/input/value",
        "after_seq": 1,
        "tool_names": ["runtime.v7"],
        "payload_equals": {"/session": "wanted"},
        "predicate": "equals",
        "expected": "雪",
    })

    assert result.outcome == OUTCOME_HOLDS
    assert result.instrument == "adapter:ledger.tool-call-string@3"
    assert result.inputs == ({
        "ref": "ledger:events", "provenance": "ledger", "role": "subject",
    },)
    observed = _execution(artifacts, result)["observation"]["snapshot"][
        "observed_eof"]
    assert observed["raw_sha256"] == hashlib.sha256(raw_at_call).hexdigest()
    assert observed["events_digest"] == content_hash(event_digests)
    assert observed["resolved_through_seq"] == wanted["seq"]
    assert observed["event_count"] == wanted["seq"]
    assert observed["byte_count"] == len(raw_at_call)
    assert observed["complete_file_at_observation"] is True
    assert observed["proves_future_closure"] is False
    assert "later appends" in observed["scope_statement"]
    assert observed["file_identity"]["bytes"] == len(raw_at_call)
    assert observed["file_identity"]["links"] == 1
    assert isinstance(observed["file_identity"]["mode"], int)
    assert observed["observation_started_at_unix_ns"] \
        <= observed["observation_completed_at_unix_ns"]
    assert observed["observation_started_monotonic_ns"] \
        <= observed["observation_completed_monotonic_ns"]
    assert observed["monotonic_clock_scope"] \
        == "process-local; not cross-process comparable"
    # AdapterRoutingDischarger appends after execution. The receipt remains
    # explicitly point-in-time and does not claim this later record was read.
    assert (state / "events.jsonl").read_bytes() != raw_at_call
    assert fallback.calls == []


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_observed_eof_treats_unicode_string_separators_as_json_content(
        tmp_path, separator):
    state, artifacts, log, fallback, routing = _harness(tmp_path)
    wanted = _append(log, "before" + separator + "after", session="wanted")

    result = _discharge(routing, {
        "scope": "observed-eof",
        "selector": "last",
        "value_pointer": "/input/value",
        "after_seq": 0,
        "tool_names": ["runtime.v7"],
        "payload_equals": {"/session": "wanted"},
        "predicate": "equals",
        "expected": "before" + separator + "after",
    })

    assert result.outcome == OUTCOME_HOLDS
    snapshot = _execution(artifacts, result)["observation"]["snapshot"][
        "observed_eof"]
    assert snapshot["resolved_through_seq"] == wanted["seq"]
    assert fallback.calls == []


def test_observed_eof_uses_only_the_stable_in_memory_event_source(
    tmp_path, monkeypatch,
):
    _state, _artifacts, log, fallback, routing = _harness(tmp_path)
    _append(log, "alpha")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("observed-eof must not reread a live prefix")

    monkeypatch.setattr(adapters_module, "_bounded_ledger_prefix", forbidden)
    result = _discharge(routing, {
        "scope": "observed-eof",
        "selector": "any",
        "value_pointer": "/input/value",
        "predicate": "nonblank",
    })

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []


def test_prefix_scope_does_not_invoke_the_observed_eof_reader(
    tmp_path, monkeypatch,
):
    _state, _artifacts, log, fallback, routing = _harness(tmp_path)
    event = _append(log, "alpha")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("prefix must keep the existing bounded reader")

    monkeypatch.setattr(adapters_module, "_read_stable_ledger_eof", forbidden)
    result = _discharge(routing, {
        "scope": "prefix",
        "selector": "any",
        "value_pointer": "/input/value",
        "through_seq": event["seq"],
        "predicate": "exists",
    })

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []


@pytest.mark.parametrize("arguments", [
    {
        "selector": "any", "value_pointer": "/input/value",
        "predicate": "exists",
    },
    {
        "scope": "other", "selector": "any",
        "value_pointer": "/input/value", "predicate": "exists",
    },
    {
        "scope": "prefix", "selector": "any",
        "value_pointer": "/input/value", "predicate": "exists",
    },
    {
        "scope": "observed-eof", "through_seq": 0,
        "selector": "any", "value_pointer": "/input/value",
        "predicate": "exists",
    },
    {
        "scope": "observed-eof", "through_seq": None,
        "selector": "any", "value_pointer": "/input/value",
        "predicate": "exists",
    },
])
def test_scope_shape_is_strictly_mutually_exclusive(tmp_path, arguments):
    _state, _artifacts, _log, fallback, routing = _harness(tmp_path)

    result = _discharge(
        routing, arguments, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert len(fallback.calls) == 1


def test_missing_subject_binding_is_rejected_before_file_read(
    tmp_path, monkeypatch,
):
    _state, _artifacts, _log, fallback, routing = _harness(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an undeclared input must not be read")

    monkeypatch.setattr(adapters_module, "_read_stable_ledger_eof", forbidden)
    result = _discharge(routing, {
        "scope": "observed-eof",
        "selector": "any",
        "value_pointer": "/input/value",
        "predicate": "exists",
    }, inputs=[])

    assert result.instrument == "fallback"
    assert "must declare subject" in fallback.calls[0]["method"][
        "adapter_fallback_reason"]


@pytest.mark.parametrize(("raw", "reason"), [
    (b"\xff\n", "strict UTF-8 JSONL"),
    (b'{"seq":1,"type":"message","payload":{}}', "incomplete final"),
    (b'{"seq":2,"type":"message","payload":{}}\n', "invalid event"),
    (b'{"seq":1,"seq":1,"type":"message","payload":{}}\n',
     "strict UTF-8 JSONL"),
    (b'{"seq":1,"type":"message","payload":{"x":NaN}}\n',
     "strict UTF-8 JSONL"),
    (b'{"seq":1,"type":"message","payload":{"x":1e999}}\n',
     "strict UTF-8 JSONL"),
    (b'[]\n', "event must be an object"),
    (b'{"seq":1,"type":7,"payload":{}}\n', "invalid event"),
    (b'\n', "blank JSONL record"),
])
def test_malformed_observed_eof_falls_back_not_absence(
    tmp_path, raw, reason,
):
    state, _artifacts, _log, fallback, routing = _harness(tmp_path)
    (state / "events.jsonl").write_bytes(raw)

    result = _discharge(routing, {
        "scope": "observed-eof",
        "selector": "any",
        "value_pointer": "/input/value",
        "predicate": "exists",
    }, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert reason in fallback.calls[0]["method"]["adapter_fallback_reason"]


def test_full_eof_is_validated_even_after_an_early_positive_match(tmp_path):
    state, _artifacts, log, fallback, routing = _harness(tmp_path)
    _append(log, "match")
    with open(state / "events.jsonl", "ab") as stream:
        stream.write(b"{not-json\n")

    result = _discharge(routing, {
        "scope": "observed-eof",
        "selector": "any",
        "value_pointer": "/input/value",
        "predicate": "equals",
        "expected": "match",
    }, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert "strict UTF-8 JSONL" in fallback.calls[0]["method"][
        "adapter_fallback_reason"]


@pytest.mark.parametrize("limited", ["bytes", "events"])
def test_observed_eof_limits_fall_back_not_absence(
    tmp_path, monkeypatch, limited,
):
    _state, _artifacts, log, fallback, routing = _harness(tmp_path)
    _append(log, "alpha")
    if limited == "bytes":
        monkeypatch.setattr(adapters_module, "MAX_LEDGER_BYTES", 1)
    else:
        monkeypatch.setattr(adapters_module, "MAX_LEDGER_EVENTS", 0)

    result = _discharge(routing, {
        "scope": "observed-eof",
        "selector": "any",
        "value_pointer": "/input/value",
        "predicate": "exists",
    }, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("file_kind", ["symlink", "hardlink", "fifo"])
def test_observed_eof_rejects_aliases_and_nonregular_files(
    tmp_path, file_kind,
):
    state, _artifacts, log, fallback, routing = _harness(tmp_path)
    _append(log, "alpha")
    path = state / "events.jsonl"
    saved = state / "saved-events.jsonl"
    if file_kind == "symlink":
        path.replace(saved)
        path.symlink_to(saved)
    elif file_kind == "hardlink":
        os.link(path, saved)
    else:
        path.unlink()
        os.mkfifo(path)
        # The adapter must fail before opening the FIFO. Avoid asking the
        # ordinary fallback audit logger to append to the deliberately
        # non-regular fixture after that failure.
        routing.log.append = lambda *_args, **_kwargs: None

    result = _discharge(routing, {
        "scope": "observed-eof",
        "selector": "any",
        "value_pointer": "/input/value",
        "predicate": "exists",
    }, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert len(fallback.calls) == 1


def test_observed_eof_rejects_dotdot_before_path_normalization(tmp_path):
    state, _artifacts, log, fallback, routing = _harness(tmp_path)
    _append(log, "alpha")
    (state / "ordinary-parent").mkdir()
    log.path = str(state / "ordinary-parent" / ".." / "events.jsonl")

    result = _discharge(routing, {
        "scope": "observed-eof",
        "selector": "any",
        "value_pointer": "/input/value",
        "predicate": "exists",
    }, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert "must not contain '..'" in fallback.calls[0]["method"][
        "adapter_fallback_reason"]


def test_observed_eof_accepts_an_unambiguous_relative_log_path(
    tmp_path, monkeypatch,
):
    _state, _artifacts, log, _fallback, _routing = _harness(tmp_path)
    event = _append(log, "alpha")
    monkeypatch.chdir(tmp_path)
    log.path = os.path.relpath(log.path, tmp_path)

    observed = adapters_module._read_stable_ledger_eof(log)

    assert observed["resolved_through_seq"] == event["seq"]
    assert observed["event_count"] == 1


def test_change_during_observed_eof_read_falls_back(tmp_path, monkeypatch):
    state, _artifacts, log, fallback, routing = _harness(tmp_path)
    _append(log, "alpha")
    original_read = adapters_module.os.read
    changed = False

    def append_during_read(fd, count):
        nonlocal changed
        chunk = original_read(fd, count)
        if chunk and not changed:
            changed = True
            with open(state / "events.jsonl", "ab") as stream:
                stream.write(json.dumps({
                    "seq": 2,
                    "ts": 0,
                    "type": "message",
                    "payload": {"text": "late"},
                }).encode("utf-8") + b"\n")
        return chunk

    monkeypatch.setattr(adapters_module.os, "read", append_during_read)
    result = _discharge(routing, {
        "scope": "observed-eof",
        "selector": "any",
        "value_pointer": "/input/value",
        "predicate": "exists",
    }, refutable_by_absence=True)

    assert changed is True
    assert result.instrument == "fallback"
    assert "changed while" in fallback.calls[0]["method"][
        "adapter_fallback_reason"]


def test_parent_component_swap_during_read_is_detected(tmp_path, monkeypatch):
    state, _artifacts, log, _fallback, _routing = _harness(tmp_path)
    _append(log, "alpha")
    original_read = adapters_module.os.read
    moved = tmp_path / "moved-state"
    swapped = False

    def swap_parent(fd, count):
        nonlocal swapped
        chunk = original_read(fd, count)
        if chunk and not swapped:
            swapped = True
            raw = (state / "events.jsonl").read_bytes()
            state.rename(moved)
            state.mkdir()
            (state / "events.jsonl").write_bytes(raw)
        return chunk

    monkeypatch.setattr(adapters_module.os, "read", swap_parent)

    with pytest.raises(AdapterNotApplicable, match="path changed"):
        adapters_module._read_stable_ledger_eof(log)
    assert swapped is True


def test_ledger_inode_replacement_during_read_is_detected(
    tmp_path, monkeypatch,
):
    state, _artifacts, log, _fallback, _routing = _harness(tmp_path)
    _append(log, "alpha")
    path = state / "events.jsonl"
    original_read = adapters_module.os.read
    replaced = False

    def replace_leaf(fd, count):
        nonlocal replaced
        chunk = original_read(fd, count)
        if chunk and not replaced:
            replaced = True
            replacement = state / "replacement.jsonl"
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        return chunk

    monkeypatch.setattr(adapters_module.os, "read", replace_leaf)

    with pytest.raises(AdapterNotApplicable, match="changed while"):
        adapters_module._read_stable_ledger_eof(log)
    assert replaced is True


def test_late_malformed_in_scope_tool_call_defeats_earlier_positive(tmp_path):
    _state, _artifacts, log, fallback, routing = _harness(tmp_path)
    _append(log, "match")
    log.append("tool_call", {
        "tool": "runtime.v7",
        "session": "wanted",
        "input": {"value": 7},
    })

    result = _discharge(routing, {
        "scope": "observed-eof",
        "selector": "any",
        "value_pointer": "/input/value",
        "tool_names": ["runtime.v7"],
        "payload_equals": {"/session": "wanted"},
        "predicate": "equals",
        "expected": "match",
    }, refutable_by_absence=True)

    assert result.instrument == "fallback"
    assert "is not a string" in fallback.calls[0]["method"][
        "adapter_fallback_reason"]


def test_observed_eof_exact_filters_after_seq_and_compact_counterexample(
    tmp_path,
):
    _state, artifacts, log, fallback, routing = _harness(tmp_path)
    _append(log, "bad-old", session="wanted")
    _append(log, "ignored-tool", tool="other", session="wanted")
    start = _append(log, "good", session="wanted")
    _append(log, "ignored-session", session="other")
    counterexample = _append(log, "", session="wanted")

    result = _discharge(routing, {
        "scope": "observed-eof",
        "selector": "all",
        "value_pointer": "/input/value",
        "after_seq": start["seq"] - 1,
        "tool_names": ["runtime.v7"],
        "payload_equals": {"/session": "wanted"},
        "predicate": "nonempty",
    })

    assert result.outcome == OUTCOME_REFUTED
    assert result.witness_kind == WITNESS_RECORD_CONTRADICTION
    witness = _execution(artifacts, result)["observation"]["witness"]
    assert witness["candidate_count"] == 2
    assert witness["predicate_match_count"] == 1
    assert len(witness["items"]) == 1
    assert witness["items"][0]["seq"] == counterexample["seq"]
    assert fallback.calls == []


def test_empty_observed_eof_uses_the_existing_absence_gate(tmp_path):
    state, _artifacts, _log, fallback, routing = _harness(tmp_path)
    (state / "events.jsonl").touch()
    arguments = {
        "scope": "observed-eof",
        "selector": "any",
        "value_pointer": "/input/value",
        "predicate": "exists",
    }

    deferred = _discharge(routing, arguments)
    refuted = _discharge(
        routing, arguments, refutable_by_absence=True)

    assert deferred.instrument == "fallback"
    assert refuted.outcome == OUTCOME_REFUTED
    assert refuted.witness_kind == WITNESS_ABSENCE
    assert len(fallback.calls) == 1


def test_default_catalog_registers_v3_without_changing_v1_v2_contracts():
    catalog = json.loads(default_adapter_registry().catalog())
    entries = {entry["version"]: entry for entry in catalog
               if entry["name"] == "ledger.tool-call-string"}

    assert set(entries) == {1, 2, 3}
    assert entries[1]["arguments"] == LedgerToolCallStringAdapter.arguments
    assert entries[2]["arguments"] == LedgerToolCallStringAdapterV2.arguments
    assert entries[3]["arguments"] == LedgerToolCallStringAdapterV3.arguments
    assert entries[3]["accepted_method_reuse"] is False
    assert entries[3]["fixed_input_binding"] == entries[2][
        "fixed_input_binding"]
    registry = default_adapter_registry()
    arguments = {
        "scope": "observed-eof",
        "selector": "any",
        "value_pointer": "/input/value",
        "predicate": "exists",
    }
    assert registry.supports_method_reuse(_method(arguments)) is False
    unbound = _method(arguments, inputs=[])
    unbound["adapter_input_binding"] = "fixed-v1"
    assert registry.bind_inputs(unbound)["adapter_inputs"] == [{
        "ref": "ledger:events", "provenance": "ledger", "role": "subject",
    }]


def test_v3_result_still_requires_fresh_oracle_and_judger_on_every_run(
    tmp_path,
):
    _state, artifacts, log, fallback, routing = _harness(
        tmp_path, default_registry=True)
    _append(log, "alpha")
    typed = _method({
        "scope": "observed-eof",
        "selector": "any",
        "value_pointer": "/input/value",
        "tool_names": ["runtime.v7"],
        "predicate": "equals",
        "expected": "alpha",
    })
    backend = FixtureBackend({
        "oracles": {"methods": [typed]},
        "judger": {
            "sufficient": "yes",
            "faithful": "yes",
            "degradation": "n/a",
            "reason": "fresh fixture audit",
        },
    })
    cache = AcceptedMethodCache()
    game = BindingGame(
        backend, artifacts, log, routing,
        live_world=False, method_cache=cache)
    obligation = Obligation(
        "o1", "the retained exact string equals alpha", BASIS_RECORD)

    first = game.run(ROW, obligation)
    second = game.run(ROW, obligation)

    assert first.outcome == second.outcome == OUTCOME_HOLDS
    assert [call["role"] for call in backend.calls] == [
        "oracles", "judger", "oracles", "judger"]
    assert len(cache) == 0
    assert fallback.calls == []
    event_types = [event["type"] for event in log.replay()]
    assert "method_cache_hit" not in event_types
    assert event_types.count("judger_verdict") == 2
