"""Generality, replay, and fail-closed tests for ``record.program@1``."""
from __future__ import annotations

import json
import os

import pytest

import verifier.execution.record_program as record_program
from verifier.core.events import EventLog
from verifier.core.model import BASIS_RECORD, BASIS_WORLD, OUTCOME_HOLDS, Obligation
from verifier.engine.typed_judger_packet import (
    build_typed_adapter_judger_packet,
)
from verifier.execution.adapters import AdapterRoutingDischarger, default_adapter_registry
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore


ROW = {"id": "claim", "kind": "final", "text": "the recorded action occurred"}


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
    state.mkdir()
    (state / "task.md").write_text(
        "Verify only what the immutable record establishes.", encoding="utf-8")
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = _Fallback(artifacts)
    route = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=False,
    )
    obligation = Obligation(
        "claim", "the ledger contains the stated interaction", BASIS_RECORD)
    return state, artifacts, log, fallback, route, obligation


def _method(source, *, scope="observed-eof", through_seq=None,
            basis=BASIS_RECORD, binding=True, inputs=None):
    arguments = {"scope": scope, "source": source}
    if through_seq is not None:
        arguments["through_seq"] = through_seq
    method = {
        "procedure": "inspect the normalized immutable event projection",
        "artifact": "a bounded pure record-program receipt",
        "level": "deterministic pure inspection",
        "basis": basis,
        "strength": "decisive",
        "bridges": [],
        "adapter": "record.program",
        "adapter_version": 1,
        "adapter_arguments": json.dumps(
            arguments, ensure_ascii=False, sort_keys=True),
        "adapter_inputs": [] if inputs is None else inputs,
    }
    if binding:
        method["adapter_input_binding"] = "fixed-v1"
    return method


def _bound(route, method):
    return route.prepare_method(method)


def _wrapper(command, *, tail=""):
    arguments = json.dumps({
        "cmd": command, "workdir": "/app", "yield_time_ms": 10000,
    }, ensure_ascii=False)
    return f"const r = await tools.exec_command({arguments});\ntext(r.output);\n{tail}"


def _pair(log, command, *, call_id="call-1", tool="exec", result="done"):
    common = {
        "tool": tool, "tool_use_id": call_id, "session": "s1", "agent": None,
    }
    call = log.append("tool_call", {
        **common,
        "input": (_wrapper(command) if tool == "exec" else {"cmd": command}),
    })
    output = log.append("tool_result", {
        **common,
        "excerpt": result,
        "excerpt_truncated": False,
        "response_chars": len(result),
        "response_sha256": "0" * 64,
    })
    return call, output


def _program(command, *, scope="observed-eof", require_pair=True):
    if scope == "prefix":
        witness_scope = (
            '{"kind": record["scope"]["kind"], '
            '"through_seq": record["scope"]["through_seq"], '
            '"complete_through_boundary": '
            'record["scope"]["complete_through_boundary"], '
            '"proves_future_closure": record["scope"]["proves_future_closure"], '
            '"source_events_digest": record["source_events_digest"]}'
        )
    else:
        witness_scope = (
            '{"kind": record["scope"]["kind"], '
            '"complete_file_at_observation": '
            'record["scope"]["complete_file_at_observation"], '
            '"proves_future_closure": record["scope"]["proves_future_closure"], '
            '"source_events_digest": record["source_events_digest"]}'
        )
    pair_clause = (' and event["correlation"]["status"] == "paired"'
                   if require_pair else "")
    return f'''record = input_json(0)
found = None
for event in record["events"]:
    if event["type"] == "tool_call" and event["invocation"]["status"] == "decoded" and event["invocation"]["command"] == {json.dumps(command, ensure_ascii=False)}{pair_clause}:
        found = event
scope = {witness_scope}
require(found is not None, "the exact recorded invocation exists", {{"event_refs": [{{"seq": found["seq"], "sha256": found["event_sha256"]}}], "facts": {{"command": found["invocation"]["command"], "correlation": found["correlation"]["status"]}}, "scope": scope}})
'''


def test_catalog_and_fixed_binding_are_record_only(tmp_path):
    _state, _artifacts, _log, _fallback, route, _obligation = _environment(tmp_path)
    entry = next(item for item in json.loads(route.adapter_catalog())
                 if item["name"] == "record.program")

    assert entry["version"] == 1
    assert entry["basis"] == BASIS_RECORD
    assert entry["accepted_method_reuse"] is False
    assert entry["fixed_input_binding"] == {
        "mode": "fixed-v1",
        "inputs": [{
            "ref": "ledger:events", "provenance": "ledger", "role": "subject",
        }],
    }
    assert "no shell parsing" in entry["normalization"]["invocations"]
    assert "future closure" in entry["normalization"]["observed_eof"]
    assert "ordinary fresh toolful Judger" in entry["judgment_transport"]
    assert "task" not in entry["example"]
    assert entry["normalized_document_shape"]["schema"] \
        == record_program.NORMALIZATION_SCHEMA
    assert entry["field_applicability"]["invocation"]["tool_call"] \
        ["statuses"] == ["decoded", "opaque"]
    assert entry["field_applicability"]["invocation"] \
        ["all_other_source_events"]["status"] == "not-applicable"
    assert entry["field_applicability"]["correlation"] \
        ["tool_call_and_tool_result"]["statuses"] == [
            "paired", "unpaired", "ambiguous"]
    assert {recipe["operation"] for recipe in entry["recipes"]} == {
        "any-paired-decoded-invocation",
        "prefix-call-result-pair",
        "last-decoded-command-equals",
        "tool-result-string-equals",
        "aggregate-source-event-count",
    }
    prefix_pair = next(
        recipe for recipe in entry["recipes"]
        if recipe["operation"] == "prefix-call-result-pair")
    assert "correlation" in prefix_pair["source"]
    last_command = next(
        recipe for recipe in entry["recipes"]
        if recipe["operation"] == "last-decoded-command-equals")
    assert "invocation" in last_command["source"]
    assert "terminal" not in last_command["source"]


@pytest.mark.parametrize("scope", ["prefix", "observed-eof"])
def test_unicode_command_and_unique_pair_are_purely_inspected(
    tmp_path, scope,
):
    _state, artifacts, log, fallback, route, obligation = _environment(tmp_path)
    call, result_event = _pair(log, "printf '雪∕é'")
    through = result_event["seq"] if scope == "prefix" else None
    method = _bound(route, _method(
        _program("printf '雪∕é'", scope=scope),
        scope=scope, through_seq=through,
    ))

    result = route.discharge(ROW, obligation, method)

    assert result.instrument == "adapter:record.program@1"
    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    evidence = artifacts.get(result.artifact)
    observed = evidence["execution"]["observation"]
    assert observed["normalization_schema"] == record_program.NORMALIZATION_SCHEMA
    assert observed["record_scope"]["proves_future_closure"] is False
    assert observed["typed_observation_validation"] == {
        "valid": True,
        "referenced_events": [{
            "seq": call["seq"],
            "sha256": EventLog.event_digest(call),
        }],
    }
    # Arbitrary record programs stay on the ordinary fresh toolful Judger
    # path.  A digest and sparse event references are not a complete
    # independently inspectable ledger projection.
    packet = build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result,
        "(none — decisive method, no degradation to audit)",
    )
    assert packet is None


def test_observed_eof_projection_ignores_verifier_events_not_source_events(tmp_path):
    _state, artifacts, log, fallback, route, obligation = _environment(tmp_path)
    _pair(log, "echo final")
    method = _bound(route, _method(_program("echo final")))
    first = route.discharge(ROW, obligation, method)
    assert first.instrument == "adapter:record.program@1"

    log.append("status_change", {"status": "verifying"})
    second = route.discharge(ROW, obligation, method)
    assert second.instrument == "adapter:record.program@1"
    assert artifacts.get(first.artifact)["execution"]["observation"][
        "projection"] == artifacts.get(second.artifact)["execution"][
            "observation"]["projection"]
    log.append("message", {"text": "late solver continuation"})
    third = route.discharge(ROW, obligation, method)
    assert third.instrument == "adapter:record.program@1"
    assert artifacts.get(second.artifact)["execution"]["observation"][
        "projection"] != artifacts.get(third.artifact)["execution"][
            "observation"]["projection"]
    assert fallback.calls == []


def test_prefix_execution_does_not_read_a_malformed_later_suffix(tmp_path):
    state, artifacts, log, fallback, route, obligation = _environment(tmp_path)
    _call, result_event = _pair(log, "echo bounded")
    method = _bound(route, _method(
        _program("echo bounded", scope="prefix"),
        scope="prefix", through_seq=result_event["seq"],
    ))
    first = route.discharge(ROW, obligation, method)
    assert first.instrument == "adapter:record.program@1"

    with (state / "events.jsonl").open("ab") as handle:
        handle.write(b'{"incomplete":')
    second = route.discharge(ROW, obligation, method)
    assert second.instrument == "adapter:record.program@1"
    assert artifacts.get(first.artifact)["execution"]["observation"] \
        == artifacts.get(second.artifact)["execution"]["observation"]
    assert fallback.calls == []


@pytest.mark.parametrize("tail", [
    "text('extra');",
    "await tools.exec_command({\"cmd\":\"second\"});",
])
def test_non_exact_or_multi_call_wrapper_is_opaque(tmp_path, tail):
    _state, artifacts, log, fallback, route, obligation = _environment(tmp_path)
    common = {
        "tool": "exec", "tool_use_id": "opaque-1", "session": "s", "agent": None,
    }
    call = log.append("tool_call", {
        **common, "input": _wrapper("echo safe", tail=tail),
    })
    log.append("tool_result", {**common, "excerpt": "", "excerpt_truncated": False})
    source = '''record = input_json(0)
found = None
for event in record["events"]:
    if event["type"] == "tool_call" and event["invocation"]["status"] == "opaque":
        found = event
scope = {"kind": record["scope"]["kind"], "complete_file_at_observation": record["scope"]["complete_file_at_observation"], "proves_future_closure": record["scope"]["proves_future_closure"], "source_events_digest": record["source_events_digest"]}
require(found is not None, "unknown transport remains opaque", {"event_refs": [{"seq": found["seq"], "sha256": found["event_sha256"]}], "facts": {"status": found["invocation"]["status"]}, "scope": scope})
'''
    method = _bound(route, _method(source))

    result = route.discharge(ROW, obligation, method)

    assert result.instrument == "adapter:record.program@1"
    assert fallback.calls == []
    assert artifacts.get(result.artifact)["execution"]["observation"][
        "typed_observation_validation"]["referenced_events"][0]["seq"] == call["seq"]


@pytest.mark.parametrize("mutation", ["duplicate", "metadata", "reversed"])
def test_ambiguous_or_mismatched_records_never_pair(tmp_path, mutation):
    _state, _artifacts, log, fallback, route, obligation = _environment(tmp_path)
    common = {"tool": "exec", "tool_use_id": "same", "session": "s", "agent": None}
    if mutation == "reversed":
        log.append("tool_result", {**common, "excerpt": "early"})
        log.append("tool_call", {**common, "input": _wrapper("echo x")})
    else:
        log.append("tool_call", {**common, "input": _wrapper("echo x")})
        if mutation == "duplicate":
            log.append("tool_call", {**common, "input": _wrapper("echo y")})
        log.append("tool_result", {
            **common,
            **({"session": "other"} if mutation == "metadata" else {}),
            "excerpt": "done",
        })
    # A paired requirement must fail and route to the unchanged executor.
    method = _bound(route, _method(_program("echo x")))

    result = route.discharge(ROW, obligation, method)

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("left,right", [
    (1, True),
    (1, 1.0),
    ([1], [True]),
])
def test_correlation_metadata_equality_is_json_type_sensitive(
    tmp_path, left, right,
):
    _state, _artifacts, log, fallback, route, obligation = _environment(tmp_path)
    call_id = "typed-metadata"
    log.append("tool_call", {
        "tool": "exec", "tool_use_id": call_id, "session": left,
        "agent": None, "input": _wrapper("echo x"),
    })
    log.append("tool_result", {
        "tool": "exec", "tool_use_id": call_id, "session": right,
        "agent": None, "excerpt": "done",
    })
    method = _bound(route, _method(_program("echo x")))

    result = route.discharge(ROW, obligation, method)

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("scope,through", [
    ("prefix", None),
    ("prefix", 0),
    ("observed-eof", 2),
    ("unknown", None),
])
def test_invalid_scope_shapes_fallback(tmp_path, scope, through):
    _state, _artifacts, log, fallback, route, obligation = _environment(tmp_path)
    _pair(log, "echo x")
    method = _bound(route, _method(
        _program("echo x", scope=("prefix" if scope == "prefix" else "observed-eof")),
        scope=scope, through_seq=through,
    ))

    result = route.discharge(ROW, obligation, method)

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


def test_wrong_basis_nonfixed_or_extra_subject_fallback(tmp_path):
    _state, _artifacts, log, fallback, route, obligation = _environment(tmp_path)
    _pair(log, "echo x")
    source = _program("echo x")
    fixed = {"ref": "ledger:events", "provenance": "ledger", "role": "subject"}
    extra = {"ref": "task.md", "provenance": "task_given", "role": "subject"}

    methods = [
        _method(source, basis=BASIS_WORLD),
        _method(source, binding=False, inputs=[fixed]),
        _method(source, inputs=[extra]),
    ]
    for method in methods:
        result = route.discharge(ROW, obligation, _bound(route, method))
        assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 3


def test_record_program_refuses_world_floor_even_with_record_method(tmp_path):
    _state, _artifacts, log, fallback, route, _obligation = _environment(tmp_path)
    _pair(log, "echo x")
    world_obligation = Obligation("claim", "a world fact", BASIS_WORLD)
    method = _bound(route, _method(_program("echo x")))

    result = route.discharge(ROW, world_obligation, method)

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("raw", [
    b'{"seq":1,"seq":1,"type":"message","payload":{}}\n',
    b'{"seq":2,"type":"message","payload":{}}\n',
    b'{"seq":1,"type":"message","payload":{"x":NaN}}\n',
    b'{"seq":1,"type":"message","payload":{}}',
    b'{"seq":1,"type":"message","payload":{"x":"\\ud800"}}\n',
    b'{"seq":1,"type":"tool_call","payload":{"tool_use_id":"\\ud800"}}\n',
])
def test_malformed_jsonl_or_unicode_falls_back(tmp_path, raw):
    state, _artifacts, _log, fallback, route, obligation = _environment(tmp_path)
    (state / "events.jsonl").write_bytes(raw)
    source = '''record = input_json(0)
require(len(record["events"]) > 0, "has event", {"event_refs": [], "facts": {}, "scope": {}})
'''
    method = _bound(route, _method(source))

    result = route.discharge(ROW, obligation, method)

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


def test_hard_linked_ledger_is_rejected(tmp_path):
    state, _artifacts, log, fallback, route, obligation = _environment(tmp_path)
    _pair(log, "echo x")
    os.link(state / "events.jsonl", state / "alias.jsonl")
    method = _bound(route, _method(_program("echo x")))

    result = route.discharge(ROW, obligation, method)

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("observed", [
    'event["payload"]',
    '{"event_refs": [{"seq": event["seq"], "sha256": "0" * 64}], "facts": {}, "scope": scope}',
    '{"event_refs": [], "facts": {}, "scope": scope}',
])
def test_weak_or_forged_observation_uses_toolful_judger(tmp_path, observed):
    _state, artifacts, log, fallback, route, obligation = _environment(tmp_path)
    call, _result_event = _pair(log, "echo x")
    source = f'''record = input_json(0)
event = record["events"][0]
scope = {{"kind": record["scope"]["kind"], "complete_file_at_observation": record["scope"]["complete_file_at_observation"], "proves_future_closure": record["scope"]["proves_future_closure"], "source_events_digest": record["source_events_digest"]}}
require(event["seq"] == {call["seq"]}, "record dependency", {observed})
'''
    method = _bound(route, _method(source))
    result = route.discharge(ROW, obligation, method)

    assert result.instrument == "adapter:record.program@1"
    assert fallback.calls == []
    assert build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result, "none") is None


def test_projection_size_cap_falls_back(tmp_path, monkeypatch):
    _state, _artifacts, log, fallback, route, obligation = _environment(tmp_path)
    _pair(log, "echo a moderately sized command")
    monkeypatch.setattr(record_program, "MAX_INPUT_BYTES", 64)
    method = _bound(route, _method(_program("echo a moderately sized command")))

    result = route.discharge(ROW, obligation, method)

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1
