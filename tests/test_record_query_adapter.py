"""General, fail-closed contracts for ``record.query@1``."""
from __future__ import annotations

import copy
import json

import pytest

from verifier.core.events import EventLog
from verifier.core.model import BASIS_RECORD, OUTCOME_FAILED, OUTCOME_HOLDS, Obligation
from verifier.engine.game import BindingGame
from verifier.engine.typed_judger_packet import (
    TypedAdapterJudgerPacket,
    build_typed_adapter_judger_packet,
)
from verifier.execution.adapters import (
    AdapterContext,
    AdapterInvocation,
    AdapterNotApplicable,
    AdapterRegistry,
    AdapterRoutingDischarger,
    default_adapter_registry,
)
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.execution.record_query import (
    DECODER_POLICY,
    RecordQueryAdapter,
    typed_record_query_observation_is_complete,
)
from verifier.llm.schema import FixtureBackend


class _Fallback:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append((row, obligation, method, tag))
        return DischargeResult(
            OUTCOME_HOLDS, None,
            self.artifacts.put({"fallback": len(self.calls)}),
            "agent:fallback",
        )


def _environment(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text(
        "Audit only the immutable record named by the obligation.",
        encoding="utf-8",
    )
    log = EventLog(str(state / "events.jsonl"))
    artifacts = ArtifactStore(str(state / "artifacts"))
    fallback = _Fallback(artifacts)
    registry = AdapterRegistry([RecordQueryAdapter()])
    route = AdapterRoutingDischarger(
        fallback, registry, artifacts, log, str(state), live_world=False)
    obligation = Obligation(
        "record-claim", "the declared immutable record fact holds",
        BASIS_RECORD,
    )
    return state, log, artifacts, fallback, route, obligation


def _method(query, *, scope="observed-eof", through_seq=None, after_seq=0):
    arguments = {"scope": scope, "query": query}
    if through_seq is not None:
        arguments["through_seq"] = through_seq
    if after_seq:
        arguments["after_seq"] = after_seq
    return {
        "procedure": "evaluate the declared immutable record query",
        "artifact": "a complete bounded declarative record receipt",
        "level": "deterministic record certificate",
        "basis": BASIS_RECORD,
        "strength": "decisive",
        "bridges": [],
        "adapter": "record.query",
        "adapter_version": 1,
        "adapter_input_binding": "fixed-v1",
        "adapter_arguments": json.dumps(
            arguments, ensure_ascii=False, sort_keys=True),
        "adapter_inputs": [{
            "ref": "ledger:events", "provenance": "ledger",
            "role": "subject",
        }],
    }


def _wrapper(command):
    arguments = json.dumps({
        "cmd": command, "workdir": "/workspace", "yield_time_ms": 10_000,
    }, ensure_ascii=False)
    return (
        "const r = await tools.exec_command(" + arguments + ");\n"
        "text(r.output);"
    )


def _pair(log, command="printf '雪'", *, call_id="pair-1", result="完成"):
    common = {
        "tool": "exec", "tool_use_id": call_id,
        "session": "session", "agent": None,
    }
    call = log.append("tool_call", {**common, "input": _wrapper(command)})
    output = log.append("tool_result", {
        **common, "excerpt": result, "excerpt_truncated": False,
    })
    return call, output


def _run(route, obligation, query, **kwargs):
    method = _method(query, **kwargs)
    result = route.discharge(
        {"id": obligation.claim_id, "kind": "final"}, obligation, method)
    return method, result


def test_catalog_is_declarative_record_only_and_task_agnostic():
    entry = RecordQueryAdapter().catalog_entry()
    rendered = json.dumps(entry, ensure_ascii=False)

    assert entry["name"] == "record.query"
    assert entry["version"] == 1
    assert entry["basis"] == BASIS_RECORD
    assert set(entry["query_kinds"]) == {
        "event-at", "exists", "paired-exchange",
        "last-decoded-command", "terminal-exact-command", "count",
    }
    assert entry["normalization"]["decoder_policy"] == DECODER_POLICY
    assert "never read from the ledger" in entry["normalization"][
        "expected_values"]
    assert "gcode" not in rendered.lower()
    assert "/app/out.txt" not in rendered
    assert all(set(example["adapter_arguments"]) == {"scope", "query"}
               for example in entry["examples"])
    evidence = entry["evidence_contract"]
    assert evidence["observation"]["schema"] == "gv-record-query-receipt-v1"
    assert set(evidence["observation"]["required_fields"]) == {
        "schema", "query", "query_digest", "matched", "snapshot",
        "evaluation",
    }
    assert set(evidence["observation"]["evaluation_fields_by_query"]) == {
        "event-at", "exists", "paired-exchange", "last-decoded-command",
        "terminal-exact-command", "count",
    }
    assert evidence["retention_limits"]["complete_ledger_bytes_retained"] \
        is False
    assert "never that no later opaque call" in entry["query_kinds"][
        "last-decoded-command"]
    assert "opaque or different later tool_call makes the query false" in \
        entry["query_kinds"]["terminal-exact-command"]
    assert "use this closed declarative contract" in entry["selection"]
    assert "Never weaken temporal scope" in entry["selection"]
    assert "unique exact JSON values" in entry["event_predicate"]["tests"][
        "one-of"]
    assert "decode=any-supported-v1" in entry["event_predicate"][
        "invocation_test_requirement"]
    terminal_examples = [
        example for example in entry["examples"]
        if example["adapter_arguments"]["query"]["kind"]
        == "terminal-exact-command"
    ]
    assert len(terminal_examples) == 1
    assert terminal_examples[0]["adapter_arguments"] == {
        "scope": "observed-eof",
        "query": {
            "kind": "terminal-exact-command",
            "operation": "exec_command",
            "predicate": "equals",
            "expected": "<exact command from the obligation>",
        },
    }
    shell_examples = [
        example for example in entry["examples"]
        if example["purpose"] ==
        "one decoded nonblank direct shell command exists"
    ]
    assert len(shell_examples) == 1
    shell_tests = shell_examples[0]["adapter_arguments"]["query"]["event"][
        "tests"]
    assert shell_tests[0]["predicate"] == "one-of"
    assert shell_tests[0]["expected"] == ["bash", "exec_command"]
    assert shell_tests[1]["predicate"] == "nonblank"


def test_default_registry_publishes_binds_and_oracle_sees_record_query(
    tmp_path,
):
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text(
        "Audit the immutable record.", encoding="utf-8")
    log = EventLog(str(state / "events.jsonl"))
    log.append("message", {"text": "generic retained fact"})
    artifacts = ArtifactStore(str(state / "artifacts"))
    fallback = _Fallback(artifacts)
    registry = default_adapter_registry()
    route = AdapterRoutingDischarger(
        fallback, registry, artifacts, log, str(state), live_world=False)
    query = {
        "kind": "exists",
        "event": {
            "event_type": "message",
            "tests": [{
                "source": "payload", "pointer": "/text",
                "predicate": "equals", "expected": "generic retained fact",
            }],
        },
    }
    oracle_method = _method(query)
    catalog = json.loads(registry.catalog())
    entries = [entry for entry in catalog
               if entry["name"] == "record.query" and entry["version"] == 1]
    assert len(entries) == 1
    assert entries[0]["fixed_input_binding"] == {
        "mode": "fixed-v1",
        "inputs": [{
            "ref": "ledger:events", "provenance": "ledger",
            "role": "subject",
        }],
    }
    unmaterialized = dict(oracle_method, adapter_inputs=[])
    assert registry.bind_inputs(unmaterialized)["adapter_inputs"] == [{
        "ref": "ledger:events", "provenance": "ledger", "role": "subject",
    }]

    backend = FixtureBackend({
        "oracles": {"methods": [oracle_method]},
        "judger": {
            "sufficient": "yes", "faithful": "yes", "degradation": "n/a",
            "reason": "the exact generic record receipt covers the obligation",
        },
    })
    game = BindingGame(
        backend, artifacts, log, route, live_world=False)
    obligation = Obligation(
        "record-claim", "the retained message has the declared exact text",
        BASIS_RECORD,
    )

    result = game.run(
        {"id": obligation.claim_id, "kind": "final", "text": "recorded"},
        obligation,
    )

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    assert [call["role"] for call in backend.calls] == ["oracles", "judger"]
    assert '"name":"record.query"' in backend.calls[0]["variables"][
        "adapter_catalog"]


def test_event_at_exact_seq_type_tool_and_result(tmp_path):
    _state, log, artifacts, fallback, route, obligation = _environment(tmp_path)
    for index in range(127):
        log.append("message", {"text": f"record-{index}"})
    event = log.append("tool_result", {
        "tool": "fixture.exec", "session": None, "agent": None,
        "tool_use_id": "result-128", "excerpt": "alpha 雪",
    })
    assert event["seq"] == 128
    query = {
        "kind": "event-at", "seq": 128,
        "event": {
            "event_type": "tool_result", "tool": "fixture.exec",
            "tests": [{
                "source": "payload", "pointer": "/excerpt",
                "predicate": "equals", "expected": "alpha 雪",
            }],
        },
    }

    method, result = _run(route, obligation, query)

    assert result.instrument == "adapter:record.query@1"
    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    receipt = artifacts.get(result.artifact)["execution"]["observation"]
    assert receipt["evaluation"]["selected"][0]["seq"] == 128
    assert receipt["evaluation"]["selected"][0]["payload"]["excerpt"] \
        == "alpha 雪"
    invocation = AdapterInvocation.from_method(method)
    assert invocation is not None
    assert typed_record_query_observation_is_complete(
        invocation, artifacts.get(result.artifact), {
            "outcome": result.outcome,
            "witness": result.witness,
            "witness_kind": result.witness_kind,
        })


def test_exists_supports_a_closed_one_of_predicate(tmp_path):
    _state, log, artifacts, fallback, route, obligation = _environment(tmp_path)
    _pair(log, "printf 'selected'", call_id="selected")
    query = {
        "kind": "exists",
        "event": {
            "event_type": "tool_call",
            "decode": DECODER_POLICY,
            "tests": [{
                "source": "invocation",
                "pointer": "/operation",
                "predicate": "one-of",
                "expected": ["bash", "exec_command"],
            }, {
                "source": "invocation",
                "pointer": "/command",
                "predicate": "nonblank",
            }],
        },
    }

    method, result = _run(route, obligation, query)

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    evidence = artifacts.get(result.artifact)
    assert evidence["execution"]["observation"]["evaluation"][
        "matching_event_count"] == 1
    packet = build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result, "decisive")
    assert type(packet) is TypedAdapterJudgerPacket


@pytest.mark.parametrize("expected", [[], ["bash", "bash"]])
def test_one_of_requires_a_nonempty_unique_expected_set(tmp_path, expected):
    _state, log, _artifacts, _fallback, _route, obligation = _environment(
        tmp_path)
    context = AdapterContext(str(tmp_path / "state"), False, log)
    query = {
        "kind": "exists",
        "event": {
            "event_type": "tool_call",
            "decode": DECODER_POLICY,
            "tests": [{
                "source": "invocation", "pointer": "/operation",
                "predicate": "one-of", "expected": expected,
            }],
        },
    }
    invocation = AdapterInvocation.from_method(_method(query))
    assert invocation is not None

    with pytest.raises(AdapterNotApplicable):
        RecordQueryAdapter().execute(invocation, obligation, context)


def test_any_supported_decoded_call_pairs_with_exact_result(tmp_path):
    _state, log, artifacts, fallback, route, obligation = _environment(tmp_path)
    call, output = _pair(log, "printf 'café 雪'", result="done café")
    query = {
        "kind": "paired-exchange",
        "call": {
            "event_type": "tool_call", "decode": DECODER_POLICY,
            "tests": [{
                "source": "invocation", "pointer": "/operation",
                "predicate": "equals", "expected": "exec_command",
            }, {
                "source": "invocation", "pointer": "/command",
                "predicate": "equals", "expected": "printf 'café 雪'",
            }],
        },
        "result": {
            "event_type": "tool_result",
            "tests": [{
                "source": "payload", "pointer": "/excerpt",
                "predicate": "contains", "expected": "café",
            }],
        },
    }

    method, result = _run(route, obligation, query)

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    evidence = artifacts.get(result.artifact)
    selected = evidence["execution"]["observation"]["evaluation"]["selected"]
    assert [item["seq"] for item in selected] == [call["seq"], output["seq"]]
    assert selected[0]["invocation"]["profile"] == "codex-exec-wrapper-v1"
    assert selected[0]["invocation"]["operation"] == "exec_command"

    packet = build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result,
        "(none — decisive method)",
    )
    assert type(packet) is TypedAdapterJudgerPacket
    assert packet.revalidate() is True
    document = json.loads(packet.render())
    assert document["adapter_contract"]["identity"] == "record.query@1"
    assert "events.jsonl" not in document[
        "content_addressed_evidence"]["raw_json_text"]


@pytest.mark.parametrize(("tool", "input_value", "operation", "profile"), [
    ("exec_command", {"cmd": "structured"}, "exec_command",
     "structured-exec-command-v1"),
    ("Bash", {"command": "claude"}, "bash", "claude-bash-command-v1"),
    ("bash", {"keystrokes": "terminal"}, "terminal_input",
     "lav-bash-keystrokes-v1"),
])
def test_any_supported_policy_accepts_each_exact_structured_transport(
    tmp_path, tool, input_value, operation, profile,
):
    _state, log, _artifacts, fallback, route, obligation = _environment(tmp_path)
    common = {
        "tool": tool, "tool_use_id": "transport", "session": None,
        "agent": None,
    }
    log.append("tool_call", {**common, "input": input_value})
    log.append("tool_result", {**common, "excerpt": "ok"})
    query = {
        "kind": "paired-exchange",
        "call": {
            "event_type": "tool_call", "decode": DECODER_POLICY,
            "tests": [{
                "source": "invocation", "pointer": "/operation",
                "predicate": "equals", "expected": operation,
            }],
        },
        "result": {"event_type": "tool_result", "tests": []},
    }

    _method_value, result = _run(route, obligation, query)

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    evidence = _artifacts.get(result.artifact)
    assert evidence["execution"]["observation"]["evaluation"]["selected"][0][
        "invocation"]["profile"] == profile


def test_last_decoded_command_selects_before_comparing_expected(tmp_path):
    _state, log, artifacts, fallback, route, obligation = _environment(tmp_path)
    _pair(log, "first", call_id="first")
    log.append("message", {"text": "a later non-command source action"})
    _pair(log, "second", call_id="second")
    query = {
        "kind": "last-decoded-command", "operation": "exec_command",
        "predicate": "equals", "expected": "first",
    }

    _method_value, failed = _run(route, obligation, query)

    assert failed.instrument == "adapter:record.query@1"
    assert failed.outcome == OUTCOME_FAILED
    assert failed.witness["failure_kind"] == "query-false"
    assert failed.witness_kind == "none"
    assert fallback.calls == []
    receipt = artifacts.get(failed.artifact)["execution"]["observation"]
    assert receipt["evaluation"]["selected_command"] == "second"
    assert receipt["evaluation"]["later_decoded_invocation_count"] == 0

    query["expected"] = "second"
    _method_value, held = _run(route, obligation, query)
    assert held.outcome == OUTCOME_HOLDS


def test_later_message_is_suffix_but_not_a_later_decoded_command(tmp_path):
    _state, log, artifacts, fallback, route, obligation = _environment(tmp_path)
    _pair(log, "final")
    log.append("message", {"text": "ordinary final response"})
    query = {
        "kind": "last-decoded-command", "operation": "exec_command",
        "predicate": "equals", "expected": "final",
    }

    _method_value, result = _run(route, obligation, query)

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    evaluation = artifacts.get(result.artifact)["execution"]["observation"][
        "evaluation"]
    # The paired tool result and later message are both source suffix events;
    # neither is a decoded invocation.
    assert evaluation["scanned_source_suffix_count"] == 2
    assert evaluation["later_decoded_invocation_count"] == 0


def test_terminal_exact_command_holds_at_declared_boundary_with_noncall_suffix(
    tmp_path,
):
    _state, log, artifacts, fallback, route, obligation = _environment(tmp_path)
    _pair(log, "earlier", call_id="earlier")
    terminal, _result = _pair(log, "terminal", call_id="terminal")
    boundary = log.append("message", {"text": "a later non-call source event"})
    query = {
        "kind": "terminal-exact-command", "operation": "exec_command",
        "predicate": "equals", "expected": "terminal",
    }
    method, result = _run(route, obligation, query)

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    evidence = artifacts.get(result.artifact)
    evaluation = evidence["execution"]["observation"]["evaluation"]
    assert evaluation == {
        "kind": "terminal-exact-command",
        "tool_call_count": 2,
        "selected": [evaluation["selected"][0]],
        "selected_decode_status": "decoded",
        "selected_operation": "exec_command",
        "selected_command": "terminal",
        "scanned_source_suffix_count": 2,
        "later_tool_call_count": 0,
    }
    assert evaluation["selected"][0]["seq"] == terminal["seq"]
    packet = build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result, "decisive")
    assert type(packet) is TypedAdapterJudgerPacket
    assert packet.revalidate() is True


@pytest.mark.parametrize(
    "later_kind", ["opaque", "different-command", "different-operation"])
def test_terminal_exact_command_does_not_skip_a_later_tool_call(
    tmp_path, later_kind,
):
    _state, log, artifacts, fallback, route, obligation = _environment(tmp_path)
    _pair(log, "expected", call_id="expected")
    if later_kind == "opaque":
        later = log.append("tool_call", {
            "tool": "unrecognized.transport",
            "tool_use_id": "later", "session": None, "agent": None,
            "input": {"cmd": "expected"},
        })
    elif later_kind == "different-command":
        later = log.append("tool_call", {
            "tool": "exec_command",
            "tool_use_id": "later", "session": None, "agent": None,
            "input": {"cmd": "different"},
        })
    else:
        later = log.append("tool_call", {
            "tool": "Bash",
            "tool_use_id": "later", "session": None, "agent": None,
            "input": {"command": "expected"},
        })
    query = {
        "kind": "terminal-exact-command", "operation": "exec_command",
        "predicate": "equals", "expected": "expected",
    }

    _method_value, result = _run(route, obligation, query)

    assert result.outcome == OUTCOME_FAILED
    assert result.instrument == "adapter:record.query@1"
    assert fallback.calls == []
    evaluation = artifacts.get(result.artifact)["execution"]["observation"][
        "evaluation"]
    assert evaluation["selected"][0]["seq"] == later["seq"]
    assert evaluation["tool_call_count"] == 2
    assert evaluation["selected_decode_status"] == (
        "opaque" if later_kind == "opaque" else "decoded")
    assert evaluation["later_tool_call_count"] == 0


def test_terminal_exact_command_rejects_prefix_scope(tmp_path):
    _state, log, _artifacts, fallback, route, obligation = _environment(tmp_path)
    _call, boundary = _pair(log, "terminal")
    log.append("tool_call", {
        "tool": "unrecognized.transport", "tool_use_id": "outside-prefix",
        "session": None, "agent": None, "input": {"cmd": "terminal"},
    })
    query = {
        "kind": "terminal-exact-command", "operation": "exec_command",
        "predicate": "equals", "expected": "terminal",
    }

    _method_value, result = _run(
        route, obligation, query, scope="prefix",
        through_seq=boundary["seq"])

    assert result.outcome == OUTCOME_HOLDS
    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


def test_complete_no_match_is_failed_never_refuted_or_agent_fallback(tmp_path):
    _state, log, _artifacts, fallback, route, obligation = _environment(tmp_path)
    log.append("message", {"text": "different"})
    query = {
        "kind": "exists",
        "event": {
            "event_type": "message",
            "tests": [{
                "source": "payload", "pointer": "/text",
                "predicate": "equals", "expected": "missing",
            }],
        },
    }

    _method_value, result = _run(route, obligation, query)

    assert result.outcome == OUTCOME_FAILED
    assert result.witness_kind == "none"
    assert result.instrument == "adapter:record.query@1"
    assert fallback.calls == []


def test_duplicate_pair_identity_is_ambiguous_and_falls_back(tmp_path):
    _state, log, _artifacts, fallback, route, obligation = _environment(tmp_path)
    _pair(log, "one", call_id="duplicate")
    log.append("tool_call", {
        "tool": "exec", "tool_use_id": "duplicate", "session": "session",
        "agent": None, "input": _wrapper("two"),
    })
    query = {
        "kind": "paired-exchange",
        "call": {
            "event_type": "tool_call", "decode": DECODER_POLICY,
            "tests": [],
        },
        "result": {"event_type": "tool_result", "tests": []},
    }

    _method_value, result = _run(route, obligation, query)

    assert result.instrument == "agent:fallback"
    assert len(fallback.calls) == 1


def test_count_retains_complete_aggregate_not_the_whole_ledger(tmp_path):
    _state, log, artifacts, fallback, route, obligation = _environment(tmp_path)
    for index in range(5):
        log.append("message", {"text": "selected" if index < 3 else "other"})
    query = {
        "kind": "count",
        "event": {
            "event_type": "message",
            "tests": [{
                "source": "payload", "pointer": "/text",
                "predicate": "equals", "expected": "selected",
            }],
        },
        "comparison": {"operator": "equals", "expected": 3},
    }

    method, result = _run(route, obligation, query)

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    evidence = artifacts.get(result.artifact)
    evaluation = evidence["execution"]["observation"]["evaluation"]
    assert evaluation["count"] == 3
    assert len(evaluation["representative"]) == 1
    assert len(evaluation["matching_events_digest"]) == 64
    packet = build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result, "decisive")
    assert type(packet) is TypedAdapterJudgerPacket


def test_zero_count_hold_is_eligible_for_restricted_packet(tmp_path):
    _state, log, artifacts, fallback, route, obligation = _environment(tmp_path)
    log.append("message", {"text": "other"})
    query = {
        "kind": "count",
        "event": {
            "event_type": "tool_call", "tests": [],
        },
        "comparison": {"operator": "equals", "expected": 0},
    }

    method, result = _run(route, obligation, query)

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    evaluation = artifacts.get(result.artifact)["execution"]["observation"][
        "evaluation"]
    assert evaluation["count"] == 0
    assert evaluation["representative"] == []
    packet = build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result, "decisive")
    assert type(packet) is TypedAdapterJudgerPacket


def test_prefix_does_not_read_malformed_later_suffix(tmp_path):
    state, log, _artifacts, fallback, route, obligation = _environment(tmp_path)
    event = log.append("message", {"text": "bounded"})
    query = {
        "kind": "event-at", "seq": event["seq"],
        "event": {"event_type": "message", "tests": []},
    }
    with (state / "events.jsonl").open("ab") as handle:
        handle.write(b'{"incomplete":')

    _method_value, result = _run(
        route, obligation, query, scope="prefix", through_seq=event["seq"])

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []


@pytest.mark.parametrize("mutation", [
    "event", "operation", "later", "scope", "digest", "consumed",
    "extra-evaluation", "out-of-scope",
])
def test_typed_gate_rejects_mutated_complete_receipt(tmp_path, mutation):
    _state, log, artifacts, _fallback, route, obligation = _environment(tmp_path)
    _pair(log, "final")
    query = {
        "kind": "last-decoded-command", "operation": "exec_command",
        "predicate": "equals", "expected": "final",
    }
    method, result = _run(route, obligation, query)
    evidence = artifacts.get(result.artifact)
    changed = copy.deepcopy(evidence)
    observation = changed["execution"]["observation"]
    if mutation == "event":
        observation["evaluation"]["selected"][0]["payload"]["input"] = "x"
    elif mutation == "operation":
        observation["evaluation"]["selected_operation"] = "bash"
    elif mutation == "later":
        observation["evaluation"]["later_decoded_invocation_count"] = 1
    elif mutation == "scope":
        observation["snapshot"]["complete_file_at_observation"] = False
    elif mutation == "digest":
        observation["snapshot"]["projection_digest"] = "0" * 64
    elif mutation == "extra-evaluation":
        observation["evaluation"]["untrusted"] = True
    elif mutation == "out-of-scope":
        observation["evaluation"]["selected"][0]["seq"] = 0
    else:
        changed["execution"]["consumed"][0]["source_event_count"] += 1
    invocation = AdapterInvocation.from_method(method)
    assert invocation is not None

    assert not typed_record_query_observation_is_complete(
        invocation, changed, {
            "outcome": result.outcome,
            "witness": result.witness,
            "witness_kind": result.witness_kind,
        })


@pytest.mark.parametrize("mutation", [
    "capsule", "decode-status", "operation", "command", "call-count",
    "later-call", "extra-field",
])
def test_typed_terminal_gate_rejects_mutated_receipt(tmp_path, mutation):
    _state, log, artifacts, _fallback, route, obligation = _environment(tmp_path)
    _pair(log, "terminal")
    query = {
        "kind": "terminal-exact-command", "operation": "exec_command",
        "predicate": "equals", "expected": "terminal",
    }
    method, result = _run(route, obligation, query)
    changed = copy.deepcopy(artifacts.get(result.artifact))
    evaluation = changed["execution"]["observation"]["evaluation"]
    if mutation == "capsule":
        evaluation["selected"][0]["payload"]["input"] = "opaque"
    elif mutation == "decode-status":
        evaluation["selected_decode_status"] = "opaque"
    elif mutation == "operation":
        evaluation["selected_operation"] = "bash"
    elif mutation == "command":
        evaluation["selected_command"] = "different"
    elif mutation == "call-count":
        evaluation["tool_call_count"] = 0
    elif mutation == "later-call":
        evaluation["later_tool_call_count"] = 1
    else:
        evaluation["additional"] = True
    invocation = AdapterInvocation.from_method(method)
    assert invocation is not None

    assert not typed_record_query_observation_is_complete(
        invocation, changed, {
            "outcome": result.outcome,
            "witness": result.witness,
            "witness_kind": result.witness_kind,
        })


@pytest.mark.parametrize("mutation", ["peer-digest", "peer-seq", "metadata"])
def test_typed_pair_gate_rejects_incomplete_correlation(
    tmp_path, mutation,
):
    _state, log, artifacts, _fallback, route, obligation = _environment(tmp_path)
    _pair(log, "paired")
    query = {
        "kind": "paired-exchange",
        "call": {
            "event_type": "tool_call", "decode": DECODER_POLICY,
            "tests": [],
        },
        "result": {"event_type": "tool_result", "tests": []},
    }
    method, result = _run(route, obligation, query)
    evidence = copy.deepcopy(artifacts.get(result.artifact))
    selected = evidence["execution"]["observation"]["evaluation"]["selected"]
    if mutation == "peer-digest":
        selected[0]["correlation"]["peer_sha256"] = "0" * 64
    elif mutation == "peer-seq":
        selected[0]["correlation"]["peer_seq"] = selected[0]["seq"]
    else:
        selected[1]["payload"]["session"] = "different"
        selected[1]["event_sha256"] = EventLog.event_digest({
            "seq": selected[1]["seq"], "type": selected[1]["type"],
            "payload": selected[1]["payload"],
        })
        selected[0]["correlation"]["peer_sha256"] = selected[1][
            "event_sha256"]
    invocation = AdapterInvocation.from_method(method)
    assert invocation is not None

    assert not typed_record_query_observation_is_complete(
        invocation, evidence, {
            "outcome": result.outcome,
            "witness": result.witness,
            "witness_kind": result.witness_kind,
        })


def test_source_append_breaks_packet_replay_but_verifier_append_does_not(tmp_path):
    _state, log, artifacts, _fallback, route, obligation = _environment(tmp_path)
    _pair(log, "final")
    query = {
        "kind": "last-decoded-command", "operation": "exec_command",
        "predicate": "equals", "expected": "final",
    }
    method, result = _run(route, obligation, query)
    packet = build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result, "decisive")
    assert type(packet) is TypedAdapterJudgerPacket

    log.append("judger_verdict", {"verifier_only": True})
    assert packet.revalidate() is True
    log.append("message", {"text": "late source continuation"})
    assert packet.revalidate() is False


def test_terminal_packet_replay_rejects_a_later_opaque_call(tmp_path):
    _state, log, artifacts, _fallback, route, obligation = _environment(tmp_path)
    _pair(log, "terminal")
    query = {
        "kind": "terminal-exact-command", "operation": "exec_command",
        "predicate": "equals", "expected": "terminal",
    }
    method, result = _run(route, obligation, query)
    packet = build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result, "decisive")
    assert type(packet) is TypedAdapterJudgerPacket
    assert packet.revalidate() is True

    log.append("tool_call", {
        "tool": "unrecognized.transport", "tool_use_id": "late-opaque",
        "session": None, "agent": None, "input": {"cmd": "terminal"},
    })

    assert packet.revalidate() is False


@pytest.mark.parametrize("defect", [
    "unknown-query", "non-exact-command", "invocation-without-decode",
])
def test_malformed_method_is_inapplicable(defect, tmp_path):
    _state, log, _artifacts, _fallback, _route, obligation = _environment(tmp_path)
    context = AdapterContext(str(tmp_path / "state"), False, log)
    if defect == "unknown-query":
        query = {"kind": "shell", "command": "true"}
    elif defect == "non-exact-command":
        query = {
            "kind": "last-decoded-command", "operation": "exec_command",
            "predicate": "contains", "expected": "true",
        }
    else:
        query = {
            "kind": "exists",
            "event": {
                "event_type": "tool_call",
                "tests": [{
                    "source": "invocation", "pointer": "/command",
                    "predicate": "nonempty",
                }],
            },
        }
    method = _method(query)
    invocation = AdapterInvocation.from_method(method)
    assert invocation is not None

    with pytest.raises(AdapterNotApplicable):
        RecordQueryAdapter().execute(invocation, obligation, context)
