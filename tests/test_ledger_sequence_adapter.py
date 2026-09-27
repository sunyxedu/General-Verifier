"""Fail-closed, task-independent semantics for ``ledger.sequence@1``."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import verifier.execution.ledger_sequence as ledger_sequence
from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_RECORD,
    OUTCOME_FAILED,
    OUTCOME_HOLDS,
    WITNESS_NONE,
    Obligation,
)
from verifier.engine.binding.game import BindingGame
from verifier.execution.judger_packets.typed import (
    TypedAdapterJudgerPacket,
    build_typed_adapter_judger_packet,
)
from verifier.execution.adapters import (
    AdapterContext,
    AdapterInvocation,
    AdapterNotApplicable,
    AdapterRoutingDischarger,
    default_adapter_registry,
)
from verifier.execution.instruments import ArtifactStore
from verifier.execution.ledger_sequence import (
    LedgerSequenceAdapter,
    LedgerSequenceCompleteAuditAdapter,
    typed_sequence_observation_is_complete,
)
from verifier.llm.schema import FixtureBackend


def _method(arguments, *, version=1):
    return {
        "procedure": "inspect one declared ordered record sequence",
        "artifact": "a bounded canonical ledger receipt",
        "level": "deterministic",
        "basis": BASIS_RECORD,
        "strength": "decisive",
        "bridges": [],
        "adapter": "ledger.sequence",
        "adapter_version": version,
        "adapter_input_binding": "fixed-v1",
        "adapter_arguments": json.dumps(
            arguments, ensure_ascii=False, sort_keys=True),
        "adapter_inputs": [{
            "ref": "ledger:events", "provenance": "ledger", "role": "subject",
        }],
    }


def _environment(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text(
        "Inspect the declared record sequence.", encoding="utf-8")
    log = EventLog(str(state / "events.jsonl"))
    return log, AdapterContext(str(state), live_world=False, log=log)


class _NoFallback:
    def discharge(self, *_args, **_kwargs):
        raise AssertionError("the sequence adapter must not fall back")


def _execute(context, arguments):
    adapter = LedgerSequenceAdapter()
    method = _method(arguments)
    invocation = AdapterInvocation.from_method(method)
    assert invocation is not None
    adapter.validate(invocation, method)
    obligation = Obligation(
        "neutral", "the declared record relation holds", BASIS_RECORD)
    return adapter.execute(invocation, obligation, context)


def _execute_complete_audit(context, arguments, *, refutable=False):
    adapter = LedgerSequenceCompleteAuditAdapter()
    method = _method(arguments, version=2)
    invocation = AdapterInvocation.from_method(method)
    assert invocation is not None
    adapter.validate(invocation, method)
    obligation = Obligation(
        "neutral", "the declared record relation holds", BASIS_RECORD,
        refutable_by_absence=refutable)
    return adapter.execute(invocation, obligation, context)


def _wrapper(command, *, leading="", trailing=""):
    encoded = json.dumps({
        "cmd": command, "workdir": "/app", "yield_time_ms": 10_000,
    }, ensure_ascii=False)
    return (
        f"{leading}const r = await tools.exec_command({encoded});\n"
        f"text(r.output);\n{trailing}"
    )


def _pair(log, command="printf 雪", *, call_id="call-1", session="s1",
          agent=None, call_tool="exec", result_tool=None):
    common = {
        "tool": call_tool, "tool_use_id": call_id,
        "session": session, "agent": agent,
    }
    call = log.append("tool_call", {
        **common, "input": _wrapper(command), "label": "café 雪",
    })
    result = log.append("tool_result", {
        **{**common, "tool": result_tool or call_tool},
        "excerpt": "完成 雪", "response_sha256": "0" * 64,
    })
    return call, result


def _paired_arguments(scope, *, through_seq=None, terminal=False):
    arguments = {
        "scope": scope,
        "clauses": [{
            "id": "invoke", "event_type": "tool_call", "tool": "exec",
            "session": "s1", "agent": None,
            "payload_strings": [{
                "pointer": "/label", "predicate": "contains", "expected": "fé 雪",
            }],
            "record_program": {
                "decoder": "record.program@1",
                "profile": "codex-exec-wrapper-v1",
                "strings": [{
                    "pointer": "/arguments/cmd", "predicate": "equals",
                    "expected": "printf 雪",
                }, {
                    "pointer": "/arguments/workdir", "predicate": "nonempty",
                }],
            },
        }, {
            "id": "result", "event_type": "tool_result", "tool": "exec",
            "session": "s1", "agent": None,
            "payload_strings": [{
                "pointer": "/excerpt", "predicate": "contains", "expected": "雪",
            }],
        }],
        "relations": [{
            "kind": "ordered", "before": "invoke", "after": "result",
        }],
    }
    if scope == "observed-eof":
        arguments["relations"].append({
            "kind": "paired", "call": "invoke", "result": "result",
        })
    if through_seq is not None:
        arguments["through_seq"] = through_seq
    if terminal:
        arguments["terminal"] = {
            "source": "result", "policy": "no-later-solver-action-v1",
        }
    return arguments


def test_prefix_holds_with_full_canonical_witness_and_structured_relations(
    tmp_path,
):
    log, context = _environment(tmp_path)
    log.append("message", {"text": "unrelated"})
    call, result_event = _pair(log)

    result = _execute(context, _paired_arguments(
        "prefix", through_seq=result_event["seq"]))

    assert result.outcome == OUTCOME_HOLDS
    assert result.witness is None
    assert result.witness_kind == WITNESS_NONE
    observed = result.observation
    assert [item["seq"] for item in observed["witness_events"]] == [
        call["seq"], result_event["seq"],
    ]
    for declaration, event in zip(observed["witness_events"], (call, result_event)):
        assert declaration["type"] == event["type"]
        assert declaration["payload"] == event["payload"]
        assert declaration["event_digest"] == EventLog.event_digest(event)
        assert "canonical_event" not in declaration
        assert declaration["evaluation"]["matched"] is True
    assert observed["witness_events"][0]["evaluation"][
        "record_program"]["string_predicates"][0] == {
            "pointer": "/arguments/cmd", "predicate": "equals",
            "expected": "printf 雪", "actual": "printf 雪", "matched": True,
        }
    assert [relation["kind"] for relation in observed["relations"]] == [
        "ordered",
    ]
    assert observed["terminal"] is None
    assert observed["snapshot"]["projection_digest"] == result.consumed[0][
        "projection_digest"]


def test_observed_eof_retains_stable_source_projection_identity(tmp_path):
    log, context = _environment(tmp_path)
    _call, result_event = _pair(log)

    first = _execute(context, _paired_arguments(
        "observed-eof", terminal=True))
    log.append("adapter_discharge", {"verifier_only": True})
    second = _execute(context, _paired_arguments(
        "observed-eof", terminal=True))

    snapshot = first.observation["snapshot"]
    assert snapshot["resolved_source_through_seq"] == result_event["seq"]
    assert snapshot["source_event_count"] == 2
    assert snapshot["complete_file_at_observation"] is True
    assert snapshot["proves_future_closure"] is False
    assert "observed_eof" not in snapshot
    assert "raw_sha256" not in json.dumps(snapshot)
    assert first.observation == second.observation
    assert first.consumed == second.consumed
    assert first.observation["terminal"] == {
        "source": "result",
        "source_seq": result_event["seq"],
        "policy": "no-later-solver-action-v1",
        "solver_action_types": ["commit", "message", "session_start", "tool_call"],
        "no_later_solver_action": True,
        "scanned_source_suffix_count": 0,
        "later_solver_action_count": 0,
        "scanned_source_event_count": 2,
        "scanned_source_events_digest": snapshot["source_events_digest"],
        "proves_future_closure": False,
    }

    log.append("message", {"text": "a new source action"})
    with pytest.raises(AdapterNotApplicable):
        _execute(context, _paired_arguments("observed-eof", terminal=True))


def test_offline_extractor_graph_does_not_create_a_later_solver_action(
        tmp_path):
    log, context = _environment(tmp_path)
    _call, result_event = _pair(log)
    log.append("extracted_commit", {
        "label": "F", "kind": "final", "text": "reconstructed graph",
        "deps": [],
    })

    result = _execute(
        context, _paired_arguments("observed-eof", terminal=True))

    assert result.outcome == OUTCOME_HOLDS
    assert result.observation["snapshot"]["resolved_source_through_seq"] \
        == result_event["seq"]
    assert result.observation["terminal"]["later_solver_action_count"] == 0


def test_observed_eof_source_append_changes_replay_identity(tmp_path):
    log, context = _environment(tmp_path)
    log.append("message", {"text": "selected"})
    arguments = {
        "scope": "observed-eof",
        "clauses": [{
            "id": "message", "event_type": "message",
            "payload_strings": [{
                "pointer": "/text", "predicate": "equals",
                "expected": "selected",
            }],
        }],
    }
    first = _execute(context, arguments)
    log.append("stop", {"reason": "source lifecycle"})
    second = _execute(context, arguments)

    assert first.observation["witness_events"] \
        == second.observation["witness_events"]
    assert first.observation["snapshot"]["source_event_count"] == 1
    assert second.observation["snapshot"]["source_event_count"] == 2
    assert first.observation != second.observation
    assert first.consumed != second.consumed


def test_prefix_does_not_read_or_parse_a_malformed_suffix(tmp_path):
    log, context = _environment(tmp_path)
    _call, result_event = _pair(log)
    path = tmp_path / "state" / "events.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"malformed":')

    result = _execute(context, _paired_arguments(
        "prefix", through_seq=result_event["seq"]))

    assert result.outcome == OUTCOME_HOLDS
    assert result.observation["snapshot"]["through_seq"] \
        == result_event["seq"]


@pytest.mark.parametrize("input_value", [
    "xconst r = await tools.exec_command({\"cmd\":\"printf 雪\"});\ntext(r.output);\n",
    "const r = await tools.exec_command({\"cmd\":\"printf 雪\"});\ntext(r.output);\ny",
    "const r = await tools.exec_command({\"cmd\":\"printf 雪\",\"cmd\":\"other\"});\ntext(r.output);\n",
])
def test_frozen_invocation_decoder_rejects_wrong_envelope_prefix_suffix_or_json(
    tmp_path, input_value,
):
    log, context = _environment(tmp_path)
    call, _result = _pair(log)
    call["payload"]["input"] = input_value
    # Rewrite the fixture before a reader has observed it.
    path = tmp_path / "state" / "events.jsonl"
    path.write_text(json.dumps(call, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(AdapterNotApplicable):
        _execute(context, {
            "scope": "prefix", "through_seq": 1,
            "clauses": [{
                "id": "call", "event_type": "tool_call", "tool": "exec",
                "record_program": {
                    "decoder": "record.program@1",
                    "profile": "codex-exec-wrapper-v1",
                    "strings": [{
                        "pointer": "/command", "predicate": "equals",
                        "expected": "printf 雪",
                    }],
                },
            }],
        })


@pytest.mark.parametrize("mutation", [
    {"agent": False},
    {"session": 1},
    {"tool": 7},
    {"tool_use_id": 1},
])
def test_outer_selectors_and_pair_ids_never_coerce_types(tmp_path, mutation):
    log, context = _environment(tmp_path)
    call, result_event = _pair(log)
    call["payload"].update(mutation)
    path = tmp_path / "state" / "events.jsonl"
    records = [call, result_event]
    path.write_text("".join(
        json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8")

    with pytest.raises(AdapterNotApplicable):
        _execute(context, _paired_arguments("observed-eof"))


def test_duplicate_pair_identifier_is_ambiguous_and_falls_back(tmp_path):
    log, context = _environment(tmp_path)
    _call, result_event = _pair(log)
    log.append("tool_result", {
        "tool": "exec", "tool_use_id": "call-1", "session": "s1",
        "agent": None, "excerpt": "duplicate 雪",
    })

    with pytest.raises(AdapterNotApplicable, match="unambiguous"):
        _execute(context, _paired_arguments("observed-eof"))


def test_pair_uniqueness_covers_events_before_after_seq(tmp_path):
    log, context = _environment(tmp_path)
    earlier = log.append("tool_result", {
        "tool": "exec", "tool_use_id": "call-1", "session": "s1",
        "agent": None, "excerpt": "earlier duplicate",
    })
    _call, result_event = _pair(log)
    arguments = _paired_arguments("observed-eof")
    arguments["after_seq"] = earlier["seq"]

    with pytest.raises(AdapterNotApplicable, match="unambiguous"):
        _execute(context, arguments)


def test_paired_relation_is_forbidden_for_oracle_chosen_prefix(tmp_path):
    log, context = _environment(tmp_path)
    _call, result_event = _pair(log)
    log.append("tool_result", {
        "tool": "exec", "tool_use_id": "call-1", "session": "s1",
        "agent": None, "excerpt": "later duplicate",
    })
    arguments = _paired_arguments("prefix", through_seq=result_event["seq"])
    arguments["relations"].append({
        "kind": "paired", "call": "invoke", "result": "result",
    })

    with pytest.raises(AdapterNotApplicable, match="complete observed-eof"):
        _execute(context, arguments)


def test_clause_declaration_order_is_semantic(tmp_path):
    log, context = _environment(tmp_path)
    result_event = log.append("tool_result", {
        "tool": "exec", "tool_use_id": "call-1", "session": "s1",
        "agent": None, "excerpt": "完成 雪",
    })
    call = log.append("tool_call", {
        "tool": "exec", "tool_use_id": "call-1", "session": "s1",
        "agent": None, "input": _wrapper("printf 雪"), "label": "café 雪",
    })

    with pytest.raises(AdapterNotApplicable):
        _execute(context, _paired_arguments(
            "prefix", through_seq=call["seq"]))
    assert result_event["seq"] < call["seq"]


def test_terminal_rejects_any_later_solver_action_at_observed_eof(tmp_path):
    log, context = _environment(tmp_path)
    _call, result_event = _pair(log)
    later = log.append("tool_call", {
        "tool": "different", "session": "s1", "agent": None,
        "tool_use_id": "late", "input": "opaque",
    })

    with pytest.raises(AdapterNotApplicable):
        _execute(context, _paired_arguments(
            "observed-eof", terminal=True))
    assert result_event["seq"] < later["seq"]


def test_terminal_policy_cannot_be_narrowed_by_session_or_agent(tmp_path):
    log, context = _environment(tmp_path)
    _call, result_event = _pair(log)
    later = log.append("tool_call", {
        "tool": "different", "session": "another", "agent": None,
        "tool_use_id": "late", "input": "opaque",
    })

    with pytest.raises(AdapterNotApplicable):
        _execute(context, _paired_arguments("observed-eof", terminal=True))
    assert result_event["seq"] < later["seq"]


def test_terminal_policy_is_forbidden_for_prefix_even_at_its_watermark(tmp_path):
    log, context = _environment(tmp_path)
    _call, result_event = _pair(log)

    with pytest.raises(AdapterNotApplicable, match="only for observed-eof"):
        _execute(context, _paired_arguments(
            "prefix", through_seq=result_event["seq"], terminal=True))


def test_terminal_policy_does_not_count_results_or_stop_as_solver_actions(tmp_path):
    log, context = _environment(tmp_path)
    _call, result_event = _pair(log)
    log.append("tool_result", {
        "tool": "other", "tool_use_id": "other-id", "session": "s2",
        "agent": "worker", "excerpt": "observation only",
    })
    log.append("stop", {"reason": "lifecycle marker"})

    result = _execute(context, _paired_arguments("observed-eof", terminal=True))

    assert result.outcome == OUTCOME_HOLDS
    assert result.observation["terminal"]["source_seq"] == result_event["seq"]


def test_contains_rejects_an_empty_expected_string(tmp_path):
    log, context = _environment(tmp_path)
    event = log.append("message", {"text": "anything"})

    with pytest.raises(AdapterNotApplicable, match="non-empty for contains"):
        _execute(context, {
            "scope": "prefix", "through_seq": event["seq"],
            "clauses": [{
                "id": "message", "event_type": "message",
                "payload_strings": [{
                    "pointer": "/text", "predicate": "contains", "expected": "",
                }],
            }],
        })


def test_pair_pointer_is_fixed_to_tool_use_id(tmp_path):
    log, context = _environment(tmp_path)
    _pair(log)
    arguments = _paired_arguments("observed-eof")
    arguments["relations"][-1]["id_pointer"] = "/session"

    with pytest.raises(AdapterNotApplicable, match="invalid relation shape"):
        _execute(context, arguments)


def test_verifier_events_cannot_be_selected_as_source_witnesses(tmp_path):
    log, context = _environment(tmp_path)
    log.append("judger_verdict", {"verdict": "verified"})

    with pytest.raises(AdapterNotApplicable, match="source-side"):
        _execute(context, {
            "scope": "observed-eof",
            "clauses": [{"id": "self", "event_type": "judger_verdict"}],
        })


def test_fixed_binding_and_record_basis_are_independent_contracts(tmp_path):
    log, context = _environment(tmp_path)
    event = log.append("message", {"text": "bounded source"})
    arguments = {
        "scope": "prefix", "through_seq": event["seq"],
        "clauses": [{
            "id": "message", "event_type": "message",
            "payload_strings": [{
                "pointer": "/text", "predicate": "equals",
                "expected": "bounded source",
            }],
        }],
    }
    adapter = LedgerSequenceAdapter()
    method = _method(arguments)
    method.pop("adapter_input_binding")
    invocation = AdapterInvocation.from_method(method)
    assert invocation is not None
    with pytest.raises(AdapterNotApplicable, match="fixed-v1"):
        adapter.validate(invocation, method)

    bound_method = _method(arguments)
    bound = AdapterInvocation.from_method(bound_method)
    assert bound is not None
    with pytest.raises(AdapterNotApplicable, match="non-record"):
        adapter.execute(
            bound,
            Obligation("neutral", "world claim", "world"),
            context,
        )


def test_rfc6901_array_indices_are_ascii_only(tmp_path):
    log, context = _environment(tmp_path)
    event = log.append("message", {"items": ["first"]})

    with pytest.raises(AdapterNotApplicable):
        _execute(context, {
            "scope": "prefix", "through_seq": event["seq"],
            "clauses": [{
                "id": "message", "event_type": "message",
                "payload_strings": [{
                    "pointer": "/items/٠", "predicate": "equals",
                    "expected": "first",
                }],
            }],
        })


def test_deterministic_selection_uses_earliest_ordered_tuple(tmp_path):
    log, context = _environment(tmp_path)
    events = [log.append("message", {"text": text})
              for text in ("α", "β", "γ")]
    arguments = {
        "scope": "prefix", "through_seq": events[-1]["seq"],
        "clauses": [{
            "id": "first", "event_type": "message",
            "payload_strings": [{
                "pointer": "/text", "predicate": "nonempty",
            }],
        }, {
            "id": "second", "event_type": "message",
            "payload_strings": [{
                "pointer": "/text", "predicate": "nonempty",
            }],
        }],
    }

    first = _execute(context, arguments)
    second = _execute(context, arguments)

    assert [item["seq"] for item in first.observation["witness_events"]] == [1, 2]
    assert first.observation["witness_events"] == second.observation["witness_events"]
    assert first.observation["snapshot"]["projection_digest"] \
        == second.observation["snapshot"]["projection_digest"]


def test_absence_is_always_inapplicable_even_for_refutable_obligation(tmp_path):
    log, context = _environment(tmp_path)
    event = log.append("message", {"text": "different"})

    with pytest.raises(AdapterNotApplicable, match="no matching"):
        _execute(context, {
            "scope": "prefix", "through_seq": event["seq"],
            "clauses": [{
                "id": "wanted", "event_type": "message",
                "payload_strings": [{
                    "pointer": "/text", "predicate": "equals",
                    "expected": "not present",
                }],
            }],
        })


@pytest.mark.parametrize("refutable", [False, True])
def test_complete_audit_no_match_is_failed_never_a_verdict(
    tmp_path, refutable,
):
    log, context = _environment(tmp_path)
    event = log.append("message", {"text": "different"})
    arguments = {
        "scope": "observed-eof",
        "clauses": [{
            "id": "wanted", "event_type": "message",
            "payload_strings": [{
                "pointer": "/text", "predicate": "equals",
                "expected": "not present",
            }],
        }],
    }

    result = _execute_complete_audit(
        context, arguments, refutable=refutable)

    assert result.outcome == OUTCOME_FAILED
    assert result.witness_kind == WITNESS_NONE
    assert result.observation == {
        "schema": "gv-ledger-sequence-complete-audit-v1",
        "matched": False,
        "audit_complete": True,
        "failure_kind": "clause-no-match",
        "query_digest": ledger_sequence.content_hash(arguments),
        "clause_order": ["wanted"],
        "missing_clauses": ["wanted"],
        "candidate_counts": [{"clause": "wanted", "count": 0}],
        "search_states": 0,
        "examined_string_bytes": len("different"),
        "snapshot": result.consumed[0],
    }
    assert result.consumed[0]["source_event_count"] == 1
    assert result.consumed[0]["resolved_source_through_seq"] == event["seq"]
    assert result.consumed[0]["complete_file_at_observation"] is True
    assert "absence" not in json.dumps(
        {"witness": result.witness, "observation": result.observation})


def test_complete_audit_exhausts_relations_before_failed(tmp_path):
    log, context = _environment(tmp_path)
    result_event = log.append("tool_result", {
        "tool": "exec", "tool_use_id": "call-1", "session": "s1",
        "agent": None, "excerpt": "完成 雪",
    })
    call = log.append("tool_call", {
        "tool": "exec", "tool_use_id": "call-1", "session": "s1",
        "agent": None, "input": _wrapper("printf 雪"), "label": "café 雪",
    })

    result = _execute_complete_audit(
        context, _paired_arguments("observed-eof"))

    assert result.outcome == OUTCOME_FAILED
    assert result.observation["failure_kind"] == "sequence-no-match"
    assert result.observation["candidate_counts"] == [
        {"clause": "invoke", "count": 1},
        {"clause": "result", "count": 1},
    ]
    assert result.observation["search_states"] > 0
    assert result_event["seq"] < call["seq"]


def test_complete_audit_does_not_hide_a_later_clause_resource_limit(
    tmp_path, monkeypatch,
):
    log, context = _environment(tmp_path)
    log.append("message", {"text": "one"})
    last = log.append("message", {"text": "two"})
    monkeypatch.setattr(ledger_sequence, "MAX_CANDIDATES_PER_CLAUSE", 1)

    with pytest.raises(AdapterNotApplicable, match="candidate cap"):
        _execute_complete_audit(context, {
            "scope": "prefix", "through_seq": last["seq"],
            "clauses": [{
                "id": "missing", "event_type": "commit",
            }, {
                "id": "too-many", "event_type": "message",
                "payload_strings": [{
                    "pointer": "/text", "predicate": "nonempty",
                }],
            }],
        })


def test_complete_audit_prefix_no_match_still_falls_back(tmp_path):
    log, context = _environment(tmp_path)
    event = log.append("message", {"text": "different"})

    with pytest.raises(AdapterNotApplicable, match="stable observed-eof"):
        _execute_complete_audit(context, {
            "scope": "prefix", "through_seq": event["seq"],
            "clauses": [{
                "id": "wanted", "event_type": "message",
                "payload_strings": [{
                    "pointer": "/text", "predicate": "equals",
                    "expected": "not present",
                }],
            }],
        })


def test_complete_audit_missing_clause_still_checks_global_pair_boundaries(
    tmp_path,
):
    log, context = _environment(tmp_path)
    _pair(log, call_id="x" * (ledger_sequence.MAX_IDENTIFIER_BYTES + 1))
    arguments = _paired_arguments("observed-eof")
    arguments["clauses"].append({
        "id": "missing", "event_type": "commit",
    })

    with pytest.raises(AdapterNotApplicable, match="pair identifier exceeds"):
        _execute_complete_audit(context, arguments)


def test_caps_fail_closed_without_truncating_candidates_or_clauses(
    tmp_path, monkeypatch,
):
    log, context = _environment(tmp_path)
    first = log.append("message", {"text": "one"})
    second = log.append("message", {"text": "two"})
    clause = {
        "id": "match", "event_type": "message",
        "payload_strings": [{"pointer": "/text", "predicate": "nonempty"}],
    }
    with pytest.raises(AdapterNotApplicable, match="1..8"):
        _execute(context, {
            "scope": "prefix", "through_seq": second["seq"],
            "clauses": [{**clause, "id": f"c{index}"} for index in range(9)],
        })

    monkeypatch.setattr(ledger_sequence, "MAX_CANDIDATES_PER_CLAUSE", 1)
    with pytest.raises(AdapterNotApplicable, match="candidate cap"):
        _execute(context, {
            "scope": "prefix", "through_seq": second["seq"],
            "clauses": [clause],
        })
    assert first["seq"] == 1


def test_receipt_cap_is_measured_in_utf8_bytes(tmp_path):
    log, context = _environment(tmp_path)
    event = log.append("message", {"text": "雪" * 1_800})

    with pytest.raises(AdapterNotApplicable, match="receipt exceeds"):
        _execute(context, {
            "scope": "prefix", "through_seq": event["seq"],
            "clauses": [{
                "id": "large", "event_type": "message",
                "payload_strings": [{
                    "pointer": "/text", "predicate": "nonempty",
                }],
            }],
        })


def test_module_contract_is_record_only_fixed_input_and_task_agnostic(tmp_path):
    adapter = LedgerSequenceAdapter()
    entry = adapter.catalog_entry()
    assert adapter.identity == "ledger.sequence@1"
    assert adapter.basis == BASIS_RECORD
    assert adapter.accepted_method_reuse is False
    assert adapter.fixed_inputs[0].public() == {
        "ref": "ledger:events", "provenance": "ledger", "role": "subject",
    }
    assert entry["direction"].startswith("holds-only")
    assert "required multi-event order" in entry["selection"]
    assert "one record.query exactly covers" in entry["selection"]
    assert "terminal-solver-action" in entry["capabilities"]
    assert "terminal-is-not-last-decoded-command" in entry["limitations"]
    terminal_example = next(
        example for example in entry["examples"]
        if "terminal solver action" in example["purpose"])
    assert "final recorded command" not in terminal_example["purpose"]
    assert "stronger than proving the last decoded command" \
        in terminal_example["warning"]
    assert "record.query last-decoded-command" in terminal_example["warning"]
    assert "record.query terminal-exact-command" in terminal_example[
        "warning"]
    assert set(entry["source_projection"]["event_types"]) == {
        "session_start", "tool_call", "tool_result", "message", "commit", "stop",
    }
    # Generality is structural: the implementation contains no task/file
    # selector or domain-specific subject vocabulary.
    text = Path(ledger_sequence.__file__).read_text(encoding="utf-8")
    assert "task_id" not in text
    assert "filename" not in text
    assert "subprocess" not in text
    assert "shell=True" not in text

    complete = LedgerSequenceCompleteAuditAdapter()
    complete_entry = complete.catalog_entry()
    assert complete.identity == "ledger.sequence@2"
    assert complete.complete_no_match is True
    assert ("complete valid observed-eof no-match => failed"
            in complete_entry["direction"])
    assert complete_entry["complete_audit"]["future_closure"] is False
    assert "never a refutation" in complete_entry["complete_audit"]["no_match"]


def test_default_registry_publishes_and_materializes_fixed_sequence_input():
    registry = default_adapter_registry()
    sequence_entries = [
        item for item in json.loads(registry.catalog())
        if item["name"] == "ledger.sequence"
    ]
    assert [item["version"] for item in sequence_entries] == [1, 2]
    entry = sequence_entries[0]
    assert entry["version"] == 1
    assert entry["fixed_input_binding"] == {
        "mode": "fixed-v1",
        "inputs": [{
            "ref": "ledger:events", "provenance": "ledger",
            "role": "subject",
        }],
    }
    method = _method({
        "scope": "prefix", "through_seq": 1,
        "clauses": [{"id": "one", "event_type": "message"}],
    })
    method["adapter_inputs"] = []
    bound = registry.bind_inputs(method)
    assert bound["adapter_inputs"] == entry["fixed_input_binding"]["inputs"]


def test_complete_audit_failed_routes_without_agent_fallback(tmp_path):
    log, context = _environment(tmp_path)
    log.append("message", {"text": "different"})
    state = Path(context.state_dir)
    artifacts = ArtifactStore(str(state / "artifacts"))
    route = AdapterRoutingDischarger(
        _NoFallback(), default_adapter_registry(), artifacts, log,
        str(state), live_world=False)
    arguments = {
        "scope": "observed-eof",
        "clauses": [{
            "id": "wanted", "event_type": "message",
            "payload_strings": [{
                "pointer": "/text", "predicate": "equals",
                "expected": "wanted",
            }],
        }],
    }
    method = _method(arguments, version=2)
    obligation = Obligation(
        "sequence-claim", "the declared relation is recorded", BASIS_RECORD)

    result = route.discharge(
        {"id": obligation.claim_id, "kind": "final"}, obligation, method)

    assert result.outcome == OUTCOME_FAILED
    evidence = artifacts.get(result.artifact)
    assert evidence["execution"]["outcome"] == OUTCOME_FAILED
    assert evidence["execution"]["observation"]["audit_complete"] is True
    adapter_events = [event for event in log.replay()
                      if event["type"] == "adapter_discharge"]
    assert len(adapter_events) == 1
    assert adapter_events[0]["payload"]["adapter"] == "ledger.sequence@2"
    assert not [event for event in log.replay()
                if event["type"] == "adapter_fallback"]


@pytest.mark.parametrize("live_world", [False, True])
def test_complete_audit_failure_keeps_existing_oracle_repair_and_judger_flow(
    tmp_path, live_world,
):
    log, context = _environment(tmp_path)
    log.append("message", {"text": "present"})
    state = Path(context.state_dir)
    artifacts = ArtifactStore(str(state / "artifacts"))
    route = AdapterRoutingDischarger(
        _NoFallback(), default_adapter_registry(), artifacts, log,
        str(state), live_world=live_world)
    missing = _method({
        "scope": "observed-eof",
        "clauses": [{
            "id": "message", "event_type": "message",
            "payload_strings": [{
                "pointer": "/text", "predicate": "equals",
                "expected": "missing",
            }],
        }],
    }, version=2)
    present = _method({
        "scope": "observed-eof",
        "clauses": [{
            "id": "message", "event_type": "message",
            "payload_strings": [{
                "pointer": "/text", "predicate": "equals",
                "expected": "present",
            }],
        }],
    })
    present["procedure"] = "inspect the repaired exact record query"
    calls = 0

    def oracle(_variables):
        nonlocal calls
        calls += 1
        return {"methods": [missing] if calls == 1 else [present]}

    backend = FixtureBackend({
        "oracles": oracle,
        "judger": {
            "sufficient": "yes", "faithful": "yes", "degradation": "n/a",
            "reason": "the fresh positive sequence receipt is complete",
        },
    })
    game = BindingGame(
        backend, artifacts, log, route, live_world=live_world)
    obligation = Obligation(
        "sequence-claim", "the present message is recorded", BASIS_RECORD)

    result = game.run(
        {"id": obligation.claim_id, "kind": "final", "text": "recorded"},
        obligation,
    )

    assert result.outcome == OUTCOME_HOLDS
    assert [call["role"] for call in backend.calls] == [
        "oracles", "oracles", "judger"]
    assert [event["payload"]["outcome"] for event in log.replay()
            if event["type"] == "adapter_discharge"] == [
                OUTCOME_FAILED, OUTCOME_HOLDS]
    assert len([event for event in log.replay()
                if event["type"] == "judger_verdict"]) == 1


def _typed_sequence_fixture(tmp_path, *, version=1):
    log, context = _environment(tmp_path)
    _pair(log)
    state = Path(context.state_dir)
    artifacts = ArtifactStore(str(state / "artifacts"))
    route = AdapterRoutingDischarger(
        _NoFallback(), default_adapter_registry(), artifacts, log,
        str(state), live_world=False)
    obligation = Obligation(
        "sequence-claim", "the declared completed exchange is recorded",
        BASIS_RECORD)
    method = _method(
        _paired_arguments("observed-eof", terminal=True), version=version)
    result = route.discharge(
        {"id": obligation.claim_id, "kind": "final"}, obligation, method)
    evidence = artifacts.get(result.artifact)
    invocation = AdapterInvocation.from_method(method)
    assert invocation is not None
    result_identity = {
        "outcome": result.outcome,
        "witness": result.witness,
        "witness_kind": result.witness_kind,
    }
    return log, artifacts, route, obligation, method, result, evidence, \
        invocation, result_identity


@pytest.mark.parametrize("version", [1, 2])
def test_complete_sequence_receipt_uses_fresh_typed_judger_packet(
    tmp_path, version,
):
    (log, artifacts, route, obligation, method, result, evidence,
     invocation, result_identity) = _typed_sequence_fixture(
         tmp_path, version=version)

    assert typed_sequence_observation_is_complete(
        invocation, evidence, result_identity)
    packet = build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result,
        "(none — decisive method)",
    )
    assert type(packet) is TypedAdapterJudgerPacket
    assert packet.revalidate() is True
    log.append("judger_verdict", {"verifier_only": True})
    assert packet.revalidate() is True
    log.append("message", {"text": "late source continuation"})
    assert packet.revalidate() is False


@pytest.mark.parametrize("live_world", [False, True])
def test_sequence_keeps_fresh_oracle_adapter_and_judger_architecture(
    tmp_path, live_world,
):
    log, context = _environment(tmp_path)
    _pair(log)
    state = Path(context.state_dir)
    artifacts = ArtifactStore(str(state / "artifacts"))
    route = AdapterRoutingDischarger(
        _NoFallback(), default_adapter_registry(), artifacts, log,
        str(state), live_world=live_world)
    method = _method(_paired_arguments("observed-eof", terminal=True))
    backend = FixtureBackend({
        "oracles": {"methods": [method]},
        "judger": {
            "sufficient": "yes", "faithful": "yes", "degradation": "n/a",
            "reason": "the fresh receipt covers the exact record claim",
        },
    })
    game = BindingGame(
        backend, artifacts, log, route, live_world=live_world)
    obligation = Obligation(
        "sequence-claim", "the declared completed exchange is recorded",
        BASIS_RECORD)

    result = game.run(
        {"id": obligation.claim_id, "kind": "final", "text": "recorded"},
        obligation,
    )

    assert result.outcome == OUTCOME_HOLDS
    assert [call["role"] for call in backend.calls] == ["oracles", "judger"]
    verifier_events = [event["type"] for event in log.replay()
                       if event["type"] not in {"tool_call", "tool_result"}]
    assert verifier_events == [
        "methods_enumerated", "adapter_discharge", "judger_verdict",
        "method_bound",
    ]


@pytest.mark.parametrize("mutation", [
    "predicate_actual", "event_digest", "pair_unique", "pair_scan_digest",
    "terminal_action_count", "snapshot_digest", "consumed_digest",
    "candidate_count", "source_protocol",
])
def test_typed_sequence_gate_rejects_semantic_receipt_mutation(
    tmp_path, mutation,
):
    (*_prefix, evidence, invocation,
     result_identity) = _typed_sequence_fixture(tmp_path)
    changed = copy.deepcopy(evidence)
    observation = changed["execution"]["observation"]
    if mutation == "predicate_actual":
        observation["witness_events"][0]["evaluation"][
            "record_program"]["string_predicates"][0]["actual"] = "different"
    elif mutation == "event_digest":
        observation["witness_events"][0]["event_digest"] = "0" * 64
    elif mutation == "pair_unique":
        observation["relations"][-1]["unique_in_complete_source_scan"] = False
    elif mutation == "pair_scan_digest":
        observation["relations"][-1]["scanned_source_events_digest"] = "0" * 64
    elif mutation == "terminal_action_count":
        observation["terminal"]["later_solver_action_count"] = 1
    elif mutation == "snapshot_digest":
        observation["snapshot"]["source_events_digest"] = "0" * 64
    elif mutation == "consumed_digest":
        changed["execution"]["consumed"][0]["source_events_digest"] = "0" * 64
    elif mutation == "candidate_count":
        observation["candidate_counts"][0]["count"] = True
    else:
        observation["snapshot"]["source_event_types"] = ["message"]

    assert not typed_sequence_observation_is_complete(
        invocation, changed, result_identity)
