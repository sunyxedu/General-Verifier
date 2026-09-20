"""Pure policy-validity fixtures: no Docker, model, evaluator or task payloads."""
import json
import os
from pathlib import Path

import pytest

from experiments.cybergym.policy import (
    ITEM_EVENTS, MAX_SOLVER_EVENT_BYTES, SearchPolicyEvidenceError,
    SearchPolicyGuard, SolverSearchObserver, validate_policy_response, validate_policy_snapshot,
)


@pytest.mark.parametrize("event_type", sorted(ITEM_EVENTS))
@pytest.mark.parametrize("source", ["solver", "verifier"])
def test_structural_search_event_latches_without_query_or_result(source, event_type):
    guard = SearchPolicyGuard()
    secret = "fixture-query-and-credential-never-retained"
    if source == "solver":
        guard.observe_solver_event({"type": event_type, "item": {
            "type": "web_search", "query": secret, "output": secret}})
    else:
        guard.observe_backend_phase({"phase": "cli_event", "event_type": event_type,
                                     "item_kind": "web_search", "query": secret})
    state = guard.snapshot()
    assert validate_policy_snapshot(state) is True
    assert state["violation_source"] == source and state["event_type"] == event_type
    assert secret not in json.dumps(state)
    guard.observe_solver_event({"type": "item.completed", "item": {"type": "web_search"}})
    assert guard.snapshot() == state, "first event is irreversible and stable"
    state["violation"] = False
    assert guard.snapshot()["violation"] is True, "no mutable snapshot alias"


@pytest.mark.parametrize("event", [None, [], "web_search", {},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "web_search"}},
    {"type": "item.completed", "item": {"type": "command_execution", "command": "web_search"}},
    {"type": "item.completed", "item": {"type": "mcp_tool_call", "tool": "web_search"}},
    {"type": "item.completed", "item": "web_search"},
    {"type": "turn.completed", "item": {"type": "web_search"}},
    {"type": [], "item": {"type": "web_search"}}])
def test_prose_and_nonsearch_cli_items_do_not_latch(event):
    guard = SearchPolicyGuard()
    guard.observe_solver_event(event)
    assert validate_policy_snapshot(guard.snapshot()) is False


@pytest.mark.parametrize("event", [
    {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "WebSearch", "input": {"query": "secret"}}]}},
    {"type": "assistant", "message": {"content": [
        {"type": "server_tool_use", "name": "WebFetch"}]}},
    {"type": "result", "usage": {"server_tool_use": {"web_search_requests": 1}}},
])
def test_claude_structural_search_events_latch(event):
    guard = SearchPolicyGuard()
    guard.observe_solver_event(event)
    assert validate_policy_snapshot(guard.snapshot()) is True
    assert "secret" not in json.dumps(guard.snapshot())


def test_claude_nonsearch_event_does_not_latch():
    guard = SearchPolicyGuard()
    guard.observe_solver_event({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": "echo WebSearch"}}]}})
    assert validate_policy_snapshot(guard.snapshot()) is False


@pytest.mark.parametrize("record", [None, [], {},
    {"phase": "exec_heartbeat", "event_type": "item.completed", "item_kind": "web_search"},
    {"phase": "cli_event", "event_type": "turn.completed", "item_kind": "web_search"},
    {"phase": "cli_event", "event_type": [], "item_kind": "web_search"},
    {"phase": "cli_event", "event_type": "item.completed", "item_kind": "agent_message", "text": "web_search"}])
def test_only_native_backend_item_metadata_is_evidence(record):
    guard = SearchPolicyGuard()
    guard.observe_backend_phase(record)
    assert validate_policy_snapshot(guard.snapshot()) is False


@pytest.mark.parametrize("mutation", [None, {}, {"violation": 0}, {"violation": 1},
    {"schema": "wrong"}, {"web_search": "cached"}, {"event_type": "item.completed"},
    {"violation_source": "solver"}, {"extra": "value"},
    {"sealed": 1}, {"evidence_complete": 1},
    {"violation": True}, {"violation": True, "violation_source": "verifier", "event_type": "turn.completed"},
    {"violation": True, "violation_source": [], "event_type": "item.completed"}])
def test_unknown_or_contradictory_policy_snapshot_fails_closed(mutation):
    value = mutation if mutation in (None, {}) else {**SearchPolicyGuard().snapshot(), **mutation}
    with pytest.raises(SearchPolicyEvidenceError, match="snapshot_invalid"):
        validate_policy_snapshot(value)


def test_live_partial_line_is_revisited_and_continuations_are_not_skipped(tmp_path):
    path = tmp_path / "solver.jsonl"
    path.write_bytes(b'{"type":"thread.started","thread_id":"same-thread"}\n')
    guard = SearchPolicyGuard()
    reader = SolverSearchObserver(guard)
    reader.consume(path)
    first_offset = reader.offset
    with path.open("ab") as stream:
        stream.write(b'{"type":"item.started","item":')
    reader.consume(path)
    assert reader.offset == first_offset and guard.snapshot()["violation"] is False
    with path.open("ab") as stream:
        stream.write(b'{"type":"web_search","query":"discarded"}}\n')
    reader.consume(path)
    assert reader.offset == path.stat().st_size and guard.snapshot()["violation"] is True
    reader.consume(path)
    assert guard.snapshot()["event_type"] == "item.started"


def test_terminal_unterminated_line_is_checked(tmp_path):
    path = tmp_path / "solver.jsonl"
    path.write_bytes(b'{"type":"item.completed","item":{"type":"web_search"}}')
    guard = SearchPolicyGuard()
    reader = SolverSearchObserver(guard)
    reader.consume(path)
    assert reader.offset == 0
    reader.consume(path, final=True)
    assert guard.snapshot()["violation"] is True


@pytest.mark.parametrize("raw", [b'not-json\n', b'[]\n', b'\xff\n',
    b'{"type":"item.completed","type":"ignored"}\n', b'{"n":NaN}\n'])
def test_malformed_json_never_becomes_clean_policy_evidence(tmp_path, raw):
    path = tmp_path / "solver.jsonl"
    path.write_bytes(raw)
    with pytest.raises(SearchPolicyEvidenceError, match="event_invalid") as error:
        SolverSearchObserver(SearchPolicyGuard()).consume(path)
    assert raw.decode(errors="replace").strip() not in str(error.value)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "directory", "missing"])
def test_unsafe_log_cannot_be_read_or_block(tmp_path, kind):
    path = tmp_path / "solver.jsonl"
    target = tmp_path / "preserved"
    target.write_bytes(b'{"type":"turn.started"}\n')
    if kind == "symlink":
        path.symlink_to(target)
    elif kind == "hardlink":
        os.link(target, path)
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    with pytest.raises(SearchPolicyEvidenceError):
        SolverSearchObserver(SearchPolicyGuard()).consume(path)
    assert target.read_bytes() == b'{"type":"turn.started"}\n'


@pytest.mark.parametrize("change", ["truncate", "replace"])
def test_log_identity_or_cursor_drift_cannot_hide_earlier_events(tmp_path, change):
    path = tmp_path / "solver.jsonl"
    initial = b'{"type":"turn.started"}\n'
    path.write_bytes(initial)
    reader = SolverSearchObserver(SearchPolicyGuard())
    reader.consume(path)
    if change == "truncate":
        path.write_bytes(b'')
    else:
        path.rename(tmp_path / "preserved")
        path.write_bytes(initial)
    with pytest.raises(SearchPolicyEvidenceError, match="log_unsafe"):
        reader.consume(path)


def test_event_limit_is_a_parser_bound_not_a_task_deadline(tmp_path):
    path = tmp_path / "solver.jsonl"
    path.write_bytes(b'x' * (MAX_SOLVER_EVENT_BYTES + 1))
    with pytest.raises(SearchPolicyEvidenceError, match="event_too_large"):
        SolverSearchObserver(SearchPolicyGuard()).consume(path)


@pytest.mark.parametrize("mutation", [None, {}, {"instance_id": "invalid"},
    {"instance_id": []}, {"instance_id": "a" * 33}, {"schema": "wrong"},
    {"extra": "value"}, {"policy": None}])
def test_policy_response_requires_strict_process_generation_and_snapshot(mutation):
    response = {"schema": "gv-search-policy-response-v1", "instance_id": "a" * 32,
                "policy": SearchPolicyGuard().snapshot()}
    value = mutation if mutation in (None, {}) else {**response, **mutation}
    with pytest.raises(SearchPolicyEvidenceError):
        validate_policy_response(value)


def test_policy_process_generation_cannot_change_after_handshake():
    response = {"schema": "gv-search-policy-response-v1", "instance_id": "a" * 32,
                "policy": SearchPolicyGuard().snapshot()}
    assert validate_policy_response(response, expected_instance_id="a" * 32) == ("a" * 32, False)
    with pytest.raises(SearchPolicyEvidenceError, match="instance_changed"):
        validate_policy_response({**response, "instance_id": "b" * 32}, expected_instance_id="a" * 32)


@pytest.mark.parametrize("phase", ["exec_heartbeat", "pipe_drain_heartbeat", "exec_returned"])
@pytest.mark.parametrize("gap", [{"unparsed_stdout_lines": 1}, {"reader_failed": True}])
def test_native_verifier_parser_gap_cannot_become_clean_policy(phase, gap):
    guard = SearchPolicyGuard()
    guard.observe_backend_phase({"phase": phase, **gap})
    assert guard.snapshot()["violation"] is False
    assert guard.snapshot()["evidence_complete"] is False
    with pytest.raises(SearchPolicyEvidenceError, match="evidence_incomplete"):
        validate_policy_snapshot(guard.snapshot())


def test_active_call_prevents_seal_and_sealing_prevents_future_model_admission():
    guard = SearchPolicyGuard()
    guard.begin_execution()
    assert guard.try_seal() is False and guard.snapshot()["sealed"] is False
    guard.end_execution()
    assert guard.try_seal() is True and guard.snapshot()["sealed"] is True
    assert guard.try_seal() is True, "sealing is idempotent"
    with pytest.raises(SearchPolicyEvidenceError, match="after_policy_seal"):
        guard.begin_execution()


def test_actual_search_remains_violation_even_when_other_evidence_is_incomplete():
    guard = SearchPolicyGuard()
    guard.observe_solver_event({"type": "item.started", "item": {"type": "web_search"}})
    guard.observe_backend_phase({"phase": "exec_returned", "reader_failed": True})
    assert validate_policy_snapshot(guard.snapshot()) is True
