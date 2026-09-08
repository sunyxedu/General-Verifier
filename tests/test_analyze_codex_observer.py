"""No-model tests of common-observer lifecycle and token accounting."""
from copy import deepcopy
import hashlib
import json

import pytest

from experiments.terminal_bench.analyze_codex_observer import (
    SCHEMA, analyze_file, analyze_rows,
)


PLAN = {
    "context_by_cwd": {"/sealed/state": {"arm": "control", "trial": "opaque"}},
    "model": "gpt-5.6-luna", "effort": "medium", "service_tier": "standard",
    "session_summary": {
        "schema": "gv-codex-attempt-observer-summary-v1", "observer_id": "observer",
        "seam": "legacy-subprocess-run", "logical_calls_started": 1,
        "logical_calls_finished": 1, "attempts_started": 1, "attempts_finished": 1,
        "write_failures": 0, "active_calls": 0, "restoration_failed": False,
        "closed": True,
        "ledger_sha256": "c" * 64, "ledger_bytes": 123,
    },
}


def ledger(*, raised=False, seam="legacy-subprocess-run"):
    common = {
        "schema": SCHEMA, "observer_id": "observer", "seam": seam,
        "call_id": "call", "role": "judger", "model": PLAN["model"],
        "effort": PLAN["effort"], "service_tier": PLAN["service_tier"],
        "state_dir": "/sealed/state", "attribution": "attributed",
        "context": deepcopy(PLAN["context_by_cwd"]["/sealed/state"]),
    }
    cmd = {
        "attempt_id": "attempt", "attempt_index": 1,
        "command_sha256": "a" * 64, "cwd": "/sealed/state",
        "command_model": PLAN["model"], "command_effort": PLAN["effort"],
        "command_service_tier": PLAN["service_tier"],
    }
    return [
        {**common, "record_type": "codex_logical_call_started",
         "started_at_unix_ns": 10, "started_at_monotonic_ns": 10},
        {**common, **cmd, "record_type": "codex_cli_attempt_started",
         "started_at_unix_ns": 11, "started_at_monotonic_ns": 11},
        {**common, **cmd, "record_type": "codex_cli_attempt_finished",
         "started_at_unix_ns": 11, "finished_at_unix_ns": 15, "duration_ns": 4,
         "terminal_status": "returned", "returncode": 0, "process_started": True,
         "stdout_sha256": "b" * 64, "stdout_bytes": 123,
         "usage_complete": True, "usage_status": "complete", "thread_id": "thread",
         "input_tokens": 100, "cached_input_tokens": 80, "uncached_input_tokens": 20,
         "cache_write_input_tokens": 0, "output_tokens": 10, "reasoning_tokens": 4},
        {**common, "record_type": "codex_logical_call_finished",
         "started_at_unix_ns": 10, "finished_at_unix_ns": 16, "duration_ns": 6,
         "terminal_status": "raised" if raised else "returned", "attempt_count": 1,
         "error_kind": "RoleError" if raised else None},
    ]


@pytest.mark.parametrize("seam", ["legacy-subprocess-run", "run-codex-process"])
@pytest.mark.parametrize("raised", [False, True])
def test_both_seams_keep_postprocessing_failure_costs(seam, raised):
    plan = {**PLAN, "session_summary": {**PLAN["session_summary"], "seam": seam}}
    result = analyze_rows(ledger(seam=seam, raised=raised), **plan)
    assert result["accounting_complete"] is True
    assert result["all_logical_calls_returned"] is not raised
    assert result["logical_calls_raised"] == int(raised)
    assert result["tokens"]["output_tokens"]["known_sum"] == 10
    assert result["tokens"]["reasoning_tokens"]["known_sum"] == 4
    assert result["quality_acceptance"] is None
    assert result["full_e2e_wall_seconds"] is None


@pytest.mark.parametrize("removed", range(4))
def test_each_missing_lifecycle_event_is_incomplete(removed):
    rows = ledger()
    del rows[removed]
    assert analyze_rows(rows, **PLAN)["accounting_complete"] is False


@pytest.mark.parametrize("index", range(4))
def test_duplicate_rows_fail_instead_of_silently_double_counting(index):
    rows = ledger()
    rows.insert(index, deepcopy(rows[index]))
    report = analyze_rows(rows, **PLAN)
    assert "duplicate_lifecycle_event" in report["findings"]
    assert report["tokens"]["output_tokens"]["known_sum"] == 10


@pytest.mark.parametrize("field,bad", [
    ("usage_complete", 1), ("usage_status", "partial"),
    ("input_tokens", True), ("output_tokens", -1),
    ("cached_input_tokens", 101), ("uncached_input_tokens", 100),
    ("reasoning_tokens", 11), ("thread_id", None),
])
def test_invalid_core_counts_are_not_trusted(field, bad):
    rows = ledger()
    rows[2][field] = bad
    result = analyze_rows(rows, **PLAN)
    assert result["accounting_complete"] is False
    assert result["unknown_usage_attempts"] == 1
    assert result["tokens"]["output_tokens"] == {"known_sum": 0, "unknown_attempts": 1}


def test_missing_cache_writes_preserves_other_buckets_but_not_full_cost_claim():
    rows = ledger()
    rows[2]["cache_write_input_tokens"] = None
    result = analyze_rows(rows, **PLAN)
    assert result["accounting_complete"] is False
    assert result["tokens"]["output_tokens"]["known_sum"] == 10
    assert result["tokens"]["cache_write_input_tokens"]["unknown_attempts"] == 1


@pytest.mark.parametrize("field", ["model", "effort", "service_tier"])
def test_actual_command_profile_cannot_be_replaced_by_declared_profile(field):
    rows = ledger()
    for row in rows[1:3]:
        row["command_" + field] = None
    result = analyze_rows(rows, **PLAN)
    assert "actual_cli_profile_mismatch" in result["findings"]


def test_unattributed_and_changed_operator_context_rejected():
    rows = ledger()
    rows[1]["context"] = {"arm": "treatment", "trial": "opaque"}
    rows[2]["attribution"] = "unattributed"
    result = analyze_rows(rows, **PLAN)
    assert "context_attribution_mismatch" in result["findings"]


def test_multiple_attempts_all_charged_including_nonzero_return():
    rows = ledger()
    first_start, first_end = deepcopy(rows[1:3])
    first_end["returncode"] = 1
    rows[1]["attempt_id"] = rows[2]["attempt_id"] = "attempt-2"
    rows[1]["attempt_index"] = rows[2]["attempt_index"] = 2
    rows[2]["thread_id"] = "thread-2"
    rows[3]["attempt_count"] = 2
    rows[1:1] = [first_start, first_end]
    plan = {**PLAN, "session_summary": {**PLAN["session_summary"],
            "attempts_started": 2, "attempts_finished": 2}}
    result = analyze_rows(rows, **plan)
    assert result["accounting_complete"] is True
    assert result["attempt_count"] == 2
    assert result["tokens"]["output_tokens"]["known_sum"] == 20


@pytest.mark.parametrize("status", ["timeout", "launch_error", "interrupted"])
def test_interrupted_attempt_cannot_claim_complete_usage(status):
    rows = ledger(raised=True)
    rows[2]["terminal_status"] = status
    rows[2]["returncode"] = None
    assert "interrupted_usage_cannot_be_complete" in analyze_rows(rows, **PLAN)["findings"]
    assert analyze_rows(rows, **PLAN)["tokens"]["output_tokens"]["known_sum"] == 0


def test_wrong_execution_state_is_not_self_authorized_by_ledger():
    rows = ledger()
    for row in rows[1:3]:
        row["cwd"] = "/unrelated/state"
    assert "execution_cwd_not_operator_bound" in analyze_rows(rows, **PLAN)["findings"]
    allowed = {"/sealed/state": ("/unrelated/state",)}
    assert analyze_rows(rows, execution_cwds_by_state=allowed, **PLAN)["accounting_complete"] is True


@pytest.mark.parametrize("roots", [
    1, None, "/sealed/state", {"/sealed/state": True}, [],
    ["relative"], [False], ["/sealed/state", "/sealed/state"],
    ["/sealed/state/../other"],
])
def test_bad_operator_execution_roots_fail_before_analysis(roots):
    with pytest.raises(ValueError, match="execution cwd"):
        analyze_rows(ledger(), execution_cwds_by_state={"/sealed/state": roots}, **PLAN)


def test_execution_root_mapping_cannot_add_unknown_candidate():
    with pytest.raises(ValueError, match="every known state"):
        analyze_rows(ledger(), execution_cwds_by_state={
            "/sealed/state": ("/sealed/state",), "/unknown": ("/unknown",)}, **PLAN)


@pytest.mark.parametrize("status,started", [
    ("returned", False), ("returned", 1), ("timeout", False),
    ("launch_error", True), ("interrupted", "unknown"),
])
def test_process_started_flag_must_match_terminal_status(status, started):
    rows = ledger()
    rows[2]["terminal_status"] = status
    rows[2]["process_started"] = started
    assert analyze_rows(rows, **PLAN)["accounting_complete"] is False


def test_successful_zero_attempt_call_cannot_hide_beside_real_work():
    rows = ledger()
    second = [deepcopy(rows[0]), deepcopy(rows[3])]
    for row in second:
        row["call_id"] = "second"
    second[1]["attempt_count"] = 0
    summary = {**PLAN["session_summary"], "logical_calls_started": 2,
               "logical_calls_finished": 2}
    result = analyze_rows(rows + second, **{**PLAN, "session_summary": summary})
    assert "returned_logical_call_without_attempt" in result["findings"]


@pytest.mark.parametrize("bad", [None, [], "record", {"schema": SCHEMA, "record_type": []}])
def test_malformed_rows_fail_without_crashing_or_echoing_content(bad):
    assert analyze_rows([bad], **PLAN)["accounting_complete"] is False


def test_no_work_is_not_free_success():
    result = analyze_rows([], **PLAN)
    assert result["accounting_complete"] is False
    assert result["all_logical_calls_returned"] is False


def test_missing_session_summary_cannot_hide_a_fully_lost_call():
    plan = {**PLAN, "session_summary": None}
    result = analyze_rows(ledger(), **plan)
    assert result["accounting_complete"] is False
    assert "missing_independent_session_summary" in result["findings"]


@pytest.mark.parametrize("field,bad", [
    ("logical_calls_started", 2), ("logical_calls_finished", 2),
    ("attempts_started", 2), ("attempts_finished", 2),
    ("write_failures", 4), ("active_calls", 1), ("closed", False),
    ("restoration_failed", True), ("write_failures", False),
])
def test_independent_session_rejects_complete_looking_partial_ledger(field, bad):
    summary = {**PLAN["session_summary"], field: bad}
    result = analyze_rows(ledger(), **{**PLAN, "session_summary": summary})
    assert result["accounting_complete"] is False
    assert result["observer_session_complete"] is False


@pytest.mark.parametrize("index,field", [
    (0, "seam"), (2, "terminal_status"), (3, "terminal_status"),
])
def test_unhashable_enum_values_are_reported(index, field):
    rows = ledger()
    rows[index][field] = []
    assert analyze_rows(rows, **PLAN)["accounting_complete"] is False


def test_replayed_thread_usage_is_not_counted_as_independent_fresh_work():
    rows = ledger()
    second = deepcopy(rows)
    for row in second:
        row["call_id"] = "call-2"
        if "attempt_id" in row:
            row["attempt_id"] = "attempt-2"
    result = analyze_rows(rows + second, **PLAN)
    assert "thread_id_reused_across_attempts" in result["findings"]


def test_overlapping_attempts_cannot_hide_a_misbound_call():
    rows = ledger()
    start, end = deepcopy(rows[1:3])
    start["attempt_id"] = end["attempt_id"] = "attempt-2"
    start["attempt_index"] = end["attempt_index"] = 2
    end["thread_id"] = "thread-2"
    rows[3]["attempt_count"] = 2
    result = analyze_rows([rows[0], rows[1], start, rows[2], end, rows[3]], **PLAN)
    assert "overlapping_attempts_in_one_logical_call" in result["findings"]


def test_file_reader_retains_identity_and_does_not_mutate(tmp_path):
    path = tmp_path / "observer.jsonl"
    data = "".join(json.dumps(row) + "\n" for row in ledger()).encode()
    path.write_bytes(data)
    plan = {**PLAN, "session_summary": {**PLAN["session_summary"],
            "ledger_sha256": hashlib.sha256(data).hexdigest(), "ledger_bytes": len(data)}}
    result = analyze_file(path, **plan)
    assert result["accounting_complete"] is True
    assert result["ledger_sha256"] == hashlib.sha256(data).hexdigest()
    assert path.read_bytes() == data


def test_file_reader_rejects_changed_content_after_session_closed(tmp_path):
    path = tmp_path / "observer.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in ledger()))
    result = analyze_file(path, **PLAN)
    assert result["accounting_complete"] is False
    assert "session_ledger_identity_mismatch" in result["findings"]


@pytest.mark.parametrize("raw", [b'{}', b'\n', b'{"a":1,"a":2}\n', b'{"a":NaN}\n'])
def test_file_reader_rejects_partial_or_non_strict_json(tmp_path, raw):
    path = tmp_path / "observer.jsonl"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        analyze_file(path, **PLAN)
