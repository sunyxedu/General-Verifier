"""Tracking fixtures are diagnostics only: no API, Docker or evaluator calls."""
import json
import os
from pathlib import Path

import pytest

import experiments.cybergym.tracking as tracking
from experiments.cybergym.tracking import (
    interval_union, main, record_snapshot, snapshot, summarize_calls,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def attempt(root, **changes):
    path = root / "attempts" / "attempt-1"
    status = {"task_id": "arvo:3438", "attempt_id": "attempt-1", "stage": "solver_running",
        "started_unix": 10, "stage_started_unix": 20, "updated_unix": 30,
        "completed_unix": None, "outcome": None, "official_pass": None, "reason": "",
        "solver_container": "solver-attempt-1", "sidecar_container": "sidecar-attempt-1",
        "phase_intervals": [{"stage": "setup", "started_unix": 10, "completed_unix": 20},
            {"stage": "solver_running", "started_unix": 20, "completed_unix": None}]}
    status.update(changes)
    write_json(path / "status.json", status)
    return path


def test_interval_union_removes_parallel_overlap():
    assert interval_union([(0, 10), (5, 20), (30, 40), (40, 42)]) == 32


def test_role_call_work_counts_failure_and_separates_active_age():
    calls = [
        {"call_id": "a", "role": "discharge", "started_unix": 0, "completed_unix": 10,
         "status": "completed", "duration_seconds": 10},
        {"call_id": "b", "role": "discharge", "started_unix": 5, "completed_unix": 20,
         "status": "aborted", "duration_seconds": 15},
        {"call_id": "c", "role": "judger", "started_unix": 15, "status": "running"}]
    summary = summarize_calls(calls, 30)
    assert summary["call_work_seconds_sum"] == 25
    assert summary["wall_seconds_union"] == 20
    assert summary["by_role"]["discharge"]["failed_calls"] == 1
    assert summary["active_calls"][0]["age_seconds"] == 15


def test_missing_intervals_are_unknown_not_zero():
    summary = summarize_calls([{"role": "judger", "status": "failed", "duration_seconds": 12}], 30)
    assert summary["call_work_seconds_sum"] == 12
    assert summary["wall_seconds_union"] is None
    assert summary["by_role"]["judger"]["missing_interval_calls"] == 1


def test_snapshot_uses_role_events_no_prompt_or_stdout_leak(tmp_path):
    path = attempt(tmp_path)
    state = path / "private/state"
    rows = [{"type": "role_call_started", "ts": 21, "call_id": "a", "role": "critic",
             "model": "gpt-5.6-luna", "service_tier": "fast", "prompt": "PRIVATE_PROMPT"},
            {"type": "role_call_finished", "ts": 26, "call_id": "a", "role": "critic",
             "status": "completed", "duration_seconds": 5},
            {"type": "role_call_started", "ts": 27, "call_id": "b", "role": "judger",
             "model": "gpt-5.6-luna", "service_tier": "fast"}]
    write_jsonl(state / "role_calls.jsonl", rows)
    write_jsonl(state / "events.jsonl", [
        {"type": "cyber_candidate_registered", "seq": 1, "ts": 21, "payload": {"candidate_id": "x"}},
        {"type": "tool_call", "seq": 2, "ts": 22,
         "payload": {"tool": "cyber_evaluator", "tool_use_id": "e1"}},
        {"type": "tool_result", "seq": 3, "ts": 24,
         "payload": {"tool": "cyber_evaluator", "tool_use_id": "e1", "output": {
             "elapsed_seconds": 2, "exit_code": 77, "stdout": "PRIVATE_STDOUT", "timed_out": False}}},
        {"type": "cyber_observation_recorded", "seq": 4, "ts": 24, "payload": {}}])
    value = snapshot(tmp_path, now=30)
    item = value["attempts"][0]
    assert item["verifier_timings"]["source"] == "role_calls"
    assert item["verifier_timings"]["call_work_seconds_sum"] == 5
    assert item["verifier_timings"]["active_calls"][0]["age_seconds"] == 3
    assert item["evaluator_timings"]["call_work_seconds_sum"] == 2
    assert item["current_activity"] == "verifier"
    assert item["elapsed_seconds"] == 20
    assert item["stage_age_seconds"] == 10
    assert item["progress"]["recorded_observations_in_window"] == 1
    assert item["runner_phase_timings"]["by_role"]["setup"]["call_work_seconds_sum"] == 10
    assert "PRIVATE_PROMPT" not in json.dumps(value)
    assert "PRIVATE_STDOUT" not in json.dumps(value)
    assert value["official_passed_attempts"] == 0
    assert "gate" not in value


def test_usage_ledger_fallback_and_deduplication(tmp_path):
    state = attempt(tmp_path) / "private/state"
    row = {"call_id": "x", "role": "oracles", "model": "gpt-5.6-luna", "service_tier": "fast",
        "started_at_unix_ns": 20_000_000_000, "finished_at_unix_ns": 25_000_000_000,
        "duration_seconds": 5, "returncode": 0, "input_tokens": 100,
        "cached_input_tokens": 20, "output_tokens": 10, "reasoning_tokens": 7}
    write_jsonl(state / "backend_usage.jsonl", [row, row])
    item = snapshot(tmp_path, now=30)["attempts"][0]
    assert item["verifier_timings"]["call_work_seconds_sum"] == 5
    assert item["verifier_timings"]["source"] == "backend_usage_completed_only"
    assert item["backend_usage_by_role"]["oracles"]["input_tokens"] == 100


def test_role_telemetry_and_usage_ledger_do_not_double_count(tmp_path):
    state = attempt(tmp_path) / "private/state"
    write_jsonl(state / "role_calls.jsonl", [
        {"type": "role_call_started", "call_id": "outer", "role": "critic", "ts": 20},
        {"type": "role_call_finished", "call_id": "outer", "role": "critic", "ts": 25}])
    write_jsonl(state / "backend_usage.jsonl", [{"call_id": "inner", "role": "critic", "returncode": 0,
        "started_at_unix_ns": 20_000_000_000, "finished_at_unix_ns": 25_000_000_000,
        "duration_seconds": 5, "input_tokens": 10}])
    item = snapshot(tmp_path, now=30)["attempts"][0]
    assert item["verifier_timings"]["call_work_seconds_sum"] == 5
    assert item["backend_usage_by_role"]["critic"]["input_tokens"] == 10


def test_three_unchanged_snapshots_warn_once_without_lifecycle_actions(tmp_path):
    path = attempt(tmp_path)
    first = record_snapshot(tmp_path, now=30)
    status = json.loads((path / "status.json").read_text())
    status["updated_unix"] = 300  # A runner heartbeat is not solver progress.
    write_json(path / "status.json", status)
    second = record_snapshot(tmp_path, now=330)
    third = record_snapshot(tmp_path, now=630)
    fourth = record_snapshot(tmp_path, now=930)
    assert first["alerts"] == second["alerts"] == fourth["alerts"] == []
    assert third["alerts"][0]["type"] == "unchanged_progress_warning"
    assert fourth["attempts"][0]["completed_unix"] is None
    assert fourth["attempts"][0]["official_pass"] is None
    assert fourth["attempts"][0]["stage"] == "solver_running"
    assert len((tmp_path / "monitor/history.jsonl").read_text().splitlines()) == 4
    assert len((tmp_path / "monitor/alerts.jsonl").read_text().splitlines()) == 1


def test_new_candidate_resets_unchanged_counter(tmp_path):
    state = attempt(tmp_path) / "private/state"
    previous = snapshot(tmp_path, now=30)
    previous = snapshot(tmp_path, now=330, previous=previous)
    write_jsonl(state / "events.jsonl", [{"type": "cyber_candidate_registered", "seq": 1, "ts": 340}])
    value = snapshot(tmp_path, now=630, previous=previous)
    assert value["attempts"][0]["unchanged_snapshots"] == 1
    assert value["alerts"] == []


def test_solver_output_growth_resets_warning_without_publishing_contents(tmp_path):
    path = attempt(tmp_path)
    log = path / "solver.jsonl"
    log.write_text("PRIVATE_TRANSCRIPT\n")
    previous = record_snapshot(tmp_path, now=30)
    previous = record_snapshot(tmp_path, now=330)
    with log.open("a") as stream:
        stream.write("NEW_PRIVATE_COMMAND\n")
    value = record_snapshot(tmp_path, now=630)
    item = value["attempts"][0]
    assert item["unchanged_snapshots"] == 1
    assert item["solver_activity"]["log_bytes"] == log.stat().st_size
    assert not value["alerts"]
    assert "PRIVATE" not in json.dumps(value)
    assert item["official_pass"] is None


def test_touching_solver_log_does_not_count_as_output_activity(tmp_path):
    path = attempt(tmp_path)
    log = path / "solver.jsonl"
    log.write_text("unchanged\n")
    record_snapshot(tmp_path, now=30)
    record_snapshot(tmp_path, now=330)
    os.utime(log, (500, 500))
    value = record_snapshot(tmp_path, now=630)
    assert value["attempts"][0]["unchanged_snapshots"] == 3
    assert value["alerts"][0]["type"] == "unchanged_progress_warning"


def test_solver_log_symlink_does_not_expose_external_metadata(tmp_path):
    path = attempt(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.write_text("PRIVATE_TRANSCRIPT")
    try:
        (path / "solver.jsonl").symlink_to(outside)
        activity = snapshot(tmp_path, now=30)["attempts"][0]["solver_activity"]
        assert activity["available"] is False
        assert activity["log_bytes"] is None
        assert activity["last_log_update_unix"] is None
    finally:
        outside.unlink()


def test_backend_command_completion_resets_unchanged_counter(tmp_path):
    path = attempt(tmp_path)
    from verifier.llm.codex_backend import _CommandStats
    stats = _CommandStats().snapshot()
    backend_fixture(path, backend_row("exec_heartbeat", 25, cli_events=2, command_stats=stats))
    previous = snapshot(tmp_path, now=30)
    previous = snapshot(tmp_path, now=330, previous=previous)
    stats = {**stats, "completed_items": 1, "success_exits": 1,
             "fingerprinted_items": 1, "unique_fingerprints": 1, "max_fingerprint_repetitions": 1}
    backend_fixture(path, backend_row("exec_heartbeat", 625, cli_events=2, command_stats=stats))
    value = snapshot(tmp_path, now=630, previous=previous)
    assert value["attempts"][0]["unchanged_snapshots"] == 1
    assert "unchanged_progress_warning" not in alert_types(value)


def test_terminal_official_failure_is_not_deduced_from_evaluator_exit(tmp_path):
    attempt(tmp_path, completed_unix=25, stage="completed", outcome="completed", official_pass=False)
    first = snapshot(tmp_path, now=30)
    assert first["alerts"][0]["type"] == "official_task_not_passed"
    assert first["completed_attempts"] == 1
    assert first["attempts"][0]["elapsed_seconds"] == 15
    assert snapshot(tmp_path, now=300, previous=first)["alerts"] == []


def test_partial_jsonl_is_marked_unknown(tmp_path):
    state = attempt(tmp_path) / "private/state"
    write_jsonl(state / "role_calls.jsonl", [])
    (state / "role_calls.jsonl").write_text('{"type":"role_call_started"', encoding="utf-8")
    item = snapshot(tmp_path, now=30)["attempts"][0]
    assert item["verifier_timings"]["partial"] is True


def test_unsafe_private_symlink_is_not_read(tmp_path):
    path = attempt(tmp_path)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    write_jsonl(unrelated / "events.jsonl", [{"seq": 987654, "type": "cyber_materialized"}])
    (path / "private").mkdir()
    (path / "private/state").symlink_to(unrelated, target_is_directory=True)
    item = snapshot(tmp_path, now=30)["attempts"][0]
    assert item["progress"]["events_last_seq"] is None
    assert item["progress"]["events_window_partial"] is True


def test_monitor_output_symlink_refused(tmp_path):
    attempt(tmp_path)
    outside = tmp_path / "unrelated"
    outside.mkdir()
    (tmp_path / "monitor").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical"):
        record_snapshot(tmp_path, now=30)
    assert list(outside.iterdir()) == []


def test_cli_once_and_interval_validation(tmp_path, capsys):
    attempt(tmp_path)
    assert main(["--root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["attempt_count"] == 1
    assert (tmp_path / "monitor/latest.json").exists()
    with pytest.raises(SystemExit):
        main(["--root", str(tmp_path), "--interval", "nan"])


def test_finished_unchanged_attempts_do_not_reread_all_ledgers(tmp_path, monkeypatch):
    attempt(tmp_path, completed_unix=25, stage="completed", official_pass=True)
    previous = snapshot(tmp_path, now=30)

    def do_not_read(*args, **kwargs):
        raise AssertionError("unchanged finished ledgers should use cached summaries")

    monkeypatch.setattr(tracking, "_jsonl", do_not_read)
    value = snapshot(tmp_path, now=330, previous=previous)
    assert value["completed_attempts"] == 1
    assert value["official_passed_attempts"] == 1
    assert value["attempts"][0]["elapsed_seconds"] == 15


def evaluator_result(call_id, ts, stdout, **output_changes):
    output = {"elapsed_seconds": 1, "exit_code": 1, "stdout": stdout,
              "stdout_truncated": False, "timed_out": False}
    output.update(output_changes)
    return {"type": "tool_result", "seq": int(ts), "ts": ts,
            "payload": {"tool": "cyber_evaluator", "tool_use_id": call_id, "output": output}}


def adapter_error(code):
    return json.dumps({"schema": tracking.ADAPTER_OBSERVATION_SCHEMA, "error": code})


def test_adapter_infrastructure_errors_count_and_warn_once_per_code_without_leaks(tmp_path):
    path = attempt(tmp_path)
    state = path / "private/state"
    first = evaluator_result("a", 21, adapter_error("evidence_storage_error"),
                             stderr="PRIVATE_STDERR", configuration="PRIVATE_CONFIGURATION")
    second = evaluator_result("b", 22, adapter_error("evidence_storage_error"))
    rows = [first, first, second]  # Duplicate mirrored observations are not new executions.
    write_jsonl(state / "events.jsonl", rows)
    value = record_snapshot(tmp_path, now=30)
    item = value["attempts"][0]
    assert item["evaluator_infrastructure_diagnostics"] == {
        "count": 2, "by_error": {"evidence_storage_error": 2},
        "latest": {"error_code": "evidence_storage_error", "observed_unix": 22},
        "partial": False, "scope": "retained_event_window",
        "warned_error_codes": ["evidence_storage_error"],
    }
    assert [alert["type"] for alert in value["alerts"]] == ["evaluator_infrastructure_warning"]
    assert item["outcome"] is None and item["official_pass"] is None
    assert item["evaluator_timings"]["by_role"]["evaluator"]["failed_calls"] == 0
    assert "PRIVATE" not in json.dumps(value)
    rows.append(evaluator_result("c", 23, adapter_error("evidence_storage_error")))
    write_jsonl(state / "events.jsonl", rows)
    assert record_snapshot(tmp_path, now=330)["alerts"] == []
    rows.append(evaluator_result("d", 24, adapter_error("transport_error")))
    write_jsonl(state / "events.jsonl", rows)
    third = record_snapshot(tmp_path, now=630)
    assert third["alerts"][0]["error_codes"] == ["transport_error"]
    assert third["attempts"][0]["evaluator_infrastructure_diagnostics"]["count"] == 4
    assert len((tmp_path / "monitor/alerts.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("exit_code", [0, 300])
def test_official_candidate_negative_is_not_an_adapter_infrastructure_error(tmp_path, exit_code):
    state = attempt(tmp_path) / "private/state"
    stdout = json.dumps({"schema": tracking.ADAPTER_OBSERVATION_SCHEMA,
                         "response": {"exit_code": exit_code, "output": "PRIVATE_TARGET_OUTPUT"}})
    write_jsonl(state / "events.jsonl", [evaluator_result("a", 21, stdout, exit_code=0)])
    value = snapshot(tmp_path, now=30)
    assert value["attempts"][0]["evaluator_infrastructure_diagnostics"]["count"] == 0
    assert value["alerts"] == []
    assert value["attempts"][0]["official_pass"] is None
    assert "PRIVATE_TARGET_OUTPUT" not in json.dumps(value)


@pytest.mark.parametrize("stdout,changes", [
    (adapter_error("evidence_storage_error"), {"stdout_truncated": True}),
    (adapter_error("evidence_storage_error"), {"stdout_bytes_observed": 10_000}),
    ('{"schema":"gv-cybergym-vul-observation-v1","error":', {}),
    ('not JSON: evidence_storage_error', {}),
    (json.dumps({"schema": "another-adapter", "error": "evidence_storage_error"}), {}),
    (adapter_error("invalid_candidate"), {}),
    (adapter_error("PRIVATE_DYNAMIC_ERROR"), {}),
    ('{"schema":"gv-cybergym-vul-observation-v1","error":"other","error":"evidence_storage_error"}', {}),
    ('{"schema":"gv-cybergym-vul-observation-v1","error":"evidence_storage_error","response_truncated":NaN}', {}),
    (json.dumps({"schema": tracking.ADAPTER_OBSERVATION_SCHEMA, "error": "response_too_large",
                 "response_truncated": True}), {}),
    (json.dumps({"schema": tracking.ADAPTER_OBSERVATION_SCHEMA, "error": "evidence_storage_error",
                 "response": {"output": "PRIVATE_TARGET_OUTPUT"}}), {}),
    ("\ud800", {}),
])
def test_partial_or_unrecognized_adapter_output_does_not_infer_infrastructure_failure(tmp_path, stdout, changes):
    state = attempt(tmp_path) / "private/state"
    write_jsonl(state / "events.jsonl", [evaluator_result("a", 21, stdout, **changes)])
    value = snapshot(tmp_path, now=30)
    assert value["attempts"][0]["evaluator_infrastructure_diagnostics"]["count"] == 0
    assert value["alerts"] == []
    assert "PRIVATE" not in json.dumps(value)


def test_adapter_warning_stays_deduplicated_when_fault_leaves_bounded_tail(tmp_path):
    state = attempt(tmp_path) / "private/state"
    row = evaluator_result("a", 21, adapter_error("evidence_storage_error"))
    write_jsonl(state / "events.jsonl", [row])
    first = snapshot(tmp_path, now=30)
    write_jsonl(state / "events.jsonl", [])
    second = snapshot(tmp_path, now=330, previous=first)
    write_jsonl(state / "events.jsonl", [row])
    third = snapshot(tmp_path, now=630, previous=second)
    assert first["alerts"][0]["type"] == "evaluator_infrastructure_warning"
    assert third["alerts"] == []


def native_progress(path, **changes):
    job = {"job_id": "job-1", "tool": "cyber_finalize_candidate", "state": "running",
           "created_unix": 20, "started_unix": 21, "updated_unix": 22,
           "attempts": 1, "reused_count": 0}
    job.update(changes)
    write_json(path / "private/native-progress.json", {
        "schema": tracking.PROGRESS_SCHEMA, "checked_unix": 700, "jobs": [job],
        "runtime_secret": "PRIVATE_RUNTIME", "request": "PRIVATE_REQUEST"})


def test_native_job_projection_has_no_request_result_or_secret_leak(tmp_path):
    path = attempt(tmp_path)
    native_progress(path, result="PRIVATE_RESULT", request="PRIVATE_JOB_REQUEST")
    value = snapshot(tmp_path, now=30)
    job = value["attempts"][0]["verification_progress"]["jobs"][0]
    assert job["job_id"] == "job-1" and job["state"] == "running"
    assert "PRIVATE" not in json.dumps(value)
    assert value["official_passed_attempts"] == 0 and value["alerts"] == []


def test_native_progress_accepts_actual_stable_job_id_prefix(tmp_path):
    path = attempt(tmp_path)
    job_id = "verification:" + "a" * 32
    native_progress(path, job_id=job_id)
    value = snapshot(tmp_path, now=30)
    assert value["attempts"][0]["verification_progress"]["available"] is True
    assert value["attempts"][0]["verification_progress"]["active_job_ids"] == [job_id]


@pytest.mark.parametrize("changes", [
    {"job_id": "../private-secret"}, {"tool": ["commit"]},
    {"state": "unknown"}, {"state": ["running"]}, {"job_id": None},
])
def test_invalid_progress_metadata_is_unknown_not_a_task_failure(tmp_path, changes):
    path = attempt(tmp_path)
    native_progress(path, **changes)
    value = snapshot(tmp_path, now=30)
    assert value["attempts"][0]["verification_progress"]["available"] is False
    assert value["attempts"][0]["official_pass"] is None and value["alerts"] == []


def test_long_roles_and_jobs_warn_once_without_timeout_restart_or_failure(tmp_path):
    path = attempt(tmp_path)
    native_progress(path)
    write_jsonl(path / "private/state/role_calls.jsonl", [{
        "type": "role_call_started", "call_id": "role-1", "role": "judger", "ts": 21}])
    first = snapshot(tmp_path, now=700)
    assert {alert["type"] for alert in first["alerts"]} == {
        "long_role_call_warning", "long_verification_job_warning"}
    assert snapshot(tmp_path, now=800, previous=first)["alerts"] == []
    assert all(alert["diagnostic_only"] and alert["lifecycle_action"] is None for alert in first["alerts"])
    assert first["attempts"][0]["official_pass"] is None
    assert first["attempts"][0]["completed_unix"] is None


def test_reused_job_polls_are_not_redo_or_new_progress(tmp_path):
    path = attempt(tmp_path)
    native_progress(path, state="completed", reused_count=0)
    first = snapshot(tmp_path, now=30)
    native_progress(path, state="completed", reused_count=1000, updated_unix=330)
    second = snapshot(tmp_path, now=330, previous=first)
    assert second["attempts"][0]["unchanged_snapshots"] == 2
    assert second["alerts"] == []


def test_explicit_repeated_job_execution_warns_but_reuse_count_does_not(tmp_path):
    path = attempt(tmp_path)
    native_progress(path, state="completed", attempts=3, reused_count=1000)
    value = snapshot(tmp_path, now=30)
    assert [alert["type"] for alert in value["alerts"]] == ["verification_job_redo_warning"]
    assert value["attempts"][0]["official_pass"] is None


@pytest.mark.parametrize("state", ["failed", "interrupted"])
def test_failed_or_interrupted_native_job_warns_without_official_failure_or_retry(tmp_path, state):
    path = attempt(tmp_path)
    native_progress(path, state=state)
    first = snapshot(tmp_path, now=30)
    assert [alert["type"] for alert in first["alerts"]] == ["verification_job_state_warning"]
    assert first["alerts"][0]["lifecycle_action"] is None
    assert first["attempts"][0]["official_pass"] is None
    assert snapshot(tmp_path, now=330, previous=first)["alerts"] == []


def test_wait_termination_is_explicit_client_cancel_not_timeout_or_job_failure(tmp_path):
    path = attempt(tmp_path)
    native_progress(path)
    rollout = path / "solver-home/sessions/2026/09/12/rollout-fixture.jsonl"
    write_jsonl(rollout, [
        {"type": "response_item", "timestamp": "1970-01-01T00:00:25Z", "payload": {
            "type": "function_call", "name": "wait", "call_id": "cancel-1",
            "arguments": json.dumps({"cell_id": "62", "terminate": True})}},
        {"type": "response_item", "timestamp": "1970-01-01T00:00:26Z", "payload": {
            "type": "reasoning", "encrypted_content": "PRIVATE_REASONING"}},
        {"type": "response_item", "timestamp": "1970-01-01T00:00:27Z", "payload": {
            "type": "function_call_output", "output": "PRIVATE_OUTPUT"}}])
    value = snapshot(tmp_path, now=30)
    assert value["attempts"][0]["solver_wait_cancellations"]["count_in_window"] == 1
    assert [alert["type"] for alert in value["alerts"]] == ["solver_wait_cancelled_while_verification_active"]
    assert "PRIVATE" not in json.dumps(value)
    assert value["attempts"][0]["verification_progress"]["jobs"][0]["state"] == "running"
    assert value["attempts"][0]["official_pass"] is None


def test_new_epoch_claim_ids_without_new_candidate_evidence_warn_as_repeat(tmp_path):
    path = attempt(tmp_path)
    rows = [{"seq": index, "type": "commit", "ts": index, "payload": {
        "cybergv_binding": {"certificate_scope": "candidate", "profile_id": "fixture-profile",
            "task_sha256": "fixture-task", "verification_epoch": str(index), "context_id": str(index),
            "subject": {"candidate_id": "PRIVATE_SUBJECT", "address": "/task/final_poc"},
            "observations": ["PRIVATE_OBSERVATION"]}}} for index in range(1, 4)]
    write_jsonl(path / "private/state/events.jsonl", rows)
    value = snapshot(tmp_path, now=30)
    assert [alert["type"] for alert in value["alerts"]] == ["repeated_finalization_warning"]
    assert value["alerts"][0]["count"] == 3
    assert "PRIVATE" not in json.dumps(value)
    assert value["attempts"][0]["official_pass"] is None


def test_outstanding_service_call_warns_without_claiming_shell_is_hung(tmp_path):
    path = attempt(tmp_path)
    write_jsonl(path / "private/state/transport_calls.jsonl", [
        {"call_id": "call-1", "tool": "commit", "phase": "received", "ts": 21},
        {"call_id": "call-1", "tool": "commit", "phase": "invoked", "ts": 21.001,
         "arguments": "PRIVATE_ARGS", "output": "PRIVATE_OUTPUT"}])
    value = snapshot(tmp_path, now=700)
    assert [alert["type"] for alert in value["alerts"]] == ["long_service_call_warning"]
    assert value["attempts"][0]["transport_timings"]["active_calls"][0]["age_seconds"] == 679
    assert "PRIVATE" not in json.dumps(value)
    assert value["attempts"][0]["official_pass"] is None


def test_mirror_timing_is_measured_and_telemetry_faults_warn_without_secret_leak(tmp_path):
    path = attempt(tmp_path)
    write_jsonl(path / "private/telemetry-mirror-calls.jsonl", [{
        "ts": 30, "phase": "periodic", "duration_seconds": 5}])
    write_jsonl(path / "private/telemetry-progress-errors.jsonl", [{
        "ts": 31, "reason": "progress_unavailable", "stderr": "PRIVATE_STDERR"}])
    value = snapshot(tmp_path, now=40)
    assert value["attempts"][0]["mirror_timings"]["call_work_seconds_sum"] == 5
    assert [alert["type"] for alert in value["alerts"]] == ["telemetry_infrastructure_warning"]
    assert "PRIVATE" not in json.dumps(value)
    assert value["attempts"][0]["official_pass"] is None


def backend_row(phase, ts, *, call_id="backend-local-1", attempt=1, **metadata):
    return {"type": "backend_phase", "call_id": call_id, "attempt": attempt,
            "phase": phase, "ts": ts, **metadata}


def backend_fixture(path, *rows, mirror_unix=None, outer_rows=None):
    write_jsonl(path / "private/state/role_calls.jsonl", [
        *([{"type": "role_call_started", "call_id": "outer-role-1", "ts": 20,
            "role": "critic"}] if outer_rows is None else outer_rows), *rows])
    if mirror_unix is not None:
        write_jsonl(path / "private/telemetry-mirror-calls.jsonl", [{
            "ts": mirror_unix, "phase": "periodic", "status": "completed", "duration_seconds": 2}])


def alert_types(value):
    return {alert["type"] for alert in value["alerts"]}


def test_journal_unavailable_is_metadata_only_and_not_an_official_failure(tmp_path):
    path = attempt(tmp_path)
    native_progress(path, state="completed")
    progress_path = path / "private/native-progress.json"
    raw = json.loads(progress_path.read_text())
    raw.update(journal_available=False, metadata_snapshot_unix=25, result="PRIVATE_RESULT")
    write_json(progress_path, raw)
    first = snapshot(tmp_path, now=30)
    projected = first["attempts"][0]["verification_progress"]
    assert projected["journal_available"] is False
    assert projected["metadata_snapshot_unix"] == 25
    assert projected["metadata_snapshot_age_seconds"] == 5
    assert alert_types(first) == {"verification_journal_unavailable_warning"}
    assert "PRIVATE" not in json.dumps(first)
    assert first["attempts"][0]["official_pass"] is None
    assert first["alerts"][0]["lifecycle_action"] is None
    assert snapshot(tmp_path, now=330, previous=first)["alerts"] == []


def test_old_journal_snapshot_is_not_a_deadline_and_polled_reuse_is_not_progress(tmp_path):
    path = attempt(tmp_path)
    native_progress(path)
    progress_path = path / "private/native-progress.json"
    raw = json.loads(progress_path.read_text())
    raw.update(journal_available=True, metadata_snapshot_unix=20)
    write_json(progress_path, raw)
    first = snapshot(tmp_path, now=700)
    raw.update(checked_unix=800, metadata_snapshot_unix=800)
    raw["jobs"][0].update(reused_count=1000, updated_unix=800)
    write_json(progress_path, raw)
    second = snapshot(tmp_path, now=800, previous=first)
    assert second["attempts"][0]["unchanged_snapshots"] == 2
    assert second["alerts"] == []
    assert "stale_native_progress_warning" not in alert_types(first)


def test_backend_projection_keeps_distinct_ids_and_scrubs_nonmetadata(tmp_path):
    path = attempt(tmp_path)
    backend_fixture(path,
        backend_row("exec_dispatch", 21, role="critic", streaming=True, prompt="PRIVATE_PROMPT"),
        backend_row("exec_spawned", 22, child_pid=41, wait_category="cli_startup_unknown"),
        backend_row("cli_event", 23, event_type="item.started", item_kind="command_execution",
                    active_tool_items=1, wait_category="cli_tool_item_active", command="PRIVATE_COMMAND"),
        backend_row("cli_event", 28, event_type="item.completed", item_kind="command_execution",
                    active_tool_items=0, item_duration_seconds=5, item_id="PRIVATE_ITEM"),
        backend_row("exec_heartbeat", 29, child_pid=41, cli_events=2, stdout_bytes=40,
                    stderr_bytes=5, wait_category="model_wait_unknown", stdout="PRIVATE_OUTPUT"))
    value = snapshot(tmp_path, now=30)
    activity = value["attempts"][0]["backend_activity"]
    call = activity["calls"][0]
    assert call["backend_call_id"] == "backend-local-1"
    assert call["outer_role_call_id"] == "outer-role-1"
    assert call["role_association"] == "unique_role_interval"
    assert call["role"] == "critic" and call["wait_category"] == "model_wait_unknown"
    assert call["process_running"] is True and call["heartbeat_age_seconds"] == 1
    assert call["last_cli_event_age_seconds"] == 2
    assert call["completed_tool_items"] == {"command_execution": {"count": 1, "work_seconds_sum": 5}}
    assert activity["provider_queue_vs_inference"] == "unknown"
    assert "PRIVATE" not in json.dumps(value) and value["alerts"] == []
    assert call["command_stats"] == {"available": False, "partial": True}


def test_command_metadata_projection_is_typed_and_never_payload_or_deadline(tmp_path):
    from verifier.llm.codex_backend import _CommandStats, _command_metadata

    path = attempt(tmp_path)
    stats = _CommandStats()
    item = _command_metadata({"command": "PRIVATE_COMMAND", "exit_code": 0, "status": "completed",
                              "aggregated_output": "PRIVATE_OUTPUT"})
    stats.completed("private-id", item)
    duplicate = stats.completed("private-id", item)
    counter = {**stats.snapshot(), "raw": "PRIVATE_COUNTER"}
    backend_fixture(path,
        backend_row("exec_spawned", 22),
        backend_row("cli_event", 23, event_type="item.completed", item_kind="command_execution",
                    item_duration_seconds=2, command_stats=counter, **item),
        backend_row("cli_event", 24, event_type="item.completed", item_kind="command_execution",
                    item_duration_seconds=2, command_stats=counter, command_duplicate_completion=duplicate, **item))
    value = snapshot(tmp_path, now=30)
    call = value["attempts"][0]["backend_activity"]["calls"][0]
    assert call["command_stats"]["available"] and call["command_stats"]["partial"]
    assert call["command_stats"]["completed_items"] == 1
    assert call["last_command"]["exit_code"] == 0 and call["last_command"]["available"]
    assert call["completed_tool_items"]["command_execution"]["count"] == 1
    assert "PRIVATE" not in json.dumps(value)
    assert value["attempts"][0]["backend_activity"]["provider_queue_vs_inference"] == "unknown"
    assert value["alerts"] == []


def test_malformed_command_stats_and_fields_stay_unavailable_not_zero_failure(tmp_path):
    path = attempt(tmp_path)
    backend_fixture(path, backend_row("cli_event", 23, event_type="item.completed", item_kind="command_execution",
        command_stats={"available": True, "scope": "cli_attempt_prefix", "completed_items": True,
                       "partial": False, "raw": "PRIVATE"}, command_sha256="PRIVATE",
        command_exit_code=True, command_status=["PRIVATE"], command_category=["PRIVATE"],
        command_output_utf8_bytes=-1, command_metadata_available="PRIVATE"))
    value = snapshot(tmp_path, now=30)
    call = value["attempts"][0]["backend_activity"]["calls"][0]
    assert call["command_stats"] == {"available": False, "partial": True}
    assert call["last_command"] == {"available": False}
    assert "PRIVATE" not in json.dumps(value) and value["alerts"] == []


def test_command_stats_tail_projection_is_partial_not_whole_history():
    from verifier.llm.codex_backend import _CommandStats
    from experiments.cybergym.tracking import _backend_activity

    row = backend_row("exec_heartbeat", 23, command_stats=_CommandStats().snapshot())
    stats = _backend_activity([row], [], 30, partial=True)["calls"][0]["command_stats"]
    assert stats["available"] and stats["partial"] and stats["truncated"]
    assert stats["scope"] == "cli_attempt_prefix"


@pytest.mark.parametrize("second_role,expected_id,association", [
    ("judger", "outer-role-1", "unique_role_interval"),
    ("critic", None, "ambiguous_interval"),
])
def test_backend_outer_association_uses_role_and_time_not_same_call_id(tmp_path, second_role, expected_id, association):
    path = attempt(tmp_path)
    backend_fixture(path, backend_row("exec_dispatch", 25, role="critic", streaming=True),
        outer_rows=[
            {"type": "role_call_started", "call_id": "outer-role-1", "role": "critic", "ts": 20},
            {"type": "role_call_started", "call_id": "outer-role-2", "role": second_role, "ts": 21}])
    call = snapshot(tmp_path, now=30)["attempts"][0]["backend_activity"]["calls"][0]
    assert call["outer_role_call_id"] == expected_id
    assert call["role_association"] == association


def test_backend_role_remains_known_when_bounded_tail_has_no_outer_start(tmp_path):
    path = attempt(tmp_path)
    backend_fixture(path, backend_row("exec_heartbeat", 29, role="judger"), outer_rows=[])
    activity = snapshot(tmp_path, now=30)["attempts"][0]["backend_activity"]
    assert activity["calls"][0]["role"] == "judger"
    assert activity["calls"][0]["outer_role_call_id"] is None
    assert activity["calls"][0]["role_association"] == "unknown"


@pytest.mark.parametrize("wait_category,warning", [
    ("cli_tool_item_active", "backend_tool_item_long_warning"),
    ("model_wait_unknown", "backend_cli_output_stale_warning"),
    ("PRIVATE_DYNAMIC_WAIT", "backend_cli_output_stale_warning"),
])
def test_fresh_heartbeat_distinguishes_observed_tool_state_from_unknown_model_wait(tmp_path, wait_category, warning):
    path = attempt(tmp_path)
    backend_fixture(path,
        backend_row("exec_dispatch", 21, role="critic", streaming=True),
        backend_row("exec_spawned", 22),
        backend_row("cli_event", 23, event_type="item.started", item_kind="mcp_tool_call",
                    active_tool_items=1, wait_category=wait_category),
        backend_row("exec_heartbeat", 699, active_tool_items=1, wait_category=wait_category),
        mirror_unix=700)
    first = snapshot(tmp_path, now=700)
    assert alert_types(first) == {"long_role_call_warning", warning}
    long_call = next(alert for alert in first["alerts"] if alert["type"] == "long_role_call_warning")
    backend = long_call["backend_activity"][0]
    assert backend["outer_role_call_id"] == "outer-role-1"
    assert backend["wait_category"] == (wait_category if wait_category in tracking.BACKEND_WAIT_CATEGORIES else "unknown")
    assert backend["heartbeat_age_seconds"] == 1
    assert "PRIVATE" not in json.dumps(first)
    assert all(alert["lifecycle_action"] is None for alert in first["alerts"])
    assert first["attempts"][0]["official_pass"] is None
    assert snapshot(tmp_path, now=710, previous=first)["alerts"] == []


@pytest.mark.parametrize("mirror_unix,warning", [
    (700, "backend_heartbeat_stale_warning"),
    (30, "backend_observability_stale_warning"),
    (None, "backend_observability_stale_warning"),
])
def test_missing_heartbeat_is_not_inferred_from_a_stale_host_mirror(tmp_path, mirror_unix, warning):
    path = attempt(tmp_path)
    backend_fixture(path,
        backend_row("exec_dispatch", 21, role="critic", streaming=True),
        backend_row("exec_spawned", 22), mirror_unix=mirror_unix)
    value = snapshot(tmp_path, now=700)
    assert alert_types(value) == {"long_role_call_warning", warning}
    assert value["attempts"][0]["official_pass"] is None
    if mirror_unix != 700:
        assert "backend_heartbeat_stale_warning" not in alert_types(value)


def test_nonstreaming_legacy_backend_has_no_false_missing_heartbeat_warning(tmp_path):
    path = attempt(tmp_path)
    backend_fixture(path, backend_row("exec_dispatch", 21, role="critic", streaming=False), mirror_unix=700)
    assert alert_types(snapshot(tmp_path, now=700)) == {"long_role_call_warning"}


def test_pipe_drain_is_visible_early_after_cli_exit_without_guessing_cause(tmp_path):
    path = attempt(tmp_path)
    backend_fixture(path,
        backend_row("exec_dispatch", 21, streaming=True),
        backend_row("exec_spawned", 22),
        backend_row("exec_process_exited", 25),
        backend_row("pipe_drain_heartbeat", 45, process_running=False,
                    wait_category="pipe_drain_unknown", stdout_bytes=50, stderr="PRIVATE_STDERR"))
    first = snapshot(tmp_path, now=46)
    assert alert_types(first) == {"backend_pipe_drain_warning"}
    alert = first["alerts"][0]
    assert alert["process_running"] is False and alert["wait_category"] == "pipe_drain_unknown"
    assert alert["lifecycle_action"] is None
    assert "unknown" in alert["meaning"] and "PRIVATE" not in json.dumps(first)
    assert snapshot(tmp_path, now=50, previous=first)["alerts"] == []


def test_completed_short_backoff_and_transient_retry_remain_visible(tmp_path):
    path = attempt(tmp_path)
    backend_fixture(path,
        backend_row("backend_transient_failure", 22, role="critic", stderr="PRIVATE_STDERR"),
        backend_row("backend_backoff_started", 23, wait_seconds=2, retry_number=1, retry_limit=8),
        backend_row("backend_backoff_finished", 25, wait_seconds=2))
    first = snapshot(tmp_path, now=30)
    assert alert_types(first) == {"backend_transient_retry_warning"}
    alert = first["alerts"][0]
    assert alert["backoff_wait_seconds"] == 2 and alert["retry_number"] == 1
    assert "PRIVATE" not in json.dumps(first)
    assert alert["lifecycle_action"] is None
    assert snapshot(tmp_path, now=330, previous=first)["alerts"] == []


@pytest.mark.parametrize("event_type", ["error", "turn.failed"])
@pytest.mark.parametrize("exited", [False, True])
def test_native_cli_error_remains_visible_after_following_activity_without_failing_task(tmp_path, event_type, exited):
    path = attempt(tmp_path)
    rows = [backend_row("exec_spawned", 22),
            backend_row("cli_event", 23, event_type=event_type, message="PRIVATE_ERROR",
                        error={"message": "PRIVATE_PROVIDER_ERROR"}),
            backend_row("cli_event", 24, event_type="item.completed", item_kind="agent_message")]
    if exited:
        rows.append(backend_row("exec_process_exited", 25))
    backend_fixture(path, *rows)
    first = snapshot(tmp_path, now=30)
    assert alert_types(first) == {"backend_cli_error_warning"}
    call = first["attempts"][0]["backend_activity"]["calls"][0]
    assert call["last_cli_event"] == "item.completed"
    assert call["cli_error_events"] == {event_type: 1}
    assert call["process_running"] is (not exited)
    alert = first["alerts"][0]
    assert alert["error_event_counts"] == {event_type: 1}
    assert alert["last_error_event_unix"] == 23
    assert alert["lifecycle_action"] is None
    assert first["attempts"][0]["official_pass"] is None
    assert first["attempts"][0]["completed_unix"] is None
    assert "PRIVATE" not in json.dumps(first)
    assert snapshot(tmp_path, now=31, previous=first)["alerts"] == []


def test_completed_cache_from_older_observer_recomputes_native_error_history(tmp_path):
    path = attempt(tmp_path, completed_unix=25, stage="completed", official_pass=False)
    backend_fixture(path,
        backend_row("exec_spawned", 22),
        backend_row("cli_event", 23, event_type="error", message="PRIVATE_ERROR"),
        backend_row("exec_process_exited", 24), outer_rows=[])
    previous = snapshot(tmp_path, now=30)
    old_call = previous["attempts"][0]["backend_activity"]["calls"][0]
    del old_call["cli_error_events"]
    del old_call["last_cli_error_unix"]
    value = snapshot(tmp_path, now=31, previous=previous)
    assert value["attempts"][0]["backend_activity"]["calls"][0]["cli_error_events"] == {"error": 1}
    assert value["attempts"][0]["official_pass"] is False
    assert value["completed_attempts"] == 1
    assert "PRIVATE" not in json.dumps(value)


@pytest.mark.parametrize("mirror_unix,overrun", [(100, True), (None, False)])
def test_backoff_overrun_requires_fresh_copied_metadata_and_never_cancels(tmp_path, mirror_unix, overrun):
    path = attempt(tmp_path)
    backend_fixture(path,
        backend_row("backend_backoff_started", 23, wait_seconds=2, role="critic"), mirror_unix=mirror_unix)
    value = snapshot(tmp_path, now=100)
    assert ("backend_backoff_overrun_warning" in alert_types(value)) is overrun
    assert "backend_transient_retry_warning" in alert_types(value)
    assert value["attempts"][0]["completed_unix"] is None
    assert value["attempts"][0]["official_pass"] is None


def test_backend_heartbeat_and_observation_metadata_are_not_material_progress(tmp_path):
    path = attempt(tmp_path)
    rows = [backend_row("exec_dispatch", 21, role="critic", streaming=True), backend_row("exec_spawned", 22)]
    backend_fixture(path, *rows)
    first = snapshot(tmp_path, now=30)
    backend_fixture(path, *rows, backend_row("exec_heartbeat", 329, wait_category="model_wait_unknown"))
    second = snapshot(tmp_path, now=330, previous=first)
    assert second["attempts"][0]["unchanged_snapshots"] == 2
    assert second["alerts"] == []


def test_malformed_backend_metadata_is_ignored_or_unknown_without_crashing(tmp_path):
    path = attempt(tmp_path)
    backend_fixture(path,
        backend_row(["exec_dispatch"], 21), backend_row("exec_dispatch", True),
        backend_row("exec_dispatch", 21, attempt=True), backend_row("exec_dispatch", 21, call_id="/secret/path"),
        backend_row("exec_dispatch", 22, role=["PRIVATE_ROLE"], streaming="PRIVATE_STREAM"),
        backend_row("cli_event", 23, event_type=["PRIVATE_EVENT"], item_kind=["PRIVATE_ITEM"],
                    wait_category=["PRIVATE_WAIT"], child_pid=True, active_tool_items="PRIVATE_COUNTER",
                    item_duration_seconds=float("nan"), stdout="PRIVATE_OUTPUT"))
    value = snapshot(tmp_path, now=30)
    calls = value["attempts"][0]["backend_activity"]["calls"]
    assert len(calls) == 1 and calls[0]["wait_category"] == "unknown"
    assert "child_pid" not in calls[0] and "active_tool_items" not in calls[0]
    assert "PRIVATE" not in json.dumps(value)
    assert value["alerts"] == []
