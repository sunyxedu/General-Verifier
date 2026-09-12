"""Tracking fixtures are diagnostics only: no API, Docker or evaluator calls."""
import json
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
