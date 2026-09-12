import json
from pathlib import Path

import pytest

from experiments.terminal_bench import profile_typed_adapter_eval as profile


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _run(tmp_path, *, run_id="r1", arm="treatment", complete=True):
    run_dir = tmp_path / "runs" / run_id
    usage = run_dir / "usage.jsonl"
    results = run_dir / "results.jsonl"
    row = {
        "schema": profile.RUN_SCHEMA,
        "run_id": run_id,
        "arm": arm,
        "tasks": ["task-a"],
        "trials": ["task-a__one"],
        "results_path": str(results.relative_to(tmp_path)),
        "usage_paths": [str(usage.relative_to(tmp_path))],
        "exit_code": 0 if complete else 1,
        "usage_audit_complete": complete,
        "usage_audit_errors": [],
        "result_audit_errors": [],
        "source_fingerprint_unchanged": True,
        "started_at_monotonic_ns": 1,
        "finished_at_monotonic_ns": 2,
    }
    _write_jsonl(results, [{"task": "task-a", "trial": "task-a__one"}])
    return row, usage, run_dir


def _attempt(attempt, role, start, finish, *, status="succeeded", **extra):
    input_tokens = extra.pop("input_tokens", 11)
    cached = extra.pop("cached_input_tokens", 4)
    return {
        "record_type": profile.ATTEMPT_RECORD,
        "attempt_id": attempt,
        "call_id": "call-" + attempt,
        "attempt_index": 1,
        "started": True,
        "attempt_status": status,
        "usage_status": "complete",
        "usage_complete": True,
        "role": role,
        "task": "task-a",
        "started_at_unix_ns": start,
        "finished_at_unix_ns": finish,
        "duration_seconds": (finish - start) / 1e9,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "uncached_input_tokens": input_tokens - cached,
        "output_tokens": 3,
        "reasoning_tokens": 2,
        **extra,
    }


def test_profiles_overlapping_intervals_and_event_counts(tmp_path):
    run, usage, run_dir = _run(tmp_path)
    state = run_dir / "scoring" / "task-a__one" / "gv"
    rows = [
        _attempt("a1", "discharge", 0, 10_000_000_000),
        _attempt("a2", "discharge", 5_000_000_000, 15_000_000_000,
                 status="provider_retry"),
        _attempt("a3", "discharge", 20_000_000_000, 25_000_000_000),
        _attempt("a4", "oracles", 2_000_000_000, 4_000_000_000),
    ]
    for row in rows:
        row["state_dir"] = str(state)
    _write_jsonl(usage, rows)
    _write_jsonl(state / "events.jsonl", [
        {"type": "contract_cache_miss", "payload": {"secret": "never"}},
        {"type": "contract_cache_published", "payload": {}},
        {"type": "method_cache_hit", "payload": {}},
        {"type": "method_cache_direction_audit",
         "payload": {"method_digest": "private", "reason": "never"}},
        {"type": "adapter_discharge", "payload": {"procedure": "private"}},
        {"type": "adapter_fallback", "payload": {}},
        {"type": "method_rejected", "payload": {"reason": "private"}},
    ])
    ledger = tmp_path / "eval-runs.jsonl"
    _write_jsonl(ledger, [run])

    report = profile.profile(ledger)
    group, = report["groups"]
    roles = {row["role"]: row for row in group["roles"]}
    discharge = roles["discharge"]
    assert discharge["attempts"] == 3
    assert discharge["attempt_statuses"] == {
        "provider_retry": 1, "succeeded": 2}
    assert discharge["duration_seconds"] == {
        "known_count": 3,
        "unknown_started_attempts": 0,
        "sum": 25.0,
        "p50": 10.0,
        "max": 10.0,
    }
    assert discharge["call_interval_union_seconds"] == {
        "known_union": 20.0,
        "total": 20.0,
        "unknown_started_attempts": 0,
    }
    assert discharge["tokens"]["uncached_input_tokens"]["total"] == 21
    assert group["events"]["contract_cache"] == {
        "hits": 0, "misses": 1, "bypasses": 0, "published": 1}
    assert group["events"]["method_cache"] == {
        "hits": 1, "explicit_miss_events": 0,
        "misses_observable": False, "bypasses": 0,
        "direction_audit_events": 1}
    assert group["events"]["adapter"] == {
        "executions": 1, "fallbacks": 1}
    encoded = json.dumps(report)
    assert "never" not in encoded
    assert "private" not in encoded
    assert "not the dependency critical path" in encoded
    assert "direction audit events=1" in profile.render_summary(report)


def test_unknown_usage_and_unfinished_run_remain_incomplete(tmp_path):
    run, usage, run_dir = _run(tmp_path, complete=False)
    state = run_dir / "scoring" / "task-a__one" / "gv"
    _write_jsonl(usage, [{
        "record_type": profile.ATTEMPT_RECORD,
        "attempt_id": "unknown-1",
        "call_id": "call-unknown",
        "attempt_index": 1,
        "started": True,
        "attempt_status": "timeout",
        "usage_status": "unknown",
        "usage_complete": False,
        "role": "discharge",
        "task": "task-a",
        "state_dir": str(state),
        "started_at_unix_ns": 1_000_000_000,
        # no finished timestamp or token counts: genuinely unknown
        "prompt": "do not expose this prompt",
    }, {
        "record_type": profile.ATTEMPT_RECORD,
        "attempt_id": "launch-1",
        "call_id": "call-launch",
        "attempt_index": 1,
        "started": False,
        "attempt_status": "launch_error",
        "usage_status": "unknown",
        "usage_complete": False,
        "role": "discharge",
        "task": "task-a",
        "state_dir": str(state),
    }])
    _write_jsonl(state / "events.jsonl", [])
    orphan = tmp_path / "runs" / "r-running" / "run-state.json"
    orphan.parent.mkdir(parents=True)
    orphan.write_text(json.dumps({
        "schema": profile.RUN_SCHEMA,
        "run_id": "r-running",
        "status": "running",
    }))
    ledger = tmp_path / "eval-runs.jsonl"
    _write_jsonl(ledger, [run])

    report = profile.profile(ledger)
    assert report["cost_complete"] is False
    assert report["runs"]["complete"] == 0
    assert report["runs"]["incomplete_or_failed"] == 1
    assert report["runs"]["orphan_run_states"] == {"running": 1}
    role = report["groups"][0]["roles"][0]
    assert role["attempts"] == 2
    assert role["tokens"]["output_tokens"] == {
        "known_total": 0, "total": None, "unknown_started_attempts": 1}
    assert role["call_interval_union_seconds"] == {
        "known_union": 0.0,
        "total": None,
        "unknown_started_attempts": 1,
    }
    assert "do not expose this prompt" not in json.dumps(report)
    assert "INCOMPLETE" in profile.render_summary(report)


def test_missing_usage_file_is_not_a_zero_cost_success(tmp_path):
    run, usage, _run_dir = _run(tmp_path)
    assert not usage.exists()
    ledger = tmp_path / "eval-runs.jsonl"
    _write_jsonl(ledger, [run])

    report = profile.profile(ledger)
    group, = report["groups"]
    assert report["cost_complete"] is False
    assert group["usage_files"] == {"declared": 1, "readable": 0}
    assert group["roles"] == []
    assert any("usage unavailable" in issue for issue in report["issues"])


def test_first_running_process_does_not_require_a_terminal_ledger(tmp_path):
    marker = tmp_path / "runs" / "first" / "run-state.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({
        "schema": profile.RUN_SCHEMA, "run_id": "first", "status": "running",
    }), encoding="utf-8")

    report = profile.profile(tmp_path / "eval-runs.jsonl")

    assert report["cost_complete"] is False
    assert report["runs"]["complete"] == 0
    assert report["runs"]["orphan_run_states"] == {"running": 1}
    assert "INCOMPLETE" in profile.render_summary(report)


def test_missing_ledger_without_a_real_run_still_errors(tmp_path):
    with pytest.raises(profile.ProfileError):
        profile.profile(tmp_path / "eval-runs.jsonl")


def test_existing_malformed_ledger_is_not_hidden_by_a_running_marker(tmp_path):
    marker = tmp_path / "runs" / "first" / "run-state.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"run_id":"first","status":"running"}',
                      encoding="utf-8")
    ledger = tmp_path / "eval-runs.jsonl"
    ledger.write_text("not JSON\n", encoding="utf-8")

    with pytest.raises(profile.ProfileError):
        profile.profile(ledger)
