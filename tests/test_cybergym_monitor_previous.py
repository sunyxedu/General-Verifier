"""Large host monitor cache fixtures; no API, Docker, or real experiment data."""
import json
from pathlib import Path
import runpy
from unittest.mock import patch

import pytest

from experiments.cybergym import tracking


def fixture_attempt(root, index, *, active=False):
    attempt_id = f"gvroles-arvo-{100000 + index}-{index:016x}"
    path = root / "attempts" / attempt_id / "status.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"task_id": f"arvo:{100000 + index}",
        "attempt_id": attempt_id, "stage": "solver_running" if active else "completed",
        "started_unix": 10, "stage_started_unix": 10, "updated_unix": 20,
        "completed_unix": None if active else 20, "outcome": None if active else "fail",
        "official_pass": None if active else False, "phase_intervals": []}))
    return attempt_id


def latest_file(root, raw):
    path = root / "monitor/latest.json"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(raw)
    return path


def test_real_writer_over_one_mib_preserves_alert_dedup_cache_and_progress(tmp_path):
    for index in range(400):
        fixture_attempt(tmp_path, index)
    active_id = fixture_attempt(tmp_path, 400, active=True)
    first = tracking.record_snapshot(tmp_path, now=30)
    latest = tmp_path / "monitor/latest.json"
    assert latest.stat().st_size > tracking.MAX_JSON_BYTES
    assert tracking._json(latest, tmp_path) == {}  # Generic task/file cap is unchanged.
    previous, diagnostic = tracking.read_monitor_previous_snapshot(tmp_path)
    assert diagnostic["available"] is True and diagnostic["reason_code"] is None
    assert len(previous["attempts"]) == 401
    assert len(first["alerts"]) == 400
    with patch.object(tracking, "attempt_snapshot", wraps=tracking.attempt_snapshot) as observed:
        second = tracking.record_snapshot(tmp_path, now=330)
        assert observed.call_count == 1  # Only active metadata needs recomputation.
    third = tracking.record_snapshot(tmp_path, now=630)
    assert not second["alerts"]
    assert [row["type"] for row in third["alerts"]] == ["unchanged_progress_warning"]
    assert third["alerts"][0]["attempt_id"] == active_id
    assert len((tmp_path / "monitor/alerts.jsonl").read_text().splitlines()) == 401
    assert third["official_passed_attempts"] == 0  # Cache is never a success inference.


def test_oversized_previous_is_explicit_diagnostic_not_result_or_restart(tmp_path, monkeypatch):
    fixture_attempt(tmp_path, 0)
    tracking.record_snapshot(tmp_path, now=30)
    monkeypatch.setattr(tracking, "MAX_MONITOR_PREVIOUS_BYTES", 128)
    value = tracking.record_snapshot(tmp_path, now=330)
    diagnostic = value["monitor_previous_snapshot"]
    assert diagnostic["available"] is False and diagnostic["reason_code"] == "oversized"
    assert diagnostic["bytes_observed"] == 129 and diagnostic["limit_bytes"] == 128
    warning = next(row for row in value["alerts"]
                   if row["type"] == "monitor_previous_snapshot_oversized_warning")
    assert warning["available"] is False and warning["lifecycle_action"] is None
    assert warning["diagnostic_only"] is True
    assert value["attempts"][0]["official_pass"] is False
    assert value["attempts"][0]["outcome"] == "fail"
    assert tracking.MAX_JSON_BYTES == 1024 * 1024


@pytest.mark.parametrize("raw", [b"invalid-json", b"[]", b"{}",
    b'{"schema":"cybergym-tracking-v1","attempts":[null]}'])
def test_malformed_previous_fails_closed_without_retaining_payload(tmp_path, raw):
    latest_file(tmp_path, raw)
    previous, diagnostic = tracking.read_monitor_previous_snapshot(tmp_path)
    assert previous == {}
    assert diagnostic["available"] is False and diagnostic["reason_code"] == "malformed"
    assert "invalid-json" not in json.dumps(diagnostic)


def test_malformed_previous_writer_reports_static_warning(tmp_path):
    fixture_attempt(tmp_path, 0)
    latest_file(tmp_path, b"PRIVATE_MALFORMED_CONTENT")
    value = tracking.record_snapshot(tmp_path, now=30)
    warning = next(row for row in value["alerts"]
                   if row["type"] == "monitor_previous_snapshot_unavailable_warning")
    assert warning["reason_code"] == "malformed"
    assert "PRIVATE_" not in json.dumps(value["monitor_previous_snapshot"])
    assert "PRIVATE_" not in json.dumps(warning)


def test_previous_symlink_is_not_opened_or_promoted(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "monitor").mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text('{"PRIVATE_FAKE_SECRET":"never read"}')
    (root / "monitor/latest.json").symlink_to(outside)

    def forbidden_open(*_args, **_kwargs):
        pytest.fail("unsafe previous file was opened")

    monkeypatch.setattr(tracking.os, "open", forbidden_open)
    previous, diagnostic = tracking.read_monitor_previous_snapshot(root)
    assert previous == {} and diagnostic["reason_code"] == "unsafe"
    assert diagnostic["bytes_observed"] == 0


def test_previous_parent_symlink_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "latest.json").write_text('{"schema":"cybergym-tracking-v1","attempts":[]}')
    (root / "monitor").symlink_to(outside, target_is_directory=True)
    previous, diagnostic = tracking.read_monitor_previous_snapshot(root)
    assert previous == {} and diagnostic["reason_code"] == "unsafe"


def test_previous_fifo_is_rejected_without_blocking(tmp_path):
    path = tmp_path / "monitor/latest.json"
    path.parent.mkdir()
    tracking.os.mkfifo(path)
    previous, diagnostic = tracking.read_monitor_previous_snapshot(tmp_path)
    assert previous == {} and diagnostic["reason_code"] == "unsafe"


def test_read_growth_race_is_bounded_limit_plus_one(tmp_path, monkeypatch):
    raw = b'{"schema":"cybergym-tracking-v1","attempts":[]}'
    path = latest_file(tmp_path, raw)
    monkeypatch.setattr(tracking, "MAX_MONITOR_PREVIOUS_BYTES", 128)
    real_safe = tracking._safe_file

    def grow_after_safe_check(candidate, root):
        safe = real_safe(candidate, root)
        path.write_bytes(raw + b" " * 1024)
        return safe

    monkeypatch.setattr(tracking, "_safe_file", grow_after_safe_check)
    previous, diagnostic = tracking.read_monitor_previous_snapshot(tmp_path)
    assert previous == {} and diagnostic["reason_code"] == "oversized"
    assert diagnostic["bytes_observed"] == 129


def test_missing_previous_does_not_create_files_or_warning(tmp_path):
    previous, diagnostic = tracking.read_monitor_previous_snapshot(tmp_path)
    assert previous == {} and diagnostic["reason_code"] == "missing"
    assert not (tmp_path / "monitor").exists()
    value = tracking.record_snapshot(tmp_path, now=30)
    assert not value["alerts"]


def test_previous_at_exact_small_cap_is_accepted(tmp_path, monkeypatch):
    raw = b'{"schema":"cybergym-tracking-v1","attempts":[]}'
    latest_file(tmp_path, raw)
    monkeypatch.setattr(tracking, "MAX_MONITOR_PREVIOUS_BYTES", len(raw))
    previous, diagnostic = tracking.read_monitor_previous_snapshot(tmp_path)
    assert previous["attempts"] == [] and diagnostic["available"] is True


def test_operator_helper_uses_same_cap_readonly_and_projects_warning_safely(tmp_path):
    # The optional operator view lives outside the package; distributions need
    # not include it. This repository's real helper is validated when present.
    helper_path = Path(tracking.__file__).resolve().parents[3] / "cybergym-observe-eval-20260912.py"
    if not helper_path.is_file():
        pytest.skip("repository-local operator helper is not installed")
    helper = runpy.run_path(str(helper_path), run_name="offline_previous_reader_test")
    assert helper["MAX_PREVIOUS_BYTES"] == tracking.MAX_MONITOR_PREVIOUS_BYTES
    raw = b'{"schema":"cybergym-tracking-v1","attempts":[]}'
    path = latest_file(tmp_path, raw)
    before = (path.stat().st_mtime_ns, path.read_bytes())
    previous, diagnostic = helper["load_previous"](tmp_path, with_diagnostics=True)
    assert previous["attempts"] == [] and diagnostic["available"] is True
    assert before == (path.stat().st_mtime_ns, path.read_bytes())
    alert = helper["alert_projection"]({"type": "monitor_previous_snapshot_oversized_warning",
        "available": False, "reason_code": "oversized", "limit_bytes": 64,
        "bytes_observed": 65, "reason": "PRIVATE_RAW_CONTENT", "meaning": "PRIVATE_TEXT"})
    assert alert["available"] is False and alert["limit_bytes"] == 64
    assert "PRIVATE_" not in json.dumps(alert)
