"""Offline controller-exit monitoring: no Docker, evaluators, or model calls."""
import json
import threading
from types import SimpleNamespace

import pytest

from experiments.cybergym import run, tracking


@pytest.fixture
def evaluation(tmp_path):
    controller = object.__new__(run.Evaluation)
    controller.root = tmp_path
    controller.stop = threading.Event()
    controller.results = {}
    controller.frozen_commit = "fixture-commit"
    controller.pin = lambda: None
    controller.start_services = lambda: None
    controller.cleanup_services = lambda: None
    return controller


def no_periodic_tick(monkeypatch):
    monkeypatch.setattr(run.threading, "Thread",
        lambda **_kwargs: SimpleNamespace(start=lambda: None))


def terminal_attempt(controller, attempt_id, *, passed=None, infrastructure=False):
    directory = controller.root / "attempts" / attempt_id
    state = {"task_id": "synthetic:" + attempt_id, "attempt_id": attempt_id,
        "started_unix": 10, "completed_unix": 11, "updated_unix": 11,
        "stage": "infrastructure_error" if infrastructure else "completed",
        "outcome": "infrastructure_error" if infrastructure else "pass" if passed else "fail",
        "official_pass": None if infrastructure else passed}
    run.atomic_json(directory / "status.json", state)
    controller.results[state["task_id"]] = state
    return state


def read_latest(controller):
    return json.loads((controller.root / "monitor/latest.json").read_text())


def read_alerts(controller):
    path = controller.root / "monitor/alerts.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_short_pilot_failure_is_recorded_without_waiting_for_periodic_tick(evaluation, monkeypatch):
    no_periodic_tick(monkeypatch)
    tracking.record_snapshot(evaluation.root)  # Last observer saw no attempts.

    def batch(_tasks, stage):
        assert stage == "pilot5"
        terminal_attempt(evaluation, "pilot-negative", passed=False)
        return 4

    evaluation.batch = batch
    evaluation.run()
    value = read_latest(evaluation)
    assert value["completed_attempts"] == 1
    assert value["attempts"][0]["official_pass"] is False
    assert [alert["type"] for alert in read_alerts(evaluation)] == ["official_task_not_passed"]
    tracking.record_snapshot(evaluation.root)
    assert len(read_alerts(evaluation)) == 1  # Final recording uses existing deduplication.


def test_fixed30_gate_failure_keeps_terminal_latest_and_alert(evaluation, monkeypatch):
    no_periodic_tick(monkeypatch)
    stages = []

    def batch(_tasks, stage):
        stages.append(stage)
        terminal_attempt(evaluation, stage, passed=stage == "pilot5")
        return 5 if stage == "pilot5" else 25

    evaluation.batch = batch
    evaluation.run()
    assert stages == ["pilot5", "fixed30"]
    assert read_latest(evaluation)["completed_attempts"] == 2
    assert [alert["attempt_id"] for alert in read_alerts(evaluation)] == ["fixed30"]


def test_normal_finish_records_last_batch_terminal_status(evaluation, monkeypatch, tmp_path):
    no_periodic_tick(monkeypatch)
    official = tmp_path / "official-fixture"
    run.atomic_json(official / "mask_map.json", {"synthetic:all": "masked"})
    monkeypatch.setattr(run, "OFFICIAL", official)
    stages = []

    def batch(_tasks, stage):
        stages.append(stage)
        terminal_attempt(evaluation, stage, passed=True)
        return {"pilot5": 5, "fixed30": 26, "all1507": 1}[stage]

    evaluation.batch = batch
    evaluation.run()
    value = read_latest(evaluation)
    assert stages == ["pilot5", "fixed30", "all1507"]
    assert value["completed_attempts"] == value["official_passed_attempts"] == 3
    assert all(item["completed_unix"] is not None for item in value["attempts"])
    assert not read_alerts(evaluation)


def test_startup_error_still_cleans_and_records_final_snapshot(evaluation, monkeypatch):
    no_periodic_tick(monkeypatch)
    cleaned = []

    def fail_start():
        raise RuntimeError("synthetic startup failure")

    def cleanup():
        cleaned.append(True)
        terminal_attempt(evaluation, "startup-diagnostic", infrastructure=True)

    evaluation.start_services = fail_start
    evaluation.cleanup_services = cleanup
    evaluation.batch = lambda *_args: pytest.fail("startup failure must not launch a batch")
    with pytest.raises(RuntimeError, match="synthetic startup failure"):
        evaluation.run()
    assert cleaned == [True] and evaluation.stop.is_set()
    assert read_latest(evaluation)["completed_attempts"] == 1
    assert [alert["type"] for alert in read_alerts(evaluation)] == ["infrastructure_error"]
    assert evaluation.results["synthetic:startup-diagnostic"]["official_pass"] is None


def test_tracking_error_does_not_prevent_cleanup_or_change_result(evaluation, monkeypatch):
    no_periodic_tick(monkeypatch)
    state = terminal_attempt(evaluation, "official-negative", passed=False)
    before = json.dumps(evaluation.results, sort_keys=True)
    order = []
    evaluation.batch = lambda *_args: 4
    evaluation.cleanup_services = lambda: order.append("cleanup")

    def failed_tracking(_root):
        order.append("snapshot")
        raise OSError("PRIVATE_DIAGNOSTIC_SENTINEL")

    monkeypatch.setattr(tracking, "record_snapshot", failed_tracking)
    evaluation.run()
    assert order == ["cleanup", "snapshot"]
    assert json.dumps(evaluation.results, sort_keys=True) == before
    assert state["official_pass"] is False
    error = json.loads((evaluation.root / "monitor-error.json").read_text())
    assert error["reason"] == "tracking_failed" and error["type"] == "OSError"
    assert "PRIVATE_DIAGNOSTIC_SENTINEL" not in json.dumps(error)


def test_final_snapshot_is_attempted_even_if_cleanup_raises(evaluation, monkeypatch):
    no_periodic_tick(monkeypatch)
    terminal_attempt(evaluation, "finished", passed=True)
    evaluation.batch = lambda *_args: 4

    def failed_cleanup():
        raise RuntimeError("synthetic cleanup failure")

    evaluation.cleanup_services = failed_cleanup
    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        evaluation.run()
    assert read_latest(evaluation)["official_passed_attempts"] == 1


def test_periodic_and_final_writers_serialize_and_final_writer_runs_last(evaluation, monkeypatch):
    # These short guards bound this synthetic concurrency fixture only; they
    # do not add an execution deadline to the runner or any verification job.
    thread_class = threading.Thread
    entered, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    callback_lock = threading.Lock()
    active, peak, calls, errors = 0, 0, [], []
    real_record = tracking.record_snapshot

    def observed_record(root):
        nonlocal active, peak
        with callback_lock:
            active += 1
            peak = max(peak, active)
            index = len(calls)
            calls.append(threading.current_thread().name)
        try:
            if index == 0:
                entered.set()
                assert release.wait(2), "fixture failed to release periodic writer"
            return real_record(root)
        finally:
            with callback_lock:
                active -= 1

    monkeypatch.setattr(tracking, "record_snapshot", observed_record)

    def batch(_tasks, _stage):
        assert entered.wait(2), "fixture periodic writer did not start"
        terminal_attempt(evaluation, "late-negative", passed=False)
        return 4

    evaluation.batch = batch
    evaluation.cleanup_services = cleaned.set

    def run_controller():
        try:
            evaluation.run()
        except BaseException as error:
            errors.append(error)

    controller_thread = thread_class(target=run_controller, name="fixture-controller")
    controller_thread.start()
    try:
        assert cleaned.wait(2), "fixture cleanup did not run during metadata write"
        assert controller_thread.is_alive()  # Final writer is queued, not racing.
    finally:
        release.set()
        controller_thread.join(2)
    assert not controller_thread.is_alive() and not errors
    assert peak == 1 and len(calls) == 2 and calls[-1] == "fixture-controller"
    assert read_latest(evaluation)["attempts"][0]["official_pass"] is False
    assert len(read_alerts(evaluation)) == 1


def test_tracking_error_log_failure_does_not_replace_startup_error(evaluation, monkeypatch):
    no_periodic_tick(monkeypatch)
    cleaned = []
    evaluation.cleanup_services = lambda: cleaned.append(True)

    def fail_start():
        raise RuntimeError("original startup error")

    def failed_tracking(_root):
        raise OSError("tracking storage full")

    def failed_error_log(*_args, **_kwargs):
        raise OSError("diagnostic storage full")

    evaluation.start_services = fail_start
    monkeypatch.setattr(tracking, "record_snapshot", failed_tracking)
    monkeypatch.setattr(run, "atomic_json", failed_error_log)
    with pytest.raises(RuntimeError, match="original startup error"):
        evaluation.run()
    assert cleaned == [True]
