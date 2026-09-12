"""Operational lifecycle tests; no models, evaluators, Docker or network."""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
import shutil
import threading
import time
from types import SimpleNamespace

import pytest

from verifier.core.model import content_hash
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.llm.schema import FixtureBackend
from verifier.mcp_server import Service, _ToolDispatcher
from verifier.service_jobs import VerificationJobs


def request(index=0):
    return {"tool": "commit", "args": [str(index), "assertion", "fixture", []],
            "scope": {"profile": "fixture", "revision": 1}}


def settled(jobs, job_id):
    assert jobs._done[job_id].wait(2), "fixture job did not settle"
    return jobs.status(job_id)


def test_repeated_pending_and_completed_request_run_once_and_keep_result(tmp_path):
    jobs = VerificationJobs(str(tmp_path))
    entered, release = threading.Event(), threading.Event()
    calls = []

    def operation(retry):
        calls.append(retry)
        entered.set()
        release.wait(2)
        return {"status": "verified", "unicode": "独立验证"}

    first = jobs.submit(request(), operation)
    try:
        assert entered.wait(1)
        for _ in range(20):
            duplicate = jobs.submit(request(), operation)
            assert duplicate["job_id"] == first["job_id"]
            assert duplicate["pending"] is True
        assert jobs.retry(first["job_id"])["attempts"] == 1
        progress = jobs.progress()
        assert "request" not in progress["jobs"][0] and "result" not in progress["jobs"][0]
        assert progress["jobs"][0]["state"] == "running"
    finally:
        release.set()
    result = settled(jobs, first["job_id"])
    result["result"]["status"] = "tampered client copy"
    assert jobs.status(first["job_id"])["result"]["status"] == "verified"
    again = jobs.submit(request(), lambda _: pytest.fail("duplicate restarted"))
    assert again["result_is_retained"] is True and again["verified"] is False
    assert again["job_id"] == first["job_id"] and calls == [False]


def test_requests_are_frozen_and_scopes_are_not_collapsed(tmp_path):
    jobs = VerificationJobs(str(tmp_path))
    original = request()
    job = jobs.submit(original, lambda _: {"ok": True})
    original["args"][3].append("caller mutated dependency")
    assert jobs.request(job["job_id"])["args"][3] == []
    changed = request()
    changed["scope"]["revision"] = 2
    other = jobs.submit(changed, lambda _: {"ok": True})
    assert other["job_id"] != job["job_id"]
    settled(jobs, other["job_id"])


def test_one_worker_serializes_claim_mutations_but_status_is_independent(tmp_path):
    jobs = VerificationJobs(str(tmp_path))
    entered, release = threading.Event(), threading.Event()
    order = []

    def first_operation(_):
        order.append("first-start")
        entered.set()
        release.wait(2)
        order.append("first-end")
        return {"status": "unverified"}

    first = jobs.submit(request(), first_operation)
    try:
        assert entered.wait(1)
        second = jobs.submit(request(1), lambda _: order.append("second") or {"status": "unverified"})
        assert jobs.status(second["job_id"])["state"] == "queued"
        assert jobs.status(first["job_id"])["state"] == "running"
        assert order == ["first-start"]
    finally:
        release.set()
    settled(jobs, second["job_id"])
    assert order == ["first-start", "first-end", "second"]


def test_client_wait_cancellation_does_not_cancel_or_lose_worker(tmp_path):
    jobs = VerificationJobs(str(tmp_path))
    entered, release = threading.Event(), threading.Event()

    def operation(_):
        entered.set()
        release.wait(2)
        return {"status": "verified"}

    handle = jobs.submit(request(), operation)

    async def scenario():
        assert entered.wait(1)
        waiter = asyncio.create_task(asyncio.to_thread(jobs.wait, handle["job_id"], 1))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert jobs.status(handle["job_id"])["state"] == "running"
        release.set()
        assert await asyncio.to_thread(jobs._done[handle["job_id"]].wait, 2)

    try:
        asyncio.run(scenario())
    finally:
        release.set()
    assert jobs.status(handle["job_id"])["result"]["status"] == "verified"


def test_explicit_retry_is_distinct_from_reconnection_and_retains_id(tmp_path):
    jobs = VerificationJobs(str(tmp_path))
    attempts = []

    def operation(retry):
        attempts.append(retry)
        if not retry:
            raise RuntimeError("must not be exposed in progress")
        return {"status": "verified"}

    first = jobs.submit(request(), operation)
    failed = settled(jobs, first["job_id"])
    assert failed["state"] == "failed" and failed["error_type"] == "RuntimeError"
    assert "must not" not in json.dumps(failed)
    assert jobs.submit(request(), operation)["state"] == "failed"
    jobs.retry(first["job_id"])
    success = settled(jobs, first["job_id"])
    assert success["attempts"] == 2 and attempts == [False, True]
    assert jobs.retry(first["job_id"])["error"] == "verified job is not retryable"


def test_retry_after_certificate_delivery_failure_is_not_forbidden(tmp_path):
    jobs = VerificationJobs(str(tmp_path))
    req = request()
    req["tool"] = "cyber_finalize_candidate"
    first = jobs.submit(req, lambda retry: {"verdict": "verified", "materialized": retry})
    assert settled(jobs, first["job_id"])["result"]["materialized"] is False
    jobs.retry(first["job_id"])
    assert settled(jobs, first["job_id"])["result"]["materialized"] is True


def test_contextvars_survive_background_worker(tmp_path):
    value = contextvars.ContextVar("job_usage", default=None)
    jobs = VerificationJobs(str(tmp_path))
    token = value.set("fixture-accounting")
    try:
        first = jobs.submit(request(), lambda _: {"usage_context": value.get()})
    finally:
        value.reset(token)
    assert settled(jobs, first["job_id"])["result"]["usage_context"] == "fixture-accounting"


def test_completed_result_survives_restart_without_automatic_reexecution(tmp_path):
    jobs = VerificationJobs(str(tmp_path))
    first = jobs.submit(request(), lambda _: {"status": "unverified", "value": 17})
    settled(jobs, first["job_id"])
    restarted = VerificationJobs(str(tmp_path))
    assert restarted._worker is None
    duplicate = restarted.submit(request(), lambda _: pytest.fail("restarted completed work"))
    assert duplicate["job_id"] == first["job_id"]
    assert duplicate["result"]["value"] == 17


def test_interrupted_job_is_visible_and_recovers_only_on_explicit_submission(tmp_path):
    original_dir, restarted_dir = tmp_path / "old", tmp_path / "new"
    original_dir.mkdir()
    restarted_dir.mkdir()
    jobs = VerificationJobs(str(original_dir))
    entered, release = threading.Event(), threading.Event()
    first = jobs.submit(request(), lambda _: entered.set() or release.wait(2) or {})
    try:
        assert entered.wait(1)
        shutil.copy2(jobs.path, restarted_dir / "verification_jobs.jsonl")
        restarted = VerificationJobs(str(restarted_dir))
        assert restarted.status(first["job_id"])["state"] == "interrupted"
        assert restarted._worker is None
        recovered = restarted.submit(request(), lambda _: {"status": "unverified", "recovered": True})
        assert recovered["job_id"] == first["job_id"]
        assert settled(restarted, recovered["job_id"])["attempts"] == 2
    finally:
        release.set()
        settled(jobs, first["job_id"])


@pytest.mark.parametrize("kind", ["symlink", "nonprivate", "hardlink", "modified"])
def test_journal_rejects_unsafe_or_modified_authority(tmp_path, kind):
    jobs = VerificationJobs(str(tmp_path))
    first = jobs.submit(request(), lambda _: {"status": "unverified"})
    settled(jobs, first["job_id"])
    path = tmp_path / "verification_jobs.jsonl"
    if kind == "symlink":
        moved = tmp_path / "real-journal"
        path.rename(moved)
        path.symlink_to(moved)
    elif kind == "nonprivate":
        path.chmod(0o644)
    elif kind == "hardlink":
        os.link(path, tmp_path / "other-journal")
    else:
        records = path.read_text().splitlines()
        data = json.loads(records[-1])
        data["job"]["result"]["status"] = "verified"
        records[-1] = json.dumps(data)
        path.write_text("\n".join(records) + "\n")
    with pytest.raises((OSError, ValueError)):
        VerificationJobs(str(tmp_path))


def test_queued_request_is_not_executed_after_its_scope_changes(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text("Establish the fixture condition.")
    service = Service(str(state), profile="cyber", project_root=str(tmp_path),
                      kernel_factory=lambda _: SimpleNamespace())
    submitted = service._job_request("verify", ["fixture"])
    (state / "task.md").write_text("Changed authoritative task.")
    with pytest.raises(ValueError, match="task changed"):
        service._job_operation(submitted)(False)


def test_enqueue_storage_failure_never_publishes_an_unrunnable_pending_job(tmp_path, monkeypatch):
    jobs = VerificationJobs(str(tmp_path))
    def fail(_):
        raise OSError("fixture disk unavailable")
    monkeypatch.setattr(jobs, "_persist", fail)
    with pytest.raises(OSError):
        jobs.submit(request(), lambda _: pytest.fail("unretained operation ran"))
    assert jobs.progress()["jobs"] == []
    assert jobs._queue.empty() and jobs._worker is None


def test_failed_retry_retention_preserves_previous_completed_result(tmp_path, monkeypatch):
    jobs = VerificationJobs(str(tmp_path))
    first = jobs.submit(request(), lambda _: {"status": "unverified", "retained": 7})
    settled(jobs, first["job_id"])
    def fail(_):
        raise OSError("fixture retry retention failed")
    monkeypatch.setattr(jobs, "_persist", fail)
    with pytest.raises(OSError):
        jobs.retry(first["job_id"])
    prior = jobs.status(first["job_id"])
    assert prior["state"] == "completed" and prior["result"]["retained"] == 7
    assert prior["attempts"] == 1 and jobs._done[first["job_id"]].is_set()


def test_partial_write_poison_stops_subsequent_job_admission(tmp_path, monkeypatch):
    jobs = VerificationJobs(str(tmp_path))
    original_write = os.write
    writes = []
    def partial_then_fail(fd, data):
        writes.append(len(data))
        if len(writes) == 1:
            return original_write(fd, data[:11])
        raise OSError("fixture partial write")
    monkeypatch.setattr(os, "write", partial_then_fail)
    with pytest.raises(OSError, match="journal unavailable"):
        jobs.submit(request(), lambda _: pytest.fail("unretained job ran"))
    assert jobs.progress()["journal_available"] is False
    assert jobs.progress()["jobs"] == []
    before = len(writes)
    with pytest.raises(OSError, match="journal unavailable"):
        jobs.submit(request(2), lambda _: {})
    assert len(writes) == before


def test_run_transition_retention_failure_does_not_kill_worker_or_execute_request(tmp_path, monkeypatch):
    jobs = VerificationJobs(str(tmp_path))
    persist = jobs._persist
    def fail_running(job):
        if job["state"] in ("running", "failed"):
            raise ValueError("fixture corrupt journal")
        persist(job)
    monkeypatch.setattr(jobs, "_persist", fail_running)
    first = jobs.submit(request(), lambda _: pytest.fail("unretained running request executed"))
    failed = settled(jobs, first["job_id"])
    assert failed["state"] == "failed" and "result" not in failed
    assert jobs._worker.is_alive()


def test_progress_and_job_status_do_not_wait_for_blocked_journal_fsync(tmp_path, monkeypatch):
    jobs = VerificationJobs(str(tmp_path))
    first = jobs.submit(request(), lambda _: {"status": "unverified"})
    settled(jobs, first["job_id"])
    entered, release = threading.Event(), threading.Event()
    persist = jobs._persist
    def stalled(job):
        entered.set()
        release.wait(2)
        persist(job)
    monkeypatch.setattr(jobs, "_persist", stalled)
    submitting = threading.Thread(target=lambda: jobs.submit(request(), lambda _: {}), daemon=True)
    submitting.start()
    try:
        assert entered.wait(1)
        status = jobs.status(first["job_id"])
        progress = jobs.progress()
        assert status["state"] == "completed" and progress["jobs"][0]["reused_count"] == 0
        assert not release.is_set()
    finally:
        release.set()
        submitting.join(2)
    assert jobs.status(first["job_id"])["reused_count"] == 1


def test_actual_service_long_commit_returns_handle_and_never_blocks_status(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text("Establish the fixture security condition.")
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Discharger:
        def discharge(self, row, obligation, method=None, tag=""):
            calls.append(row["id"])
            entered.set()
            release.wait(3)
            return DischargeResult(outcome="holds", witness_kind="execution",
                witness="fixture observation", artifact="", instrument="fixture",
                basis="world", strength="decisive")

    def factory(service):
        backend = FixtureBackend({"obligations": {"obligations": [{
            "text": "the independent fixture security condition holds", "required_basis": "world"}]}})
        return Kernel(service.store, service.log, backend, service.artifacts,
                      critic_role=None, discharger=Discharger())

    service = Service(str(state), profile="cyber", project_root=str(tmp_path), kernel_factory=factory)
    dispatcher = _ToolDispatcher(service)
    args = ["fixture", "final", "the fixture condition holds", [], None]

    async def scenario():
        first = await asyncio.wait_for(dispatcher.call("commit", *args), 0.8)
        assert first["pending"] and first["job_state"] == "running"
        assert entered.is_set() and not release.is_set()
        status = await asyncio.wait_for(dispatcher.call("status", None), 0.5)
        assert status["status_snapshot"] == "unavailable_while_verifying"
        duplicate = await asyncio.wait_for(dispatcher.call("commit", *args), 0.8)
        assert duplicate["job_id"] == first["job_id"] and len(calls) == 1
        # This is not a failed or retried verification merely because the
        # client no longer owns a waiting cell.
        release.set()
        assert await asyncio.to_thread(service._jobs._done[first["job_id"]].wait, 2)
        result = await dispatcher.call("verification_job", first["job_id"])
        assert result["result"]["status"] == "verified" and result["attempts"] == 1
        assert len([event for event in service.log.replay() if event["type"] == "commit"]) == 1

    try:
        asyncio.run(scenario())
    finally:
        release.set()
