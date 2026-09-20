"""Registration jobs bind staging bytes, not merely their mutable path.

These are real Service/dispatcher tests with no model, evaluator or network.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from verifier.mcp_server import Service, _ToolDispatcher
from verifier.profiles.cyber.candidates import MAX_CANDIDATE_BYTES


@pytest.fixture
def service(tmp_path):
    project, state = tmp_path / "project", tmp_path / "state"
    project.mkdir()
    state.mkdir()
    (state / "task.md").write_text("Retain the exact proposed artifact.")
    return Service(str(state), project_root=str(project), profile="cyber",
                   kernel_factory=lambda _: SimpleNamespace())


def dispatch(service, path="staged", label="candidate"):
    return asyncio.run(asyncio.wait_for(
        _ToolDispatcher(service).call("cyber_register_candidate", path, label), 2))


def settle(service, handle):
    assert service._jobs._done[handle["job_id"]].wait(2), "registration did not settle"
    job = service.verification_job(handle["job_id"])
    assert job["state"] == "completed", job
    return job["result"]


def registrations(service):
    return [event for event in service.log.replay()
            if event["type"] == "cyber_candidate_registered"]


def assert_no_registration_writes(service):
    assert registrations(service) == []
    assert list(Path(service.candidates.blob_dir).iterdir()) == []
    assert list(Path(service.artifacts.root).glob("*.json")) == []


def test_identical_bytes_reconnect_but_in_place_mutation_gets_new_job(service):
    source = Path(service.project_root) / "staged"
    source.write_bytes(b"first proposal")
    first = dispatch(service)
    first_result = settle(service, first)
    same = dispatch(service)
    assert same["job_id"] == first["job_id"]
    assert settle(service, same) == first_result
    assert len(registrations(service)) == 1

    source.write_bytes(b"revised proposal")
    revised = dispatch(service)
    revised_result = settle(service, revised)
    assert revised["job_id"] != first["job_id"]
    assert revised_result["candidate_id"] == hashlib.sha256(b"revised proposal").hexdigest()
    assert revised_result["candidate_id"] != first_result["candidate_id"]
    assert Path(revised_result["blob_path"]).read_bytes() == b"revised proposal"
    assert len(registrations(service)) == 2


def test_pending_identical_registration_reconnects_to_serialized_worker(service):
    source = Path(service.project_root) / "staged"
    source.write_bytes(b"proposal")
    entered, release = threading.Event(), threading.Event()

    def block(_):
        entered.set()
        assert release.wait(2)
        return {}

    blocker = service._jobs.submit({"tool": "fixture", "args": [], "scope": {}}, block)
    try:
        assert entered.wait(1)
        first = dispatch(service)
        same = dispatch(service)
        assert same["job_id"] == first["job_id"]
        assert first["state"] == same["state"] == "queued"
        assert registrations(service) == []
    finally:
        release.set()
    assert service._jobs._done[blocker["job_id"]].wait(2)
    settle(service, first)
    assert len(registrations(service)) == 1


def test_staging_mutation_while_queued_rejected_before_registration(service):
    source = Path(service.project_root) / "staged"
    source.write_bytes(b"original")
    entered, release = threading.Event(), threading.Event()

    def block(_):
        entered.set()
        assert release.wait(2)
        return {}

    service._jobs.submit({"tool": "fixture", "args": [], "scope": {}}, block)
    try:
        assert entered.wait(1)
        queued = dispatch(service)
        assert queued["state"] == "queued"
        source.write_bytes(b"changed while queued")
    finally:
        release.set()
    rejected = settle(service, queued)
    assert "changed while queued" in rejected["error"]
    assert_no_registration_writes(service)


def test_mutation_after_request_recheck_is_rejected_before_any_write(service, monkeypatch):
    source = Path(service.project_root) / "staged"
    source.write_bytes(b"original")
    register = service._cyber_register_candidate

    def mutate_after_recheck(*args, **kwargs):
        source.write_bytes(b"changed after recheck")
        return register(*args, **kwargs)

    monkeypatch.setattr(service, "_cyber_register_candidate", mutate_after_recheck)
    rejected = settle(service, dispatch(service))
    assert "candidate source changed while queued" in rejected["error"]
    assert_no_registration_writes(service)


@pytest.mark.parametrize("kind", ["symlink", "parent_symlink", "fifo", "oversize"])
def test_unsafe_staging_source_fails_closed_before_job_admission(service, tmp_path, kind):
    source = Path(service.project_root) / "staged"
    if kind == "symlink":
        outside = tmp_path / "outside"
        outside.write_bytes(b"outside")
        source.symlink_to(outside)
    elif kind == "parent_symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "input").write_bytes(b"outside")
        source.symlink_to(outside, target_is_directory=True)
        source = source / "input"
    elif kind == "fifo":
        os.mkfifo(source)
    else:
        with source.open("wb") as stream:
            stream.truncate(MAX_CANDIDATE_BYTES + 1)
    rejected = dispatch(service, str(source))
    assert rejected.get("error")
    assert "job_id" not in rejected
    assert service.verification_jobs()["jobs"] == []
    assert_no_registration_writes(service)


def test_restart_reuses_completed_registration_with_identical_source(service):
    source = Path(service.project_root) / "staged"
    source.write_bytes(b"proposal")
    first = dispatch(service)
    first_result = settle(service, first)
    restarted = Service(service.state_dir, project_root=service.project_root, profile="cyber",
                        kernel_factory=lambda _: SimpleNamespace())
    recovered = dispatch(restarted)
    assert recovered["job_id"] == first["job_id"]
    assert settle(restarted, recovered) == first_result
    assert len(registrations(restarted)) == 1


def test_expected_identity_is_not_a_public_registration_argument():
    assert tuple(inspect.signature(Service.cyber_register_candidate).parameters) \
        == ("self", "path", "label")
