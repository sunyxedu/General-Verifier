"""Host operational job metadata contract; no evaluators, Docker, or models."""
import json

import pytest

from experiments.cybergym import tracking
from verifier.mcp_server import _JOB_VERBS
from verifier.service_jobs import VerificationJobs


def test_progress_inventory_matches_actual_mcp_durable_job_verbs():
    assert tracking.VERIFICATION_TOOLS == _JOB_VERBS


@pytest.mark.parametrize("tool", sorted(_JOB_VERBS))
@pytest.mark.parametrize("state", sorted(tracking.JOB_STATES))
def test_progress_accepts_all_real_durable_verbs_without_exposing_request_result(tool, state):
    raw = {"schema": tracking.PROGRESS_SCHEMA, "checked_unix": 100,
        "metadata_snapshot_unix": 95, "journal_available": True,
        "jobs": [{"job_id": "verification:" + "a" * 32, "tool": tool, "state": state,
            "created_unix": 90, "started_unix": 95, "updated_unix": 95,
            "attempts": 1, "reused_count": 0,
            "request": "PRIVATE_CANDIDATE_REQUEST", "result": "PRIVATE_EVALUATOR_RESULT"}]}
    projected = tracking.project_progress(raw)
    assert projected["jobs"][0]["tool"] == tool
    assert projected["jobs"][0]["state"] == state
    assert projected["active_job_ids"] == (["verification:" + "a" * 32]
                                           if state in {"queued", "running"} else [])
    assert "PRIVATE_" not in json.dumps(projected)
    assert "verified" not in projected["jobs"][0]  # Job completion is not a verdict.


@pytest.mark.parametrize("tool", ["cyber_register_candidate", "cyber_evaluate_candidate"])
def test_actual_completed_candidate_job_progress_is_accepted(tmp_path, tool):
    state = tmp_path / "state"
    state.mkdir()
    jobs = VerificationJobs(state)
    started = jobs.submit({"tool": tool, "args": ["PRIVATE_CANDIDATE_ARGUMENT"]},
        lambda _retry: {"verified": False, "result": "PRIVATE_RETAINED_RESULT"})
    completed = jobs.wait(started["job_id"], 1)
    assert completed["state"] == "completed"
    projected = tracking.project_progress(jobs.progress())
    assert projected["journal_available"] is True
    assert projected["jobs"][0]["tool"] == tool
    assert projected["jobs"][0]["state"] == "completed"
    assert not projected["active_job_ids"]
    assert "PRIVATE_" not in json.dumps(projected)


def test_unregistered_tool_is_still_rejected():
    raw = {"schema": tracking.PROGRESS_SCHEMA, "checked_unix": 100,
        "jobs": [{"job_id": "verification:" + "a" * 32,
                  "tool": "arbitrary_solver_tool", "state": "running"}]}
    with pytest.raises(ValueError, match="job identity"):
        tracking.project_progress(raw)


@pytest.mark.parametrize("tool", ["cyber_register_candidate", "cyber_evaluate_candidate"])
def test_attempt_snapshot_accepts_candidate_job_and_keeps_official_score_unknown(tmp_path, tool):
    attempt = tmp_path / "attempts/attempt-job-contract"
    private = attempt / "private"
    private.mkdir(parents=True)
    (attempt / "status.json").write_text(json.dumps({
        "task_id": "arvo:9471", "attempt_id": attempt.name,
        "stage": "solver_running", "started_unix": 90, "stage_started_unix": 90,
        "official_pass": None}))
    (private / "native-progress.json").write_text(json.dumps({
        "schema": tracking.PROGRESS_SCHEMA, "checked_unix": 99,
        "metadata_snapshot_unix": 98, "journal_available": True,
        "jobs": [{"job_id": "verification:" + "a" * 32,
            "tool": tool, "state": "running", "created_unix": 95,
            "started_unix": 98, "updated_unix": 98, "attempts": 1, "reused_count": 0}]}))
    value = tracking.snapshot(tmp_path, now=100)
    item = value["attempts"][0]
    assert item["verification_progress"]["available"] is True
    assert item["verification_progress"]["jobs"][0]["tool"] == tool
    assert item["verification_progress"]["age_seconds"] == 1
    assert item["official_pass"] is None
    assert not value["official_passed_attempts"]
