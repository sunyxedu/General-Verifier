import json
import os
from pathlib import Path

import pytest

from experiments.terminal_bench import score
from verifier import offline


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_copy_captured_world_uses_successful_manifest_roots(tmp_path):
    trial = tmp_path / "trial"
    state = tmp_path / "state"
    _write(trial / "artifacts/app/result.txt", "answer")
    _write(trial / "artifacts/etc/nginx/nginx.conf", "config")
    _write(trial / "artifacts/var/log/nginx/error.log", "not captured")
    manifest = [
        {"source": "/logs/artifacts", "destination":
         "artifacts/logs/artifacts", "type": "directory", "status": "empty"},
        {"source": "/app", "destination": "artifacts/app",
         "type": "directory", "status": "ok"},
        {"source": "/etc", "destination": "artifacts/etc",
         "type": "directory", "status": "ok"},
        {"source": "/var/log", "destination": "artifacts/var/log",
         "type": "directory", "status": "failed"},
    ]
    _write(trial / "artifacts/manifest.json", json.dumps(manifest))

    captured = score._copy_captured_world(str(trial), str(state))

    assert captured == ["/app", "/etc"]
    assert (state / "world/app/result.txt").read_text() == "answer"
    assert (state / "world/etc/nginx/nginx.conf").read_text() == "config"
    assert not (state / "world/var").exists()


def test_copy_captured_world_rejects_destination_outside_artifacts(tmp_path):
    trial = tmp_path / "trial"
    state = tmp_path / "state"
    _write(trial / "outside/secret", "gold")
    manifest = [{
        "source": "/app",
        "destination": "outside",
        "type": "directory",
        "status": "ok",
    }]
    _write(trial / "artifacts/manifest.json", json.dumps(manifest))

    assert score._copy_captured_world(str(trial), str(state)) == []
    assert not (state / "world").exists()


def test_copy_captured_world_supports_legacy_app_only_jobs(tmp_path):
    trial = tmp_path / "trial"
    state = tmp_path / "state"
    _write(trial / "artifacts/app/result.txt", "legacy")

    assert score._copy_captured_world(str(trial), str(state)) == ["/app"]
    assert (state / "world/app/result.txt").read_text() == "legacy"


def test_copy_captured_world_preserves_successful_root_symlink(tmp_path):
    trial = tmp_path / "trial"
    state = tmp_path / "state"
    link = trial / "artifacts/etc/service/current"
    link.parent.mkdir(parents=True)
    link.symlink_to("/container-only/service-v1")
    manifest = [{
        "source": "/etc/service/current",
        "destination": "artifacts/etc/service/current",
        "type": "file",
        "status": "ok",
    }]
    _write(trial / "artifacts/manifest.json", json.dumps(manifest))

    assert score._copy_captured_world(str(trial), str(state)) == [
        "/etc/service/current"]
    copied = state / "world/etc/service/current"
    assert copied.is_symlink()
    assert os.readlink(copied) == "/container-only/service-v1"


def test_sanitized_state_publishes_anchor_before_completion_marker(tmp_path):
    trial_dir = tmp_path / "trial"
    source = trial_dir / "gv-state"
    _write(source / "hook_feed.jsonl", "")
    _write(source / "task.md", "do the task")
    _write(source / "final.txt", "done")
    _write(trial_dir / "artifacts/app/result.txt", "answer")
    trial = {
        "trial": "task__trial",
        "trial_dir": str(trial_dir),
        "agent_log": str(trial_dir / "agent.log"),
    }

    state = score.sanitized_state(
        trial, str(tmp_path / "scoring"), "job", "config/rep")

    assert os.path.isfile(os.path.join(state, score.SANITIZE_MARKER))
    assert offline.verify_immutable_input_anchor(state)

    with open(os.path.join(state, "final.txt"), "a") as handle:
        handle.write(" tampered")
    with pytest.raises(offline.ImmutableInputError, match="seal mismatch"):
        score.sanitized_state(
            trial, str(tmp_path / "scoring"), "job", "config/rep")


def test_usage_contexts_accepts_per_trial_kwargs(tmp_path):
    record = {
        "job": "job",
        "task": "task",
        "trial": "task__trial",
        "config": "config",
        "rep": "r1",
    }
    state = tmp_path / "gv"

    contexts = score._usage_contexts([
        (record, str(state), {"parallel_claims": 4}),
    ])

    assert contexts == {str(state.resolve()): record}
