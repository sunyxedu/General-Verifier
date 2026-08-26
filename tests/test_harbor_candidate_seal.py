"""Harbor rec-codex discovery and candidate-freezing tests."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.common import candidate_seal
from experiments.common import harbor_candidate_seal as harbor_seal


DATASET_DIGEST = "1" * 64
PACKAGE_DIGEST = "2" * 64
TASK_DIGEST = "3" * 64


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def harbor_job(tmp_path: Path) -> tuple[Path, Path]:
    job = tmp_path / "job"
    trial = job / "generic-task__trial1"
    agent = {
        "name": "codex",
        "model_name": "gpt-test",
        "kwargs": {"reasoning_effort": "high"},
    }
    task_config = {
        "name": "org/generic-task",
        "ref": f"sha256:{PACKAGE_DIGEST}",
        "source": "org/dataset",
    }
    trial_config = {
        "task": task_config,
        "trial_name": trial.name,
        "agent": agent,
        "artifacts": ["/app", "/failed"],
    }
    trial_lock = {
        "schema_version": 1,
        "task": {
            "name": "org/generic-task",
            "type": "package",
            "digest": f"sha256:{PACKAGE_DIGEST}",
            "source": "org/dataset",
        },
        "agent": agent,
        "artifacts": ["/app", "/failed"],
    }
    job_config = {
        "job_name": "job",
        "agents": [agent],
        "datasets": [{
            "name": "org/dataset",
            "ref": f"sha256:{DATASET_DIGEST}",
            "task_names": ["org/generic-task"],
        }],
        "artifacts": ["/app", "/failed"],
    }
    result = {
        "task_name": "org/generic-task",
        "trial_name": trial.name,
        "task_id": {
            "org": "org",
            "name": "generic-task",
            "ref": f"sha256:{PACKAGE_DIGEST}",
        },
        "source": "org/dataset",
        "task_checksum": TASK_DIGEST,
        "config": {
            "task": {
                "name": "org/generic-task",
                "ref": f"sha256:{PACKAGE_DIGEST}",
                "source": "org/dataset",
            },
            "trial_name": trial.name,
            "agent": agent,
        },
        "agent_info": {
            "name": "codex",
            "model_info": {"name": "gpt-test"},
        },
        "verifier_result": {"rewards": {"reward": 1.0}},
        "exception_info": None,
        "finished_at": "2026-01-01T00:03:00Z",
        "agent_execution": {"finished_at": "2026-01-01T00:02:00Z"},
        "verifier": {"finished_at": "2026-01-01T00:02:30Z"},
    }
    _write_json(job / "config.json", job_config)
    _write_json(job / "lock.json", {"schema_version": 2, "trials": [trial_lock]})
    _write_json(trial / "config.json", trial_config)
    _write_json(trial / "lock.json", trial_lock)
    _write_json(trial / "result.json", result)
    _write_json(trial / "agent/trajectory.json", {
        "schema_version": "ATIF-v1",
        "steps": [{"source": "agent", "message": "done"}],
    })
    (trial / "artifacts/app").mkdir(parents=True)
    (trial / "artifacts/app/answer.txt").write_text("answer\n", encoding="utf-8")
    (trial / "artifacts/failed").mkdir()
    (trial / "artifacts/failed/partial.txt").write_text(
        "partial bytes\n", encoding="utf-8")
    _write_json(trial / "artifacts/manifest.json", [
        {
            "source": "/app",
            "destination": "artifacts/app",
            "type": "directory",
            "status": "ok",
        },
        {
            "source": "/failed",
            "destination": "artifacts/failed",
            "type": "directory",
            "status": "failed",
        },
    ])
    return job, trial


def _outputs(tmp_path: Path) -> dict[str, Path]:
    return {
        "specs_path": tmp_path / "out/specs.jsonl",
        "seals_path": tmp_path / "out/seals.jsonl",
        "verifier_inputs_path": tmp_path / "out/views.jsonl",
    }


def _replace_agent(job: Path, agent: dict, *, job_only: bool = False) -> None:
    """Update fixture copies without normalizing Harbor's selector fields."""
    job_config_path = job / "config.json"
    job_config = json.loads(job_config_path.read_text(encoding="utf-8"))
    job_config["agents"] = [agent]
    _write_json(job_config_path, job_config)
    if job_only:
        return

    trial = job / "generic-task__trial1"
    for path, location in (
        (job / "lock.json", "job_lock"),
        (trial / "config.json", "trial_config"),
        (trial / "lock.json", "trial_lock"),
        (trial / "result.json", "result"),
    ):
        value = json.loads(path.read_text(encoding="utf-8"))
        if location == "job_lock":
            value["trials"][0]["agent"] = agent
        elif location == "result":
            value["config"]["agent"] = agent
        else:
            value["agent"] = agent
        _write_json(path, value)


def _finish_fixture_trajectory(trial: Path) -> None:
    trajectory_path = trial / "agent/trajectory.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory["session_id"] = "session-1"
    trajectory["final_metrics"] = {"total_steps": len(trajectory["steps"])}
    _write_json(trajectory_path, trajectory)


def _add_raw_exception_rollout(
    trial: Path, *, session_id: str = "session-1",
    codex_error_info: str = "cyber_policy",
    message: str = "Request blocked by the Cyber Verification Program",
) -> Path:
    trajectory_path = trial / "agent/trajectory.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory["session_id"] = session_id
    _write_json(trajectory_path, trajectory)
    rollout = trial / "agent/sessions/2026/01/01/rollout-test.jsonl"
    rollout.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {"type": "session_meta", "payload": {"id": session_id}},
        {"type": "event_msg", "payload": {
            "type": "task_started", "turn_id": "turn-1"}},
        {"type": "event_msg", "payload": {
            "type": "task_complete",
            "turn_id": "turn-1",
            "error": {
                "message": message,
                "codex_error_info": codex_error_info,
            },
        }},
    ]
    rollout.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return rollout


def _add_runtime_snapshot(
    trial: Path, *, collector_runtime: str | None = None,
    returncode: bytes = b"0\n",
) -> Path:
    root = trial / "artifacts/gv-runtime-snapshot"
    root.mkdir(parents=True)
    manifest = {
        "schema": harbor_seal.RUNTIME_SNAPSHOT_SCHEMA,
        "status": "complete",
        "capture_boundary": harbor_seal.RUNTIME_SNAPSHOT_BOUNDARY,
        "privacy": {"process_environments_captured": False},
    }
    if collector_runtime is None:
        collector = root / "collector.py"
        collector.write_text("# frozen collector\n", encoding="utf-8")
        manifest["collector"] = {
            "path": "collector.py",
            "bytes": collector.stat().st_size,
            "sha256": candidate_seal.file_digest(collector),
        }
    else:
        manifest["collector_runtime"] = collector_runtime
        (root / "collector.sh").write_text(
            "#!/bin/sh\n", encoding="utf-8")
    _write_json(root / "manifest.json", manifest)
    (root / "collector.returncode").write_bytes(returncode)
    outer_path = trial / "artifacts/manifest.json"
    outer = json.loads(outer_path.read_text(encoding="utf-8"))
    outer.append({
        "source": harbor_seal.RUNTIME_SNAPSHOT_SOURCE,
        "destination": "artifacts/gv-runtime-snapshot",
        "type": "directory",
        "status": "ok",
    })
    _write_json(outer_path, outer)
    return root


def _clone_trial(job: Path, new_name: str) -> Path:
    source = job / "generic-task__trial1"
    target = job / new_name
    shutil.copytree(source, target)
    for relative in ("config.json", "result.json"):
        path = target / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        rendered = json.dumps(value).replace(source.name, new_name)
        _write_json(path, json.loads(rendered))
    job_lock_path = job / "lock.json"
    job_lock = json.loads(job_lock_path.read_text(encoding="utf-8"))
    job_lock["trials"].append(
        json.loads((target / "lock.json").read_text(encoding="utf-8")))
    _write_json(job_lock_path, job_lock)
    return target


def test_freeze_job_binds_harbor_identity_and_keeps_reward_private(tmp_path):
    job, trial = harbor_job(tmp_path)
    outputs = _outputs(tmp_path)

    first = harbor_seal.freeze_job(job, **outputs)
    second = harbor_seal.freeze_job(job, **outputs)

    assert first["added"] == first["frozen"] == 1
    assert second["added"] == 0
    spec = candidate_seal.load_jsonl(outputs["specs_path"])[0]
    seal = candidate_seal.load_jsonl(outputs["seals_path"])[0]
    view = candidate_seal.load_jsonl(outputs["verifier_inputs_path"])[0]
    assert spec["task_digest"] == TASK_DIGEST
    assert spec["task_package_digest"] == PACKAGE_DIGEST
    assert spec["model"] == "gpt-test"
    assert spec["effort"] == "high"
    assert spec["service_tier"] == "standard"
    assert spec["trajectory_path"] == str(trial / "agent/trajectory.json")
    assert seal["identity"]["dataset"]["sha256"] == DATASET_DIGEST
    assert seal["identity"]["task"]["sha256"] == TASK_DIGEST
    assert seal["identity"]["task"]["package_sha256"] == PACKAGE_DIGEST
    assert seal["identity"]["task"]["config_sha256"] \
        == candidate_seal.file_digest(trial / "config.json")
    assert seal["identity"]["world"]["status"] == "captured"
    assert seal["identity"]["world"]["manifest_sha256"] \
        == candidate_seal.file_digest(trial / "artifacts/manifest.json")
    assert seal["evaluation"] == {"reward": 1.0}
    assert "reward" not in json.dumps(view)
    assert "evaluation" not in view


def test_failed_partial_artifact_bytes_never_enter_world_identity(tmp_path):
    job, trial = harbor_job(tmp_path)
    first = harbor_seal.discover_completed(job)[0][0].seal

    (trial / "artifacts/failed/partial.txt").write_text(
        "different failed bytes\n", encoding="utf-8")
    failed_edit = harbor_seal.discover_completed(job)[0][0].seal
    assert failed_edit["candidate_id"] == first["candidate_id"]
    assert failed_edit["identity"]["world"] == first["identity"]["world"]

    (trial / "artifacts/app/answer.txt").write_text(
        "different answer\n", encoding="utf-8")
    successful_edit = harbor_seal.discover_completed(job)[0][0].seal
    assert successful_edit["candidate_id"] != first["candidate_id"]


def test_successful_root_symlink_is_sealed_without_following(tmp_path):
    job, trial = harbor_job(tmp_path)
    link = trial / "artifacts/etc/service/current"
    link.parent.mkdir(parents=True)
    link.symlink_to("/container-only/service-v1")
    manifest_path = trial / "artifacts/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.append({
        "source": "/etc/service/current",
        "destination": "artifacts/etc/service/current",
        "type": "file",
        "status": "ok",
    })
    _write_json(manifest_path, manifest)

    first = harbor_seal.discover_completed(job)[0][0]
    assert "/etc/service/current" in first.spec["captured_roots"]

    link.unlink()
    link.symlink_to("/container-only/service-v2")
    second = harbor_seal.discover_completed(job)[0][0]
    assert second.seal["candidate_id"] != first.seal["candidate_id"]


def test_status_ok_destination_with_escaping_symlink_parent_is_rejected(
    tmp_path,
):
    job, trial = harbor_job(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not an artifact\n", encoding="utf-8")
    (trial / "artifacts/escape").symlink_to(outside, target_is_directory=True)
    manifest_path = trial / "artifacts/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.append({
        "source": "/secret.txt",
        "destination": "artifacts/escape/secret.txt",
        "type": "file",
        "status": "ok",
    })
    _write_json(manifest_path, manifest)

    with pytest.raises(candidate_seal.SealError, match="status=ok artifact"):
        harbor_seal.discover_completed(job)


def test_incremental_freeze_skips_running_trial_and_rejects_claimed_complete(
    tmp_path,
):
    job, _ = harbor_job(tmp_path)
    running = job / "other__running"
    _write_json(running / "config.json", {})
    summary = harbor_seal.freeze_job(job, **_outputs(tmp_path))
    assert summary["skipped"] == {"other__running": "incomplete"}

    trial = job / "broken__finished"
    source = job / "generic-task__trial1"
    for name in ("config.json", "lock.json", "result.json"):
        value = json.loads((source / name).read_text(encoding="utf-8"))
        rendered = json.dumps(value).replace(
            "generic-task__trial1", "broken__finished")
        _write_json(trial / name, json.loads(rendered))
    with pytest.raises(candidate_seal.SealError, match="trajectory"):
        harbor_seal.discover_completed(job)


def test_frozen_source_skips_payload_rediscovery_after_metadata_match(tmp_path):
    job, trial = harbor_job(tmp_path)
    outputs = _outputs(tmp_path)
    first = harbor_seal.freeze_job(job, **outputs)
    (trial / "artifacts/app/answer.txt").write_text(
        "mutated after freeze\n", encoding="utf-8")

    second = harbor_seal.freeze_job(job, **outputs)

    assert first["frozen"] == second["frozen"] == 1
    assert second["reused"] == 1
    assert second["discovered"] == second["added"] == 0


def test_cross_checks_model_effort_and_package_digest(tmp_path):
    job, trial = harbor_job(tmp_path)
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["config"]["agent"]["kwargs"]["reasoning_effort"] = "low"
    _write_json(result_path, result)
    with pytest.raises(candidate_seal.SealError, match="reasoning effort disagrees"):
        harbor_seal.discover_completed(job)


def test_tiered_codex_selector_is_allowlisted_but_runtime_name_stays_logical(
    tmp_path,
):
    job, _ = harbor_job(tmp_path)
    custom = {
        "name": "codex_agent:TieredCodex",
        "model_name": "gpt-test",
        "kwargs": {
            "reasoning_effort": "high",
            "service_tier": "fast",
            "fast_mode": "enable",
        },
    }
    _replace_agent(job, custom)
    _add_runtime_snapshot(job / "generic-task__trial1")

    candidate = harbor_seal.discover_completed(job)[0][0]
    assert candidate.spec["agent_implementation"] == \
        "codex_agent:TieredCodex"
    assert candidate.spec["agent_config_selector"] == {
        "name": "codex_agent:TieredCodex", "import_path": None}
    assert candidate.spec["service_tier"] == "fast"
    assert candidate.seal["identity"]["solver"]["service_tier"] == "fast"

    result_path = job / "generic-task__trial1/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["agent_info"]["name"] = "TieredCodex"
    _write_json(result_path, result)
    with pytest.raises(candidate_seal.SealError, match="logical runtime agent codex"):
        harbor_seal.discover_completed(job)


def test_import_path_form_is_accepted_and_unallowlisted_subclass_is_rejected(
    tmp_path,
):
    job, _ = harbor_job(tmp_path)
    custom = {
        "name": None,
        "import_path": "experiments.terminal_bench.codex_agent:TieredCodex",
        "model_name": "gpt-test",
        "kwargs": {"reasoning_effort": "high"},
    }
    _replace_agent(job, custom)
    _add_runtime_snapshot(job / "generic-task__trial1")
    assert harbor_seal.discover_completed(job)[0][0].spec[
        "agent_implementation"] == custom["import_path"]

    custom["import_path"] = "third_party:DefinitelyCodex"
    _replace_agent(job, custom)
    with pytest.raises(candidate_seal.SealError, match="allowlisted"):
        harbor_seal.discover_completed(job)


def test_historical_builtin_trial_is_not_rewritten_when_job_resumes_custom(
    tmp_path,
):
    job, _ = harbor_job(tmp_path)
    custom_job_agent = {
        "name": "codex_agent:TieredCodex",
        "model_name": "gpt-test",
        "kwargs": {
            "reasoning_effort": "high",
            "service_tier": "default",
            "fast_mode": "disable",
        },
    }
    _replace_agent(job, custom_job_agent, job_only=True)

    candidate = harbor_seal.discover_completed(job)[0][0]
    assert "agent_implementation" not in candidate.spec
    assert json.loads((job / "generic-task__trial1/config.json").read_text())[
        "agent"]["name"] == "codex"


@pytest.mark.parametrize(
    ("service_tier", "fast_mode"),
    [
        ("fast", None),
        (None, "enable"),
        ("default", "enable"),
        ("fast", "disable"),
        ("standard", "disable"),
    ],
)
def test_inconsistent_solver_service_tier_is_rejected(
    tmp_path, service_tier, fast_mode,
):
    job, _ = harbor_job(tmp_path)
    kwargs = {"reasoning_effort": "high"}
    if service_tier is not None:
        kwargs["service_tier"] = service_tier
    if fast_mode is not None:
        kwargs["fast_mode"] = fast_mode
    _replace_agent(job, {
        "name": "codex_agent:TieredCodex",
        "model_name": "gpt-test",
        "kwargs": kwargs,
    })

    with pytest.raises(candidate_seal.SealError, match="inconsistent"):
        harbor_seal.discover_completed(job)


def test_explicit_standard_and_legacy_share_pool_but_fast_splits_it(tmp_path):
    legacy_job, _ = harbor_job(tmp_path / "legacy")
    legacy = harbor_seal.discover_completed(legacy_job)[0][0].seal

    standard_job, _ = harbor_job(tmp_path / "standard")
    _replace_agent(standard_job, {
        "name": "codex_agent:TieredCodex",
        "model_name": "gpt-test",
        "kwargs": {
            "reasoning_effort": "high",
            "service_tier": "default",
            "fast_mode": "disable",
        },
    })
    _add_runtime_snapshot(standard_job / "generic-task__trial1")
    standard = harbor_seal.discover_completed(standard_job)[0][0].seal

    fast_job, _ = harbor_job(tmp_path / "fast")
    _replace_agent(fast_job, {
        "name": "codex_agent:TieredCodex",
        "model_name": "gpt-test",
        "kwargs": {
            "reasoning_effort": "high",
            "service_tier": "fast",
            "fast_mode": "enable",
        },
    })
    _add_runtime_snapshot(
        fast_job / "generic-task__trial1",
        collector_runtime="posix-sh-fallback",
    )
    fast = harbor_seal.discover_completed(fast_job)[0][0].seal

    assert standard["pool_id"] == legacy["pool_id"]
    assert fast["pool_id"] != legacy["pool_id"]


def test_builtin_codex_cannot_claim_ignored_fast_kwargs(tmp_path):
    job, _ = harbor_job(tmp_path)
    _replace_agent(job, {
        "name": "codex",
        "model_name": "gpt-test",
        "kwargs": {
            "reasoning_effort": "high",
            "service_tier": "fast",
            "fast_mode": "enable",
        },
    })

    with pytest.raises(candidate_seal.SealError, match="requires.*TieredCodex"):
        harbor_seal.discover_completed(job)


def test_tiered_codex_requires_valid_post_run_runtime_capture(tmp_path):
    job, trial = harbor_job(tmp_path)
    _replace_agent(job, {
        "name": "codex_agent:TieredCodex",
        "model_name": "gpt-test",
        "kwargs": {
            "reasoning_effort": "high",
            "service_tier": "default",
            "fast_mode": "disable",
        },
    })

    with pytest.raises(candidate_seal.SealError, match="gv-runtime-snapshot"):
        harbor_seal.discover_completed(job)

    root = _add_runtime_snapshot(trial)
    candidate = harbor_seal.discover_completed(job)[0][0]
    assert harbor_seal.RUNTIME_SNAPSHOT_SOURCE in candidate.spec["captured_roots"]

    (root / "collector.py").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(candidate_seal.SealError, match="collector binding"):
        harbor_seal.discover_completed(job)


def test_shell_runtime_smoke_is_accepted_but_rc127_snapshot_is_rejected(
    tmp_path,
):
    accepted_job, accepted_trial = harbor_job(tmp_path / "accepted")
    agent = {
        "name": "codex_agent:TieredCodex",
        "model_name": "gpt-test",
        "kwargs": {
            "reasoning_effort": "high",
            "service_tier": "default",
            "fast_mode": "disable",
        },
    }
    _replace_agent(accepted_job, agent)
    _add_runtime_snapshot(
        accepted_trial, collector_runtime="posix-sh-fallback")
    assert harbor_seal.discover_completed(accepted_job)[0]

    rejected_job, rejected_trial = harbor_job(tmp_path / "rejected")
    _replace_agent(rejected_job, agent)
    root = _add_runtime_snapshot(
        rejected_trial,
        collector_runtime="posix-sh-fallback",
        returncode=b"127\n",
    )
    (root / "manifest.json").unlink()
    with pytest.raises(candidate_seal.SealError):
        harbor_seal.discover_completed(rejected_job)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "wrong/v1", "wrong schema"),
        ("status", "partial", "not complete"),
        ("capture_boundary", "after verifier", "capture boundary"),
        ("privacy", {"process_environments_captured": True}, "may not capture"),
        ("collector_runtime", "python3", "unsupported"),
    ],
)
def test_runtime_snapshot_manifest_contract_is_fail_closed(
    tmp_path, field, value, message,
):
    job, trial = harbor_job(tmp_path)
    _replace_agent(job, {
        "name": "codex_agent:TieredCodex",
        "model_name": "gpt-test",
        "kwargs": {
            "reasoning_effort": "high",
            "service_tier": "default",
            "fast_mode": "disable",
        },
    })
    root = _add_runtime_snapshot(
        trial, collector_runtime="posix-sh-fallback")
    inner_path = root / "manifest.json"
    inner = json.loads(inner_path.read_text(encoding="utf-8"))
    inner[field] = value
    _write_json(inner_path, inner)

    with pytest.raises(candidate_seal.SealError, match=message):
        harbor_seal.discover_completed(job)


def test_agent_safety_refusal_with_complete_evidence_is_candidate_outcome(
    tmp_path,
):
    job, trial = harbor_job(tmp_path)
    _finish_fixture_trajectory(trial)
    normal = harbor_seal.discover_completed(job)[0][0]
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["exception_info"] = {
        "exception_type": "AgentSafetyRefusalError",
        "exception_message": "policy boundary",
        "exception_traceback": "trace",
        "occurred_at": "2026-01-01T00:02:00Z",
    }
    _write_json(result_path, result)
    raw_rollout = _add_raw_exception_rollout(trial)

    exceptional = harbor_seal.discover_completed(job)[0][0]
    assert exceptional.spec["agent_exception_type"] == \
        "AgentSafetyRefusalError"
    assert exceptional.seal["candidate_id"] != normal.seal["candidate_id"]
    assert exceptional.seal["pool_id"] == normal.seal["pool_id"]
    assert exceptional.seal["identity"]["trajectory"]["raw_sha256"] \
        == candidate_seal.file_digest(raw_rollout)
    rendered_view = json.dumps(
        candidate_seal.verifier_view(exceptional.seal), sort_keys=True)
    assert "AgentSafetyRefusalError" not in rendered_view
    assert "reward" not in rendered_view


def test_immediate_agent_refusal_without_final_metrics_is_still_candidate(
    tmp_path,
):
    job, trial = harbor_job(tmp_path)
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["exception_info"] = {
        "exception_type": "AgentSafetyRefusalError",
        "exception_message": "zero-token policy refusal",
        "exception_traceback": "trace",
        "occurred_at": "2026-01-01T00:02:00Z",
    }
    _write_json(result_path, result)
    rollout = _add_raw_exception_rollout(trial)

    candidate = harbor_seal.discover_completed(job)[0][0]

    trajectory = json.loads(
        (trial / "agent/trajectory.json").read_text(encoding="utf-8"))
    assert "final_metrics" not in trajectory
    assert candidate.seal["identity"]["trajectory"]["raw_sha256"] \
        == candidate_seal.file_digest(rollout)
    assert candidate.spec["agent_exception_type"] == "AgentSafetyRefusalError"


def test_exceptional_rollout_must_be_unique_and_session_bound(tmp_path):
    job, trial = harbor_job(tmp_path)
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["exception_info"] = {
        "exception_type": "AgentSafetyRefusalError",
        "exception_message": "policy refusal",
        "exception_traceback": "trace",
        "occurred_at": "2026-01-01T00:02:00Z",
    }
    _write_json(result_path, result)
    first = _add_raw_exception_rollout(trial)
    second = first.with_name("rollout-ambiguous.jsonl")
    shutil.copy2(first, second)

    with pytest.raises(candidate_seal.SealError, match="exactly one raw Codex"):
        harbor_seal.discover_completed(job)

    second.unlink()
    trajectory_path = trial / "agent/trajectory.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory["session_id"] = "different-session"
    _write_json(trajectory_path, trajectory)
    with pytest.raises(candidate_seal.SealError, match="session id disagrees"):
        harbor_seal.discover_completed(job)


def test_exceptional_rollout_terminal_must_match_private_exception(tmp_path):
    job, trial = harbor_job(tmp_path)
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["exception_info"] = {
        "exception_type": "AgentSafetyRefusalError",
        "exception_message": "policy refusal",
        "exception_traceback": "trace",
        "occurred_at": "2026-01-01T00:02:00Z",
    }
    _write_json(result_path, result)
    rollout = _add_raw_exception_rollout(
        trial,
        codex_error_info="context_window_exceeded",
        message="input token count exceeds the maximum number of tokens",
    )

    with pytest.raises(candidate_seal.SealError, match="does not match"):
        harbor_seal.discover_completed(job)

    events = [json.loads(line) for line in rollout.read_text().splitlines()]
    events.pop(1)
    rollout.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    with pytest.raises(candidate_seal.SealError, match="lifecycle"):
        harbor_seal.discover_completed(job)


def test_exceptional_final_metrics_are_checked_when_present(tmp_path):
    job, trial = harbor_job(tmp_path)
    _finish_fixture_trajectory(trial)
    _add_raw_exception_rollout(trial)
    trajectory_path = trial / "agent/trajectory.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory["final_metrics"]["total_steps"] += 1
    _write_json(trajectory_path, trajectory)
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["exception_info"] = {
        "exception_type": "AgentSafetyRefusalError",
        "exception_message": "policy refusal",
        "exception_traceback": "trace",
        "occurred_at": "2026-01-01T00:02:00Z",
    }
    _write_json(result_path, result)

    with pytest.raises(candidate_seal.SealError, match="inconsistent ATIF"):
        harbor_seal.discover_completed(job)


@pytest.mark.parametrize(
    "exception_type",
    [
        "AgentAuthenticationError",
        "AgentTimeoutError",
        "ApiRateLimitError",
        "NetworkConnectionError",
    ],
)
def test_infrastructure_agent_exceptions_remain_unsealable(
    tmp_path, exception_type,
):
    job, trial = harbor_job(tmp_path)
    _finish_fixture_trajectory(trial)
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["exception_info"] = {
        "exception_type": exception_type,
        "exception_message": "not a solver outcome",
        "exception_traceback": "trace",
        "occurred_at": "2026-01-01T00:02:00Z",
    }
    _write_json(result_path, result)

    candidates, skipped, _ = harbor_seal.discover_completed(job)
    assert candidates == []
    assert skipped == {trial.name: "errored"}


def test_agent_outcome_requires_finished_trajectory_and_captured_world(tmp_path):
    job, trial = harbor_job(tmp_path)
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["exception_info"] = {
        "exception_type": "AgentSafetyRefusalError",
        "exception_message": "policy boundary",
        "exception_traceback": "trace",
        "occurred_at": "2026-01-01T00:02:00Z",
    }
    _write_json(result_path, result)
    with pytest.raises(candidate_seal.SealError, match="finished ATIF"):
        harbor_seal.discover_completed(job)

    _finish_fixture_trajectory(trial)
    _add_raw_exception_rollout(trial)
    manifest_path = trial / "artifacts/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest:
        row["status"] = "failed"
    _write_json(manifest_path, manifest)
    with pytest.raises(candidate_seal.SealError, match="captured artifact world"):
        harbor_seal.discover_completed(job)


def test_legacy_frozen_standard_spec_remains_byte_compatible(tmp_path):
    job, trial = harbor_job(tmp_path)
    outputs = _outputs(tmp_path)
    harbor_seal.freeze_job(job, **outputs)
    spec = candidate_seal.load_jsonl(outputs["specs_path"])[0]
    spec.pop("service_tier")
    spec.pop("task_config_digest")
    candidate_seal.write_jsonl_atomic(outputs["specs_path"], [spec])
    shutil.rmtree(trial / "agent")
    shutil.rmtree(trial / "artifacts")

    summary = harbor_seal.freeze_job(job, **outputs)
    assert summary["added"] == 0
    assert summary["reused"] == 1
    assert candidate_seal.load_jsonl(outputs["specs_path"])[0] == spec


def test_incremental_freeze_reuses_pruned_sources_and_only_hashes_new_trial(
    tmp_path,
):
    job, old_trial = harbor_job(tmp_path)
    outputs = _outputs(tmp_path)
    harbor_seal.freeze_job(job, **outputs)
    old_spec = candidate_seal.load_jsonl(outputs["specs_path"])[0]
    _clone_trial(job, "generic-task__trial2")
    shutil.rmtree(old_trial / "agent")
    shutil.rmtree(old_trial / "artifacts")

    summary = harbor_seal.freeze_job(job, **outputs)

    assert summary["reused"] == 1
    assert summary["discovered"] == summary["added"] == 1
    assert summary["frozen"] == 2
    specs = candidate_seal.load_jsonl(outputs["specs_path"])
    assert old_spec in specs
    assert {spec["trial"] for spec in specs} == {
        "generic-task__trial1", "generic-task__trial2"}


def test_incremental_skip_uses_dataset_identity_not_trial_name(tmp_path):
    job, trial = harbor_job(tmp_path)
    outputs = _outputs(tmp_path)
    harbor_seal.freeze_job(job, **outputs)
    shutil.rmtree(trial / "agent")
    shutil.rmtree(trial / "artifacts")

    new_dataset = "other/dataset"
    job_config_path = job / "config.json"
    job_config = json.loads(job_config_path.read_text(encoding="utf-8"))
    job_config["datasets"][0]["name"] = new_dataset
    _write_json(job_config_path, job_config)
    job_lock_path = job / "lock.json"
    job_lock = json.loads(job_lock_path.read_text(encoding="utf-8"))
    job_lock["trials"][0]["task"]["source"] = new_dataset
    _write_json(job_lock_path, job_lock)
    for path in (trial / "config.json", trial / "lock.json", trial / "result.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "config.json":
            value["task"]["source"] = new_dataset
        elif path.name == "lock.json":
            value["task"]["source"] = new_dataset
        else:
            value["source"] = new_dataset
            value["config"]["task"]["source"] = new_dataset
        _write_json(path, value)

    with pytest.raises(candidate_seal.SealError, match="trajectory"):
        harbor_seal.freeze_job(job, **outputs)


def test_frozen_trial_cannot_regress_to_incomplete_or_errored(tmp_path):
    job, trial = harbor_job(tmp_path)
    outputs = _outputs(tmp_path)
    harbor_seal.freeze_job(job, **outputs)
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["finished_at"] = None
    _write_json(result_path, result)

    with pytest.raises(candidate_seal.SealError, match="became incomplete"):
        harbor_seal.freeze_job(job, **outputs)


@pytest.mark.parametrize("corrupt", ["spec", "seal", "view"])
def test_incremental_freeze_requires_valid_spec_seal_view_triple(
    tmp_path, corrupt,
):
    job, trial = harbor_job(tmp_path)
    outputs = _outputs(tmp_path)
    harbor_seal.freeze_job(job, **outputs)
    path = outputs[f"{corrupt}s_path"] if corrupt != "view" \
        else outputs["verifier_inputs_path"]
    rows = candidate_seal.load_jsonl(path)
    if corrupt == "spec":
        rows[0]["dataset_digest"] = "f" * 64
    elif corrupt == "seal":
        rows[0]["candidate_id"] = "f" * 64
    else:
        rows[0]["pool_id"] = "f" * 64
    candidate_seal.write_jsonl_atomic(path, rows)
    shutil.rmtree(trial / "agent")
    shutil.rmtree(trial / "artifacts")

    with pytest.raises(candidate_seal.SealError):
        harbor_seal.freeze_job(job, **outputs)


def test_freeze_jobs_merges_multiple_jobs_into_one_target_pool(tmp_path):
    first_job, _ = harbor_job(tmp_path / "first")
    second_job, original = harbor_job(tmp_path / "second")
    second = _clone_trial(second_job, "generic-task__trial2")
    shutil.rmtree(original)
    outputs = _outputs(tmp_path)

    summary = harbor_seal.freeze_jobs(
        [first_job, second_job], **outputs)

    assert summary["added"] == summary["discovered"] == 2
    assert summary["frozen"] == 2
    seals = candidate_seal.load_jsonl(outputs["seals_path"])
    assert len({seal["pool_id"] for seal in seals}) == 1
    assert {seal["identity"]["trial"] for seal in seals} == {
        "generic-task__trial1", second.name}
