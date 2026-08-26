"""Candidate identity sealing and reward-safe BoN selection tests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.common import candidate_seal as candidates


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def completed_spec(
    tmp_path: Path, *, task: str = "task-a", trial: str = "trial-1",
    reward: int = 1, trajectory_reward: int = 0,
) -> dict:
    task_lock = tmp_path / f"{task}.lock"
    task_lock.write_text(f"locked task package: {task}\n", encoding="utf-8")
    instruction = tmp_path / f"{task}.md"
    instruction.write_text(f"solve {task}\n", encoding="utf-8")
    task_config = tmp_path / f"{task}.json"
    task_config.write_text(
        json.dumps({"task": task, "timeout": 900}), encoding="utf-8")
    trajectory = tmp_path / f"{trial}.json"
    trajectory.write_text(json.dumps({
        "task_name": task,
        "trial_name": trial,
        "trajectory": {"steps": [{"message": f"finished {trial}"}]},
        "reward": trajectory_reward,
    }), encoding="utf-8")
    world = tmp_path / f"world-{trial}"
    world.mkdir(exist_ok=True)
    delivered = world / "answer.txt"
    delivered.write_text(f"answer from {trial}\n", encoding="utf-8")
    delivered.chmod(0o640)
    return {
        "dataset": "terminal-bench/terminal-bench-2-1",
        "dataset_digest": digest("pinned dataset release"),
        "task": task,
        "task_digest_path": str(task_lock),
        "instruction_path": str(instruction),
        "task_config_path": str(task_config),
        "trial": trial,
        "model": "gpt-5.6-luna",
        "effort": "medium",
        "trajectory_path": str(trajectory),
        "world_path": str(world),
        "reward": reward,
    }


def test_seal_binds_candidate_and_verifier_view_excludes_reward(tmp_path):
    spec = completed_spec(tmp_path)
    seal = candidates.seal_from_spec(spec)

    assert seal["identity"]["dataset"] == {
        "name": "terminal-bench/terminal-bench-2-1",
        "sha256": digest("pinned dataset release"),
    }
    assert seal["identity"]["task"]["sha256"] == candidates.file_digest(
        spec["task_digest_path"])
    assert seal["identity"]["task"]["instruction_sha256"]
    assert seal["identity"]["task"]["config_sha256"]
    assert seal["identity"]["solver"] == {
        "model": "gpt-5.6-luna", "effort": "medium"}
    assert seal["identity"]["trajectory"]["sha256"]
    assert seal["identity"]["world"]["status"] == "captured"
    assert seal["evaluation"] == {"reward": 1.0}

    view = candidates.verifier_view(seal)
    rendered = json.dumps(view, sort_keys=True)
    assert "evaluation" not in view
    assert "reward" not in rendered
    assert set(view) == {"schema", "candidate_id", "pool_id", "identity"}


def test_reward_does_not_change_candidate_identity_but_artifacts_do(tmp_path):
    spec = completed_spec(tmp_path)
    first = candidates.seal_from_spec(spec)

    # Both private reward and a top-level official trajectory reward are gold.
    # Neither is part of the solver candidate identity.
    reward_edit = dict(spec, reward=0)
    trajectory_path = Path(spec["trajectory_path"])
    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    payload["reward"] = 1
    trajectory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    second = candidates.seal_from_spec(reward_edit)
    assert second["candidate_id"] == first["candidate_id"]
    assert second["pool_id"] == first["pool_id"]
    assert second["seal_id"] != first["seal_id"]

    payload["trajectory"]["steps"][0]["message"] = "different completion"
    trajectory_path.write_text(json.dumps(payload), encoding="utf-8")
    changed_trace = candidates.seal_from_spec(reward_edit)
    assert changed_trace["candidate_id"] != first["candidate_id"]

    # Pool identity intentionally excludes per-candidate trajectory/world.
    assert changed_trace["pool_id"] == first["pool_id"]
    Path(spec["world_path"], "answer.txt").write_text(
        "different world\n", encoding="utf-8")
    changed_world = candidates.seal_from_spec(reward_edit)
    assert changed_world["candidate_id"] != changed_trace["candidate_id"]


def test_standard_service_tier_is_legacy_compatible_and_fast_splits_pool(
    tmp_path,
):
    legacy_spec = completed_spec(tmp_path)
    legacy = candidates.seal_from_spec(legacy_spec)
    explicit_standard = candidates.seal_from_spec(
        dict(legacy_spec, service_tier="standard"))
    fast = candidates.seal_from_spec(dict(legacy_spec, service_tier="fast"))

    assert explicit_standard["candidate_id"] == legacy["candidate_id"]
    assert explicit_standard["pool_id"] == legacy["pool_id"]
    assert "service_tier" not in legacy["identity"]["solver"]
    assert fast["identity"]["solver"]["service_tier"] == "fast"
    assert fast["candidate_id"] != legacy["candidate_id"]
    assert fast["pool_id"] != legacy["pool_id"]


def test_optional_raw_trajectory_digest_binds_candidate_but_not_pool(tmp_path):
    spec = completed_spec(tmp_path)
    legacy = candidates.seal_from_spec(spec)
    raw = tmp_path / "rollout.jsonl"
    raw.write_text('{"type":"session_meta"}\n', encoding="utf-8")

    bound = candidates.seal_from_spec(dict(
        spec,
        raw_trajectory_path=str(raw),
        raw_trajectory_digest=candidates.file_digest(raw),
    ))

    assert "raw_sha256" not in legacy["identity"]["trajectory"]
    assert bound["identity"]["trajectory"]["raw_sha256"] \
        == candidates.file_digest(raw)
    assert bound["candidate_id"] != legacy["candidate_id"]
    assert bound["pool_id"] == legacy["pool_id"]

    raw.write_text('{"type":"session_meta","changed":true}\n', encoding="utf-8")
    with pytest.raises(candidates.SealError, match="raw_trajectory_digest"):
        candidates.seal_from_spec(dict(
            spec,
            raw_trajectory_path=str(raw),
            raw_trajectory_digest=bound["identity"]["trajectory"]["raw_sha256"],
        ))


def test_service_tier_rejects_unknown_value(tmp_path):
    with pytest.raises(candidates.SealError, match="service_tier"):
        candidates.seal_from_spec(
            dict(completed_spec(tmp_path), service_tier="turbo"))


def test_freeze_is_idempotent_and_rejects_mutating_a_source(tmp_path):
    spec = completed_spec(tmp_path)
    first = candidates.seal_from_spec(spec)
    seals_path = tmp_path / "seals.jsonl"

    assert candidates.freeze_seals(seals_path, [first]) == [first]
    assert candidates.freeze_seals(seals_path, [first]) == [first]

    Path(spec["world_path"], "answer.txt").write_text(
        "mutated after completion\n", encoding="utf-8")
    changed = candidates.seal_from_spec(spec)
    with pytest.raises(candidates.SealError, match="change a frozen"):
        candidates.freeze_seals(seals_path, [changed])


def make_pool(tmp_path: Path, task: str, rewards: list[int]):
    result = []
    for index, reward in enumerate(rewards):
        trial_dir = tmp_path / f"{task}-{index}"
        trial_dir.mkdir()
        spec = completed_spec(
            trial_dir, task=task, trial=f"{task}__{index}", reward=reward)
        result.append(candidates.seal_from_spec(spec))
    return result


def test_trial_specific_harbor_configs_still_form_one_five_way_pool(tmp_path):
    seals = []
    for index in range(5):
        trial = f"task-a__{index}"
        trial_dir = tmp_path / trial
        trial_dir.mkdir()
        spec = completed_spec(
            trial_dir, task="task-a", trial=trial, reward=int(index == 2))
        # Real Harbor trial configs bind the replicate name.  This must remain
        # part of the exact candidate identity without entering pool identity.
        Path(spec["task_config_path"]).write_text(
            json.dumps({
                "task": "task-a", "timeout": 900, "trial_name": trial,
            }),
            encoding="utf-8",
        )
        seals.append(candidates.seal_from_spec(spec))

    assert len({seal["identity"]["task"]["config_sha256"] for seal in seals}) \
        == 5
    assert len({seal["candidate_id"] for seal in seals}) == 5
    assert len({seal["pool_id"] for seal in seals}) == 1
    scores = [{
        "candidate_id": seal["candidate_id"],
        "scorer": "test-scorer",
        "score": float(index),
    } for index, seal in enumerate(seals)]
    report = candidates.evaluate_bon(seals, scores, expected_n=5)
    assert report["scorers"]["test-scorer"]["pools"] == 1


def test_bon_top1_is_per_pool_and_reports_success_and_selector_hits(tmp_path):
    seals = [
        *make_pool(tmp_path, "task-a", [0, 1, 0]),
        *make_pool(tmp_path, "task-b", [1, 0, 0]),
    ]
    by_trial = {seal["identity"]["trial"]: seal for seal in seals}
    scores = []
    good_scores = {
        "task-a__0": 0.1, "task-a__1": 0.9, "task-a__2": 0.2,
        "task-b__0": 0.8, "task-b__1": 0.4, "task-b__2": 0.0,
    }
    weak_scores = {
        "task-a__0": 0.9, "task-a__1": 0.8, "task-a__2": 0.1,
        "task-b__0": 0.9, "task-b__1": 0.2, "task-b__2": 0.1,
    }
    for scorer, values in (("good", good_scores), ("weak", weak_scores)):
        for trial, score in values.items():
            scores.append({
                "candidate_id": by_trial[trial]["candidate_id"],
                "scorer": scorer,
                "score": score,
            })

    report = candidates.evaluate_bon(seals, scores, expected_n=3)
    good = report["scorers"]["good"]
    weak = report["scorers"]["weak"]
    assert good["pools"] == 2
    assert good["top1_correct"] == 2
    assert good["top1_accuracy"] == 1.0
    assert good["selector_hits"] == 2
    assert good["selector_hit_rate"] == 1.0
    assert good["random_reward_mean"] == pytest.approx(1 / 3)
    assert good["oracle_reward_mean"] == 1.0
    assert weak["top1_correct"] == 1
    assert weak["top1_accuracy"] == 0.5
    assert weak["selector_hits"] == 1
    assert {row["task"] for row in good["selections"]} == {
        "task-a", "task-b"}


def test_bon_rejects_partial_pools_and_gold_in_score_rows(tmp_path):
    seals = make_pool(tmp_path, "task-a", [0, 1, 0])
    partial = [{
        "candidate_id": seals[0]["candidate_id"],
        "scorer": "partial",
        "score": 0.5,
    }]
    with pytest.raises(candidates.SealError, match="partial coverage"):
        candidates.evaluate_bon(seals, partial, expected_n=3)

    leaked = dict(partial[0], reward=0)
    with pytest.raises(candidates.SealError, match="held-out"):
        candidates.evaluate_bon(seals, [leaked])


def test_validate_rejects_tampered_pool_identity(tmp_path):
    seal = candidates.seal_from_spec(completed_spec(tmp_path))
    tampered = copy.deepcopy(seal)
    tampered["pool_id"] = "0" * 64
    unsigned = dict(tampered)
    unsigned.pop("seal_id")
    tampered["seal_id"] = candidates.stable_digest(unsigned)

    with pytest.raises(candidates.SealError, match="pool_id"):
        candidates.validate_seal(tampered)
