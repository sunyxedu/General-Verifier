"""Official Terminal-Bench 2.1 GV driver: identity and isolation tests."""
from __future__ import annotations

import importlib
import hashlib
import json
import os
from pathlib import Path

import pytest

from experiments.common import results as result_io
from experiments.terminal_bench import score_official as official
from verifier.execution.instruments import world_evidence_context

miniswe_adapter = importlib.import_module("miniswe")
TEST_PROTOCOL = (
    miniswe_adapter.TASK_WRAPPER_MARKER
    + "\n\n## Recommended Workflow\nUse bash and finish cleanly."
)


@pytest.fixture(autouse=True)
def pinned_test_protocol(monkeypatch):
    monkeypatch.setattr(
        miniswe_adapter, "TASK_WRAPPER_SHA256",
        hashlib.sha256(TEST_PROTOCOL.encode("utf-8")).hexdigest())


def wrapped_task(message: str) -> str:
    return message + "\n\n\n" + TEST_PROTOCOL


def trajectory(task: str, trial: str, reward: int = 1) -> dict:
    return {
        "trial_id": f"uuid-{trial}",
        "trial_name": trial,
        "task_name": task,
        "task_checksum": f"checksum-{task}",
        "reward": reward,
        "trajectory": {
            "schema_version": "atif-converted-1.0",
            "session_id": f"session-{trial}",
            "agent": {
                "name": "mini-swe-agent",
                "model_name": "deepseek/deepseek-v4-flash",
            },
            "steps": [
                {"step_id": 1, "source": "user",
                 "message": wrapped_task(f"solve {task}")},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": f"finished {trial}",
                    "tool_calls": [{
                        "tool_name": "bash",
                        "arguments": {"keystrokes": "printf done"},
                    }],
                    "observation": {
                        "results": [{"content": json.dumps({
                            "returncode": 0, "output": "done",
                        })}],
                    },
                },
            ],
        },
    }


def write_trace(root: Path, task: str, trial: str, reward: int = 1) -> Path:
    path = root / "mini-swe-agent_deepseek-v4-flash" / task \
        / f"{trial}_trajectory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trajectory(task, trial, reward)), encoding="utf-8")
    return path


def args_for(root: Path, results: Path, scoring_root: Path, *extra: str):
    return official.build_parser().parse_args([
        "--trajectory-root", str(root),
        "--results", str(results),
        "--scoring-root", str(scoring_root),
        *extra,
    ])


PROVIDER = official.ProviderIdentity(
    provider="custom",
    endpoint="http://127.0.0.1:8371/v1",
    requested_model="deepseek-v4-flash-0731-h200-high",
    served_model="deepseek-ai/DeepSeek-V4-Flash-0731@h200-high",
    text_only=True,
)


def test_trace_and_tree_hash_exclude_reward(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    path_left = write_trace(left, "task-a", "task-a__one", reward=0)
    path_right = write_trace(right, "task-a", "task-a__one", reward=1)

    assert official.trace_hash(str(path_left)) == official.trace_hash(
        str(path_right))
    assert official.discover_dataset(str(left)).tree_fingerprint == \
        official.discover_dataset(str(right)).tree_fingerprint

    changed = trajectory("task-a", "task-a__one", reward=1)
    changed["trajectory"]["steps"][-1]["message"] = "different completion"
    path_right.write_text(json.dumps(changed), encoding="utf-8")
    assert official.trace_hash(str(path_left)) != official.trace_hash(
        str(path_right))


def test_provider_and_served_model_are_bound_into_run_identity():
    cfg = official.offline.config_fields(model="same-model")
    openrouter = official.ProviderIdentity(
        "openrouter", "https://openrouter.ai/api", "same-model", "same-model",
        True)
    local = official.ProviderIdentity(
        "custom", "http://127.0.0.1:8371/v1", "same-model", "same-model",
        True)
    checkpoint_b = official.ProviderIdentity(
        "custom", "http://127.0.0.1:8371/v1", "same-model", "checkpoint-b",
        True)
    multimodal = official.ProviderIdentity(
        "custom", "http://127.0.0.1:8371/v1", "same-model", "same-model",
        False)

    first, _ = official.batch_identity(cfg, "tree", "adapter", openrouter, "r1")
    second, _ = official.batch_identity(cfg, "tree", "adapter", local, "r1")
    third, _ = official.batch_identity(cfg, "tree", "adapter", checkpoint_b, "r1")
    fourth, _ = official.batch_identity(cfg, "tree", "adapter", multimodal, "r1")

    assert len({first, second, third, fourth}) == 4


def test_fast_service_tier_requires_codex_seats(tmp_path):
    args = args_for(
        tmp_path / "traces",
        tmp_path / "results.jsonl",
        tmp_path / "states",
        "--service-tier", "fast",
    )
    with pytest.raises(SystemExit, match="requires --seats codex"):
        official.run(args, provider_identity=PROVIDER)


def test_dry_run_filters_and_never_converts_scores_or_writes(tmp_path, capsys):
    root = tmp_path / "data" / "terminal_bench_2.1_trajs"
    write_trace(root, "task-a", "task-a__one")
    write_trace(root, "task-b", "task-b__one")
    results = tmp_path / "out" / "scores.jsonl"
    scoring_root = tmp_path / "states"

    class BombAdapter:
        def convert(self, *_args):
            raise AssertionError("dry-run converted a trajectory")

    def bomb_score(*_args, **_kwargs):
        raise AssertionError("dry-run invoked score_many")

    args = args_for(
        root, results, scoring_root,
        "--task", "task-b", "--trial", "task-b__one",
        "--workers", "7", "--rep", "pilot-r2", "--dry-run")
    assert official.run(
        args, adapter=BombAdapter(), adapter_hash="a" * 64,
        provider_identity=PROVIDER, score_many_fn=bomb_score) == 0

    streams = capsys.readouterr()
    assert streams.out == ""
    assert "DRY-RUN" in streams.err
    assert "selected 1 traces in 1 tasks" in streams.err
    assert "with 7 workers" in streams.err
    assert "provider=custom" in streams.err
    assert "served=deepseek-ai/DeepSeek-V4-Flash-0731@h200-high" in streams.err
    assert "text_only=1" in streams.err
    assert not results.exists()
    assert not scoring_root.exists()


def test_strict_state_scoring_and_per_task_gv_ranks(tmp_path, capsys):
    root = tmp_path / "data" / "terminal_bench_2.1_trajs"
    write_trace(root, "task-a", "task-a__bad", reward=0)
    write_trace(root, "task-a", "task-a__good", reward=1)
    results = tmp_path / "output" / "official.jsonl"
    scoring_root = tmp_path / "isolated-states"
    seen = []

    class Adapter:
        def convert(self, path, state):
            miniswe_adapter.convert(path, state)

    def fake_score_many(jobs, out_path, **kwargs):
        assert kwargs["workers"] == 2
        assert kwargs["model"] == PROVIDER.requested_model
        for record, state in jobs:
            seen.append((record, state))
            assert "reward" not in record and "tb_reward" not in record
            assert record["text_only"] is True
            assert kwargs["label"](record) == \
                f"{record['task']}/{record['trial']}"
            official.validate_state(state)
            assert not os.path.exists(os.path.join(state, "world"))
            assert world_evidence_context(state, live_world=False) == (
                "unavailable", "", "")
            assert set(os.listdir(state)) == official.STATE_FILES
            if record["trial"].endswith("good"):
                features = {
                    "feature_schema_version":
                        official.offline.FEATURE_SCHEMA_VERSION,
                    "unscoreable": False, "verdict": "verified",
                    "slice_size": 2, "claims": {"verified": 2},
                    "obligations": {"verified": 3},
                    "obligations_total": 3,
                    "final_obligations": {"verified": 2},
                    "final_obligations_total": 2,
                }
            else:
                features = {
                    "feature_schema_version":
                        official.offline.FEATURE_SCHEMA_VERSION,
                    "unscoreable": False, "verdict": "refuted",
                    "slice_size": 2, "claims": {"verified": 0},
                    "obligations": {"verified": 0},
                    "obligations_total": 1,
                    "final_obligations": {"refuted": 1},
                    "final_obligations_total": 1,
                }
            scored = dict(record, features=features)
            result_io.append(out_path, scored)
            kwargs["on_record"](scored)

    args = args_for(root, results, scoring_root, "--workers", "2")
    assert official.run(
        args, adapter=Adapter(), adapter_hash="b" * 64,
        provider_identity=PROVIDER, score_many_fn=fake_score_many) == 0

    streams = capsys.readouterr()
    lines = streams.out.strip().splitlines()
    assert lines[0].endswith("\tgv_rank")
    assert "gv_contrary_signals" in lines[0]
    assert "gv_verified_frac" not in lines[0]
    by_trial = {line.split("\t")[1]: line.split("\t") for line in lines[1:]}
    assert by_trial["task-a__good"][3] == "1"
    assert by_trial["task-a__good"][-1] == "1"
    assert by_trial["task-a__bad"][3] == "0"
    assert by_trial["task-a__bad"][-1] == "2"
    assert len(seen) == 2

    persisted = result_io.load(str(results))
    assert len(persisted) == 2
    assert all("reward" not in row and "tb_reward" not in row
               for row in persisted)
    assert all(row["dataset_tree_fingerprint"] for row in persisted)
    assert all(row["adapter_fingerprint"] == "b" * 64 for row in persisted)
    assert all(row["provider_identity"] == PROVIDER.fields()
               for row in persisted)
    assert all(row["trace_hash"] and row["work_id"] for row in persisted)
    assert len({row["work_id"] for row in persisted}) == 2


@pytest.mark.parametrize(
    "tampered", [
        "task.md", "protocol.md", "response_blobs.jsonl",
        "trace_manifest.json",
    ])
def test_resume_allows_only_named_derived_state_and_sealed_sources(
        tmp_path, tampered):
    root = tmp_path / "dataset"
    write_trace(root, "task-a", "task-a__one")
    trace = official.discover_dataset(str(root)).traces[0]
    scoring_root = str(tmp_path / "scoring")
    adapter_module = miniswe_adapter

    state = official.sanitized_state(
        trace, scoring_root, "run-identity", adapter=adapter_module)
    Path(state, "events.jsonl").write_text("", encoding="utf-8")
    Path(state, "extract.json").write_text("{}", encoding="utf-8")
    Path(state, "claims.sqlite").write_bytes(b"sqlite")
    Path(state, "verification.json").write_text('{}', encoding="utf-8")
    Path(state, "artifacts").mkdir()
    Path(state, "workspace").mkdir()
    Path(state, "work", "nested").mkdir(parents=True)

    assert official.sanitized_state(
        trace, scoring_root, "run-identity", adapter=adapter_module) == state
    official.validate_state(state, allow_derived=True)

    Path(state, tampered).write_text("{}" if tampered.endswith(".json")
                                     else "tampered", encoding="utf-8")
    with pytest.raises(
            official.StateError,
            match="digest mismatch|changed since conversion|wrong schema"):
        official.sanitized_state(
            trace, scoring_root, "run-identity", adapter=adapter_module)


@pytest.mark.parametrize("bad_entry", ["world", "gold.json"])
def test_reuse_rejects_world_and_extraneous_entries(tmp_path, bad_entry):
    state = tmp_path / "gv"
    state.mkdir()
    (state / "hook_feed.jsonl").write_text("{}\n", encoding="utf-8")
    (state / "response_blobs.jsonl").write_text("", encoding="utf-8")
    (state / "task.md").write_text("task", encoding="utf-8")
    (state / "protocol.md").write_text("protocol", encoding="utf-8")
    (state / "final.txt").write_text("final", encoding="utf-8")
    (state / "world_manifest.json").write_text(
        json.dumps(official.WORLD_MANIFEST), encoding="utf-8")
    (state / "trace_manifest.json").write_text("{}", encoding="utf-8")
    if bad_entry == "world":
        (state / bad_entry).mkdir()
    else:
        (state / bad_entry).write_text("gold", encoding="utf-8")

    with pytest.raises(official.StateError, match="wrong top-level"):
        official.validate_state(str(state), allow_derived=True)


def test_reuse_rejects_symlinks_inside_derived_work(tmp_path):
    state = tmp_path / "gv"
    state.mkdir()
    (state / "hook_feed.jsonl").write_text("{}\n", encoding="utf-8")
    (state / "response_blobs.jsonl").write_text("", encoding="utf-8")
    (state / "task.md").write_text("task", encoding="utf-8")
    (state / "protocol.md").write_text("protocol", encoding="utf-8")
    (state / "final.txt").write_text("final", encoding="utf-8")
    (state / "world_manifest.json").write_text(
        json.dumps(official.WORLD_MANIFEST), encoding="utf-8")
    (state / "trace_manifest.json").write_text("{}", encoding="utf-8")
    work = state / "work"
    work.mkdir()
    (work / "escape").symlink_to(tmp_path)

    with pytest.raises(official.StateError, match="symlink forbidden"):
        official.validate_state(str(state), allow_derived=True)
