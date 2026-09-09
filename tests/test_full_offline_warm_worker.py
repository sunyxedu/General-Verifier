"""Warm treatment batches score every candidate of a task in one process."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.terminal_bench import run_full_offline_warm_worker as warm
from experiments.terminal_bench import build_full_offline_comparison_plan as builder
from test_build_full_offline_comparison_plan import _tiny_repo, _general_package
from verifier import offline


def _plan(tmp_path, monkeypatch, count=2):
    source = tmp_path / "source"
    (source / "verifier").mkdir(parents=True)
    candidates = []
    for index in range(count):
        inputs = tmp_path / f"inputs-{index}"
        inputs.mkdir()
        (inputs / "task.md").write_text("Do the task.\n")
        (inputs / "final.txt").write_text("done\n")
        (inputs / "hook_feed.jsonl").write_text("")
        (inputs / "world_manifest.json").write_text("{}")
        seal = offline.write_immutable_input_anchor(str(inputs))
        candidates.append({"candidate_id": f"cand-{index}",
                           "input_source": str(inputs.resolve()),
                           "input_seal": seal})
    pins = {name: hashlib.sha256(
        Path(warm.__file__).with_name(name).read_bytes()).hexdigest()
        for name in warm.PINNED}
    return {
        "schema": warm.SCHEMA, "source_root": str(source.resolve()),
        "source_commit": "a" * 40, "source_fingerprint": "b" * 64,
        "task_family": "task", "candidates": candidates,
        "output_root": str((tmp_path / "output").resolve()),
        "runtime_read_roots": ["/usr/bin"], "support_sha256": pins,
    }


def test_plan_validation_rejects_single_candidate_and_overlaps(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    # paid outputs may not live in host temporary trees; validate the
    # rule itself, then use a repository-local scratch root for the rest
    with pytest.raises(ValueError, match="host temporary trees"):
        warm.validate_plan(deepcopy(plan))
    import tempfile
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="gv-warm-test.", dir=root) as scratch:
        plan["output_root"] = str(Path(scratch).resolve() / "output")
        warm.validate_plan(deepcopy(plan))
        single = deepcopy(plan)
        single["candidates"] = single["candidates"][:1]
        with pytest.raises(ValueError, match="two to sixty-four"):
            warm.validate_plan(single)
        duplicate = deepcopy(plan)
        duplicate["candidates"][1]["candidate_id"] = "cand-0"
        with pytest.raises(ValueError, match="unique"):
            warm.validate_plan(duplicate)
        unpinned = deepcopy(plan)
        del unpinned["support_sha256"]["analyze_codex_observer.py"]
        with pytest.raises(ValueError, match="every warm support file"):
            warm.validate_plan(unpinned)
        tampered = deepcopy(plan)
        tampered["support_sha256"]["run_full_offline_worker.py"] = "0" * 64
        with pytest.raises(ValueError, match="support source changed"):
            warm.validate_plan(tampered)


def test_batch_scores_each_candidate_once_in_order_and_keeps_failures(
        tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch, count=3)
    env = {}
    monkeypatch.setattr(warm.os, "environ", env)
    calls = []

    def score(state, **kwargs):
        calls.append(state)
        if "cand-1" in state:
            raise ValueError("synthetic scoring failure")
        Path(state).mkdir(parents=True, exist_ok=True)
        (Path(state) / "events.jsonl").write_text(
            '{"type":"contract_cache_hit"}\n{"type":"certificate"}\n')
        return {"unscoreable": False, "verdict": "verified",
                "certificate_id": "c" * 64}

    def clone(offline_module, source, destination, expected):
        destination.mkdir(parents=True)

    offline_stub = SimpleNamespace(
        score=score,
        verify_immutable_input_anchor=lambda path: next(
            c["input_seal"] for c in plan["candidates"]
            if path.startswith(c["input_source"]) or c["candidate_id"] in path),
        immutable_input_seal=lambda path: "x")
    cold = SimpleNamespace(
        _environment=lambda p: env.update({"GV_TYPED_ADAPTERS": "1"}),
        _load_runtime=lambda _: (offline_stub, SimpleNamespace(
            _run_codex_process=lambda: None,
            codex_usage_ledger=_native)),
        _clone=clone,
        _certificate=lambda state, features: {"fixture": True},
        _write_new=warm_write_new,
        _read_jsonl=lambda path: [json.loads(line) for line in
                                  Path(path).read_text().splitlines()],
    )

    @contextmanager
    def _native(*args, **kwargs):
        yield

    @contextmanager
    def observer(*args, **kwargs):
        # every candidate state is registered before any model call
        assert set(kwargs["context_by_cwd"]) == {
            str(Path(plan["output_root"]) / c["candidate_id"] / "state")
            for c in plan["candidates"]}
        yield SimpleNamespace(summary=lambda: {"fixture": True})

    @contextmanager
    def policy(backend, bindings):
        assert len(bindings) == 3
        yield SimpleNamespace(identity={"schema": "fixture-policy"})

    support = {
        "run_full_offline_worker": cold,
        "run_typed_adapter_eval": SimpleNamespace(
            _source_fingerprint=lambda _: plan["source_fingerprint"]),
        "observe_codex_attempts": SimpleNamespace(observe_codex_attempts=observer),
        "analyze_codex_observer": SimpleNamespace(
            analyze_file=lambda *a, **k: {"accounting_complete": True}),
        "eval_offline_policy": SimpleNamespace(
            eval_offline_policy=policy, OfflinePolicyBinding=lambda **k: k),
    }
    monkeypatch.setattr(warm, "_support", lambda name: support[name])
    monkeypatch.setattr(warm, "validate_plan", lambda p: p)
    monkeypatch.setattr(warm, "_sha", lambda path: plan["support_sha256"][Path(path).name])

    result = warm.run_worker(plan)

    assert env["GV_CONTRACT_CACHE"] == "1" and env["GV_METHOD_CACHE"] == "1"
    assert [Path(c).parent.name for c in calls] == ["cand-0", "cand-1", "cand-2"]
    assert result["complete"] is False
    assert result["failures"] == ["cand-1:ValueError"]
    states = {item["candidate_id"]: item for item in result["candidates"]}
    assert states["cand-0"]["certificate"] == {"fixture": True}
    assert states["cand-0"]["event_counts"] == {"contract_cache_hit": 1, "certificate": 1}
    assert states["cand-1"]["features"] is None
    assert states["cand-1"]["failures"] == ["ValueError"]
    assert states["cand-2"]["features"]["verdict"] == "verified"
    assert all(item["wall_seconds"] >= 0 for item in result["candidates"])
    persisted = json.loads((Path(plan["output_root"]) / "worker-result.json").read_text())
    assert persisted["accounting"] == {"accounting_complete": True}


def warm_write_new(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))


def test_build_groups_candidates_by_task_family(tmp_path, monkeypatch, capsys):
    repo, _control, treatment = _tiny_repo(tmp_path)
    import shutil
    shutil.copy2(Path(warm.__file__),
                 repo / "experiments" / "terminal_bench" / "run_full_offline_warm_worker.py")
    package = _general_package(tmp_path)
    args = SimpleNamespace(repo=str(repo), commit=treatment,
                           general_inputs=str(package),
                           run_root=str(tmp_path / "run"),
                           runtime_read_root=["/usr/bin"])
    warm.build(args)
    printed = json.loads(capsys.readouterr().out)
    assert printed["model_calls"] == 0 and printed["commit"] == treatment
    assert [item["task_family"] for item in printed["plans"]] == ["task"]
    plan_path = Path(printed["plans"][0]["plan"])
    raw = plan_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == printed["plans"][0]["plan_sha256"]
    plan = json.loads(raw)
    assert [c["candidate_id"] for c in plan["candidates"]] == ["task__alpha", "task__beta"]
    for candidate in plan["candidates"]:
        assert offline.verify_immutable_input_anchor(
            candidate["input_source"]) == candidate["input_seal"]
    assert not list((tmp_path / "run").rglob("secret.json"))
    support = tmp_path / "run" / "support"
    assert (support / "run_full_offline_warm_worker.py").is_file()
    assert set(plan["support_sha256"]) == set(warm.PINNED)
