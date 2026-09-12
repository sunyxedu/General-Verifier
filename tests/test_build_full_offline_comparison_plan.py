"""The v1 plan builder freezes sources and seals inputs without a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from experiments.terminal_bench import build_full_offline_comparison_plan as builder
from verifier import offline


REPO = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True,
        text=True).stdout.strip()


def _tiny_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    (repo / "verifier").mkdir(parents=True)
    (repo / "verifier" / "__init__.py").write_text("VERSION = 1\n")
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    support = repo / "experiments" / "terminal_bench"
    support.mkdir(parents=True)
    for name in builder.SUPPORT:
        shutil.copy2(REPO / "experiments" / "terminal_bench" / name,
                     support / name)
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "-m", "control")
    control = _git(repo, "rev-parse", "HEAD")
    (repo / "verifier" / "__init__.py").write_text("VERSION = 2\n")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "-am", "treatment")
    treatment = _git(repo, "rev-parse", "HEAD")
    return repo, control, treatment


def _sealed_state(root: Path) -> str:
    root.mkdir(parents=True)
    (root / "task.md").write_text("Do the task.\n")
    (root / "final.txt").write_text("done\n")
    (root / "hook_feed.jsonl").write_text("")
    (root / "world_manifest.json").write_text(json.dumps(
        {"captured_roots": [], "world_mode": "unavailable"}))
    return offline.write_immutable_input_anchor(str(root))


def _package_set(tmp_path: Path) -> Path:
    package_set = tmp_path / "package-set"
    packages = []
    for name in ("task__alpha", "task__beta"):
        seal = _sealed_state(package_set / name / "source")
        packages.append({"package": name, "immutable_input_seal": seal})
    (package_set / "package-manifest.json").write_text(json.dumps(
        {"schema": "gv-terminal-bench-strict-input-package-set-v1",
         "packages": packages}))
    return package_set


def _general_package(tmp_path: Path) -> Path:
    package = tmp_path / "general"
    cases = []
    for index, candidate in enumerate(("task__alpha", "task__beta"), 1):
        case_id = f"case-{index:03d}"
        seal = _sealed_state(package / "model-inputs" / "cases" / case_id / "gv")
        cases.append({"case_id": case_id, "candidate_id": candidate,
                      "task_family": "task", "input_seal": seal,
                      "truth": {"label": index % 2}})
    (package / "model-inputs" / "input-manifest.json").write_text("{}")
    (package / "operator-manifest.json").write_text(json.dumps(
        {"schema": "gv-general-offline-eval-operator-v1", "cases": cases}))
    (package / "coordinator-only-truth").mkdir()
    (package / "coordinator-only-truth" / "secret.json").write_text("{}")
    return package


def _build(tmp_path, repo, control, treatment, **inputs):
    args = SimpleNamespace(
        repo=str(repo), control_commit=control, treatment_commit=treatment,
        inputs=None, general_inputs=None,
        run_root=str(tmp_path / "run"), comparison_id="test",
        runtime_read_root=["/usr/bin"])
    for key, value in inputs.items():
        setattr(args, key, value)
    return builder.build(args)


def _check_plan(plan_path: Path, control: str, treatment: str):
    run_root = plan_path.parents[1]
    plan = json.loads(plan_path.read_text())
    assert plan["schema"] == builder.COMPARISON_SCHEMA
    assert plan["coordinator_sha256"] == hashlib.sha256(
        (run_root / "support" / "run_full_offline_comparison.py").read_bytes()
    ).hexdigest()
    assert [job["arm"] for pair in plan["pairs"] for job in pair["jobs"]] \
        == ["control", "treatment", "treatment", "control"]
    seen = set()
    for pair in plan["pairs"]:
        for path, digest in pair["input_attestations"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
            assert Path(path).is_relative_to(run_root / "inputs")
        for job in pair["jobs"]:
            raw = Path(job["plan_path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == job["plan_sha256"]
            worker = json.loads(raw)
            assert worker["schema"] == builder.WORKER_SCHEMA
            assert worker["arm"] == job["arm"]
            assert worker["judger_prefetch"] is (job["arm"] == "treatment")
            assert worker["source_commit"] == (
                control if job["arm"] == "control" else treatment)
            source = Path(worker["source_root"])
            assert not (source / ".git").exists()
            assert (source / "verifier" / "__init__.py").read_text() == (
                "VERSION = 1\n" if job["arm"] == "control" else "VERSION = 2\n")
            assert offline.verify_immutable_input_anchor(
                worker["input_source"]) == worker["input_seal"]
            assert worker["output_root"] not in seen
            seen.add(worker["output_root"])
            assert set(worker["support_sha256"]) == set(builder.WORKER_PINNED)
    return plan


def test_builds_from_a_strict_package_set(tmp_path, capsys):
    repo, control, treatment = _tiny_repo(tmp_path)
    plan_path = _build(tmp_path, repo, control, treatment,
                       inputs=str(_package_set(tmp_path)))
    plan = _check_plan(plan_path, control, treatment)
    assert [pair["candidate_id"] for pair in plan["pairs"]] == ["alpha", "beta"]
    printed = json.loads(capsys.readouterr().out)
    assert printed["model_calls"] == 0
    assert printed["commits"] == {"control": control, "treatment": treatment}


def test_builds_from_a_general_package_without_copying_truth(tmp_path):
    repo, control, treatment = _tiny_repo(tmp_path)
    plan_path = _build(tmp_path, repo, control, treatment,
                       general_inputs=str(_general_package(tmp_path)))
    plan = _check_plan(plan_path, control, treatment)
    assert [pair["candidate_id"] for pair in plan["pairs"]] \
        == ["task__alpha", "task__beta"]
    run_root = plan_path.parents[1]
    assert not list(run_root.rglob("secret.json"))
    assert not list(run_root.rglob("operator-manifest.json"))
    assert "label" not in (run_root / "plans").read_text() \
        if (run_root / "plans").is_file() else True
    for path in (run_root / "plans").iterdir():
        assert "label" not in path.read_text()
