#!/usr/bin/env python3
"""Build a sealed v1 full-offline comparison run from two pinned commits.

This is a no-model schema join that reproduces the layout of the retained
``full-offline-<commit>-<date>`` runs: frozen ``git archive`` exports of the
control and treatment commits, a verbatim copy of one strict input package
set, pinned copies of the evaluation support files, one worker plan per
candidate and arm, and the balanced comparison plan the v1 coordinator
validates.  Truth never enters worker plans.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import io

SUPPORT = (
    "run_full_offline_worker.py", "observe_codex_attempts.py",
    "analyze_codex_observer.py", "eval_offline_policy.py",
    "run_typed_adapter_eval.py", "run_full_offline_comparison.py",
    "project_full_offline_result.py",
)
WORKER_PINNED = (
    "run_full_offline_worker.py", "observe_codex_attempts.py",
    "analyze_codex_observer.py", "eval_offline_policy.py",
    "run_typed_adapter_eval.py",
)
WORKER_SCHEMA = "gv-full-offline-worker-plan-v1"
COMPARISON_SCHEMA = "gv-full-offline-comparison-plan-v1"
DEFAULT_RUNTIME_READ_ROOTS = (
    "/opt/homebrew/bin", "/opt/homebrew/Cellar", "/opt/homebrew/opt",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _archive(repo: Path, commit: str, target: Path) -> str:
    full = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", commit + "^{commit}"],
        check=True, capture_output=True, text=True).stdout.strip()
    raw = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", full],
        check=True, capture_output=True).stdout
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        archive.extractall(target, filter="data")
    return full


def _module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "_gv_plan_support_" + _sha(path)[:16], path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sealed(source: Path) -> str:
    document = json.loads((source / "immutable_input_seal.json").read_text())
    return document["seal"]


def _file_attestations(source: Path) -> dict:
    """Digest every regular file directly under one sealed input root.

    The captured world tree is covered by the immutable input seal itself;
    attesting its individual files would exceed the coordinator's bound.
    """
    return {str(item): _sha(item) for item in sorted(source.iterdir())
            if item.is_file()}


def _strict_package_candidates(inputs: Path) -> list:
    manifest = json.loads((inputs / "package-manifest.json").read_text())
    result = []
    for package in manifest["packages"]:
        name = package["package"]
        source = inputs / name / "source"
        seal = _sealed(source)
        if seal != package["immutable_input_seal"]:
            raise SystemExit(f"input seal disagrees for {name}")
        attestations = {str(inputs / "package-manifest.json"):
                        _sha(inputs / "package-manifest.json")}
        preparation = inputs / name / "input-preparation.json"
        if preparation.is_file():
            attestations[str(preparation)] = _sha(preparation)
        attestations.update(_file_attestations(source))
        result.append((name.rsplit("__", 1)[-1], source, seal, attestations))
    return result


def _general_candidates(run_root: Path, package: Path) -> list:
    """Copy a general offline evaluation package's model inputs verbatim.

    Only ``model-inputs`` is copied; the coordinator-only truth stays where
    it is and never enters a worker plan.  Candidate identifiers keep the
    package's task-qualified names so a later truth join is exact.
    """
    operator = json.loads((package / "operator-manifest.json").read_text())
    inputs = run_root / "inputs" / "model-inputs"
    shutil.copytree(package / "model-inputs", inputs, symlinks=False)
    manifest = inputs / "input-manifest.json"
    result = []
    for case in operator["cases"]:
        source = inputs / "cases" / case["case_id"] / "gv"
        seal = _sealed(source)
        if seal != case["input_seal"]:
            raise SystemExit(f"input seal disagrees for {case['case_id']}")
        attestations = {str(manifest): _sha(manifest)}
        attestations.update(_file_attestations(source))
        result.append((case["candidate_id"], source, seal, attestations))
    return result


def build(args) -> Path:
    repo = Path(args.repo).resolve()
    run_root = Path(args.run_root)
    if run_root.exists():
        raise SystemExit(f"run root exists: {run_root}")
    run_root.mkdir(parents=True)
    run_root = run_root.resolve()

    commits = {}
    for arm, commit in (("control", args.control_commit),
                        ("treatment", args.treatment_commit)):
        commits[arm] = _archive(repo, commit, run_root / "sources" / arm)

    if bool(args.inputs) == bool(args.general_inputs):
        raise SystemExit("give exactly one of --inputs or --general-inputs")
    if args.inputs:
        package_source = Path(args.inputs).resolve()
        inputs = run_root / "inputs" / "package-set"
        shutil.copytree(package_source, inputs, symlinks=False)
        candidates = _strict_package_candidates(inputs)
    else:
        candidates = _general_candidates(
            run_root, Path(args.general_inputs).resolve())

    support = run_root / "support"
    support.mkdir()
    for name in SUPPORT:
        shutil.copy2(repo / "experiments" / "terminal_bench" / name,
                     support / name)
    support_sha = {name: _sha(support / name) for name in WORKER_PINNED}
    tools = _module(support / "run_typed_adapter_eval.py")
    fingerprints = {
        arm: tools._source_fingerprint(run_root / "sources" / arm)
        for arm in commits
    }

    plans = run_root / "plans"
    plans.mkdir()
    (run_root / "worker-outputs").mkdir()
    pairs = []
    for index, (candidate, source, seal, attestations) in enumerate(candidates):
        jobs = []
        order = (("control", "treatment") if index % 2 == 0
                 else ("treatment", "control"))
        for arm in order:
            plan = {
                "schema": WORKER_SCHEMA,
                "source_root": str(run_root / "sources" / arm),
                "source_commit": commits[arm],
                "source_fingerprint": fingerprints[arm],
                "input_source": str(source),
                "input_seal": seal,
                "output_root": str(run_root / "worker-outputs"
                                   / f"{candidate}-{arm}"),
                "arm": arm,
                "judger_prefetch": arm == "treatment",
                "runtime_read_roots": list(args.runtime_read_root
                                           or DEFAULT_RUNTIME_READ_ROOTS),
                "support_sha256": support_sha,
            }
            plan_path = plans / f"{candidate}-{arm}.json"
            digest = _write_json(plan_path, plan)
            jobs.append({"arm": arm, "plan_path": str(plan_path),
                         "plan_sha256": digest})
        pairs.append({"candidate_id": candidate,
                      "input_attestations": attestations, "jobs": jobs})

    comparison = {
        "schema": COMPARISON_SCHEMA,
        "comparison_id": args.comparison_id,
        "output_root": str(run_root / "comparison"),
        "worker_path": str(support / "run_full_offline_worker.py"),
        "worker_sha256": support_sha["run_full_offline_worker.py"],
        "projection_path": str(support / "project_full_offline_result.py"),
        "projection_sha256": _sha(support / "project_full_offline_result.py"),
        "coordinator_sha256": _sha(support / "run_full_offline_comparison.py"),
        "pairs": pairs,
    }
    path = plans / "comparison.json"
    digest = _write_json(path, comparison)
    print(json.dumps({
        "plan": str(path), "plan_sha256": digest,
        "commits": commits, "fingerprints": fingerprints,
        "coordinator": str(support / "run_full_offline_comparison.py"),
        "model_calls": 0,
    }, sort_keys=True, indent=1))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--control-commit", required=True)
    parser.add_argument("--treatment-commit", required=True)
    parser.add_argument("--inputs",
                        help="an existing strict input package-set directory")
    parser.add_argument("--general-inputs",
                        help="a general offline evaluation package root "
                             "(operator-manifest.json + model-inputs/)")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--runtime-read-root", action="append")
    build(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
