"""No-model tests for the task-general full-offline v2 plan builder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from experiments.terminal_bench import build_full_offline_comparison_v2_plan as builder


def _write(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _records(tmp_path: Path):
    model_root = tmp_path / "model-inputs"
    truth_root = tmp_path / "coordinator-only"
    records = []
    ordinal = 0
    for family in ("opaque-alpha", "opaque-beta", "opaque-gamma"):
        for label in ("positive", "negative"):
            ordinal += 1
            candidate = f"candidate-{ordinal}"
            inputs = model_root / f"case-{ordinal}" / "gv"
            inputs.mkdir(parents=True)
            (inputs / "task.md").write_text(f"opaque {ordinal}\n")
            (inputs / "world_manifest.json").write_text("{}\n")
            seal = hashlib.sha256(candidate.encode()).hexdigest()
            truth_path = truth_root / f"truth-{ordinal}.json"
            truth_sha = _write(truth_path, {
                "schema": "gv-retained-official-truth-v1",
                "candidate_id": candidate, "task_family": family,
                "label": label, "input_seal": seal,
                "official_identity": hashlib.sha256(
                    f"official-{ordinal}".encode()).hexdigest(),
                "source_truth_manifest_sha256": "9" * 64,
            })
            records.append({
                "case_id": f"case-{ordinal}", "candidate_id": candidate,
                "task_family": family, "input_source": str(inputs.resolve()),
                "input_seal": seal,
                "official_truth": {
                    "path": str(truth_path.resolve()), "sha256": truth_sha},
            })
    path = tmp_path / "materialized" / "records.json"
    digest = _write(path, {
        "schema": "gv-offline-v2-materialized-records-v1",
        "package_id": "opaque-package",
        "model_input_manifest_sha256": "8" * 64,
        "truth_manifest_sha256": "9" * 64, "records": records,
    })
    return path, digest, records


def _build(tmp_path, monkeypatch, **overrides):
    records_path, records_sha, records = _records(tmp_path)
    source = tmp_path / "frozen-source"
    source.mkdir()
    support = source / "experiments" / "terminal_bench"
    support.mkdir(parents=True)
    support_names = {
        *builder.worker.SUPPORT_FILES,
        "project_full_offline_result.py",
        "run_full_offline_comparison_v2.py",
        "run_full_offline_comparison.py",
    }
    for name in support_names:
        (support / name).write_text(f"# frozen fixture {name}\n")
    run_root = tmp_path / "persistent-eval"
    # pytest's temp root is intentionally forbidden in production.  Isolate
    # that host-path policy here while preserving all freshness checks.
    monkeypatch.setattr(builder.tempfile, "gettempdir", lambda: "/not-test-root")
    original_inside = builder._inside
    monkeypatch.setattr(
        builder, "_inside",
        lambda parent, child: False
        if str(parent) in {"/private/var/folders", "/private/tmp", "/tmp"}
        else original_inside(parent, child))
    values = {
        "records_path": str(records_path.resolve()),
        "records_sha256": records_sha,
        "source_root": str(source.resolve()), "source_commit": "a" * 40,
        "run_root": str(run_root.resolve()), "comparison_id": "opaque-run",
        "runtime_read_roots": ["/usr/bin"],
        "attestation_paths": ["world_manifest.json"],
    }
    values.update(overrides)
    return builder.build_plan(**values), records, run_root


def test_builds_same_source_pinned_balanced_truth_isolated_plans(
        tmp_path, monkeypatch):
    result, records, run_root = _build(tmp_path, monkeypatch)
    comparison_plan = json.loads(
        Path(result["comparison_plan_path"]).read_text())
    assert comparison_plan["source"]["root"].endswith("frozen-source")
    assert len(comparison_plan["source"]["fingerprint"]) == 64
    assert comparison_plan["coordinator_sha256"] == builder._sha(
        Path(comparison_plan["source"]["root"]) / "experiments"
        / "terminal_bench" / "run_full_offline_comparison_v2.py")
    assert comparison_plan["projection_sha256"] == builder._sha(
        Path(comparison_plan["projection_path"]))
    assert comparison_plan["accounting_helper_sha256"] == builder._sha(
        Path(comparison_plan["source"]["root"]) / "experiments"
        / "terminal_bench" / "run_full_offline_comparison.py")
    first_arms = [pair["jobs"][0]["arm"]
                  for pair in comparison_plan["pairs"]]
    assert first_arms == [
        "control", "treatment", "control",
        "treatment", "control", "treatment"]
    for pair, record in zip(comparison_plan["pairs"], records):
        assert pair["official_truth"] == record["official_truth"]
        assert {Path(path).relative_to(record["input_source"]).as_posix()
                for path in pair["input_attestations"]} == {
                    "task.md", "world_manifest.json"}
        for job in pair["jobs"]:
            worker_plan = json.loads(Path(job["plan_path"]).read_text())
            assert worker_plan["source_root"] == comparison_plan["source"]["root"]
            assert worker_plan["source_commit"] == "a" * 40
            assert worker_plan["source_fingerprint"] == comparison_plan[
                "source"]["fingerprint"]
            assert worker_plan["runtime_read_roots"] == ["/usr/bin"]
            rendered = json.dumps(worker_plan, sort_keys=True)
            assert "official_truth" not in rendered
            assert record["official_truth"]["path"] not in rendered
            assert record["official_truth"]["sha256"] not in rendered
            assert not Path(worker_plan["output_root"]).exists()
    assert not Path(comparison_plan["output_root"]).exists()
    assert (run_root.stat().st_mode & 0o777) == 0o700
    assert all((path.stat().st_mode & 0o777) == 0o600
               for path in (run_root / "plans").rglob("*.json"))


def test_arbitrary_family_names_use_identical_schema_only_flow(
        tmp_path, monkeypatch):
    result, _records_value, _run_root = _build(tmp_path, monkeypatch)
    plan = json.loads(Path(result["comparison_plan_path"]).read_text())
    assert [pair["task_family"] for pair in plan["pairs"]] == [
        "opaque-alpha", "opaque-alpha", "opaque-beta", "opaque-beta",
        "opaque-gamma", "opaque-gamma"]


def test_rejects_truth_inside_model_inputs_before_writing(tmp_path, monkeypatch):
    records_path, digest, records = _records(tmp_path)
    record = records[0]
    truth_path = Path(record["input_source"]) / "truth.json"
    record["official_truth"] = {
        "path": str(truth_path.resolve()),
        "sha256": _write(truth_path, {
            "schema": "gv-retained-official-truth-v1",
            "candidate_id": record["candidate_id"],
            "task_family": record["task_family"], "label": "positive",
            "input_seal": record["input_seal"],
            "official_identity": "7" * 64,
            "source_truth_manifest_sha256": "9" * 64,
        }),
    }
    manifest = json.loads(records_path.read_text())
    manifest["records"] = records
    digest = _write(records_path, manifest)
    source = tmp_path / "source"; source.mkdir()
    run_root = tmp_path / "run"
    monkeypatch.setattr(builder, "_fresh_root", lambda value: Path(value))
    with pytest.raises(builder.PlanBuildError, match="official truth overlaps"):
        builder.build_plan(
            records_path=str(records_path.resolve()), records_sha256=digest,
            source_root=str(source.resolve()), source_commit="a" * 40,
            run_root=str(run_root.resolve()), comparison_id="run",
            runtime_read_roots=["/usr/bin"])
    assert not run_root.exists()


def test_rejects_truth_hidden_in_a_different_candidates_input(
        tmp_path, monkeypatch):
    records_path, _digest, records = _records(tmp_path)
    record = records[0]
    truth_path = Path(records[1]["input_source"]) / "cross-case-truth.json"
    record["official_truth"] = {
        "path": str(truth_path.resolve()),
        "sha256": _write(truth_path, {
            "schema": "gv-retained-official-truth-v1",
            "candidate_id": record["candidate_id"],
            "task_family": record["task_family"], "label": "positive",
            "input_seal": record["input_seal"],
            "official_identity": "7" * 64,
            "source_truth_manifest_sha256": "9" * 64,
        }),
    }
    manifest = json.loads(records_path.read_text())
    manifest["records"] = records
    digest = _write(records_path, manifest)
    source = tmp_path / "source"; source.mkdir()
    run_root = tmp_path / "run"
    monkeypatch.setattr(builder, "_fresh_root", lambda value: Path(value))
    with pytest.raises(builder.PlanBuildError, match="outside every input"):
        builder.build_plan(
            records_path=str(records_path.resolve()), records_sha256=digest,
            source_root=str(source.resolve()), source_commit="a" * 40,
            run_root=str(run_root.resolve()), comparison_id="run",
            runtime_read_roots=["/usr/bin"])
    assert not run_root.exists()


def test_rejects_missing_relative_attestation_and_runtime_roots(
        tmp_path, monkeypatch):
    records_path, digest, _records_value = _records(tmp_path)
    source = tmp_path / "source"; source.mkdir()
    run_root = tmp_path / "run"
    monkeypatch.setattr(builder, "_fresh_root", lambda value: Path(value))
    with pytest.raises(builder.PlanBuildError, match="missing input attestation"):
        builder.build_plan(
            records_path=str(records_path.resolve()), records_sha256=digest,
            source_root=str(source.resolve()), source_commit="a" * 40,
            run_root=str(run_root.resolve()), comparison_id="run",
            runtime_read_roots=["/usr/bin"],
            attestation_paths=["not-present.txt"])
    with pytest.raises(builder.PlanBuildError, match="explicitly provided"):
        builder.build_plan(
            records_path=str(records_path.resolve()), records_sha256=digest,
            source_root=str(source.resolve()), source_commit="a" * 40,
            run_root=str(run_root.resolve()), comparison_id="run",
            runtime_read_roots=[])


def test_rejects_stale_records_source_pin_and_nonpersistent_root(
        tmp_path, monkeypatch):
    records_path, digest, _records_value = _records(tmp_path)
    source = tmp_path / "source"; source.mkdir()
    original_fresh_root = builder._fresh_root
    monkeypatch.setattr(builder, "_fresh_root", lambda value: Path(value))
    with pytest.raises(builder.PlanBuildError, match="records changed"):
        builder.build_plan(
            records_path=str(records_path.resolve()), records_sha256="0" * 64,
            source_root=str(source.resolve()), source_commit="a" * 40,
            run_root=str((tmp_path / "run-a").resolve()), comparison_id="run",
            runtime_read_roots=["/usr/bin"])
    with pytest.raises(builder.PlanBuildError, match="operator pin"):
        builder.build_plan(
            records_path=str(records_path.resolve()), records_sha256=digest,
            source_root=str(source.resolve()), source_commit="a" * 40,
            expected_source_fingerprint="f" * 64,
            run_root=str((tmp_path / "run-b").resolve()), comparison_id="run",
            runtime_read_roots=["/usr/bin"])
    with pytest.raises(builder.PlanBuildError, match="persistent root"):
        original_fresh_root(
            str((Path(tempfile.gettempdir()) / "gv-run").resolve()))
