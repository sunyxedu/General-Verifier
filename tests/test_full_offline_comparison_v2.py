"""No-model tests for the same-source full-offline v2 evaluator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.terminal_bench import run_full_offline_comparison_v2 as comparison
from experiments.terminal_bench import run_full_offline_worker_v2 as worker


def _write(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _fixture_plan(tmp_path, monkeypatch):
    source = tmp_path / "frozen-source"
    source.mkdir(parents=True)
    worker_file = tmp_path / "worker.py"
    projection_file = tmp_path / "projection.py"
    worker_file.write_text("# pinned worker fixture\n")
    projection_file.write_text("# pinned projection fixture\n")
    fake_worker = SimpleNamespace(
        SCHEMA="gv-full-offline-worker-plan-v2", RUNTIME=comparison.RUNTIME,
        validate_plan=lambda value: value)
    fake_projection = SimpleNamespace(project_full_offline_result=lambda *a: {})
    monkeypatch.setattr(
        comparison, "_load_pinned",
        lambda name, path, digest: (
            fake_worker if name == "worker" else fake_projection))
    source_identity = {
        "root": str(source.resolve()), "commit": "a" * 40,
        "fingerprint": "b" * 64,
    }
    pairs = []
    ordinal = 0
    for family in ("documents", "filesystem", "records"):
        for label in ("positive", "negative"):
            ordinal += 1
            candidate = f"candidate-{ordinal}"
            inputs = tmp_path / "inputs" / candidate
            inputs.mkdir(parents=True)
            task = inputs / "task.md"
            task.write_text("opaque task fixture\n")
            input_seal = hashlib.sha256(candidate.encode()).hexdigest()
            truth_path = tmp_path / "truth" / f"{candidate}.json"
            truth_sha = _write(truth_path, {
                "schema": comparison.TRUTH_SCHEMA,
                "candidate_id": candidate, "task_family": family,
                "label": label, "input_seal": input_seal,
                "official_identity": hashlib.sha256(
                    (candidate + ":official").encode()).hexdigest(),
                "source_truth_manifest_sha256": "9" * 64,
            })
            jobs = []
            order = ("control", "treatment") if ordinal % 2 else (
                "treatment", "control")
            for arm in order:
                worker_plan = {
                    "schema": "gv-full-offline-worker-plan-v2",
                    "source_root": source_identity["root"],
                    "source_commit": source_identity["commit"],
                    "source_fingerprint": source_identity["fingerprint"],
                    "input_source": str(inputs.resolve()),
                    "input_seal": input_seal,
                    "output_root": f"/Users/gv-offline-v2-{tmp_path.name}-{ordinal}-{arm}",
                    "arm": arm, "runtime_read_roots": [],
                    "support_sha256": {"fixture": "c" * 64},
                }
                plan_path = tmp_path / "plans" / f"{candidate}-{arm}.json"
                jobs.append({
                    "arm": arm, "plan_path": str(plan_path.resolve()),
                    "plan_sha256": _write(plan_path, worker_plan),
                })
            pairs.append({
                "candidate_id": candidate, "task_family": family,
                "official_truth": {
                    "path": str(truth_path.resolve()), "sha256": truth_sha,
                },
                "input_attestations": {
                    str(task.resolve()): hashlib.sha256(task.read_bytes()).hexdigest(),
                },
                "jobs": jobs,
            })
    return {
        "schema": comparison.SCHEMA, "comparison_id": "fixture-v2",
        "regime": "cold",
        "output_root": f"/Users/gv-offline-v2-coordinator-{tmp_path.name}",
        "source": source_identity,
        "worker_path": str(worker_file.resolve()),
        "worker_sha256": comparison._sha(worker_file),
        "projection_path": str(projection_file.resolve()),
        "projection_sha256": comparison._sha(projection_file),
        "coordinator_sha256": comparison._sha(comparison.__file__),
        "accounting_helper_sha256": comparison._sha(
            comparison._accounting.__file__),
        "pairs": pairs,
    }


def test_plan_enforces_same_current_source_and_three_balanced_families(
        tmp_path, monkeypatch):
    plan = _fixture_plan(tmp_path, monkeypatch)
    _worker, _projection, pairs = comparison.validate_plan(plan)
    assert len(pairs) == 6
    assert {pair["task_family"] for pair in pairs} == {
        "documents", "filesystem", "records"}
    assert all(job["worker_plan"]["source_commit"] == "a" * 40
               for pair in pairs for job in pair["jobs"])


def test_plan_rejects_source_confounding_and_non_arm_differences(
        tmp_path, monkeypatch):
    plan = _fixture_plan(tmp_path, monkeypatch)
    job = plan["pairs"][0]["jobs"][1]
    worker_plan = json.loads(Path(job["plan_path"]).read_text())
    worker_plan["source_commit"] = "d" * 40
    job["plan_sha256"] = _write(Path(job["plan_path"]), worker_plan)
    with pytest.raises(ValueError, match="exact source"):
        comparison.validate_plan(plan)

    plan = _fixture_plan(tmp_path / "other", monkeypatch)
    job = plan["pairs"][0]["jobs"][1]
    worker_plan = json.loads(Path(job["plan_path"]).read_text())
    worker_plan["support_sha256"] = {"fixture": "e" * 64}
    job["plan_sha256"] = _write(Path(job["plan_path"]), worker_plan)
    with pytest.raises(ValueError, match="beyond arm/output"):
        comparison.validate_plan(plan)


def test_plan_requires_positive_and_negative_in_each_family(tmp_path, monkeypatch):
    plan = _fixture_plan(tmp_path, monkeypatch)
    pair = plan["pairs"][0]
    truth = json.loads(Path(pair["official_truth"]["path"]).read_text())
    truth["label"] = "negative"
    pair["official_truth"]["sha256"] = _write(
        Path(pair["official_truth"]["path"]), truth)
    with pytest.raises(ValueError, match="positive/negative"):
        comparison.validate_plan(plan)


def test_truth_is_external_and_never_enters_worker_command(tmp_path, monkeypatch):
    plan = _fixture_plan(tmp_path, monkeypatch)
    _worker, _projection, pairs = comparison.validate_plan(plan)
    pair, job = pairs[0], pairs[0]["jobs"][0]
    rendered_plan = json.dumps(job["worker_plan"], sort_keys=True)
    command = comparison._worker_command(plan, job)
    assert pair["truth_sha256"] not in rendered_plan
    assert pair["truth"]["label"] not in rendered_plan
    assert pair["truth_sha256"] not in " ".join(command)
    assert pair["truth"]["label"] not in " ".join(command)

    raw_pair = plan["pairs"][0]
    input_truth = Path(json.loads(
        Path(raw_pair["jobs"][0]["plan_path"]).read_text())["input_source"]
        ) / "official-truth.json"
    raw_pair["official_truth"]["path"] = str(input_truth.resolve())
    raw_pair["official_truth"]["sha256"] = _write(input_truth, {
        "schema": comparison.TRUTH_SCHEMA,
        "candidate_id": raw_pair["candidate_id"],
        "task_family": raw_pair["task_family"], "label": "positive",
        "input_seal": json.loads(Path(
            raw_pair["jobs"][0]["plan_path"]).read_text())["input_seal"],
        "official_identity": "8" * 64,
        "source_truth_manifest_sha256": "9" * 64,
    })
    with pytest.raises(ValueError, match="outside model-visible"):
        comparison.validate_plan(plan)


def _job(verdict, *, complete=True, eligible=True):
    view = None if verdict is None else {
        "verdict": verdict, "ordered_contract": [{"id": "o"}],
        "ordered_outcomes": [{"status": verdict}],
    }
    return {
        "worker_complete": complete, "accounting_complete": complete,
        "truth_eligible": eligible, "quality_view": view,
    }


def test_no_regression_and_absolute_truth_are_reported_separately():
    same_error = comparison.compare_pair(
        _job("refuted"), _job("refuted"), {"label": "positive"})
    assert same_error["no_regression_pass"] is True
    assert same_error["absolute_truth_pass"] is False
    assert same_error["truth_correct"] == {
        "control": False, "treatment": False}

    regression = comparison.compare_pair(
        _job("verified"), _job("refuted"), {"label": "positive"})
    assert regression["no_regression_pass"] is False
    improvement = comparison.compare_pair(
        _job("refuted"), _job("verified"), {"label": "positive"})
    assert improvement["no_regression_pass"] is True
    assert improvement["absolute_truth_pass"] is False


@pytest.mark.parametrize("verdict", ["partial", "unverified"])
def test_partial_and_unverified_never_pass_quality(verdict):
    report = comparison.compare_pair(
        _job(verdict, eligible=False), _job(verdict, eligible=False),
        {"label": "positive"})
    assert report["complete"] is False
    assert report["no_regression_pass"] is False
    assert report["absolute_truth_pass"] is False


def test_unknown_cost_makes_ten_x_gate_indeterminate():
    cost = {"complete": False, "control_over_treatment": None}
    latency = {"complete": True, "control_over_treatment": 20.0}
    assert comparison._ten_x_gate(True, cost, latency) is None
    assert comparison._ten_x_gate(False, cost, latency) is False


def test_stratum_does_not_drop_a_missing_candidate(monkeypatch):
    monkeypatch.setattr(comparison._accounting, "_accounting_summary",
                        lambda jobs, expected_jobs: {
                            "complete": False, "control_over_treatment": None,
                            "expected_jobs": expected_jobs})
    report = comparison._stratum_report(
        "documents|positive", ["a", "b"], [], [{
            "candidate_id": "a", "no_regression_pass": True,
            "absolute_truth_pass": True,
        }])
    assert report["no_regression_pass"] is False
    assert report["all_truth_correct"] is False
    assert report["ten_x_no_regression_acceptance"] is False


def test_warm_regime_is_explicitly_rejected(tmp_path, monkeypatch):
    plan = _fixture_plan(tmp_path, monkeypatch)
    plan["regime"] = "warm"
    with pytest.raises(ValueError, match="warm is fail-closed"):
        comparison.validate_plan(plan)


def test_general_package_manifests_materialize_without_task_logic(tmp_path):
    model_root = tmp_path / "model"
    state = model_root / "cases/case-1/gv"
    state.mkdir(parents=True)
    seal = "a" * 64
    visible_case = {
        "case_id": "case-1", "task_family": "arbitrary-family",
        "candidate_id": "opaque-candidate", "state": "cases/case-1/gv",
        "input_seal": seal,
    }
    model_path = model_root / "input-manifest.json"
    model_sha = _write(model_path, {
        "schema": "gv-general-offline-model-inputs-v1",
        "package_id": "package", "cases": [visible_case],
    })
    truth_path = tmp_path / "truth/truth-manifest.json"
    truth_sha = _write(truth_path, {
        "schema": "gv-general-offline-coordinator-truth-v1",
        "package_id": "package", "operator_manifest_sha256": "b" * 64,
        "model_input_manifest_sha256": model_sha,
        "model_input_root": str(model_root.resolve()),
        "cases": [{
            "case_id": "case-1", "task_family": "arbitrary-family",
            "candidate_id": "opaque-candidate", "input_seal": seal,
            "label": 1.0, "identity": "c" * 64,
            "provenance": {"kind": "retained-but-not-model-visible"},
        }],
    })
    output = tmp_path / "materialized"
    result = comparison.materialize_package_records(
        str(model_path.resolve()), model_sha, str(truth_path.resolve()),
        truth_sha, str(output.resolve()))
    assert result["schema"] == comparison.MATERIALIZED_RECORDS_SCHEMA
    record = result["records"][0]
    assert record["input_source"] == str(state.resolve())
    retained = json.loads(Path(record["official_truth"]["path"]).read_text())
    assert retained["label"] == "positive"
    assert retained["official_identity"] == "c" * 64
    assert retained["source_truth_manifest_sha256"] == truth_sha


def _worker_plan(tmp_path, monkeypatch, arm="control"):
    source, inputs = tmp_path / "source", tmp_path / "inputs"
    source.mkdir(parents=True)
    inputs.mkdir()
    monkeypatch.setattr(worker, "_sha", lambda _: "a" * 64)
    return {
        "schema": worker.SCHEMA, "source_root": str(source.resolve()),
        "source_commit": "c" * 40, "source_fingerprint": "b" * 64,
        "input_source": str(inputs.resolve()), "input_seal": "d" * 64,
        "output_root": f"/Users/gv-worker-v2-{tmp_path.name}-{arm}",
        "arm": arm, "runtime_read_roots": [],
        "support_sha256": {name: "a" * 64 for name in worker.SUPPORT_FILES},
    }


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_worker_accepts_same_current_source_for_both_arms(
        arm, tmp_path, monkeypatch):
    plan = _worker_plan(tmp_path, monkeypatch, arm)
    assert worker.validate_plan(plan) is plan


@pytest.mark.parametrize("certificate", [
    None, {}, {"certificate": None}, {"certificate": {}},
    {"certificate": {"verdict": None}},
    {"certificate": {"verdict": "unknown"}},
    {"certificate": {"verdict": "partial"}},
    {"certificate": {"verdict": "unverified"}},
])
def test_worker_truth_eligibility_is_closed_for_missing_or_unsettled_verdict(
        certificate):
    assert worker._truth_eligible(certificate) is False


@pytest.mark.parametrize("verdict", ["verified", "supported", "refuted"])
def test_worker_truth_eligibility_accepts_only_settled_verdicts(verdict):
    assert worker._truth_eligible({
        "certificate": {"verdict": verdict}}) is True


@pytest.mark.parametrize("arm,flag", [("control", "0"), ("treatment", "1")])
def test_worker_environment_changes_only_general_optimization_flags(
        arm, flag, tmp_path, monkeypatch):
    plan = _worker_plan(tmp_path, monkeypatch, arm)
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "ambient")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "ambient")
    worker._environment(plan)
    assert worker.os.environ["GV_TYPED_ADAPTERS"] == flag
    assert worker.os.environ["GV_TYPED_JUDGER_PACKET"] == flag
    assert worker.os.environ["GV_CONTRACT_CACHE"] == "0"
    assert worker.os.environ["GV_APPEND_ONLY_DELTAS"] == "0"


@pytest.mark.parametrize(
    "verdict,eligible",
    [("verified", True), ("supported", True), ("refuted", True),
     ("partial", False), ("unverified", False), (None, False),
     ("unknown", False)],
)
def test_worker_truth_eligibility_is_an_explicit_settled_allowlist(
        verdict, eligible):
    certificate = {"certificate": {"verdict": verdict}}
    assert worker._truth_eligible(certificate) is eligible


def _put_artifact(state: Path, value) -> str:
    digest = worker._content_digest(value)
    root = state / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{digest}.json").write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")))
    return digest


def _architecture_fixture(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    final_text = "The delivered candidate satisfies the task."
    final = worker._content_digest({
        "kind": "final", "text": final_text, "deps": []})
    compiled_obligation = {
        "text": "subject is valid", "required_basis": "world",
        "refutable_by_absence": False, "premise_scope": [],
        "severity": "must", "clause": "The subject must be valid.",
    }
    oid = worker._content_digest({
        "schema": "gv-obligation-v2", "text": "subject is valid",
        "required": "world", "refutable_by_absence": False,
        "premise_scope": [],
    })
    compiled_obligation["id"] = oid
    certificate_id = "c" * 64
    method = {"procedure": "inspect the complete subject", "level": "exact",
              "strength": "decisive"}
    audit = {"sufficient": "yes", "faithful": "yes",
             "degradation": "n/a", "reason": "fresh review"}
    execution = _put_artifact(state, {
        "instrument": "agent:discharge", "reported": {}, "files": {}})
    binding = {
        "instrument": "binding-game", "obligation": "subject is valid",
        "attempts": [{"method": method, "discharge_artifact": execution,
                      "audit": audit}],
        "bound": {"method": method, "discharge_artifact": execution,
                  "audit": audit},
    }
    binding_digest = _put_artifact(state, binding)
    evidence = {"artifact": binding_digest, "evidence_id": "evidence-1",
                "outcome": "holds", "basis": "world",
                "strength": "decisive"}
    certificate = {
        "final": final, "verdict": "verified", "verdict_strength": "decisive",
        "slice": [{"id": final, "kind": "final", "text": final_text,
                   "obligations": [{
            **compiled_obligation, "status": "verified",
            "evidence": [evidence],
        }]}],
    }
    events = [
        {"type": "commit", "payload": {
            "label": "final", "kind": "final", "text": final_text,
            "deps": []}},
        {"type": "decomposition_attacked", "payload": {
            "claim": final, "verdict": "pass"}},
        {"type": "obligations_compiled", "payload": {
            "claim": final, "obligations": [compiled_obligation]}},
        {"type": "methods_enumerated", "payload": {
            "claim": final, "obligation": oid, "methods": [method]}},
        {"type": "judger_verdict", "payload": {
            "obligation": oid, "procedure": method["procedure"], **audit}},
        {"type": "method_bound", "payload": {
            "obligation": oid, "procedure": method["procedure"]}},
        {"type": "discharge", "payload": {
            "obligation": oid, "artifact": binding_digest,
            "evidence_id": "evidence-1"}},
        {"type": "certificate", "payload": {
            "final": final, "digest": certificate_id}},
    ]
    for seq, event in enumerate(events, 1):
        event["seq"] = seq
    (state / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events))
    return state, certificate, certificate_id


def test_worker_architecture_audit_requires_complete_fresh_lineage(tmp_path):
    state, certificate, certificate_id = _architecture_fixture(tmp_path)
    report = worker._audit_architecture(state, certificate, certificate_id)
    assert report["lineages"][0]["execution_kind"] == "agent"
    rows, _ = worker._read_jsonl(state / "events.jsonl")
    rows = [row for row in rows if row["type"] != "decomposition_attacked"]
    for seq, row in enumerate(rows, 1):
        row["seq"] = seq
    (state / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Critic pass"):
        worker._audit_architecture(state, certificate, certificate_id)
