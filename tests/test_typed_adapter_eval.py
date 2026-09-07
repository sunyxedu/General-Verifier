import json
from pathlib import Path

from experiments.terminal_bench import analyze_typed_adapter_eval as analyze
from experiments.terminal_bench import run_typed_adapter_eval as runner


MANIFEST = (
    Path(__file__).parents[1]
    / "experiments/terminal_bench/typed_adapter_eval_manifest.json"
)


def _attempt(**updates):
    row = {
        "schema_version": 1,
        "record_type": "codex_cli_attempt",
        "attempt_schema_version": 1,
        "call_id": "call-1",
        "attempt_id": "attempt-1",
        "attempt_index": 1,
        "started": True,
        "attempt_status": "succeeded",
        "usage_status": "complete",
        "usage_complete": True,
        "input_tokens": 10,
        "uncached_input_tokens": 6,
        "cached_input_tokens": 4,
        "output_tokens": 3,
    }
    row.update(updates)
    return row


def test_canary_warm_is_one_process_per_task_and_arm():
    manifest = runner._manifest(MANIFEST)
    selected = runner._selected(manifest, "canary")
    groups = runner._groups("warm", selected)

    assert len(groups) == 3
    assert all(len(group.tasks) == 1 for group in groups)
    assert all(len(group.trials) == 2 for group in groups)

    schedule = runner._schedule("warm", groups, ("control", "treatment"))
    assert len(schedule) == 6
    for group in groups:
        assert {arm for arm, item in schedule if item == group} == {
            "control", "treatment",
        }


def test_canary_all_plan_has_every_phase_in_one_ledger():
    manifest = runner._manifest(MANIFEST)
    rows = runner._plan_rows(
        manifest, MANIFEST.parents[2], "full-e2e", "canary",
        runner.PHASES, ("control", "treatment"), Path("/tmp/eval"),
    )

    assert len(rows) == 22
    assert {row["phase"] for row in rows} == set(runner.PHASES)


def test_batch_schedule_is_globally_serial_and_each_arm_contiguous():
    manifest = runner._manifest(MANIFEST)
    groups = runner._groups(
        "batch", runner._selected(manifest, "acceptance"))
    schedule = runner._schedule("batch", groups, ("control", "treatment"))

    arms = [arm for arm, _group in schedule]
    split = len(groups)
    assert len(schedule) == 2 * split
    assert len(set(arms[:split])) == 1
    assert len(set(arms[split:])) == 1
    assert arms[0] != arms[-1]
    assert [group.key for _arm, group in schedule[:split]] == [
        group.key for group in groups
    ]
    assert [group.key for _arm, group in schedule[split:]] == [
        group.key for group in groups
    ]


def test_source_fingerprint_includes_score_local_dependencies(tmp_path):
    paths = (
        "experiments/terminal_bench/score.py",
        "experiments/terminal_bench/harborjobs.py",
        "experiments/common/scoring.py",
        "verifier/offline.py",
        "protocol/solver-protocol.md",
        "pyproject.toml",
        "uv.lock",
    )
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    before = runner._source_fingerprint(tmp_path)

    (tmp_path / "experiments/common/scoring.py").write_text(
        "changed", encoding="utf-8")

    assert runner._source_fingerprint(tmp_path) != before


def test_wrapper_attempt_audit_rejects_unknown_or_inconsistent_cost(tmp_path):
    usage = tmp_path / "usage.jsonl"
    usage.write_text(json.dumps(_attempt()) + "\n", encoding="utf-8")
    assert runner._attempt_audit(usage) == (True, [])

    usage.write_text(json.dumps(_attempt(
        usage_complete=False,
        usage_status="unknown",
        input_tokens=None,
        uncached_input_tokens=None,
        cached_input_tokens=None,
        output_tokens=None,
    )) + "\n", encoding="utf-8")
    valid, errors = runner._attempt_audit(usage)
    assert not valid
    assert "incomplete/invalid token usage" in " ".join(errors)

    usage.write_text(json.dumps(_attempt(
        uncached_input_tokens=7,
    )) + "\n", encoding="utf-8")
    valid, errors = runner._attempt_audit(usage)
    assert not valid
    assert "inconsistent input token usage" in " ".join(errors)


def test_wrapper_result_audit_requires_raw_contract_not_speculative_digest(
        tmp_path):
    result = tmp_path / "results.jsonl"
    result.write_text(json.dumps({
        "trial": "task__candidate",
        "features": {
            "contract_audit_status": "available",
            "final_obligation_contract": [{"ordinal": 0}],
            "final_obligation_outcomes_ordered": [{"ordinal": 0}],
        },
    }) + "\n", encoding="utf-8")

    assert runner._result_audit(result, ("task__candidate",)) == []

    result.write_text(json.dumps({
        "trial": "task__candidate", "features": {},
    }) + "\n", encoding="utf-8")
    assert "raw contract audit" in " ".join(
        runner._result_audit(result, ("task__candidate",)))

    result.write_text(json.dumps({
        "trial": "task__candidate",
        "features": {
            "contract_audit_status": "missing",
            "final_obligation_contract": [],
            "final_obligation_outcomes_ordered": [],
        },
    }) + "\n", encoding="utf-8")
    assert runner._result_audit(result, ("task__candidate",)) == []


def test_analyzer_rejects_paths_outside_a_run_root(tmp_path):
    run = {"run_id": "run-1"}
    inside = tmp_path / "runs/run-1/results.jsonl"
    outside = tmp_path / "other/results.jsonl"

    assert analyze._private_path(
        run, tmp_path, str(inside), "results")[1] is None
    assert analyze._private_path(
        run, tmp_path, str(outside), "results")[1] is not None


def test_analyze_validates_task_and_job_from_current_metadata_row(
        tmp_path, monkeypatch):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    trials = [
        "gcode-to-text__ELuvDeE",
        "gcode-to-text__HFh2qJM",
    ]
    run_id = "warm-control-gcode"
    results = tmp_path / "runs" / run_id / "results.jsonl"
    results.parent.mkdir(parents=True)
    results.write_text("", encoding="utf-8")
    row = {
        "schema": analyze.RUN_SCHEMA,
        "tier": "canary", "comparison": "full-e2e", "phase": "warm",
        "arm": "control", "run_id": run_id, "source_commit": "a" * 40,
        "verifier_fingerprint": "fingerprint",
        "verifier_fingerprint_after": "fingerprint",
        "source_fingerprint_unchanged": True,
        "manifest_sha256": analyze._manifest_sha256(MANIFEST),
        "tasks": ["gcode-to-text"], "job": "lav-official-g2-k5",
        "trials": trials,
        "results_path": str(results.relative_to(tmp_path)),
        "usage_paths": [], "started_at_monotonic_ns": 1,
        "finished_at_monotonic_ns": 2, "wall_seconds": 1e-9,
        "exit_code": 0, "usage_audit_complete": True,
        "usage_audit_errors": [], "result_audit_errors": [],
        "fresh_scoring_root": True, "fresh_process_cache": True,
        "wall_includes_contract_build": True, "shared_toolchain": True,
        "workers": 1,
        **manifest["verifier"],
        **manifest["comparisons"]["full-e2e"]["control"],
    }
    runs = tmp_path / "eval-runs.jsonl"
    runs.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(analyze, "_quality", lambda *args: {
        "status": "pass", "contract_review": {"status": "exact"}})
    monkeypatch.setattr(analyze, "_cost", lambda *args: {"status": "pass"})
    monkeypatch.setattr(analyze, "_latency", lambda *args: {"status": "pass"})

    report = analyze.analyze(
        MANIFEST, runs, "canary", "full-e2e", ("warm",))

    assert not any("task list mismatch" in error for error in report["audit_errors"])
    assert not any("archive job mismatch" in error for error in report["audit_errors"])


def _quality_row(*, text="must equal 1", oid="o1", status="verified"):
    contract = [{
        "ordinal": 0,
        "id": oid,
        "text": text,
        "premise_scope": [],
        "required_basis": "world",
        "severity": "must",
        "clause": "output must equal 1",
        "refutable_by_absence": False,
    }]
    outcomes = [{
        "ordinal": 0,
        "obligation_id": oid,
        "evaluated": True,
        "status": status,
        "reason": "fixture",
        "basis": "world",
        "strength": "decisive",
    }]
    return {
        "reward": 1,
        "features": {
            "feature_schema_version": 3,
            "unscoreable": False,
            "verdict": "verified",
            "verdict_strength": "decisive",
            "final_obligations": {"verified": 1},
            "final_obligations_total": 1,
            "final_contrary_signals": 0,
            "immutable_input_seal": "sealed",
            "contract_audit_status": "available",
            "final_obligation_contract": contract,
            "final_obligation_outcomes_ordered": outcomes,
        },
    }


def test_full_e2e_reports_raw_contract_drift_without_invented_digest_gate():
    trial = "task__candidate"
    control = _quality_row(text="value is at least 1", oid="control-id")
    treatment = _quality_row(text="value is greater than 1", oid="treatment-id")
    report = analyze._quality(
        {
            ("warm", "control", trial): control,
            ("warm", "treatment", trial): treatment,
        },
        {trial: {"immutable_input_seal": "sealed"}},
        "full-e2e",
        ("warm",),
    )

    assert report["status"] == "pass"
    assert report["observed_mismatches"] == []
    assert report["contract_review"]["status"] == "drift"
    assert report["contract_review"]["semantic_equivalence_inferred_from_digest"] \
        is False
    review = report["candidate_reviews"][0]
    assert review["ordered_contract_exact"] is False
    assert review["control"]["ordered_contract"][0]["text"] \
        == "value is at least 1"
    assert review["treatment"]["ordered_contract"][0]["text"] \
        == "value is greater than 1"


def test_equal_contract_compares_aligned_outcomes_not_only_histogram():
    trial = "task__candidate"
    control = _quality_row()
    treatment = _quality_row()
    treatment["features"]["final_obligation_outcomes_ordered"][0][
        "basis"] = "record"
    report = analyze._quality(
        {
            ("cold", "control", trial): control,
            ("cold", "treatment", trial): treatment,
        },
        {trial: {"immutable_input_seal": "sealed"}},
        "full-e2e",
        ("cold",),
    )

    assert report["status"] == "fail"
    assert report["contract_review"]["status"] == "exact"
    assert report["observed_mismatches"][0]["fields"] \
        == ["ordered_obligation_outcomes"]
