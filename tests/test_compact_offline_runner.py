"""No-model tests for explicit compact inspection in the offline runner."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.common import scoring
from experiments.terminal_bench import run_typed_adapter_eval as runner
from experiments.terminal_bench import score


MANIFEST = (
    Path(__file__).parents[1]
    / "experiments/terminal_bench/typed_adapter_eval_manifest.json"
)
RUNTIME_ID = "a" * 64


def _compact_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    environment = {
        "GV_TYPED_ADAPTERS": "1",
        "GV_CONTRACT_CACHE": "0",
        "GV_APPEND_ONLY_DELTAS": "0",
    }
    manifest["comparisons"]["compact-inspection"] = {
        "acceptance_eligible": False,
        "control": {**environment, "compact_inspection": False},
        "treatment": {**environment, "compact_inspection": True},
    }
    return manifest


def _attempt(index=1, **updates):
    row = {
        "schema_version": 1,
        "record_type": "codex_cli_attempt",
        "attempt_schema_version": 1,
        "call_id": f"call-{index}",
        "attempt_id": f"attempt-{index}",
        "attempt_index": 1,
        "started": True,
        "attempt_status": "succeeded",
        "usage_status": "complete",
        "usage_complete": True,
        "input_tokens": 10,
        "uncached_input_tokens": 6,
        "cached_input_tokens": 4,
        "output_tokens": 3,
        "compact_inspection": False,
        "runtime_instruction_identity": RUNTIME_ID,
    }
    row.update(updates)
    return row


def test_compact_plan_changes_only_explicit_cli_profile(tmp_path):
    manifest = _compact_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    parsed = runner._manifest(path)
    selected = runner._selected(parsed, "canary")
    rows = runner._plan_rows(
        parsed, MANIFEST.parents[2], "compact-inspection", "canary",
        ("warm",), ("control", "treatment"), tmp_path / "output")

    assert rows
    by_arm = {arm: [row for row in rows if row["arm"] == arm]
              for arm in ("control", "treatment")}
    assert len(by_arm["control"]) == len(by_arm["treatment"])
    for arm, compact in (("control", False), ("treatment", True)):
        for row in by_arm[arm]:
            assert row["compact_inspection"] is compact
            position = row["command"].index("--compact-inspection")
            assert row["command"][position + 1] == ("1" if compact else "0")
            assert row["environment"] == {
                "GV_TYPED_ADAPTERS": "1",
                "GV_CONTRACT_CACHE": "0",
                "GV_APPEND_ONLY_DELTAS": "0",
            }
    assert len(selected) == 6


def test_compact_comparison_rejects_a_second_policy_change(tmp_path):
    manifest = _compact_manifest()
    manifest["comparisons"]["compact-inspection"]["treatment"][
        "GV_CONTRACT_CACHE"] = "1"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(runner.PlanError, match="together with GV settings"):
        runner._manifest(path)


@pytest.mark.parametrize("value", [0, 1, "1", None])
def test_compact_manifest_requires_a_real_boolean(tmp_path, value):
    manifest = _compact_manifest()
    manifest["comparisons"]["compact-inspection"]["treatment"][
        "compact_inspection"] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(runner.PlanError, match="must be boolean"):
        runner._manifest(path)


def test_attempt_audit_binds_profile_and_runtime_identity(tmp_path):
    usage = tmp_path / "usage.jsonl"
    usage.write_text(json.dumps(_attempt()) + "\n", encoding="utf-8")
    assert runner._attempt_audit(
        usage, compact_inspection=False) == (True, [])

    usage.write_text(json.dumps(_attempt(compact_inspection=True)) + "\n",
                     encoding="utf-8")
    valid, errors = runner._attempt_audit(
        usage, compact_inspection=False)
    assert valid is False
    assert "differs from the arm" in " ".join(errors)

    usage.write_text("".join(json.dumps(row) + "\n" for row in (
        _attempt(1), _attempt(2, runtime_instruction_identity="b" * 64),
    )), encoding="utf-8")
    valid, errors = runner._attempt_audit(
        usage, compact_inspection=False)
    assert valid is False
    assert "mixes runtime instruction identities" in " ".join(errors)

    usage.write_text(json.dumps([
        "not", "an", {"runtime_instruction_identity": []},
    ]) + "\n", encoding="utf-8")
    valid, errors = runner._attempt_audit(
        usage, compact_inspection=False)
    assert valid is False
    assert "non-object row" in " ".join(errors)
    assert runner._attempt_runtime_identity(usage) is None


def test_malformed_result_rows_are_diagnostic_not_exceptions(tmp_path):
    result = tmp_path / "results.jsonl"
    result.write_text("[]\n" + json.dumps({
        "trial": "task__one", "features": [],
    }) + "\n", encoding="utf-8")

    errors = runner._result_audit(
        result, ("task__one",), compact_inspection=False)

    assert "non-object row" in " ".join(errors)
    assert "malformed features" in " ".join(errors)


def test_score_many_forwards_compact_without_changing_schedule(
        tmp_path, monkeypatch):
    calls = []

    def fake_score(state, **kwargs):
        calls.append((state, kwargs))
        return {"unscoreable": True}

    monkeypatch.setattr(scoring.offline, "score", fake_score)
    jobs = [({"trial": "task__one"}, str(tmp_path / "state"))]
    records = scoring.score_many(
        jobs, str(tmp_path / "results.jsonl"), workers=1, heartbeat=0,
        model="gpt-5.6-luna", seats="codex", effort="medium",
        service_tier="standard", compact_inspection=True)

    assert len(records) == 1
    assert len(calls) == 1
    assert calls[0][1]["compact_inspection"] is True
    assert calls[0][1]["model"] == "gpt-5.6-luna"
    assert calls[0][1]["effort"] == "medium"
    assert calls[0][1]["service_tier"] == "standard"


def test_score_cli_passes_explicit_compact_to_config(
        tmp_path, monkeypatch):
    captured = []
    original = score.offline.config_fields

    def config_fields(**kwargs):
        captured.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(score.offline, "config_fields", config_fields)
    monkeypatch.setattr(score.harborjobs, "trials", lambda _path: [])
    monkeypatch.setattr(score.res, "recorded", lambda *_args: set())
    monkeypatch.setattr(score.res, "load", lambda *_args: [])
    monkeypatch.setattr(sys, "argv", [
        "score.py", "--job", "fixture", "--seats", "codex",
        "--model", "gpt-5.6-luna", "--effort", "medium",
        "--service-tier", "standard", "--compact-inspection", "1",
        "--scores-path", str(tmp_path / "results.jsonl"),
    ])

    score.main()

    assert len(captured) == 1
    assert captured[0]["compact_inspection"] is True


def test_run_record_and_child_environment_are_unambiguous(
        tmp_path, monkeypatch):
    manifest = _compact_manifest()
    group = runner.Group(
        "cold-job-one", "job", ("task",), ("task__one",))
    source_root = MANIFEST.parents[2]
    fingerprint = runner._source_fingerprint(source_root)
    captured = {}

    def fake_run(command, *, cwd, env, stdout, stderr, text, check):
        captured.update(command=list(command), cwd=cwd, env=dict(env))
        scores = Path(command[command.index("--scores-path") + 1])
        scores.write_text(json.dumps({
            "trial": "task__one",
            "compact_inspection": True,
            "features": {
                "compact_inspection": True,
                "contract_audit_status": "missing",
                "final_obligation_contract": [],
                "final_obligation_outcomes_ordered": [],
            },
        }) + "\n", encoding="utf-8")
        Path(str(scores) + ".usage.jsonl").write_text(
            json.dumps(_attempt(compact_inspection=True)) + "\n",
            encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    record = runner._run_one(
        manifest=manifest, comparison="compact-inspection", tier="canary",
        phase="cold", arm="treatment", group=group,
        output=tmp_path / "output", source_root=source_root,
        source_commit="1" * 40, source_fingerprint=fingerprint,
        manifest_sha256="2" * 64)

    assert record["compact_inspection"] is True
    assert record["runtime_instruction_identity"] == RUNTIME_ID
    assert record["usage_audit_complete"] is True
    assert captured["env"].get("compact_inspection") is None
    assert captured["env"].get("GV_COMPACT_INSPECTION") is None
    position = captured["command"].index("--compact-inspection")
    assert captured["command"][position + 1] == "1"
    persisted = json.loads(
        (tmp_path / "output/eval-runs.jsonl").read_text())
    assert persisted["compact_inspection"] is True
    assert persisted["runtime_instruction_identity"] == RUNTIME_ID


def test_result_audit_does_not_treat_legacy_missing_as_false(tmp_path):
    result = tmp_path / "results.jsonl"
    base = {
        "trial": "task__one",
        "features": {
            "contract_audit_status": "missing",
            "final_obligation_contract": [],
            "final_obligation_outcomes_ordered": [],
        },
    }
    result.write_text(json.dumps(base) + "\n", encoding="utf-8")
    assert runner._result_audit(result, ("task__one",)) == []
    errors = runner._result_audit(
        result, ("task__one",), compact_inspection=False)
    assert "differs from the arm" in " ".join(errors)

    current = copy.deepcopy(base)
    current["compact_inspection"] = False
    current["features"]["compact_inspection"] = False
    result.write_text(json.dumps(current) + "\n", encoding="utf-8")
    assert runner._result_audit(
        result, ("task__one",), compact_inspection=False) == []
