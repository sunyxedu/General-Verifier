"""No-model tests for the balanced full-offline comparison coordinator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.terminal_bench import run_full_offline_comparison as comparison


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_plan(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "SCHEMA = 'fake-worker-v1'\n"
        "def validate_plan(plan):\n"
        "    return plan\n")
    projection = tmp_path / "projection.py"
    projection.write_text(
        "def project_full_offline_result(path, certificate):\n"
        "    return {'complete': True}\n")
    attestation = tmp_path / "input-manifest.json"
    attestation.write_text("sealed\n")
    pairs = []
    for pair_index, order in enumerate((("control", "treatment"),
                                        ("treatment", "control"))):
        jobs = []
        for arm in order:
            worker_plan = {
                "arm": arm,
                "input_source": f"/Users/fixture-input-{pair_index}",
                "input_seal": chr(97 + pair_index) * 64,
                "output_root": f"/Users/fixture-worker-{tmp_path.name}-{pair_index}-{arm}",
            }
            plan_path = tmp_path / f"plan-{pair_index}-{arm}.json"
            plan_sha = _write(plan_path, worker_plan)
            jobs.append({"arm": arm, "plan_path": str(plan_path.resolve()),
                         "plan_sha256": plan_sha})
        pairs.append({
            "candidate_id": f"candidate-{pair_index}",
            "input_attestations": {
                str(attestation.resolve()): hashlib.sha256(
                    attestation.read_bytes()).hexdigest()},
            "jobs": jobs,
        })
    return {
        "schema": comparison.SCHEMA,
        "comparison_id": "fixture",
        "output_root": f"/Users/fixture-comparison-{tmp_path.name}",
        "worker_path": str(worker.resolve()),
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "projection_path": str(projection.resolve()),
        "projection_sha256": hashlib.sha256(projection.read_bytes()).hexdigest(),
        "coordinator_sha256": hashlib.sha256(Path(
            comparison.__file__).read_bytes()).hexdigest(),
        "pairs": pairs,
    }


def _accounting(complete=True, unknown=0, base=10):
    return {
        "accounting_complete": complete,
        "tokens": {
            name: {"known_sum": base, "unknown_attempts": unknown}
            for name in ("input_tokens", "uncached_input_tokens",
                         "cached_input_tokens", "cache_write_input_tokens",
                         "output_tokens", "reasoning_tokens")
        },
    }


def test_validates_two_balanced_opaque_pairs_without_writes(tmp_path):
    plan = _fixture_plan(tmp_path)
    _worker, _projection, pairs = comparison.validate_plan(plan)
    assert [pair["jobs"][0]["arm"] for pair in pairs] == ["control", "treatment"]
    assert not Path(plan["output_root"]).exists()


def test_unbalanced_order_and_mismatched_pair_inputs_are_rejected(tmp_path):
    plan = _fixture_plan(tmp_path)
    plan["pairs"][1]["jobs"].reverse()
    with pytest.raises(ValueError, match="balanced"):
        comparison.validate_plan(plan)
    plan = _fixture_plan(tmp_path)
    second = plan["pairs"][0]["jobs"][1]
    value = json.loads(Path(second["plan_path"]).read_text())
    value["input_seal"] = "f" * 64
    second["plan_sha256"] = _write(Path(second["plan_path"]), value)
    with pytest.raises(ValueError, match="identical sealed input"):
        comparison.validate_plan(plan)


def test_worker_plan_helper_and_input_attestation_are_all_pinned(tmp_path):
    plan = _fixture_plan(tmp_path)
    Path(plan["worker_path"]).write_text("changed\n")
    with pytest.raises(ValueError, match="helper source changed"):
        comparison.validate_plan(plan)
    plan = _fixture_plan(tmp_path)
    path = Path(next(iter(plan["pairs"][0]["input_attestations"])))
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="attestation mismatch"):
        comparison.validate_plan(plan)


def test_pair_equality_is_true_false_or_inconclusive_not_conflated():
    base = {"worker_complete": True, "accounting_complete": True,
            "prefetch_exercised": True, "quality_view": {"verdict": "verified"}}
    assert comparison.compare_pair(base, base)["exact_quality_equal"] is True
    changed = {**base, "quality_view": {"verdict": "refuted"}}
    assert comparison.compare_pair(base, changed)["exact_quality_equal"] is False
    incomplete = {**base, "accounting_complete": False}
    assert comparison.compare_pair(base, incomplete)["exact_quality_equal"] is None
    no_packet = {**base, "prefetch_exercised": False}
    assert comparison.compare_pair(base, no_packet)["exact_quality_equal"] is None


def test_incomplete_cost_retains_every_known_bucket_but_has_no_ratio():
    jobs = [
        {"arm": "control", "accounting": _accounting(base=20)},
        {"arm": "treatment", "accounting": _accounting(
            complete=False, unknown=1, base=7)},
    ]
    report = comparison._accounting_summary(jobs)
    assert report["complete"] is False
    assert report["control_over_treatment"] is None
    assert report["tokens"]["output_tokens"] == {
        "control": {"known_sum": 20, "unknown_attempts": 0},
        "treatment": {"known_sum": 7, "unknown_attempts": 1},
    }


def test_complete_cost_uses_output_once_and_computes_ratio():
    jobs = [
        {"arm": "control", "accounting": _accounting(base=100)},
        {"arm": "treatment", "accounting": _accounting(base=10)},
    ]
    report = comparison._accounting_summary(jobs)
    assert report["complete"] is True
    assert report["control_over_treatment"] == pytest.approx(10)
    # Reasoning is a diagnostic subset of output and is not billed twice.
    expected = (100 * .20 + 100 * .02 + 100 * 1.20) / 1_000_000
    assert report["price_equivalent_usd"]["control"] == pytest.approx(expected)


def test_rank_projection_requires_original_feature_schema():
    features = {
        "feature_schema_version": 3, "unscoreable": False,
        "verdict": "verified", "final_obligations": {"verified": 2},
        "final_obligations_total": 2, "final_contrary_signals": 1,
    }
    assert comparison._rank_key(features) == [4, -1]
    features["feature_schema_version"] = 2
    assert comparison._rank_key(features) is None
