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
    value = {
        "accounting_complete": complete,
        "tokens": {
            name: {"known_sum": base, "unknown_attempts": unknown}
            for name in ("input_tokens", "uncached_input_tokens",
                         "cached_input_tokens", "cache_write_input_tokens",
                         "output_tokens", "reasoning_tokens")
        },
    }
    value["tokens"]["cache_write_input_tokens"]["known_sum"] = 0
    return value


def _priced_job(tmp_path, arm, attempts, *, label=None):
    output = tmp_path / ((label or arm) + "-worker")
    output.mkdir()
    rows = []
    totals = {name: 0 for name in (
        "input_tokens", "uncached_input_tokens", "cached_input_tokens",
        "cache_write_input_tokens", "output_tokens", "reasoning_tokens")}
    for index, usage in enumerate(attempts, 1):
        call_id = f"call-{arm}-{index}"
        attempt_id = f"attempt-{arm}-{index}"
        common = {
            "schema": "gv-codex-attempt-observer-v1",
            **comparison.RUNTIME,
            "call_id": call_id,
        }
        command = {
            **common, "attempt_id": attempt_id, "attempt_index": 1,
            "command_sha256": str(index) * 64,
            "cwd": f"/fixture/{arm}",
            **{"command_" + key: value
               for key, value in comparison.RUNTIME.items()},
        }
        rows.extend([
            {**common, "record_type": "codex_logical_call_started"},
            {**command, "record_type": "codex_cli_attempt_started"},
            {**command, "record_type": "codex_cli_attempt_finished",
             "usage_complete": usage.get("usage_complete", True),
             "usage_status": usage.get("usage_status", "complete"),
             "terminal_status": usage.get("terminal_status", "returned"),
             "process_started": usage.get("process_started", True),
             **{name: usage[name] for name in totals}},
            {**common, "record_type": "codex_logical_call_finished"},
        ])
        for name in totals:
            totals[name] += usage[name]
    ledger = output / "observer.jsonl"
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode()
                   for row in rows)
    ledger.write_bytes(raw)
    accounting = {
        "accounting_complete": True,
        "ledger_sha256": hashlib.sha256(raw).hexdigest(),
        "ledger_bytes": len(raw),
        "attempt_count": len(attempts),
        "expected_profile": dict(comparison.RUNTIME),
        "observed_event_counts": {
            "logical_calls_started": len(attempts),
            "logical_calls_finished": len(attempts),
            "attempts_started": len(attempts),
            "attempts_finished": len(attempts),
        },
        "tokens": {name: {"known_sum": value, "unknown_attempts": 0}
                   for name, value in totals.items()},
    }
    return {"arm": arm, "worker_output_root": str(output),
            "accounting": accounting}


def _usage(input_tokens=100, cached=20, cache_write=0, output=10, reasoning=2,
           **extra):
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "uncached_input_tokens": input_tokens - cached,
        "cache_write_input_tokens": cache_write,
        "output_tokens": output,
        "reasoning_tokens": reasoning,
        **extra,
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
    typed_only = {**no_packet, "typed_judger_exercised": True}
    assert comparison.compare_pair(base, typed_only)["exact_quality_equal"] is True


def test_incomplete_cost_retains_every_known_bucket_but_has_no_ratio():
    jobs = [
        {"arm": "control", "accounting": _accounting(base=20)},
        {"arm": "treatment", "accounting": _accounting(
            complete=False, unknown=1, base=7)},
    ]
    report = comparison._accounting_summary(jobs)
    assert report["complete"] is False
    assert report["control_over_treatment"] is None
    assert "job_accounting_incomplete" in report["findings"]
    assert report["tokens"]["output_tokens"] == {
        "control": {"known_sum": 20, "unknown_attempts": 0},
        "treatment": {"known_sum": 7, "unknown_attempts": 1},
    }


def test_complete_cost_uses_output_once_and_computes_ratio(tmp_path):
    jobs = [
        _priced_job(tmp_path, "control", [_usage(
            input_tokens=200, cached=100, output=100, reasoning=100)]),
        _priced_job(tmp_path, "treatment", [_usage(
            input_tokens=20, cached=10, output=10, reasoning=10)]),
    ]
    report = comparison._accounting_summary(jobs)
    assert report["complete"] is True
    assert report["control_over_treatment"] == pytest.approx(10)
    # Reasoning is a diagnostic subset of output and is not billed twice.
    expected = (100 * .20 + 100 * .02 + 100 * 1.20) / 1_000_000
    assert report["price_equivalent_usd"]["control"] == pytest.approx(expected)
    assert report["priced_observer_attempts"] == {"control": 1, "treatment": 1}
    assert report["price_equivalent_profile"] == comparison.RUNTIME
    assert report["invoice_claim"] is False


def test_missing_planned_job_makes_cost_inconclusive():
    control = _accounting(base=100)
    missing = comparison._accounting_summary([
        {"arm": "control", "accounting": control},
    ], expected_jobs=2)
    assert missing["complete"] is False
    assert {"planned_job_count_incomplete", "arm_count_unbalanced"} <= set(
        missing["findings"])


def test_per_attempt_long_context_threshold_and_cache_write(tmp_path):
    at_threshold = _usage(
        input_tokens=272_000, cached=72_000, cache_write=1_000,
        output=10_000, reasoning=0)
    above = _usage(
        input_tokens=272_001, cached=72_001, cache_write=2_000,
        output=10_000, reasoning=0)
    jobs = [
        _priced_job(tmp_path, "control", [at_threshold, above]),
        _priced_job(tmp_path, "treatment", [at_threshold]),
    ]
    report = comparison._accounting_summary(jobs)
    base = (199_000 * .20 + 72_000 * .02 + 1_000 * .25
            + 10_000 * 1.20) / 1_000_000
    long = ((198_000 * .20 + 72_001 * .02 + 2_000 * .25) * 2
            + 10_000 * 1.20 * 1.5) / 1_000_000
    assert report["complete"] is True
    assert report["price_equivalent_usd"]["control"] == pytest.approx(base + long)
    assert report["price_equivalent_usd"]["treatment"] == pytest.approx(base)
    assert report["long_context_attempts"] == {"control": 1, "treatment": 0}
    assert report["long_context_pricing_inference"]["per"] == "observer_attempt"


def test_cache_write_cannot_exceed_uncached_input(tmp_path):
    control = _priced_job(tmp_path, "control", [
        _usage(input_tokens=100, cached=90, cache_write=11)])
    treatment = _priced_job(tmp_path, "treatment", [_usage()])
    report = comparison._accounting_summary([control, treatment])
    assert report["complete"] is False
    assert report["price_equivalent_usd"] is None
    assert "observer_attempt_usage_invalid" in report["findings"]


@pytest.mark.parametrize("tamper", ["hash", "bytes", "attempts", "profile"])
def test_observer_accounting_identity_and_profile_must_match(tmp_path, tamper):
    control = _priced_job(tmp_path, "control", [_usage()])
    treatment = _priced_job(
        tmp_path, "treatment", [_usage()], label="missing-treatment")
    if tamper == "hash":
        control["accounting"]["ledger_sha256"] = "f" * 64
    elif tamper == "bytes":
        control["accounting"]["ledger_bytes"] += 1
    elif tamper == "attempts":
        control["accounting"]["attempt_count"] += 1
    else:
        control["accounting"]["expected_profile"]["effort"] = "high"
    report = comparison._accounting_summary([control, treatment])
    assert report["complete"] is False
    assert report["price_equivalent_usd"] is None
    assert report["control_over_treatment"] is None


def test_missing_tampered_and_incomplete_usage_are_never_zero_cost(tmp_path):
    missing = {"arm": "control", "worker_output_root": str(tmp_path / "missing"),
               "accounting": _accounting()}
    treatment = _priced_job(tmp_path, "treatment", [_usage()])
    report = comparison._accounting_summary([missing, treatment])
    assert report["complete"] is False
    assert report["price_equivalent_usd"] is None
    assert "observer_ledger_unavailable" in report["findings"]

    control = _priced_job(
        tmp_path, "control", [_usage()], label="tampered-control")
    ledger = Path(control["worker_output_root"]) / "observer.jsonl"
    ledger.write_bytes(ledger.read_bytes() + b"{}\n")
    report = comparison._accounting_summary([control, treatment])
    assert report["complete"] is False
    assert "observer_ledger_identity_mismatch" in report["findings"]

    failed = _priced_job(tmp_path, "control", [_usage(
        usage_complete=False, usage_status="partial")], label="failed-control")
    report = comparison._accounting_summary([failed, treatment])
    assert report["complete"] is False
    assert report["price_equivalent_usd"] is None
    assert "observer_attempt_usage_incomplete" in report["findings"]

    failed_terminal = _priced_job(
        tmp_path, "control", [_usage(terminal_status="timeout")],
        label="failed-terminal-control")
    report = comparison._accounting_summary([failed_terminal, treatment])
    assert report["complete"] is False
    assert report["price_equivalent_usd"] is None
    assert "observer_attempt_usage_incomplete" in report["findings"]


def test_rank_projection_requires_original_feature_schema():
    features = {
        "feature_schema_version": 3, "unscoreable": False,
        "verdict": "verified", "final_obligations": {"verified": 2},
        "final_obligations_total": 2, "final_contrary_signals": 1,
    }
    assert comparison._rank_key(features) == [4, -1]
    features["feature_schema_version"] = 2
    assert comparison._rank_key(features) is None
