"""No-model tests for the fail-closed full-online A/B coordinator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.terminal_bench import run_full_online_comparison as comparison
from verifier.engine.typed_judger_packet import TYPED_JUDGER_PACKET_PROTOCOL
from verifier.llm.typed_judger_backend import TYPED_JUDGER_BOUNDARY_VERSION


def test_transport_pins_match_production_protocols():
    assert comparison.TYPED_PACKET_PROTOCOL == TYPED_JUDGER_PACKET_PROTOCOL
    assert comparison.TYPED_BOUNDARY == TYPED_JUDGER_BOUNDARY_VERSION


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(root, label):
    root.mkdir()
    (root / "task.md").write_text(f"Inspect opaque candidate {label}.\n")
    (root / "live-world.json").write_text(
        json.dumps({"candidate": label}) + "\n")
    records = {}
    for name in ("task.md", "live-world.json"):
        raw = (root / name).read_bytes()
        records[name] = {"sha256": hashlib.sha256(raw).hexdigest(),
                         "bytes": len(raw)}
    encoded = json.dumps(
        [{"name": name, **records[name]}
         for name in ("task.md", "live-world.json")],
        sort_keys=True, separators=(",", ":")).encode()
    return {"seal": hashlib.sha256(encoded).hexdigest(), "files": records}


def _fixture_plan(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source"
    worker = source / "experiments" / "terminal_bench" \
        / "run_full_online_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text(
        "import hashlib, json\n"
        "from pathlib import Path\n"
        "SCHEMA = 'gv-full-online-worker-plan-v2'\n"
        "RESULT_SCHEMA = 'gv-full-online-worker-result-v2'\n"
        "RUNTIME = {'model':'gpt-5.6-luna','effort':'medium',"
        "'service_tier':'standard'}\n"
        "INPUT_FILES = ('task.md', 'live-world.json')\n"
        "def validate_plan(plan): return plan\n"
        "def input_snapshot(root):\n"
        " records = {}\n"
        " for name in INPUT_FILES:\n"
        "  raw = (Path(root) / name).read_bytes()\n"
        "  records[name] = {'sha256':hashlib.sha256(raw).hexdigest(),"
        "'bytes':len(raw)}\n"
        " payload = [{'name':name, **records[name]} for name in INPUT_FILES]\n"
        " raw = json.dumps(payload,sort_keys=True,separators=(',',':')).encode()\n"
        " return {'seal':hashlib.sha256(raw).hexdigest(),'files':records}\n"
        "def _certificate_semantics(value): return value['semantic']\n"
        "def _score_semantics(value): return value['semantic']\n"
        "def _content_digest(value):\n"
        " raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode()\n"
        " return hashlib.sha256(raw).hexdigest()\n",
        encoding="utf-8")
    (source / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    cli = runtime / "bin" / "codex"
    cli.write_text("fixture cli\n")
    cli.chmod(0o700)
    source_identity = {
        "source_commit": "a" * 40,
        "source_fingerprint": comparison._source_fingerprint(source),
    }
    runtime_identity = {
        "codex_cli_path": str(cli.resolve()),
        "codex_cli_realpath": str(cli.resolve()),
        "codex_cli_sha256": comparison._sha(cli),
        "codex_cli_version": "codex-cli fixture",
        "runtime_read_root_fingerprints": {
            str(runtime.resolve()): comparison._runtime_root_fingerprint(runtime),
        },
    }
    pairs = []
    cells = (("eligible", "eligible"), ("eligible", "ineligible"),
             ("ineligible", "ineligible"))
    for index in range(6):
        order = (("control", "treatment") if index % 2 == 0
                 else ("treatment", "control"))
        input_root = tmp_path / f"input-{index}"
        snapshot = _inputs(input_root, str(index))
        attestations = {
            str((input_root / name).resolve()): snapshot["files"][name]["sha256"]
            for name in ("task.md", "live-world.json")
        }
        jobs = []
        for arm in order:
            worker_plan = {
                "schema": "gv-full-online-worker-plan-v2",
                "source_root": str(source.resolve()),
                "source_commit": source_identity["source_commit"],
                "source_fingerprint": source_identity["source_fingerprint"],
                "input_source": str(input_root.resolve()),
                "input_seal": snapshot["seal"],
                "output_root": f"/Users/fixture-online-worker-{tmp_path.name}-{index}-{arm}",
                "arm": arm, "runtime_read_roots": [str(runtime.resolve())],
                "support_sha256": {"worker.py": comparison._sha(worker)},
            }
            path = tmp_path / f"plan-{index}-{arm}.json"
            jobs.append({"arm": arm, "plan_path": str(path.resolve()),
                         "plan_sha256": _write(path, worker_plan)})
        candidate = f"candidate-{index}"
        adapter, packet = cells[index % len(cells)]
        truth = {
            "schema": comparison.TRUTH_SCHEMA, "candidate_id": candidate,
            "input_seal": snapshot["seal"],
            "task_family": ("documents", "filesystem", "records")[index % 3],
            "polarity": "positive" if index < 3 else "negative",
            "typed_adapter_eligibility": adapter,
            "typed_packet_eligibility": packet, "browser_required": False,
            "expected_final_status": "verified" if index < 3 else "refuted",
            "acceptable_verdicts": (["verified", "supported"] if index < 3
                                    else ["refuted"]),
            "minimum_evidence_count": 1, "required_evidence": [],
        }
        truth["acceptable_semantic_evidence"] = [{
            "schema": comparison.SEMANTIC_SCHEMA,
            "input_tree_seal": snapshot["seal"],
            "verdict": truth["acceptable_verdicts"][0],
            "verdict_strength": 1, "truth_eligible": True,
            "obligations": [{"evidence": [{}]}],
        }]
        truth_path = tmp_path / f"truth-{index}.json"
        pairs.append({"candidate_id": candidate,
                      "truth_path": str(truth_path.resolve()),
                      "truth_sha256": _write(truth_path, truth), "jobs": jobs})
    return {
        "schema": comparison.SCHEMA, "comparison_id": "fixture",
        "output_root": f"/Users/fixture-online-comparison-{tmp_path.name}",
        "worker_path": str(worker.resolve()),
        "worker_sha256": comparison._sha(worker),
        "coordinator_sha256": comparison._sha(comparison.__file__),
        "source_identity": source_identity,
        "runtime_identity": runtime_identity, "pairs": pairs,
    }


def _replace_worker_plan(plan, pair_index, arm, change):
    job = next(item for item in plan["pairs"][pair_index]["jobs"]
               if item["arm"] == arm)
    path = Path(job["plan_path"])
    value = json.loads(path.read_text())
    change(value)
    job["plan_sha256"] = _write(path, value)


def test_valid_balanced_plan_is_read_only_and_pins_exact_inputs(tmp_path):
    plan = _fixture_plan(tmp_path / "source-mismatch")
    _worker, pairs = comparison.validate_plan(plan)
    assert [pair["jobs"][0]["arm"] for pair in pairs] == [
        "control", "treatment", "control", "treatment", "control",
        "treatment"]
    assert not Path(plan["output_root"]).exists()


def test_pair_may_differ_only_by_arm_and_output(tmp_path):
    plan = _fixture_plan(tmp_path / "attestation-mismatch")
    _replace_worker_plan(
        plan, 0, "treatment", lambda value: value.update(input_seal="f" * 64))
    with pytest.raises(ValueError, match="differ only"):
        comparison.validate_plan(plan)


def test_order_source_and_truth_binding_fail_closed(tmp_path):
    plan = _fixture_plan(tmp_path / "order-case")
    plan["pairs"][1]["jobs"].reverse()
    with pytest.raises(ValueError, match="counterbalanced"):
        comparison.validate_plan(plan)

    plan = _fixture_plan(tmp_path / "commit-case")
    plan["source_identity"]["source_commit"] = "b" * 40
    with pytest.raises(ValueError, match="source identity"):
        comparison.validate_plan(plan)

    plan = _fixture_plan(tmp_path / "truth-case")
    truth_path = Path(plan["pairs"][0]["truth_path"])
    truth = json.loads(truth_path.read_text())
    truth["input_seal"] = "f" * 64
    plan["pairs"][0]["truth_sha256"] = _write(truth_path, truth)
    with pytest.raises(ValueError, match="truth does not bind"):
        comparison.validate_plan(plan)


def test_source_and_cli_runtime_content_are_verified(tmp_path):
    plan = _fixture_plan(tmp_path / "source-case")
    source = Path(json.loads(Path(
        plan["pairs"][0]["jobs"][0]["plan_path"]).read_text())["source_root"])
    (source / "pyproject.toml").write_text("changed\n")
    with pytest.raises(ValueError, match="archive fingerprint"):
        comparison.validate_plan(plan)


def test_worker_is_bound_before_import_to_the_canonical_frozen_source(tmp_path):
    plan = _fixture_plan(tmp_path / "worker-binding")
    attacker = tmp_path / "attacker.py"
    attacker.write_text("raise RuntimeError('worker executed')\n")
    plan["worker_path"] = str(attacker.resolve())
    plan["worker_sha256"] = comparison._sha(attacker)
    with pytest.raises(ValueError, match="canonical frozen-source worker"):
        comparison.validate_plan(plan)

    plan = _fixture_plan(tmp_path / "runtime-case")
    cli = Path(plan["runtime_identity"]["codex_cli_realpath"])
    cli.write_text("changed cli\n")
    with pytest.raises(ValueError, match="executable identity"):
        comparison.validate_plan(plan)


def _quality(label="same"):
    certificate_semantic = {"final": label, "slice": [
        {"id": label, "status": "verified"}]}
    score = {"verdict": "verified", "label": label}
    document = {
        "schema": comparison.SEMANTIC_SCHEMA,
        "input_tree_seal": "a" * 64, "verdict": "verified",
        "verdict_strength": 1, "truth_eligible": True,
        "obligations": [{"evidence": [{"kind": "document", "value": label}]}],
    }
    return {
        "projection": {
            "schema": "gv-full-offline-result-projection-v1",
            "events_sha256": "1" * 64, "events_rows": 7,
            "final_claim_id": label,
        },
        "ordered_contract": [{"text": label}],
        "ordered_outcomes": [{"status": "verified"}],
        "semantic_evidence": {
            "document": document,
            "content_sha256": comparison._canonical_json_sha(document),
        },
        "certificate": certificate_semantic,
        "score": {"features": score, "rank_key": [4, 0]},
    }, {"certificate": {"semantic": certificate_semantic, "final": label,
                         "slice": certificate_semantic["slice"]},
        "features": {"semantic": score}}


def test_quality_is_rederived_and_pair_acceptance_requires_both_exercises():
    quality, chain = _quality()
    worker = SimpleNamespace(
        _certificate_semantics=lambda value: value["semantic"],
        _score_semantics=lambda value: value["semantic"])
    result = {
        "chain": {**chain, "quality_view": quality},
        "execution_audit": {"events_sha256": "1" * 64, "events_rows": 7},
    }
    worker._content_digest = comparison._canonical_json_sha
    result["truth_eligible"] = True
    semantic, projection, final_status = comparison._quality_view(worker, result)
    assert semantic == quality["semantic_evidence"]["document"]
    assert projection == quality["projection"]
    assert final_status == "verified"
    result["chain"]["quality_view"]["certificate"] = {"tampered": True}
    with pytest.raises(ValueError, match="identity"):
        comparison._quality_view(worker, result)

    truth = {"typed_adapter_eligibility": "eligible",
             "typed_packet_eligibility": "eligible"}
    truth_eval = {"absolute_truth_correct": True, "truth_distance": 0}
    complete = {
        "worker_complete": True, "accounting_complete": True,
        "transport_audit_complete": True, "semantic_evidence": semantic,
        "semantic_evidence_sha256": comparison._canonical_json_sha(semantic),
        "truth_evaluation": truth_eval, "truth_eligible": True,
        "typed_adapter_exercised": True,
        "typed_transport": {"packet_attached_attempts": 1,
                            "restricted_attempted": True,
                            "restricted_fallback_attempted": False,
                            "packet_unavailable_fallback": False},
    }
    assert comparison.compare_pair(complete, complete, truth)[
        "sealed_truth_no_regression"] is True
    no_adapter = {**complete, "typed_adapter_exercised": False}
    assert comparison.compare_pair(complete, no_adapter, truth)["complete"] is False
    no_packet = {**complete, "typed_transport": {
        "packet_attached_attempts": 0, "restricted_attempted": False}}
    assert comparison.compare_pair(complete, no_packet, truth)["complete"] is False

    other_quality, other_chain = _quality()
    other_quality["projection"]["events_sha256"] = "2" * 64
    other_result = {
        "chain": {**other_chain, "quality_view": other_quality},
        "execution_audit": {"events_sha256": "2" * 64, "events_rows": 7},
    }
    other_result["truth_eligible"] = True
    other_semantic, _, _ = comparison._quality_view(worker, other_result)
    assert comparison.compare_pair(
        complete, {**complete, "semantic_evidence": other_semantic,
                   "semantic_evidence_sha256":
                   comparison._canonical_json_sha(other_semantic)}, truth)[
            "sealed_truth_no_regression"] is True


def _usage(input_tokens=100, cached=20, cache_write=0, output=10, reasoning=2,
           **extra):
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "uncached_input_tokens": input_tokens - cached,
        "cache_write_input_tokens": cache_write,
        "output_tokens": output, "reasoning_tokens": reasoning, **extra,
    }


def _priced_job(tmp_path, arm, usages, *, accounting_complete=True):
    output = tmp_path / f"{arm}-{len(list(tmp_path.iterdir()))}"
    output.mkdir()
    rows = []
    totals = {name: {"known_sum": 0, "unknown_attempts": 0}
              for name in comparison._TOKENS}
    for index, usage in enumerate(usages, 1):
        common = {
            "schema": "gv-codex-attempt-observer-v1", **comparison.RUNTIME,
            "call_id": f"call-{arm}-{index}",
        }
        attempt = {
            **common, "attempt_id": f"attempt-{arm}-{index}",
            "attempt_index": 1, "command_sha256": str(index) * 64,
            "cwd": f"/fixture/{arm}",
            **{"command_" + key: value
               for key, value in comparison.RUNTIME.items()},
        }
        rows.extend([
            {**common, "record_type": "codex_logical_call_started"},
            {**attempt, "record_type": "codex_cli_attempt_started"},
            {**attempt, "record_type": "codex_cli_attempt_finished",
             "usage_complete": usage.get("usage_complete", True),
             "usage_status": usage.get("usage_status", "complete"),
             "terminal_status": usage.get("terminal_status", "returned"),
             "process_started": usage.get("process_started", True),
             **{name: usage[name] for name in comparison._TOKENS}},
            {**common, "record_type": "codex_logical_call_finished"},
        ])
        for name in comparison._TOKENS:
            totals[name]["known_sum"] += usage[name]
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode()
                   for row in rows)
    (output / "observer.jsonl").write_bytes(raw)
    accounting = {
        "accounting_complete": accounting_complete,
        "ledger_sha256": hashlib.sha256(raw).hexdigest(),
        "ledger_bytes": len(raw), "attempt_count": len(usages),
        "expected_profile": dict(comparison.RUNTIME),
        "observed_event_counts": {
            "logical_calls_started": len(usages),
            "logical_calls_finished": len(usages),
            "attempts_started": len(usages), "attempts_finished": len(usages),
        },
        "tokens": totals,
    }
    return {"arm": arm, "worker_output_root": str(output),
            "accounting": accounting,
            "accounting_complete": accounting_complete,
            "worker_complete": True}


def test_attempt_pricing_is_per_call_and_cache_write_replaces_uncached(tmp_path):
    at = _usage(input_tokens=272_000, cached=72_000, cache_write=1_000,
                output=10_000, reasoning=0)
    above = _usage(input_tokens=272_001, cached=72_001, cache_write=2_000,
                   output=10_000, reasoning=0)
    report = comparison._accounting_summary([
        _priced_job(tmp_path, "control", [at, above]),
        _priced_job(tmp_path, "treatment", [at]),
    ])
    base = (199_000 * .20 + 72_000 * .02 + 1_000 * .25
            + 10_000 * 1.20) / 1_000_000
    long = ((198_000 * .20 + 72_001 * .02 + 2_000 * .25) * 2
            + 10_000 * 1.20 * 1.5) / 1_000_000
    assert report["complete"] is True
    assert report["price_equivalent_usd"]["control"] == pytest.approx(base + long)
    assert report["long_context_attempts"] == {"control": 1, "treatment": 0}
    assert report["cache_write_rate_replaces_uncached_rate"] is True


def test_failed_accounting_retains_known_minimum_but_never_a_ratio(tmp_path):
    control = _priced_job(tmp_path, "control", [_usage()])
    treatment = _priced_job(
        tmp_path, "treatment", [_usage()], accounting_complete=False)
    report = comparison._accounting_summary([control, treatment])
    assert report["complete"] is False
    assert report["price_equivalent_usd"] is None
    assert report["control_over_treatment"] is None
    assert report["known_minimum_price_equivalent_usd"]["treatment"] > 0
    assert report["tokens"]["output_tokens"]["treatment"]["known_sum"] == 10


def test_missing_worker_result_still_recovers_observer_known_minimum(tmp_path):
    control = _priced_job(tmp_path, "control", [_usage(output=7)])
    treatment = _priced_job(tmp_path, "treatment", [_usage(output=11)])
    treatment["accounting"] = None
    treatment["accounting_complete"] = False
    report = comparison._accounting_summary([control, treatment])
    assert report["complete"] is False
    assert report["control_over_treatment"] is None
    assert report["tokens"]["output_tokens"]["treatment"]["known_sum"] == 11
    assert report["known_minimum_price_equivalent_usd"]["treatment"] > 0


def test_torn_observer_tail_keeps_complete_prefix_as_known_minimum(tmp_path):
    control = _priced_job(tmp_path, "control", [_usage(output=7)])
    treatment = _priced_job(tmp_path, "treatment", [_usage(output=11)])
    ledger = Path(treatment["worker_output_root"]) / "observer.jsonl"
    with ledger.open("ab") as handle:
        handle.write(b'{"record_type":')
    treatment["accounting"] = None
    treatment["accounting_complete"] = False
    report = comparison._accounting_summary([control, treatment])
    assert report["complete"] is False
    assert report["control_over_treatment"] is None
    assert "observer_ledger_torn_or_empty" in report["findings"]
    assert report["tokens"]["output_tokens"]["treatment"]["known_sum"] == 11


def test_accounting_aggregates_multiple_jobs_without_cross_job_mismatch(tmp_path):
    jobs = [_priced_job(tmp_path, arm, [_usage(output=amount)])
            for amount in (3, 5) for arm in ("control", "treatment")]
    report = comparison._accounting_summary(jobs, expected_jobs=4)
    assert report["complete"] is True
    assert report["tokens"]["output_tokens"]["control"]["known_sum"] == 8
    assert report["tokens"]["output_tokens"]["treatment"]["known_sum"] == 8


def test_missing_attempt_finish_is_unknown_not_zero(tmp_path):
    control = _priced_job(tmp_path, "control", [_usage()])
    treatment = _priced_job(tmp_path, "treatment", [_usage()])
    ledger = Path(treatment["worker_output_root"]) / "observer.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows = [row for row in rows
            if row["record_type"] != "codex_cli_attempt_finished"]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = comparison._accounting_summary([control, treatment])
    assert report["complete"] is False
    assert report["control_over_treatment"] is None
    assert report["tokens"]["output_tokens"]["treatment"][
        "unknown_attempts"] == 1


def test_typed_transport_audit_consumes_native_observer_bijection():
    base = {
        "native_attempts": 2, "observer_finished_attempts": 2,
        "typed_judger_transport": {
            "enabled": True, "protocol": comparison.TYPED_PACKET_PROTOCOL,
            "boundary": comparison.TYPED_BOUNDARY,
            "packet_eligible_bound_methods": 0,
            "packet_attached_attempts": 0, "restricted_attempted": False,
            "ordinary_judger_attempts": 1,
            "raw_prefetch_attached_attempts": 0,
            "restricted_fallback_attempted": False,
            "packet_unavailable_fallback": False,
        },
    }
    assert comparison._transport_audit(base, "treatment") is True
    assert comparison._transport_audit(
        {**base, "observer_finished_attempts": 1}, "treatment") is False
    unavailable = json.loads(json.dumps(base))
    unavailable["typed_judger_transport"]["packet_unavailable_fallback"] = True
    assert comparison._transport_audit(unavailable, "treatment") is False
    assert comparison._transport_audit(base, "control") is False


def test_partial_or_unverified_never_passes_truth_or_no_regression():
    truth = {
        "input_seal": "a" * 64, "acceptable_verdicts": ["verified"],
        "expected_final_status": "verified", "minimum_evidence_count": 1,
        "required_evidence": [], "typed_adapter_eligibility": "ineligible",
        "typed_packet_eligibility": "ineligible",
        "acceptable_semantic_evidence": [],
    }
    semantic = {
        "input_tree_seal": "a" * 64, "verdict": "unverified",
        "truth_eligible": False, "obligations": [{"evidence": [{}]}],
    }
    evaluation = comparison._truth_evaluation(truth, semantic, "unverified")
    assert evaluation["absolute_truth_correct"] is False
    job = {
        "worker_complete": True, "accounting_complete": True,
        "transport_audit_complete": True, "semantic_evidence": semantic,
        "semantic_evidence_sha256": comparison._canonical_json_sha(semantic),
        "truth_evaluation": evaluation, "truth_eligible": False,
        "typed_adapter_exercised": False,
        "typed_transport": {"packet_attached_attempts": 0,
                            "restricted_attempted": False},
    }
    assert comparison.compare_pair(job, job, truth)[
        "sealed_truth_no_regression"] is False


@pytest.mark.parametrize("replacement", [True, 1.0])
def test_absolute_truth_requires_exact_json_types(replacement):
    expected = {
        "schema": comparison.SEMANTIC_SCHEMA,
        "input_tree_seal": "a" * 64, "verdict": "verified",
        "verdict_strength": 1, "truth_eligible": True,
        "obligations": [{"evidence": [{"value": "expected"}]}],
    }
    truth = {
        "input_seal": "a" * 64, "acceptable_verdicts": ["verified"],
        "expected_final_status": "verified", "minimum_evidence_count": 1,
        "required_evidence": [{"value": "expected"}],
        "acceptable_semantic_evidence": [expected],
    }
    assert comparison._truth_evaluation(
        truth, expected, "verified")["absolute_truth_correct"] is True
    changed = json.loads(json.dumps(expected))
    changed["verdict_strength"] = replacement
    report = comparison._truth_evaluation(truth, changed, "verified")
    assert report["absolute_truth_correct"] is False
    assert "complete_semantic_evidence" in report["findings"]


def test_unknown_verdict_and_restricted_fallback_fail_closed():
    truth = {
        "input_seal": "a" * 64, "acceptable_verdicts": ["verified"],
        "expected_final_status": "verified", "minimum_evidence_count": 1,
        "required_evidence": [], "typed_adapter_eligibility": "eligible",
        "typed_packet_eligibility": "eligible",
        "acceptable_semantic_evidence": [],
    }
    semantic = {
        "input_tree_seal": "a" * 64, "verdict": "future-verdict",
        "truth_eligible": False, "obligations": [{"evidence": [{}]}],
    }
    evaluation = comparison._truth_evaluation(truth, semantic, "verified")
    assert "nonterminal_or_unknown_verdict" in evaluation["findings"]
    job = {
        "worker_complete": True, "accounting_complete": True,
        "transport_audit_complete": True, "semantic_evidence": semantic,
        "semantic_evidence_sha256": comparison._canonical_json_sha(semantic),
        "truth_evaluation": evaluation, "truth_eligible": False,
        "typed_adapter_exercised": True,
        "typed_transport": {
            "packet_attached_attempts": 1, "restricted_attempted": True,
            "restricted_fallback_attempted": True,
            "packet_unavailable_fallback": False,
        },
    }
    report = comparison.compare_pair(job, job, truth)
    assert report["complete"] is False
    assert report["sealed_truth_no_regression"] is False


def test_default_cli_only_validates_and_paid_gate_is_exact(tmp_path, capsys,
                                                           monkeypatch):
    plan = _fixture_plan(tmp_path)
    path = tmp_path / "comparison-plan.json"
    digest = _write(path, plan)
    assert comparison.main([
        "--plan", str(path), "--plan-sha256", digest]) == 0
    assert json.loads(capsys.readouterr().out)["model_calls"] == 0
    assert not Path(plan["output_root"]).exists()

    monkeypatch.setattr(comparison, "run_comparison",
                        lambda _plan: pytest.fail("paid run was reached"))
    with pytest.raises(SystemExit):
        comparison.main([
            "--plan", str(path), "--plan-sha256", digest, "--execute",
            "--confirm-paid", "gpt-5.6-luna"])


def test_ten_x_acceptance_requires_truth_and_transport_but_keeps_latency(
        tmp_path, monkeypatch):
    full_quality, _chain = _quality()
    semantic = full_quality["semantic_evidence"]["document"]
    pairs = [{
        "candidate_id": f"candidate-{index}",
        "truth": {"typed_adapter_eligibility": "eligible",
                  "typed_packet_eligibility": "eligible"},
        "stratum": {"id": "fixture-stratum"},
        "jobs": [{"arm": arm, "worker_plan": {"output_root": f"/{arm}-{index}"}}
                 for arm in order],
    } for index, order in enumerate((
        ("control", "treatment"), ("treatment", "control")))]
    monkeypatch.setattr(comparison, "preflight", lambda plan: {"model_calls": 0})
    monkeypatch.setattr(
        comparison, "validate_plan", lambda plan: (SimpleNamespace(), pairs))
    exercised = {"value": True}

    def fake_job(_plan, _worker, _output, pair, job, _ordinal):
        arm = job["arm"]
        return {
            "candidate_id": pair["candidate_id"], "arm": arm,
            "worker_complete": True, "accounting_complete": True,
            "accounting": {}, "semantic_evidence": semantic,
            "semantic_evidence_sha256": comparison._canonical_json_sha(semantic),
            "truth_evaluation": {"absolute_truth_correct": True,
                                 "truth_distance": 0},
            "truth_eligible": True, "transport_audit_complete": True,
            "typed_adapter_exercised": (exercised["value"]
                                         if arm == "treatment" else False),
            "typed_transport": {
                "packet_attached_attempts": (1 if arm == "treatment" else 0),
                "restricted_attempted": arm == "treatment",
                "restricted_fallback_attempted": False,
                "packet_unavailable_fallback": False},
            "spawn_to_exit_wall_seconds": 1.0 if arm == "treatment" else 10.0,
        }

    monkeypatch.setattr(comparison, "_run_job", fake_job)
    monkeypatch.setattr(comparison, "_accounting_summary", lambda jobs, **kwargs: {
        "complete": True, "control_over_treatment": 10.0})
    plan = {"output_root": str(tmp_path / "accepted")}
    result = comparison.run_comparison(plan)
    assert result["sampled_exact_semantic_quality_no_loss"] is True
    assert result["sampled_all_truth_correct"] is True
    assert result["latency"]["control_over_treatment"] == 10
    assert result["sampled_candidate_10x_acceptance"] is True

    exercised["value"] = False
    plan = {"output_root": str(tmp_path / "rejected")}
    result = comparison.run_comparison(plan)
    assert result["sampled_exact_semantic_quality_no_loss"] is False
    assert result["latency"]["control_over_treatment"] == 10
    assert result["sampled_candidate_10x_acceptance"] is False
