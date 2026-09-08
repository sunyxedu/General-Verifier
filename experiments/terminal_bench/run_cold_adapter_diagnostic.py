"""Single-candidate cold adapter diagnostic, never production scoring.

Three fresh processes: real extraction/Compiler/Critic setup, then independent
control and treatment Oracle/executor/Judger arms. No warm reuse, outer retry,
candidate substitution, or full-E2E performance acceptance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from experiments.terminal_bench.prepare_shared_contracts import (
    SOURCE_ROOT, codex_service, prepare_shared_contracts,
)
from experiments.terminal_bench.replay_prepared_contracts import replay_prepared_contracts, _audit_model_work, RESULT_SCHEMA
from experiments.terminal_bench.analyze_typed_adapter_eval import _outcome_decisions, _audit_raw_contract
from experiments.terminal_bench.run_typed_adapter_eval import (
    _source_fingerprint, _validate_execute_output_root,
)
from experiments.terminal_bench.shared_setup import COST_FIELDS, charge_pair
from experiments.terminal_bench.analyze_model_spans import MAX_LEDGER_BYTES
from experiments.terminal_bench.validate_prepared_contract_entry import _private_read, MAX_RECEIPT_BYTES
from verifier import offline
from verifier.core.model import content_hash
from verifier.execution.config import offline_runtime_read_roots
from verifier.engine.contract_bundle import _strict_json, MAX_CONTRACT_BUNDLE_BYTES


SCHEMA = "gv-single-cold-adapter-diagnostic/v1"
STAGES = ("setup", "control", "treatment")


def _environment(request, stage):
    # Pin public runtime access; never inherit a shared cache/toolchain or
    # browser instrument from a previous experiment.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GV_") and key not in ("PYTHONPATH", "PYTHONSTARTUP")}
    env.update({
        "PATH": request["runtime_path"],
        "GV_TYPED_ADAPTERS": "0" if stage == "control" else "1",
        "GV_CONTRACT_CACHE": "1",
        "GV_MAX_CONCURRENT_SEATS": str(request["max_concurrent_seats"]),
        "GV_OFFLINE_RUNTIME_READ_ROOTS": json.dumps(request["runtime_read_roots"]),
        "TZ": request.get("timezone", "Europe/London"),
    })
    return env


def _config():
    return offline.config_fields(model="gpt-5.6-luna", seats="codex",
                                 effort="medium", service_tier="standard",
                                 max_rounds=3, critic_rounds=4)


def _json(path):
    value = _strict_json(_private_read(Path(path), MAX_RECEIPT_BYTES))
    if type(value) is not dict:
        raise ValueError("diagnostic record must be an object")
    return value


def _check_setup_bills(root, state, accounting):
    """Bind the trusted terminal metrics to every still-intact raw bill."""
    expected = [("extract", root / "extract-usage.jsonl")]
    expected += [(row["claim_id"], root / "claims" / row["claim_id"] / "prepare-usage.jsonl")
                 for row in state["claims"]]
    records = accounting.get("source_records", [])
    if len(records) != len(expected) or any(row.get("status") != "finished" for row in state["claims"]):
        raise ValueError("setup bill coverage differs from completed plan")
    for label, suffix in (("usage", ""), ("spans", ".spans.jsonl")):
        digest, total = hashlib.sha256(), 0
        for record, (cid, usage_path) in zip(records, expected):
            item = record[label]
            path = Path(str(usage_path) + suffix)
            if (record.get("component_id") != cid or item.get("path") != str(path)
                    or item.get("included") is not True or item.get("stable") is not True):
                raise ValueError("setup bill provenance changed")
            raw = _private_read(path, MAX_LEDGER_BYTES - total)
            if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("setup component bill changed")
            digest.update(raw)
            total += len(raw)
        aggregate = _private_read(root / "accounting" / ("setup-usage.jsonl" + suffix), MAX_LEDGER_BYTES)
        if len(aggregate) != total or hashlib.sha256(aggregate).digest() != digest.digest():
            raise ValueError("setup aggregate bill changed")


def _recorded_config(request, stage):
    record = _json(Path(request["output"]) / (stage + "-config.json"))
    if record.get("stage") != stage or record.get("request_sha256") != content_hash(request):
        raise ValueError("worker config record differs from request")
    expected = {"model": "gpt-5.6-luna", "seats": "codex", "effort": "medium",
                "service_tier": "standard", "max_rounds": 3, "critic_rounds": 4,
                "typed_adapters": stage != "control", "contract_cache": True,
                "offline_runtime_read_roots": request["runtime_read_roots"], "browser_observer": None}
    expected["config"] = content_hash({"pipeline": offline.pipeline_fingerprint(), **expected})[:16]
    if content_hash(record.get("config")) != content_hash(expected):
        raise ValueError("worker config differs from fixed experimental profile")
    return expected


def _admit_setup(request, *, runtime_config=None):
    """Read the trusted worker's three terminal records, never just its pin."""
    root = Path(request["output"]) / "setup"
    names = ("setup-state.json", "setup-result.json", "bundle-pin.json", "contract-plan.json", "contracts.json")
    before = {name: _private_read(root / name, MAX_CONTRACT_BUNDLE_BYTES) for name in names}
    state, result, pin = (_json(root / name) for name in (
        "setup-state.json", "setup-result.json", "bundle-pin.json"))
    if (state.get("status") != "complete" or result.get("complete") is not True
            or state.get("findings") != [] or result.get("findings") != []):
        raise ValueError("setup has no complete terminal admission")
    for record in (state, result, pin):
        if (record.get("test_only") is not False
                or record.get("setup_id") != request["setup_id"]
                or record.get("source_fingerprint") != request["source_fingerprint"]):
            raise ValueError("setup provenance differs from paid request")
    if (state.get("source") != request["source"]
            or state.get("source_seal") != request["source_seal"]
            or state.get("source_root") != str(SOURCE_ROOT)
            or state.get("source_fingerprint_after") != request["source_fingerprint"]
            or result.get("source_fingerprint_after") != request["source_fingerprint"]
            or result.get("extracted_source") != str(root / "extracted")
            or result.get("bundle_path") != str(root / "contracts.json")
            or result.get("identity") != pin.get("identity")):
        raise ValueError("setup source or identity differs from request")
    # Config is derived under each arm's explicit environment. Only adapter
    # dispatch may differ; the compilation identity deliberately excludes it.
    current = _config() if runtime_config is None else runtime_config
    setup_config = state.get("config", {})
    if ({k: v for k, v in setup_config.items() if k not in ("config", "typed_adapters")}
            != {k: v for k, v in current.items() if k not in ("config", "typed_adapters")}
            or setup_config.get("typed_adapters") is not True):
        raise ValueError("setup runtime differs from replay runtime")
    if not result.get("accounting", {}).get("complete") or not result.get("metrics", {}).get("cost_complete"):
        raise ValueError("setup accounting is incomplete")
    _check_setup_bills(root, state, result["accounting"])
    if (pin.get("accounting_usage_path") != str(root / "accounting" / "setup-usage.jsonl")
            or pin.get("accounting_spans_path") != str(root / "accounting" / "setup-usage.jsonl.spans.jsonl")):
        raise ValueError("setup bill is not in the expected private location")
    plan_sha = content_hash(_json(root / "contract-plan.json"))
    if any(record.get("contract_plan_sha256") != plan_sha for record in (state, result, pin)):
        raise ValueError("setup plan changed")
    bundle = _private_read(root / "contracts.json", MAX_CONTRACT_BUNDLE_BYTES)
    sha = hashlib.sha256(bundle).hexdigest()
    if any(record.get("bundle_sha256") != sha for record in (state, result, pin)):
        raise ValueError("setup bundle changed")
    if offline.verify_immutable_input_anchor(str(root / "extracted")) != request["source_seal"]:
        raise ValueError("extracted source changed")
    if any(_private_read(root / name, MAX_CONTRACT_BUNDLE_BYTES) != raw for name, raw in before.items()):
        raise ValueError("setup changed during admission")
    return result, bundle


def _check_arm_bills(request, stage, result):
    root = Path(request["output"]) / stage
    cfg = _recorded_config(request, stage)
    prepared, _ = _admit_setup(request, runtime_config=_recorded_config(request, "setup"))
    context = {"replay_id": request["run_id"] + "-" + stage,
               "setup_id": request["setup_id"], "arm": stage}
    if (any(result.get(key) != value for key, value in context.items())
            or result.get("schema") != RESULT_SCHEMA or result.get("status") != "terminal"
            or result.get("usage_path") != "replay-usage.jsonl"
            or result.get("spans_path") != "replay-usage.jsonl.spans.jsonl"
            or result.get("bundle_sha256") != prepared["bundle_sha256"]
            or result.get("contract_identity") != prepared["identity"]
            or result.get("wrapper_config_sha256") != content_hash(cfg)):
        raise ValueError("replay provenance differs from request")
    paths = {"usage": root / "replay-usage.jsonl", "spans": root / "replay-usage.jsonl.spans.jsonl"}
    before = {label: _private_read(path, MAX_LEDGER_BYTES) for label, path in paths.items()}
    if any(hashlib.sha256(raw).hexdigest() != result.get(label + "_sha256") for label, raw in before.items()):
        raise ValueError("replay bill differs from terminal digest")
    audit = _audit_model_work(paths["usage"], paths["spans"], wall_seconds=result["arm_wall_seconds"],
                             context=context, state=root / "state")
    if (audit["complete"] is not True or any(audit["metrics"][key] != result["cost_report"]["metrics"][key]
                                           for key in COST_FIELDS)):
        raise ValueError("replay bill differs from terminal charge")
    if any(_private_read(paths[label], MAX_LEDGER_BYTES) != raw for label, raw in before.items()):
        raise ValueError("replay bills changed during final admission")


def _worker(request_path, expected_sha, stage):
    request = _json(request_path)
    if content_hash(request) != expected_sha or request.get("schema") != SCHEMA:
        raise ValueError("worker request differs from parent pin")
    if _source_fingerprint(SOURCE_ROOT) != request["source_fingerprint"]:
        raise ValueError("running source differs from parent pin")
    expected_env = _environment(request, stage)
    for key in ("PATH", "GV_TYPED_ADAPTERS", "GV_CONTRACT_CACHE",
                "GV_MAX_CONCURRENT_SEATS", "GV_OFFLINE_RUNTIME_READ_ROOTS", "TZ"):
        if os.environ.get(key) != expected_env[key]:
            raise ValueError("worker environment differs from request: " + key)
    if any(key.startswith("GV_") and key not in expected_env for key in os.environ):
        raise ValueError("worker inherited unpinned verifier environment")
    if offline.verify_immutable_input_anchor(request["source"]) != request["source_seal"]:
        raise ValueError("original input changed")
    output = Path(request["output"])
    _validate_execute_output_root(output)
    config = _config()
    offline._write_json_atomic(str(output / (stage + "-config.json")), {
        "request_sha256": expected_sha, "stage": stage, "config": config})
    if stage == "setup":
        result = prepare_shared_contracts(request["source"], output / stage,
            setup_id=request["setup_id"], config=config,
            expected_source_fingerprint=request["source_fingerprint"])
        if result["complete"]:
            _admit_setup(request)
    else:
        prepared, bundle = _admit_setup(request)
        result = replay_prepared_contracts(
            prepared["extracted_source"], output / stage,
            replay_id=request["run_id"] + "-" + stage,
            setup_id=request["setup_id"], arm=stage,
            service_factory=lambda state: codex_service(state, config), config=config,
            extracted_rounds=prepared["extracted_rounds"], bundle_bytes=bundle,
            expected_bundle_sha256=prepared["bundle_sha256"],
            expected_identity=prepared["identity"])
        _admit_setup(request)  # Final source/bundle guard after evidence work.
    if _source_fingerprint(SOURCE_ROOT) != request["source_fingerprint"]:
        raise ValueError("running source changed")
    return 0 if result["complete"] else 2


def _unknown_metrics():
    return {"complete": False, "cost_complete": False, "wall_complete": False,
            **{key: None for key in COST_FIELDS}, "wall_seconds": None}


def _compare_features(output, records):
    if any(records[stage]["status"] != "complete" for stage in STAGES):
        return {"available": False}
    try:
        results = {arm: _json(output / arm / "replay-result.json") for arm in ("control", "treatment")}
        if any(content_hash(row) != records[arm].get("worker_result_sha256") for arm, row in results.items()):
            raise ValueError("replay result changed after process admission")
        features = {arm: row["features"] for arm, row in results.items()}
        if any(type(row) is not dict or _audit_raw_contract({"features": row}) for row in features.values()):
            raise ValueError("replay feature contract is malformed")
        left, right = features["control"], features["treatment"]
        semantic = {arm: {**{k: v for k, v in row.items()
                                if k not in ("certificate_id", "final_obligation_outcomes_ordered")},
                          "final_obligation_outcomes_ordered": _outcome_decisions({"features": row})}
                    for arm, row in features.items()}
        return {"available": True,
                "verdicts": {arm: row["verdict"] for arm, row in features.items()},
                "rank_keys": {arm: list(offline.rank_key(row)) for arm, row in features.items()},
                "same_verdict": left["verdict"] == right["verdict"],
                "same_verdict_strength": left.get("verdict_strength") == right.get("verdict_strength"),
                "same_semantic_features": semantic["control"] == semantic["treatment"],
                "same_rank_key": offline.rank_key(left) == offline.rank_key(right),
                "same_final_contract": (left.get("contract_audit_status") == "available"
                    and right.get("contract_audit_status") == "available"
                    and left["final_obligation_contract"] == right["final_obligation_contract"]),
                "same_ordered_final_outcomes": (_outcome_decisions({"features": left})
                                               == _outcome_decisions({"features": right})),
                "zero_loss_proven": False}
    except (OSError, ValueError, KeyError, TypeError):
        return {"available": False}


def _run_stage(request, stage):
    root = Path(request["output"])
    command = [sys.executable, "-m", "experiments.terminal_bench.run_cold_adapter_diagnostic",
               "--worker", stage, "--request", str(root / "request.json"),
               "--request-sha256", content_hash(request), "--confirm-paid", "GPT-5.6-Luna"]
    started = time.monotonic_ns()
    record = {"stage": stage, "status": "running", "command": command,
              "started_at_monotonic_ns": started, "started_at_unix_ns": time.time_ns()}
    offline._write_json_atomic(str(root / (stage + "-process.json")), record)
    proc, interrupt, returncode = None, None, None
    findings = []
    try:
        if (_source_fingerprint(SOURCE_ROOT) != request["source_fingerprint"]
                or offline.verify_immutable_input_anchor(request["source"]) != request["source_seal"]):
            raise ValueError("source changed before process launch")
        with (root / (stage + ".stdout.log")).open("x") as stdout, \
                (root / (stage + ".stderr.log")).open("x") as stderr:
            proc = subprocess.Popen(command, cwd=SOURCE_ROOT, env=_environment(request, stage),
                                    stdout=stdout, stderr=stderr, start_new_session=True)
            returncode = proc.wait()
    except BaseException as error:
        findings.append("process_failed:" + type(error).__name__)
        if proc is not None:
            # Reap this worker group. Model CLI attempts create their own
            # sessions: abrupt cancellation cannot prove their cleanup here.
            # Record that limitation explicitly; never retry this stage.
            record["cleanup_complete"] = False
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                returncode = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                returncode = proc.wait()
        if not isinstance(error, Exception):
            interrupt = error
    finished = time.monotonic_ns()
    filename = "setup-result.json" if stage == "setup" else "replay-result.json"
    try:
        result = _json(root / stage / filename)
        metrics = dict(result["metrics"] if stage == "setup" else result["cost_report"]["metrics"])
        if stage != "setup" and result.get("complete") is True and (
                result.get("verification_complete") is not True
                or result.get("accounting_complete") is not True
                or result.get("diagnostic_status") != "complete"):
            raise ValueError("complete replay has inconsistent terminal flags")
        if stage == "setup" and result.get("complete") is True:
            _admit_setup(request, runtime_config=_recorded_config(request, "setup"))
        elif stage != "setup" and result.get("accounting_complete") is True:
            _check_arm_bills(request, stage, result)
        if result.get("complete") is not True:
            findings.append("worker_result_inconclusive")
        if content_hash(_json(root / stage / filename)) != content_hash(result):
            raise ValueError("worker result changed during process admission")
    except (OSError, ValueError, KeyError, TypeError):
        result, metrics = None, _unknown_metrics()
        findings.append("worker_result_unavailable")
    if returncode != 0:
        findings.append("worker_exit_nonzero")
    try:
        if _source_fingerprint(SOURCE_ROOT) != request["source_fingerprint"]:
            findings.append("experiment_source_changed")
        if offline.verify_immutable_input_anchor(request["source"]) != request["source_seal"]:
            findings.append("original_input_changed")
    except (OSError, ValueError):
        findings.append("source_revalidation_failed")
    metrics.update(wall_seconds=(finished - started) / 1e9, wall_complete=True)
    metrics["complete"] = metrics.get("cost_complete") is True
    record.update(status="complete" if not findings else "inconclusive",
                  exit_code=returncode, finished_at_monotonic_ns=finished,
                  wall_seconds=metrics["wall_seconds"], metrics=metrics, findings=findings,
                  worker_result_sha256=content_hash(result) if result is not None else None,
                  worker_result_path=str(root / stage / filename))
    offline._write_json_atomic(str(root / (stage + "-process.json")), record)
    if interrupt is not None:
        raise interrupt
    return record


def run(request):
    """Trusted local coordinator; no factory injection or paid worker retry."""
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=False)
    offline._write_json_atomic(str(output / "request.json"), request)
    started = time.monotonic_ns()
    records = {stage: {"stage": stage, "status": "planned", "metrics": _unknown_metrics()}
               for stage in STAGES}
    offline._write_json_atomic(str(output / "diagnostic-state.json"), {
        "schema": SCHEMA, "status": "running", "stages": records})
    try:
        for stage in STAGES:
            if stage != "setup" and records["setup"]["status"] != "complete":
                # No process was launched, hence no model work. This is not an
                # unknown bill or a successful arm and cannot support comparison.
                records[stage] = {"stage": stage, "status": "not_started", "metrics": {
                    "complete": True, "cost_complete": True, "wall_complete": True,
                    **{key: 0 for key in COST_FIELDS}, "wall_seconds": 0.0}}
            else:
                records[stage] = _run_stage(request, stage)
            offline._write_json_atomic(str(output / "diagnostic-state.json"), {
                "schema": SCHEMA, "status": "running", "stages": records})
    except BaseException as error:
        offline._write_json_atomic(str(output / "diagnostic-state.json"), {
            "schema": SCHEMA, "status": "inconclusive", "stages": records,
            "error_class": type(error).__name__, "note": "See per-stage process records and retained raw bills."})
        raise
    charges = charge_pair(*(records[stage]["metrics"] for stage in STAGES))
    comparison = _compare_features(output, records)
    equivalent = (comparison.get("available") is True
                  and all(comparison.get(key) is True for key in (
                      "same_verdict", "same_verdict_strength", "same_semantic_features",
                      "same_rank_key", "same_final_contract", "same_ordered_final_outcomes")))
    result = {"schema": SCHEMA, "diagnostic_only": True,
              "comparison_complete": (all(row["status"] == "complete" for row in records.values())
                                      and charges["complete"] is True and comparison.get("available") is True),
              "full_e2e_acceptance": False, "stages": records, "charges": charges,
              "result_comparison": comparison, "result_equivalent": equivalent,
              "coordinator_wall_seconds": (time.monotonic_ns() - started) / 1e9,
              "note": "Cold diagnostic only; setup charged once per arm, once physically. No zero-loss or 10x claim."}
    offline._write_json_atomic(str(output / "diagnostic-result.json"), result)
    offline._write_json_atomic(str(output / "diagnostic-state.json"), {
        "schema": SCHEMA, "status": "complete" if result["comparison_complete"] else "inconclusive",
        "stages": records})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-fingerprint")
    parser.add_argument("--runtime-path")
    parser.add_argument("--runtime-read-root", action="append", default=[])
    parser.add_argument("--max-concurrent-seats", type=int, default=12)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid")
    parser.add_argument("--worker", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--request-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.confirm_paid != "GPT-5.6-Luna" or not args.request or not args.request_sha256:
            parser.error("worker requires the paid confirmation and pinned request")
        return _worker(args.request, args.request_sha256, args.worker)
    if not all((args.source, args.output, args.source_fingerprint, args.runtime_path)):
        parser.error("source, output, source-fingerprint and runtime-path are required")
    if args.execute and args.confirm_paid != "GPT-5.6-Luna":
        parser.error("execution requires --confirm-paid GPT-5.6-Luna")
    if not 1 <= args.max_concurrent_seats <= 64:
        parser.error("max-concurrent-seats must be between 1 and 64")
    if not args.output.is_absolute() or ".." in args.output.parts or os.path.lexists(args.output):
        parser.error("output must be a new absolute normalized root")
    source, output = args.source.resolve(strict=True), args.output.resolve()
    if os.path.commonpath((source, output)) in (str(source), str(output)):
        parser.error("source and output must be disjoint")
    _validate_execute_output_root(output)
    if _source_fingerprint(SOURCE_ROOT) != args.source_fingerprint:
        parser.error("source fingerprint differs from operator pin")
    roots = offline_runtime_read_roots({"GV_OFFLINE_RUNTIME_READ_ROOTS": json.dumps(args.runtime_read_root)})
    run_id = "cold-" + uuid.uuid4().hex
    request = {"schema": SCHEMA, "run_id": run_id, "setup_id": run_id + "-setup",
               "source": str(source), "source_seal": offline.verify_immutable_input_anchor(str(source)),
               "source_root": str(SOURCE_ROOT), "source_fingerprint": args.source_fingerprint,
               "output": str(output), "runtime_path": args.runtime_path,
               "runtime_read_roots": list(roots), "max_concurrent_seats": args.max_concurrent_seats,
               "timezone": "Europe/London",
               "model": "gpt-5.6-luna", "effort": "medium", "service_tier": "standard"}
    if not args.execute:
        print(json.dumps(request, indent=2))
        return 0
    result = run(request)
    print(json.dumps({"output": str(output), "comparison_complete": result["comparison_complete"],
                      "result_equivalent": result["result_equivalent"], "full_e2e_acceptance": False}))
    return 0 if result["comparison_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
