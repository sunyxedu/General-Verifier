"""Run a pinned, balanced, cold comparison of full offline verifier sources.

The coordinator is task-agnostic.  Candidate identifiers are opaque and each
pair uses the same sealed input for one original-source control and one
treatment.  It measures subprocess spawn-to-exit wall time, preserves failed
attempt accounting, and projects both versions onto one exact contract view.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid


SCHEMA = "gv-full-offline-comparison-plan-v1"
RESULT_SCHEMA = "gv-full-offline-comparison-result-v1"
RUNTIME = {"model": "gpt-5.6-luna", "effort": "medium", "service_tier": "standard"}
FEATURE_SCHEMA_VERSION = 3
COST_RATES_USD_PER_MILLION = {
    "uncached_input_tokens": 0.20,
    "cached_input_tokens": 0.02,
    "output_tokens": 1.20,
}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_PLAN_KEYS = {
    "schema", "comparison_id", "output_root", "worker_path",
    "worker_sha256", "projection_path", "projection_sha256",
    "coordinator_sha256", "pairs",
}
_PAIR_KEYS = {"candidate_id", "input_attestations", "jobs"}
_JOB_KEYS = {"arm", "plan_path", "plan_sha256"}
_INTEGRITY_FAILURES = {
    "InputSealChanged", "SourceChanged", "EvaluationSupportChanged",
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError("non-finite JSON constant")


def _read_json(path, cap=64 * 1024 * 1024):
    with Path(path).open("rb") as handle:
        raw = handle.read(cap + 1)
    if len(raw) > cap:
        raise ValueError("comparison JSON exceeds size limit")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object,
                       parse_constant=_nonfinite)
    return value, hashlib.sha256(raw).hexdigest()


def _canonical(value, *, existing=False):
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ValueError("comparison paths must be absolute strings")
    path = Path(value)
    if str(path.resolve(strict=existing)) != value:
        raise ValueError("comparison paths must be canonical")
    return path


def _inside(parent, child):
    return os.path.commonpath((str(parent), str(child))) == str(parent)


def _load_pinned(name, path, expected):
    if not isinstance(expected, str) or not _HASH.fullmatch(expected):
        raise ValueError("invalid helper digest")
    file_path = _canonical(path, existing=True)
    if not file_path.is_file() or _sha(file_path) != expected:
        raise ValueError("comparison helper source changed")
    module_name = f"_gv_comparison_{name}_{expected[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ValueError("comparison helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _validate_attestations(value):
    if type(value) is not dict or not value or len(value) > 64:
        raise ValueError("each candidate needs bounded input attestations")
    result = {}
    for path_text, digest in value.items():
        path = _canonical(path_text, existing=True)
        if (not path.is_file() or not isinstance(digest, str)
                or not _HASH.fullmatch(digest) or _sha(path) != digest):
            raise ValueError("input attestation mismatch")
        result[path_text] = digest
    return result


def validate_plan(plan, *, require_fresh_outputs=True):
    """Validate all sources and plans without creating an output or model call."""
    if type(plan) is not dict or set(plan) != _PLAN_KEYS or plan.get("schema") != SCHEMA:
        raise ValueError("wrong comparison plan schema or fields")
    if (not isinstance(plan["comparison_id"], str) or not plan["comparison_id"]
            or len(plan["comparison_id"]) > 256):
        raise ValueError("comparison_id must be a bounded opaque string")
    output = _canonical(plan["output_root"])
    if require_fresh_outputs and (output.exists() or output.is_symlink()):
        raise ValueError("comparison output must be new")
    for root in (Path("/private/tmp"), Path("/private/var/folders")):
        if _inside(root, output):
            raise ValueError("paid comparison cannot use a host temporary output")
    if (not isinstance(plan["coordinator_sha256"], str)
            or not _HASH.fullmatch(plan["coordinator_sha256"])
            or _sha(__file__) != plan["coordinator_sha256"]):
        raise ValueError("comparison coordinator source changed")
    worker = _load_pinned("worker", plan["worker_path"], plan["worker_sha256"])
    projection = _load_pinned(
        "projection", plan["projection_path"], plan["projection_sha256"])
    if not callable(getattr(worker, "validate_plan", None)) or not callable(
            getattr(projection, "project_full_offline_result", None)):
        raise ValueError("comparison helpers do not provide required APIs")
    pairs = plan["pairs"]
    if (not isinstance(pairs, list) or len(pairs) < 2 or len(pairs) > 16
            or len(pairs) % 2):
        raise ValueError("use a bounded even number of candidate pairs")
    seen_candidates = set()
    seen_outputs = set()
    first_arms = []
    normalized = []
    for pair in pairs:
        if type(pair) is not dict or set(pair) != _PAIR_KEYS:
            raise ValueError("wrong candidate pair schema")
        candidate = pair["candidate_id"]
        if (not isinstance(candidate, str) or not candidate or len(candidate) > 256
                or candidate in seen_candidates):
            raise ValueError("candidate identifiers must be unique opaque strings")
        seen_candidates.add(candidate)
        attestations = _validate_attestations(pair["input_attestations"])
        jobs = pair["jobs"]
        if not isinstance(jobs, list) or len(jobs) != 2:
            raise ValueError("each candidate requires exactly two arms")
        arms = []
        loaded_jobs = []
        pair_input = None
        pair_seal = None
        for job in jobs:
            if type(job) is not dict or set(job) != _JOB_KEYS:
                raise ValueError("wrong comparison job schema")
            arm = job["arm"]
            if arm not in ("control", "treatment") or arm in arms:
                raise ValueError("each pair needs one control and one treatment")
            plan_path = _canonical(job["plan_path"], existing=True)
            worker_plan, plan_digest = _read_json(plan_path)
            if (not isinstance(job["plan_sha256"], str)
                    or plan_digest != job["plan_sha256"]):
                raise ValueError("worker plan digest mismatch")
            worker.validate_plan(worker_plan)
            if worker_plan["arm"] != arm:
                raise ValueError("job arm differs from worker plan")
            current_input = worker_plan["input_source"]
            current_seal = worker_plan["input_seal"]
            if pair_input is None:
                pair_input, pair_seal = current_input, current_seal
            elif (current_input, current_seal) != (pair_input, pair_seal):
                raise ValueError("paired arms must share an identical sealed input")
            worker_output = worker_plan["output_root"]
            if worker_output in seen_outputs:
                raise ValueError("worker outputs must be unique")
            seen_outputs.add(worker_output)
            if require_fresh_outputs and Path(worker_output).exists():
                raise ValueError("worker output is not fresh")
            if _inside(output, Path(worker_output)) or _inside(Path(worker_output), output):
                raise ValueError("comparison and worker outputs must be disjoint")
            for private_name in ("source_root", "input_source"):
                private_value = worker_plan.get(private_name)
                if isinstance(private_value, str) and os.path.isabs(private_value):
                    private = Path(private_value)
                    if _inside(private, output) or _inside(output, private):
                        raise ValueError(
                            "comparison output overlaps a source or sealed input")
            arms.append(arm)
            loaded_jobs.append({**job, "worker_plan": worker_plan})
        first_arms.append(arms[0])
        normalized.append({"candidate_id": candidate,
                           "input_attestations": attestations,
                           "jobs": loaded_jobs})
    if first_arms.count("control") != first_arms.count("treatment"):
        raise ValueError("candidate pair execution order must be balanced")
    return worker, projection, normalized


def _write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _private_output(path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(fd, "wb")


def _subprocess_env():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONSTARTUP", None)
    return env


def preflight(plan):
    worker, _projection, pairs = validate_plan(plan)
    reports = []
    for pair in pairs:
        for job in pair["jobs"]:
            command = [sys.executable, plan["worker_path"], "--plan", job["plan_path"],
                       "--plan-sha256", job["plan_sha256"], "--preflight"]
            completed = subprocess.run(
                command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=_subprocess_env(), timeout=300)
            if completed.returncode != 0:
                raise ValueError("worker preflight failed")
            try:
                report = json.loads(completed.stdout.decode("utf-8"),
                                    object_pairs_hook=_strict_object,
                                    parse_constant=_nonfinite)
            except (UnicodeError, ValueError, TypeError) as error:
                raise ValueError("worker preflight output is malformed") from error
            if (type(report) is not dict or report.get("preflight_complete") is not True
                    or report.get("model_calls") != 0
                    or Path(job["worker_plan"]["output_root"]).exists()):
                raise ValueError("worker preflight did not remain read-only")
            reports.append({"candidate_id": pair["candidate_id"],
                            "arm": job["arm"], "report": report})
    # Recheck all hashes after the subprocess imports.
    validate_plan(plan)
    return {"schema": SCHEMA, "preflight_complete": True, "model_calls": 0,
            "runtime": RUNTIME, "jobs": reports,
            "worker_schema": worker.SCHEMA}


def _rank_key(features):
    ranks = {"verified": 4, "supported": 3, "partial": 2,
             "unverified": 1, "refuted": 0}
    if (type(features) is not dict or features.get("unscoreable") is not False
            or features.get("feature_schema_version") != FEATURE_SCHEMA_VERSION):
        return None
    counts, total = features.get("final_obligations"), features.get(
        "final_obligations_total")
    contrary = features.get("final_contrary_signals", 0)
    if (type(counts) is not dict or type(total) is not int or total < 0
            or any(type(value) is not int or value < 0 for value in counts.values())
            or sum(counts.values()) != total or features.get("verdict") not in ranks
            or type(contrary) is not int or contrary < 0):
        return None
    return [ranks[features["verdict"]], -contrary]


def _quality_view(result, projection_report):
    features = result.get("features") if type(result) is dict else None
    if (type(features) is not dict or type(projection_report) is not dict
            or projection_report.get("complete") is not True):
        return None
    rank = _rank_key(features)
    if rank is None:
        return None
    return {
        "ordered_contract": projection_report["ordered_contract"],
        "ordered_outcomes": projection_report["ordered_outcomes"],
        "feature_schema_version": features.get("feature_schema_version"),
        "verdict": features.get("verdict"),
        "verdict_strength": features.get("verdict_strength"),
        "rank_key": rank,
        "final_obligations": features.get("final_obligations"),
        "final_obligations_total": features.get("final_obligations_total"),
        "final_contrary_signals": features.get("final_contrary_signals"),
    }


def compare_pair(control, treatment):
    left, right = control.get("quality_view"), treatment.get("quality_view")
    complete = (control.get("worker_complete") is True
                and treatment.get("worker_complete") is True
                and control.get("accounting_complete") is True
                and treatment.get("accounting_complete") is True
                and treatment.get("prefetch_exercised") is True
                and left is not None and right is not None)
    equal = (left == right) if complete else None
    return {"complete": complete, "exact_quality_equal": equal,
            "control": left, "treatment": right,
            "prefetch_exercised": treatment.get("prefetch_exercised")}


def _accounting_summary(jobs):
    buckets = {name: {
                    "control": {"known_sum": 0, "unknown_attempts": 0},
                    "treatment": {"known_sum": 0, "unknown_attempts": 0},
                }
               for name in ("input_tokens", "uncached_input_tokens",
                            "cached_input_tokens", "cache_write_input_tokens",
                            "output_tokens", "reasoning_tokens")}
    complete = bool(jobs)
    for job in jobs:
        accounting = job.get("accounting")
        if type(accounting) is not dict:
            complete = False
            continue
        if accounting.get("accounting_complete") is not True:
            complete = False
        tokens = accounting.get("tokens")
        if type(tokens) is not dict:
            complete = False
            continue
        for name in buckets:
            row = tokens.get(name)
            if (type(row) is not dict or type(row.get("known_sum")) is not int
                    or row["known_sum"] < 0
                    or type(row.get("unknown_attempts")) is not int
                    or row["unknown_attempts"] < 0):
                complete = False
            else:
                buckets[name][job["arm"]]["known_sum"] += row["known_sum"]
                buckets[name][job["arm"]]["unknown_attempts"] += row[
                    "unknown_attempts"]
                if row["unknown_attempts"]:
                    complete = False
    estimate = None
    ratio = None
    if complete:
        estimate = {}
        for arm in ("control", "treatment"):
            estimate[arm] = sum(
                buckets[name][arm]["known_sum"] * rate / 1_000_000
                for name, rate in COST_RATES_USD_PER_MILLION.items())
        if estimate["treatment"] > 0:
            ratio = estimate["control"] / estimate["treatment"]
    return {"complete": complete, "tokens": buckets,
            "price_equivalent_usd": estimate,
            "control_over_treatment": ratio,
            "rates_usd_per_million": COST_RATES_USD_PER_MILLION,
            "invoice_claim": False}


def _read_worker_result(job):
    path = Path(job["worker_plan"]["output_root"]) / "worker-result.json"
    value, digest = _read_json(path)
    if (type(value) is not dict or value.get("schema") !=
            "gv-full-offline-worker-result-v1" or value.get("status") != "terminal"
            or value.get("plan") != job["worker_plan"]):
        raise ValueError("worker result does not match committed job plan")
    return value, digest


def _postflight_integrity(plan, pair, job):
    if (_sha(__file__) != plan["coordinator_sha256"]
            or _sha(plan["worker_path"]) != plan["worker_sha256"]
            or _sha(plan["projection_path"]) != plan["projection_sha256"]
            or _sha(job["plan_path"]) != job["plan_sha256"]):
        raise ValueError("comparison control source changed during execution")
    _validate_attestations(pair["input_attestations"])


def _run_job(plan, projection, output, pair, job, ordinal):
    stdout_path = output / f"job-{ordinal:02d}.stdout"
    stderr_path = output / f"job-{ordinal:02d}.stderr"
    started = time.monotonic_ns()
    command = [sys.executable, plan["worker_path"], "--plan", job["plan_path"],
               "--plan-sha256", job["plan_sha256"], "--execute",
               "--confirm-paid", "GPT-5.6-Luna"]
    with _private_output(stdout_path) as stdout, _private_output(stderr_path) as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   env=_subprocess_env(), start_new_session=True)
        _write_new(output / f"job-{ordinal:02d}-start.json", {
            "candidate_id": pair["candidate_id"], "arm": job["arm"],
            "pid": process.pid, "started_at_unix_ns": time.time_ns(),
            "automatic_outer_retries": 0,
        })
        try:
            returncode = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            except ProcessLookupError:
                process.wait()
            raise
    wall = (time.monotonic_ns() - started) / 1e9
    result, result_digest = _read_worker_result(job)
    _postflight_integrity(plan, pair, job)
    certificate = result.get("certificate")
    certificate = certificate.get("certificate") if type(certificate) is dict else None
    projection_report = projection.project_full_offline_result(
        str(Path(job["worker_plan"]["output_root"]) / "state" / "events.jsonl"),
        certificate)
    accounting = result.get("accounting")
    audit = result.get("role_and_cache_audit")
    summary = {
        "candidate_id": pair["candidate_id"], "arm": job["arm"],
        "returncode": returncode, "spawn_to_exit_wall_seconds": wall,
        "worker_result_sha256": result_digest,
        "worker_complete": result.get("complete") is True and returncode == 0,
        "worker_failures": result.get("failures"),
        "accounting": accounting,
        "accounting_complete": (type(accounting) is dict
                                and accounting.get("accounting_complete") is True),
        "projection": projection_report,
        "quality_view": _quality_view(result, projection_report),
        "prefetch_exercised": (audit.get("prefetch_exercised")
                               if type(audit) is dict else None),
        "stdout_sha256": _sha(stdout_path), "stderr_sha256": _sha(stderr_path),
    }
    _write_new(output / f"job-{ordinal:02d}-result.json", summary)
    return summary


def run_comparison(plan):
    preflight_report = preflight(plan)
    _worker, projection, pairs = validate_plan(plan)
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic_ns()
    run_id = uuid.uuid4().hex
    _write_new(output / "comparison-start.json", {
        "schema": RESULT_SCHEMA, "status": "started", "run_id": run_id,
        "plan": plan, "preflight": preflight_report,
        "runtime": RUNTIME, "automatic_outer_retries": 0,
        "latency_boundary": "subprocess-spawn-to-exit",
    })
    jobs = []
    pairs_report = []
    integrity_stop = None
    ordinal = 0
    for pair in pairs:
        pair_jobs = []
        for job in pair["jobs"]:
            ordinal += 1
            if integrity_stop is not None:
                skipped = {"candidate_id": pair["candidate_id"], "arm": job["arm"],
                           "skipped": True, "reason": integrity_stop}
                _write_new(output / f"job-{ordinal:02d}-result.json", skipped)
                jobs.append(skipped)
                pair_jobs.append(skipped)
                continue
            try:
                summary = _run_job(plan, projection, output, pair, job, ordinal)
            except BaseException as error:
                summary = {"candidate_id": pair["candidate_id"], "arm": job["arm"],
                           "supervisor_failure": type(error).__name__,
                           "worker_complete": False, "accounting_complete": False}
                _write_new(output / f"job-{ordinal:02d}-result.json", summary)
                integrity_stop = "supervisor_integrity_unavailable"
            failures = summary.get("worker_failures")
            if isinstance(failures, list) and _INTEGRITY_FAILURES.intersection(failures):
                integrity_stop = "worker_reported_integrity_change"
            jobs.append(summary)
            pair_jobs.append(summary)
        by_arm = {job.get("arm"): job for job in pair_jobs}
        comparison = compare_pair(by_arm.get("control", {}), by_arm.get("treatment", {}))
        pairs_report.append({"candidate_id": pair["candidate_id"], **comparison})
    costs = _accounting_summary([job for job in jobs if not job.get("skipped")])
    control_wall = sum(job.get("spawn_to_exit_wall_seconds", 0) for job in jobs
                       if job.get("arm") == "control")
    treatment_wall = sum(job.get("spawn_to_exit_wall_seconds", 0) for job in jobs
                         if job.get("arm") == "treatment")
    latency_complete = (len(jobs) == 2 * len(pairs)
                        and all(type(job.get("spawn_to_exit_wall_seconds")) is float
                                for job in jobs))
    latency_ratio = (control_wall / treatment_wall
                     if latency_complete and treatment_wall > 0 else None)
    quality = (None if any(not pair["complete"] for pair in pairs_report)
               else all(pair["exact_quality_equal"] is True
                        for pair in pairs_report))
    ten_x = bool(quality is True and costs["complete"]
                 and costs["control_over_treatment"] is not None
                 and costs["control_over_treatment"] >= 10
                 and latency_ratio is not None and latency_ratio >= 10)
    terminal = {
        "schema": RESULT_SCHEMA, "status": "terminal", "run_id": run_id,
        "jobs": jobs, "pairs": pairs_report, "cost": costs,
        "latency": {"complete": latency_complete,
                    "control_wall_seconds": control_wall,
                    "treatment_wall_seconds": treatment_wall,
                    "control_over_treatment": latency_ratio,
                    "batch_makespan_seconds": (time.monotonic_ns() - started) / 1e9},
        "sampled_quality_no_loss": quality,
        "sampled_candidate_10x_acceptance": ten_x,
        "general_no_loss_claim": False, "general_10x_claim": False,
        "integrity_stop": integrity_stop,
    }
    _write_new(output / "comparison-result.json", terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid")
    args = parser.parse_args()
    plan, digest = _read_json(args.plan)
    if digest != args.plan_sha256:
        parser.error("operator comparison-plan digest mismatch")
    if args.preflight and args.execute:
        parser.error("preflight and execute are mutually exclusive")
    if args.preflight:
        print(json.dumps(preflight(plan), sort_keys=True))
        return 0
    if not args.execute:
        validate_plan(plan)
        print(json.dumps({"plan": plan, "model_calls": 0}, sort_keys=True))
        return 0
    if args.confirm_paid != "GPT-5.6-Luna":
        parser.error("execution requires --confirm-paid GPT-5.6-Luna")
    result = run_comparison(plan)
    print(json.dumps({"sampled_candidate_10x_acceptance":
                      result["sampled_candidate_10x_acceptance"]}, sort_keys=True))
    return 0 if result["sampled_candidate_10x_acceptance"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
