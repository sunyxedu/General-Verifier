"""Coordinate a same-source, cold, full-offline control/treatment comparison.

The cohort is fixed before execution. Official truth is retained in separate,
pinned coordinator-only files and is never included in worker plans or model
inputs. Warm evaluation is intentionally unsupported until every verifier
stage can emit an exact signed cross-run hit receipt.
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
import stat
import subprocess
import sys
import time
import uuid

try:
    from . import run_full_offline_comparison as _accounting
except ImportError:  # direct script execution
    import run_full_offline_comparison as _accounting


SCHEMA = "gv-full-offline-comparison-plan-v2"
RESULT_SCHEMA = "gv-full-offline-comparison-result-v2"
TRUTH_SCHEMA = "gv-retained-official-truth-v1"
MATERIALIZED_RECORDS_SCHEMA = "gv-offline-v2-materialized-records-v1"
RUNTIME = {
    "model": "gpt-5.6-luna", "effort": "medium",
    "service_tier": "standard",
}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_PLAN_KEYS = {
    "schema", "comparison_id", "regime", "output_root",
    "source", "worker_path", "worker_sha256", "projection_path",
    "projection_sha256", "coordinator_sha256",
    "accounting_helper_sha256", "pairs",
}
_SOURCE_KEYS = {"root", "commit", "fingerprint"}
_PAIR_KEYS = {
    "candidate_id", "task_family", "official_truth",
    "input_attestations", "jobs",
}
_TRUTH_KEYS = {"path", "sha256"}
_TRUTH_DOCUMENT_KEYS = {
    "schema", "candidate_id", "task_family", "label", "input_seal",
    "official_identity", "source_truth_manifest_sha256",
}
_JOB_KEYS = {"arm", "plan_path", "plan_sha256"}
_INTEGRITY_FAILURES = {
    "InputSealChanged", "SourceChanged", "EvaluationSupportChanged",
}


def _sha(path) -> str:
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
    raw_path = Path(path)
    info = raw_path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_size > cap):
        raise ValueError("comparison JSON must be a bounded private regular file")
    with raw_path.open("rb") as handle:
        raw = handle.read(cap + 1)
    after = raw_path.lstat()
    if (len(raw) > cap or (info.st_dev, info.st_ino, info.st_size,
                           info.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError("comparison JSON changed while it was read")
    return (json.loads(raw.decode("utf-8", errors="strict"),
                       object_pairs_hook=_strict_object,
                       parse_constant=_nonfinite),
            hashlib.sha256(raw).hexdigest())


def _path(value, *, existing=False) -> Path:
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ValueError("comparison paths must be canonical absolute strings")
    path = Path(value)
    if str(path.resolve(strict=existing)) != value:
        raise ValueError("comparison paths must contain no aliases")
    return path


def _inside(parent: Path, child: Path) -> bool:
    try:
        return os.path.commonpath((str(parent), str(child))) == str(parent)
    except ValueError:
        return False


def _load_pinned(name, path, digest):
    if not isinstance(digest, str) or not _HASH.fullmatch(digest):
        raise ValueError("helper digest is invalid")
    source = _path(path, existing=True)
    if not source.is_file() or _sha(source) != digest:
        raise ValueError("comparison helper source changed")
    spec = importlib.util.spec_from_file_location(
        f"_gv_offline_comparison_v2_{name}_{digest[:16]}", source)
    if spec is None or spec.loader is None:
        raise ValueError("comparison helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _truth(value, candidate, family, input_seal, private_roots):
    if type(value) is not dict or set(value) != _TRUTH_KEYS:
        raise ValueError("official truth reference is malformed")
    path = _path(value["path"], existing=True)
    if any(_inside(root, path) or _inside(path, root) for root in private_roots):
        raise ValueError("official truth must remain outside model-visible roots")
    document, digest = _read_json(path, cap=16 * 1024)
    if (not isinstance(value["sha256"], str) or digest != value["sha256"]
            or type(document) is not dict
            or set(document) != _TRUTH_DOCUMENT_KEYS
            or document.get("schema") != TRUTH_SCHEMA
            or document.get("candidate_id") != candidate
            or document.get("task_family") != family
            or document.get("label") not in {"positive", "negative"}
            or document.get("input_seal") != input_seal
            or not isinstance(document.get("official_identity"), str)
            or not _HASH.fullmatch(document["official_identity"])
            or not isinstance(
                document.get("source_truth_manifest_sha256"), str)
            or not _HASH.fullmatch(
                document["source_truth_manifest_sha256"])):
        raise ValueError("official truth identity or schema mismatch")
    return document


def materialize_package_records(
    model_manifest_path, model_manifest_sha256,
    truth_manifest_path, truth_manifest_sha256, output_root,
):
    """Convert the general package builder's aggregate manifest generically.

    The conversion is a deterministic schema join. It does not inspect task
    text, candidate artifacts, or domain-specific outputs, and it emits only
    coordinator-side individual truth records plus input identities suitable
    for constructing v2 worker pairs.
    """
    for digest in (model_manifest_sha256, truth_manifest_sha256):
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise ValueError("package manifest identity is invalid")
    model_path = _path(model_manifest_path, existing=True)
    truth_path = _path(truth_manifest_path, existing=True)
    destination = _path(output_root)
    if destination.exists() or destination.is_symlink():
        raise ValueError("materialized truth record root must be fresh")
    model, observed_model_sha = _read_json(model_path)
    truth, observed_truth_sha = _read_json(truth_path)
    if observed_model_sha != model_manifest_sha256 \
            or observed_truth_sha != truth_manifest_sha256:
        raise ValueError("package manifest changed")
    if (type(model) is not dict
            or set(model) != {"schema", "package_id", "cases"}
            or model.get("schema") != "gv-general-offline-model-inputs-v1"
            or type(truth) is not dict
            or set(truth) != {
                "schema", "package_id", "operator_manifest_sha256",
                "model_input_manifest_sha256", "model_input_root", "cases",
            }
            or truth.get("schema")
            != "gv-general-offline-coordinator-truth-v1"
            or truth.get("package_id") != model.get("package_id")
            or truth.get("model_input_manifest_sha256") != observed_model_sha
            or type(model.get("cases")) is not list
            or type(truth.get("cases")) is not list):
        raise ValueError("general package manifests are incompatible")
    model_root = _path(truth.get("model_input_root"), existing=True)
    if model_path != model_root / "input-manifest.json" \
            or _inside(model_root, destination) \
            or _inside(destination, model_root):
        raise ValueError("materialized records must stay outside model inputs")
    model_by_case = {}
    for item in model["cases"]:
        if type(item) is not dict or set(item) != {
                "case_id", "task_family", "candidate_id", "state",
                "input_seal"} or item["case_id"] in model_by_case:
            raise ValueError("model package case is malformed or duplicated")
        model_by_case[item["case_id"]] = item
    truth_by_case = {}
    for item in truth["cases"]:
        if type(item) is not dict or set(item) != {
                "case_id", "task_family", "candidate_id", "input_seal",
                "label", "identity", "provenance"} \
                or item["case_id"] in truth_by_case:
            raise ValueError("truth package case is malformed or duplicated")
        truth_by_case[item["case_id"]] = item
    if set(model_by_case) != set(truth_by_case) or not model_by_case:
        raise ValueError("model/truth package case inventories disagree")
    prepared = []
    for ordinal, case_id in enumerate(sorted(model_by_case), 1):
        visible, retained = model_by_case[case_id], truth_by_case[case_id]
        common = ("task_family", "candidate_id", "input_seal")
        if any(visible.get(key) != retained.get(key) for key in common):
            raise ValueError("model/truth package case identities disagree")
        label = retained.get("label")
        if isinstance(label, bool) or label not in {0, 0.0, 1, 1.0}:
            raise ValueError("offline v2 requires binary numeric official truth")
        identity = retained.get("identity")
        seal = retained.get("input_seal")
        if (not isinstance(identity, str) or not _HASH.fullmatch(identity)
                or not isinstance(seal, str) or not _HASH.fullmatch(seal)):
            raise ValueError("package truth identity is malformed")
        state = (model_root / visible["state"]).resolve(strict=True)
        if not state.is_dir() or not _inside(model_root, state):
            raise ValueError("model package state escapes its root")
        document = {
            "schema": TRUTH_SCHEMA,
            "candidate_id": retained["candidate_id"],
            "task_family": retained["task_family"],
            "label": "positive" if float(label) == 1.0 else "negative",
            "input_seal": seal, "official_identity": identity,
            "source_truth_manifest_sha256": observed_truth_sha,
        }
        prepared.append((case_id, state, document,
                         f"truth-{ordinal:04d}.json"))
    destination.mkdir(mode=0o700)
    records = []
    for case_id, state, document, filename in prepared:
        record_path = destination / filename
        _write_new(record_path, document)
        records.append({
            "case_id": case_id, "candidate_id": document["candidate_id"],
            "task_family": document["task_family"],
            "input_source": str(state), "input_seal": document["input_seal"],
            "official_truth": {
                "path": str(record_path), "sha256": _sha(record_path),
            },
        })
    manifest = {
        "schema": MATERIALIZED_RECORDS_SCHEMA,
        "package_id": model["package_id"],
        "model_input_manifest_sha256": observed_model_sha,
        "truth_manifest_sha256": observed_truth_sha,
        "records": records,
    }
    _write_new(destination / "records.json", manifest)
    return manifest


def validate_plan(plan, *, require_fresh_outputs=True):
    if type(plan) is not dict or set(plan) != _PLAN_KEYS \
            or plan.get("schema") != SCHEMA:
        raise ValueError("wrong offline comparison v2 schema or fields")
    if plan.get("regime") != "cold":
        raise ValueError("v2 supports cold evaluation only; warm is fail-closed")
    if (not isinstance(plan.get("comparison_id"), str)
            or not plan["comparison_id"] or len(plan["comparison_id"]) > 256):
        raise ValueError("comparison_id must be a bounded opaque string")
    output = _path(plan["output_root"])
    if require_fresh_outputs and (output.exists() or output.is_symlink()):
        raise ValueError("comparison output must be fresh")
    for temporary in (Path("/private/tmp"), Path("/private/var/folders")):
        if _inside(temporary, output):
            raise ValueError("paid output must not use a host temporary tree")
    source = plan.get("source")
    if type(source) is not dict or set(source) != _SOURCE_KEYS:
        raise ValueError("comparison requires one exact source identity")
    source_root = _path(source["root"], existing=True)
    if (not source_root.is_dir() or (source_root / ".git").exists()
            or not isinstance(source["commit"], str)
            or not _COMMIT.fullmatch(source["commit"])
            or not isinstance(source["fingerprint"], str)
            or not _HASH.fullmatch(source["fingerprint"])):
        raise ValueError("comparison source must be one frozen exact revision")
    if (not isinstance(plan["coordinator_sha256"], str)
            or _sha(__file__) != plan["coordinator_sha256"]
            or not isinstance(plan["accounting_helper_sha256"], str)
            or _sha(_accounting.__file__)
            != plan["accounting_helper_sha256"]):
        raise ValueError("comparison coordinator/support source changed")
    worker = _load_pinned("worker", plan["worker_path"],
                          plan["worker_sha256"])
    projection = _load_pinned("projection", plan["projection_path"],
                              plan["projection_sha256"])
    if (getattr(worker, "SCHEMA", None) != "gv-full-offline-worker-plan-v2"
            or getattr(worker, "RUNTIME", None) != RUNTIME
            or not callable(getattr(worker, "validate_plan", None))
            or not callable(getattr(
                projection, "project_full_offline_result", None))):
        raise ValueError("comparison helpers do not implement offline v2")
    pairs = plan.get("pairs")
    if (type(pairs) is not list or not 6 <= len(pairs) <= 60
            or len(pairs) % 2):
        raise ValueError("use an even bounded cohort of at least six candidates")
    candidates, outputs, truth_paths = set(), set(), set()
    first_arms, family_labels, normalized = [], {}, []
    for pair in pairs:
        if type(pair) is not dict or set(pair) != _PAIR_KEYS:
            raise ValueError("candidate pair schema is malformed")
        candidate, family = pair["candidate_id"], pair["task_family"]
        if (not isinstance(candidate, str) or not candidate
                or len(candidate) > 256 or candidate in candidates
                or not isinstance(family, str) or not family
                or len(family) > 128):
            raise ValueError("candidate/family identities must be bounded and unique")
        candidates.add(candidate)
        jobs = pair["jobs"]
        if type(jobs) is not list or len(jobs) != 2:
            raise ValueError("each candidate requires exactly two arms")
        loaded, arms, common_plan = [], [], None
        input_root = None
        for job in jobs:
            if type(job) is not dict or set(job) != _JOB_KEYS:
                raise ValueError("comparison job schema is malformed")
            if job["arm"] not in {"control", "treatment"} \
                    or job["arm"] in arms:
                raise ValueError("each candidate requires one arm of each kind")
            plan_path = _path(job["plan_path"], existing=True)
            worker_plan, digest = _read_json(plan_path)
            if (not isinstance(job["plan_sha256"], str)
                    or digest != job["plan_sha256"]):
                raise ValueError("worker plan identity mismatch")
            worker.validate_plan(worker_plan)
            if (worker_plan.get("arm") != job["arm"]
                    or {"root": worker_plan.get("source_root"),
                        "commit": worker_plan.get("source_commit"),
                        "fingerprint": worker_plan.get("source_fingerprint")}
                    != source):
                raise ValueError("all arms must use the comparison's exact source")
            comparable = {key: value for key, value in worker_plan.items()
                          if key not in {"arm", "output_root"}}
            if common_plan is None:
                common_plan = comparable
                input_root = Path(worker_plan["input_source"])
            elif comparable != common_plan:
                raise ValueError("paired worker plans differ beyond arm/output flags")
            worker_output = Path(worker_plan["output_root"])
            if str(worker_output) in outputs:
                raise ValueError("worker outputs must be unique")
            outputs.add(str(worker_output))
            if require_fresh_outputs and worker_output.exists():
                raise ValueError("worker output is not fresh")
            for private in (source_root, input_root, worker_output):
                if _inside(private, output) or _inside(output, private):
                    raise ValueError("comparison output overlaps a private root")
            arms.append(job["arm"])
            loaded.append({**job, "worker_plan": worker_plan})
        attestations = pair["input_attestations"]
        if type(attestations) is not dict or not attestations \
                or len(attestations) > 64:
            raise ValueError("candidate input attestations are required")
        for path_text, digest in attestations.items():
            path = _path(path_text, existing=True)
            if (not path.is_file() or not isinstance(digest, str)
                    or not _HASH.fullmatch(digest) or _sha(path) != digest
                    or not _inside(input_root, path)):
                raise ValueError("candidate input attestation mismatch")
        truth = _truth(
            pair["official_truth"], candidate, family,
            common_plan["input_seal"],
            (source_root, input_root, output,
             *(Path(job["worker_plan"]["output_root"]) for job in loaded)))
        if pair["official_truth"]["path"] in truth_paths:
            raise ValueError("each candidate requires an independent truth record")
        truth_paths.add(pair["official_truth"]["path"])
        family_labels.setdefault(family, set()).add(truth["label"])
        first_arms.append(arms[0])
        normalized.append({
            "candidate_id": candidate, "task_family": family,
            "truth": truth, "truth_sha256": pair["official_truth"]["sha256"],
            "input_attestations": dict(attestations), "jobs": loaded,
        })
    if (len(family_labels) < 3
            or any(labels != {"positive", "negative"}
                   for labels in family_labels.values())):
        raise ValueError("cohort needs positive/negative cases in at least three families")
    if first_arms.count("control") != first_arms.count("treatment"):
        raise ValueError("candidate execution order must be balanced")
    return worker, projection, normalized


def _write_new(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _private_output(path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(descriptor, "wb")


def _subprocess_env():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in list(env):
        if name.startswith("GV_") or name in {
                "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}:
            env.pop(name, None)
    return env


def _worker_command(plan, job, *, preflight=False):
    command = [sys.executable, plan["worker_path"], "--plan", job["plan_path"],
               "--plan-sha256", job["plan_sha256"]]
    if preflight:
        return [*command, "--preflight"]
    return [*command, "--execute", "--confirm-paid", "GPT-5.6-Luna"]


def preflight(plan):
    worker, _projection, pairs = validate_plan(plan)
    reports = []
    for pair in pairs:
        for job in pair["jobs"]:
            completed = subprocess.run(
                _worker_command(plan, job, preflight=True), check=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=_subprocess_env(), timeout=300)
            try:
                report = json.loads(
                    completed.stdout.decode("utf-8", errors="strict"),
                    object_pairs_hook=_strict_object,
                    parse_constant=_nonfinite)
            except (UnicodeError, ValueError, TypeError) as error:
                raise ValueError("worker preflight output is malformed") from error
            if (completed.returncode != 0 or type(report) is not dict
                    or report.get("preflight_complete") is not True
                    or report.get("model_calls") != 0
                    or report.get("runtime") != RUNTIME
                    or Path(job["worker_plan"]["output_root"]).exists()):
                raise ValueError("worker preflight did not remain read-only")
            reports.append({"candidate_id": pair["candidate_id"],
                            "arm": job["arm"], "report": report})
    validate_plan(plan)
    return {
        "schema": SCHEMA, "preflight_complete": True, "model_calls": 0,
        "runtime": RUNTIME, "regime": "cold", "jobs": reports,
        "worker_schema": worker.SCHEMA,
        "warm_cache": {
            "supported": False,
            "reason": ("no complete Extractor-through-Judger cross-run "
                       "exact signed artifact hit/receipt protocol"),
        },
    }


def _quality_view(result, projection):
    features = result.get("features") if type(result) is dict else None
    if type(features) is not dict or type(projection) is not dict \
            or projection.get("complete") is not True:
        return None
    return {
        "ordered_contract": projection.get("ordered_contract"),
        "ordered_outcomes": projection.get("ordered_outcomes"),
        "verdict": features.get("verdict"),
        "verdict_strength": features.get("verdict_strength"),
        "final_obligations": features.get("final_obligations"),
        "final_obligations_total": features.get("final_obligations_total"),
        "final_contrary_signals": features.get("final_contrary_signals"),
    }


def compare_pair(control, treatment, truth):
    expected = truth.get("label") if type(truth) is dict else None
    views = {"control": control.get("quality_view"),
             "treatment": treatment.get("quality_view")}
    verdicts = {arm: (view.get("verdict") if type(view) is dict else None)
                for arm, view in views.items()}
    predictions = {
        arm: ("positive" if verdict in {"verified", "supported"}
              else "negative" if verdict == "refuted" else None)
        for arm, verdict in verdicts.items()
    }
    arm_eligible = {
        arm: (job.get("worker_complete") is True
              and job.get("accounting_complete") is True
              and job.get("truth_eligible") is True
              and views[arm] is not None
              and predictions[arm] is not None)
        for arm, job in (("control", control), ("treatment", treatment))
    }
    truth_correct = {arm: arm_eligible[arm] and predictions[arm] == expected
                     for arm in arm_eligible}
    exact = views["control"] == views["treatment"] \
        if all(arm_eligible.values()) else None
    # Truth-relative no-regression is distinct from correctness: two equal,
    # settled substantive errors are not a regression, but remain incorrect.
    no_regression = bool(
        all(arm_eligible.values())
        and (not truth_correct["control"] or truth_correct["treatment"]))
    absolute_truth_pass = bool(all(truth_correct.values()))
    return {
        "complete": all(arm_eligible.values()),
        "official_label": expected, "verdicts": verdicts,
        "truth_eligible": arm_eligible, "truth_correct": truth_correct,
        "exact_quality_equal": exact, "no_regression_pass": no_regression,
        "absolute_truth_pass": absolute_truth_pass,
        "control": views["control"], "treatment": views["treatment"],
    }


def _latency_summary(jobs, expected_jobs):
    arms = {"control": 0.0, "treatment": 0.0}
    complete = len(jobs) == expected_jobs
    for job in jobs:
        wall = job.get("spawn_to_exit_wall_seconds")
        arm = job.get("arm")
        if arm not in arms or type(wall) is not float or wall < 0:
            complete = False
        else:
            arms[arm] += wall
    ratio = (arms["control"] / arms["treatment"]
             if complete and arms["treatment"] > 0 else None)
    return {"complete": complete, "seconds": arms,
            "control_over_treatment": ratio,
            "boundary": "subprocess-spawn-to-exit"}


def _ten_x_gate(quality_pass, cost, latency):
    if not quality_pass:
        return False
    if (cost.get("complete") is not True
            or cost.get("control_over_treatment") is None
            or latency.get("complete") is not True
            or latency.get("control_over_treatment") is None):
        return None
    return (cost["control_over_treatment"] >= 10
            and latency["control_over_treatment"] >= 10)


def _read_worker_result(job):
    result, digest = _read_json(
        Path(job["worker_plan"]["output_root"]) / "worker-result.json")
    if (type(result) is not dict
            or result.get("schema") != "gv-full-offline-worker-result-v2"
            or result.get("status") != "terminal"
            or result.get("plan") != job["worker_plan"]):
        raise ValueError("worker result does not match its committed plan")
    return result, digest


def _postflight(plan, pair, job):
    if (_sha(__file__) != plan["coordinator_sha256"]
            or _sha(_accounting.__file__) != plan["accounting_helper_sha256"]
            or _sha(plan["worker_path"]) != plan["worker_sha256"]
            or _sha(plan["projection_path"]) != plan["projection_sha256"]
            or _sha(job["plan_path"]) != job["plan_sha256"]
            or _sha(pair["official_truth"]["path"])
            != pair["official_truth"]["sha256"]):
        raise ValueError("comparison source or retained truth changed")
    for path, digest in pair["input_attestations"].items():
        if _sha(path) != digest:
            raise ValueError("candidate input attestation changed")


def _run_job(plan, projection, output, pair, job, ordinal):
    stdout_path = output / f"job-{ordinal:03d}.stdout"
    stderr_path = output / f"job-{ordinal:03d}.stderr"
    with _private_output(stdout_path) as stdout, \
            _private_output(stderr_path) as stderr:
        started = time.monotonic_ns()
        process = subprocess.Popen(
            _worker_command(plan, job), stdout=stdout, stderr=stderr,
            env=_subprocess_env(), start_new_session=True)
        _write_new(output / f"job-{ordinal:03d}-start.json", {
            "candidate_id": pair["candidate_id"], "arm": job["arm"],
            "pid": process.pid, "started_at_unix_ns": time.time_ns(),
            "automatic_outer_retries": 0,
        })
        try:
            returncode = process.wait()
            finished = time.monotonic_ns()
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
    base = {
        "candidate_id": pair["candidate_id"], "arm": job["arm"],
        "worker_output_root": job["worker_plan"]["output_root"],
        "returncode": returncode,
        "spawn_to_exit_wall_seconds": (finished - started) / 1e9,
        "stdout_sha256": _sha(stdout_path), "stderr_sha256": _sha(stderr_path),
    }
    try:
        result, result_digest = _read_worker_result(job)
        _postflight(plan, pair, job)
        cert = result.get("certificate")
        cert = cert.get("certificate") if type(cert) is dict else None
        projected = projection.project_full_offline_result(
            str(Path(job["worker_plan"]["output_root"]) / "state/events.jsonl"),
            cert)
    except BaseException as error:
        recovered, result_digest = _recover(job)
        summary = {
            **base, "supervisor_failure": type(error).__name__,
            "worker_complete": False, "accounting": recovered,
            "accounting_complete": False,
            "recovered_worker_result_sha256": result_digest,
        }
        _write_new(output / f"job-{ordinal:03d}-result.json", summary)
        return summary
    accounting = result.get("accounting")
    summary = {
        **base, "worker_result_sha256": result_digest,
        "worker_complete": result.get("complete") is True and returncode == 0,
        "worker_failures": result.get("failures"),
        "accounting": accounting,
        "accounting_complete": (type(accounting) is dict
                                and accounting.get("accounting_complete") is True),
        "truth_eligible": result.get("truth_eligible") is True,
        "projection": projected, "quality_view": _quality_view(result, projected),
        "attempt_audit": result.get("attempt_audit"),
    }
    _write_new(output / f"job-{ordinal:03d}-result.json", summary)
    return summary


def _recover(job):
    try:
        result, digest = _read_json(
            Path(job["worker_plan"]["output_root"]) / "worker-result.json")
    except BaseException:
        return None, None
    accounting = result.get("accounting") if type(result) is dict else None
    return accounting if type(accounting) is dict else None, digest


def _stratum_report(key, candidate_ids, jobs, pair_reports):
    cost = _accounting._accounting_summary(jobs, expected_jobs=2 * len(candidate_ids))
    latency = _latency_summary(jobs, 2 * len(candidate_ids))
    relevant = [row for row in pair_reports
                if row["candidate_id"] in set(candidate_ids)]
    no_regression = (len(relevant) == len(candidate_ids)
                     and all(row["no_regression_pass"] is True
                             for row in relevant))
    absolute_truth = (len(relevant) == len(candidate_ids)
                      and all(row["absolute_truth_pass"] is True
                              for row in relevant))
    return {
        "stratum": key, "candidate_ids": list(candidate_ids),
        "no_regression_pass": no_regression,
        "all_truth_correct": absolute_truth,
        "cost": cost, "latency": latency,
        "ten_x_no_regression_acceptance": _ten_x_gate(
            no_regression, cost, latency),
        "ten_x_all_truth_correct_acceptance": _ten_x_gate(
            absolute_truth, cost, latency),
    }


def run_comparison(plan):
    preflight_report = preflight(plan)
    _worker, projection, pairs = validate_plan(plan)
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    run_id, started = uuid.uuid4().hex, time.monotonic_ns()
    _write_new(output / "comparison-start.json", {
        "schema": RESULT_SCHEMA, "status": "started", "run_id": run_id,
        "plan": plan, "preflight": preflight_report, "runtime": RUNTIME,
        "regime": "cold", "latency_boundary": "subprocess-spawn-to-exit",
        "automatic_outer_retries": 0,
    })
    jobs, pair_reports, integrity_stop = [], [], None
    ordinal = 0
    original_by_id = {pair["candidate_id"]: raw
                      for pair, raw in zip(pairs, plan["pairs"])}
    for pair in pairs:
        pair_jobs = []
        raw_pair = original_by_id[pair["candidate_id"]]
        for job in pair["jobs"]:
            ordinal += 1
            if integrity_stop is not None:
                summary = {
                    "candidate_id": pair["candidate_id"], "arm": job["arm"],
                    "skipped": True, "reason": integrity_stop,
                    "worker_complete": False, "accounting_complete": False,
                }
            else:
                try:
                    summary = _run_job(
                        plan, projection, output, raw_pair, job, ordinal)
                except BaseException as error:
                    recovered, digest = _recover(job)
                    summary = {
                        "candidate_id": pair["candidate_id"], "arm": job["arm"],
                        "worker_output_root": job["worker_plan"]["output_root"],
                        "supervisor_failure": type(error).__name__,
                        "worker_complete": False, "accounting": recovered,
                        "accounting_complete": False,
                        "recovered_worker_result_sha256": digest,
                    }
                    integrity_stop = "supervisor_integrity_unavailable"
            if summary.get("skipped") or "supervisor_failure" in summary:
                if summary.get("skipped"):
                    _write_new(output / f"job-{ordinal:03d}-result.json", summary)
                if "supervisor_failure" in summary:
                    integrity_stop = "supervisor_integrity_unavailable"
            failures = summary.get("worker_failures")
            if isinstance(failures, list) \
                    and _INTEGRITY_FAILURES.intersection(failures):
                integrity_stop = "worker_reported_integrity_change"
            jobs.append(summary)
            pair_jobs.append(summary)
        by_arm = {job.get("arm"): job for job in pair_jobs}
        pair_reports.append({
            "candidate_id": pair["candidate_id"],
            "task_family": pair["task_family"],
            "truth_sha256": pair["truth_sha256"],
            **compare_pair(by_arm.get("control", {}),
                           by_arm.get("treatment", {}), pair["truth"]),
        })
    aggregate_cost = _accounting._accounting_summary(
        jobs, expected_jobs=2 * len(pairs))
    aggregate_latency = _latency_summary(jobs, 2 * len(pairs))
    cohort_no_regression = (
        len(pair_reports) == len(pairs)
        and all(row["no_regression_pass"] for row in pair_reports))
    cohort_truth_correct = (
        len(pair_reports) == len(pairs)
        and all(row["absolute_truth_pass"] for row in pair_reports))
    strata = {}
    for pair in pairs:
        key = pair["task_family"] + "|" + pair["truth"]["label"]
        strata.setdefault(key, []).append(pair["candidate_id"])
    stratum_reports = []
    for key, ids in sorted(strata.items()):
        selected_jobs = [job for job in jobs if job.get("candidate_id") in ids]
        stratum_reports.append(
            _stratum_report(key, ids, selected_jobs, pair_reports))
    aggregate_ten_x = _ten_x_gate(
        cohort_no_regression, aggregate_cost, aggregate_latency)
    strict_ten_x = _ten_x_gate(
        cohort_truth_correct, aggregate_cost, aggregate_latency)
    if aggregate_ten_x is True and any(
            row["ten_x_no_regression_acceptance"] is not True
            for row in stratum_reports):
        aggregate_ten_x = False
    if strict_ten_x is True and any(
            row["ten_x_all_truth_correct_acceptance"] is not True
            for row in stratum_reports):
        strict_ten_x = False
    result = {
        "schema": RESULT_SCHEMA, "status": "terminal", "run_id": run_id,
        "regime": "cold", "source": plan["source"], "runtime": RUNTIME,
        "jobs": jobs, "pairs": pair_reports, "strata": stratum_reports,
        "aggregate": {
            "planned_candidates": len(pairs),
            "reported_candidates": len(pair_reports),
            "no_regression_pass": cohort_no_regression,
            "all_truth_correct": cohort_truth_correct,
            "cost": aggregate_cost,
            "latency": {**aggregate_latency,
                        "batch_makespan_seconds":
                            (time.monotonic_ns() - started) / 1e9},
            "ten_x_no_regression_acceptance": aggregate_ten_x,
            "ten_x_all_truth_correct_acceptance": strict_ten_x,
        },
        "warm_cache": {
            "supported": False, "acceptance": None,
            "reason": ("no complete Extractor-through-Judger cross-run "
                       "exact signed artifact hit/receipt protocol"),
        },
        "selection_policy": "all-precommitted-candidates-and-strata",
        "general_no_loss_claim": False, "general_10x_claim": False,
        "integrity_stop": integrity_stop,
    }
    _write_new(output / "comparison-result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid")
    args = parser.parse_args(argv)
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
    print(json.dumps({"ten_x_no_regression_acceptance":
                      result["aggregate"]["ten_x_no_regression_acceptance"],
                      "ten_x_all_truth_correct_acceptance": result[
                          "aggregate"]["ten_x_all_truth_correct_acceptance"]},
                     sort_keys=True))
    return 0 if result["aggregate"][
        "ten_x_no_regression_acceptance"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
