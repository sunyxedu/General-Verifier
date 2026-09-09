"""Warm treatment scoring: every candidate of one task in one process.

The cold worker scores one candidate per fresh process.  Best-of-N
selection scores several candidates of the same task, and the treatment
source can share audited work between them in one process: the
Critic-approved, candidate-blind final contract (``GV_CONTRACT_CACHE=1``)
and Judger-accepted typed methods in their accepted direction
(``GV_METHOD_CACHE=1``).  This worker runs the sealed candidates of one task
family sequentially in one process with the treatment profile plus those
two flags, keeps one observer and one usage ledger for the batch with a
per-candidate context, and records per-candidate features, certificates,
wall time and cache events.  Truth never enters the plan.  Nothing is
retried; a failed candidate keeps its charges and the batch continues.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import uuid

SCHEMA = "gv-full-offline-warm-worker-plan-v1"
RESULT_SCHEMA = "gv-full-offline-warm-worker-result-v1"
RUNTIME = {"model": "gpt-5.6-luna", "effort": "medium",
           "service_tier": "standard"}
PINNED = (
    "run_full_offline_worker.py", "observe_codex_attempts.py",
    "analyze_codex_observer.py", "eval_offline_policy.py",
    "run_typed_adapter_eval.py",
)
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_PLAN_KEYS = {
    "schema", "source_root", "source_commit", "source_fingerprint",
    "task_family", "candidates", "output_root", "runtime_read_roots",
    "support_sha256",
}
_CANDIDATE_KEYS = {"candidate_id", "input_source", "input_seal"}
CACHE_EVENTS = (
    "contract_cache_hit", "contract_cache_published", "contract_cache_miss",
    "contract_cache_bypass", "method_cache_hit",
    "method_cache_direction_audit", "method_cache_bypass",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _support(name):
    path = Path(__file__).with_name(name)
    module_name = "_gv_warm_support_" + name + "_" + _sha(path)[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_json(path, cap=64 * 1024):
    with Path(path).open("rb") as handle:
        raw = handle.read(cap + 1)
    if len(raw) > cap:
        raise ValueError("JSON input exceeds worker size limit")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object,
                       parse_constant=lambda _v: (_ for _ in ()).throw(
                           ValueError("non-finite JSON constant")))
    return value, hashlib.sha256(raw).hexdigest()


def _path(value, *, existing=False):
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ValueError("worker paths must be explicit absolute strings")
    path = Path(value)
    if str(path.resolve(strict=existing)) != value:
        raise ValueError("worker paths must be canonical, without aliases")
    return path


def _inside(parent, child):
    return os.path.commonpath((str(parent), str(child))) == str(parent)


def validate_plan(plan):
    if type(plan) is not dict or set(plan) != _PLAN_KEYS or plan["schema"] != SCHEMA:
        raise ValueError("wrong warm worker plan schema or fields")
    if (not isinstance(plan["source_commit"], str)
            or not _COMMIT.fullmatch(plan["source_commit"])):
        raise ValueError("source commit must be pinned")
    if (not isinstance(plan["source_fingerprint"], str)
            or not _HASH.fullmatch(plan["source_fingerprint"])):
        raise ValueError("source fingerprint must be pinned")
    if not isinstance(plan["task_family"], str) or not plan["task_family"]:
        raise ValueError("task family must be named")
    source = _path(plan["source_root"], existing=True)
    if (source / ".git").exists():
        raise ValueError("use a frozen source archive, not an active repository")
    output = _path(plan["output_root"])
    if output.exists() or output.is_symlink():
        raise ValueError("worker output must be new; no resume or overwrite")
    for root in (Path("/private/tmp"), Path("/private/var/folders")):
        if _inside(root, output):
            raise ValueError("paid worker output must not be in host temporary trees")
    candidates = plan["candidates"]
    if (not isinstance(candidates, list) or not 2 <= len(candidates) <= 64):
        raise ValueError("a warm batch needs two to sixty-four candidates")
    seen = set()
    for candidate in candidates:
        if type(candidate) is not dict or set(candidate) != _CANDIDATE_KEYS:
            raise ValueError("wrong warm candidate schema")
        identifier = candidate["candidate_id"]
        if (not isinstance(identifier, str) or not identifier
                or len(identifier) > 256 or identifier in seen
                or "/" in identifier or identifier in {".", ".."}):
            raise ValueError("candidate identifiers must be unique path-safe strings")
        seen.add(identifier)
        inputs = _path(candidate["input_source"], existing=True)
        if not inputs.is_dir():
            raise ValueError("candidate input must be a directory")
        if (not isinstance(candidate["input_seal"], str)
                or not _HASH.fullmatch(candidate["input_seal"])):
            raise ValueError("candidate input seal must be pinned")
        for left, right in ((source, inputs), (source, output), (inputs, output)):
            if _inside(left, right) or _inside(right, left):
                raise ValueError("source, input and output roots must be disjoint")
    roots = plan["runtime_read_roots"]
    if not isinstance(roots, list) or len(roots) > 16:
        raise ValueError("runtime roots must be an explicit bounded list")
    for value in roots:
        root = _path(value, existing=True)
        if not root.is_dir() or value == "/":
            raise ValueError("runtime roots must be real directories")
        if _inside(root, Path.home().resolve()) or _inside(Path.home().resolve(), root):
            raise ValueError("runtime roots overlap the user home")
    pins = plan["support_sha256"]
    if type(pins) is not dict or set(pins) != set(PINNED):
        raise ValueError("every warm support file must be pinned")
    for name, digest in pins.items():
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise ValueError("invalid support source digest")
        if _sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("warm evaluation support source changed")
    return plan


def _environment(plan, cold):
    """The treatment profile plus explicit process-local reuse."""
    cold._environment({"arm": "treatment",
                       "runtime_read_roots": plan["runtime_read_roots"]})
    os.environ["GV_CONTRACT_CACHE"] = "1"
    os.environ["GV_METHOD_CACHE"] = "1"


def _event_counts(state):
    counts = {}
    for row in _support("run_full_offline_worker")._read_jsonl(
            Path(state) / "events.jsonl"):
        kind = row.get("type")
        if isinstance(kind, str):
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def run_worker(plan):
    validate_plan(plan)
    started_unix = time.time_ns()
    started_mono = time.monotonic_ns()
    cold = _support("run_full_offline_worker")
    _environment(plan, cold)
    source = Path(plan["source_root"])
    output = Path(plan["output_root"])
    source_tools = _support("run_typed_adapter_eval")
    if source_tools._source_fingerprint(source) != plan["source_fingerprint"]:
        raise ValueError("frozen verifier fingerprint mismatch")
    observer = _support("observe_codex_attempts")
    analyzer = _support("analyze_codex_observer")
    policy = _support("eval_offline_policy")
    offline, backend = cold._load_runtime(source)
    if not hasattr(backend, "_run_codex_process"):
        raise ValueError("warm scoring requires the treatment observer seam")
    for candidate in plan["candidates"]:
        if offline.verify_immutable_input_anchor(
                candidate["input_source"]) != candidate["input_seal"]:
            raise ValueError("operator input seal mismatch before scoring")
    output.mkdir(parents=True, exist_ok=False)
    worker_id = uuid.uuid4().hex
    states = {}
    contexts = {}
    for candidate in plan["candidates"]:
        state = output / candidate["candidate_id"] / "state"
        states[candidate["candidate_id"]] = state
        contexts[str(state)] = {
            "arm": "treatment-warm", "worker_id": worker_id,
            "candidate_id": candidate["candidate_id"]}
    start = {
        "schema": RESULT_SCHEMA, "status": "started", "plan": plan,
        "runtime": RUNTIME, "worker_id": worker_id,
        "started_at_unix_ns": started_unix,
        "scope": "full-offline-score-warm-task-batch",
        "certificate_scope": "final",
        "contract_cache": True, "accepted_method_reuse": True,
        "automatic_outer_retries": 0,
    }
    cold._write_new(output / "worker-start.json", start)
    results = []
    failures = []
    observation = overlay = accounting = None
    try:
        bindings = tuple(
            policy.OfflinePolicyBinding(
                state_dir=str(state),
                runtime_read_roots=tuple(plan["runtime_read_roots"]))
            for state in states.values())
        with policy.eval_offline_policy(backend, bindings=bindings) as overlay:
            with observer.observe_codex_attempts(
                    backend, output / "observer.jsonl",
                    seam="run-codex-process", context_by_cwd=contexts) as observation:
                with backend.codex_usage_ledger(
                        str(output / "native-usage.jsonl"),
                        context_by_cwd=contexts):
                    for candidate in plan["candidates"]:
                        identifier = candidate["candidate_id"]
                        state = states[identifier]
                        record = {"candidate_id": identifier,
                                  "input_seal": candidate["input_seal"],
                                  "features": None, "certificate": None,
                                  "event_counts": None, "failures": [],
                                  "wall_seconds": None}
                        results.append(record)
                        begun = time.monotonic_ns()
                        try:
                            state.parent.mkdir(parents=True, exist_ok=False)
                            cold._clone(offline, Path(candidate["input_source"]),
                                        state, candidate["input_seal"])
                            features = offline.score(
                                str(state), **RUNTIME, seats="codex",
                                max_rounds=3, parallel_claims=4,
                                parallel_obligations=8, shared_dir=None,
                                compact_inspection=False,
                                judger_prefetch=True)
                            record["features"] = features
                            record["certificate"] = cold._certificate(
                                state, features)
                        except BaseException as error:
                            record["failures"].append(type(error).__name__)
                            failures.append(f"{identifier}:{type(error).__name__}")
                            if not isinstance(error, Exception):
                                raise
                        finally:
                            record["wall_seconds"] = (
                                time.monotonic_ns() - begun) / 1e9
                            try:
                                record["event_counts"] = _event_counts(state)
                            except Exception:
                                record["event_counts"] = None
    except BaseException as error:
        failures.append(type(error).__name__)
        raise
    finally:
        try:
            if observation is not None:
                accounting = analyzer.analyze_file(
                    output / "observer.jsonl", context_by_cwd=contexts,
                    **RUNTIME, session_summary=observation.summary(),
                    authority_scratch_prefix=os.path.join(
                        os.path.realpath(tempfile.gettempdir()),
                        "gv-authority-"))
                if not accounting["accounting_complete"]:
                    failures.append("AccountingIncomplete")
            else:
                failures.append("ObserverUnavailable")
            for candidate in plan["candidates"]:
                state = states[candidate["candidate_id"]]
                if (offline.verify_immutable_input_anchor(
                        candidate["input_source"]) != candidate["input_seal"]
                        or (state.exists()
                            and offline.verify_immutable_input_anchor(str(state))
                            != candidate["input_seal"])):
                    failures.append("InputSealChanged")
            if source_tools._source_fingerprint(source) != plan["source_fingerprint"]:
                failures.append("SourceChanged")
            if any(_sha(Path(__file__).with_name(name)) != digest
                   for name, digest in plan["support_sha256"].items()):
                failures.append("EvaluationSupportChanged")
        except BaseException as error:
            failures.append("Postflight" + type(error).__name__)
        result = {
            **start, "status": "terminal", "complete": not failures,
            "failures": failures, "candidates": results,
            "accounting": accounting,
            "observer_summary": (observation.summary()
                                 if observation is not None else None),
            "isolation_overlay_identity": (overlay.identity
                                           if overlay is not None else None),
            "finished_at_unix_ns": time.time_ns(),
            "batch_wall_seconds": (time.monotonic_ns() - started_mono) / 1e9,
        }
        cold._write_new(output / "worker-result.json", result)
    return result


def build(args):
    """Freeze one commit and write one warm plan per task family."""
    try:
        from . import build_full_offline_comparison_plan as cold_builder
    except ImportError:  # direct script execution
        import build_full_offline_comparison_plan as cold_builder
    import shutil
    repo = Path(args.repo).resolve()
    run_root = Path(args.run_root)
    if run_root.exists():
        raise SystemExit(f"run root exists: {run_root}")
    run_root.mkdir(parents=True)
    run_root = run_root.resolve()
    commit = cold_builder._archive(repo, args.commit, run_root / "sources" / "treatment")
    support = run_root / "support"
    support.mkdir()
    names = (*PINNED, "run_full_offline_warm_worker.py")
    for name in names:
        shutil.copy2(repo / "experiments" / "terminal_bench" / name, support / name)
    pins = {name: cold_builder._sha(support / name) for name in PINNED}
    tools = cold_builder._module(support / "run_typed_adapter_eval.py")
    fingerprint = tools._source_fingerprint(run_root / "sources" / "treatment")
    package = Path(args.general_inputs).resolve()
    operator = json.loads((package / "operator-manifest.json").read_text())
    inputs = run_root / "inputs" / "model-inputs"
    shutil.copytree(package / "model-inputs", inputs, symlinks=False)
    families = {}
    for case in operator["cases"]:
        source = inputs / "cases" / case["case_id"] / "gv"
        seal = cold_builder._sealed(source)
        if seal != case["input_seal"]:
            raise SystemExit(f"input seal disagrees for {case['case_id']}")
        families.setdefault(case["task_family"], []).append({
            "candidate_id": case["candidate_id"], "input_source": str(source),
            "input_seal": seal})
    plans = run_root / "plans"
    plans.mkdir()
    (run_root / "worker-outputs").mkdir()
    written = []
    for family, candidates in sorted(families.items()):
        if len(candidates) < 2:
            continue
        plan = {
            "schema": SCHEMA,
            "source_root": str(run_root / "sources" / "treatment"),
            "source_commit": commit, "source_fingerprint": fingerprint,
            "task_family": family, "candidates": candidates,
            "output_root": str(run_root / "worker-outputs" / family),
            "runtime_read_roots": list(
                args.runtime_read_root
                or cold_builder.DEFAULT_RUNTIME_READ_ROOTS),
            "support_sha256": pins,
        }
        path = plans / f"warm-{family}.json"
        digest = cold_builder._write_json(path, plan)
        written.append({"plan": str(path), "plan_sha256": digest,
                        "task_family": family,
                        "candidates": len(candidates)})
    print(json.dumps({"commit": commit, "fingerprint": fingerprint,
                      "worker": str(support / "run_full_offline_warm_worker.py"),
                      "plans": written, "model_calls": 0},
                     indent=1, sort_keys=True))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--plan-sha256", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--confirm-paid")
    make = sub.add_parser("build")
    make.add_argument("--repo", required=True)
    make.add_argument("--commit", required=True)
    make.add_argument("--general-inputs", required=True)
    make.add_argument("--run-root", required=True)
    make.add_argument("--runtime-read-root", action="append")
    args = parser.parse_args()
    if args.command == "build":
        return build(args)
    plan, digest = _read_json(args.plan)
    if digest != args.plan_sha256:
        parser.error("operator warm-plan digest mismatch")
    validate_plan(plan)
    if not args.execute:
        print(json.dumps({"plan": plan, "model_calls": 0}, sort_keys=True))
        return 0
    if args.confirm_paid != "GPT-5.6-Luna":
        parser.error("execution requires --confirm-paid GPT-5.6-Luna")
    result = run_worker(plan)
    print(json.dumps({"complete": result["complete"],
                      "batch_wall_seconds": result["batch_wall_seconds"],
                      "candidates": [
                          {"candidate_id": item["candidate_id"],
                           "wall_seconds": item["wall_seconds"],
                           "verdict": (item["features"] or {}).get("verdict")}
                          for item in result["candidates"]]}))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
