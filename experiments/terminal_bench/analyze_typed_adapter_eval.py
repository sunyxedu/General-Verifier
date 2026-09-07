"""Fail-closed analysis for the ordinary Terminal-Bench typed-adapter eval.

The input run ledger is JSONL.  Each row describes one completed score.py
process and has this shape (paths are relative to the run ledger):

    {
      "schema": "gv-terminal-bench-eval-run/v1",
      "tier": "acceptance", "comparison": "full-e2e|adapter-isolation",
      "phase": "cold|warm|batch",
      "arm": "control|treatment", "run_id": "...",
      "source_commit": "...", "verifier_fingerprint": "...",
      "trials": ["task__trial", ...],
      "results_path": "...jsonl", "usage_paths": ["...usage.jsonl"],
      "started_at_unix_ns": 1, "finished_at_unix_ns": 2,
      "wall_seconds": 1.0, "wall_includes_contract_build": true,
      "usage_audit_complete": true, "exit_code": 0
    }

No reward appears in the source manifest.  It is read only from completed
result rows, after scoring.  A missing attempt, token count, contract seal, or
run timestamp makes the corresponding gate incomplete; it is never silently
treated as zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RUN_SCHEMA = "gv-terminal-bench-eval-run/v1"
ATTEMPT_RECORD = "codex_cli_attempt"
PHASES = ("cold", "warm", "batch")
ARMS = ("control", "treatment")
TOKEN_FIELDS = (
    "uncached_input_tokens", "cached_input_tokens", "output_tokens",
)
USAGE_TOKEN_FIELDS = ("input_tokens", *TOKEN_FIELDS)
ATTEMPT_STATUSES = frozenset({
    "succeeded", "provider_retry", "provider_exhausted", "nonzero_exit",
    "timeout", "telemetry_error", "output_error", "schema_error",
    "interrupted", "launch_error",
})
CONTRACT_FIELDS = (
    "obligation_contract_bundle_sha256",
    "task_contract_sha256",
    "compiler_fingerprint",
)
OUTCOME_FIELD = "final_obligation_outcomes_sha256"
SCOPE_FIELDS = ("task_contract_scope_sha256", "final_scope_outcomes_sha256")


class EvalError(ValueError):
    pass


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvalError(f"cannot read {path}: {exc}") from exc
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalError(f"invalid JSON at {path}:{lineno}: {exc}") from exc
        if not isinstance(value, dict):
            raise EvalError(f"non-object JSON at {path}:{lineno}")
        rows.append(value)
    return rows


def _resolve(base: Path, value: str) -> Path:
    path = Path(os.path.expanduser(value))
    return path if path.is_absolute() else base / path


def _candidate_index(manifest: dict[str, Any], tier: str) \
        -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    tiers = manifest.get("tiers") or {}
    spec = tiers.get(tier)
    if not isinstance(spec, dict):
        raise EvalError(f"unknown tier {tier!r}")
    wanted_tasks = set(spec.get("tasks") or [])
    wanted_trials = set(spec.get("trials") or [])
    tasks: dict[str, dict[str, Any]] = {}
    trials: dict[str, dict[str, Any]] = {}
    for task in manifest.get("tasks") or []:
        if not isinstance(task, dict) or not isinstance(task.get("task"), str):
            raise EvalError("manifest task is malformed")
        include_task = not wanted_tasks and not wanted_trials \
            or task["task"] in wanted_tasks
        included = []
        for candidate in task.get("candidates") or []:
            trial = candidate.get("trial") if isinstance(candidate, dict) else None
            if not isinstance(trial, str):
                raise EvalError(f"malformed candidate in {task['task']}")
            if include_task or trial in wanted_trials:
                if trial in trials:
                    raise EvalError(f"duplicate trial in manifest: {trial}")
                trials[trial] = {**candidate, "task": task["task"]}
                included.append(trial)
        if included:
            tasks[task["task"]] = {**task, "selected_trials": included}
    if not trials:
        raise EvalError(f"tier {tier!r} selects no trials")
    return tasks, trials


def _feature(record: dict[str, Any], name: str) -> Any:
    features = record.get("features") or {}
    if name in features:
        return features[name]
    execution = features.get("execution") or {}
    return execution.get(name)


def _topology(
    runs: list[dict[str, Any]], tasks: dict[str, dict[str, Any]],
    trials: dict[str, dict[str, Any]], phases: tuple[str, ...],
) -> list[str]:
    """Ensure cold/warm/batch labels describe the promised state regimes."""
    errors: list[str] = []
    expected_by_job: dict[str, set[str]] = defaultdict(set)
    for trial, candidate in trials.items():
        expected_by_job[tasks[candidate["task"]]["archive_job"]].add(trial)
    warm_seen: set[tuple[str, str]] = set()
    batch_seen: set[tuple[str, str]] = set()
    batch_intervals: list[tuple[int, int, str]] = []
    for run in runs:
        phase, arm = run.get("phase"), run.get("arm")
        selected = run.get("trials")
        if phase not in phases or arm not in ARMS or not isinstance(selected, list):
            continue
        selected_set = set(selected)
        if phase == "cold" and len(selected) != 1:
            errors.append(f"{run.get('run_id')}: cold must contain one trial")
        elif phase == "warm":
            selected_tasks = {trials[item]["task"] for item in selected
                              if item in trials}
            if len(selected_tasks) != 1:
                errors.append(f"{run.get('run_id')}: warm must contain one task")
                continue
            task = next(iter(selected_tasks))
            key = (arm, task)
            if key in warm_seen:
                errors.append(f"duplicate warm process: {arm}/{task}")
            warm_seen.add(key)
            expected = set(tasks[task]["selected_trials"])
            if selected_set != expected:
                errors.append(f"{run.get('run_id')}: incomplete warm task set")
            seed = tasks[task].get("warm_seed_trial")
            if seed != min(expected):
                errors.append(f"{task}: warm seed is not lexical first")
        elif phase == "batch":
            jobs = {tasks[trials[item]["task"]]["archive_job"]
                    for item in selected if item in trials}
            if len(jobs) != 1:
                errors.append(f"{run.get('run_id')}: batch must contain one job")
                continue
            job = next(iter(jobs))
            key = (arm, job)
            if key in batch_seen:
                errors.append(f"duplicate batch process: {arm}/{job}")
            batch_seen.add(key)
            if selected_set != expected_by_job[job]:
                errors.append(f"{run.get('run_id')}: incomplete batch job set")
            start = run.get("started_at_monotonic_ns")
            finish = run.get("finished_at_monotonic_ns")
            if isinstance(start, int) and isinstance(finish, int):
                batch_intervals.append((start, finish, str(run.get("run_id"))))
    ordered_batch = sorted(batch_intervals)
    for previous, current in zip(ordered_batch, ordered_batch[1:]):
        if current[0] < previous[1]:
            errors.append(
                f"batch processes overlap despite fixed global concurrency=1: "
                f"{previous[2]} and {current[2]}")
    expected_warm = ({(arm, task) for arm in ARMS for task in tasks}
                     if "warm" in phases else set())
    expected_batch = ({(arm, job) for arm in ARMS for job in expected_by_job}
                      if "batch" in phases else set())
    for item in sorted(expected_warm - warm_seen):
        errors.append(f"missing warm process: {item[0]}/{item[1]}")
    for item in sorted(expected_batch - batch_seen):
        errors.append(f"missing batch process: {item[0]}/{item[1]}")
    return errors


def _contract_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(_feature(record, field) for field in CONTRACT_FIELDS)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value)) and value >= 0


def _is_exact_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _ratio(treatment: float, control: float) -> float | None:
    if control <= 0:
        return None
    return treatment / control


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_metadata_errors(
    run: dict[str, Any], manifest: dict[str, Any], comparison: str,
    manifest_sha256: str,
) -> list[str]:
    rid = str(run.get("run_id"))
    errors: list[str] = []
    verifier = manifest["verifier"]
    expected = {
        "manifest_sha256": manifest_sha256,
        "model": verifier["model"],
        "seats": verifier["seats"],
        "effort": verifier["effort"],
        "service_tier": verifier["service_tier"],
        "mode": verifier["mode"],
        "browser_observer": verifier["browser_observer"],
    }
    phase, arm = run.get("phase"), run.get("arm")
    if phase in PHASES:
        expected.update({
            "workers": manifest["phases"][phase]["workers"],
            "fresh_scoring_root": True,
            "fresh_process_cache": True,
            "shared_toolchain": manifest["phases"][phase]["shared_toolchain"],
            "wall_includes_contract_build": True,
        })
    if arm in ARMS:
        expected.update(manifest["comparisons"][comparison][arm])
    for field, value in expected.items():
        if run.get(field) != value:
            errors.append(
                f"{rid}: {field}={run.get(field)!r}, expected {value!r}")
    if run.get("source_fingerprint_unchanged") is not True \
            or run.get("verifier_fingerprint_after") \
            != run.get("verifier_fingerprint"):
        errors.append(f"{rid}: verifier source changed during the run")
    if run.get("exit_code") != 0:
        errors.append(f"{rid}: score process did not exit successfully")
    if run.get("usage_audit_complete") is not True:
        errors.append(f"{rid}: wrapper usage audit is incomplete")
    if run.get("usage_audit_errors") not in (None, []):
        errors.append(f"{rid}: wrapper usage audit recorded errors")
    if run.get("result_audit_errors") not in (None, []):
        errors.append(f"{rid}: wrapper result audit recorded errors")
    start = run.get("started_at_monotonic_ns")
    finish = run.get("finished_at_monotonic_ns")
    wall = run.get("wall_seconds")
    if (not _is_exact_nonnegative_int(start)
            or not _is_exact_nonnegative_int(finish) or finish <= start
            or not _is_number(wall)):
        errors.append(f"{rid}: invalid monotonic/wall timing")
    else:
        measured = (finish - start) / 1e9
        tolerance = max(0.01, measured * 1e-6)
        if abs(float(wall) - measured) > tolerance:
            errors.append(f"{rid}: wall_seconds disagrees with monotonic bounds")
    return errors


def _private_path(
    run: dict[str, Any], base: Path, value: str, label: str,
) -> tuple[Path, str | None]:
    rid = str(run.get("run_id"))
    path = _resolve(base, value).resolve()
    run_root = (base / "runs" / rid).resolve()
    try:
        inside = os.path.commonpath((run_root, path)) == os.fspath(run_root)
    except ValueError:
        inside = False
    return path, None if inside else f"{rid}: {label} escapes private run root"


def _quality(
    results: dict[tuple[str, str, str], dict[str, Any]],
    trials: dict[str, dict[str, Any]],
    comparison: str, phases: tuple[str, ...],
) -> dict[str, Any]:
    errors: list[str] = []
    mismatches: list[dict[str, Any]] = []
    rewards: dict[str, float] = {}
    for phase in phases:
        for trial, expected in trials.items():
            pair = []
            for arm in ARMS:
                row = results.get((phase, arm, trial))
                if row is None:
                    errors.append(f"missing result: {phase}/{arm}/{trial}")
                pair.append(row)
            if None in pair:
                continue
            control, treatment = pair
            for arm, row in zip(ARMS, pair):
                features = row.get("features") or {}
                if features.get("unscoreable"):
                    errors.append(f"unscoreable: {phase}/{arm}/{trial}")
                seal = features.get("immutable_input_seal")
                if seal != expected.get("immutable_input_seal"):
                    errors.append(f"wrong immutable input seal: {phase}/{arm}/{trial}")
                if comparison == "adapter-isolation":
                    missing = [field for field, value in zip(
                        CONTRACT_FIELDS, _contract_signature(row)) if not value]
                    if missing:
                        errors.append(
                            f"missing frozen contract fields {missing}: "
                            f"{phase}/{arm}/{trial}")
                    if not _feature(row, OUTCOME_FIELD):
                        errors.append(
                            f"missing {OUTCOME_FIELD}: {phase}/{arm}/{trial}")
                else:
                    missing = [field for field in SCOPE_FIELDS
                               if not _feature(row, field)]
                    if missing:
                        errors.append(
                            f"missing scope audit fields {missing}: "
                            f"{phase}/{arm}/{trial}")
            if comparison == "adapter-isolation":
                if _contract_signature(control) != _contract_signature(treatment):
                    errors.append(
                        f"contract differs between arms: {phase}/{trial}")
                if _feature(control, OUTCOME_FIELD) != _feature(treatment, OUTCOME_FIELD):
                    mismatches.append({
                        "phase": phase, "trial": trial,
                        "fields": [OUTCOME_FIELD],
                        "control_verdict": (
                            control.get("features") or {}).get("verdict"),
                        "treatment_verdict": (
                            treatment.get("features") or {}).get("verdict"),
                    })
            else:
                differing_scope = [field for field in SCOPE_FIELDS
                                   if _feature(control, field)
                                   != _feature(treatment, field)]
                if differing_scope:
                    mismatches.append({
                        "phase": phase, "trial": trial,
                        "fields": differing_scope,
                        "control_verdict": (
                            control.get("features") or {}).get("verdict"),
                        "treatment_verdict": (
                            treatment.get("features") or {}).get("verdict"),
                    })
            comparable = {
                "feature_schema_version": lambda row: (
                    row.get("features") or {}).get("feature_schema_version"),
                "verdict": lambda row: (row.get("features") or {}).get("verdict"),
                "verdict_strength": lambda row: (
                    row.get("features") or {}).get("verdict_strength"),
                "final_obligations": lambda row: (
                    row.get("features") or {}).get("final_obligations"),
                "final_obligations_total": lambda row: (
                    row.get("features") or {}).get("final_obligations_total"),
                "final_contrary_signals": lambda row: (
                    row.get("features") or {}).get("final_contrary_signals", 0),
            }
            differing = [name for name, getter in comparable.items()
                         if getter(control) != getter(treatment)]
            if differing:
                mismatches.append({
                    "phase": phase, "trial": trial, "fields": differing,
                    "control_verdict": comparable["verdict"](control),
                    "treatment_verdict": comparable["verdict"](treatment),
                })
            reward_values = {row.get("reward") for row in pair}
            if len(reward_values) != 1 or not all(
                    _is_number(value) for value in reward_values):
                errors.append(f"missing/inconsistent post-run reward: {phase}/{trial}")
            else:
                reward = float(next(iter(reward_values)))
                if trial in rewards and rewards[trial] != reward:
                    errors.append(f"reward changes across phases: {trial}")
                rewards[trial] = reward
    # Zero observed disagreements is a suite gate, not a universal proof.
    n = len(trials)
    upper_95 = 1.0 - math.pow(0.05, 1.0 / n) if n else 1.0
    status = "pass" if not errors and not mismatches else "fail"
    return {
        "status": status,
        "errors": errors,
        "mismatches": mismatches,
        "unique_candidates": n,
        "reward_counts": {
            "positive": sum(value > 0 for value in rewards.values()),
            "zero": sum(value == 0 for value in rewards.values()),
        },
        "zero_mismatch_95pct_disagreement_upper_bound": upper_95,
        "scope_note": "Passing applies only to these unique archived candidates.",
    }


def _usage_for_run(row: dict[str, Any], base: Path) \
        -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    paths = row.get("usage_paths")
    if not isinstance(paths, list) or not paths:
        return [], [f"{row.get('run_id')}: usage_paths is empty"]
    attempts: list[dict[str, Any]] = []
    for value in paths:
        if not isinstance(value, str):
            errors.append(f"{row.get('run_id')}: non-string usage path")
            continue
        path, error = _private_path(row, base, value, "usage path")
        if error:
            errors.append(error)
            continue
        attempts.extend(_jsonl(path))
    return attempts, errors


def _cost(
    runs: list[dict[str, Any]], trials: dict[str, dict[str, Any]], base: Path,
    factor: float, verifier: dict[str, Any], phases: tuple[str, ...],
) -> dict[str, Any]:
    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"started_attempts": 0, **{field: 0 for field in TOKEN_FIELDS}})
    errors: list[str] = []
    seen_attempts: set[str] = set()
    seen_usage_paths: set[Path] = set()
    call_owners: dict[str, str] = {}
    for run in runs:
        phase, arm = run.get("phase"), run.get("arm")
        key = (phase, arm)
        if phase not in phases or arm not in ARMS:
            continue
        if run.get("usage_audit_complete") is not True:
            errors.append(f"{run.get('run_id')}: usage audit is not complete")
        usage_paths = run.get("usage_paths") or []
        for value in usage_paths:
            if not isinstance(value, str):
                continue
            path = _resolve(base, value).resolve()
            if path in seen_usage_paths:
                errors.append(f"{run.get('run_id')}: usage path is reused")
            seen_usage_paths.add(path)
        attempts, read_errors = _usage_for_run(run, base)
        errors.extend(read_errors)
        # Every usage path is process-private.  This deliberately charges the
        # task-only contract build and any failed/retried attempt whose trial
        # context could not be populated, instead of filtering such rows out.
        selected = attempts
        if not selected:
            errors.append(f"{run.get('run_id')}: no enumerated attempts")
        call_indexes: dict[str, list[int]] = defaultdict(list)
        for attempt in selected:
            if attempt.get("record_type") != ATTEMPT_RECORD:
                errors.append(f"{run.get('run_id')}: legacy/non-attempt usage row")
                continue
            if attempt.get("schema_version") != 1 \
                    or attempt.get("attempt_schema_version") != 1:
                errors.append(f"{run.get('run_id')}: wrong attempt schema")
            if (not isinstance(attempt.get("role"), str)
                    or not attempt.get("role")
                    or attempt.get("model") != verifier["model"]
                    or attempt.get("effort") != verifier["effort"]
                    or attempt.get("service_tier") != verifier["service_tier"]):
                errors.append(
                    f"{run.get('run_id')}: attempt runtime identity mismatch")
            if (attempt.get("job") != run.get("job")
                    or attempt.get("trial") not in set(run.get("trials") or [])
                    or attempt.get("rep") != run.get("run_id")):
                errors.append(
                    f"{run.get('run_id')}: attempt context mismatch")
            prompt_hash = attempt.get("prompt_sha256")
            if (not isinstance(prompt_hash, str) or len(prompt_hash) != 64
                    or any(char not in "0123456789abcdef"
                           for char in prompt_hash)):
                errors.append(f"{run.get('run_id')}: invalid prompt hash")
            state_dir = attempt.get("state_dir")
            scoring_root = (
                base / "runs" / str(run.get("run_id")) / "scoring").resolve()
            try:
                state_inside = (isinstance(state_dir, str)
                    and os.path.commonpath((scoring_root, Path(state_dir).resolve()))
                    == os.fspath(scoring_root))
            except ValueError:
                state_inside = False
            if not state_inside:
                errors.append(
                    f"{run.get('run_id')}: attempt state escapes scoring root")
            missing_identity = [field for field in (
                "call_id", "attempt_id", "attempt_index", "attempt_status",
                "usage_status",
            ) if attempt.get(field) is None]
            if missing_identity:
                errors.append(
                    f"{run.get('run_id')}: attempt lacks {missing_identity}")
                continue
            if attempt.get("attempt_status") not in ATTEMPT_STATUSES:
                errors.append(
                    f"{run.get('run_id')}: invalid attempt_status "
                    f"{attempt.get('attempt_status')!r}")
            call_id = attempt["call_id"]
            index = attempt["attempt_index"]
            if not isinstance(call_id, str) or not call_id \
                    or not isinstance(index, int) or isinstance(index, bool) \
                    or index < 1:
                errors.append(f"{run.get('run_id')}: invalid call/index identity")
            else:
                call_indexes[call_id].append(index)
                owner = call_owners.setdefault(call_id, str(run.get("run_id")))
                if owner != str(run.get("run_id")):
                    errors.append(f"call_id reused across processes: {call_id}")
            attempt_id = attempt.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id:
                errors.append(f"{run.get('run_id')}: missing attempt_id")
                continue
            if attempt_id in seen_attempts:
                errors.append(f"duplicate charged attempt: {attempt_id}")
                continue
            seen_attempts.add(attempt_id)
            started = attempt.get("started")
            if not isinstance(started, bool):
                errors.append(f"{run.get('run_id')}: started is not boolean")
                continue
            started_ns = attempt.get("started_at_unix_ns")
            finished_ns = attempt.get("finished_at_unix_ns")
            duration_ns = attempt.get("duration_ns")
            duration_seconds = attempt.get("duration_seconds")
            if (not _is_exact_nonnegative_int(started_ns)
                    or not _is_exact_nonnegative_int(finished_ns)
                    or not _is_exact_nonnegative_int(duration_ns)
                    or finished_ns < started_ns
                    or not _is_number(duration_seconds)
                    or abs(float(duration_seconds) - duration_ns / 1e9)
                    > max(1e-9, duration_ns / 1e15)):
                errors.append(f"{run.get('run_id')}: invalid attempt timing")
            if not started:
                # A launch error consumes wall time but no provider tokens.
                if (attempt.get("attempt_status") != "launch_error"
                        or attempt.get("usage_complete") is not False
                        or any(attempt.get(field) is not None
                               for field in USAGE_TOKEN_FIELDS)):
                    errors.append(
                        f"{run.get('run_id')}: invalid non-started attempt")
                continue
            totals[key]["started_attempts"] += 1
            nulls = [field for field in USAGE_TOKEN_FIELDS
                     if not _is_exact_nonnegative_int(attempt.get(field))]
            if (attempt.get("usage_complete") is not True
                    or attempt.get("usage_status") != "complete" or nulls):
                errors.append(
                    f"incomplete token usage: {phase}/{arm}/{attempt_id}; "
                    f"status={attempt.get('attempt_status')}; missing={nulls}")
                continue
            if (attempt["cached_input_tokens"] > attempt["input_tokens"]
                    or attempt["uncached_input_tokens"]
                    != attempt["input_tokens"]
                    - attempt["cached_input_tokens"]):
                errors.append(
                    f"inconsistent input tokens: {phase}/{arm}/{attempt_id}")
                continue
            for field in TOKEN_FIELDS:
                totals[key][field] += attempt[field]
        for call_id, indexes in call_indexes.items():
            if sorted(indexes) != list(range(1, len(indexes) + 1)):
                errors.append(
                    f"{run.get('run_id')}: non-contiguous attempt_index for "
                    f"call {call_id}")
    ratios: dict[str, dict[str, float | None]] = {}
    failures: list[str] = []
    limit = 1.0 / factor
    for phase in phases:
        control, treatment = totals[(phase, "control")], totals[(phase, "treatment")]
        ratios[phase] = {}
        for field in ("started_attempts", *TOKEN_FIELDS):
            value = _ratio(treatment[field], control[field])
            ratios[phase][field] = value
            if value is None or value > limit:
                failures.append(f"{phase}/{field}: ratio={value!r}, limit={limit}")
    status = "incomplete" if errors else ("fail" if failures else "pass")
    return {
        "status": status,
        "errors": errors,
        "failures": failures,
        "totals": {f"{phase}/{arm}": values
                   for (phase, arm), values in sorted(totals.items())},
        "treatment_over_control": ratios,
        "required_max_ratio": limit,
    }


def _latency(
    runs: list[dict[str, Any]], trials: dict[str, dict[str, Any]], factor: float,
    phases: tuple[str, ...],
) -> dict[str, Any]:
    errors: list[str] = []
    totals: dict[tuple[str, str], float] = defaultdict(float)
    cold: dict[tuple[str, str], float] = {}
    batch_bounds: dict[str, list[int]] = defaultdict(list)
    for run in runs:
        phase, arm = run.get("phase"), run.get("arm")
        if phase not in phases or arm not in ARMS:
            continue
        rid = run.get("run_id")
        wall = run.get("wall_seconds")
        if (run.get("exit_code") != 0 or not _is_number(wall)
                or run.get("wall_includes_contract_build") is not True):
            errors.append(f"{rid}: invalid/incomplete end-to-end wall record")
            continue
        totals[(phase, arm)] += float(wall)
        if phase == "cold":
            selected = [trial for trial in run.get("trials") or [] if trial in trials]
            if len(selected) != 1:
                errors.append(f"{rid}: cold run must contain exactly one candidate")
            else:
                key = (arm, selected[0])
                if key in cold:
                    errors.append(f"duplicate cold wall record: {arm}/{selected[0]}")
                cold[key] = float(wall)
        if phase == "batch":
            start = run.get("started_at_monotonic_ns")
            finish = run.get("finished_at_monotonic_ns")
            if not isinstance(start, int) or not isinstance(finish, int) or finish <= start:
                errors.append(f"{rid}: batch run lacks valid monotonic bounds")
            else:
                batch_bounds[arm].extend((start, finish))
    # Batch's primary metric is makespan, not the sum of concurrent model time.
    if "batch" in phases:
        for arm in ARMS:
            bounds = batch_bounds[arm]
            if bounds:
                starts, finishes = bounds[0::2], bounds[1::2]
                totals[("batch", arm)] = (max(finishes) - min(starts)) / 1e9
            else:
                errors.append(f"batch/{arm}: no valid process bounds")
    ratios: dict[str, float | None] = {}
    failures: list[str] = []
    limit = 1.0 / factor
    for phase in phases:
        value = _ratio(totals[(phase, "treatment")], totals[(phase, "control")])
        ratios[phase] = value
        if value is None or value > limit:
            failures.append(f"{phase}: ratio={value!r}, limit={limit}")
    regressions = []
    if "cold" in phases:
        for trial in trials:
            control = cold.get(("control", trial))
            treatment = cold.get(("treatment", trial))
            if control is None or treatment is None:
                errors.append(f"missing paired cold wall: {trial}")
            elif treatment > control:
                regressions.append({"trial": trial, "control": control,
                                    "treatment": treatment})
    if regressions:
        failures.append("one or more candidates regress in cold wall latency")
    status = "incomplete" if errors else ("fail" if failures else "pass")
    return {
        "status": status,
        "errors": errors,
        "failures": failures,
        "wall_seconds": {f"{phase}/{arm}": value
                         for (phase, arm), value in sorted(totals.items())},
        "treatment_over_control": ratios,
        "cold_regressions": regressions,
        "required_max_ratio": limit,
    }


def analyze(manifest_path: Path, runs_path: Path, tier: str,
            comparison: str, phases: tuple[str, ...] = PHASES) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot read manifest: {exc}") from exc
    if manifest.get("schema") != "gv-terminal-bench-typed-adapter-eval/v1":
        raise EvalError("wrong eval manifest schema")
    tasks, trials = _candidate_index(manifest, tier)
    comparisons = manifest.get("comparisons") or {}
    if comparison not in comparisons:
        raise EvalError(f"unknown comparison {comparison!r}")
    runs = [row for row in _jsonl(runs_path)
            if row.get("tier") == tier and row.get("comparison") == comparison
            and row.get("phase") in phases]
    errors: list[str] = []
    manifest_digest = _manifest_sha256(manifest_path)
    source_commits = {row.get("source_commit") for row in runs}
    fingerprints = {row.get("verifier_fingerprint") for row in runs}
    run_ids: set[str] = set()
    for row in runs:
        if row.get("schema") != RUN_SCHEMA:
            errors.append(f"{row.get('run_id')}: wrong run schema")
        if row.get("arm") not in ARMS or row.get("phase") not in PHASES:
            errors.append(f"{row.get('run_id')}: invalid phase/arm")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in run_ids:
            errors.append(f"{run_id}: missing/duplicate run_id")
        else:
            run_ids.add(run_id)
        selected = row.get("trials")
        if not isinstance(selected, list) or len(selected) != len(set(selected)):
            errors.append(f"{row.get('run_id')}: trials are absent or repeated")
        elif any(trial not in trials for trial in selected):
            errors.append(f"{row.get('run_id')}: trial outside tier")
        else:
            expected_tasks = sorted({trials[trial]["task"] for trial in selected})
            expected_jobs = {
                tasks[trials[trial]["task"]]["archive_job"]
                for trial in selected
            }
            if run.get("tasks") != expected_tasks:
                errors.append(f"{row.get('run_id')}: task list mismatch")
            if len(expected_jobs) != 1 or run.get("job") not in expected_jobs:
                errors.append(f"{row.get('run_id')}: archive job mismatch")
        errors.extend(_run_metadata_errors(
            row, manifest, comparison, manifest_digest))
    if len(source_commits) != 1 or None in source_commits:
        errors.append("arms/phases do not share one source commit")
    if len(fingerprints) != 1 or None in fingerprints:
        errors.append("arms/phases do not share one verifier fingerprint")
    errors.extend(_topology(runs, tasks, trials, phases))

    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    result_paths: set[Path] = set()
    configs: dict[str, set[str]] = defaultdict(set)
    pipeline_fingerprints: set[str] = set()
    base = runs_path.parent
    for run in runs:
        value = run.get("results_path")
        if not isinstance(value, str):
            errors.append(f"{run.get('run_id')}: missing results_path")
            continue
        results_path, path_error = _private_path(
            run, base, value, "results path")
        if path_error:
            errors.append(path_error)
            continue
        if results_path in result_paths:
            errors.append(f"{run.get('run_id')}: results path is reused")
        result_paths.add(results_path)
        result_rows = _jsonl(results_path)
        expected_trials = set(run.get("trials") or [])
        actual_trials = [result.get("trial") for result in result_rows]
        if len(result_rows) != len(expected_trials) \
                or set(actual_trials) != expected_trials \
                or len(actual_trials) != len(set(actual_trials)):
            errors.append(
                f"{run.get('run_id')}: results are not the exact run trial set")
        arm_cfg = manifest["comparisons"][comparison].get(run.get("arm"), {})
        expected_typed = arm_cfg.get("GV_TYPED_ADAPTERS") == "1"
        expected_contract = arm_cfg.get("GV_CONTRACT_CACHE") == "1"
        for result in result_rows:
            trial = result.get("trial")
            if trial not in expected_trials:
                continue
            task = trials[trial]["task"]
            exact = {
                "job": run.get("job"), "task": task,
                "rep": run.get("run_id"),
                "model": manifest["verifier"]["model"],
                "seats": manifest["verifier"]["seats"],
                "effort": manifest["verifier"]["effort"],
                "service_tier": manifest["verifier"]["service_tier"],
                "typed_adapters": expected_typed,
                "contract_cache": expected_contract,
                "browser_observer": None,
            }
            features = result.get("features") or {}
            for field, expected in exact.items():
                if result.get(field) != expected:
                    errors.append(
                        f"{run.get('run_id')}/{trial}: result {field} mismatch")
                if field not in {"job", "task", "rep"} \
                        and features.get(field) != expected:
                    errors.append(
                        f"{run.get('run_id')}/{trial}: feature {field} mismatch")
            if features.get("immutable_input_boundary") != "adapter-sealed":
                errors.append(
                    f"{run.get('run_id')}/{trial}: input is not adapter-sealed")
            config = result.get("config")
            if (not isinstance(config, str) or not config
                    or features.get("config") != config):
                errors.append(
                    f"{run.get('run_id')}/{trial}: config identity mismatch")
            else:
                configs[str(run.get("arm"))].add(config)
            pipeline = features.get("fingerprint")
            if not isinstance(pipeline, str) or not pipeline:
                errors.append(
                    f"{run.get('run_id')}/{trial}: missing pipeline fingerprint")
            else:
                pipeline_fingerprints.add(pipeline)
            key = (run.get("phase"), run.get("arm"), trial)
            if key in results:
                errors.append(f"duplicate result: {'/'.join(key)}")
            results[key] = result

    for arm in ARMS:
        if len(configs[arm]) != 1:
            errors.append(f"{arm}: results do not share one config identity")
    if len(pipeline_fingerprints) != 1:
        errors.append("results do not share one pipeline fingerprint")

    factor = float(manifest["acceptance"]["cost"]["required_reduction_factor"])
    quality = _quality(results, trials, comparison, phases)
    cost = _cost(runs, trials, base, factor, manifest["verifier"], phases)
    latency = _latency(runs, trials, factor, phases)
    eligible = bool((manifest.get("tiers") or {}).get(tier, {}).get(
        "acceptance_eligible")) and bool(
            comparisons[comparison].get("acceptance_eligible"))
    failed = (bool(errors) or quality["status"] != "pass"
              or cost["status"] != "pass" or latency["status"] != "pass")
    if not eligible or phases != PHASES:
        overall = "preliminary_fail" if failed else "preliminary_pass"
    elif failed:
        overall = "fail"
    else:
        overall = "pass"
    return {
        "schema": "gv-terminal-bench-typed-adapter-eval-analysis/v1",
        "tier": tier,
        "comparison": comparison,
        "phases": list(phases),
        "acceptance_eligible": eligible,
        "overall": overall,
        "audit_errors": errors,
        "tasks": sorted(tasks),
        "unique_candidates": len(trials),
        "quality_gate": quality,
        "cost_gate": cost,
        "latency_gate": latency,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--tier", choices=("canary", "core", "acceptance"),
                        default="acceptance")
    parser.add_argument("--comparison",
                        choices=("full-e2e", "adapter-isolation"),
                        default="full-e2e")
    parser.add_argument("--phase", choices=(*PHASES, "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        phases = PHASES if args.phase == "all" else (args.phase,)
        report = analyze(
            args.manifest, args.runs, args.tier, args.comparison, phases)
    except EvalError as exc:
        print(f"eval analysis error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["overall"] in ("pass", "preliminary_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
