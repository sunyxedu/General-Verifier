"""Plan or execute the pinned typed-adapter Terminal-Bench evaluation.

Planning is the default and writes nothing. Paid model calls require both
``--execute`` and ``--confirm-paid GPT-5.6-Luna``. Full-E2E runs each arm in a
fresh process. Adapter-isolation execution is intentionally blocked until the
shared sealed-contract charge/export hook described in TYPED_ADAPTER_EVAL.md
is available; separate processes would not share the process-local cache and
would not be a valid experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = HERE.parent.parent
DEFAULT_MANIFEST = HERE / "typed_adapter_eval_manifest.json"
DEFAULT_JOBS_ROOT = HERE / "results" / "jobs"
DEFAULT_OUTPUT_ROOT = HERE / "results" / "gv-typed-adapter-eval"
RUN_SCHEMA = "gv-terminal-bench-eval-run/v1"
ATTEMPT_RECORD = "codex_cli_attempt"
PHASES = ("cold", "warm", "batch")
TOKEN_FIELDS = (
    "input_tokens", "uncached_input_tokens", "cached_input_tokens",
    "output_tokens",
)
ATTEMPT_STATUSES = frozenset({
    "succeeded", "provider_retry", "provider_exhausted", "nonzero_exit",
    "timeout", "telemetry_error", "output_error", "schema_error",
    "interrupted", "launch_error",
})
_BROWSER_ENV = (
    "GV_BROWSER_OBSERVER", "GV_BROWSER_OBSERVER_SHA256",
    "GV_BROWSER_EXECUTABLE", "GV_BROWSER_EXECUTABLE_SHA256",
)
_CLEARED_ENV = (
    *_BROWSER_ENV, "GV_SHARED_DIR", "GV_OFFLINE_RUNTIME_READ_ROOTS",
)


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class Group:
    key: str
    job: str
    tasks: tuple[str, ...]
    trials: tuple[str, ...]


def _manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read manifest {path}: {exc}") from exc
    if value.get("schema") != "gv-terminal-bench-typed-adapter-eval/v1":
        raise PlanError("wrong eval manifest schema")
    verifier = value.get("verifier")
    if not isinstance(verifier, dict) or type(
            verifier.get("compact_inspection", False)) is not bool:
        raise PlanError("verifier.compact_inspection must be a boolean")
    for comparison in (value.get("comparisons") or {}):
        _validate_compact_comparison(value, comparison)
    return value


def _arm_profile(
    manifest: dict[str, Any], comparison: str, arm: str,
) -> tuple[bool, dict[str, str]]:
    """Resolve one explicit CLI setting and the existing GV environment."""
    try:
        spec = manifest["comparisons"][comparison][arm]
    except (KeyError, TypeError) as error:
        raise PlanError(f"missing {comparison}/{arm} arm configuration") from error
    if not isinstance(spec, dict):
        raise PlanError(f"{comparison}/{arm} arm configuration must be an object")
    compact = spec.get(
        "compact_inspection",
        manifest.get("verifier", {}).get("compact_inspection", False),
    )
    if type(compact) is not bool:
        raise PlanError(f"{comparison}/{arm} compact_inspection must be boolean")
    unknown = set(spec) - {"compact_inspection"} - {
        key for key in spec if isinstance(key, str) and key.startswith("GV_")
    }
    if unknown:
        raise PlanError(
            f"{comparison}/{arm} has unsupported arm fields: {sorted(unknown)}")
    environment = {
        key: str(value) for key, value in spec.items() if key.startswith("GV_")
    }
    return compact, environment


def _validate_compact_comparison(
    manifest: dict[str, Any], comparison: str,
) -> None:
    """A compact A/B may not silently change adapters or cache policy too."""
    spec = (manifest.get("comparisons") or {}).get(comparison)
    if not isinstance(spec, dict) or not all(
            isinstance(spec.get(arm), dict) for arm in ("control", "treatment")):
        return
    control, control_env = _arm_profile(manifest, comparison, "control")
    treatment, treatment_env = _arm_profile(manifest, comparison, "treatment")
    if control != treatment and control_env != treatment_env:
        raise PlanError(
            f"{comparison} changes compact_inspection together with GV settings")


def _selected(manifest: dict[str, Any], tier: str) \
        -> dict[str, dict[str, Any]]:
    spec = (manifest.get("tiers") or {}).get(tier)
    if not isinstance(spec, dict):
        raise PlanError(f"unknown tier {tier!r}")
    wanted_tasks = set(spec.get("tasks") or [])
    wanted_trials = set(spec.get("trials") or [])
    selected: dict[str, dict[str, Any]] = {}
    for task in manifest.get("tasks") or []:
        include_task = task.get("task") in wanted_tasks
        for candidate in task.get("candidates") or []:
            trial = candidate.get("trial")
            if include_task or trial in wanted_trials:
                selected[trial] = {**candidate, **{
                    "task": task["task"], "job": task["archive_job"]}}
    if not selected:
        raise PlanError(f"tier {tier!r} selects no candidates")
    return selected


def _groups(phase: str, selected: dict[str, dict[str, Any]]) -> list[Group]:
    grouped: dict[tuple[str, ...], list[str]] = defaultdict(list)
    if phase == "cold":
        for trial, item in selected.items():
            grouped[(item["job"], item["task"], trial)].append(trial)
    elif phase == "warm":
        for trial, item in selected.items():
            grouped[(item["job"], item["task"])].append(trial)
    elif phase == "batch":
        for trial, item in selected.items():
            grouped[(item["job"],)].append(trial)
    else:
        raise PlanError(f"unknown phase {phase!r}")
    result = []
    for identity, trials in sorted(grouped.items()):
        job = identity[0]
        tasks = tuple(sorted({selected[trial]["task"] for trial in trials}))
        label = identity[-1].split("__", 1)[-1] if phase == "cold" \
            else "-".join(tasks)
        result.append(Group(
            key=f"{phase}-{job}-{label}", job=job, tasks=tasks,
            trials=tuple(sorted(trials))))
    return result


def _source_fingerprint(source_root: Path) -> str:
    paths = [
        source_root / "pyproject.toml",
        source_root / "uv.lock",
    ]
    # Include the runner, preparation/replay and accounting code too. Do not
    # recurse into results/: archived frozen sources are data, not this run.
    paths.extend(sorted((source_root / "experiments" / "terminal_bench").glob("*.py")))
    paths.extend(sorted(
        (source_root / "experiments" / "common").rglob("*.py")))
    paths.extend(sorted((source_root / "verifier").rglob("*.py")))
    paths.extend(sorted((source_root / "protocol").rglob("*")))
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        rel = path.relative_to(source_root).as_posix().encode()
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_execute_output_root(output: Path) -> None:
    """Keep offline states out of host temp trees readable under :minimal."""
    resolved = output.resolve()
    unsafe_roots = {
        Path(tempfile.gettempdir()).resolve(),
        Path("/private/tmp").resolve(),
    }
    for root in unsafe_roots:
        try:
            inside = os.path.commonpath((root, resolved)) == os.fspath(root)
        except ValueError:
            inside = False
        if inside:
            raise PlanError(
                "paid output root must not be inside a host temporary tree; "
                "use an independent directory under ~/Documents")


def _validate_source_root(source_root: Path) -> None:
    required = (
        source_root / "verifier" / "offline.py",
        source_root / "protocol" / "solver-protocol.md",
        source_root / "experiments" / "terminal_bench" / "score.py",
        source_root / "experiments" / "terminal_bench" / "harborjobs.py",
        source_root / "experiments" / "common" / "scoring.py",
        source_root / "pyproject.toml",
        source_root / "uv.lock",
    )
    missing = [os.fspath(path) for path in required if not path.is_file()]
    if missing:
        raise PlanError(f"source root is incomplete; missing {missing}")


def _commit(source_root: Path, override: str) -> str:
    if override:
        if not (7 <= len(override) <= 40
                and all(char in "0123456789abcdef" for char in override)):
            raise PlanError("--source-commit must be a 7-40 digit lowercase hex SHA")
        return override
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source_root, check=True,
            text=True, capture_output=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise PlanError(f"cannot resolve source commit: {exc}") from exc


def _command(
    manifest: dict[str, Any], source_root: Path, group: Group, phase: str,
    scores: Path, rep: str, comparison: str, arm: str,
) -> list[str]:
    cfg = manifest["verifier"]
    compact, _environment = _arm_profile(manifest, comparison, arm)
    command = [
        sys.executable,
        os.fspath(source_root / "experiments" / "terminal_bench" / "score.py"),
        "--job", group.job,
        "--model", cfg["model"], "--seats", cfg["seats"],
        "--effort", cfg["effort"], "--service-tier", cfg["service_tier"],
        "--compact-inspection", "1" if compact else "0",
        "--workers", str(manifest["phases"][phase]["workers"]),
        "--parallel-obligations", str(cfg["parallel_obligations"]),
        "--parallel-claims", str(cfg["parallel_claims"]),
        "--rep", rep, "--scores-path", os.fspath(scores),
        "--task", *group.tasks, "--trial", *group.trials,
    ]
    if phase == "cold":
        command.append("--no-shared-toolchain")
    return command


def _prepare_jobs(
    source_root: Path, jobs_root: Path, selected: dict[str, dict[str, Any]],
) -> None:
    """Expose only selected immutable archives at score.py's fixed job path."""
    if not jobs_root.is_absolute():
        raise PlanError("--jobs-root must resolve to an absolute path")
    target_root = source_root / "experiments" / "terminal_bench" / "results" / "jobs"
    target_root.mkdir(parents=True, exist_ok=True)
    for job in sorted({item["job"] for item in selected.values()}):
        source = jobs_root / job
        if not source.is_dir():
            raise PlanError(f"archived job is absent: {source}")
        target = target_root / job
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise PlanError(
                    f"source archive job link has wrong target: {target}")
            continue
        target.symlink_to(source.resolve(), target_is_directory=True)


def _arm_order(group: Group, arms: tuple[str, ...]) -> tuple[str, ...]:
    """Use a precommitted group hash to balance first-arm time drift."""
    if len(arms) != 2:
        return arms
    digest = hashlib.sha256(
        f"gv-a-b-order/v1\0{group.key}".encode("utf-8")).digest()
    return arms if digest[0] % 2 == 0 else tuple(reversed(arms))


def _schedule(
    phase: str, groups: list[Group], arms: tuple[str, ...],
) -> list[tuple[str, Group]]:
    # Batch makespan is measured from an arm's first job start through its
    # last job finish. Keep each arm contiguous so running the other arm is
    # never charged as latency, while preserving the same sorted job order.
    if phase == "batch" and len(arms) == 2:
        ordered_arms = _arm_order(
            Group("batch-global-arm-order", "", (), ()), arms)
        return [(arm, group) for arm in ordered_arms for group in groups]
    return [(arm, group) for group in groups
            for arm in _arm_order(group, arms)]


def _attempt_audit(
    path: Path, *, compact_inspection: bool | None = None,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        rows = [json.loads(line) for line in path.read_text(
            encoding="utf-8").split("\n") if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"cannot audit usage ledger: {exc}"]
    if not rows:
        return False, ["usage ledger has no attempts"]
    ids: set[str] = set()
    by_call: dict[str, list[int]] = defaultdict(list)
    runtime_identities: set[str] = set()
    for row in rows:
        if type(row) is not dict:
            errors.append("usage ledger contains a non-object row")
            continue
        if row.get("record_type") != ATTEMPT_RECORD:
            errors.append("usage ledger contains a legacy/non-attempt row")
            continue
        if row.get("schema_version") != 1 \
                or row.get("attempt_schema_version") != 1:
            errors.append("usage ledger has the wrong attempt schema")
        if compact_inspection is not None:
            if row.get("compact_inspection") is not compact_inspection:
                errors.append(
                    "usage ledger compact_inspection differs from the arm")
            runtime_identity = row.get("runtime_instruction_identity")
            if (not isinstance(runtime_identity, str)
                    or len(runtime_identity) != 64
                    or any(char not in "0123456789abcdef"
                           for char in runtime_identity)):
                errors.append("usage ledger has invalid runtime instruction identity")
            else:
                runtime_identities.add(runtime_identity)
        attempt = row.get("attempt_id")
        call = row.get("call_id")
        index = row.get("attempt_index")
        if not isinstance(attempt, str) or not attempt or attempt in ids:
            errors.append("usage ledger has a missing/duplicate attempt_id")
        else:
            ids.add(attempt)
        if not isinstance(call, str) or not call or not isinstance(index, int) \
                or isinstance(index, bool) or index < 1:
            errors.append("usage ledger has an invalid call/index identity")
        else:
            by_call[call].append(index)
        started = row.get("started")
        status = row.get("attempt_status")
        if not isinstance(started, bool) or status not in ATTEMPT_STATUSES:
            errors.append("usage ledger has invalid started/attempt_status")
            continue
        if not started:
            if status != "launch_error" or row.get("usage_complete") is not False:
                errors.append("non-started attempt must be an incomplete launch_error")
            if any(row.get(field) is not None for field in TOKEN_FIELDS):
                errors.append("non-started attempt unexpectedly has token usage")
            continue
        token_values = {field: row.get(field) for field in TOKEN_FIELDS}
        if (row.get("usage_complete") is not True
                or row.get("usage_status") != "complete"
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 0 for value in token_values.values())):
            errors.append("started attempt has incomplete/invalid token usage")
            continue
        if (token_values["cached_input_tokens"]
                > token_values["input_tokens"]
                or token_values["uncached_input_tokens"]
                != token_values["input_tokens"]
                - token_values["cached_input_tokens"]):
            errors.append("started attempt has inconsistent input token usage")
    for call, indexes in by_call.items():
        if sorted(indexes) != list(range(1, len(indexes) + 1)):
            errors.append(f"attempt indexes are not contiguous for call {call}")
    if compact_inspection is not None and len(runtime_identities) != 1:
        errors.append("usage ledger mixes runtime instruction identities")
    return not errors, errors


def _attempt_runtime_identity(path: Path) -> str | None:
    """Return the one observed runtime identity, never derive it from a flag."""
    try:
        rows = [json.loads(line) for line in path.read_text(
            encoding="utf-8").split("\n") if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    identities: set[str] = set()
    for row in rows:
        if type(row) is not dict:
            return None
        if row.get("record_type") != ATTEMPT_RECORD:
            continue
        identity = row.get("runtime_instruction_identity")
        if not isinstance(identity, str):
            return None
        identities.add(identity)
    if len(identities) != 1:
        return None
    identity = next(iter(identities))
    if (not isinstance(identity, str) or len(identity) != 64
            or any(char not in "0123456789abcdef" for char in identity)):
        return None
    return identity


def _result_audit(
    path: Path, trials: tuple[str, ...], *,
    compact_inspection: bool | None = None,
) -> list[str]:
    try:
        rows = [json.loads(line) for line in path.read_text(
            encoding="utf-8").split("\n") if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot audit results: {exc}"]
    counts = defaultdict(int)
    errors: list[str] = []
    for row in rows:
        if type(row) is not dict:
            errors.append("results ledger contains a non-object row")
            continue
        trial = row.get("trial")
        features = row.get("features")
        if trial in trials:
            counts[trial] += 1
            if type(features) is not dict:
                errors.append(f"{trial} has malformed features")
                continue
            if compact_inspection is not None and (
                    row.get("compact_inspection") is not compact_inspection
                    or features.get("compact_inspection") is not
                    compact_inspection):
                errors.append(
                    f"{trial} compact_inspection differs from the arm")
            audit_status = features.get("contract_audit_status")
            contract = features.get("final_obligation_contract")
            outcomes = features.get("final_obligation_outcomes_ordered")
            known_statuses = {
                "available", "missing", "empty", "malformed",
                "outcomes_malformed", "unavailable_unscoreable",
            }
            if (audit_status not in known_statuses
                    or not isinstance(contract, list)
                    or not isinstance(outcomes, list)):
                errors.append(
                    f"{trial} has an invalid raw contract audit envelope")
            elif audit_status == "available" and (
                    not contract or len(outcomes) != len(contract)):
                errors.append(
                    f"{trial} has a misaligned available raw contract audit")
    errors.extend(
        f"expected one result for {trial}, got {counts[trial]}"
        for trial in trials if counts[trial] != 1
    )
    return errors


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_state(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_one(
    *, manifest: dict[str, Any], comparison: str, tier: str, phase: str,
    arm: str, group: Group, output: Path, source_root: Path,
    source_commit: str,
    source_fingerprint: str, manifest_sha256: str,
) -> dict[str, Any]:
    run_id = f"{comparison}-{tier}-{arm}-{group.key}-{uuid.uuid4().hex[:8]}"
    run_dir = output / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    scores = run_dir / "results.jsonl"
    usage = Path(f"{scores}.usage.jsonl")
    stdout_path = run_dir / "results.tsv"
    stderr_path = run_dir / "run.log"
    scoring_root = run_dir / "scoring"
    rep = run_id
    command = _command(
        manifest, source_root, group, phase, scores, rep, comparison, arm)
    compact_inspection, arm_cfg = _arm_profile(manifest, comparison, arm)
    env = os.environ.copy()
    env.update(arm_cfg)
    env["GV_SCORING_ROOT"] = os.fspath(scoring_root)
    for name in _CLEARED_ENV:
        env.pop(name, None)
    env["PATH"] = manifest["verifier"]["runtime_path"]
    env["GV_MAX_CONCURRENT_SEATS"] = str(
        manifest["verifier"]["max_concurrent_seats"])
    env["GV_OFFLINE_RUNTIME_READ_ROOTS"] = json.dumps(
        manifest["verifier"]["offline_runtime_read_roots"],
        separators=(",", ":"),
    )
    start_wall = time.time_ns()
    start_mono = time.monotonic_ns()
    state_path = run_dir / "run-state.json"
    _write_state(state_path, {
        "schema": RUN_SCHEMA, "run_id": run_id, "status": "running",
        "compact_inspection": compact_inspection,
        "started_at_unix_ns": start_wall,
        "started_at_monotonic_ns": start_mono,
    })
    with stdout_path.open("w", encoding="utf-8") as stdout, \
            stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(
            command, cwd=source_root, env=env, stdout=stdout, stderr=stderr,
            text=True, check=False)
    finish_mono = time.monotonic_ns()
    finish_wall = time.time_ns()
    fingerprint_after = _source_fingerprint(source_root)
    usage_ok, usage_errors = _attempt_audit(
        usage, compact_inspection=compact_inspection)
    runtime_instruction_identity = _attempt_runtime_identity(usage)
    result_errors = _result_audit(
        scores, group.trials, compact_inspection=compact_inspection)
    fingerprint_unchanged = fingerprint_after == source_fingerprint
    usage_ok = (usage_ok and proc.returncode == 0 and not result_errors
                and fingerprint_unchanged)
    record = {
        "schema": RUN_SCHEMA, "comparison": comparison, "tier": tier,
        "phase": phase, "arm": arm, "run_id": run_id,
        "source_commit": source_commit,
        "source_root": os.fspath(source_root),
        "verifier_fingerprint": source_fingerprint,
        "verifier_fingerprint_after": fingerprint_after,
        "source_fingerprint_unchanged": fingerprint_unchanged,
        "manifest_sha256": manifest_sha256,
        "job": group.job, "tasks": list(group.tasks),
        "trials": list(group.trials),
        "results_path": os.path.relpath(scores, output),
        "usage_paths": [os.path.relpath(usage, output)],
        "stdout_path": os.path.relpath(stdout_path, output),
        "stderr_path": os.path.relpath(stderr_path, output),
        "started_at_unix_ns": start_wall, "finished_at_unix_ns": finish_wall,
        "started_at_monotonic_ns": start_mono,
        "finished_at_monotonic_ns": finish_mono,
        "wall_seconds": (finish_mono - start_mono) / 1e9,
        "wall_includes_contract_build": True,
        "usage_audit_complete": usage_ok,
        "usage_audit_errors": usage_errors,
        "result_audit_errors": result_errors,
        "exit_code": proc.returncode,
        "command": command,
        "model": manifest["verifier"]["model"],
        "seats": manifest["verifier"]["seats"],
        "effort": manifest["verifier"]["effort"],
        "service_tier": manifest["verifier"]["service_tier"],
        "compact_inspection": compact_inspection,
        "runtime_instruction_identity": runtime_instruction_identity,
        "mode": manifest["verifier"]["mode"],
        "max_concurrent_seats": manifest["verifier"]["max_concurrent_seats"],
        "runtime_path": manifest["verifier"]["runtime_path"],
        "offline_runtime_read_roots":
            manifest["verifier"]["offline_runtime_read_roots"],
        "browser_observer": None,
        "workers": manifest["phases"][phase]["workers"],
        "fresh_scoring_root": True, "fresh_process_cache": True,
        "shared_toolchain": phase != "cold", **arm_cfg,
    }
    _append_jsonl(output / "eval-runs.jsonl", record)
    _write_state(state_path, {
        "schema": RUN_SCHEMA, "run_id": run_id, "status": "terminal",
        "exit_code": proc.returncode,
        "started_at_unix_ns": start_wall,
        "finished_at_unix_ns": finish_wall,
        "started_at_monotonic_ns": start_mono,
        "finished_at_monotonic_ns": finish_mono,
        "usage_audit_complete": usage_ok,
        "compact_inspection": compact_inspection,
        "runtime_instruction_identity": runtime_instruction_identity,
        "result_audit_errors": result_errors,
        "source_fingerprint_unchanged": fingerprint_unchanged,
    })
    return record


def _plan_rows(
    manifest: dict[str, Any], source_root: Path, comparison: str, tier: str,
    phases: tuple[str, ...],
    arms: tuple[str, ...], output: Path,
) -> list[dict[str, Any]]:
    selected = _selected(manifest, tier)
    rows = []
    for phase in phases:
        groups = _groups(phase, selected)
        for arm, group in _schedule(phase, groups, arms):
            run_id = f"PLAN-{comparison}-{tier}-{arm}-{group.key}"
            scores = output / "runs" / run_id / "results.jsonl"
            rows.append({
                "comparison": comparison, "tier": tier, "phase": phase,
                "arm": arm, "job": group.job, "tasks": list(group.tasks),
                "trials": list(group.trials),
                "environment": _arm_profile(
                    manifest, comparison, arm)[1],
                "compact_inspection": _arm_profile(
                    manifest, comparison, arm)[0],
                "command": _command(
                    manifest, source_root, group, phase, scores, run_id,
                    comparison, arm),
            })
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--comparison",
                        choices=("full-e2e", "adapter-isolation",
                                 "compact-inspection"),
                        default="full-e2e")
    parser.add_argument("--tier", choices=("canary", "core", "acceptance"),
                        default="canary")
    parser.add_argument("--phase", choices=(*PHASES, "all"),
                        required=True)
    parser.add_argument("--arm", choices=("control", "treatment", "both"),
                        default="both")
    parser.add_argument("--output-root", type=Path,
                        default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path,
                        default=DEFAULT_SOURCE_ROOT,
                        help="frozen verifier source/archive root")
    parser.add_argument("--source-commit", default="",
                        help="commit stamped onto an archive without .git")
    parser.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT,
                        help="absolute root containing the archived jobs")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid", default="")
    args = parser.parse_args(argv)
    try:
        manifest = _manifest(args.manifest)
        arms = ("control", "treatment") if args.arm == "both" else (args.arm,)
        phases = PHASES if args.phase == "all" else (args.phase,)
        output = args.output_root.expanduser().resolve()
        source_root = args.source_root.expanduser().resolve()
        jobs_root = args.jobs_root.expanduser().resolve()
        _validate_source_root(source_root)
        plan = _plan_rows(
            manifest, source_root, args.comparison, args.tier, phases, arms,
            output)
        if not args.execute:
            print(json.dumps(plan, indent=2))
            return 0
        if args.confirm_paid != "GPT-5.6-Luna":
            raise PlanError(
                "paid execution requires --confirm-paid GPT-5.6-Luna")
        if args.comparison == "adapter-isolation":
            raise PlanError(
                "adapter-isolation execution requires shared sealed-contract "
                "charge/export support; separate processes are invalid")
        _validate_execute_output_root(output)
        if output.exists() and any(output.iterdir()):
            raise PlanError(f"output root must be new or empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        source_commit = _commit(source_root, args.source_commit)
        source_fingerprint = _source_fingerprint(source_root)
        manifest_sha256 = _file_sha256(args.manifest)
        selected = _selected(manifest, args.tier)
        _prepare_jobs(source_root, jobs_root, selected)
        for phase in phases:
            groups = _groups(phase, selected)
            for arm, group in _schedule(phase, groups, arms):
                if _source_fingerprint(source_root) != source_fingerprint:
                    raise PlanError(
                        "verifier source changed during the evaluation")
                record = _run_one(
                    manifest=manifest, comparison=args.comparison,
                    tier=args.tier, phase=phase, arm=arm, group=group,
                    output=output, source_root=source_root,
                    source_commit=source_commit,
                    source_fingerprint=source_fingerprint,
                    manifest_sha256=manifest_sha256)
                print(json.dumps({
                    "run_id": record["run_id"],
                    "exit_code": record["exit_code"],
                    "wall_seconds": record["wall_seconds"],
                    "usage_audit_complete": record["usage_audit_complete"],
                    "result_audit_errors": record["result_audit_errors"],
                }, sort_keys=True), flush=True)
                if (record["exit_code"] or record["result_audit_errors"]
                        or not record["usage_audit_complete"]
                        or not record["source_fingerprint_unchanged"]):
                    return 1
        return 0
    except PlanError as exc:
        print(f"eval runner error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
