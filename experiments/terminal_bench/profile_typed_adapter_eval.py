"""Read-only latency/cost profile for a typed-adapter evaluation run ledger.

This is deliberately a diagnostic, not an acceptance analyzer.  It reads the
runner's terminal ``eval-runs.jsonl`` rows, their private usage ledgers, and
the derived verifier event logs under each run directory.  It never emits
prompts, method text, ledger payloads, or evidence content.

Usage::

    python experiments/terminal_bench/profile_typed_adapter_eval.py \
        /path/to/eval-runs.jsonl --json-out /path/to/profile.json

The human summary goes to stdout.  ``--json-only`` instead prints the compact
machine-readable report to stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROFILE_SCHEMA = "gv-typed-adapter-eval-profile/v1"
RUN_SCHEMA = "gv-terminal-bench-eval-run/v1"
ATTEMPT_RECORD = "codex_cli_attempt"
TOKEN_FIELDS = (
    "input_tokens",
    "uncached_input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
)
COST_TOKEN_FIELDS = (
    "uncached_input_tokens", "cached_input_tokens", "output_tokens",
)
EVENT_TYPES = (
    "contract_cache_hit",
    "contract_cache_miss",
    "contract_cache_bypass",
    "contract_cache_published",
    "method_cache_hit",
    "method_cache_miss",
    "method_cache_bypass",
    "method_cache_direction_audit",
    "adapter_discharge",
    "adapter_fallback",
    "methods_enumerated",
    "discharge",
    "judger_verdict",
    "method_redo",
    "method_rejected",
)
MAX_JSONL_BYTES = 512 * 1024 * 1024
MAX_JSONL_ROWS = 2_000_000
MAX_EVENT_FILES = 20_000
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9_.-]{1,240}")


class ProfileError(ValueError):
    pass


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value)) and value >= 0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _label(value: Any, fallback: str = "<unknown>") -> str:
    if not isinstance(value, str) or not value:
        return fallback
    # Labels are useful grouping dimensions, not a channel for raw content.
    return value[:160]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ProfileError(f"cannot stat {path.name}: {error}") from error
    if size > MAX_JSONL_BYTES:
        raise ProfileError(f"{path.name} exceeds the profiling size cap")
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if lineno > MAX_JSONL_ROWS:
                    raise ProfileError(
                        f"{path.name} exceeds the profiling row cap")
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ProfileError(
                        f"{path.name}:{lineno} is not a JSON object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProfileError(f"cannot read {path.name}: {error}") from error
    return rows


def _private_path(base: Path, run_id: str, value: Any) -> Path:
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ProfileError("run_id is not a safe path component")
    if not isinstance(value, str) or not value:
        raise ProfileError("private path is missing")
    candidate = Path(os.path.expanduser(value))
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    root = (base / "runs" / run_id).resolve()
    try:
        inside = os.path.commonpath((root, candidate)) == os.fspath(root)
    except ValueError:
        inside = False
    if not inside:
        raise ProfileError("private path escapes the run directory")
    return candidate


def _interval_union(intervals: Iterable[tuple[int, int]]) -> float:
    ordered = sorted(intervals)
    if not ordered:
        return 0.0
    total = 0
    start, finish = ordered[0]
    for next_start, next_finish in ordered[1:]:
        if next_start <= finish:
            finish = max(finish, next_finish)
        else:
            total += finish - start
            start, finish = next_start, next_finish
    total += finish - start
    return total / 1e9


class _RoleProfile:
    def __init__(self) -> None:
        self.attempts = 0
        self.started = 0
        self.not_started = 0
        self.start_unknown = 0
        self.statuses: Counter[str] = Counter()
        self.usage_statuses: Counter[str] = Counter()
        self.token_known = Counter()
        self.token_unknown = Counter()
        self.durations: list[float] = []
        self.duration_unknown = 0
        self.intervals: list[tuple[int, int]] = []
        self.interval_unknown = 0

    def add(self, row: dict[str, Any], issues: list[str], run_id: str) -> None:
        self.attempts += 1
        started = row.get("started")
        if started is True:
            self.started += 1
        elif started is False:
            self.not_started += 1
        else:
            self.start_unknown += 1
        self.statuses[_label(row.get("attempt_status"))] += 1
        self.usage_statuses[_label(row.get("usage_status"))] += 1

        usage_exact = (started is True
                       and row.get("usage_complete") is True
                       and row.get("usage_status") == "complete")
        input_tokens = row.get("input_tokens")
        cached = row.get("cached_input_tokens")
        uncached = row.get("uncached_input_tokens")
        if usage_exact and all(_nonnegative_int(item)
                               for item in (input_tokens, cached, uncached)) \
                and (cached > input_tokens
                     or uncached != input_tokens - cached):
            # Internally inconsistent "complete" telemetry is unknown cost,
            # not a known total accompanied by a warning.
            usage_exact = False
            issues.append(
                f"{run_id}: inconsistent complete input-token accounting")
        for field in TOKEN_FIELDS:
            value = row.get(field)
            if usage_exact and _nonnegative_int(value):
                self.token_known[field] += value
            elif started is not False:
                self.token_unknown[field] += 1

        start = row.get("started_at_unix_ns")
        finish = row.get("finished_at_unix_ns")
        valid_interval = (started is True and _nonnegative_int(start)
                          and _nonnegative_int(finish) and finish >= start)
        supplied_duration = row.get("duration_seconds")
        if valid_interval:
            duration = (finish - start) / 1e9
            self.intervals.append((start, finish))
            self.durations.append(duration)
            if _number(supplied_duration):
                tolerance = max(0.01, duration * 1e-6)
                if abs(float(supplied_duration) - duration) > tolerance:
                    issues.append(
                        f"{run_id}: duration_seconds disagrees with timestamps")
        elif started is True:
            self.interval_unknown += 1
            if _number(supplied_duration):
                self.durations.append(float(supplied_duration))
            else:
                self.duration_unknown += 1

    def result(self, role: str) -> dict[str, Any]:
        tokens = {}
        for field in TOKEN_FIELDS:
            unknown = self.token_unknown[field]
            known = self.token_known[field]
            tokens[field] = {
                "known_total": known,
                "total": known if unknown == 0 else None,
                "unknown_started_attempts": unknown,
            }
        durations = self.durations
        return {
            "role": role,
            "attempts": self.attempts,
            "started": self.started,
            "not_started": self.not_started,
            "start_unknown": self.start_unknown,
            "attempt_statuses": dict(sorted(self.statuses.items())),
            "usage_statuses": dict(sorted(self.usage_statuses.items())),
            "cost_complete": all(
                tokens[field]["total"] is not None
                for field in COST_TOKEN_FIELDS),
            "tokens": tokens,
            "duration_seconds": {
                "known_count": len(durations),
                "unknown_started_attempts": self.duration_unknown,
                "sum": sum(durations),
                "p50": statistics.median(durations) if durations else None,
                "max": max(durations) if durations else None,
            },
            "call_interval_union_seconds": {
                "known_union": _interval_union(self.intervals),
                "total": (_interval_union(self.intervals)
                          if self.interval_unknown == 0 else None),
                "unknown_started_attempts": self.interval_unknown,
            },
        }


class _GroupProfile:
    def __init__(self) -> None:
        self.roles: dict[str, _RoleProfile] = defaultdict(_RoleProfile)
        self.events: Counter[str] = Counter()
        self.run_ids: set[str] = set()
        self.complete_run_ids: set[str] = set()
        self.usage_files = 0
        self.usage_files_readable = 0

    def result(self, arm: str, task: str) -> dict[str, Any]:
        roles = [self.roles[name].result(name) for name in sorted(self.roles)]
        event_counts = {name: self.events[name] for name in EVENT_TYPES}
        return {
            "arm": arm,
            "task": task,
            "runs": {
                "total": len(self.run_ids),
                "complete": len(self.complete_run_ids),
                "incomplete": len(self.run_ids - self.complete_run_ids),
            },
            "usage_files": {
                "declared": self.usage_files,
                "readable": self.usage_files_readable,
            },
            "roles": roles,
            "events": {
                "contract_cache": {
                    "hits": event_counts["contract_cache_hit"],
                    "misses": event_counts["contract_cache_miss"],
                    "bypasses": event_counts["contract_cache_bypass"],
                    "published": event_counts["contract_cache_published"],
                },
                "method_cache": {
                    "hits": event_counts["method_cache_hit"],
                    # The current engine does not emit a miss for an empty or
                    # ineligible method-cache lookup.  Keep explicit misses
                    # separate instead of deriving a false zero.
                    "explicit_miss_events": event_counts["method_cache_miss"],
                    "misses_observable": False,
                    "bypasses": event_counts["method_cache_bypass"],
                    # A new independent audit is not a cache hit. One event
                    # can include multiple real CLI attempts after retries;
                    # those remain charged separately in the role totals.
                    "direction_audit_events": event_counts[
                        "method_cache_direction_audit"],
                },
                "adapter": {
                    "executions": event_counts["adapter_discharge"],
                    "fallbacks": event_counts["adapter_fallback"],
                },
                "binding": {
                    "methods_enumerated": event_counts["methods_enumerated"],
                    "discharges_recorded": event_counts["discharge"],
                    "judger_verdicts": event_counts["judger_verdict"],
                    "redos": event_counts["method_redo"],
                    "rejections": event_counts["method_rejected"],
                },
            },
        }


def _run_complete(run: dict[str, Any]) -> bool:
    return (
        run.get("schema") == RUN_SCHEMA
        and run.get("exit_code") == 0
        and run.get("usage_audit_complete") is True
        and run.get("usage_audit_errors") in (None, [])
        and run.get("result_audit_errors") in (None, [])
        and run.get("source_fingerprint_unchanged") is True
        and _nonnegative_int(run.get("started_at_monotonic_ns"))
        and _nonnegative_int(run.get("finished_at_monotonic_ns"))
        and run["finished_at_monotonic_ns"] > run["started_at_monotonic_ns"]
    )


def _attempt_usage_observed(row: dict[str, Any]) -> bool:
    """Whether this row independently supports the runner's usage audit."""
    if row.get("record_type") != ATTEMPT_RECORD:
        return False
    started = row.get("started")
    if started is False:
        return (row.get("attempt_status") == "launch_error"
                and row.get("usage_complete") is False)
    if started is not True or row.get("usage_complete") is not True \
            or row.get("usage_status") != "complete":
        return False
    if not all(_nonnegative_int(row.get(field))
               for field in ("input_tokens", *COST_TOKEN_FIELDS)):
        return False
    return (
        row["cached_input_tokens"] <= row["input_tokens"]
        and row["uncached_input_tokens"]
        == row["input_tokens"] - row["cached_input_tokens"]
    )


def _result_task_map(base: Path, run: dict[str, Any], run_id: str,
                     issues: list[str]) -> dict[str, str]:
    value = run.get("results_path")
    if not value:
        return {}
    try:
        path = _private_path(base, run_id, value)
        rows = _read_jsonl(path)
    except ProfileError as error:
        issues.append(f"{run_id}: results unavailable ({error})")
        return {}
    result = {}
    for row in rows:
        trial, task = row.get("trial"), row.get("task")
        if isinstance(trial, str) and trial and isinstance(task, str) and task:
            result[trial] = task
    return result


def _task_for_state(path: Path, trial_tasks: dict[str, str],
                    default: str) -> str:
    matches = {task for trial, task in trial_tasks.items()
               if trial in path.parts}
    return next(iter(matches)) if len(matches) == 1 else default


def _state_paths(run_root: Path, usage_states: dict[Path, set[str]],
                 trial_tasks: dict[str, str], default_task: str,
                 issues: list[str], run_id: str) -> dict[Path, str]:
    states = {}
    for path, tasks in usage_states.items():
        task = next(iter(tasks)) if len(tasks) == 1 else default_task
        states[path] = task
        if len(tasks) > 1:
            issues.append(f"{run_id}: one state has conflicting task labels")

    scoring = run_root / "scoring"
    if scoring.is_dir():
        count = 0
        try:
            for event_path in scoring.rglob("events.jsonl"):
                count += 1
                if count > MAX_EVENT_FILES:
                    issues.append(f"{run_id}: event-file scan exceeded cap")
                    break
                state = event_path.parent.resolve()
                try:
                    inside = os.path.commonpath(
                        (scoring.resolve(), state)) == os.fspath(scoring.resolve())
                except ValueError:
                    inside = False
                if inside:
                    states.setdefault(
                        state,
                        _task_for_state(state, trial_tasks, default_task))
        except OSError as error:
            issues.append(f"{run_id}: cannot scan state events ({error})")
    return states


def profile(run_ledger: Path | str) -> dict[str, Any]:
    ledger = Path(run_ledger).expanduser().resolve()
    base = ledger.parent
    try:
        runs = _read_jsonl(ledger)
    except ProfileError:
        # The wrapper appends its first ledger row only after the first
        # scoring process finishes. A real in-flight run is not an invalid
        # input path, but it is necessarily incomplete; retain the orphan
        # state audit below without inventing a successful zero-cost row.
        if ledger.exists() or not any(
                path.is_file()
                for path in (base / "runs").glob("*/run-state.json")):
            raise
        runs = []
    issues: list[str] = []
    groups: dict[tuple[str, str], _GroupProfile] = defaultdict(_GroupProfile)
    seen_run_ids: set[str] = set()
    seen_attempts: set[tuple[str, str]] = set()
    run_counts = Counter()

    for ordinal, run in enumerate(runs, 1):
        run_id = _label(run.get("run_id"), f"<run-{ordinal}>")
        if run_id in seen_run_ids:
            issues.append(f"{run_id}: duplicate terminal run row")
            run_counts["duplicate_rows"] += 1
            continue
        seen_run_ids.add(run_id)
        complete = _run_complete(run)
        run_counts["complete" if complete else "incomplete_or_failed"] += 1
        arm = _label(run.get("arm"))
        declared_tasks = {
            _label(task) for task in (run.get("tasks") or [])
            if isinstance(task, str) and task
        }
        default_task = (next(iter(declared_tasks))
                        if len(declared_tasks) == 1 else "<unknown>")
        if not declared_tasks:
            declared_tasks = {default_task}
        for task in declared_tasks:
            group = groups[(arm, task)]
            group.run_ids.add(run_id)
            if complete:
                group.complete_run_ids.add(run_id)

        trial_tasks = _result_task_map(base, run, run_id, issues)
        usage_states: dict[Path, set[str]] = defaultdict(set)
        usage_paths = run.get("usage_paths")
        usage_observed = True
        if not isinstance(usage_paths, list) or not usage_paths:
            issues.append(f"{run_id}: no usage ledger declared")
            usage_paths = []
            usage_observed = False
        for value in usage_paths:
            for task in declared_tasks:
                groups[(arm, task)].usage_files += 1
            try:
                usage_path = _private_path(base, run_id, value)
                usage_rows = _read_jsonl(usage_path)
            except ProfileError as error:
                issues.append(f"{run_id}: usage unavailable ({error})")
                usage_observed = False
                continue
            for task in declared_tasks:
                groups[(arm, task)].usage_files_readable += 1
            if not usage_rows:
                issues.append(f"{run_id}: usage ledger is empty")
                usage_observed = False
            for line_no, row in enumerate(usage_rows, 1):
                if not _attempt_usage_observed(row):
                    usage_observed = False
                attempt_id = row.get("attempt_id")
                if isinstance(attempt_id, str) and attempt_id:
                    identity = (run_id, attempt_id)
                    if identity in seen_attempts:
                        issues.append(f"{run_id}: duplicate attempt_id")
                        continue
                    seen_attempts.add(identity)
                if row.get("record_type") != ATTEMPT_RECORD:
                    issues.append(f"{run_id}: legacy or unknown usage record")
                task = _label(row.get("task"), default_task)
                if task not in declared_tasks and default_task != "<unknown>":
                    issues.append(f"{run_id}: usage task is outside run tasks")
                    task = default_task
                group = groups[(arm, task)]
                group.run_ids.add(run_id)
                if complete:
                    group.complete_run_ids.add(run_id)
                role = _label(row.get("role"))
                group.roles[role].add(row, issues, run_id)
                state_value = row.get("state_dir")
                if isinstance(state_value, str) and state_value:
                    try:
                        state = _private_path(base, run_id, state_value)
                    except ProfileError:
                        issues.append(
                            f"{run_id}: usage state_dir escapes private run")
                    else:
                        usage_states[state].add(task)

        if complete and not usage_observed:
            # Do not trust a contradictory terminal success marker.  The
            # profiler independently observed missing/incomplete cost data.
            complete = False
            run_counts["complete"] -= 1
            run_counts["incomplete_or_failed"] += 1
            for group in groups.values():
                group.complete_run_ids.discard(run_id)

        try:
            run_root = (base / "runs" / run_id).resolve()
            states = _state_paths(
                run_root, usage_states, trial_tasks, default_task,
                issues, run_id)
        except (OSError, ProfileError) as error:
            issues.append(f"{run_id}: cannot resolve run states ({error})")
            states = {}
        for state, task in states.items():
            try:
                events = _read_jsonl(state / "events.jsonl")
            except ProfileError as error:
                issues.append(f"{run_id}: events unavailable ({error})")
                continue
            group = groups[(arm, task)]
            group.run_ids.add(run_id)
            if complete:
                group.complete_run_ids.add(run_id)
            for event in events:
                event_type = event.get("type")
                if event_type in EVENT_TYPES:
                    group.events[event_type] += 1

    # A killed runner leaves run-state.json but no terminal eval-runs row.
    # Count it explicitly; never let absence from the terminal ledger read as
    # a successful zero-cost run.
    orphan_states = Counter()
    runs_root = base / "runs"
    if runs_root.is_dir():
        try:
            for child in runs_root.iterdir():
                state_path = child / "run-state.json"
                if not state_path.is_file():
                    continue
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    orphan_states["unreadable"] += 1
                    continue
                run_id = state.get("run_id")
                if run_id in seen_run_ids:
                    continue
                status = _label(state.get("status"))
                orphan_states[status] += 1
        except OSError as error:
            issues.append(f"cannot scan unfinished run states ({error})")

    group_results = [
        groups[key].result(*key) for key in sorted(groups)
    ]
    unknown_cost_attempts = sum(
        role["tokens"][field]["unknown_started_attempts"]
        for group in group_results for role in group["roles"]
        for field in COST_TOKEN_FIELDS
    )
    report = {
        "schema": PROFILE_SCHEMA,
        "definitions": {
            "duration_sum": "sum of observed model-attempt durations",
            "call_interval_union": (
                "union of observed wall-clock attempt intervals within one "
                "arm/task/role; it is not the dependency critical path"
            ),
            "unknown_cost": (
                "a started attempt without complete telemetry makes its "
                "token total null; known_total is only the exact observed "
                "lower component"
            ),
            "method_cache_miss": (
                "not observable in the current engine; explicit event count "
                "is reported but absence is not interpreted as zero misses"
            ),
        },
        "runs": {
            "ledger_rows": len(runs),
            "unique_terminal": len(seen_run_ids),
            "complete": run_counts["complete"],
            "incomplete_or_failed": run_counts["incomplete_or_failed"],
            "duplicate_rows": run_counts["duplicate_rows"],
            "orphan_run_states": dict(sorted(orphan_states.items())),
        },
        "cost_complete": unknown_cost_attempts == 0
            and run_counts["incomplete_or_failed"] == 0
            and not orphan_states,
        "unknown_cost_field_attempts": unknown_cost_attempts,
        "groups": group_results,
        "issues": issues,
    }
    return report


def _token_cell(token: dict[str, Any]) -> str:
    total = token["total"]
    if total is not None:
        return str(total)
    return f"? (known {token['known_total']})"


def render_summary(report: dict[str, Any]) -> str:
    runs = report["runs"]
    orphan_count = sum(runs["orphan_run_states"].values())
    lines = [
        f"Runs: {runs['complete']} complete, "
        f"{runs['incomplete_or_failed']} incomplete/failed, "
        f"{orphan_count} unfinished/orphan; "
        f"cost={'complete' if report['cost_complete'] else 'INCOMPLETE'}",
        "Interval union is observed call occupancy, not dependency critical path.",
    ]
    for group in report["groups"]:
        lines.append(f"\n[{group['arm']}] {group['task']}")
        for role in group["roles"]:
            tokens = role["tokens"]
            duration = role["duration_seconds"]
            union = role["call_interval_union_seconds"]
            lines.append(
                "  " + role["role"]
                + f": attempts={role['attempts']} "
                + f"status={role['attempt_statuses']} "
                + "tokens(u/c/o)="
                + "/".join(_token_cell(tokens[name]) for name in (
                    "uncached_input_tokens", "cached_input_tokens",
                    "output_tokens"))
                + f" duration(sum/p50/max)={duration['sum']:.2f}/"
                + (f"{duration['p50']:.2f}/{duration['max']:.2f}s"
                   if duration["p50"] is not None else "?/?s")
                + " interval_union="
                + (f"{union['total']:.2f}s" if union["total"] is not None
                   else f"? (known {union['known_union']:.2f}s)")
            )
        events = group["events"]
        contract, method = events["contract_cache"], events["method_cache"]
        adapter = events["adapter"]
        lines.append(
            "  events: contract h/m/b/p="
            f"{contract['hits']}/{contract['misses']}/"
            f"{contract['bypasses']}/{contract['published']}; "
            "method h/m*/b="
            f"{method['hits']}/{method['explicit_miss_events']}/"
            f"{method['bypasses']}; direction audit events="
            f"{method['direction_audit_events']}; adapter exec/fallback="
            f"{adapter['executions']}/{adapter['fallbacks']}"
        )
    if report["issues"]:
        lines.append(f"\nDiagnostics: {len(report['issues'])} issue(s); see JSON.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ledger", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = profile(args.run_ledger)
    except ProfileError as error:
        print(f"profile error: {error}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    if args.json_out is not None:
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded if args.json_only else render_summary(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
