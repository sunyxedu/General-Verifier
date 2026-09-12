"""Metadata-only CyberGym progress tracking, never a solver input or score gate.

The runner owns ``attempts/<id>/status.json`` and its exact phase intervals.
Sidecars write role_call_started / role_call_finished to the private state's
role_calls.jsonl. Existing Codex usage ledgers provide token accounting.
The tracker never runs model calls, Docker commands, task evaluators, or task
lifecycle actions. An unchanged-progress warning is not a timeout or failure.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time
from typing import Any

MAX_JSON_BYTES = 1024 * 1024
MAX_JSONL_BYTES = 16 * 1024 * 1024
USAGE_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens",
                "reasoning_tokens", "reasoning_output_tokens")
FINISHED_STATES = frozenset({"completed", "failed", "aborted", "error"})


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _safe_file(path: Path, root: Path) -> bool:
    """Do not follow solver-controlled symlinks into unrelated local data."""
    try:
        if not path.resolve().is_relative_to(root.resolve()):
            return False
        current = path
        while current != root:
            if current.is_symlink():
                return False
            current = current.parent
        return stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
    except OSError:
        return False


def _json(path: Path, root: Path) -> dict:
    try:
        if not _safe_file(path, root) or path.stat().st_size > MAX_JSON_BYTES:
            return {}
        value = json.loads(path.read_bytes())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _jsonl(path: Path, root: Path) -> tuple[list[dict], bool]:
    """Bounded tail read; missing or partial telemetry is explicitly unknown."""
    if not path.exists():
        return [], False
    if not _safe_file(path, root):
        return [], True
    rows, partial = [], False
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            partial = size > MAX_JSONL_BYTES
            if partial:
                stream.seek(size - MAX_JSONL_BYTES)
                stream.readline()
            data = stream.read(MAX_JSONL_BYTES)
        for raw in data.splitlines(keepends=True):
            if not raw.endswith(b"\n"):
                partial = True
                continue
            try:
                value = json.loads(raw)
            except ValueError:
                partial = True
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                partial = True
    except OSError:
        partial = True
    return rows, partial


def interval_union(intervals: list[tuple[float, float]]) -> float:
    """Elapsed interval union, not CPU time or aggregate parallel call work."""
    merged: list[list[float]] = []
    for start, finish in sorted(intervals):
        if finish < start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], finish)
        else:
            merged.append([start, finish])
    return round(sum(finish - start for start, finish in merged), 6)


def summarize_calls(calls: list[dict], now: float) -> dict:
    """Keep failed-call work and still-running ages; do not invent missing times."""
    by_role: dict[str, dict] = {}
    all_intervals: list[tuple[float, float]] = []
    missing_interval = False
    active = []
    for call in calls:
        role = str(call.get("role", "unknown"))
        item = by_role.setdefault(role, {"finished_calls": 0, "completed_calls": 0,
            "failed_calls": 0, "running_calls": 0, "call_work_seconds_sum": 0.0,
            "wall_seconds_union": 0.0, "missing_interval_calls": 0,
            "intervals": []})
        start, end = _number(call.get("started_unix")), _number(call.get("completed_unix"))
        status = call.get("status", "running")
        if status not in FINISHED_STATES:
            item["running_calls"] += 1
            active.append({key: call.get(key) for key in
                ("call_id", "role", "model", "service_tier", "started_unix")})
            active[-1]["age_seconds"] = round(max(0, now - start), 6) if start is not None else None
            continue
        item["finished_calls"] += 1
        item["completed_calls" if status == "completed" else "failed_calls"] += 1
        duration = _number(call.get("duration_seconds"))
        valid_interval = start is not None and end is not None and end >= start
        if duration is None and valid_interval:
            duration = end - start
        if duration is not None:
            item["call_work_seconds_sum"] += duration
        if valid_interval:
            item["intervals"].append((start, end))
            all_intervals.append((start, end))
        else:
            item["missing_interval_calls"] += 1
            missing_interval = True
    for item in by_role.values():
        intervals = item.pop("intervals")
        item["call_work_seconds_sum"] = round(item["call_work_seconds_sum"], 6)
        item["wall_seconds_union"] = (None if item["missing_interval_calls"]
                                        else interval_union(intervals))
    return {"by_role": by_role, "active_calls": active,
            "call_work_seconds_sum": round(sum(x["call_work_seconds_sum"]
                for x in by_role.values()), 6),
            "wall_seconds_union": None if missing_interval else interval_union(all_intervals)}


def _role_calls(rows: list[dict]) -> list[dict]:
    calls: dict[str, dict] = {}
    for row in rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else row
        kind = row.get("type") or payload.get("type")
        if kind not in {"role_call_started", "role_call_finished"}:
            continue
        call_id = payload.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            continue
        call = calls.setdefault(call_id, {"call_id": call_id})
        for key in ("role", "model", "service_tier", "duration_seconds"):
            if key in payload:
                call[key] = payload[key]
        ts = _number(row.get("ts", payload.get("ts")))
        if kind == "role_call_started":
            call["started_unix"] = payload.get("started_unix", ts)
            # A repeated start record cannot undo a retained completion.
            call.setdefault("status", "running")
        else:
            call["completed_unix"] = payload.get("completed_unix", ts)
            call["status"] = payload.get("status", "completed")
            if "started_unix" in payload:
                call["started_unix"] = payload["started_unix"]
    return list(calls.values())


def _usage_calls(rows: list[dict]) -> list[dict]:
    calls = {}
    for row in rows:
        call_id = row.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            continue
        start, end = _number(row.get("started_at_unix_ns")), _number(row.get("finished_at_unix_ns"))
        calls[call_id] = {"call_id": call_id, "role": row.get("role", "unknown"),
            "model": row.get("model"), "service_tier": row.get("service_tier"),
            "started_unix": start / 1e9 if start is not None else None,
            "completed_unix": end / 1e9 if end is not None else None,
            "duration_seconds": row.get("duration_seconds"),
            "status": "completed" if row.get("returncode") == 0 else "failed"}
    return list(calls.values())


def _usage_summary(rows: list[dict]) -> dict:
    unique = {row["call_id"]: row for row in rows if isinstance(row.get("call_id"), str)}
    result: dict[str, dict] = {}
    for row in unique.values():
        item = result.setdefault(str(row.get("role", "unknown")), {"recorded_calls": 0})
        item["recorded_calls"] += 1
        for field in USAGE_FIELDS:
            value = row.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                item[field] = item.get(field, 0) + value
    return result


def _evaluator_calls(events: list[dict]) -> list[dict]:
    calls: dict[str, dict] = {}
    for row in events:
        payload = row.get("payload", {})
        if not isinstance(payload, dict) or payload.get("tool") != "cyber_evaluator":
            continue
        call_id = payload.get("tool_use_id")
        if not isinstance(call_id, str):
            continue
        call = calls.setdefault(call_id, {"call_id": call_id, "role": "evaluator"})
        if row.get("type") == "tool_call":
            call["started_unix"] = row.get("ts")
            call.setdefault("status", "running")
        elif row.get("type") == "tool_result":
            output = payload.get("output", {})
            if not isinstance(output, dict):
                output = {}
            call.update(completed_unix=row.get("ts"),
                duration_seconds=output.get("elapsed_seconds"),
                status="failed" if output.get("error") or output.get("execution_error")
                    or output.get("timed_out") else "completed")
    return list(calls.values())


def _phases(status: dict, now: float) -> dict:
    intervals = status.get("phase_intervals", [])
    calls = []
    for i, phase in enumerate(intervals if isinstance(intervals, list) else []):
        if not isinstance(phase, dict):
            continue
        calls.append({"call_id": str(i), "role": phase.get("stage", "unknown"),
            "started_unix": phase.get("started_unix"),
            "completed_unix": phase.get("completed_unix"),
            "status": "completed" if _number(phase.get("completed_unix")) is not None else "running"})
    return summarize_calls(calls, now)


def _fingerprint(value: dict) -> str:
    stable = {key: value.get(key) for key in ("stage", "outcome", "official_pass", "completed_unix")}
    stable["progress"] = value["progress"]
    stable["roles"] = [{key: call.get(key) for key in ("call_id", "role", "status", "completed_unix")}
        for call in value["_calls"]]
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def _source_signature(status_path: Path, root: Path) -> dict:
    """Finished attempts normally never change; avoid rereading 1507 ledgers."""
    state = status_path.parent / "private" / "state"
    signatures = {}
    for path in (status_path, state / "role_calls.jsonl", state / "backend_usage.jsonl", state / "events.jsonl"):
        try:
            info = path.stat() if _safe_file(path, root) else None
        except OSError:  # A mirrored file can be atomically replaced between stats.
            info = None
        signatures[str(path.relative_to(status_path.parent))] = (
            [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns] if info else None)
    return signatures


def attempt_snapshot(status_path: Path, root: Path, now: float, *, status: dict | None = None) -> dict:
    status = _json(status_path, root) if status is None else status
    state = status_path.parent / "private" / "state"
    role_rows, role_partial = _jsonl(state / "role_calls.jsonl", root)
    usage_rows, usage_partial = _jsonl(state / "backend_usage.jsonl", root)
    events, event_partial = _jsonl(state / "events.jsonl", root)
    calls = _role_calls(role_rows)
    source = "role_calls" if calls else "backend_usage_completed_only"
    if not calls:
        calls = _usage_calls(usage_rows)
    evaluator = _evaluator_calls(events)
    counts = Counter(row.get("type") for row in events)
    value = {key: status.get(key) for key in ("task_id", "attempt_id", "stage", "started_unix",
        "stage_started_unix", "updated_unix", "completed_unix", "outcome", "official_pass",
        "solver_container", "sidecar_container")}
    value["reason"] = str(status.get("reason") or "")[:512]
    start, end = _number(status.get("started_unix")), _number(status.get("completed_unix"))
    stage_start = _number(status.get("stage_started_unix"))
    value.update(elapsed_seconds=round(max(0, (end if end is not None else now) - start), 6)
                 if start is not None else None,
        stage_age_seconds=round(max(0, now - stage_start), 6) if stage_start is not None and end is None else None,
        runner_phase_timings=_phases(status, now),
        verifier_timings={"source": source, "available": bool(role_rows or usage_rows),
                          "partial": role_partial or usage_partial,
                          **summarize_calls(calls, now)},
        evaluator_timings={"available": bool(events), "partial": event_partial,
                           **summarize_calls(evaluator, now)},
        backend_usage_by_role=_usage_summary(usage_rows),
        progress={"events_last_seq": events[-1].get("seq") if events else None,
            "events_in_window": len(events), "events_window_partial": event_partial,
            "registered_candidates_in_window": counts["cyber_candidate_registered"],
            "recorded_observations_in_window": counts["cyber_observation_recorded"],
            "materializations_in_window": counts["cyber_materialized"],
            "completed_role_calls": sum(c.get("status") in FINISHED_STATES for c in calls),
            "completed_evaluator_calls": sum(c.get("status") in FINISHED_STATES for c in evaluator)},
        current_activity="verifier" if any(c.get("status") not in FINISHED_STATES for c in calls)
            else "evaluator" if any(c.get("status") not in FINISHED_STATES for c in evaluator)
            else status.get("stage"), _calls=calls + evaluator)
    value["progress_fingerprint"] = _fingerprint(value)
    value["source_signature"] = _source_signature(status_path, root)
    value.pop("_calls")
    return value


def snapshot(root: Path | str, *, now: float | None = None, previous: dict | None = None) -> dict:
    root = Path(root).absolute()
    if root.is_symlink() or root.resolve() != root or not root.is_dir():
        raise ValueError("monitor root must be an existing canonical directory")
    now = time.time() if now is None else now
    prior = {a.get("attempt_id"): a for a in (previous or {}).get("attempts", [])}
    attempts = []
    for path in sorted((root / "attempts").glob("*/status.json")):
        status = _json(path, root)
        if not status:
            continue
        old = prior.get(status.get("attempt_id"), {})
        if (old and _number(status.get("completed_unix")) is not None
                and old.get("source_signature") == _source_signature(path, root)
                and not old["verifier_timings"]["active_calls"]
                and not old["evaluator_timings"]["active_calls"]):
            value = dict(old)
        else:
            value = attempt_snapshot(path, root, now, status=status)
        unchanged = old.get("progress_fingerprint") == value["progress_fingerprint"]
        value["unchanged_snapshots"] = old.get("unchanged_snapshots", 1) + 1 if unchanged else 1
        attempts.append(value)
    alerts = []
    for value in attempts:
        old = prior.get(value["attempt_id"], {})
        base = {key: value[key] for key in ("task_id", "attempt_id", "stage", "reason")}
        if _number(value["completed_unix"]) is None and value["unchanged_snapshots"] == 3:
            alerts.append({"type": "unchanged_progress_warning", "ts": now, **base,
                "meaning": "Three unchanged observations; inspect trace. This is not a timeout, failure, or restart instruction."})
        if value["official_pass"] is False and _number(value["completed_unix"]) is not None and old.get("official_pass") is not False:
            alerts.append({"type": "official_task_not_passed", "ts": now, **base})
        if value["outcome"] in {"infrastructure_error", "runner_error"} and old.get("outcome") != value["outcome"]:
            alerts.append({"type": "infrastructure_error", "ts": now, **base})
    free = os.statvfs(root)
    # Counts describe attempts only. The runner independently authenticates
    # official identities and decides task-set gates; this module never does.
    return {"schema": "cybergym-tracking-v1", "checked_unix": now,
        "root": str(root), "free_gib": round(free.f_bavail * free.f_frsize / 1024**3, 3),
        "attempt_count": len(attempts), "completed_attempts": sum(_number(a["completed_unix"]) is not None for a in attempts),
        "official_passed_attempts": sum(a["official_pass"] is True for a in attempts),
        "attempts": attempts, "alerts": alerts,
        "measurement": "Call-work sum includes parallel work; interval union removes overlap. Neither is CPU time. No success gates or lifecycle actions are inferred."}


def _atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".latest-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def record_snapshot(root: Path | str, *, now: float | None = None) -> dict:
    root = Path(root).absolute()
    monitor = root / "monitor"
    if root.resolve() != root or not root.is_dir() or monitor.is_symlink():
        raise ValueError("monitor destination must be canonical and task-independent")
    monitor.mkdir(exist_ok=True)
    for name in ("latest.json", "history.jsonl", "alerts.jsonl"):
        if (monitor / name).is_symlink():
            raise ValueError("monitor output must not be a symlink")
    value = snapshot(root, now=now, previous=_json(monitor / "latest.json", root))
    with (monitor / "history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    if value["alerts"]:
        with (monitor / "alerts.jsonl").open("a", encoding="utf-8") as stream:
            for alert in value["alerts"]:
                stream.write(json.dumps(alert, ensure_ascii=False, sort_keys=True) + "\n")
    _atomic_json(monitor / "latest.json", value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=300)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("interval must be finite and positive")
    while True:
        value = record_snapshot(args.root)
        print(json.dumps({key: value[key] for key in ("checked_unix", "attempt_count",
            "completed_attempts", "official_passed_attempts", "free_gib", "alerts")},
            ensure_ascii=False, sort_keys=True), flush=True)
        if not args.watch:
            return 0
        if (args.root / "stop-monitor.request").exists():
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
