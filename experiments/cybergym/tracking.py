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
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any

MAX_JSON_BYTES = 1024 * 1024
MAX_MONITOR_PREVIOUS_BYTES = 64 * 1024 * 1024
MAX_JSONL_BYTES = 16 * 1024 * 1024
MAX_ROLLOUT_BYTES = 2 * 1024 * 1024
PROGRESS_SCHEMA = "gv-verification-progress-v1"
JOB_STATES = frozenset({"queued", "running", "completed", "failed", "interrupted"})
# Durable service jobs include candidate registration/evaluation as well as
# adjudication. This is an operational verb inventory, not verified outcomes.
VERIFICATION_TOOLS = frozenset({"commit", "verify", "certificate", "cyber_register_candidate",
    "cyber_evaluate_candidate", "cyber_finalize_candidate"})
TRANSPORT_TOOLS = VERIFICATION_TOOLS | frozenset({"status", "cyber_checkpoint",
    "cyber_register_candidate", "cyber_candidates", "cyber_evaluate_candidate",
    "verification_job", "verification_jobs", "verification_retry"})
LONG_CALL_SECONDS = 600
BACKEND_HEARTBEAT_STALE_SECONDS = 300
FRESH_METADATA_SECONDS = 120
BACKEND_PHASES = frozenset({"exec_dispatch", "exec_spawned", "exec_spawn_failed",
    "cli_event", "exec_heartbeat", "exec_process_exited", "pipe_drain_heartbeat",
    "exec_returned", "backend_transient_failure", "backend_backoff_started",
    "backend_backoff_finished"})
CLI_EVENT_TYPES = frozenset({"thread.started", "turn.started", "turn.completed", "turn.failed",
    "item.started", "item.updated", "item.completed", "error"})
CLI_ITEM_KINDS = frozenset({"command_execution", "mcp_tool_call", "web_search", "file_change",
    "agent_message", "reasoning", "plan_update", "other"})
COMMAND_STAT_COUNTERS = ("item_limit", "counter_limit", "completed_items", "fingerprinted_items",
    "unique_fingerprints", "repeated_fingerprint_items", "max_fingerprint_repetitions",
    "duplicate_completions", "untracked_completions", "success_exits", "nonzero_exits", "unknown_exits")
BACKEND_WAIT_CATEGORIES = frozenset({"unknown", "cli_startup_unknown", "cli_finalization_unknown",
    "cli_activity_unknown", "cli_tool_item_active", "model_wait_unknown", "pipe_drain_unknown"})
BACKEND_ROLES = frozenset({"obligations", "critic", "oracles", "discharge", "judger"})
USAGE_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens",
                "reasoning_tokens", "reasoning_output_tokens")
FINISHED_STATES = frozenset({"completed", "failed", "aborted", "error"})
ADAPTER_OBSERVATION_SCHEMA = "gv-cybergym-vul-observation-v1"
# Host/transport/evidence faults only. In particular, invalid_candidate is
# not inferred to be infrastructure failure, nor is any target exit code.
ADAPTER_INFRASTRUCTURE_ERRORS = frozenset({
    "invalid_arguments", "invalid_configuration", "unexpected_redirect",
    "unexpected_http_status", "transport_error", "invalid_response",
    "response_too_large", "evidence_storage_error",
})


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


def _valid_monitor_previous(value: Any) -> bool:
    """Validate only cache identities and shapes consumed by the tracker."""
    if (not isinstance(value, dict) or value.get("schema") != "cybergym-tracking-v1"
            or not isinstance(value.get("attempts"), list)):
        return False
    seen = set()
    for attempt in value["attempts"]:
        if not isinstance(attempt, dict):
            return False
        attempt_id = attempt.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id.strip() or attempt_id in seen:
            return False
        seen.add(attempt_id)
        if "unchanged_snapshots" in attempt and (
                type(attempt["unchanged_snapshots"]) is not int or attempt["unchanged_snapshots"] <= 0):
            return False
        if "performance_warning_keys" in attempt and (
                not isinstance(attempt["performance_warning_keys"], list)
                or any(not isinstance(key, str) for key in attempt["performance_warning_keys"])):
            return False
        if "evaluator_infrastructure_diagnostics" in attempt:
            diagnostics = attempt["evaluator_infrastructure_diagnostics"]
            if not isinstance(diagnostics, dict):
                return False
            if "warned_error_codes" in diagnostics and (
                    not isinstance(diagnostics["warned_error_codes"], list)
                    or any(not isinstance(code, str) for code in diagnostics["warned_error_codes"])):
                return False
        for field in ("verifier_timings", "evaluator_timings"):
            if field in attempt:
                timings = attempt[field]
                if not isinstance(timings, dict) or ("active_calls" in timings and (
                        not isinstance(timings["active_calls"], list)
                        or any(not isinstance(call, dict) for call in timings["active_calls"]))):
                    return False
    return True


def read_monitor_previous_snapshot(root: Path | str) -> tuple[dict, dict]:
    """Larger bounded host-only cache; task/evaluator JSON limits stay unchanged."""
    root = Path(root).absolute()
    path = root / "monitor/latest.json"
    diagnostic = {"schema": "gv-monitor-previous-read-v1", "available": False,
        "reason_code": "missing", "limit_bytes": MAX_MONITOR_PREVIOUS_BYTES,
        "bytes_observed": 0}
    try:
        if root.resolve() != root or not root.is_dir() or not _safe_file(path, root):
            # Missing on the first tick is normal; unsafe ancestors/files are not.
            try:
                path.lstat()
            except FileNotFoundError:
                return {}, diagnostic
            diagnostic["reason_code"] = "unsafe"
            return {}, diagnostic
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                             | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                diagnostic["reason_code"] = "unsafe"
                return {}, diagnostic
            # Do not rely on a prior stat: atomic replacement/growth may race us.
            raw = stream.read(MAX_MONITOR_PREVIOUS_BYTES + 1)
        diagnostic["bytes_observed"] = len(raw)
        if len(raw) > MAX_MONITOR_PREVIOUS_BYTES:
            diagnostic["reason_code"] = "oversized"
            return {}, diagnostic
        value = json.loads(raw)
        if not _valid_monitor_previous(value):
            diagnostic["reason_code"] = "malformed"
            return {}, diagnostic
        diagnostic.update(available=True, reason_code=None)
        return value, diagnostic
    except (ValueError, UnicodeError, RecursionError):
        diagnostic["reason_code"] = "malformed"
    except OSError:
        diagnostic["reason_code"] = "unreadable"
    return {}, diagnostic


def _jsonl(path: Path, root: Path, *, limit_bytes: int = MAX_JSONL_BYTES) -> tuple[list[dict], bool]:
    """Bounded tail read; missing or partial telemetry is explicitly unknown."""
    if not path.exists():
        return [], False
    if not _safe_file(path, root):
        return [], True
    rows, partial = [], False
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            partial = size > limit_bytes
            if partial:
                stream.seek(size - limit_bytes)
                stream.readline()
            data = stream.read(limit_bytes)
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


def project_progress(value: Any) -> dict:
    """Only authenticated native job metadata; never requests/results/secrets."""
    if (not isinstance(value, dict) or value.get("schema") != PROGRESS_SCHEMA
            or not isinstance(value.get("jobs"), list) or len(value["jobs"]) > 10000
            or _number(value.get("checked_unix")) is None):
        raise ValueError("invalid verification progress metadata")
    jobs, seen = [], set()
    for job in value["jobs"]:
        if not isinstance(job, dict):
            raise ValueError("invalid verification progress job")
        job_id = job.get("job_id")
        if (not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_:-]{1,128}", job_id)
                or job_id in seen or not isinstance(job.get("tool"), str) or job["tool"] not in VERIFICATION_TOOLS
                or not isinstance(job.get("state"), str) or job["state"] not in JOB_STATES):
            raise ValueError("invalid verification progress job identity")
        seen.add(job_id)
        clean = {key: job[key] for key in ("job_id", "tool", "state")}
        for key in ("created_unix", "started_unix", "updated_unix"):
            clean[key] = _number(job.get(key))
        for key in ("attempts", "reused_count"):
            number = job.get(key)
            clean[key] = number if isinstance(number, int) and not isinstance(number, bool) and number >= 0 else None
        jobs.append(clean)
    return {"schema": PROGRESS_SCHEMA, "checked_unix": value["checked_unix"],
        "journal_available": value.get("journal_available") if isinstance(value.get("journal_available"), bool) else None,
        "metadata_snapshot_unix": _number(value.get("metadata_snapshot_unix")),
        "jobs": jobs, "active_job_ids": [job["job_id"] for job in jobs if job["state"] in {"queued", "running"}]}


def _backend_activity(rows: list[dict], outer_calls: list[dict], now: float, partial: bool) -> dict:
    """Observed CLI stages only; queue versus inference cannot be recovered."""
    calls = {}
    for row in sorted(rows, key=lambda item: _number(item.get("ts")) or 0):
        if (row.get("type") != "backend_phase" or not isinstance(row.get("phase"), str)
                or row["phase"] not in BACKEND_PHASES):
            continue
        call_id, attempt, ts = row.get("call_id"), row.get("attempt"), _number(row.get("ts"))
        if (not isinstance(call_id, str) or not re.fullmatch(r"[A-Za-z0-9_:-]{1,128}", call_id)
                or not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0 or ts is None):
            continue
        key = (call_id, attempt)
        call = calls.setdefault(key, {"backend_call_id": call_id, "attempt": attempt,
            "started_unix": ts, "phase": "exec_dispatch", "phase_started_unix": ts,
            "wait_category": "unknown", "process_running": None,
            "latest_unix": ts, "last_heartbeat_unix": None, "last_cli_event_unix": None,
            "last_cli_event": None, "last_item_kind": None, "phase_counts": {},
            "completed_tool_items": {}, "command_stats": {"available": False, "partial": True}})
        phase = row["phase"]
        if phase != call["phase"]:
            call["phase_started_unix"] = ts
        call.update(phase=phase, latest_unix=ts)
        call["phase_counts"][phase] = call["phase_counts"].get(phase, 0) + 1
        role = row.get("role")
        if isinstance(role, str) and role in BACKEND_ROLES:
            call["role"] = role
        wait_category = row.get("wait_category")
        if isinstance(wait_category, str):
            call["wait_category"] = wait_category if wait_category in BACKEND_WAIT_CATEGORIES else "unknown"
        for field in ("child_pid", "stdout_bytes", "stderr_bytes", "cli_events", "active_tool_items",
                      "unparsed_stdout_lines", "retry_number", "retry_limit"):
            number = row.get(field)
            if isinstance(number, int) and not isinstance(number, bool) and number >= 0:
                call[field] = number
        for field in ("item_state_complete", "reader_failed", "streaming"):
            if isinstance(row.get(field), bool):
                call[field] = row[field]
        stats = row.get("command_stats")
        if stats is not None:
            valid = (isinstance(stats, dict) and stats.get("available") is True
                and stats.get("scope") == "cli_attempt_prefix"
                and all(isinstance(stats.get(field), int) and not isinstance(stats[field], bool)
                        and 0 <= stats[field] < 2 ** 31 for field in COMMAND_STAT_COUNTERS)
                and all(isinstance(stats.get(field), bool) for field in ("partial", "saturated", "truncated")))
            call["command_stats"] = ({"available": True, "scope": "cli_attempt_prefix",
                **{field: stats[field] for field in COMMAND_STAT_COUNTERS},
                "partial": stats["partial"] or stats["saturated"] or stats["truncated"] or partial,
                "saturated": stats["saturated"], "truncated": stats["truncated"] or partial}
                if valid else {"available": False, "partial": True})
        if phase in {"exec_spawned", "exec_heartbeat"}:
            call["process_running"] = True
        elif phase in {"exec_process_exited", "pipe_drain_heartbeat", "exec_returned", "exec_spawn_failed"}:
            call["process_running"] = False
        if phase in {"exec_heartbeat", "pipe_drain_heartbeat"}:
            call["last_heartbeat_unix"] = ts
        if phase == "exec_process_exited":
            call["process_exited_unix"] = ts
        if phase == "backend_backoff_started":
            call["backoff_started_unix"] = ts
            call["backoff_wait_seconds"] = _number(row.get("wait_seconds"))
        if isinstance(row.get("last_cli_event"), str) and row["last_cli_event"] in CLI_EVENT_TYPES:
            call["last_cli_event"] = row["last_cli_event"]
        if phase == "cli_event":
            event_type = row.get("event_type")
            if isinstance(event_type, str) and event_type in CLI_EVENT_TYPES:
                call.update(last_cli_event=event_type, last_cli_event_unix=ts)
            item_kind = row.get("item_kind")
            if isinstance(item_kind, str) and item_kind in CLI_ITEM_KINDS:
                call["last_item_kind"] = item_kind
            duration = _number(row.get("item_duration_seconds"))
            if item_kind == "command_execution":
                metadata = {"available": row.get("command_metadata_available") is True}
                digest = row.get("command_sha256")
                if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
                    metadata["sha256"] = digest
                for field, values in (("category", {"compiler", "build", "test", "other"}),
                                      ("status", {"in_progress", "completed", "failed", "unknown"})):
                    value = row.get("command_" + field)
                    if isinstance(value, str) and value in values:
                        metadata[field] = value
                for source, destination, low, high in (("command_exit_code", "exit_code", -(2 ** 31), 2 ** 31),
                                                     ("command_output_utf8_bytes", "output_utf8_bytes", 0, 2 ** 31)):
                    value = row.get(source)
                    if isinstance(value, int) and not isinstance(value, bool) and low <= value < high:
                        metadata[destination] = value
                for field in ("metadata_truncated", "duplicate_completion"):
                    if isinstance(row.get("command_" + field), bool):
                        metadata[field] = row["command_" + field]
                call["last_command"] = metadata
            if (event_type == "item.completed" and duration is not None
                    and isinstance(item_kind, str) and item_kind in {"command_execution", "mcp_tool_call", "web_search"}
                    and row.get("command_duplicate_completion") is not True):
                item = call["completed_tool_items"].setdefault(item_kind, {"count": 0, "work_seconds_sum": 0.0})
                item["count"] += 1
                item["work_seconds_sum"] = round(item["work_seconds_sum"] + duration, 6)
    for call in calls.values():
        candidates = []
        for outer in outer_calls:
            begin, end = _number(outer.get("started_unix")), _number(outer.get("completed_unix"))
            if (begin is not None and begin <= call["started_unix"] <= (end if end is not None else now)
                    and (call.get("role") is None or outer.get("role") == call["role"])):
                candidates.append(outer)
        roles = {outer.get("role") for outer in candidates}
        if call.get("role") is None and len(roles) == 1:
            call["role"] = next(iter(roles))
        call["outer_role_call_id"] = candidates[0].get("call_id") if len(candidates) == 1 else None
        call["role_association"] = "unique_role_interval" if len(candidates) == 1 else "ambiguous_interval" if candidates else "unknown"
        call.setdefault("role", None)
        for field, timestamp in (("observation_age_seconds", "latest_unix"),
                                 ("heartbeat_age_seconds", "last_heartbeat_unix"),
                                 ("last_cli_event_age_seconds", "last_cli_event_unix"),
                                 ("phase_age_seconds", "phase_started_unix")):
            stamp = call.get(timestamp)
            call[field] = max(0, now - stamp) if stamp is not None else None
        call["elapsed_seconds"] = max(0, now - call["started_unix"])
    return {"available": bool(calls), "partial": partial, "calls": list(calls.values()),
        "provider_queue_vs_inference": "unknown",
        "measurement": "CLI heartbeats/item stages and completed tool-item wall durations only. Tool-work sums may overlap; they are not CPU time or a provider inference breakdown."}


def _transport_calls(rows: list[dict]) -> list[dict]:
    calls = {}
    for row in rows:
        call_id, tool, phase = row.get("call_id"), row.get("tool"), row.get("phase")
        if (not isinstance(call_id, str) or not call_id or tool not in TRANSPORT_TOOLS
                or phase not in {"received", "invoked", "finished", "failed"}):
            continue
        call = calls.setdefault(call_id, {"call_id": call_id, "role": tool, "status": "running"})
        ts = _number(row.get("ts"))
        if phase == "received":
            call.setdefault("started_unix", ts)
        elif phase == "invoked":
            call["invoked_unix"] = ts
            call.setdefault("started_unix", ts)
        else:
            call.update(status="failed" if phase == "failed" else "completed", completed_unix=ts,
                        duration_seconds=_number(row.get("elapsed_seconds")))
    return list(calls.values())


def _solver_wait_cancellations(directory: Path, root: Path) -> dict:
    """Bounded metadata projection of explicit code-mode wait cancellation."""
    sessions = directory / "solver-home" / "sessions"
    if sessions.is_symlink() or not sessions.is_dir() or sessions.resolve() != sessions:
        return {"count_in_window": 0, "partial": False, "latest": None}
    paths = sorted(sessions.glob("**/rollout-*.jsonl"))
    seen, latest, partial = set(), None, len(paths) > 8
    for path in paths[-8:]:
        rows, incomplete = _jsonl(path, root, limit_bytes=MAX_ROLLOUT_BYTES)
        partial |= incomplete
        for row in rows:
            payload = row.get("payload")
            if (row.get("type") != "response_item" or not isinstance(payload, dict)
                    or payload.get("type") != "function_call" or payload.get("name") != "wait"):
                continue
            arguments = payload.get("arguments")
            if not isinstance(arguments, str) or len(arguments) > 4096:
                continue
            try:
                arguments = json.loads(arguments)
            except ValueError:
                continue
            if not isinstance(arguments, dict) or arguments.get("terminate") is not True:
                continue
            call_id = payload.get("call_id")
            if not isinstance(call_id, str) or not call_id or call_id in seen:
                continue
            seen.add(call_id)
            ts = row.get("timestamp")
            try:
                observed = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() if isinstance(ts, str) else None
                observed = _number(observed)
            except ValueError:
                observed = None
            if observed is not None and (latest is None or observed > latest["observed_unix"]):
                latest = {"observed_unix": observed, "event_id_sha256": hashlib.sha256(call_id.encode()).hexdigest()}
    return {"count_in_window": len(seen), "partial": partial, "latest": latest}


def _repeated_finalizations(events: list[dict]) -> list[dict]:
    contexts = {}
    for event in events:
        payload = event.get("payload")
        if event.get("type") != "commit" or not isinstance(payload, dict):
            continue
        binding = payload.get("cybergv_binding")
        if not isinstance(binding, dict) or binding.get("certificate_scope") != "candidate":
            continue
        # Verification epoch / claim UUID changes are not new candidate or
        # evaluator evidence. Do not reveal the actual subject or observation.
        identity = {key: binding.get(key) for key in ("profile_id", "task_sha256", "subject", "observations")}
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        item = contexts.setdefault(signature, {"context_sha256": signature, "count": 0})
        item["count"] += 1
    return list(contexts.values())


def _telemetry_health(directory: Path, root: Path) -> dict:
    errors, partial = [], False
    for name in ("telemetry-mirror-errors.jsonl", "telemetry-progress-errors.jsonl"):
        rows, incomplete = _jsonl(directory / "private" / name, root)
        partial |= incomplete
        for row in rows:
            if row.get("reason") not in {"mirror_failed", "progress_unavailable"}:
                continue
            ts = _number(row.get("ts"))
            if ts is not None:
                errors.append({"reason": row["reason"], "observed_unix": ts})
    latest = max(errors, key=lambda item: item["observed_unix"]) if errors else None
    return {"count_in_window": len(errors), "by_reason": dict(Counter(item["reason"] for item in errors)),
            "partial": partial, "latest": latest}


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


def _adapter_infrastructure_error(output: dict) -> str | None:
    """Recognize complete static adapter faults; never retain raw stdout."""
    stdout = output.get("stdout")
    if not isinstance(stdout, str) or output.get("stdout_truncated") not in (None, False):
        return None
    try:
        encoded = stdout.encode("utf-8")
    except UnicodeError:
        return None
    if len(encoded) > MAX_JSON_BYTES:
        return None
    observed = _number(output.get("stdout_bytes_observed"))
    if observed is not None and observed > len(encoded):
        return None

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate adapter field")
            value[key] = item
        return value

    def invalid_constant(_):
        raise ValueError("non-JSON adapter constant")

    try:
        value = json.loads(stdout, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, RecursionError):
        return None
    if (not isinstance(value, dict) or value.get("schema") != ADAPTER_OBSERVATION_SCHEMA
            or set(value) != {"schema", "error"}):
        return None
    error = value.get("error")
    return error if isinstance(error, str) and error in ADAPTER_INFRASTRUCTURE_ERRORS else None


def _evaluator_infrastructure_diagnostics(events: list[dict], partial: bool) -> dict:
    faults = {}
    for row in events:
        payload = row.get("payload", {})
        if (row.get("type") != "tool_result" or not isinstance(payload, dict)
                or payload.get("tool") != "cyber_evaluator"):
            continue
        call_id, output = payload.get("tool_use_id"), payload.get("output")
        if not isinstance(call_id, str) or not call_id or not isinstance(output, dict):
            continue
        error = _adapter_infrastructure_error(output)
        if error is not None:
            faults[call_id] = {"error_code": error, "observed_unix": _number(row.get("ts"))}
    latest = max(faults.values(), key=lambda value: value["observed_unix"] or 0, default=None)
    return {"count": len(faults), "by_error": dict(sorted(Counter(
        fault["error_code"] for fault in faults.values()).items())),
        "latest": latest, "partial": partial, "scope": "retained_event_window"}


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
    stable["jobs"] = [{key: job.get(key) for key in ("job_id", "state", "attempts")}
        for job in value.get("verification_progress", {}).get("jobs", [])]
    stable["solver_output"] = {key: value.get("solver_activity", {}).get(key)
                               for key in ("available", "log_bytes")}
    stable["backend_cli"] = [{
        **{key: call.get(key) for key in ("backend_call_id", "attempt", "cli_events", "active_tool_items")},
        "completed_commands": call.get("command_stats", {}).get("completed_items"),
        "completed_tools": {kind: item.get("count")
                            for kind, item in call.get("completed_tool_items", {}).items()},
    } for call in value.get("backend_activity", {}).get("calls", [])]
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def _solver_activity(directory: Path, root: Path, now: float) -> dict:
    """Output activity only; never read or publish solver transcript contents."""
    path = directory / "solver.jsonl"
    try:
        info = path.stat(follow_symlinks=False) if _safe_file(path, root) else None
        if info is not None and not stat.S_ISREG(info.st_mode):
            info = None
    except OSError:
        info = None
    return {"available": info is not None,
            "log_bytes": info.st_size if info else None,
            "last_log_update_unix": info.st_mtime if info else None,
            "log_age_seconds": max(0, now - info.st_mtime) if info else None,
            "measurement": "Log output and CLI items are observed activity, not successful task progress or CPU time."}


def _source_signature(status_path: Path, root: Path) -> dict:
    """Finished attempts normally never change; avoid rereading 1507 ledgers."""
    state = status_path.parent / "private" / "state"
    signatures = {}
    for path in (status_path, status_path.parent / "solver.jsonl", state / "role_calls.jsonl", state / "backend_usage.jsonl", state / "events.jsonl",
                 state / "transport_calls.jsonl", status_path.parent / "private/native-progress.json",
                 status_path.parent / "private/telemetry-mirror-calls.jsonl",
                 status_path.parent / "private/telemetry-mirror-errors.jsonl",
                 status_path.parent / "private/telemetry-progress-errors.jsonl"):
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
    transport_rows, transport_partial = _jsonl(state / "transport_calls.jsonl", root)
    transport = _transport_calls(transport_rows)
    raw_progress = _json(status_path.parent / "private/native-progress.json", root)
    try:
        jobs = {"available": True, **project_progress(raw_progress)}
    except ValueError:
        jobs = {"available": False, "jobs": [], "active_job_ids": [], "checked_unix": None,
                "journal_available": None, "metadata_snapshot_unix": None}
    jobs["age_seconds"] = max(0, now - jobs["checked_unix"]) if jobs["checked_unix"] is not None else None
    snapshot_unix = jobs["metadata_snapshot_unix"]
    jobs["metadata_snapshot_age_seconds"] = max(0, now - snapshot_unix) if snapshot_unix is not None else None
    for job in jobs["jobs"]:
        began = job.get("started_unix") if job.get("started_unix") is not None else job.get("created_unix")
        job["age_seconds"] = max(0, now - began) if began is not None and job["state"] in {"queued", "running"} else None
    mirror_rows, mirror_partial = _jsonl(status_path.parent / "private/telemetry-mirror-calls.jsonl", root)
    mirror_calls = []
    for index, row in enumerate(mirror_rows):
        duration, end = _number(row.get("duration_seconds")), _number(row.get("ts"))
        if duration is not None and end is not None and row.get("phase") in {"periodic", "final", "cleanup"}:
            mirror_calls.append({"call_id": str(index), "role": row["phase"],
                "status": "failed" if row.get("status") == "failed" else "completed",
                "started_unix": max(0, end - duration), "completed_unix": end, "duration_seconds": duration})
    calls = _role_calls(role_rows)
    source = "role_calls" if calls else "backend_usage_completed_only"
    if not calls:
        calls = _usage_calls(usage_rows)
    backend = _backend_activity(role_rows, calls, now, role_partial)
    mirror_ends = [call["completed_unix"] for call in mirror_calls
                   if call["status"] == "completed"]
    backend["last_metadata_mirror_unix"] = max(mirror_ends) if mirror_ends else None
    backend["metadata_mirror_age_seconds"] = (max(0, now - max(mirror_ends))
                                               if mirror_ends else None)
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
        backend_activity=backend,
        solver_activity=_solver_activity(status_path.parent, root, now),
        evaluator_timings={"available": bool(events), "partial": event_partial,
                           **summarize_calls(evaluator, now)},
        evaluator_infrastructure_diagnostics=_evaluator_infrastructure_diagnostics(events, event_partial),
        verification_progress=jobs,
        transport_timings={"available": bool(transport_rows), "partial": transport_partial,
                           **summarize_calls(transport, now)},
        mirror_timings={"available": bool(mirror_rows), "partial": mirror_partial,
                        **summarize_calls(mirror_calls, now)},
        solver_wait_cancellations=_solver_wait_cancellations(status_path.parent, root),
        repeated_finalization_contexts=_repeated_finalizations(events),
        telemetry_health=_telemetry_health(status_path.parent, root),
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


def _performance_warnings(value: dict, old: dict, now: float) -> list[dict]:
    """Early inspection signals, not deadlines, verdicts, or lifecycle actions."""
    seen = set(old.get("performance_warning_keys", []))
    warnings = []
    base = {key: value.get(key) for key in ("task_id", "attempt_id", "stage")}

    def warn(kind, key, **metadata):
        token = kind + ":" + key
        if token not in seen:
            seen.add(token)
            warnings.append({"type": kind, "ts": now, **base, **metadata,
                "diagnostic_only": True, "lifecycle_action": None})

    if _number(value.get("completed_unix")) is None:
        health = value["telemetry_health"]
        if health["latest"] is not None:
            warn("telemetry_infrastructure_warning", str(health["latest"]["observed_unix"]),
                 error_codes=sorted(health["by_reason"]), counts=health["by_reason"],
                 meaning="Native progress or forensic mirroring failed. Telemetry is incomplete; task/job state and official score are unchanged.")
        for kind, summary in (("long_role_call_warning", value["verifier_timings"]),
                              ("long_service_call_warning", value["transport_timings"])):
            for call in summary["active_calls"]:
                age = _number(call.get("age_seconds"))
                if age is not None and age >= LONG_CALL_SECONDS:
                    associated = [_backend_warning_metadata(item) for item in value["backend_activity"]["calls"]
                                  if item["outer_role_call_id"] == call.get("call_id")]
                    warn(kind, str(call.get("call_id")), role=call.get("role"), age_seconds=age,
                         backend_activity=associated if kind == "long_role_call_warning" else [],
                         meaning="Long outstanding call. Provider/tool activity is unknown; this is not proof of a hung shell, timeout, or failure.")
        _backend_warnings(value["backend_activity"], warn)
        progress = value["verification_progress"]
        if progress["journal_available"] is False:
            warn("verification_journal_unavailable_warning", "journal-unavailable",
                 meaning="The verification journal is unavailable. Inspect storage/integrity diagnostics; this is not an official task failure, a retry instruction, or permission to accept an unretained result.")
        if progress["available"] and progress["age_seconds"] >= LONG_CALL_SECONDS:
            warn("stale_native_progress_warning", str(progress["checked_unix"]), age_seconds=progress["age_seconds"],
                 meaning="Native progress polling is stale; inspect monitoring transport, not the task answer.")
        for job in progress["jobs"]:
            if job["state"] in {"failed", "interrupted"}:
                warn("verification_job_state_warning", job["job_id"] + ":" + job["state"],
                     job_id=job["job_id"], job_state=job["state"],
                     meaning="The retained job is no longer executing normally. Inspect its existing result; neither waiting cancellation nor this diagnostic is an official task failure or an automatic retry instruction.")
            age = _number(job.get("age_seconds"))
            if age is not None and age >= LONG_CALL_SECONDS:
                warn("long_verification_job_warning", job["job_id"], job_id=job["job_id"], job_state=job["state"],
                     age_seconds=age, meaning="Query this existing job and its retained stage evidence; do not create a replacement verification attempt merely because waiting is long.")
            if job.get("attempts") is not None and job["attempts"] >= 3:
                warn("verification_job_redo_warning", job["job_id"] + ":" + str(job["attempts"]),
                     job_id=job["job_id"], attempts=job["attempts"],
                     meaning="The same verification job has executed repeatedly. Inspect rejection/new evidence before another explicit retry. Reused polls are not executions.")
        for context in value["repeated_finalization_contexts"]:
            if context["count"] >= 3:
                warn("repeated_finalization_warning", context["context_sha256"] + ":" + str(context["count"]),
                     count=context["count"], context_sha256=context["context_sha256"],
                     meaning="Multiple final claims reuse the same candidate and evaluator evidence; epoch/claim UUID changes alone are not progress.")
        cancelled = value["solver_wait_cancellations"]
        active = (progress["active_job_ids"] or value["verifier_timings"]["active_calls"]
                  or value["transport_timings"]["active_calls"])
        if cancelled["latest"] is not None and active:
            warn("solver_wait_cancelled_while_verification_active", cancelled["latest"]["event_id_sha256"],
                 count_in_window=cancelled["count_in_window"], partial=cancelled["partial"],
                 meaning="The client explicitly terminated a code-mode waiting cell while GV activity exists. Waiting cancellation is not task/job cancellation or a failed verdict; query the stable job id.")
    value["performance_warning_keys"] = sorted(seen)
    return warnings


def _backend_warning_metadata(call: dict) -> dict:
    return {key: call.get(key) for key in ("backend_call_id", "attempt", "role",
        "outer_role_call_id", "role_association", "phase", "wait_category", "process_running",
        "observation_age_seconds", "heartbeat_age_seconds", "last_cli_event_age_seconds",
        "last_cli_event", "last_item_kind", "active_tool_items", "item_state_complete")}


def _backend_warnings(activity: dict, warn) -> None:
    """Observed client stages, never a guessed provider phase or timeout."""
    mirror_age = activity["metadata_mirror_age_seconds"]
    fresh_copy = mirror_age is not None and mirror_age <= FRESH_METADATA_SECONDS
    for call in activity["calls"]:
        key = call["backend_call_id"] + ":" + str(call["attempt"])
        metadata = _backend_warning_metadata(call)
        counts = call["phase_counts"]
        if counts.get("backend_transient_failure") or counts.get("backend_backoff_started"):
            warn("backend_transient_retry_warning", key, **metadata,
                 retry_number=call.get("retry_number"), retry_limit=call.get("retry_limit"),
                 backoff_wait_seconds=call.get("backoff_wait_seconds"),
                 meaning="The client recorded a transient failure/retry or backoff. This wall time is not task reasoning time; provider queue versus inference remains unknown. No retry is triggered by this alert.")
        if call["phase"] == "backend_backoff_started":
            wait = call.get("backoff_wait_seconds")
            age = call["phase_age_seconds"]
            if wait is not None and age > wait + 60 and fresh_copy:
                warn("backend_backoff_overrun_warning", key, **metadata,
                     backoff_wait_seconds=wait, observed_backoff_age_seconds=age,
                     meaning="A fresh metadata copy still shows backoff beyond its declared wait plus 60 seconds. Inspect client scheduling; this is not a model/task timeout or cancellation.")
        if counts.get("pipe_drain_heartbeat"):
            warn("backend_pipe_drain_warning", key, **metadata,
                 meaning="The CLI exited but its output readers remained pending for at least one drain heartbeat. Unread output versus a descendant holding a pipe open is unknown; this is not provider inference or proof of a hung target.")
        if call.get("streaming") is not True or call["process_running"] is not True:
            continue
        observed_age = call["observation_age_seconds"]
        if observed_age >= BACKEND_HEARTBEAT_STALE_SECONDS:
            if fresh_copy:
                warn("backend_heartbeat_stale_warning", key, **metadata,
                     meaning="A fresh metadata copy contains no recent client observation/heartbeat. Client liveness and its underlying wait are unknown; inspect instrumentation/process health without cancelling or failing the task.")
            else:
                warn("backend_observability_stale_warning", key, **metadata,
                     metadata_mirror_age_seconds=mirror_age,
                     meaning="Host client metadata is old and no fresh successful mirror confirms the native state. Missing native heartbeats or a hung client cannot be inferred.")
            continue
        heartbeat_age = call["heartbeat_age_seconds"]
        cli_age = call["last_cli_event_age_seconds"]
        if cli_age is None:
            cli_age = call["elapsed_seconds"]
        if (heartbeat_age is not None and heartbeat_age <= FRESH_METADATA_SECONDS
                and cli_age >= LONG_CALL_SECONDS):
            tool_active = call["wait_category"] == "cli_tool_item_active"
            warn("backend_tool_item_long_warning" if tool_active else "backend_cli_output_stale_warning",
                 key, **metadata,
                 meaning="Recent client heartbeats show the CLI has not exited, but no new CLI event has been observed for 600 seconds. The CLI tool-item state is observational, not proof of a live or hung shell; provider queue, inference, and transport waits remain unknown.")


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
        if (old and "evaluator_infrastructure_diagnostics" in old
                and _number(status.get("completed_unix")) is not None
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
        alerts.extend(_performance_warnings(value, old, now))
        base = {key: value[key] for key in ("task_id", "attempt_id", "stage", "reason")}
        if _number(value["completed_unix"]) is None and value["unchanged_snapshots"] == 3:
            alerts.append({"type": "unchanged_progress_warning", "ts": now, **base,
                "meaning": "Three unchanged observations; inspect trace. This is not a timeout, failure, or restart instruction."})
        if value["official_pass"] is False and _number(value["completed_unix"]) is not None and old.get("official_pass") is not False:
            alerts.append({"type": "official_task_not_passed", "ts": now, **base})
        if value["outcome"] in {"infrastructure_error", "runner_error"} and old.get("outcome") != value["outcome"]:
            alerts.append({"type": "infrastructure_error", "ts": now, **base})
        diagnostics = dict(value["evaluator_infrastructure_diagnostics"])
        value["evaluator_infrastructure_diagnostics"] = diagnostics
        previous_diagnostics = old.get("evaluator_infrastructure_diagnostics", {})
        warned = previous_diagnostics.get("warned_error_codes", [])
        if not isinstance(warned, list):
            warned = []
        seen = {code for code in warned if isinstance(code, str) and code in ADAPTER_INFRASTRUCTURE_ERRORS}
        new_errors = set(diagnostics["by_error"]) - seen
        diagnostics["warned_error_codes"] = sorted(seen | set(diagnostics["by_error"]))
        if new_errors:
            alerts.append({"type": "evaluator_infrastructure_warning", "ts": now,
                **{key: value[key] for key in ("task_id", "attempt_id", "stage")},
                "error_codes": sorted(new_errors),
                "counts": {code: diagnostics["by_error"][code] for code in sorted(new_errors)},
                "meaning": "Static adapter infrastructure diagnostic, not a candidate negative, official task result, timeout, or lifecycle instruction."})
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
    previous, previous_diagnostic = read_monitor_previous_snapshot(root)
    try:
        value = snapshot(root, now=now, previous=previous)
    except (TypeError, KeyError, AttributeError):
        if not previous:
            raise
        # A cache is only a hint. Recompute from native metadata; errors from
        # that fresh read must still propagate, never become official results.
        value = snapshot(root, now=now, previous={})
        previous_diagnostic = {**previous_diagnostic, "available": False, "reason_code": "malformed"}
    value["monitor_previous_snapshot"] = previous_diagnostic
    if not previous_diagnostic["available"] and previous_diagnostic["reason_code"] != "missing":
        value["alerts"].append({"type": "monitor_previous_snapshot_oversized_warning"
            if previous_diagnostic["reason_code"] == "oversized"
            else "monitor_previous_snapshot_unavailable_warning",
            "ts": value["checked_unix"], "diagnostic_only": True, "lifecycle_action": None,
            **{key: previous_diagnostic[key] for key in
               ("available", "reason_code", "limit_bytes", "bytes_observed")}})
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
