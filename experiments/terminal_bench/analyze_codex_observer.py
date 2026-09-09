"""Read-only accounting gate for the common, non-mutating CLI observer.

This audits the observer ledger, not verdict quality, process-wall latency,
source isolation or completeness of the surrounding evaluation process.
An outer run manifest must establish those independently.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "gv-codex-attempt-observer-v1"
REPORT_SCHEMA = "gv-codex-observer-accounting-v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_KINDS = {
    "codex_logical_call_started", "codex_logical_call_finished",
    "codex_cli_attempt_started", "codex_cli_attempt_finished",
}
_TOKENS = (
    "input_tokens", "cached_input_tokens", "uncached_input_tokens",
    "cache_write_input_tokens", "output_tokens", "reasoning_tokens",
)
_IDENTITY = (
    "observer_id", "seam", "call_id", "role", "model", "state_dir",
    "attribution", "context", "effort", "service_tier",
)
_COMMAND = (
    "attempt_id", "attempt_index", "command_sha256", "cwd",
    "command_model", "command_effort", "command_service_tier",
)


def _uint(value: Any) -> bool:
    return type(value) is int and value >= 0


def _digest(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX.fullmatch(value))


def _execution_roots(value, contexts):
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(contexts):
        raise ValueError("execution cwd mapping must bind every known state exactly")
    result = {}
    for state, roots in value.items():
        if not isinstance(roots, (list, tuple)) or not 1 <= len(roots) <= 32:
            raise ValueError("execution cwd roots must be a bounded nonempty list or tuple")
        checked = []
        for root in roots:
            if (not isinstance(root, str) or not os.path.isabs(root)
                    or os.path.realpath(root) != root or root in checked):
                raise ValueError("execution cwd roots must be distinct canonical absolute paths")
            checked.append(root)
        result[state] = tuple(checked)
    return result


_AUTHORITY_SCRATCH_ROLES = frozenset({"obligations", "critic"})


def _authority_scratch(value):
    if value is None:
        return None
    if (not isinstance(value, str) or not os.path.isabs(value)
            or not value.endswith("-")
            or os.path.realpath(os.path.dirname(value)) != os.path.dirname(value)):
        raise ValueError(
            "authority scratch prefix must be a canonical absolute path prefix")
    return value


def analyze_rows(
    rows: Iterable[Mapping[str, Any]], *,
    context_by_cwd: Mapping[str, Mapping[str, Any]],
    model: str, effort: str, service_tier: str,
    session_summary: Mapping[str, Any] | None = None,
    execution_cwds_by_state: Mapping[str, Iterable[str]] | None = None,
    authority_scratch_prefix: str | None = None,
) -> dict[str, Any]:
    """Reject missing lifecycle events, attribution and actual-CLI profile drift.

    ``context_by_cwd`` and the expected profile come from the operator's pinned
    run plan, never from the ledger being checked. A raised logical call does
    not erase its known token charges; missing usage never becomes zero.

    ``authority_scratch_prefix`` names the operator's temporary-directory
    prefix under which the verifier compiles a final contract through its
    candidate-blind authority boundary; only compiler and Critic seats may
    run there, each in a direct child of that prefix.
    """
    execution_cwds_by_state = _execution_roots(
        execution_cwds_by_state, context_by_cwd)
    authority_scratch_prefix = _authority_scratch(authority_scratch_prefix)
    authority_scratch_attempts = 0
    findings: set[str] = set()
    calls: dict[str, dict[str, tuple[int, Mapping[str, Any]]]] = {}
    attempts: dict[str, dict[str, tuple[int, Mapping[str, Any]]]] = {}
    identities = set()
    totals = {key: {"known_sum": 0, "unknown_attempts": 0} for key in _TOKENS}
    profile = {"model": model, "effort": effort, "service_tier": service_tier}
    event_counts = {
        "logical_calls_started": 0, "logical_calls_finished": 0,
        "attempts_started": 0, "attempts_finished": 0,
    }

    def same(a, b, fields, code):
        if any(key not in a or key not in b or a[key] != b[key] for key in fields):
            findings.add(code)

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            findings.add("non_object_record")
            continue
        kind = row.get("record_type")
        if (row.get("schema") != SCHEMA or not isinstance(kind, str)
                or kind not in _KINDS):
            findings.add("unknown_record_schema_or_type")
            continue
        if (not isinstance(row.get("observer_id"), str)
                or not row["observer_id"] or not isinstance(row.get("seam"), str)
                or row.get("seam") not in {
                    "legacy-subprocess-run", "run-codex-process"}):
            findings.add("invalid_observer_identity")
        else:
            identities.add((row["observer_id"], row["seam"]))
        if any(row.get(key) != expected or key not in row
               for key, expected in profile.items()):
            findings.add("declared_profile_mismatch")
        state = row.get("state_dir")
        if (not isinstance(state, str) or state not in context_by_cwd
                or row.get("attribution") != "attributed"
                or "context" not in row
                or row["context"] != context_by_cwd.get(state)):
            findings.add("context_attribution_mismatch")
        if not isinstance(row.get("role"), str) or not row["role"]:
            findings.add("missing_role")
        if not _uint(row.get("started_at_unix_ns")):
            findings.add("invalid_start_time")
        event = "started" if kind.endswith("_started") else "finished"
        event_counts[("logical_calls_" if kind.startswith("codex_logical")
                      else "attempts_") + event] += 1
        if event == "started":
            if not _uint(row.get("started_at_monotonic_ns")):
                findings.add("invalid_monotonic_start")
        elif (not _uint(row.get("finished_at_unix_ns"))
              or not _uint(row.get("duration_ns"))):
            findings.add("invalid_terminal_time")
        call_id = row.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            findings.add("missing_call_id")
            continue
        if kind.startswith("codex_logical"):
            bucket = calls.setdefault(call_id, {})
        else:
            attempt_id = row.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id:
                findings.add("missing_attempt_id")
                continue
            bucket = attempts.setdefault(attempt_id, {})
            if not _uint(row.get("attempt_index")) or row["attempt_index"] < 1:
                findings.add("invalid_attempt_index")
            if not _digest(row.get("command_sha256")):
                findings.add("invalid_command_digest")
            allowed_cwds = (state,)
            if execution_cwds_by_state is not None:
                allowed_cwds = (execution_cwds_by_state.get(state, ())
                                if isinstance(state, str) else ())
            cwd = row.get("cwd")
            in_authority_scratch = (
                authority_scratch_prefix is not None
                and isinstance(cwd, str)
                and cwd.startswith(authority_scratch_prefix)
                and "/" not in cwd[len(authority_scratch_prefix):]
                and len(cwd) > len(authority_scratch_prefix)
                and row.get("role") in _AUTHORITY_SCRATCH_ROLES)
            if in_authority_scratch:
                if kind.endswith("_started"):
                    authority_scratch_attempts += 1
            elif not isinstance(cwd, str) or cwd not in allowed_cwds:
                findings.add("execution_cwd_not_operator_bound")
            if any(row.get("command_" + key) != expected
                   or "command_" + key not in row
                   for key, expected in profile.items()):
                findings.add("actual_cli_profile_mismatch")
        if event in bucket:
            findings.add("duplicate_lifecycle_event")
        else:
            bucket[event] = (index, row)

    if len(identities) != 1:
        findings.add("observer_identity_not_unique")
    session_findings = set()
    if not isinstance(session_summary, Mapping):
        session_findings.add("missing_independent_session_summary")
    else:
        if session_summary.get("schema") != "gv-codex-attempt-observer-summary-v1":
            session_findings.add("invalid_session_summary_schema")
        if (len(identities) != 1 or any(
                session_summary.get("observer_id") != observer_id
                or session_summary.get("seam") != seam
                for observer_id, seam in identities)):
            session_findings.add("session_observer_identity_mismatch")
        for key, count in event_counts.items():
            if not _uint(session_summary.get(key)) or session_summary[key] != count:
                session_findings.add("session_event_count_mismatch")
        for key in ("write_failures", "active_calls"):
            if not _uint(session_summary.get(key)) or session_summary[key] != 0:
                session_findings.add("observer_session_unhealthy")
        if (session_summary.get("closed") is not True
                or session_summary.get("restoration_failed") is not False):
            session_findings.add("observer_session_not_cleanly_closed")
        if (not _digest(session_summary.get("ledger_sha256"))
                or not _uint(session_summary.get("ledger_bytes"))):
            session_findings.add("missing_session_ledger_identity")
    findings.update(session_findings)
    if not calls or not attempts:
        findings.add("no_observed_model_work")
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    unknown_usage = 0
    billed_threads = set()
    for pair in attempts.values():
        if set(pair) != {"started", "finished"}:
            findings.add("incomplete_attempt_lifecycle")
        start = pair.get("started")
        end = pair.get("finished")
        if start is not None:
            grouped.setdefault(start[1]["call_id"], []).append(start)
        if start is not None and end is not None:
            if start[0] >= end[0]:
                findings.add("attempt_finished_before_start")
            same(start[1], end[1], (*_IDENTITY, *_COMMAND,
                 "started_at_unix_ns"), "attempt_identity_changed")
        row = end[1] if end is not None else {}
        core_usage_valid = (row.get("terminal_status") == "returned"
                       and row.get("process_started") is True
                       and row.get("usage_complete") is True
                       and row.get("usage_status") == "complete"
                       and all(_uint(row.get(key)) for key in _TOKENS
                               if key != "cache_write_input_tokens"))
        if core_usage_valid:
            core_usage_valid = (
                row["cached_input_tokens"] <= row["input_tokens"]
                and row["uncached_input_tokens"] ==
                row["input_tokens"] - row["cached_input_tokens"]
                and row["reasoning_tokens"] <= row["output_tokens"]
                and isinstance(row.get("thread_id"), str)
                and bool(row["thread_id"]))
        if not core_usage_valid or not _uint(row.get("cache_write_input_tokens")):
            unknown_usage += 1
            findings.add("incomplete_or_invalid_usage")
        if core_usage_valid:
            if row["thread_id"] in billed_threads:
                findings.add("thread_id_reused_across_attempts")
            billed_threads.add(row["thread_id"])
        for key in _TOKENS:
            # Retain each independently known bucket even when cache-write
            # telemetry is missing. Inconsistent core counters are not trusted.
            if core_usage_valid and _uint(row.get(key)):
                totals[key]["known_sum"] += row[key]
            else:
                totals[key]["unknown_attempts"] += 1
        if end is not None:
            status = row.get("terminal_status")
            if not isinstance(status, str) or status not in {
                    "returned", "timeout", "launch_error", "interrupted"}:
                findings.add("invalid_attempt_terminal_status")
            if status == "returned":
                if row.get("process_started") is not True:
                    findings.add("contradictory_process_start_status")
                if type(row.get("returncode")) is not int:
                    findings.add("missing_returncode")
                if not _digest(row.get("stdout_sha256")) or not _uint(row.get("stdout_bytes")):
                    findings.add("invalid_stdout_identity")
            elif row.get("usage_complete") is True:
                findings.add("interrupted_usage_cannot_be_complete")
            if status == "timeout" and row.get("process_started") is not True:
                findings.add("contradictory_process_start_status")
            if status == "launch_error" and row.get("process_started") is not False:
                findings.add("contradictory_process_start_status")
            if status == "interrupted" and (
                    "process_started" not in row
                    or (row["process_started"] is not None
                        and type(row["process_started"]) is not bool)):
                findings.add("invalid_process_start_status")
            if status != "returned" and row.get("returncode") is not None:
                findings.add("unexpected_returncode_for_raised_process")

    returned = raised = 0
    for call_id, pair in calls.items():
        if set(pair) != {"started", "finished"}:
            findings.add("incomplete_logical_call_lifecycle")
        entries = sorted(grouped.get(call_id, []))
        if [row.get("attempt_index") for _, row in entries] != list(range(1, len(entries) + 1)):
            findings.add("non_contiguous_attempt_indices")
        start, end = pair.get("started"), pair.get("finished")
        if start is not None and end is not None:
            if start[0] >= end[0]:
                findings.add("logical_call_finished_before_start")
            same(start[1], end[1], (*_IDENTITY, "started_at_unix_ns"),
                 "logical_call_identity_changed")
            previous_end = None
            for pos, attempt in entries:
                same(start[1], attempt, _IDENTITY, "attempt_call_identity_mismatch")
                terminal = attempts[attempt["attempt_id"]].get("finished")
                if not start[0] < pos < end[0] or (terminal is not None and terminal[0] >= end[0]):
                    findings.add("attempt_outside_logical_call")
                if previous_end is not None and previous_end >= pos:
                    findings.add("overlapping_attempts_in_one_logical_call")
                previous_end = terminal[0] if terminal is not None else None
        if end is not None:
            row = end[1]
            if not _uint(row.get("attempt_count")) or row["attempt_count"] != len(entries):
                findings.add("logical_attempt_count_mismatch")
            status = row.get("terminal_status")
            returned += status == "returned"
            raised += status == "raised"
            if status == "returned" and not entries:
                findings.add("returned_logical_call_without_attempt")
            if not isinstance(status, str) or status not in {"returned", "raised"}:
                findings.add("invalid_logical_terminal_status")
    if set(grouped) - set(calls):
        findings.add("attempt_without_logical_call")
    return {
        "schema": REPORT_SCHEMA,
        "accounting_complete": not findings,
        "observer_session_complete": not session_findings,
        "ledger_identity_complete": None,
        "findings": sorted(findings),
        "observed_event_counts": event_counts,
        "authority_scratch_attempts": authority_scratch_attempts,
        "call_count": len(calls), "attempt_count": len(attempts),
        "unknown_usage_attempts": unknown_usage,
        "logical_calls_returned": returned, "logical_calls_raised": raised,
        "all_logical_calls_returned": bool(calls) and returned == len(calls),
        "tokens": totals,
        "expected_profile": profile,
        "quality_acceptance": None, "full_e2e_wall_seconds": None,
        "notes": [
            "known_sum is not a complete total when accounting_complete=false",
            "reasoning tokens are included in output tokens; do not add them again",
            "failed logical calls retain their charges; this report does not establish result quality",
            "outer source/input/process lifecycle and isolation checks are still required",
        ],
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("non-finite JSON number")


def analyze_file(path: str | Path, **kwargs) -> dict[str, Any]:
    """Read one retained ledger without editing it or emitting its raw content."""
    with Path(path).open("rb") as handle:
        raw = handle.read(64 * 1024 * 1024 + 1)
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("observer ledger exceeds analysis limit")
    if not raw.endswith(b"\n"):
        raise ValueError("observer ledger has no complete final newline")
    rows = [json.loads(line, object_pairs_hook=_unique_object,
                       parse_constant=_reject_constant)
            for line in raw.decode("utf-8", errors="strict").split("\n")[:-1]]
    report = analyze_rows(rows, **kwargs)
    report["ledger_sha256"] = hashlib.sha256(raw).hexdigest()
    report["ledger_bytes"] = len(raw)
    summary = kwargs.get("session_summary")
    identity_matches = (isinstance(summary, Mapping)
                        and summary.get("ledger_sha256") == report["ledger_sha256"]
                        and type(summary.get("ledger_bytes")) is int
                        and summary["ledger_bytes"] == len(raw))
    report["ledger_identity_complete"] = identity_matches
    if not identity_matches:
        report["accounting_complete"] = False
        report["observer_session_complete"] = False
        report["findings"] = sorted(set(report["findings"]) | {"session_ledger_identity_mismatch"})
    return report
