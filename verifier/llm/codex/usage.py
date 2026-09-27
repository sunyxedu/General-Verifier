"""The Codex usage ledger: one ``codex_cli_attempt`` row per CLI attempt
while a capture is active, with exact provider-reported token counters (or
an explicit unknown/partial/invalid status, never a fabricated zero) and
the attempt's content-free execution diagnostic.

The typed-Judger attribution of an attempt is carried per logical call in a
context variable, so concurrent calls never share it."""
from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional

from ..model_spans import model_span_capture
from ..schema import RoleError
from .diagnostics import _codex_execution_diagnostic
from .stream import (
    _ExecTelemetry,
    _USAGE_REQUIRED_FIELDS,
    _native_jsonl_lines,
    _parse_exec_accounting,
    _stream_text,
    _strict_usage_event,
    _usage_event_type_or_full_parse,
)


_USAGE_CAPTURE_LOCK = threading.Lock()
_USAGE_WRITE_LOCK = threading.Lock()
_USAGE_CAPTURE: Optional["_UsageCapture"] = None
_TYPED_JUDGER_ATTEMPT: ContextVar[Optional[tuple[str, str]]] = ContextVar(
    "gv_typed_judger_attempt", default=None)


def _typed_judger_attempt_fields() -> Dict[str, Any]:
    """Return immutable per-call transport attribution for attempt ledgers."""
    profile = _TYPED_JUDGER_ATTEMPT.get()
    if profile is None:
        return {
            "typed_judger_transport": False,
            "typed_judger_protocol": None,
            "typed_judger_packet_sha256": None,
        }
    protocol, digest = profile
    return {
        "typed_judger_transport": True,
        "typed_judger_protocol": protocol,
        "typed_judger_packet_sha256": digest,
    }


@contextmanager
def _typed_judger_attempt_scope(
    protocol: str,
    packet_sha256: str,
) -> Iterator[None]:
    """Bind one rendered packet to one logical call without shared mutation."""
    if (not isinstance(protocol, str) or not protocol
            or len(protocol) > 512
            or not isinstance(packet_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", packet_sha256) is None):
        raise RoleError("invalid typed Judger attempt identity")
    if _TYPED_JUDGER_ATTEMPT.get() is not None:
        raise RoleError("nested typed Judger attempt identity")
    token = _TYPED_JUDGER_ATTEMPT.set((protocol, packet_sha256))
    try:
        yield
    finally:
        _TYPED_JUDGER_ATTEMPT.reset(token)


@dataclass(frozen=True)
class _UsageCapture:
    path: str
    context_by_cwd: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class _AttemptUsage:
    telemetry: Optional[_ExecTelemetry]
    thread_id: Optional[str]
    status: str
    error: Optional[str]


@contextmanager
def codex_usage_ledger(
    path: str,
    context_by_cwd: Mapping[str, Mapping[str, Any]],
) -> Iterator[None]:
    """Capture exact ``codex exec --json`` usage for calls in this block.

    ``context_by_cwd`` gives the scorer a race-free association from each
    isolated state directory to its job/trial/config/rep identity.  It lives
    only in this Python process (and is therefore not exposed to the seat's
    shell environment).  One capture may be active at a time; worker threads
    share it and append under ``_USAGE_WRITE_LOCK``.
    """
    global _USAGE_CAPTURE
    capture = _UsageCapture(
        path=os.path.abspath(os.fspath(path)),
        context_by_cwd={
            os.path.realpath(os.path.abspath(os.fspath(cwd))): dict(context)
            for cwd, context in context_by_cwd.items()
        },
    )
    with _USAGE_CAPTURE_LOCK:
        if _USAGE_CAPTURE is not None:
            raise RuntimeError("a Codex usage ledger is already active")
        _USAGE_CAPTURE = capture
    try:
        with model_span_capture(capture.path):
            yield
    finally:
        with _USAGE_CAPTURE_LOCK:
            if _USAGE_CAPTURE is not capture:
                raise RuntimeError("Codex usage ledger capture changed unexpectedly")
            _USAGE_CAPTURE = None


def _usage_target(cwd: Optional[str]) -> Optional[tuple[str, Dict[str, Any]]]:
    with _USAGE_CAPTURE_LOCK:
        capture = _USAGE_CAPTURE
    if capture is None:
        return None
    if not cwd:
        raise RoleError("Codex usage capture requires a seat cwd")
    canonical = os.path.realpath(os.path.abspath(cwd))
    context = capture.context_by_cwd.get(canonical)
    if context is None:
        raise RoleError(
            f"Codex usage capture has no scoring context for cwd {canonical!r}")
    return capture.path, dict(context)


def _usage_error_code(error: Optional[str]) -> Optional[str]:
    """Persist bounded parser categories, never untrusted field values."""
    if error is None:
        return None
    for field in (*_USAGE_REQUIRED_FIELDS, "cache_write_input_tokens"):
        if error.startswith(f"codex exec turn.completed usage has invalid {field}:"):
            return "invalid_counter:" + field
    for prefix, code in (
        ("codex exec --json emitted malformed line", "malformed_jsonl"),
        ("codex exec --json emitted a non-object", "non_object_event"),
        ("codex exec --json must emit exactly one thread.started", "thread_identity"),
        ("codex exec --json must emit exactly one turn.completed usage", "completion_or_usage_object"),
        ("codex exec reported cached_input_tokens greater than input_tokens", "cached_exceeds_input"),
    ):
        if error.startswith(prefix):
            return code
    return "incomplete_or_unclassified"


def _append_usage_record(path: str, record: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        dict(record), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    with _USAGE_WRITE_LOCK:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())


def _attempt_usage(stdout: Any, *, force_incomplete: bool = False,
                   incomplete_reason: Optional[str] = None) -> _AttemptUsage:
    """Classify usage without ever turning missing counters into zero."""
    text = _stream_text(stdout)
    thread_ids = []
    thread_events = 0
    completions = 0
    malformed = False
    for raw in _native_jsonl_lines(text):
        if not raw.strip():
            continue
        try:
            event_type = _usage_event_type_or_full_parse(raw)
            if event_type not in {"thread.started", "turn.completed"}:
                continue
            event = _strict_usage_event(raw)
        except (ValueError, TypeError, RecursionError):
            malformed = True
            continue
        if event_type == "thread.started":
            thread_events += 1
            thread_id = event.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                thread_ids.append(thread_id)
        else:
            completions += 1
    unique_threads = set(thread_ids)
    thread_id = next(iter(unique_threads)) if len(unique_threads) == 1 else None

    if force_incomplete:
        status = "partial" if text.strip() else "unknown"
        return _AttemptUsage(
            telemetry=None,
            thread_id=thread_id,
            status=status,
            error=(incomplete_reason or
                   "CLI completion was not observed as trustworthy"),
        )
    try:
        telemetry = _parse_exec_accounting(text)
    except RoleError as error:
        if not text.strip():
            status = "unknown"
        elif (malformed or thread_events > 1 or completions > 1
              or (thread_events == 1 and not thread_ids)
              or (thread_events == 1 and completions == 1)):
            status = "invalid"
        else:
            status = "partial"
        return _AttemptUsage(
            telemetry=None,
            thread_id=thread_id,
            status=status,
            error=str(error)[:300],
        )
    return _AttemptUsage(
        telemetry=telemetry,
        thread_id=telemetry.thread_id,
        status="complete",
        error=None,
    )


def _record_codex_attempt(
    usage_target,
    *,
    base: Mapping[str, Any],
    attempt_id: str,
    attempt_index: int,
    started: bool,
    attempt_status: str,
    returncode: Optional[int],
    started_at_unix_ns: int,
    finished_at_unix_ns: int,
    duration_ns: int,
    stdout: Any = "",
    force_usage_incomplete: bool = False,
    usage_incomplete_reason: Optional[str] = None,
    error_kind: Optional[str] = None,
    error: Optional[str] = None,
) -> _AttemptUsage:
    """Append one terminal attempt row when usage capture is active."""
    usage = _attempt_usage(
        stdout,
        force_incomplete=force_usage_incomplete,
        incomplete_reason=usage_incomplete_reason,
    )
    if usage_target is None:
        return usage
    usage_path, usage_context = usage_target
    execution_diagnostic = _codex_execution_diagnostic(stdout)
    counts = usage.telemetry.usage if usage.telemetry is not None else {}
    input_tokens = counts.get("input_tokens")
    cached_tokens = counts.get("cached_input_tokens")
    uncached_tokens = (
        input_tokens - cached_tokens
        if input_tokens is not None and cached_tokens is not None else None
    )
    _append_usage_record(usage_path, {
        **usage_context,
        **base,
        # Preserve the legacy successful-row schema and fields while adding
        # an explicit attempt contract for complete failure accounting.
        "schema_version": 1,
        "record_type": "codex_cli_attempt",
        "attempt_schema_version": 1,
        "attempt_id": attempt_id,
        "attempt_index": attempt_index,
        "started": started,
        "attempt_status": attempt_status,
        "usage_status": usage.status,
        "usage_complete": usage.telemetry is not None,
        "usage_error_code": _usage_error_code(usage.error),
        "usage_error": ({
            "unknown": "usage telemetry was not observed",
            "partial": "usage telemetry was incomplete",
            "invalid": "usage telemetry failed validation",
        }.get(usage.status) if usage.error is not None else None),
        "thread_id": usage.thread_id,
        "started_at_unix_ns": started_at_unix_ns,
        "finished_at_unix_ns": finished_at_unix_ns,
        "duration_ns": duration_ns,
        "duration_seconds": duration_ns / 1_000_000_000,
        "returncode": returncode,
        "error_kind": error_kind,
        "error": error[:300] if error else None,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "uncached_input_tokens": uncached_tokens,
        # Older CLI streams did not expose cache writes. Unknown remains null.
        "cache_write_input_tokens": counts.get("cache_write_input_tokens"),
        "output_tokens": counts.get("output_tokens"),
        "reasoning_tokens": counts.get("reasoning_output_tokens"),
        "execution_diagnostic": execution_diagnostic,
    })
    return usage
