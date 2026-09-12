"""Operator-owned causal spans for verifier model calls.

This module is deliberately independent from model prompts and usage rows.
It records bounded identifiers and scheduler timing in a sidecar while a
Codex usage ledger is active.  Telemetry failures are best-effort and can
never change verifier control flow or model results.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional


SPAN_SCHEMA_VERSION = 1
SPAN_PATH_SUFFIX = ".spans.jsonl"
_MAX_TEXT_CHARS = 128
_MAX_INDEX = 1_000_000

_TEXT_FIELDS = frozenset({
    "claim_id", "claim_kind", "locus", "obligation_id", "stage",
})
_INT_FIELDS = frozenset({
    "round", "method_attempt", "redo", "judger_retry_index",
})
_BOOL_FIELDS = frozenset({"independent_judger"})


@dataclass(frozen=True)
class _Capture:
    path: str


_CAPTURE_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()
_CAPTURE: Optional[_Capture] = None
_METADATA: ContextVar[Mapping[str, Any]] = ContextVar(
    "gv_model_span_metadata", default={})
_ROLE: ContextVar[Optional[Mapping[str, Any]]] = ContextVar(
    "gv_model_role_span", default=None)
_ROLE_ATTEMPT: ContextVar[Optional[Mapping[str, Any]]] = ContextVar(
    "gv_model_role_attempt", default=None)


def span_sidecar_path(usage_path: str) -> str:
    """Return the deterministic model-span sidecar beside a usage ledger."""
    return os.path.abspath(os.fspath(usage_path)) + SPAN_PATH_SUFFIX


def _capture() -> Optional[_Capture]:
    with _CAPTURE_LOCK:
        return _CAPTURE


@contextmanager
def model_span_capture(usage_path: str) -> Iterator[None]:
    """Activate one process-wide span sidecar for an existing usage capture."""
    global _CAPTURE
    capture = _Capture(span_sidecar_path(usage_path))
    with _CAPTURE_LOCK:
        if _CAPTURE is not None:
            raise RuntimeError("a model span capture is already active")
        _CAPTURE = capture
    try:
        yield
    finally:
        with _CAPTURE_LOCK:
            if _CAPTURE is capture:
                _CAPTURE = None
            # A corrupted telemetry lifecycle must not mask scorer failures.


def _sanitize(values: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in values.items():
        if key in _TEXT_FIELDS:
            if isinstance(value, str) and value:
                out[key] = value[:_MAX_TEXT_CHARS]
        elif key in _INT_FIELDS:
            if (isinstance(value, int) and not isinstance(value, bool)
                    and 0 <= value <= _MAX_INDEX):
                out[key] = value
        elif key in _BOOL_FIELDS and isinstance(value, bool):
            out[key] = value
    return out


@contextmanager
def model_scope(**metadata: Any) -> Iterator[None]:
    """Nest bounded causal metadata without exposing it to a model seat."""
    merged = dict(_METADATA.get())
    merged.update(_sanitize(metadata))
    token = _METADATA.set(merged)
    try:
        yield
    finally:
        _METADATA.reset(token)


def _append(record: Mapping[str, Any]) -> None:
    capture = _capture()
    if capture is None:
        return
    try:
        encoded = json.dumps(
            dict(record), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        with _WRITE_LOCK:
            os.makedirs(os.path.dirname(capture.path) or ".", exist_ok=True)
            with open(capture.path, "a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
    except Exception:
        # Attribution is diagnostic. It must never alter scoring, retries, or
        # the original exception (including KeyboardInterrupt/SystemExit).
        return


def _time_fields() -> Dict[str, int]:
    return {
        "unix_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
    }


def _status(error: Optional[BaseException]) -> str:
    if error is None:
        return "succeeded"
    return "error" if isinstance(error, Exception) else "interrupted"


def seat_wait_started() -> Optional[int]:
    """Timestamp immediately before a role attempts to acquire SEAT_GATE."""
    return time.monotonic_ns() if _capture() is not None else None


def seat_acquired(wait_started_ns: Optional[int]) -> Optional[int]:
    """Record exact gate queue time, excluding sidecar writes themselves."""
    if wait_started_ns is None or _capture() is None:
        return None
    acquired_ns = time.monotonic_ns()
    context: Dict[str, Any] = {}
    if _ROLE.get() is not None:
        context.update(_ROLE.get())
    if _ROLE_ATTEMPT.get() is not None:
        context.update(_ROLE_ATTEMPT.get())
    _append({
        "record_type": "seat_acquired",
        "span_schema_version": SPAN_SCHEMA_VERSION,
        **context,
        **dict(_METADATA.get()),
        "monotonic_ns": acquired_ns,
        "unix_ns": time.time_ns(),
        "queue_duration_ns": max(0, acquired_ns - wait_started_ns),
    })
    return acquired_ns


def seat_released(acquired_ns: Optional[int]) -> None:
    """Record how long a role held SEAT_GATE, even on BaseException."""
    if acquired_ns is None or _capture() is None:
        return
    released_ns = time.monotonic_ns()
    context: Dict[str, Any] = {}
    if _ROLE.get() is not None:
        context.update(_ROLE.get())
    if _ROLE_ATTEMPT.get() is not None:
        context.update(_ROLE_ATTEMPT.get())
    _append({
        "record_type": "seat_released",
        "span_schema_version": SPAN_SCHEMA_VERSION,
        **context,
        **dict(_METADATA.get()),
        "monotonic_ns": released_ns,
        "unix_ns": time.time_ns(),
        "held_duration_ns": max(0, released_ns - acquired_ns),
    })


@contextmanager
def role_span(role: str) -> Iterator[None]:
    """Time one logical AgentRole call, including seat-gate queueing."""
    if _capture() is None:
        yield
        return
    span_id = uuid.uuid4().hex
    started = time.monotonic_ns()
    metadata = dict(_METADATA.get())
    context = {
        "role_span_id": span_id,
        "role": str(role)[:_MAX_TEXT_CHARS],
    }
    token = _ROLE.set(context)
    try:
        _append({
            "record_type": "role_span_started",
            "span_schema_version": SPAN_SCHEMA_VERSION,
            **context,
            **metadata,
            **_time_fields(),
        })
    except BaseException:
        _ROLE.reset(token)
        raise
    error: Optional[BaseException] = None
    try:
        yield
    except BaseException as caught:
        error = caught
        raise
    finally:
        finished = time.monotonic_ns()
        try:
            record = {
                "record_type": "role_span_finished",
                "span_schema_version": SPAN_SCHEMA_VERSION,
                **context,
                **metadata,
                **_time_fields(),
                "duration_ns": max(0, finished - started),
                "status": _status(error),
            }
            if error is not None:
                record["error_class"] = type(error).__name__[:_MAX_TEXT_CHARS]
            _append(record)
        finally:
            _ROLE.reset(token)


@contextmanager
def role_attempt(index: int) -> Iterator[None]:
    """Time one schema attempt inside a logical AgentRole call."""
    if _capture() is None:
        yield
        return
    parent = _ROLE.get()
    attempt_id = uuid.uuid4().hex
    metadata = dict(_METADATA.get())
    context = {
        "role_attempt_id": attempt_id,
        "role_attempt_index": int(index),
        **({"role_span_id": parent["role_span_id"]} if parent else {}),
    }
    token = _ROLE_ATTEMPT.set(context)
    started = time.monotonic_ns()
    try:
        _append({
            "record_type": "role_attempt_started",
            "span_schema_version": SPAN_SCHEMA_VERSION,
            **context,
            **metadata,
            **_time_fields(),
        })
    except BaseException:
        _ROLE_ATTEMPT.reset(token)
        raise
    error: Optional[BaseException] = None
    try:
        yield
    except BaseException as caught:
        error = caught
        raise
    finally:
        finished = time.monotonic_ns()
        try:
            record = {
                "record_type": "role_attempt_finished",
                "span_schema_version": SPAN_SCHEMA_VERSION,
                **context,
                **metadata,
                **_time_fields(),
                "duration_ns": max(0, finished - started),
                "status": _status(error),
            }
            if error is not None:
                record["error_class"] = type(error).__name__[:_MAX_TEXT_CHARS]
            _append(record)
        finally:
            _ROLE_ATTEMPT.reset(token)


@contextmanager
def model_call_span(call_id: str, *, role: str, model: str) -> Iterator[None]:
    """Join one backend call to its role attempt and existing usage call_id."""
    if _capture() is None:
        yield
        return
    parent_role = _ROLE.get()
    parent_attempt = _ROLE_ATTEMPT.get()
    attribution = "attributed" if parent_role is not None else "unattributed"
    context: Dict[str, Any] = {
        "call_id": str(call_id)[:_MAX_TEXT_CHARS],
        "role": str(role)[:_MAX_TEXT_CHARS],
        "model": str(model)[:_MAX_TEXT_CHARS],
        "attribution": attribution,
        **dict(_METADATA.get()),
    }
    if parent_role is not None:
        context["role_span_id"] = parent_role["role_span_id"]
    if parent_attempt is not None:
        context.update(parent_attempt)
    started = time.monotonic_ns()
    _append({
        "record_type": "model_call_started",
        "span_schema_version": SPAN_SCHEMA_VERSION,
        **context,
        **_time_fields(),
    })
    error: Optional[BaseException] = None
    try:
        yield
    except BaseException as caught:
        error = caught
        raise
    finally:
        finished = time.monotonic_ns()
        record = {
            "record_type": "model_call_finished",
            "span_schema_version": SPAN_SCHEMA_VERSION,
            **context,
            **_time_fields(),
            "duration_ns": max(0, finished - started),
            "status": _status(error),
        }
        if error is not None:
            record["error_class"] = type(error).__name__[:_MAX_TEXT_CHARS]
        _append(record)


def submit_with_context(pool, fn: Callable[..., Any], /, *args, **kwargs):
    """Submit with a fresh Context copy safe for one concurrent worker."""
    if _capture() is None:
        return pool.submit(fn, *args, **kwargs)
    context = copy_context()
    return pool.submit(context.run, fn, *args, **kwargs)
