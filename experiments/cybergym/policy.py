"""Evaluation-validity guard, separate from GV claims and judgments.

This detects observed hosted-search CLI events; it does not claim that an
unparsed request or a quiet trace proves the complete absence of a tool.
Only fixed structural identifiers are retained, never queries or results.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import threading

POLICY_SCHEMA = "gv-search-policy-v1"
ITEM_EVENTS = frozenset({"item.started", "item.updated", "item.completed"})
MAX_SOLVER_EVENT_BYTES = 2 * 1024 * 1024


class SearchPolicyViolation(RuntimeError):
    """An observed search makes this attempt unscorable, not a model failure."""


class SearchPolicyEvidenceError(RuntimeError):
    """The host cannot safely inspect the required policy evidence."""


class SearchPolicyGuard:
    def __init__(self):
        self._lock = threading.Lock()
        self._source = None
        self._event_type = None
        self._sealed = False
        self._active_executions = 0
        self._evidence_complete = True

    def _observe(self, source, event_type):
        if not isinstance(event_type, str) or event_type not in ITEM_EVENTS:
            return
        with self._lock:
            if self._source is None:
                self._source, self._event_type = source, event_type

    def observe_backend_phase(self, record):
        phase = record.get("phase") if isinstance(record, dict) else None
        if isinstance(phase, str) and phase in {
                "exec_heartbeat", "pipe_drain_heartbeat", "exec_returned"}:
            missed = record.get("unparsed_stdout_lines")
            if (type(missed) is int and missed > 0) or record.get("reader_failed") is True:
                with self._lock:
                    self._evidence_complete = False
        if (isinstance(record, dict) and record.get("phase") == "cli_event"
                and record.get("item_kind") == "web_search"):
            self._observe("verifier", record.get("event_type"))

    def observe_solver_event(self, event):
        if not isinstance(event, dict):
            return
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "web_search":
            self._observe("solver", event.get("type"))
            return
        # Claude Code stream-json records tool calls as content blocks.
        if event.get("type") == "assistant":
            message = event.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                for block in content:
                    if (isinstance(block, dict)
                            and block.get("type") in ("tool_use", "server_tool_use")
                            and block.get("name") in ("WebSearch", "WebFetch")):
                        self._observe("solver", "item.completed")
                        return
        if event.get("type") == "result":
            usage = event.get("usage")
            server = usage.get("server_tool_use") if isinstance(usage, dict) else None
            if isinstance(server, dict) and any(
                    type(server.get(field)) is int and server[field] > 0
                    for field in ("web_search_requests", "web_fetch_requests")):
                self._observe("solver", "item.completed")

    def snapshot(self):
        with self._lock:
            return {"schema": POLICY_SCHEMA, "web_search": "disabled",
                    "violation": self._source is not None,
                    "violation_source": self._source, "event_type": self._event_type,
                    "sealed": self._sealed, "evidence_complete": self._evidence_complete}

    def begin_execution(self):
        with self._lock:
            if self._sealed:
                raise SearchPolicyEvidenceError("verification_execution_after_policy_seal")
            self._active_executions += 1

    def end_execution(self):
        with self._lock:
            if self._active_executions <= 0:
                raise SearchPolicyEvidenceError("verification_execution_accounting_invalid")
            self._active_executions -= 1

    def try_seal(self):
        with self._lock:
            if self._active_executions:
                return False
            self._sealed = True
            return True


def validate_policy_snapshot(value):
    """Fail closed on missing/malformed responses; return observed violation."""
    if (not isinstance(value, dict)
            or set(value) != {"schema", "web_search", "violation", "violation_source", "event_type",
                              "sealed", "evidence_complete"}
            or value.get("schema") != POLICY_SCHEMA or value.get("web_search") != "disabled"
            or type(value.get("violation")) is not bool or type(value.get("sealed")) is not bool
            or type(value.get("evidence_complete")) is not bool):
        raise SearchPolicyEvidenceError("search_policy_snapshot_invalid")
    if value["violation"]:
        if (not isinstance(value["violation_source"], str)
                or value["violation_source"] not in {"solver", "verifier"}
                or not isinstance(value["event_type"], str)
                or value["event_type"] not in ITEM_EVENTS):
            raise SearchPolicyEvidenceError("search_policy_snapshot_invalid")
    elif value["violation_source"] is not None or value["event_type"] is not None:
        raise SearchPolicyEvidenceError("search_policy_snapshot_invalid")
    if value["violation"]:
        return True
    if not value["evidence_complete"]:
        raise SearchPolicyEvidenceError("verifier_search_policy_evidence_incomplete")
    return False


def validate_policy_response(value, *, expected_instance_id=None):
    """Fence a process-local latch: a restarted sidecar is not clean evidence."""
    if (not isinstance(value, dict) or set(value) != {"schema", "instance_id", "policy"}
            or value.get("schema") != "gv-search-policy-response-v1"
            or type(value.get("instance_id")) is not str
            or re.fullmatch(r"[0-9a-f]{32}", value["instance_id"]) is None):
        raise SearchPolicyEvidenceError("search_policy_response_invalid")
    if expected_instance_id is not None and value["instance_id"] != expected_instance_id:
        raise SearchPolicyEvidenceError("search_policy_instance_changed")
    return value["instance_id"], validate_policy_snapshot(value["policy"])


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate event key")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError("nonfinite event value")


class SolverSearchObserver:
    """Incrementally inspect a host-owned append-only CLI log, including resumes.

    Read only the current bounded-line snapshot, not future writer output.
    Incomplete live lines are revisited; terminal unterminated lines are read.
    Unsafe files, oversized lines and malformed JSON cannot authorize scoring.
    The log is forensic evidence, not writable scratch mounted into the solver.
    """
    def __init__(self, guard):
        self.guard = guard
        self.offset = 0
        self._identity = None
        self.verifier_instance_id = None

    def consume(self, path: Path, *, final=False):
        descriptor = None
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            info = os.fstat(descriptor)
            identity = (info.st_dev, info.st_ino)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_nlink != 1 or info.st_size < self.offset
                    or self._identity is not None and identity != self._identity):
                raise SearchPolicyEvidenceError("solver_search_policy_log_unsafe")
            self._identity = identity
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = None
                stream.seek(self.offset)
                remaining = info.st_size - self.offset
                while remaining:
                    raw = stream.readline(min(MAX_SOLVER_EVENT_BYTES + 1, remaining))
                    if not raw:
                        raise SearchPolicyEvidenceError("solver_search_policy_log_changed")
                    if len(raw) > MAX_SOLVER_EVENT_BYTES:
                        raise SearchPolicyEvidenceError("solver_search_policy_event_too_large")
                    if not raw.endswith(b"\n") and not final:
                        break
                    if raw.strip():
                        try:
                            event = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                                               parse_constant=_nonfinite)
                        except (ValueError, UnicodeError, RecursionError):
                            raise SearchPolicyEvidenceError("solver_search_policy_event_invalid") from None
                        if not isinstance(event, dict):
                            raise SearchPolicyEvidenceError("solver_search_policy_event_invalid")
                        self.guard.observe_solver_event(event)
                    self.offset += len(raw)
                    remaining -= len(raw)
        except OSError:
            raise SearchPolicyEvidenceError("solver_search_policy_log_unreadable") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
