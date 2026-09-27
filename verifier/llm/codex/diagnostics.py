"""The content-free execution diagnostic persisted with each CLI attempt:
counts and sizes of the native JSONL stream's events and items, never
their contents."""
from __future__ import annotations

import json
from typing import Any, Dict

from .stream import MAX_PARSED_STREAM_CHARS, _stream_text


_EXECUTION_DIAGNOSTIC_SCHEMA = "gv-codex-jsonl-execution-diagnostic-v1"
# The subprocess has already captured stdout.  These caps bound only the
# additional parsing, allocation, and retained identifier state used by the
# content-free diagnostic below.
_EXECUTION_DIAGNOSTIC_MAX_STREAM_CHARS = MAX_PARSED_STREAM_CHARS
_EXECUTION_DIAGNOSTIC_MAX_STREAM_BYTES = 32 * 1024 * 1024
_EXECUTION_DIAGNOSTIC_MAX_LINE_CHARS = 1024 * 1024
_EXECUTION_DIAGNOSTIC_MAX_EVENTS = 50_000
_EXECUTION_DIAGNOSTIC_MAX_ITEM_IDS = 20_000
_EXECUTION_DIAGNOSTIC_MAX_ITEM_ID_CHARS = 256
_EXECUTION_DIAGNOSTIC_EVENT_TYPES = (
    "thread.started", "turn.started", "turn.completed", "turn.failed",
    "item.started", "item.updated", "item.completed", "error",
)
_EXECUTION_DIAGNOSTIC_ITEM_TYPES = (
    "agent_message", "reasoning", "command_execution", "file_change",
    "mcp_tool_call", "web_search", "image_generation", "todo_list", "error",
)


def _execution_diagnostic_base() -> Dict[str, Any]:
    """Return the fixed, content-free shape persisted for one CLI attempt."""
    return {
        "schema": _EXECUTION_DIAGNOSTIC_SCHEMA,
        # ``complete`` describes this diagnostic only. It never changes token
        # accounting completeness or whether the role response is accepted.
        "complete": False,
        "stream_present": False,
        "stream_bytes": 0,
        "stream_truncated": False,
        "invalid_utf8": False,
        "scanned_line_count": 0,
        "nonempty_line_count": 0,
        "parsed_event_count": 0,
        "malformed_event_count": 0,
        "non_object_event_count": 0,
        "unknown_event_count": 0,
        "known_event_type_counts": {},
        "item_event_counts": {
            "started": 0, "updated": 0, "completed": 0,
        },
        "completed_item_count": 0,
        "known_completed_item_type_counts": {},
        "unknown_item_type_count": 0,
        "malformed_item_count": 0,
        "duplicate_top_level_lifecycle_count": 0,
        "invalid_top_level_lifecycle_count": 0,
        "duplicate_item_lifecycle_count": 0,
        "invalid_item_lifecycle_count": 0,
        "unfinished_item_count": 0,
        "lifecycle_complete": False,
        "command_output_observed_count": 0,
        "command_output_utf8_bytes": 0,
        "max_command_output_utf8_bytes": 0,
        "agent_message_observed_count": 0,
        "agent_message_utf8_bytes": 0,
        "max_agent_message_utf8_bytes": 0,
        "oversized_line_count": 0,
        "untracked_item_event_count": 0,
        "overflow_reasons": [],
        "internal_error": False,
    }


def _diagnostic_utf8_bytes(value: str) -> int:
    """Count encoded bytes with bounded temporary allocations."""
    chunk_chars = 64 * 1024
    return sum(
        len(value[offset:offset + chunk_chars].encode("utf-8"))
        for offset in range(0, len(value), chunk_chars)
    )


def _diagnostic_unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _diagnostic_no_constant(_value):
    raise ValueError("non-finite JSON value")


def _codex_execution_diagnostic_unchecked(stdout: Any) -> Dict[str, Any]:
    """Summarize native Codex JSONL without retaining event contents.

    Counts describe CLI stream events/items, never internal model calls. The
    parser ignores command text, paths, reasoning text, and all tool payloads;
    only final command-output and agent-message UTF-8 sizes are measured.
    """
    result = _execution_diagnostic_base()
    overflow = set()
    tail_truncated = False

    if isinstance(stdout, bytes):
        result["stream_present"] = bool(stdout)
        result["stream_bytes"] = len(stdout)
        if len(stdout) > _EXECUTION_DIAGNOSTIC_MAX_STREAM_BYTES:
            stdout = stdout[:_EXECUTION_DIAGNOSTIC_MAX_STREAM_BYTES]
            tail_truncated = True
            overflow.add("stream_size_limit")
        try:
            text = stdout.decode("utf-8")
        except UnicodeDecodeError:
            text = stdout.decode("utf-8", errors="replace")
            result["invalid_utf8"] = True
    else:
        text = _stream_text(stdout)
        result["stream_present"] = bool(text)
        if len(text) > _EXECUTION_DIAGNOSTIC_MAX_STREAM_CHARS:
            # Exact UTF-8 sizing would require work proportional to the
            # unbounded tail. Preserve unknown as null rather than inventing
            # a byte count from the Python character count.
            result["stream_bytes"] = None
            text = text[:_EXECUTION_DIAGNOSTIC_MAX_STREAM_CHARS]
            tail_truncated = True
            overflow.add("stream_size_limit")
        else:
            try:
                result["stream_bytes"] = _diagnostic_utf8_bytes(text)
            except UnicodeEncodeError:
                result["stream_bytes"] = None
                result["invalid_utf8"] = True
    result["stream_truncated"] = tail_truncated

    event_counts = {kind: 0 for kind in _EXECUTION_DIAGNOSTIC_EVENT_TYPES}
    item_type_counts = {
        kind: 0 for kind in _EXECUTION_DIAGNOSTIC_ITEM_TYPES}
    item_event_counts = result["item_event_counts"]
    active_items: Dict[str, str] = {}
    completed_items = set()
    terminal_events = 0
    parsed_budget = 0
    top_level_phase = "new"

    cursor = 0
    text_length = len(text)
    while cursor < text_length:
        newline = text.find("\n", cursor)
        if newline < 0:
            line_end = text_length
            has_newline = False
        else:
            line_end = newline
            has_newline = True
        result["scanned_line_count"] += 1
        line_chars = line_end - cursor

        # A stream-cap prefix may end in the middle of a private event. Do not
        # materialize or parse that fragment as if it were a malformed line.
        if tail_truncated and not has_newline:
            break
        if line_chars > _EXECUTION_DIAGNOSTIC_MAX_LINE_CHARS:
            result["oversized_line_count"] += 1
            overflow.add("line_size_limit")
            cursor = line_end + (1 if has_newline else 0)
            continue

        raw = text[cursor:line_end]
        cursor = line_end + (1 if has_newline else 0)
        if not raw.strip():
            continue
        result["nonempty_line_count"] += 1
        parsed_budget += 1
        if parsed_budget > _EXECUTION_DIAGNOSTIC_MAX_EVENTS:
            overflow.add("event_count_limit")
            continue

        try:
            event = json.loads(
                raw,
                object_pairs_hook=_diagnostic_unique_object,
                parse_constant=_diagnostic_no_constant,
            )
        except (ValueError, TypeError, RecursionError):
            result["malformed_event_count"] += 1
            continue
        if not isinstance(event, dict):
            result["non_object_event_count"] += 1
            continue
        event_type = event.get("type")
        if not isinstance(event_type, str):
            result["malformed_event_count"] += 1
            continue
        if event_type not in event_counts:
            result["unknown_event_count"] += 1
            continue

        event_counts[event_type] += 1
        result["parsed_event_count"] += 1
        if event_type == "thread.started" \
                and event_counts[event_type] > 1:
            result["duplicate_top_level_lifecycle_count"] += 1
        elif event_type == "turn.started" \
                and event_counts[event_type] > 1:
            result["duplicate_top_level_lifecycle_count"] += 1
        elif event_type in {"turn.completed", "turn.failed"}:
            terminal_events += 1
            if terminal_events > 1:
                result["duplicate_top_level_lifecycle_count"] += 1

        if event_type == "thread.started":
            if top_level_phase != "new":
                result["invalid_top_level_lifecycle_count"] += 1
            else:
                top_level_phase = "thread"
        elif event_type == "turn.started":
            if top_level_phase != "thread":
                result["invalid_top_level_lifecycle_count"] += 1
            else:
                top_level_phase = "turn"
        elif event_type in {"turn.completed", "turn.failed"}:
            if top_level_phase != "turn":
                result["invalid_top_level_lifecycle_count"] += 1
            else:
                top_level_phase = "terminal"

        if event_type not in {
                "item.started", "item.updated", "item.completed"}:
            continue
        if top_level_phase != "turn":
            result["invalid_top_level_lifecycle_count"] += 1
        lifecycle = event_type.removeprefix("item.")
        item_event_counts[lifecycle] += 1
        item = event.get("item")
        if not isinstance(item, dict):
            result["malformed_item_count"] += 1
            continue
        item_id = item.get("id")
        item_type = item.get("type")
        if (not isinstance(item_id, str) or not item_id
                or len(item_id) > _EXECUTION_DIAGNOSTIC_MAX_ITEM_ID_CHARS
                or not isinstance(item_type, str) or not item_type):
            result["malformed_item_count"] += 1
            continue

        already_tracked = item_id in active_items or item_id in completed_items
        if (not already_tracked
                and len(active_items) + len(completed_items)
                >= _EXECUTION_DIAGNOSTIC_MAX_ITEM_IDS):
            overflow.add("item_id_limit")
            result["untracked_item_event_count"] += 1
            continue

        if event_type == "item.started":
            if already_tracked:
                result["duplicate_item_lifecycle_count"] += 1
            else:
                active_items[item_id] = item_type
            continue
        if event_type == "item.updated":
            if active_items.get(item_id) != item_type:
                result["invalid_item_lifecycle_count"] += 1
            continue

        # Completion-only items are part of the native JSONL format. Count a
        # completed item and its potentially large output at most once per id.
        if item_id in completed_items:
            result["duplicate_item_lifecycle_count"] += 1
            continue
        active_type = active_items.pop(item_id, None)
        if active_type is not None and active_type != item_type:
            result["invalid_item_lifecycle_count"] += 1
            completed_items.add(item_id)
            continue
        completed_items.add(item_id)
        result["completed_item_count"] += 1
        if item_type in item_type_counts:
            item_type_counts[item_type] += 1
        else:
            result["unknown_item_type_count"] += 1

        if item_type == "command_execution":
            output = item.get("aggregated_output")
            if isinstance(output, str):
                try:
                    size = _diagnostic_utf8_bytes(output)
                except UnicodeEncodeError:
                    result["malformed_item_count"] += 1
                    continue
                result["command_output_observed_count"] += 1
                result["command_output_utf8_bytes"] += size
                result["max_command_output_utf8_bytes"] = max(
                    result["max_command_output_utf8_bytes"], size)
            else:
                result["malformed_item_count"] += 1
        elif item_type == "agent_message":
            message = item.get("text")
            if isinstance(message, str):
                try:
                    size = _diagnostic_utf8_bytes(message)
                except UnicodeEncodeError:
                    result["malformed_item_count"] += 1
                    continue
                result["agent_message_observed_count"] += 1
                result["agent_message_utf8_bytes"] += size
                result["max_agent_message_utf8_bytes"] = max(
                    result["max_agent_message_utf8_bytes"], size)
            else:
                result["malformed_item_count"] += 1

    result["known_event_type_counts"] = {
        kind: event_counts[kind]
        for kind in _EXECUTION_DIAGNOSTIC_EVENT_TYPES
        if event_counts[kind]
    }
    result["known_completed_item_type_counts"] = {
        kind: item_type_counts[kind]
        for kind in _EXECUTION_DIAGNOSTIC_ITEM_TYPES
        if item_type_counts[kind]
    }
    result["unfinished_item_count"] = len(active_items)
    result["lifecycle_complete"] = (
        event_counts["thread.started"] == 1
        and event_counts["turn.started"] == 1
        and terminal_events == 1
        and top_level_phase == "terminal"
        and not active_items
        and not result["duplicate_top_level_lifecycle_count"]
        and not result["invalid_top_level_lifecycle_count"]
        and not result["duplicate_item_lifecycle_count"]
        and not result["invalid_item_lifecycle_count"]
    )
    result["overflow_reasons"] = sorted(overflow)
    result["complete"] = bool(
        result["lifecycle_complete"]
        and not overflow
        and not result["malformed_event_count"]
        and not result["non_object_event_count"]
        and not result["unknown_event_count"]
        and not result["unknown_item_type_count"]
        and not result["malformed_item_count"]
        and not result["invalid_utf8"]
    )
    return result


def _codex_execution_diagnostic(stdout: Any) -> Dict[str, Any]:
    """Return non-fatal, content-free execution stats for one attempt."""
    try:
        return _codex_execution_diagnostic_unchecked(stdout)
    except Exception:
        # Diagnostics must never change a role verdict or make cost appear
        # complete. Do not persist an exception message: it may embed stdout.
        result = _execution_diagnostic_base()
        result["stream_bytes"] = None
        result["internal_error"] = True
        result["overflow_reasons"] = ["internal_error"]
        return result
