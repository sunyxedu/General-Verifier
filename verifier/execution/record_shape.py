"""Content-free shape of the retained record for the Oracle.

An Oracle that binds a typed record adapter must name the recorded dialect
exactly: which tool names occur, which decoder profile decodes their
invocations, and which payload keys exist.  Guessing a profile from a
catalog example produces a binding that matches nothing and falls back to a
long agent execution.  This summary is mechanical and carries no recorded
content: only event types, tool names, payload key names, decoder profiles,
operations, and counts.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Mapping

from ..core.events import EventLog, SourceProjectionUnavailable
from .record_program import _normalized_invocation

RECORD_SHAPE_SCHEMA = "gv-record-shape-v1"
MAX_DISTINCT_NAMES = 32
MAX_NAME_CHARS = 64


def _bounded_counter(counter: Counter) -> Dict[str, int]:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    kept = {}
    for name, count in items[:MAX_DISTINCT_NAMES]:
        kept[str(name)[:MAX_NAME_CHARS]] = count
    if len(items) > MAX_DISTINCT_NAMES:
        kept["(other)"] = sum(count for _name, count in items[MAX_DISTINCT_NAMES:])
    return kept


def record_shape(log: EventLog) -> Dict[str, Any]:
    """Summarize the source projection of ``log`` without any content."""
    try:
        events = log.source_projection()
    except SourceProjectionUnavailable as error:
        return {"schema": RECORD_SHAPE_SCHEMA, "available": False,
                "reason": str(error)[:200]}
    event_types: Counter = Counter()
    tools: Counter = Counter()
    input_keys: Counter = Counter()
    result_keys: Counter = Counter()
    profiles: Counter = Counter()
    operations: Counter = Counter()
    opaque_calls = 0
    paired_calls = 0
    call_ids = set()
    result_ids = set()
    last_seq = 0
    for event in events:
        event_types[event["type"]] += 1
        last_seq = max(last_seq, event["seq"])
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if event["type"] == "tool_call":
            tool = payload.get("tool")
            tools[tool if isinstance(tool, str) else type(tool).__name__] += 1
            value = payload.get("input")
            if isinstance(value, Mapping):
                for key in value:
                    input_keys[str(key)] += 1
            else:
                input_keys[f"({type(value).__name__})"] += 1
            invocation = _normalized_invocation(event)
            if invocation.get("status") == "decoded":
                profiles[invocation["profile"]] += 1
                operations[invocation["operation"]] += 1
            else:
                opaque_calls += 1
            identity = payload.get("tool_use_id")
            if isinstance(identity, str) and identity:
                call_ids.add(identity)
        elif event["type"] == "tool_result":
            for key in payload:
                result_keys[str(key)] += 1
            identity = payload.get("tool_use_id")
            if isinstance(identity, str) and identity:
                result_ids.add(identity)
    paired_calls = len(call_ids & result_ids)
    return {
        "schema": RECORD_SHAPE_SCHEMA,
        "available": True,
        "source_event_count": len(events),
        "last_source_seq": last_seq,
        "event_types": _bounded_counter(event_types),
        "tool_call": {
            "tools": _bounded_counter(tools),
            "input_keys": _bounded_counter(input_keys),
            "decoded_profiles": _bounded_counter(profiles),
            "decoded_operations": _bounded_counter(operations),
            "opaque_invocations": opaque_calls,
            "calls_paired_with_a_result_by_tool_use_id": paired_calls,
        },
        "tool_result": {"payload_keys": _bounded_counter(result_keys)},
    }


def record_shape_json(log: EventLog) -> str:
    try:
        return json.dumps(record_shape(log), ensure_ascii=False,
                          sort_keys=True, separators=(",", ":"))
    except Exception as error:  # never block enumeration on a summary
        return json.dumps({"schema": RECORD_SHAPE_SCHEMA, "available": False,
                           "reason": type(error).__name__})
