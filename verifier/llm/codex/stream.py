"""Reading a ``codex exec --json`` event stream.

Native Codex JSONL framing, the strict parse of the two accounting events
(``thread.started``, ``turn.completed``) and of a completed turn's final
agent message, and the answer's JSON object in the final output. Every
parser fails closed and never quotes stream contents in its errors.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..schema import RoleError


_USAGE_REQUIRED_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)
# The subprocess has already captured stdout. A stream this long is not
# decoded in full a second time (by the event-type fallback here, or by the
# content-free execution diagnostic).
MAX_PARSED_STREAM_CHARS = 8 * 1024 * 1024


def _native_jsonl_lines(text: str):
    """Frame native Codex JSONL only at its U+000A record delimiter.

    ``str.splitlines()`` also treats NEL, LS, and PS characters as line
    boundaries.  Those characters are legal, unescaped JSON string content,
    so using it here can split a valid tool-result event and hide the terminal
    usage record that follows it.
    """
    return text.split("\n")


@dataclass(frozen=True)
class _ExecTelemetry:
    thread_id: str
    usage: Mapping[str, int]
    last_message: str


def _exact_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RoleError(
            f"codex exec turn.completed usage has invalid {field}: "
            "expected a nonnegative integer")
    return value


def _usage_event_type(raw: str) -> Optional[str]:
    """Read the required first ``type`` member without touching its payload.

    Only ``thread.started`` and ``turn.completed`` carry billing identity or
    counters. Once a different type has been framed as the first object
    member, its potentially huge, deep, truncated, or malformed payload is
    irrelevant to billing and is not decoded. Accounting events are decoded
    strictly in full by ``_strict_usage_event``. Requiring ``type`` first
    makes a partial or ambiguous prefix fail closed instead of guessing where
    an accounting discriminator might be hidden.
    """
    decoder = json.JSONDecoder(parse_constant=_reject_usage_constant)
    index = 0
    while index < len(raw) and raw[index] in " \t\r\n":
        index += 1
    if index >= len(raw) or raw[index] != "{":
        raise ValueError("Codex JSONL event is not an object")
    index += 1
    while index < len(raw) and raw[index] in " \t\r\n":
        index += 1
    key, index = decoder.raw_decode(raw, index)
    if key != "type":
        raise ValueError("Codex JSONL event type is not the first member")
    while index < len(raw) and raw[index] in " \t\r\n":
        index += 1
    if index >= len(raw) or raw[index] != ":":
        raise ValueError("Codex JSONL event has an ambiguous type prefix")
    index += 1
    while index < len(raw) and raw[index] in " \t\r\n":
        index += 1
    event_type, index = decoder.raw_decode(raw, index)
    if not isinstance(event_type, str):
        raise ValueError("Codex JSONL event type is not a string")
    while index < len(raw) and raw[index] in " \t\r\n":
        index += 1
    if index >= len(raw) or raw[index] not in ",}":
        raise ValueError("Codex JSONL event has an ambiguous type prefix")
    if raw[index] == "}" and raw[index + 1:].strip():
        raise ValueError("Codex JSONL event has trailing prefix data")
    return event_type


def _usage_event_type_or_full_parse(raw: str) -> Optional[str]:
    """Frame the event type cheaply; decode the whole line only if needed.

    A valid JSON object whose ``type`` is not its first member is still a
    well-formed Codex event.  Rejecting it made an otherwise complete,
    fully billed Judger attempt fail as "malformed" and forced paid retries
    that reproduced the same stream.  The full decode is bounded by the
    subprocess capture and never touches billing counters: those are still
    read only from strictly decoded accounting events.
    """
    try:
        return _usage_event_type(raw)
    except ValueError as framing_error:
        if len(raw) > MAX_PARSED_STREAM_CHARS:
            raise
        try:
            event = _strict_usage_event(raw)
        except (ValueError, TypeError, RecursionError):
            raise framing_error from None
        event_type = event.get("type")
        if not isinstance(event_type, str):
            raise framing_error from None
        return event_type


def _unique_usage_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate accounting event field")
        value[key] = item
    return value


def _reject_usage_constant(_value):
    raise ValueError("non-finite accounting event value")


def _strict_usage_event(raw: str) -> Mapping[str, Any]:
    event = json.loads(
        raw,
        object_pairs_hook=_unique_usage_object,
        parse_constant=_reject_usage_constant,
    )
    if not isinstance(event, dict):
        raise ValueError("accounting event is not an object")
    return event


def _parse_exec_telemetry(stdout: str) -> _ExecTelemetry:
    """Parse one subprocess-scoped Codex JSONL stream, failing closed.

    The live ``turn.completed`` event is authoritative.  Looking for the
    newest file in the global sessions tree would race when scorer workers run
    concurrently; the stream's ``thread.started`` event binds the counts to
    this exact invocation instead.
    """
    thread_ids = []
    completions = []
    last_message = ""
    for number, raw in enumerate(_native_jsonl_lines(stdout or ""), 1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except (ValueError, RecursionError) as err:
            raise RoleError(
                f"codex exec --json emitted malformed line {number}: "
                f"{_content_free_reason(err)}") from err
        if not isinstance(event, dict):
            raise RoleError(
                f"codex exec --json emitted a non-object on line {number}")
        event_type = event.get("type")
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                thread_ids.append(thread_id)
        elif event_type == "turn.completed":
            completions.append(event.get("usage"))
        elif event_type == "item.completed":
            item = event.get("item")
            if (isinstance(item, dict)
                    and item.get("type") == "agent_message"
                    and isinstance(item.get("text"), str)):
                last_message = item["text"].strip()

    if len(thread_ids) != 1:
        raise RoleError(
            "codex exec --json must emit exactly one thread.started event "
            f"(got {len(thread_ids)})")
    if len(completions) != 1 or not isinstance(completions[0], dict):
        raise RoleError(
            "codex exec --json must emit exactly one turn.completed usage "
            f"object (got {len(completions)})")

    raw_usage = completions[0]
    usage = {
        field: _exact_nonnegative_int(raw_usage.get(field), field)
        for field in _USAGE_REQUIRED_FIELDS
    }
    cached = usage["cached_input_tokens"]
    if cached > usage["input_tokens"]:
        raise RoleError(
            "codex exec reported cached_input_tokens greater than input_tokens")
    cache_write = raw_usage.get("cache_write_input_tokens")
    if cache_write is not None:
        usage["cache_write_input_tokens"] = _exact_nonnegative_int(
            cache_write, "cache_write_input_tokens")
    return _ExecTelemetry(
        thread_id=thread_ids[0], usage=usage, last_message=last_message)


def _parse_exec_accounting(stdout: str) -> _ExecTelemetry:
    """Parse only subprocess-bound accounting events, failing closed.

    Non-accounting event payloads are deliberately not recursively decoded.
    They remain subject to the existing response/event parser before a role
    output is accepted.
    """
    thread_ids = []
    completions = []
    for number, raw in enumerate(_native_jsonl_lines(stdout or ""), 1):
        if not raw.strip():
            continue
        try:
            event_type = _usage_event_type_or_full_parse(raw)
            if event_type not in {"thread.started", "turn.completed"}:
                continue
            event = _strict_usage_event(raw)
        except (ValueError, TypeError, RecursionError) as error:
            raise RoleError(
                f"codex exec --json emitted malformed line {number}: "
                f"{_content_free_reason(error)}") from error
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                thread_ids.append(thread_id)
        else:
            completions.append(event.get("usage"))

    if len(thread_ids) != 1:
        raise RoleError(
            "codex exec --json must emit exactly one thread.started event "
            f"(got {len(thread_ids)})")
    if len(completions) != 1 or not isinstance(completions[0], dict):
        raise RoleError(
            "codex exec --json must emit exactly one turn.completed usage "
            f"object (got {len(completions)})")

    raw_usage = completions[0]
    usage = {
        field: _exact_nonnegative_int(raw_usage.get(field), field)
        for field in _USAGE_REQUIRED_FIELDS
    }
    cached = usage["cached_input_tokens"]
    if cached > usage["input_tokens"]:
        raise RoleError(
            "codex exec reported cached_input_tokens greater than input_tokens")
    cache_write = raw_usage.get("cache_write_input_tokens")
    if cache_write is not None:
        usage["cache_write_input_tokens"] = _exact_nonnegative_int(
            cache_write, "cache_write_input_tokens")
    return _ExecTelemetry(
        thread_id=thread_ids[0], usage=usage, last_message="")


def _content_free_reason(error: BaseException) -> str:
    """A parser's fixed message or a decode position, never stream text."""
    if isinstance(error, json.JSONDecodeError):
        return f"{type(error).__name__} at char {error.pos}"
    if isinstance(error, RecursionError):
        return "RecursionError"
    text = str(error)
    if text.startswith(("Codex JSONL event", "duplicate accounting",
                        "non-finite accounting", "accounting event")):
        return text
    return type(error).__name__


def _completed_response_without_usage(
    stdout: str, *, permitted_startup_notice: Optional[str] = None,
    allow_message_sequence: bool = False,
) -> Optional[str]:
    """Recover only a structurally complete response, never token counters.

    A billing-only gap must not force another paid generation of an already
    complete answer. This deliberately narrower fallback requires strict,
    ordered JSONL and a final agent message; malformed, partial, ambiguous or
    failed streams remain failures. The role's capability policy and output
    schema are still checked afterwards. Usage remains invalid/unknown.
    """
    if not isinstance(stdout, str) or len(stdout) > 32 * 1024 * 1024:
        return None

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate event key")
            result[key] = value
        return result

    def no_constant(_value):
        raise ValueError("non-finite event value")

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite event value")
        return parsed

    if type(allow_message_sequence) is not bool:
        return None
    phase, last_message = "new", None
    message_phase = False
    active_items, completed_items = {}, set()
    notice_seen = False
    try:
        for number, line in enumerate(_native_jsonl_lines(stdout), 1):
            if number > 100_000:
                return None
            if not line.strip():
                continue
            event = json.loads(line, object_pairs_hook=unique_object,
                               parse_constant=no_constant,
                               parse_float=finite_float)
            if not isinstance(event, dict):
                return None
            kind = event.get("type")
            if phase == "new" and kind == "thread.started":
                thread_id = event.get("thread_id")
                if not isinstance(thread_id, str) or not thread_id.strip():
                    return None
                phase = "thread"
            elif (phase == "thread" and kind == "item.completed"
                  and permitted_startup_notice is not None):
                item = event.get("item")
                if (notice_seen or not isinstance(item, dict) or item.get("type") != "error"
                        or item.get("message") != permitted_startup_notice
                        or not set(item) <= {"id", "type", "message"}
                        or not isinstance(item.get("id"), str) or not item["id"]
                        or item["id"] in completed_items):
                    return None
                completed_items.add(item["id"])
                notice_seen = True
            elif phase == "thread" and kind == "turn.started":
                phase = "turn"
            elif phase == "turn" and kind in (
                    "item.started", "item.updated", "item.completed"):
                item = event.get("item")
                if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                    return None
                item_id = item.get("id")
                if (item["type"] == "error"
                        or (last_message is not None
                            and not allow_message_sequence)
                        or (message_phase
                            and item["type"] != "agent_message")
                        or not isinstance(item_id, str) or not item_id
                        or item_id in completed_items):
                    return None
                if kind == "item.started":
                    if item_id in active_items:
                        return None
                    active_items[item_id] = item["type"]
                elif kind == "item.updated":
                    if active_items.get(item_id) != item["type"]:
                        return None
                else:
                    # Completion-only items are part of the documented JSONL
                    # shape. A started item must still finish with its type.
                    if item_id in active_items and active_items[item_id] != item["type"]:
                        return None
                    active_items.pop(item_id, None)
                    completed_items.add(item_id)
                if item["type"] == "agent_message":
                    message_phase = True
                if kind == "item.completed" and item["type"] == "agent_message":
                    if not isinstance(item.get("text"), str) or not item["text"].strip():
                        return None
                    if active_items:
                        return None
                    last_message = item["text"].strip()
            elif (phase == "turn" and kind == "turn.completed"
                  and last_message and not active_items):
                phase = "complete"
            else:
                return None
    except (ValueError, TypeError, RecursionError):
        return None
    return last_message if phase == "complete" else None


def _stream_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def extract_json(text: str, accept: Optional[Any] = None) -> Any:
    """The answer's JSON object in `text` (fences and prose tolerated): the
    last complete top-level object, or with ``accept`` (a validator that
    raises on a wrong shape) the last one it accepts. The answer comes last
    by contract; a JSON-like fragment in the prose before it is not it."""
    decoder = json.JSONDecoder()
    found = []
    i = text.find("{")
    while i >= 0:
        try:
            obj, end = decoder.raw_decode(text, i)
        except (json.JSONDecodeError, RecursionError):
            i = text.find("{", i + 1)
            continue
        found.append(obj)
        i = text.find("{", end)
    if not found:
        raise RoleError(f"no JSON object in output: {text[:200]!r}")
    if accept is not None:
        for obj in reversed(found):
            try:
                accept(obj)
            except RoleError:
                continue
            return obj
    return found[-1]
