"""Billing frames must not fail on well-formed events with unusual key order."""
from __future__ import annotations

import json

import pytest

from verifier.llm.codex_backend import (
    _attempt_usage,
    _parse_exec_accounting,
    _parse_exec_telemetry,
    _usage_event_type_or_full_parse,
)
from verifier.llm.schema import RoleError


USAGE = {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 3,
         "reasoning_output_tokens": 1}


def _stream(*middle):
    events = [
        {"type": "thread.started", "thread_id": "t-1"},
        {"type": "turn.started"},
        *middle,
        {"type": "item.completed", "item": {
            "id": "m", "type": "agent_message", "text": "{}"}},
        {"type": "turn.completed", "usage": USAGE},
    ]
    return "\n".join(json.dumps(event) for event in events)


def test_type_after_payload_is_still_a_well_formed_event():
    line = json.dumps({"item": {"id": "c", "type": "command_execution",
                                "aggregated_output": "x" * 2048},
                       "type": "item.completed"})
    assert _usage_event_type_or_full_parse(line) == "item.completed"
    stream = _stream(json.loads(line))
    assert _parse_exec_accounting(stream).usage["input_tokens"] == 10
    assert _parse_exec_telemetry(stream).usage["input_tokens"] == 10
    attempt = _attempt_usage(stream)
    assert attempt.telemetry is not None and attempt.status == "complete"


def test_truly_malformed_line_is_rejected_with_a_content_free_reason():
    good = _stream().split("\n")
    stream = "\n".join(good[:2] + ['{"item":SECRET,"type":"item.completed"}'] + good[2:])
    with pytest.raises(RoleError) as accounting:
        _parse_exec_accounting(stream)
    with pytest.raises(RoleError) as telemetry:
        _parse_exec_telemetry(stream)
    for error in (accounting.value, telemetry.value):
        message = str(error)
        assert message.startswith("codex exec --json emitted malformed line 3")
        assert "SECRET" not in message
    assert "JSONDecodeError at char" in str(telemetry.value)
    assert "Codex JSONL event type is not the first member" \
        in str(accounting.value)
    assert _attempt_usage(stream).telemetry is None


def test_accounting_ignores_an_undecodable_non_accounting_payload():
    # A framed non-accounting event is never decoded for billing; the
    # response parser still reports the exact line without its content.
    good = _stream().split("\n")
    stream = "\n".join(good[:2] + ['{"type":"item.completed","item":SECRET'] + good[2:])
    assert _parse_exec_accounting(stream).usage["input_tokens"] == 10
    with pytest.raises(RoleError) as telemetry:
        _parse_exec_telemetry(stream)
    assert "SECRET" not in str(telemetry.value)
    assert str(telemetry.value).startswith(
        "codex exec --json emitted malformed line 3: JSONDecodeError at char")


def test_accounting_counters_come_only_from_strict_accounting_events():
    # A turn.completed with a duplicated key is not trusted even though a
    # lenient decoder could read it.
    good = _stream().split("\n")
    duplicated = '{"type":"turn.completed","usage":{"input_tokens":1,' \
        '"input_tokens":2,"cached_input_tokens":0,"output_tokens":0,' \
        '"reasoning_output_tokens":0}}'
    stream = "\n".join(good[:-1] + [duplicated])
    with pytest.raises(RoleError, match="malformed line"):
        _parse_exec_accounting(stream)
