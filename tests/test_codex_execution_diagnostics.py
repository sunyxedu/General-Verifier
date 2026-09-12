"""Content-free profiling must never change billing or verification behavior."""
import json

import pytest

from verifier.llm import codex_backend as backend


USAGE = {"input_tokens": 100, "cached_input_tokens": 30,
         "output_tokens": 20, "reasoning_output_tokens": 5}


def item(kind="command_execution", item_id="private-id", phase="completed", **kw):
    return {"type": "item." + phase,
            "item": {"id": item_id, "type": kind, **kw}}


def stream(*items):
    events = [{"type": "thread.started", "thread_id": "private-thread"},
              {"type": "turn.started"}, *items,
              {"type": "turn.completed", "usage": USAGE}]
    return "\n".join(json.dumps(e) for e in events) + "\n"


def test_counts_utf8_bytes_once_and_retains_no_private_contents():
    text = stream(
        item(phase="started", command="secret command", aggregated_output="partial"),
        item(phase="updated", aggregated_output="partial"),
        item(aggregated_output="\u4e2d\U0001f600", command="secret command"),
        item("agent_message", "private-answer", text="answer \u4e2d"),
        item("reasoning", "private-reasoning", text="private reasoning"))
    result = backend._codex_execution_diagnostic(text)
    assert result["complete"] is True
    assert result["command_output_utf8_bytes"] == 7
    assert result["max_command_output_utf8_bytes"] == 7
    assert result["command_output_observed_count"] == 1
    assert result["agent_message_utf8_bytes"] == 10
    assert result["stream_bytes"] == len(text.encode())
    assert result["known_completed_item_type_counts"] == {
        "command_execution": 1, "agent_message": 1, "reasoning": 1}
    assert result["item_event_counts"] == {"started": 1, "updated": 1, "completed": 3}
    encoded = json.dumps(result)
    for secret in ("private-id", "private-answer", "private-thread", "secret command",
                   "private reasoning", "answer", "partial"):
        assert secret not in encoded


def test_duplicate_lifecycle_never_multiplies_output():
    command = item(aggregated_output="large" * 1000)
    result = backend._codex_execution_diagnostic(stream(command, command))
    assert result["complete"] is False
    assert result["duplicate_item_lifecycle_count"] == 1
    assert result["command_output_utf8_bytes"] == 5000
    assert result["completed_item_count"] == 1


@pytest.mark.parametrize("value", [None, "", b"", '{"type":"thread.started"}\n'])
def test_empty_and_partial_streams_are_not_complete(value):
    result = backend._codex_execution_diagnostic(value)
    assert result["complete"] is False
    assert result["internal_error"] is False


@pytest.mark.parametrize("bad", ["{", "[]", '{"type":[]}',
                                 '{"type":"error","type":"turn.started"}',
                                 '{"type":"error","x":NaN}'])
def test_malformed_json_is_counted_without_internal_failure(bad):
    text = stream().replace('{"type": "turn.completed"', bad + '\n{"type": "turn.completed"')
    result = backend._codex_execution_diagnostic(text)
    assert result["complete"] is False
    assert result["malformed_event_count"] + result["non_object_event_count"] == 1
    assert result["internal_error"] is False


def test_unknown_kinds_are_not_retained_as_map_keys():
    result = backend._codex_execution_diagnostic(stream(
        {"type": "private-unknown-event"}, item("private-unknown-item")))
    assert result["complete"] is False
    assert result["unknown_event_count"] == 1
    assert result["unknown_item_type_count"] == 1
    assert "private-unknown" not in json.dumps(result)


@pytest.mark.parametrize("event", [item(), item(aggregated_output=None),
                                  item("agent_message"), item("agent_message", text=4),
                                  item(aggregated_output="\ud800")])
def test_missing_or_invalid_observed_payload_is_not_known_zero(event):
    result = backend._codex_execution_diagnostic(stream(event))
    assert result["complete"] is False
    assert result["malformed_item_count"] == 1
    assert result["command_output_observed_count"] == 0
    assert result["agent_message_observed_count"] == 0


def test_invalid_utf8_and_lone_surrogate_stream_are_incomplete():
    for text in (stream().encode() + b"\xff", stream() + "\ud800"):
        result = backend._codex_execution_diagnostic(text)
        assert result["complete"] is False
        assert result["invalid_utf8"] is True


@pytest.mark.parametrize("which", ["top_order", "item_after_terminal", "unfinished",
                                  "update_without_start", "changed_type", "duplicate_start"])
def test_invalid_lifecycle_is_never_complete(which):
    events = [json.loads(line) for line in stream().splitlines()]
    if which == "top_order":
        events = [events[2], events[0], events[1]]
    elif which == "item_after_terminal":
        events.append(item(aggregated_output="x"))
    elif which == "unfinished":
        events.insert(2, item(phase="started"))
    elif which == "update_without_start":
        events.insert(2, item(phase="updated"))
    elif which == "changed_type":
        events[2:2] = [item(phase="started"), item("agent_message", text="x")]
    else:
        events.insert(1, events[0])
    result = backend._codex_execution_diagnostic("\n".join(map(json.dumps, events)))
    assert result["complete"] is False
    assert result["internal_error"] is False


@pytest.mark.parametrize("limit,amount,reason,as_bytes", [
    ("MAX_STREAM_CHARS", 25, "stream_size_limit", False),
    ("MAX_STREAM_BYTES", 25, "stream_size_limit", True),
    ("MAX_LINE_CHARS", 25, "line_size_limit", False),
    ("MAX_EVENTS", 2, "event_count_limit", False),
    ("MAX_ITEM_IDS", 1, "item_id_limit", False),
])
def test_parser_caps_make_limits_explicit(monkeypatch, limit, amount, reason, as_bytes):
    monkeypatch.setattr(backend, "_EXECUTION_DIAGNOSTIC_" + limit, amount)
    text = stream(item(aggregated_output="x"), item(item_id="other", aggregated_output="y"))
    result = backend._codex_execution_diagnostic(text.encode() if as_bytes else text)
    assert result["complete"] is False
    assert reason in result["overflow_reasons"]
    assert len(json.dumps(result)) < 2400


def test_internal_diagnostic_error_is_content_free_and_nonfatal(monkeypatch):
    def fail(_):
        raise RuntimeError("private command and output")
    monkeypatch.setattr(backend, "_codex_execution_diagnostic_unchecked", fail)
    result = backend._codex_execution_diagnostic("private stream")
    assert result["internal_error"] is True
    assert result["complete"] is False
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("status,force_incomplete", [
    ("succeeded", False), ("nonzero_exit", False), ("timeout", True),
])
def test_attempt_record_keeps_billing_independent(tmp_path, status, force_incomplete):
    path = tmp_path / "usage.jsonl"
    text = stream(item(aggregated_output="secret output"), item("agent_message", "a", text='{"ok":true}'))
    usage = backend._record_codex_attempt(
        (str(path), {}), base={"role": "judger"}, attempt_id="attempt",
        attempt_index=1, started=True, attempt_status=status,
        returncode=0 if status == "succeeded" else 1,
        started_at_unix_ns=10, finished_at_unix_ns=20, duration_ns=10,
        stdout=text, force_usage_incomplete=force_incomplete)
    row = json.loads(path.read_text())
    assert row["execution_diagnostic"]["command_output_utf8_bytes"] == 13
    assert row["usage_complete"] is (not force_incomplete)
    assert row["input_tokens"] == (None if force_incomplete else 100)
    assert row["uncached_input_tokens"] == (None if force_incomplete else 70)
    assert (usage.telemetry is None) is force_incomplete
    assert "secret output" not in path.read_text()


def test_diagnostic_failure_does_not_erase_complete_bill(tmp_path, monkeypatch):
    def fail(_):
        raise RuntimeError("private")
    monkeypatch.setattr(backend, "_codex_execution_diagnostic_unchecked", fail)
    path = tmp_path / "usage.jsonl"
    backend._record_codex_attempt(
        (str(path), {}), base={}, attempt_id="attempt", attempt_index=1,
        started=True, attempt_status="succeeded", returncode=0,
        started_at_unix_ns=0, finished_at_unix_ns=1, duration_ns=1,
        stdout=stream(item("agent_message", text='{"ok":true}')))
    row = json.loads(path.read_text())
    assert row["usage_complete"] is True
    assert row["input_tokens"] == 100
    assert row["execution_diagnostic"]["internal_error"] is True
