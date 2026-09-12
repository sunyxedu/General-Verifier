"""The Oracle sees the record's dialect, never its content."""
from __future__ import annotations

import json

from verifier.core.events import EventLog
from verifier.execution.record_shape import record_shape, record_shape_json


def _ledger(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("session_start", {"agent": "x"})
    for index in range(3):
        log.append("tool_call", {
            "tool": "bash", "tool_use_id": f"call-{index}",
            "input": {"keystrokes": f"echo SECRET-{index}"}})
        log.append("tool_result", {
            "tool": "bash", "tool_use_id": f"call-{index}",
            "excerpt": f"SECRET-{index}", "exit_code": 0})
    log.append("tool_call", {"tool": "browser", "input": {"url": "http://x"}})
    log.append("commit", {"label": "f", "kind": "final", "text": "done",
                          "deps": []})
    # verifier events are outside the record
    log.append("status_change", {"claim": "c", "status": "verified"})
    return log


def test_shape_names_dialect_without_content(tmp_path):
    shape = record_shape(_ledger(tmp_path))
    assert shape["available"] is True
    assert shape["source_event_count"] == 9
    assert shape["last_source_seq"] == 9
    assert shape["event_types"] == {"tool_call": 4, "tool_result": 3,
                                    "session_start": 1, "commit": 1}
    call = shape["tool_call"]
    assert call["tools"] == {"bash": 3, "browser": 1}
    assert call["input_keys"] == {"keystrokes": 3, "url": 1}
    assert call["decoded_profiles"] == {"lav-bash-keystrokes-v1": 3}
    assert call["decoded_operations"] == {"terminal_input": 3}
    assert call["opaque_invocations"] == 1
    assert call["calls_paired_with_a_result_by_tool_use_id"] == 3
    assert shape["tool_result"]["payload_keys"] == {
        "tool": 3, "tool_use_id": 3, "excerpt": 3, "exit_code": 3}
    rendered = record_shape_json(_ledger(tmp_path))
    assert "SECRET" not in rendered and "http://x" not in rendered
    assert json.loads(rendered)["schema"] == "gv-record-shape-v1"


def test_shape_is_bounded_and_survives_an_unreadable_ledger(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    for index in range(40):
        log.append("tool_call", {"tool": f"tool-{index}", "input": {}})
    tools = record_shape(log)["tool_call"]["tools"]
    assert len(tools) == 33 and tools["(other)"] == 8
    broken = EventLog(str(tmp_path / "broken.jsonl"))
    with open(broken.path, "wb") as handle:
        handle.write(b"not json\n")
    shape = record_shape(broken)
    assert shape["available"] is False
    assert "SECRET" not in json.dumps(shape)


def test_oracle_receives_the_record_shape(tmp_path, monkeypatch):
    from verifier.llm.schema import FixtureBackend
    from test_frozen_record import _fixtures, _service

    backend = FixtureBackend(_fixtures())
    service = _service(tmp_path, backend, monkeypatch)
    service.drain_hooks()
    service.commit("c1", "assertion", "the run issued one command",
                   ["effect:2"])
    oracle_calls = [call for call in backend.calls if call["role"] == "oracles"]
    assert oracle_calls
    shape = json.loads(oracle_calls[0]["variables"]["record_shape"])
    assert shape["tool_call"]["tools"] == {"bash": 1}
    assert shape["tool_call"]["input_keys"] == {"command": 1}
