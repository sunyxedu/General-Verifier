import json

import pytest

from verifier.events import EventLog


def test_append_assigns_monotonic_seq(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    e1 = log.append("session_start", {})
    e2 = log.append("message", {"text": "hi"})
    assert (e1["seq"], e2["seq"]) == (1, 2)


def test_replay_roundtrip_and_reopen_continues_seq(tmp_path):
    path = str(tmp_path / "events.jsonl")
    log = EventLog(path)
    log.append("session_start", {})
    log.append("tool_call", {"tool": "python", "cmd": "print(1)"})

    reopened = EventLog(path)  # a new handle must continue, not restart
    e3 = reopened.append("tool_result", {"exit": 0})
    assert e3["seq"] == 3
    assert [e["type"] for e in reopened.replay()] == [
        "session_start", "tool_call", "tool_result",
    ]


def test_replay_detects_tampered_seq(tmp_path):
    path = str(tmp_path / "events.jsonl")
    log = EventLog(path)
    log.append("session_start", {})
    log.append("message", {"text": "x"})
    lines = open(path, "r", encoding="utf-8").read().splitlines()
    with open(path, "w", encoding="utf-8") as f:  # drop line 1: silent truncation
        f.write(lines[1] + "\n")
    with pytest.raises(ValueError, match="corrupted"):
        list(EventLog(path).replay())


def test_unknown_event_type_rejected(tmp_path):
    log = EventLog(str(tmp_path / "e.jsonl"))
    with pytest.raises(ValueError):
        log.append("banana", {})


def test_event_digest_ignores_ts_but_not_payload(tmp_path):
    log = EventLog(str(tmp_path / "e.jsonl"))
    e = log.append("tool_result", {"exit": 0, "stdout": "ok"})
    same = dict(e, ts=0.0)
    other = dict(e, payload={"exit": 1, "stdout": "ok"})
    assert EventLog.event_digest(e) == EventLog.event_digest(same)
    assert EventLog.event_digest(e) != EventLog.event_digest(other)


def test_get_by_seq(tmp_path):
    log = EventLog(str(tmp_path / "e.jsonl"))
    log.append("session_start", {})
    target = log.append("tool_result", {"exit": 0})
    assert EventLog(log.path).get(2)["payload"] == target["payload"]
    assert EventLog(log.path).get(99) is None
