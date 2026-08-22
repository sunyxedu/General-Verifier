"""Codex rollout -> hook feed: the ledger format is the interface, so a
codex trace must reach the offline pipeline byte-compatible with what the
Claude Code hooks write.
"""
from __future__ import annotations

import json
import os

from verifier.core.events import EventLog
from verifier.core.store import ClaimStore
from verifier.offline import ingest_feed
from verifier.stream import codex
from verifier.stream.builder import Builder

ROLLOUT = [
    {"type": "session_meta", "payload": {"session_id": "s1"}},
    {"type": "response_item", "payload": {
        "type": "function_call", "name": "exec_command",
        "call_id": "call_1", "arguments": '{"cmd":"echo 42"}'}},
    {"type": "event_msg", "payload": {"type": "agent_reasoning",
                                      "text": "thinking"}},
    {"type": "response_item", "payload": {
        "type": "function_call_output", "call_id": "call_1",
        "output": "42\n"}},
    {"type": "response_item", "payload": {
        "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": "FINAL ANSWER: 42"}]}},
]


def rollout_file(tmp_path, events=ROLLOUT):
    path = str(tmp_path / "rollout.jsonl")
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        f.write("not json\n")        # torn/foreign line: skipped, not fatal
    return path


def test_rollout_becomes_referenceable_effects(tmp_path):
    state = str(tmp_path / "gv")
    counts = codex.convert(rollout_file(tmp_path), state)

    assert counts == {"events": 2, "tool_calls": 1}
    log = EventLog(str(tmp_path / "events.jsonl"))
    assert ingest_feed(os.path.join(state, "hook_feed.jsonl"), log) == 2
    builder = Builder(log, ClaimStore())
    for event in log.replay():
        builder.feed(event)
    # the result pairs back to its call by call_id, exactly as for hooks
    assert "effect:2" in builder.labels


def test_final_text_is_the_last_assistant_message(tmp_path):
    state = str(tmp_path / "gv")
    codex.convert(rollout_file(tmp_path), state)

    with open(os.path.join(state, "final.txt")) as f:
        assert f.read() == "FINAL ANSWER: 42"


def test_arguments_are_decoded_into_structured_input(tmp_path):
    state = str(tmp_path / "gv")
    codex.convert(rollout_file(tmp_path), state)

    with open(os.path.join(state, "hook_feed.jsonl")) as f:
        call = json.loads(f.readline())
    assert call["payload"]["input"] == {"cmd": "echo 42"}
    assert call["payload"]["tool_use_id"] == "call_1"


def test_result_carries_the_tool_name_like_the_hooks_do(tmp_path):
    state = str(tmp_path / "gv")
    codex.convert(rollout_file(tmp_path), state)

    with open(os.path.join(state, "hook_feed.jsonl")) as f:
        call, result = (json.loads(line) for line in f)
    # the hooks record tool_name on results; the converted feed must too,
    # or ledger obligations refute honest records on a null field
    assert result["payload"]["tool"] == "exec_command"
    assert call["payload"]["session"] == "s1"
    assert result["payload"]["session"] == "s1"
    assert result["payload"]["response_chars"] == len(
        result["payload"]["excerpt"])
    assert result["payload"]["excerpt_truncated"] is False


def test_result_marks_a_truncated_effect(tmp_path):
    events = [
        {"type": "response_item", "payload": {
            "type": "function_call", "name": "exec_command",
            "call_id": "call_1", "arguments": "{}"}},
        {"type": "response_item", "payload": {
            "type": "function_call_output", "call_id": "call_1",
            "output": "x" * (codex.EXCERPT_CHARS + 1)}},
    ]
    state = str(tmp_path / "gv")
    codex.convert(rollout_file(tmp_path, events), state)

    with open(os.path.join(state, "hook_feed.jsonl")) as f:
        _, result = (json.loads(line) for line in f)
    payload = result["payload"]
    assert payload["excerpt_truncated"] is True
    assert payload["response_chars"] > len(payload["excerpt"])
    assert len(payload["excerpt"]) == codex.EXCERPT_CHARS


def test_task_instruction_skips_harness_noise(tmp_path):
    events = [
        {"type": "response_item", "payload": {
            "type": "message", "role": "user", "content": [
                {"type": "input_text",
                 "text": "<recommended_plugins>\n- Spotify\n"},
                {"type": "input_text",
                 "text": "<environment_context>\n  <cwd>/app</cwd>\n"},
            ]}},
        {"type": "response_item", "payload": {
            "type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "Count the tokens."}]}},
    ]
    path = rollout_file(tmp_path, events=events)
    assert codex.task_instruction(path) == "Count the tokens."


def test_task_instruction_empty_when_only_noise(tmp_path):
    events = [
        {"type": "response_item", "payload": {
            "type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "<environment_context>x"}]}},
    ]
    assert codex.task_instruction(rollout_file(tmp_path, events)) == ""


def test_newest_rollout_picks_the_latest(tmp_path):
    sessions = tmp_path / "sessions" / "2026" / "08" / "20"
    sessions.mkdir(parents=True)
    old, new = sessions / "rollout-a.jsonl", sessions / "rollout-b.jsonl"
    old.write_text("{}\n")
    new.write_text("{}\n")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    assert codex.newest_rollout(str(tmp_path / "sessions")) == str(new)
    assert codex.newest_rollout(str(tmp_path / "sessions"), since=3) is None
