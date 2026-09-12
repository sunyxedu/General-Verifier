"""Strict, diagnostic-only ATIF-to-ledger conversion for replay_lav.

These tests exercise local JSON conversion and the existing deterministic
ledger adapter only.  They never start Docker or a model process.
"""
from __future__ import annotations

import copy
import json
import re

import pytest

from experiments.terminal_bench import replay_lav
from test_ledger_query_adapter import (
    discharge as discharge_ledger_query,
    harness as adapter_harness,
)
from verifier import offline
from verifier.core.model import OUTCOME_HOLDS


PROFILE = "atif-single-call-per-step-v1"
_MISSING = object()


def _call(tool="bash", arguments=_MISSING):
    if arguments is _MISSING:
        arguments = {"keystrokes": "printf 'exact value'"}
    return {"tool_name": tool, "arguments": arguments}


def _step(index, *, call=None, result=...):
    row = {
        "step_id": index,
        "source": "agent",
        "tool_calls": [] if call is None else [call],
    }
    if result is not ...:
        row["observation"] = {
            "results": [] if result is None else [{"content": result}],
        }
    return row


def _trajectory(steps, *, session="source-session", agent="source-agent"):
    return {
        "reward": "TOP-LEVEL-GOLD-MUST-NOT-LEAK",
        "trial_name": "diagnostic-trial",
        "trajectory": {
            "schema_version": "atif-converted-1.0",
            "session_id": session,
            "agent": {"name": agent},
            "steps": steps,
        },
    }


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_valid_conversion_preserves_source_values_and_is_deterministic(
        tmp_path):
    structured = json.dumps({"returncode": 0, "output": "first\n"})
    traj = _trajectory([
        _step(7, call=_call("bash", {
            "keystrokes": "printf 'first\\n'", "timeout": 2,
            "nested": {"flags": [False, None, 3]},
        }), result=structured),
        # A duplicate source step_id must not collide: source position, not a
        # possibly malformed external identifier, determines the call id.
        _step(7, call=_call("UnKnOwN.Tool", ["raw", None, {"x": 1}]),
              result="not-json; preserve exactly"),
    ])
    original = copy.deepcopy(traj)
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"

    report1 = replay_lav.convert_feed(traj, str(first))
    report2 = replay_lav.convert_feed(traj, str(second))

    assert traj == original
    assert first.read_bytes() == second.read_bytes()
    assert report1 == report2
    assert {key: report1[key] for key in (
        "diagnostic_only", "profile", "source_tool_calls", "tool_results",
        "calls_without_result", "unsupported_replay_calls",
    )} == {
        "diagnostic_only": True,
        "profile": PROFILE,
        "source_tool_calls": 2,
        "tool_results": 2,
        "calls_without_result": 0,
        "unsupported_replay_calls": 1,
    }
    assert report1["feed_bytes"] == len(first.read_bytes())
    assert len(report1["feed_sha256"]) == 64
    assert len(report1["source_projection_sha256"]) == 64
    assert report1["not_preserved"] == [
        "non-tool messages", "complete result metadata"]
    rows = _rows(first)
    assert [row["type"] for row in rows] == [
        "tool_call", "tool_result", "tool_call", "tool_result"]
    calls, results = rows[::2], rows[1::2]
    assert [row["payload"]["tool"] for row in calls] == [
        "bash", "UnKnOwN.Tool"]
    assert calls[0]["payload"]["input"] == \
        traj["trajectory"]["steps"][0]["tool_calls"][0]["arguments"]
    assert calls[1]["payload"]["input"] == ["raw", None, {"x": 1}]
    assert [json.loads(row["payload"]["excerpt"])[0]["text"]
            for row in results] == ["first\n", "not-json; preserve exactly"]
    assert all(row["payload"]["session"] == "source-session" for row in rows)
    assert all(row["payload"]["agent"] == "source-agent" for row in rows)
    assert [row["payload"]["tool_use_id"] for row in calls] == [
        row["payload"]["tool_use_id"] for row in results]
    assert len({row["payload"]["tool_use_id"] for row in calls}) == 2
    assert "TOP-LEVEL-GOLD-MUST-NOT-LEAK" not in first.read_text()
    assert "TOP-LEVEL-GOLD-MUST-NOT-LEAK" not in json.dumps(report1)


def test_ids_depend_only_on_step_and_call_position(tmp_path):
    steps = [_step(9, call=_call(), result="done")]
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"
    replay_lav.convert_feed(_trajectory(steps, session="one", agent="a"), str(first))
    replay_lav.convert_feed(_trajectory(steps, session="two", agent="b"), str(second))

    first_id = _rows(first)[0]["payload"]["tool_use_id"]
    second_id = _rows(second)[0]["payload"]["tool_use_id"]
    assert first_id == second_id
    # In particular this is not the old random UUID/session construction.
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        first_id,
    ) is None


@pytest.mark.parametrize("arguments", [{}, None, [], "", 0, False])
def test_every_json_arguments_value_is_preserved(tmp_path, arguments):
    path = tmp_path / "feed.jsonl"
    traj = _trajectory([_step(
        1,
        call={"tool_name": "arbitrary", "arguments": arguments},
        result="result",
    )])

    report = replay_lav.convert_feed(traj, str(path))

    assert _rows(path)[0]["payload"]["input"] == arguments
    assert report["source_tool_calls"] == 1
    assert report["unsupported_replay_calls"] == 1


@pytest.mark.parametrize(("content", "expected"), [
    (json.dumps({"output": "decoded", "returncode": 0}), "decoded"),
    (json.dumps({"output": 7}), json.dumps({"output": 7})),
    (json.dumps(["list", "value"]), json.dumps(["list", "value"])),
    (json.dumps("JSON string"), json.dumps("JSON string")),
    ("{not-json", "{not-json"),
    ("", ""),
])
def test_result_extracts_only_a_json_object_string_output(
        tmp_path, content, expected):
    path = tmp_path / "feed.jsonl"

    replay_lav.convert_feed(
        _trajectory([_step(1, call=_call(), result=content)]), str(path))

    result = _rows(path)[1]
    assert result["type"] == "tool_result"
    assert json.loads(result["payload"]["excerpt"]) == [{
        "text": expected, "type": "input_text"}]


def test_missing_result_keeps_honest_pending_call(tmp_path):
    path = tmp_path / "feed.jsonl"

    report = replay_lav.convert_feed(
        _trajectory([_step(1, call=_call(), result=None)]), str(path))

    rows = _rows(path)
    assert [row["type"] for row in rows] == ["tool_call"]
    assert report["source_tool_calls"] == 1
    assert report["tool_results"] == 0
    assert report["calls_without_result"] == 1


def test_missing_source_session_and_agent_are_explicit_null(tmp_path):
    path = tmp_path / "feed.jsonl"
    traj = _trajectory([_step(1, call=_call(), result="done")])
    del traj["trajectory"]["session_id"]
    del traj["trajectory"]["agent"]

    replay_lav.convert_feed(traj, str(path))

    assert all(row["payload"]["session"] is None for row in _rows(path))
    assert all(row["payload"]["agent"] is None for row in _rows(path))


def test_explicit_source_call_id_is_preserved_and_must_match_result(tmp_path):
    path = tmp_path / "feed.jsonl"
    call = {**_call(), "tool_call_id": "source-call-9"}
    result = {"content": "done", "source_call_id": "source-call-9"}
    step = {"step_id": 1, "tool_calls": [call],
            "observation": {"results": [result]}}

    replay_lav.convert_feed(_trajectory([step]), str(path))

    assert [row["payload"]["tool_use_id"] for row in _rows(path)] == [
        "source-call-9", "source-call-9"]


@pytest.mark.parametrize("change", ["mismatch", "duplicate"])
def test_source_call_id_mismatch_or_duplicate_is_rejected(tmp_path, change):
    path = tmp_path / "feed.jsonl"
    call = {**_call(), "tool_call_id": "source-call-9"}
    if change == "mismatch":
        steps = [{
            "step_id": 1, "tool_calls": [call],
            "observation": {"results": [{
                "content": "done", "source_call_id": "other-call"}]},
        }]
    else:
        steps = [
            _step(1, call=call, result="one"),
            _step(2, call=copy.deepcopy(call), result="two"),
        ]

    with pytest.raises(ValueError):
        replay_lav.convert_feed(_trajectory(steps), str(path))
    assert not path.exists()


def test_replay_support_is_counted_without_dropping_unsupported_calls(tmp_path):
    path = tmp_path / "feed.jsonl"
    traj = _trajectory([
        _step(1, call=_call("bash", {"keystrokes": "pwd"}), result="/app"),
        _step(2, call=_call("Bash", {"keystrokes": "case matters"}), result="x"),
        _step(3, call=_call("bash", {"keystrokes": ["not", "string"]}), result="x"),
        _step(4, call=_call("browser", {"url": "https://example.invalid"}), result="x"),
    ])

    report = replay_lav.convert_feed(traj, str(path))

    assert report["source_tool_calls"] == 4
    assert report["unsupported_replay_calls"] == 3
    assert [row["payload"]["tool"] for row in _rows(path)
            if row["type"] == "tool_call"] == ["bash", "Bash", "bash", "browser"]


def _invalid_documents():
    valid_call = _call()
    return [
        None,
        {},
        {"trajectory": None},
        {"trajectory": {}},
        {"trajectory": {"steps": {}}},
        {"trajectory": {"steps": ["not-an-object"]}},
        _trajectory([{"step_id": 1, "tool_calls": {}}]),
        _trajectory([{"step_id": 1, "tool_calls": ["not-an-object"]}]),
        _trajectory([_step(1, call={}, result="x")]),
        _trajectory([_step(1, call={"tool_name": "bash"}, result="x")]),
        _trajectory([_step(1, call={"arguments": {}}, result="x")]),
        _trajectory([_step(1, call={"tool_name": "", "arguments": {}}, result="x")]),
        _trajectory([{
            "step_id": 1, "tool_calls": [valid_call, valid_call],
            "observation": {"results": []},
        }]),
        _trajectory([{
            "step_id": 1, "tool_calls": [valid_call],
            "observation": {"results": [
                {"content": "one"}, {"content": "two"}]},
        }]),
        _trajectory([{
            "step_id": 1, "tool_calls": [],
            "observation": {"results": [{"content": "orphan"}]},
        }]),
        _trajectory([{
            "step_id": 1, "tool_calls": [valid_call],
            "observation": [],
        }]),
        _trajectory([{
            "step_id": 1, "tool_calls": [valid_call],
            "observation": {"results": {}},
        }]),
        _trajectory([{
            "step_id": 1, "tool_calls": [valid_call],
            "observation": {"results": ["not-an-object"]},
        }]),
        _trajectory([{
            "step_id": 1, "tool_calls": [valid_call],
            "observation": {"results": [{"content": 7}]},
        }]),
    ]


@pytest.mark.parametrize("document", _invalid_documents())
def test_ambiguous_or_malformed_source_is_rejected_before_output_mutation(
        tmp_path, document):
    new_path = tmp_path / "new.jsonl"
    with pytest.raises(ValueError):
        replay_lav.convert_feed(copy.deepcopy(document), str(new_path))
    assert not new_path.exists()

    existing = tmp_path / "existing.jsonl"
    existing.write_bytes(b"operator-owned-existing-content\n")
    with pytest.raises(ValueError):
        replay_lav.convert_feed(copy.deepcopy(document), str(existing))
    assert existing.read_bytes() == b"operator-owned-existing-content\n"


def test_late_invalid_step_does_not_publish_valid_prefix_or_overwrite_feed(
        tmp_path):
    path = tmp_path / "existing.jsonl"
    original = b"operator-owned-existing-content\n"
    path.write_bytes(original)
    valid = _step(1, call=_call(), result="valid-prefix-result")
    invalid = {
        "step_id": 2,
        "tool_calls": [_call(), _call("other", {})],
        "observation": {"results": []},
    }

    with pytest.raises(ValueError):
        replay_lav.convert_feed(_trajectory([valid, invalid]), str(path))

    assert path.read_bytes() == original


def test_task_gold_and_final_metrics_are_outside_the_tool_projection(tmp_path):
    step = _step(
        1, call=_call("bash", {"keystrokes": "printf stable"}),
        result=json.dumps({"returncode": 0, "output": "stable"}),
    )
    first_document = _trajectory([step])
    first_document.update({
        "task_name": "task-one", "trial_name": "trial-one",
        "reward": {"gold": "first-secret"},
    })
    first_document["trajectory"]["final_metrics"] = {
        "reward": 0, "grader": "first-private-value"}
    second_document = copy.deepcopy(first_document)
    second_document.update({
        "task_name": "different-task", "trial_name": "different-trial",
        "reward": {"gold": "second-secret"},
    })
    second_document["trajectory"]["final_metrics"] = {
        "reward": 1, "grader": "second-private-value"}
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"

    first_report = replay_lav.convert_feed(first_document, str(first))
    second_report = replay_lav.convert_feed(second_document, str(second))

    assert first.read_bytes() == second.read_bytes()
    assert first_report == second_report
    rendered = first.read_text(encoding="utf-8") + json.dumps(first_report)
    assert "secret" not in rendered
    assert "private-value" not in rendered
    assert first_report["not_preserved"] == [
        "non-tool messages", "complete result metadata"]


def test_nonfinite_argument_is_rejected_without_output(tmp_path):
    path = tmp_path / "feed.jsonl"
    document = _trajectory([_step(
        1,
        call=_call("bash", {"keystrokes": "pwd", "timeout": float("nan")}),
        result="never published",
    )])

    with pytest.raises(ValueError):
        replay_lav.convert_feed(document, str(path))

    assert not path.exists()


def test_converted_bash_command_survives_ingest_and_exact_ledger_query(
        tmp_path):
    command = "python - <<'PY'\nprint({\"exact\": [1, None, False]})\nPY"
    feed = tmp_path / "feed.jsonl"
    replay_lav.convert_feed(
        _trajectory([_step(
            1, call=_call("bash", {"keystrokes": command}),
            result=json.dumps({"output": "done", "returncode": 0}),
        )]),
        str(feed),
    )
    adapter_root = tmp_path / "adapter"
    adapter_root.mkdir()
    _state, _artifacts, log, fallback, routing = adapter_harness(adapter_root)

    assert offline.ingest_feed(str(feed), log) == 2
    result = discharge_ledger_query(routing, {
        "selector": "any",
        "value_pointer": "/input/keystrokes",
        "through_seq": 2,
        "tool_names": ["bash"],
        "expected": command,
    })

    assert result.outcome == OUTCOME_HOLDS
    assert result.instrument == "adapter:ledger.tool-call-string@1"
    assert fallback.calls == []


def test_unsupported_tool_refuses_world_replay_before_docker_or_result(
        tmp_path, monkeypatch):
    trajectory_path = tmp_path / "source.json"
    document = _trajectory([
        _step(1, call=_call("browser", {"url": "https://example.invalid"}),
              result="not replayable"),
    ])
    document.update({"trial_name": "trial-1", "task_name": "ordinary-task"})
    trajectory_path.write_text(json.dumps(document), encoding="utf-8")
    calls = []

    def forbidden_docker(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unsupported records must not start Docker")

    monkeypatch.setattr(replay_lav.replay, "docker", forbidden_docker)
    result = replay_lav.build_trial(
        str(trajectory_path), str(tmp_path / "jobs"), "unused-image")

    assert result["status"] == "unsupported-replay-tools"
    assert result["feed_conversion"]["unsupported_replay_calls"] == 1
    assert calls == []
    trial = tmp_path / "jobs" / "trial-1"
    assert (trial / "gv-state" / "hook_feed.jsonl").exists()
    assert not (trial / "result.json").exists()
