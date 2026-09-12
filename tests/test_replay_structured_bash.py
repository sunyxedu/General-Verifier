"""Compatibility tests for source-preserving bash records."""
from __future__ import annotations

import json

from experiments.terminal_bench import lav, replay


def _write_feed(path, events):
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def _call(tool, input_value, call_id="call-1"):
    return {
        "type": "tool_call",
        "payload": {
            "tool": tool,
            "tool_use_id": call_id,
            "input": input_value,
        },
    }


def _result(tool, output="done", call_id="call-1"):
    return {
        "type": "tool_result",
        "payload": {
            "tool": tool,
            "tool_use_id": call_id,
            "excerpt": json.dumps([{"type": "input_text", "text": output}]),
        },
    }


def test_render_trace_extracts_explicit_bash_keystrokes(tmp_path):
    _write_feed(tmp_path / "hook_feed.jsonl", [
        _call("bash", {"keystrokes": "printf 'kept\\n'"}),
        _result("bash", "kept\n"),
    ])
    (tmp_path / "final.txt").write_text("finished", encoding="utf-8")

    rendered = lav.render_trace(str(tmp_path))

    assert "[Command] printf 'kept\\n'" in rendered
    assert '"keystrokes"' not in rendered
    assert "--- Agent Final Message ---\nfinished" in rendered


def test_render_trace_preserves_legacy_and_non_bash_rendering(tmp_path):
    legacy = "const r = await tools.exec_command({\"cmd\":\"echo old\"});"
    _write_feed(tmp_path / "hook_feed.jsonl", [
        _call("exec", legacy, "legacy"),
        _result("exec", "old", "legacy"),
        _call("other", {"keystrokes": "must-not-special-case"}, "other"),
        _result("other", "ignored", "other"),
    ])

    rendered = lav.render_trace(str(tmp_path))

    assert f"[Command] {legacy}" in rendered
    assert '[Command] {"keystrokes": "must-not-special-case"}' in rendered


def test_render_trace_renders_non_object_json_inputs_without_crashing(tmp_path):
    _write_feed(tmp_path / "hook_feed.jsonl", [
        _call("other", ["alpha", 2], "list"),
        _result("other", "list-output", "list"),
        _call("other", 7, "integer"),
        _result("other", "integer-output", "integer"),
        _call("other", True, "boolean"),
        _result("other", "boolean-output", "boolean"),
        _call("other", None, "null"),
        _result("other", "null-output", "null"),
    ])

    rendered = lav.render_trace(str(tmp_path))

    assert '[Command] ["alpha", 2]' in rendered
    assert "[Command] 7" in rendered
    assert "[Command] true" in rendered
    assert "null-output" in rendered


def test_parse_feed_accepts_only_structured_bash_keystrokes(tmp_path):
    feed = tmp_path / "feed.jsonl"
    _write_feed(feed, [
        _call("bash", {"keystrokes": "printf new"}, "new"),
        _result("bash", "new-output", "new"),
        _call("other", {"keystrokes": "printf forbidden"}, "other"),
        _result("other", "other-output", "other"),
        _call("bash", {"command": "printf wrong-key"}, "malformed"),
        _result("bash", "malformed-output", "malformed"),
    ])

    parsed = list(replay.parse_feed(str(feed)))

    assert parsed[0] == (
        "exec", {"cmd": "printf new", "workdir": "/app"}, "new-output")
    assert parsed[1][0] == "skip"
    assert parsed[1][1]["reason"] == "unsupported tool 'other'"
    assert parsed[2][0] == "skip"
    assert parsed[2][1]["reason"] == \
        "bash input lacks string keystrokes"


def test_parse_feed_preserves_legacy_exec_wrapper(tmp_path):
    feed = tmp_path / "feed.jsonl"
    wrapper = (
        'const r = await tools.exec_command({'
        '"cmd":"printf legacy","workdir":"/work"});\ntext(r.output);'
    )
    _write_feed(feed, [
        _call("exec", wrapper),
        _result("exec", "legacy-output"),
    ])

    assert list(replay.parse_feed(str(feed))) == [(
        "exec",
        {"cmd": "printf legacy", "workdir": "/work"},
        "legacy-output",
    )]


def test_parse_feed_does_not_execute_wrapper_from_another_tool(tmp_path):
    feed = tmp_path / "feed.jsonl"
    wrapper = 'tools.exec_command({"cmd":"printf forbidden"})'
    _write_feed(feed, [
        _call("other", wrapper),
        _result("other", "must-not-run"),
    ])

    parsed = list(replay.parse_feed(str(feed)))

    assert len(parsed) == 1
    assert parsed[0][0] == "skip"
    assert parsed[0][1]["reason"] == "unsupported tool 'other'"
