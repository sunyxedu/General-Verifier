"""mini-swe-agent ATIF -> sanitized General Verifier state."""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from experiments.terminal_bench import miniswe as minswe
from verifier.core.events import EventLog
from verifier.core.model import canonical
from verifier.core.store import ClaimStore
from verifier.offline import ingest_feed
from verifier.stream.builder import Builder

TEST_PROTOCOL = (
    minswe.TASK_WRAPPER_MARKER
    + "\n\n## Recommended Workflow\nUse bash and finish cleanly."
)


@pytest.fixture(autouse=True)
def pinned_test_protocol(monkeypatch):
    monkeypatch.setattr(
        minswe, "TASK_WRAPPER_SHA256",
        hashlib.sha256(TEST_PROTOCOL.encode("utf-8")).hexdigest())


def wrapped_task(message: str) -> str:
    return message + "\n\n\n" + TEST_PROTOCOL


def trajectory_file(tmp_path, steps, *, name="trial", **metadata):
    document = {
        "trial_name": "synthetic__trial",
        "task_name": "synthetic",
        "reward": metadata.pop("reward", "TOP-LEVEL-GOLD-MUST-NOT-LEAK"),
        "trajectory": {
            "schema_version": "atif-converted-1.0",
            "session_id": "session-1",
            "agent": {
                "name": "mini-swe-agent",
                "model_name": "deepseek/deepseek-v4-flash",
            },
            "steps": steps,
        },
        **metadata,
    }
    path = tmp_path / f"{name}_trajectory.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def user(message="Solve the task without using gold.", *, wrapped=True):
    return {
        "step_id": 1,
        "source": "user",
        "message": wrapped_task(message) if wrapped else message,
    }


def agent(step_id, calls, results, message=""):
    return {
        "step_id": step_id,
        "source": "agent",
        "message": message,
        "tool_calls": [
            {"tool_name": "bash", "arguments": {"keystrokes": command}}
            for command in calls
        ],
        "observation": {
            "results": [{"content": result} for result in results],
        },
    }


def read_feed(state):
    with open(os.path.join(state, "hook_feed.jsonl"), encoding="utf-8") \
            as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_convert_writes_only_the_trace_only_sanitized_state(tmp_path):
    source = trajectory_file(tmp_path, [
        user("  Count the tokens.  "),
        agent(2, ["wc -w input.txt"], [
            json.dumps({"returncode": 0, "output": "42 input.txt\n"}),
        ], message="The answer is 42."),
    ])
    state = tmp_path / "gv"

    counts = minswe.convert(source, str(state))

    assert counts == {"events": 2, "tool_calls": 1}
    assert {path.name for path in state.iterdir()} == {
        "hook_feed.jsonl", "response_blobs.jsonl", "task.md",
        "protocol.md", "final.txt", "world_manifest.json",
        "trace_manifest.json",
    }
    assert not (state / "world").exists()
    assert (state / "task.md").read_text() == "Count the tokens."
    assert (state / "protocol.md").read_text() == TEST_PROTOCOL
    assert (state / "final.txt").read_text() == "The answer is 42."
    assert json.loads((state / "world_manifest.json").read_text()) == {
        "captured_roots": [],
        "world_mode": "unavailable",
        "evidence": "trace-only",
    }
    manifest = json.loads((state / "trace_manifest.json").read_text())
    assert manifest["schema"] == minswe.TRACE_MANIFEST_SCHEMA
    assert manifest["evidence_mode"] == "trace-only"
    assert manifest["coverage"] == {
        "adapter_excerpt_truncations": 0,
        "agent_responses": 1,
        "harness_results": 0,
        "matched_tool_results": 1,
        "multi_tool_responses": 0,
        "pending_tool_calls": 0,
        "response_blob_records": 0,
        "source_reported_truncations": 0,
        "source_stream_claim": "all ATIF steps present in released record",
        "tool_calls": 1,
        "zero_tool_responses": 0,
    }
    output = "".join(path.read_text() for path in state.iterdir())
    assert "TOP-LEVEL-GOLD-MUST-NOT-LEAK" not in output
    assert "trial_name" not in output


def test_calls_and_results_pair_stably_and_build_effects(tmp_path):
    source = trajectory_file(tmp_path, [
        user(),
        agent(7, ["first", "second"], [
            json.dumps({"returncode": 0, "output": "one"}),
            json.dumps({"returncode": 0, "output": "two"}),
        ]),
    ])

    first = list(minswe.feed_lines(source))
    second = list(minswe.feed_lines(source))

    assert first == second
    assert [event["type"] for event in first] == [
        "tool_call", "tool_call", "tool_result", "tool_result",
    ]
    call_ids = [event["payload"]["tool_use_id"] for event in first[:2]]
    result_ids = [event["payload"]["tool_use_id"] for event in first[2:]]
    assert call_ids == result_ids
    assert len(set(call_ids)) == 2
    assert all(event["payload"]["tool"] == "bash" for event in first)
    assert all(event["payload"]["session"] == "session-1"
               for event in first)

    state = tmp_path / "gv"
    minswe.convert(source, str(state))
    log = EventLog(str(tmp_path / "events.jsonl"))
    assert ingest_feed(str(state / "hook_feed.jsonl"), log) == 4
    builder = Builder(log, ClaimStore()).replay()
    assert set(builder.labels) == {"effect:3", "effect:4"}


def test_result_uses_canonical_hash_and_bounded_excerpt(tmp_path):
    response = {"z": "x" * minswe.EXCERPT_CHARS, "a": 1}
    source = trajectory_file(tmp_path, [
        user(), agent(2, ["large-output"], [json.dumps(response)]),
    ])

    result = list(minswe.feed_lines(source))[1]["payload"]
    blob = canonical(response)

    assert result["response_sha256"] == hashlib.sha256(
        blob.encode("utf-8")).hexdigest()
    assert result["response_chars"] == len(blob)
    assert result["excerpt"] == blob[:minswe.EXCERPT_CHARS]
    assert len(result["excerpt"]) == minswe.EXCERPT_CHARS
    assert result["excerpt_truncated"] is True
    assert result["full_response_ref"] == {
        "file": minswe.RESPONSE_BLOBS_FILE,
        "sha256": result["response_sha256"],
    }

    state = tmp_path / "gv"
    minswe.convert(source, str(state))
    rows = [json.loads(line) for line in
            (state / minswe.RESPONSE_BLOBS_FILE).read_text().splitlines()]
    assert rows == [{
        "canonical_response": blob,
        "sha256": result["response_sha256"],
    }]


def test_extra_harness_result_is_retained_without_unpaired_event(tmp_path):
    real = json.dumps({"returncode": 0, "output": "real output"})
    diagnostic = (
        "Your previous response reached the output token limit before a "
        "tool call."
    )
    source = trajectory_file(tmp_path, [
        user(), agent(2, ["do-work"], [real, diagnostic]),
    ])

    events = list(minswe.feed_lines(source))

    assert [event["type"] for event in events] == [
        "tool_call", "tool_result",
    ]
    payload = events[1]["payload"]
    assert payload["excerpt"] == canonical(json.loads(real))
    assert len(payload["harness_results"]) == 1
    harness = payload["harness_results"][0]
    assert harness["excerpt"] == canonical(diagnostic)
    assert harness["response_sha256"] == hashlib.sha256(
        canonical(diagnostic).encode("utf-8")).hexdigest()


def test_observation_without_any_call_fails_instead_of_inventing_one(tmp_path):
    source = trajectory_file(tmp_path, [
        user(), agent(2, [], ["harness diagnostic"]),
    ])

    with pytest.raises(ValueError, match="no tool call"):
        list(minswe.feed_lines(source))


def test_missing_observation_remains_an_honest_pending_call(tmp_path):
    source = trajectory_file(tmp_path, [
        user(), agent(2, ["started", "never-returned"], [
            json.dumps({"returncode": 0, "output": "done"}),
        ]),
    ])

    events = list(minswe.feed_lines(source))

    assert [event["type"] for event in events] == [
        "tool_call", "tool_call", "tool_result",
    ]
    assert events[2]["payload"]["tool_use_id"] == \
        events[0]["payload"]["tool_use_id"]
    assert events[1]["payload"]["tool_use_id"] != \
        events[2]["payload"]["tool_use_id"]


def test_empty_final_uses_last_agent_action_as_completion_candidate(tmp_path):
    source = trajectory_file(tmp_path, [
        user(),
        agent(2, ["work-in-progress"], ["ok"]),
        agent(3, ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"], ["done"]),
    ])

    final = minswe.final_text(source)

    assert final.startswith(minswe.FINAL_FALLBACK_PREFIX)
    assert json.loads(final[len(minswe.FINAL_FALLBACK_PREFIX):]) == {
        "tool": "bash",
        "input": {"keystrokes": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
    }
    assert minswe.final_text(source) == final
    assert "TOP-LEVEL-GOLD-MUST-NOT-LEAK" not in final


def test_last_nonempty_agent_message_is_the_final_text(tmp_path):
    source = trajectory_file(tmp_path, [
        user(),
        agent(2, ["implement"], ["ok"], message="First draft"),
        agent(3, ["verify"], ["all tests pass"], message="Final report"),
        agent(4, ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"], ["done"]),
    ])

    assert minswe.final_text(source) == "Final report"


def test_task_instruction_skips_tiny_terminal_prompt(tmp_path):
    source = trajectory_file(tmp_path, [
        user("$ ", wrapped=False), user("Actual task statement"),
    ])

    assert minswe.task_instruction(source) == "Actual task statement"


def test_task_split_is_strict_and_protocol_is_not_task_authority(
        tmp_path, monkeypatch):
    source = trajectory_file(tmp_path, [user("Semantic contract")])

    assert minswe.task_parts(source) == ("Semantic contract", TEST_PROTOCOL)

    missing = trajectory_file(
        tmp_path, [user("No wrapper here", wrapped=False)], name="missing")
    with pytest.raises(ValueError, match="found 0"):
        minswe.task_instruction(missing)

    repeated_message = (
        wrapped_task("Task") + minswe.TASK_WRAPPER_DELIMITER + "again")
    repeated = trajectory_file(
        tmp_path, [user(repeated_message, wrapped=False)], name="repeated")
    with pytest.raises(ValueError, match="found 2"):
        minswe.task_instruction(repeated)

    monkeypatch.setattr(minswe, "TASK_WRAPPER_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="wrapper digest changed"):
        minswe.task_instruction(source)


def test_trace_manifest_keeps_zero_tool_response_and_final_boundary(tmp_path):
    source = trajectory_file(tmp_path, [
        user(),
        agent(2, [], [], message="thinking without a tool"),
        agent(3, ["work"], ["ok"], message="finished"),
        agent(4, [minswe.COMPLETION_COMMAND], ["submitted"]),
    ])
    state = tmp_path / "gv"

    minswe.convert(source, str(state))
    manifest = json.loads((state / "trace_manifest.json").read_text())

    assert manifest["coverage"]["agent_responses"] == 3
    assert manifest["response_envelopes"][0]["tool_use_ids"] == []
    assert manifest["response_envelopes"][0]["message_nonempty"] is True
    boundary = manifest["final_boundary"]
    assert boundary["last_text_step_id"] == 3
    assert boundary["selected_completion"]["step_id"] == 4
    assert boundary["selected_completion"]["result_recorded"] is True
    assert boundary["tool_calls_after_selected_completion"] == 0
    assert boundary["final_text_source"] == "last_nonempty_agent_message"


def test_trace_manifest_records_nested_source_truncation(tmp_path):
    nested = json.dumps({
        "returncode": 0,
        "output": json.dumps({"elided_chars": 17, "warning": "too long"}),
    })
    source = trajectory_file(tmp_path, [
        user(), agent(2, ["large"], [nested], message="done"),
    ])
    state = tmp_path / "gv"

    minswe.convert(source, str(state))
    manifest = json.loads((state / "trace_manifest.json").read_text())
    summary = manifest["response_envelopes"][0]["result_summaries"][0]

    assert "$.output#json.elided_chars" in summary["source_truncation_paths"]
    assert manifest["coverage"]["source_reported_truncations"] == 1
