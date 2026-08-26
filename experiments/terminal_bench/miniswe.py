"""Convert official mini-swe-agent ATIF trajectories to a sanitized GV state.

The Terminal-Bench release wraps the ATIF trajectory in a JSON object which
also contains the official reward.  The converter writes seven sanitized inputs
needed by offline GV scoring and deliberately does not copy the source JSON or
create a ``world/`` tree::

    python experiments/terminal_bench/miniswe.py TRAJECTORY.json STATE_DIR

ATIF has no tool-use IDs, so IDs are derived deterministically from the step
position and tool-call position.  Observation results pair positionally with
tool calls.  A few released traces contain an additional harness diagnostic
after the real observation.  Those diagnostics are retained as bounded
``harness_results`` metadata on the last matched result; they never become an
unmatched ``tool_result`` event.

The release's first user message concatenates the benchmark task contract with
one fixed mini-swe-agent interaction wrapper.  The two are split by an exact,
release-pinned delimiter and wrapper digest: ``task.md`` is semantic task
authority, while ``protocol.md`` is retained for audit but never fed to the GV
task compiler.  ``trace_manifest.json`` preserves response envelopes and the
submission boundary without expanding the verifier's tool-event ledger.
Tool responses beyond the ledger excerpt boundary are retained, deduplicated,
in ``response_blobs.jsonl`` and referenced by content hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterator
from typing import Any

from verifier.core.model import canonical

EXCERPT_CHARS = 30000  # same evidence boundary as hooks/post_tool_use.py
RESPONSE_BLOBS_FILE = "response_blobs.jsonl"
TRACE_MANIFEST_SCHEMA = "terminal-bench-2.1-atif-adapter-v2"
TASK_WRAPPER_MARKER = (
    "You can execute bash commands and edit files to implement the necessary "
    "changes."
)
TASK_WRAPPER_DELIMITER = "\n\n\n" + TASK_WRAPPER_MARKER
# Canonical UTF-8 bytes of the wrapper beginning at TASK_WRAPPER_MARKER.  All
# 445 official Terminal-Bench 2.1 trajectories carry this exact suffix.  A
# changed release must be reviewed instead of being split heuristically.
TASK_WRAPPER_SHA256 = (
    "bf895c9a01b25aad433dbcad7d86e2efb0b0dfb2afe1386059cb0047ec765e44"
)
COMPLETION_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
FINAL_FALLBACK_PREFIX = (
    "[No textual final response was recorded. The final recorded agent "
    "action is the completion candidate.]\n"
)
WORLD_MANIFEST = {
    "captured_roots": [],
    "world_mode": "unavailable",
    "evidence": "trace-only",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_document(trajectory_path: str) -> dict[str, Any]:
    with open(trajectory_path, encoding="utf-8", errors="replace") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise TypeError("trajectory file must contain a JSON object")
    trajectory = document.get("trajectory")
    if not isinstance(trajectory, dict):
        raise TypeError("trajectory file has no ATIF trajectory object")
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise TypeError("ATIF trajectory has no steps list")
    return document


def _load(trajectory_path: str) -> dict[str, Any]:
    document = _load_document(trajectory_path)
    trajectory = document["trajectory"]
    return trajectory


def _result_value(result: Any) -> Any:
    """Recover the tool response represented by an ATIF result wrapper."""

    raw = result.get("content") if isinstance(result, dict) \
        and set(result) == {"content"} else result
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _bounded_response(response: Any) -> dict[str, Any]:
    blob = canonical(response)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    bounded = {
        "response_sha256": digest,
        "response_chars": len(blob),
        "excerpt_truncated": len(blob) > EXCERPT_CHARS,
        "excerpt": blob[:EXCERPT_CHARS],
    }
    if bounded["excerpt_truncated"]:
        bounded["full_response_ref"] = {
            "file": RESPONSE_BLOBS_FILE,
            "sha256": digest,
        }
    return bounded


def _tool_use_id(step: dict[str, Any], step_index: int,
                 call_index: int) -> str:
    # step_id is useful to a human reading the ledger; step_index makes the ID
    # collision-free even for malformed inputs containing duplicate step IDs.
    step_id = step.get("step_id", step_index + 1)
    return f"atif:{step_id}:{step_index}:{call_index}"


def _step_io(step: dict[str, Any], step_index: int) \
        -> tuple[list[Any], list[Any]]:
    calls = step.get("tool_calls") or []
    observation = step.get("observation") or {}
    results = observation.get("results") or [] \
        if isinstance(observation, dict) else []
    if not isinstance(calls, list) or not isinstance(results, list):
        raise TypeError(
            f"ATIF step {step.get('step_id', step_index + 1)!r} has "
            "non-list tool calls or results")
    if any(not isinstance(call, dict) for call in calls):
        raise TypeError(
            f"ATIF step {step.get('step_id', step_index + 1)!r} "
            "contains a non-object tool call")
    return calls, results


def _feed_lines(trajectory: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield hook-feed events from one already-validated trajectory.

    Calls and observation results are paired positionally within each step.
    Missing observations remain visibly pending calls.  Results beyond the
    number of calls are release-harness diagnostics: they are retained as
    metadata on the last real result instead of inventing a tool invocation.
    """

    session = trajectory.get("session_id")
    agent_info = trajectory.get("agent")
    agent = agent_info.get("name") if isinstance(agent_info, dict) else None

    for step_index, step in enumerate(trajectory["steps"]):
        if not isinstance(step, dict):
            continue
        calls, results = _step_io(step, step_index)

        call_events: list[dict[str, Any]] = []
        for call_index, call in enumerate(calls):
            call_events.append({
                "type": "tool_call",
                "payload": {
                    "tool": call.get("tool_name"),
                    "tool_use_id": _tool_use_id(
                        step, step_index, call_index),
                    "input": call.get("arguments"),
                    "session": session,
                    "agent": agent,
                },
            })

        result_events: list[dict[str, Any]] = []
        for call_index, result in enumerate(results[:len(calls)]):
            response = _result_value(result)
            result_events.append({
                "type": "tool_result",
                "payload": {
                    "tool": calls[call_index].get("tool_name"),
                    "tool_use_id": _tool_use_id(
                        step, step_index, call_index),
                    **_bounded_response(response),
                    "session": session,
                    "agent": agent,
                },
            })

        extras = results[len(calls):]
        if extras:
            if not result_events:
                raise ValueError(
                    f"ATIF step {step.get('step_id', step_index + 1)!r} "
                    "has observation results but no tool call to carry them")
            result_events[-1]["payload"]["harness_results"] = [
                _bounded_response(_result_value(result)) for result in extras
            ]

        yield from call_events
        yield from result_events


def feed_lines(trajectory_path: str) -> Iterator[dict[str, Any]]:
    """Yield hook-feed events from a mini-swe-agent ATIF JSON file."""

    yield from _feed_lines(_load(trajectory_path))


def _task_message(trajectory: dict[str, Any]) -> tuple[str, int, Any]:
    """Return the first substantive user message and its source position."""

    for step_index, step in enumerate(trajectory["steps"]):
        if not isinstance(step, dict) or step.get("source") != "user":
            continue
        message = step.get("message")
        if isinstance(message, str) and message.strip():
            # Tiny ``$`` messages are terminal prompts, not task statements.
            if message.startswith("$") and len(message.strip()) < 5:
                continue
            return message, step_index, step.get("step_id")
    raise ValueError("ATIF trajectory has no substantive user task message")


def _split_task_message(message: str) -> tuple[str, str]:
    """Split the pinned TB2.1 task contract from its agent wrapper.

    This is intentionally strict.  Missing, repeated, or changed delimiters
    mean the dataset contract changed; silently treating the wrapper as task
    authority would reintroduce protocol obligations into GV correctness.
    """

    occurrences = message.count(TASK_WRAPPER_DELIMITER)
    if occurrences != 1:
        raise ValueError(
            "expected exactly one Terminal-Bench 2.1 task/wrapper delimiter; "
            f"found {occurrences}")
    task, wrapper_tail = message.split(TASK_WRAPPER_DELIMITER, 1)
    task = task.strip()
    wrapper = (TASK_WRAPPER_MARKER + wrapper_tail).strip()
    if not task:
        raise ValueError("Terminal-Bench task contract is empty after split")
    actual = _sha256_text(wrapper)
    if actual != TASK_WRAPPER_SHA256:
        raise ValueError(
            "Terminal-Bench 2.1 agent wrapper digest changed: "
            f"expected {TASK_WRAPPER_SHA256}, got {actual}")
    return task, wrapper


def task_parts(trajectory_path: str) -> tuple[str, str]:
    """Return ``(semantic task contract, retained agent protocol)``."""

    message, _step_index, _step_id = _task_message(_load(trajectory_path))
    return _split_task_message(message)


def task_instruction(trajectory_path: str) -> str:
    """Return only the authoritative semantic task contract."""

    return task_parts(trajectory_path)[0]


def _completion_candidate(trajectory: dict[str, Any]) -> str:
    for step in reversed(trajectory["steps"]):
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        calls = step.get("tool_calls") or []
        if not isinstance(calls, list) or not calls:
            continue
        actions = [
            {
                "tool": call.get("tool_name"),
                "input": call.get("arguments"),
            }
            for call in calls if isinstance(call, dict)
        ]
        if actions:
            candidate: Any = actions[0] if len(actions) == 1 else actions
            return FINAL_FALLBACK_PREFIX + canonical(candidate)
    return FINAL_FALLBACK_PREFIX + "(no agent action was recorded)"


def _final_text(trajectory: dict[str, Any]) -> str:
    last = ""
    for step in trajectory["steps"]:
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        message = step.get("message")
        if isinstance(message, str) and message.strip():
            last = message
    return last if last else _completion_candidate(trajectory)


def final_text(trajectory_path: str) -> str:
    """Return the last textual agent answer, or a deterministic action fallback."""

    return _final_text(_load(trajectory_path))


def _reported_truncation_paths(value: Any, path: str = "$", depth: int = 0,
                               budget: int = 256) -> list[str]:
    """Find explicit nested source truncation flags, including JSON strings.

    ATIF results occasionally wrap structured tool output inside another JSON
    string.  The outer adapter excerpt may be complete while an inner producer
    says its own output was truncated, so both boundaries are recorded.
    """

    if depth > 12 or budget <= 0:
        return []
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            lowered = str(key).lower()
            is_truncation_field = (
                "truncat" in lowered
                or lowered in {"elided_chars", "omitted_chars"}
            )
            if is_truncation_field and child not in (
                    False, None, 0, "", "false", "False", "0"):
                found.append(child_path)
            if len(found) >= budget:
                break
            found.extend(_reported_truncation_paths(
                child, child_path, depth + 1, budget - len(found)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_reported_truncation_paths(
                child, f"{path}[{index}]", depth + 1,
                budget - len(found)))
            if len(found) >= budget:
                break
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, (dict, list)):
                found.extend(_reported_truncation_paths(
                    decoded, f"{path}#json", depth + 1,
                    budget - len(found)))
    return found[:budget]


def _response_envelopes(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Retain response boundaries that the tool-only GV ledger cannot express."""

    envelopes: list[dict[str, Any]] = []
    for step_index, step in enumerate(trajectory["steps"]):
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        calls, results = _step_io(step, step_index)
        matched = min(len(calls), len(results))
        message = step.get("message")
        message_text = message if isinstance(message, str) else None
        result_summaries = []
        for call_index, result in enumerate(results[:matched]):
            response = _result_value(result)
            bounded = _bounded_response(response)
            result_summaries.append({
                "tool_use_id": _tool_use_id(step, step_index, call_index),
                "response_sha256": bounded["response_sha256"],
                "response_chars": bounded["response_chars"],
                "adapter_excerpt_truncated": bounded["excerpt_truncated"],
                "full_response_ref": bounded.get("full_response_ref"),
                "source_truncation_paths": _reported_truncation_paths(response),
            })
        harness_summaries = []
        for result in results[len(calls):]:
            response = _result_value(result)
            bounded = _bounded_response(response)
            harness_summaries.append({
                "response_sha256": bounded["response_sha256"],
                "response_chars": bounded["response_chars"],
                "adapter_excerpt_truncated": bounded["excerpt_truncated"],
                "full_response_ref": bounded.get("full_response_ref"),
                "source_truncation_paths": _reported_truncation_paths(response),
            })
        tool_use_ids = [
            _tool_use_id(step, step_index, call_index)
            for call_index in range(len(calls))
        ]
        envelopes.append({
            "response_index": len(envelopes),
            "step_index": step_index,
            "step_id": step.get("step_id"),
            "message_present": message_text is not None,
            "message_nonempty": bool(message_text and message_text.strip()),
            "message_sha256": (_sha256_text(message_text)
                               if message_text is not None else None),
            "message_chars": (len(message_text)
                              if message_text is not None else 0),
            "tool_use_ids": tool_use_ids,
            "tool_names": [call.get("tool_name") for call in calls],
            "observation_results": len(results),
            "matched_results": matched,
            "pending_tool_use_ids": tool_use_ids[matched:],
            "result_summaries": result_summaries,
            "harness_results": harness_summaries,
        })
    return envelopes


def _call_command(call: dict[str, Any]) -> Any:
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return None
    if "keystrokes" in arguments:
        return arguments["keystrokes"]
    return arguments.get("command")


def _final_boundary(trajectory: dict[str, Any], final: str) -> dict[str, Any]:
    agent_steps: list[tuple[int, dict[str, Any]]] = []
    text_steps: list[tuple[int, dict[str, Any]]] = []
    all_calls: list[dict[str, Any]] = []
    completion_mentions: list[dict[str, Any]] = []
    completion_calls: list[dict[str, Any]] = []
    for step_index, step in enumerate(trajectory["steps"]):
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        agent_steps.append((step_index, step))
        message = step.get("message")
        if isinstance(message, str) and message.strip():
            text_steps.append((step_index, step))
        calls, results = _step_io(step, step_index)
        for call_index, call in enumerate(calls):
            record = {
                "step_index": step_index,
                "step_id": step.get("step_id"),
                "call_index": call_index,
                "tool_use_id": _tool_use_id(step, step_index, call_index),
                "result_recorded": call_index < len(results),
            }
            all_calls.append(record)
            command = _call_command(call)
            if (call.get("tool_name") == "bash"
                    and isinstance(command, str)
                    and COMPLETION_COMMAND in command):
                mention = {**record, "exact_standalone": (
                    command == COMPLETION_COMMAND)}
                completion_mentions.append(mention)
                if mention["exact_standalone"]:
                    completion_calls.append(record)
    selected = completion_calls[-1] if completion_calls else None
    after = 0
    if selected is not None:
        selected_key = (selected["step_index"], selected["call_index"])
        after = sum(
            (record["step_index"], record["call_index"]) > selected_key
            for record in all_calls
        )
    last_agent = agent_steps[-1] if agent_steps else None
    last_text = text_steps[-1] if text_steps else None
    return {
        "last_agent_step_index": last_agent[0] if last_agent else None,
        "last_agent_step_id": last_agent[1].get("step_id")
        if last_agent else None,
        "last_text_step_index": last_text[0] if last_text else None,
        "last_text_step_id": last_text[1].get("step_id")
        if last_text else None,
        "final_text_source": ("last_nonempty_agent_message"
                              if last_text else "last_agent_action_fallback"),
        "final_text_sha256": _sha256_text(final),
        "completion_command": COMPLETION_COMMAND,
        "completion_mentions": completion_mentions,
        "completion_calls": completion_calls,
        "selected_completion": selected,
        "tool_calls_after_selected_completion": after,
    }


def _response_blobs(trajectory: dict[str, Any]) -> tuple[str, int]:
    """Serialize unique full responses whose ledger excerpts are truncated."""

    blobs: dict[str, str] = {}
    for step_index, step in enumerate(trajectory["steps"]):
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        _calls, results = _step_io(step, step_index)
        for result in results:
            response = _result_value(result)
            blob = canonical(response)
            if len(blob) <= EXCERPT_CHARS:
                continue
            digest = _sha256_text(blob)
            blobs.setdefault(digest, blob)
    rendered = "".join(json.dumps({
        "sha256": digest,
        "canonical_response": blobs[digest],
    }, ensure_ascii=False, sort_keys=True) + "\n" for digest in sorted(blobs))
    return rendered, len(blobs)


def _trace_manifest(document: dict[str, Any], task: str, protocol: str,
                    final: str, lines: list[dict[str, Any]],
                    hook_feed: str, response_blobs: str,
                    response_blob_records: int) -> dict[str, Any]:
    trajectory = document["trajectory"]
    task_message, task_step_index, task_step_id = _task_message(trajectory)
    envelopes = _response_envelopes(trajectory)
    result_summaries = [
        result
        for envelope in envelopes
        for result in envelope["result_summaries"]
    ]
    harness_summaries = [
        result
        for envelope in envelopes
        for result in envelope["harness_results"]
    ]
    no_gold = dict(document)
    no_gold.pop("reward", None)
    world_manifest = json.dumps(
        WORLD_MANIFEST, ensure_ascii=False, sort_keys=True) + "\n"
    return {
        "schema": TRACE_MANIFEST_SCHEMA,
        "evidence_mode": "trace-only",
        "source": {
            "trajectory_sha256": _sha256_text(canonical(no_gold)),
            "trajectory_schema_version": trajectory.get("schema_version"),
            "step_count": len(trajectory["steps"]),
            "session_sha256": _sha256_text(str(trajectory.get("session_id"))),
            "hash_scope": "canonical top-level document without held-out label",
        },
        "task_contract": {
            "source_step_index": task_step_index,
            "source_step_id": task_step_id,
            "source_message_sha256": _sha256_text(task_message),
            "splitter": "exact-delimiter-and-wrapper-digest-v1",
            "delimiter_sha256": _sha256_text(TASK_WRAPPER_DELIMITER),
            "task_sha256": _sha256_text(task),
            "protocol_sha256": _sha256_text(protocol),
        },
        "response_envelopes": envelopes,
        "coverage": {
            "agent_responses": len(envelopes),
            "zero_tool_responses": sum(
                not e["tool_use_ids"] for e in envelopes),
            "multi_tool_responses": sum(
                len(e["tool_use_ids"]) > 1 for e in envelopes),
            "tool_calls": sum(len(e["tool_use_ids"]) for e in envelopes),
            "matched_tool_results": len(result_summaries),
            "pending_tool_calls": sum(
                len(e["pending_tool_use_ids"]) for e in envelopes),
            "harness_results": len(harness_summaries),
            "adapter_excerpt_truncations": sum(
                bool(r["adapter_excerpt_truncated"])
                for r in [*result_summaries, *harness_summaries]),
            "response_blob_records": response_blob_records,
            "source_reported_truncations": sum(
                bool(r["source_truncation_paths"])
                for r in [*result_summaries, *harness_summaries]),
            "source_stream_claim": "all ATIF steps present in released record",
        },
        "final_boundary": _final_boundary(trajectory, final),
        "converted_files": {
            "hook_feed.jsonl": _sha256_text(hook_feed),
            "task.md": _sha256_text(task),
            "protocol.md": _sha256_text(protocol),
            "final.txt": _sha256_text(final),
            "world_manifest.json": _sha256_text(world_manifest),
            RESPONSE_BLOBS_FILE: _sha256_text(response_blobs),
        },
        "event_count": len(lines),
    }


def convert(trajectory_path: str, state_dir: str) -> dict[str, int]:
    """Write a trace-only sanitized GV state and return event counts.

    The output set is exactly ``hook_feed.jsonl``, ``response_blobs.jsonl``,
    ``task.md``, ``protocol.md``, ``final.txt``, ``world_manifest.json``, and
    ``trace_manifest.json`` when ``state_dir`` starts empty.  In particular,
    top-level reward metadata and a filesystem world are never copied into the
    verifier boundary.
    """

    # Finish parsing before creating output, so malformed ATIF cannot leave a
    # plausible-looking partial state behind.
    document = _load_document(trajectory_path)
    trajectory = document["trajectory"]
    lines: list[dict[str, Any]] = list(_feed_lines(trajectory))
    task_message, _task_step_index, _task_step_id = _task_message(trajectory)
    task, protocol = _split_task_message(task_message)
    final = _final_text(trajectory)
    hook_feed = "".join(
        json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    response_blobs, response_blob_records = _response_blobs(trajectory)
    manifest = _trace_manifest(
        document, task, protocol, final, lines, hook_feed,
        response_blobs, response_blob_records)

    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "hook_feed.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write(hook_feed)
    with open(os.path.join(state_dir, RESPONSE_BLOBS_FILE), "w",
              encoding="utf-8") as handle:
        handle.write(response_blobs)
    with open(os.path.join(state_dir, "task.md"), "w",
              encoding="utf-8") as handle:
        handle.write(task)
    with open(os.path.join(state_dir, "protocol.md"), "w",
              encoding="utf-8") as handle:
        handle.write(protocol)
    with open(os.path.join(state_dir, "final.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(final)
    with open(os.path.join(state_dir, "world_manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump(WORLD_MANIFEST, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(state_dir, "trace_manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    return {
        "events": len(lines),
        "tool_calls": sum(1 for line in lines
                          if line["type"] == "tool_call"),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    print(json.dumps(convert(sys.argv[1], sys.argv[2])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
