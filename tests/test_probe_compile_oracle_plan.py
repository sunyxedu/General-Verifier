"""No-model boundary tests for the Compiler/Critic -> Oracle probe."""
from pathlib import Path

import pytest

from experiments.terminal_bench.probe_compile_oracle_plan import _source_prefix
from verifier.core.events import EventLog


def _final(log: EventLog):
    return log.append("extracted_commit", {
        "label": "final",
        "kind": "final",
        "text": "The selected candidate satisfies the task.",
        "deps": [],
    })


def test_source_prefix_stops_before_verifier_events(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("tool_call", {"tool": "exec", "tool_use_id": "one"})
    log.append("tool_result", {"tool": "exec", "tool_use_id": "one"})
    final = _final(log)
    log.append("decomposition_attacked", {
        "claim": "0" * 64,
        "verdict": "pass",
    })

    prefix = _source_prefix(Path(tmp_path))

    assert [event["seq"] for event in prefix] == [1, 2, 3]
    assert prefix[-1] == {
        key: final[key] for key in ("seq", "type", "payload")
    }


def test_source_prefix_requires_exactly_one_final(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("extracted_commit", {
        "label": "support",
        "kind": "assertion",
        "text": "A support statement.",
        "deps": [],
    })

    with pytest.raises(ValueError, match="exactly one final"):
        _source_prefix(Path(tmp_path))
