"""Every seat verifying a claim reads the same immutable ledger prefix.

The verifier appends its own audit events to the ledger while seats are
reading it.  Without a frozen record, an executor's complete scan and the
independent Judger's recount disagree by exactly those appends, and a
correct holds is redone as "stale".  These tests pin the boundary.
"""
from __future__ import annotations

import copy
import json
import os
import stat

import pytest

from verifier.core.events import RECORD_DIRECTORY, EventLog
from verifier.llm.schema import FixtureBackend
from verifier.app.assembly import build_kernel
from verifier.mcp_server import Service


def _ledger(tmp_path, count):
    log = EventLog(str(tmp_path / "events.jsonl"))
    for index in range(count):
        log.append("message", {"text": f"record-{index}"})
    return log


def _raw_lines(path):
    with open(path, "rb") as handle:
        return handle.read().split(b"\n")[:-1]


def test_freeze_record_publishes_the_exact_prefix_read_only(tmp_path):
    log = _ledger(tmp_path, 5)
    frozen = log.freeze_record(3)

    assert os.path.dirname(frozen) == str(tmp_path / RECORD_DIRECTORY)
    assert EventLog.record_boundary_of(frozen) == 3
    assert _raw_lines(frozen) == _raw_lines(log.path)[:3]
    assert not (stat.S_IMODE(os.stat(frozen).st_mode) & 0o222)
    # idempotent, and unaffected by later appends
    assert log.freeze_record(3) == frozen
    log.append("status_change", {"claim": "c", "status": "verified"})
    assert log.freeze_record(3) == frozen
    assert _raw_lines(frozen) == _raw_lines(log.path)[:3]
    assert len(_raw_lines(log.path)) == 6


def test_freeze_record_rejects_invalid_or_tampered_boundaries(tmp_path):
    log = _ledger(tmp_path, 2)
    for bad in (-1, True, "1", 1.0):
        with pytest.raises(ValueError):
            log.freeze_record(bad)
    with pytest.raises(ValueError, match="beyond the ledger"):
        log.freeze_record(3)
    empty = log.freeze_record(0)
    assert os.path.getsize(empty) == 0

    frozen = log.freeze_record(2)
    os.chmod(frozen, 0o644)
    with open(frozen, "ab") as handle:
        handle.write(b'{"seq":3,"type":"message","payload":{}}\n')
    with pytest.raises(ValueError, match="does not match its name"):
        log.freeze_record(2)


def test_record_path_defaults_to_the_live_ledger(tmp_path):
    log = _ledger(tmp_path, 1)
    assert log.record_path_for("unknown") == os.path.abspath(log.path)
    assert log.record_path_for(None) == os.path.abspath(log.path)
    frozen = log.freeze_record(1)
    log.bind_record("claim", frozen)
    assert log.record_path_for("claim") == frozen
    assert log.is_record_path(frozen)
    assert not log.is_record_path(os.path.abspath(log.path))


PASS = {"verdict": "pass", "counterexample_world": "n/a",
        "missing_condition": "n/a"}
JUDGE = {"sufficient": "yes", "faithful": "yes", "degradation": "n/a",
         "reason": "the retained record decides the obligation"}
METHOD = {
    "procedure": "scan the retained record for the recorded command",
    "artifact": "a retained complete scan report",
    "level": "exact", "basis": "record", "strength": "decisive",
    "bridges": [],
}


def _fixtures():
    return {
        "obligations": {"obligations": [{
            "text": "the retained record contains one issued command",
            "required_basis": "record", "severity": "must",
            "clause": "Issue one command.", "refutable_by_absence": True,
        }]},
        "critic": copy.deepcopy(PASS),
        "oracles": {"methods": [copy.deepcopy(METHOD)]},
        "discharge": {
            "outcome": "holds", "witness": "one command is recorded",
            "artifact": "complete scan", "assumed_premises": [],
            "witness_kind": "none",
            "inputs": [{"ref": "ledger:events", "provenance": "ledger",
                        "role": "subject"}],
        },
        "judger": copy.deepcopy(JUDGE),
    }


def _service(tmp_path, backend, monkeypatch):
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "0")
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text("Issue one command.\n", encoding="utf-8")
    with open(state / "hook_feed.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "tool_call", "payload": {
            "tool": "bash", "input": {"command": "true"},
            "tool_use_id": "call-1"}}) + "\n")
        handle.write(json.dumps({"type": "tool_result", "payload": {
            "tool": "bash", "tool_use_id": "call-1", "excerpt": ""}}) + "\n")
    return Service(
        str(state), live_world=False,
        kernel_factory=lambda sc: build_kernel(sc, None, backend=backend),
    )


def test_every_seat_reads_the_claim_commit_prefix(tmp_path, monkeypatch):
    backend = FixtureBackend(_fixtures())
    service = _service(tmp_path, backend, monkeypatch)
    service.drain_hooks()

    result = service.commit("c1", "assertion",
                            "the run issued one command", ["effect:2"])
    assert result["status"] == "verified"

    events = list(service.log.replay())
    commit = next(e for e in events if e["type"] == "commit")
    frozen_events = [e for e in events if e["type"] == "record_frozen"]
    assert len(frozen_events) == 1
    assert frozen_events[0]["payload"]["through_seq"] == commit["seq"]
    frozen = os.path.join(service.state_dir, RECORD_DIRECTORY,
                          frozen_events[0]["payload"]["record"])
    assert EventLog.record_boundary_of(frozen) == commit["seq"]
    # The record ends at the claim's own commit: no verifier audit event and
    # no later append is inside it, whatever was appended afterwards.
    frozen_rows = [json.loads(line) for line in _raw_lines(frozen)]
    assert [row["type"] for row in frozen_rows] == [
        "tool_call", "tool_result", "commit"]
    assert frozen_rows == events[:commit["seq"]]
    assert any(e["type"] == "judger_verdict" for e in events)

    seats = {call["role"]: call["variables"] for call in backend.calls}
    assert set(seats) >= {"obligations", "critic", "oracles", "discharge",
                          "judger"}
    for role in ("obligations", "critic", "oracles", "discharge", "judger"):
        assert seats[role]["ledger_path"] == frozen, role
    assert service.log.record_path_for(result["claim_id"]) == frozen


def test_second_claim_gets_its_own_later_boundary(tmp_path, monkeypatch):
    backend = FixtureBackend(_fixtures())
    service = _service(tmp_path, backend, monkeypatch)
    service.drain_hooks()
    first = service.commit("c1", "assertion", "first statement", ["effect:2"])
    second = service.commit("c2", "assertion", "second statement", ["c1"])
    assert first["status"] == second["status"] == "verified"

    events = list(service.log.replay())
    frozen = [e["payload"] for e in events if e["type"] == "record_frozen"]
    commits = [e["seq"] for e in events if e["type"] == "commit"]
    assert [f["through_seq"] for f in frozen] == commits
    assert frozen[0]["record"] != frozen[1]["record"]
    later = os.path.join(service.state_dir, RECORD_DIRECTORY,
                         frozen[1]["record"])
    rows = [json.loads(line) for line in _raw_lines(later)]
    # the second record includes the first claim's audit trail verbatim
    assert rows == events[:commits[1]]


def test_publication_failure_keeps_the_live_ledger(tmp_path, monkeypatch):
    backend = FixtureBackend(_fixtures())
    service = _service(tmp_path, backend, monkeypatch)
    service.drain_hooks()

    def broken(_through_seq):
        raise OSError("read-only state")

    monkeypatch.setattr(service.log, "freeze_record", broken)
    result = service.commit("c1", "assertion", "statement", ["effect:2"])
    assert result["status"] == "verified"
    events = list(service.log.replay())
    frozen = [e["payload"] for e in events if e["type"] == "record_frozen"]
    assert frozen and frozen[0]["error"] == "read-only state"
    assert "record" not in frozen[0]
    live = os.path.abspath(service.log.path)
    for call in backend.calls:
        assert call["variables"]["ledger_path"] == live, call["role"]
