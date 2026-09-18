"""World epoch: a world-basis verdict is about the world as it was when
the check ran. A world-mutating solver action advances the epoch; a
world-basis obligation node settled in an earlier epoch is reopened and
checked afresh instead of reused, while record-basis verdicts (about the
immutable ledger) are reused across epochs. Offline the world is fixed.
"""
from __future__ import annotations

import os

from verifier.core.events import EventLog
from verifier.core.model import Claim, CLAIM_ASSERTION
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.core.world import tool_mutates_world, world_epoch
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend
from verifier.mcp_server import _residue


def compile_role(basis):
    return lambda v: {"obligations": [
        {"text": "the delivered file has the required content",
         "required_basis": basis}]}


class Discharger:
    def __init__(self, basis):
        self.basis = basis
        self.seen = 0

    def discharge(self, row, obligation, method=None, tag=""):
        self.seen += 1
        return DischargeResult(witness_kind="execution", outcome="holds",
                               witness=None, artifact="", instrument="stub",
                               basis=self.basis)


def build(tmp_path, basis, live_world=True):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    d = Discharger(basis)
    Kernel(store, log, FixtureBackend({"obligations": compile_role(basis)}),
           ArtifactStore(os.path.join(tmp_path, "art")), critic_role=None,
           discharger=d, live_world=live_world)
    return store, log, d


def test_tool_kinds_and_epoch_count():
    assert not tool_mutates_world("Read")
    assert not tool_mutates_world("mcp__gv__commit")
    assert tool_mutates_world("Bash") and tool_mutates_world("Write")
    assert tool_mutates_world(None)   # unknown reads as mutating
    events = [{"type": "tool_call", "payload": {"tool": "Read"}},
              {"type": "tool_call", "payload": {"tool": "Bash"}},
              {"type": "tool_result", "payload": {"tool": "Bash"}},
              {"type": "tool_call", "payload": {"tool": "Edit"}}]
    assert world_epoch(events) == 2


def test_world_verdict_is_reopened_after_the_world_moves(tmp_path):
    store, log, d = build(tmp_path, "world")
    a = store.add_claim(Claim(CLAIM_ASSERTION, "first"))
    assert store.status(a) == "verified" and d.seen == 1
    # same obligation, same epoch: reused
    b = store.add_claim(Claim(CLAIM_ASSERTION, "second"))
    assert store.status(b) == "verified" and d.seen == 1

    log.append("tool_call", {"tool": "Bash", "input": {"command": "sed -i"}})
    c = store.add_claim(Claim(CLAIM_ASSERTION, "third"))

    assert store.status(c) == "verified"
    assert d.seen == 2                      # checked afresh
    refused = [e["payload"] for e in log.replay()
               if e["type"] == "dedup_hit" and e["payload"].get("refused")]
    assert refused and refused[0]["refused"] == "world moved"
    node = store.obligation_children(c)[0]
    assert node["status"] == "verified"
    assert store.evidence_for(node["id"])[-1]["world_epoch"] == 1


def test_record_verdicts_survive_the_world_moving(tmp_path):
    store, log, d = build(tmp_path, "record")
    store.add_claim(Claim(CLAIM_ASSERTION, "first"))
    log.append("tool_call", {"tool": "Write", "input": {}})
    store.add_claim(Claim(CLAIM_ASSERTION, "second"))
    assert d.seen == 1


def test_offline_world_is_fixed(tmp_path):
    store, log, d = build(tmp_path, "world", live_world=False)
    store.add_claim(Claim(CLAIM_ASSERTION, "first"))
    log.append("tool_call", {"tool": "Bash", "input": {}})
    store.add_claim(Claim(CLAIM_ASSERTION, "second"))
    assert d.seen == 1


def test_residue_names_what_still_holds(tmp_path):
    def compile_two(v):
        return {"obligations": [
            {"text": "the file exists", "required_basis": "world"},
            {"text": "the file is well-formed", "required_basis": "world",
             "task_bearing": "beyond_task"}]}

    class Mixed:
        def discharge(self, row, obligation, method=None, tag=""):
            bad = "well-formed" in obligation.text
            return DischargeResult(
                witness_kind="execution",
                outcome="refuted" if bad else "holds",
                witness="parse error at byte 3" if bad else None,
                artifact="", instrument="stub", basis="world")

    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    Kernel(store, log, FixtureBackend({"obligations": compile_two}),
           ArtifactStore(os.path.join(tmp_path, "art")), critic_role=None,
           discharger=Mixed())
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the delivery is right"))
    assert store.status(cid) == "refuted"
    residue = _residue(store, cid)
    assert [o["text"] for o in residue["obligations"]] == ["the file is well-formed"]
    assert residue["obligations"][0]["required_by_task"] == "beyond_task"
    assert residue["still_holding"] == ["the file exists"]
