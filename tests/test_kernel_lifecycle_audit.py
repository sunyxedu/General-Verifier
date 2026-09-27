"""Claim lifecycle under concurrency: a check never settles a claim (or an
obligation node) that a revision invalidated while it ran, the ledger
reports exactly the status the store holds, one failing check does not
starve its batch, a node wait does not freeze the world epoch, a seat
outage does not mark a world as fully checked, a final's verdicts are not
reused for another candidate, and a final arriving during the contract
review speculates on the compiler's draft with a bounded wait.

Fixture-backed and instrument-free.
"""
from __future__ import annotations

import os
import threading

import pytest

from verifier.core.events import EventLog
from verifier.core.model import Claim, CLAIM_ASSERTION, CLAIM_FINAL
from verifier.core.store import ClaimStore, TRIGGER_LAZY, TRIGGER_ON_COMMIT
from verifier.engine import kernel as kernel_module
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.agents import CRITIC_AGENT, SCREENER_AGENT
from verifier.llm.schema import FixtureBackend


WAIT = 10
ONE = {"obligations": [{"text": "the file holds 15",
                        "required_basis": "world"}]}


class Gate:
    """Discharger whose first call blocks until released."""

    def __init__(self, outcome="holds", block=True):
        self.outcome = outcome
        self.block = block
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append((row["id"], obligation.text))
        if self.block and len(self.calls) == 1:
            self.entered.set()
            assert self.release.wait(WAIT)
        return DischargeResult(
            outcome=self.outcome, witness="observed 14", artifact="d",
            instrument="stub", basis="world", strength="decisive",
            witness_kind="world_state")


def build(tmp_path, discharger, roles=None, trigger=TRIGGER_LAZY, **kw):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=trigger)
    backend = FixtureBackend(roles or {"obligations": ONE})
    kw.setdefault("critic_role", None)
    kernel = Kernel(store, log, backend,
                    ArtifactStore(os.path.join(tmp_path, "art")),
                    discharger=discharger, **kw)
    return store, log, backend, kernel


def changes(log, claim=None, key="claim"):
    return [e["payload"] for e in log.replay()
            if e["type"] == "status_change"
            and (claim is None or e["payload"].get(key) == claim)]


def in_thread(fn, *args):
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()
    return t


# 1. a superseded claim's in-flight check: the ledger tells the truth
def test_a_revised_claims_late_verdict_is_refused_and_reported(tmp_path):
    gate = Gate(outcome="refuted")
    store, log, _, kernel = build(tmp_path, gate)
    a = store.add_claim(Claim(CLAIM_ASSERTION, "run.py bounds concurrency"))
    t = in_thread(kernel.verify, a)
    assert gate.entered.wait(WAIT)
    store.add_claim(Claim(CLAIM_ASSERTION, "run.py bounds it, revised",
                          revises=a))
    gate.release.set()
    t.join(WAIT)

    assert store.status(a) == "stale"
    node, = store.obligation_children(a)
    assert node["status"] == "stale"     # the node's settlement refused too
    cone = changes(log, a)[0]
    assert cone["status"] == "stale" and cone["reason"] == "cone"
    last = changes(log, a)[-1]
    assert last["status"] == "stale"
    assert last["proposed"] == "refuted"
    assert last["settle_refused"] == "superseded by a revision"
    # obligation nodes are reported under their own key
    assert changes(log, node["id"], key="obligation")[0]["reason"] == "cone"
    # no refutation evidence lands on a claim the store never refuted
    assert not any(e.get("refuting_obligation")
                   for e in store.evidence_for(a))


# 4. a consumer re-staled under its check stays stale, and is rechecked
def test_a_consumer_restaled_under_its_check_is_not_settled(tmp_path):
    gate = Gate()
    store, log, _, kernel = build(tmp_path, gate)
    p = store.add_claim(Claim(CLAIM_ASSERTION, "the premise"))
    c = store.add_claim(Claim(CLAIM_ASSERTION, "the consumer", deps=(p,)))
    t = in_thread(kernel.verify, c)
    assert gate.entered.wait(WAIT)
    store.add_claim(Claim(CLAIM_ASSERTION, "the premise, revised",
                          revises=p))
    gate.release.set()
    t.join(WAIT)

    assert store.status(c) == "stale"
    assert changes(log, c)[-1]["settle_refused"] == \
        "invalidated while the check ran"
    # the next request (a certificate, gv.verify) checks it afresh
    assert kernel.verify(c) == "verified"
    assert len(gate.calls) == 2


def test_settlement_refuses_a_changed_generation(tmp_path):
    store = ClaimStore(trigger=TRIGGER_LAZY)
    a = store.add_claim(Claim(CLAIM_ASSERTION, "a"))
    g = store.generation(a)
    store.set_stale(a)
    assert store.generation(a) == g + 1
    assert store.set_verified(a, generation=g) is False
    assert store.status(a) == "stale"
    assert store.set_verified(a, generation=g + 1) is True


# 2. admission and batch isolation
def test_a_screener_verdict_is_a_no_op_on_stale_or_revised(tmp_path):
    def screened(cid):
        return store.set_screened(cid, "verified", "r", basis="world",
                                  strength="decisive")
    store = ClaimStore(trigger=TRIGGER_LAZY)
    a = store.add_claim(Claim(CLAIM_ASSERTION, "a"))
    b = store.add_claim(Claim(CLAIM_ASSERTION, "b"))
    store.set_stale(a)
    assert screened(a) is False
    store.add_claim(Claim(CLAIM_ASSERTION, "b2", revises=b))
    assert screened(b) is False
    c = store.add_claim(Claim(CLAIM_ASSERTION, "c"))
    assert screened(c) is True
    assert store.status(c) == "verified" and store.get(c)["confident"]


def test_a_claim_revised_during_screening_is_not_admitted(tmp_path):
    holder = {}

    def screener(variables):
        # the revision lands while the screener reads the record
        holder["store"].add_claim(Claim(
            CLAIM_ASSERTION, "revised", revises=holder["claim"]))
        return {"verdict": "verified", "confidence": "high",
                "basis": "world", "grounds": "seq 1", "reason": "r"}

    gate = Gate(block=False)
    store, log, _, kernel = build(
        tmp_path, gate, {"obligations": ONE, "screener": screener},
        screener_role=SCREENER_AGENT)
    a = store.add_claim(Claim(CLAIM_ASSERTION, "a"))
    holder.update(store=store, claim=a)
    kernel.verify(a)          # no IllegalTransition

    row = store.get(a)
    assert row["status"] == "stale" and not row["confident"]
    assert gate.calls == []
    assert changes(log, a)[-1]["reason"] == "screener verdict refused"


def test_one_failing_check_does_not_starve_its_batch(tmp_path):
    def compile_(variables):
        if variables["claim_text"] == "x":
            raise RuntimeError("HTTP 403")
        return ONE

    gate = Gate(block=False)
    store, log, _, kernel = build(tmp_path, gate, {"obligations": compile_},
                                  trigger=TRIGGER_ON_COMMIT)
    with store.deferred_triggers() as queued:
        x = store.add_claim(Claim(CLAIM_ASSERTION, "x"))
        y = store.add_claim(Claim(CLAIM_ASSERTION, "y"))
    with pytest.raises(Exception, match="403"):
        store.fire(queued)

    assert store.status(y) == "verified"
    assert store.status(x) == "unverified"
    assert "HTTP 403" in store.evidence_for(x)[-1]["error"]
    assert changes(log, x)[-1]["reason"] == "check raised"


# 1. the other unverified branches move the store as the ledger says
def test_unverified_conclusions_are_held_by_the_store(tmp_path):
    gate = Gate(block=False)
    store, log, _, kernel = build(
        tmp_path, gate, {"obligations": {"obligations": []}})
    a = store.add_claim(Claim(CLAIM_ASSERTION, "a"))
    store.set_stale(a)
    kernel.verify(a)
    assert store.status(a) == "unverified"
    assert changes(log, a)[-1] == {"claim": a, "status": "unverified",
                                   "reason": "no obligations"}


# 3. the epoch is read once the node lock is held
def test_a_node_wait_does_not_freeze_the_world_epoch(tmp_path):
    gate = Gate()
    store, log, _, kernel = build(tmp_path, gate, live_world=True)
    waiting = threading.Event()
    real_lock_for = kernel._lock_for

    class Announcing:
        def __init__(self, lock):
            self.lock = lock

        def __enter__(self):
            if threading.current_thread().name == "second":
                waiting.set()
            return self.lock.__enter__()

        def __exit__(self, *exc):
            return self.lock.__exit__(*exc)

    def lock_for(table, key):
        lock = real_lock_for(table, key)
        return Announcing(lock) if table is kernel._node_locks else lock

    kernel._lock_for = lock_for
    first = store.add_claim(Claim(CLAIM_ASSERTION, "first"))
    second = store.add_claim(Claim(CLAIM_ASSERTION, "second"))
    t1 = in_thread(kernel.verify, first)
    assert gate.entered.wait(WAIT)
    t2 = threading.Thread(target=kernel.verify, args=(second,),
                          name="second", daemon=True)
    t2.start()
    assert waiting.wait(WAIT)
    # the solver changes the world while the second claim waits
    log.append("tool_call", {"tool": "Bash", "input": {"command": "edit"}})
    gate.release.set()
    t1.join(WAIT)
    t2.join(WAIT)

    refused = [e["payload"] for e in log.replay()
               if e["type"] == "dedup_hit"
               and e["payload"].get("refused") == "world moved"]
    assert refused and refused[0]["claim"] == second
    assert len(gate.calls) == 2       # checked again on the world as it is


# 6. a seat outage does not stamp the world as fully checked
def test_a_seat_error_leaves_no_world_fingerprint(tmp_path):
    class OutageBinder:
        def __init__(self, log):
            self.log = log

        def run(self, row, obligation):
            self.log.append("judger_call_failed",
                            {"obligation": obligation.id, "will_retry": False,
                             "error": "HTTP 403"})
            return DischargeResult(outcome="failed",
                                   witness="all candidate methods rejected",
                                   instrument="binding-game")

    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_LAZY)
    kernel = Kernel(store, log, FixtureBackend({"obligations": ONE}),
                    ArtifactStore(os.path.join(tmp_path, "art")),
                    critic_role=None, discharger=Gate(block=False),
                    binder=OutageBinder(log))
    kernel._world_fingerprint = lambda: "same world"
    a = store.add_claim(Claim(CLAIM_ASSERTION, "a"))
    assert kernel.verify(a) == "unverified"

    epoch = [e for e in store.evidence_for(a) if e["outcome"] == "epoch"][-1]
    assert epoch["world_fingerprint"] is None
    assert epoch["infrastructure_failures"] == 1
    assert not kernel._checked_on_this_world(a)


def test_a_complete_failed_check_is_not_rerun_on_the_same_world(tmp_path):
    gate = Gate(outcome="failed", block=False)
    store, log, _, kernel = build(tmp_path, gate)
    kernel._world_fingerprint = lambda: "same world"
    a = store.add_claim(Claim(CLAIM_ASSERTION, "a"))
    kernel.verify(a)
    kernel.verify(a)
    assert len(gate.calls) == 1
    # a revision in its cone reopens it: that old check no longer counts
    store.set_stale(a)
    store.set_unverified(a)
    assert not kernel._checked_on_this_world(a)


# 5. a final's verdict is about its candidate response
def test_a_final_verdict_is_not_reused_for_another_candidate(tmp_path):
    gate = Gate(block=False)
    store, log, _, kernel = build(tmp_path, gate)
    one = store.add_claim(Claim(CLAIM_FINAL, "the answer is 15"))
    two = store.add_claim(Claim(CLAIM_FINAL, "the answer is 14"))
    assert kernel.verify(one) == "verified"
    assert kernel.verify(two) == "verified"
    assert len(gate.calls) == 2
    refused = [e["payload"] for e in log.replay()
               if e["type"] == "dedup_hit" and e["payload"].get("refused")]
    assert refused[0]["refused"] == "another candidate response"


def test_the_same_candidate_reuses_its_verdict(tmp_path):
    gate = Gate(block=False)
    store, log, _, kernel = build(tmp_path, gate)
    one = store.add_claim(Claim(CLAIM_FINAL, "the answer is 15"))
    assert kernel.verify(one) == "verified"
    oid = store.obligation_children(one)[0]["id"]
    assert kernel._settle_one(store.get(one),
                              kernel_module.Obligation(one, "the file holds 15",
                                                       "world"))[0] == "holds"
    assert len(gate.calls) == 1
    assert store.status(oid) == "verified"


# 8. inheritance carries the source's epoch
def test_inheritance_stamps_the_source_epoch(tmp_path):
    gate = Gate(block=False)
    roles = {"obligations": {"obligations": [
        {"text": "the source statement", "required_basis": "record"}]}}
    store, log, _, kernel = build(tmp_path, gate, roles, live_world=True)
    src = store.add_claim(Claim(CLAIM_ASSERTION, "the source statement"))
    store.record_evidence(src, {"outcome": "epoch", "world_epoch": 0})
    store.set_verified(src, basis="record")
    log.append("tool_call", {"tool": "Bash", "input": {"command": "edit"}})
    c = store.add_claim(Claim(CLAIM_ASSERTION, "relies on the source"))
    kernel.verify(c)
    node, = store.obligation_children(c)
    inherited = store.evidence_for(node["id"])[-1]
    assert inherited["instrument"] == "graph"
    assert inherited["world_epoch"] == 0
    assert gate.calls == []


# 7. a final arriving during the contract review speculates on the draft
CONTRACT = {"obligations": [
    {"text": "the deliverable exists at the task's address",
     "required_basis": "world", "refutable_by_absence": True},
    {"text": "the deliverable's content is what the task requires",
     "required_basis": "world"},
]}
PASS = {"verdict": "pass", "counterexample_world": "n/a",
        "missing_condition": "n/a"}


def test_a_final_arriving_during_review_speculates_on_the_draft(tmp_path):
    gate = threading.Event()
    seen = []

    class Binder:
        """Discharger and binder in one: records the obligations it
        settles and releases the critic once the draft was speculated."""
        def run(self, row, obligation):
            seen.append(obligation.text)
            if len(seen) >= len(CONTRACT["obligations"]):
                gate.set()
            return DischargeResult(witness_kind="execution", outcome="holds",
                                   witness=None, artifact="d",
                                   instrument="stub", basis="world")

        def discharge(self, row, obligation, method=None, tag=""):
            return self.run(row, obligation)

    def critic(variables):
        gate.wait(WAIT)
        return PASS

    (tmp_path / "task.md").write_text("Deliver the thing")
    binder = Binder()
    store, log, backend, kernel = build(
        tmp_path, binder, {"obligations": CONTRACT, "critic": critic},
        trigger=TRIGGER_ON_COMMIT, critic_role=CRITIC_AGENT, binder=binder,
        speculative_discharge=True)
    kernel.start_final_precompile()
    for _ in range(200):
        if getattr(kernel, "_draft_contract", None):
            break
        threading.Event().wait(0.05)
    assert kernel._draft_contract

    final = store.add_claim(Claim(CLAIM_FINAL, "the task is done"))

    assert store.status(final) == "verified"
    assert sorted(seen) == sorted(o["text"] for o in CONTRACT["obligations"])
    spec = [e["payload"] for e in log.replay()
            if e["type"] == "speculative_discharge"]
    assert spec and spec[0]["source"] == "precompile draft"
    hits = [e["payload"] for e in log.replay()
            if e["type"] == "contract_cache_hit"]
    assert hits[0]["claim"] == final
    assert [c["role"] for c in backend.calls] == ["obligations", "critic"]


def test_the_wait_for_a_precompile_under_review_is_bounded(tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(kernel_module, "PRECOMPILE_WAIT_SECONDS", 0.2)
    stuck = threading.Event()
    critic_calls = []

    def critic(variables):
        critic_calls.append(1)
        if len(critic_calls) == 1:
            stuck.wait(WAIT)          # the precompile's review hangs
        return PASS

    (tmp_path / "task.md").write_text("Deliver the thing")
    binder = Gate(block=False)
    binder.run = lambda row, obligation: binder.discharge(row, obligation)
    store, log, backend, kernel = build(
        tmp_path, binder, {"obligations": CONTRACT, "critic": critic},
        trigger=TRIGGER_ON_COMMIT, critic_role=CRITIC_AGENT, binder=binder)
    kernel.start_final_precompile()
    for _ in range(200):
        if critic_calls:
            break
        threading.Event().wait(0.05)

    final = store.add_claim(Claim(CLAIM_FINAL, "the task is done"))
    stuck.set()

    assert store.status(final) == "verified"
    timed_out = [e["payload"] for e in log.replay()
                 if e["type"] == "contract_precompiled"
                 and e["payload"].get("claim") == final]
    assert timed_out and "compiled inline" in timed_out[0]["reason"]
    assert [c["role"] for c in backend.calls].count("obligations") == 2
