"""Speculative discharge: the compiler's obligations are settled while the
critic reviews them. Kept obligations are already settled when the review
ends (content-addressed nodes), added ones are settled afterwards, dropped
ones lose their edge to the claim, and the claim's verdict folds over the
reviewed decomposition only. A critic that fails closed leaves the claim
unverified with no obligation edges.
"""
from __future__ import annotations

import os
import threading

from verifier.core.events import EventLog
from verifier.core.model import Claim, CLAIM_ASSERTION
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.agents import CRITIC_AGENT
from verifier.llm.schema import FixtureBackend


def compiler(first, repaired):
    def role(variables):
        return repaired if variables["critic_feedback"] else first
    return role


def obligations(*texts):
    return {"obligations": [{"text": t, "required_basis": "world"}
                            for t in texts]}


ATTACK = {"verdict": "insufficient",
          "counterexample_world": "SUFFICIENCY GAP: y is missing",
          "missing_condition": "APPEND OBLIGATION: y"}
PASS = {"verdict": "pass", "counterexample_world": "n/a",
        "missing_condition": "n/a"}


class Discharger:
    """Records what it settled, and lets a test hold the discharge until
    the critic has spoken (to prove the overlap really happened)."""
    def __init__(self, outcomes=None):
        self.seen = []
        self.outcomes = outcomes or {}
        self.gate = threading.Event()
        self.gate.set()

    def run(self, row, obligation):
        return self.discharge(row, obligation)

    def discharge(self, row, obligation, method=None, tag=""):
        self.gate.wait(5)
        self.seen.append(obligation.text)
        outcome = self.outcomes.get(obligation.text, "holds")
        return DischargeResult(
            witness_kind="execution", outcome=outcome,
            witness=("bad" if outcome == "refuted" else None),
            artifact="", instrument="stub", basis="world")


def build(tmp_path, roles, discharger, speculative=True):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend(roles)
    Kernel(store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
           critic_role=CRITIC_AGENT, discharger=discharger,
           binder=discharger, speculative_discharge=speculative,
           critic_rounds_nonfinal=1)
    return store, log, backend


def test_kept_obligations_settle_during_review_and_added_ones_after(tmp_path):
    d = Discharger()
    store, log, backend = build(
        tmp_path,
        {"obligations": compiler(obligations("x", "z"),
                                 obligations("x", "z", "y")),
         "critic": ATTACK}, d)

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the claim"))

    assert store.status(cid) == "verified"
    # x and z were settled once (during the review), y once (after it)
    assert sorted(d.seen) == ["x", "y", "z"]
    edges = {o["text"] for o in store.obligation_children(cid)}
    assert edges == {"x", "y", "z"}
    spec = [e["payload"] for e in log.replay()
            if e["type"] == "speculative_discharge"]
    assert spec and len(spec[0]["obligations"]) == 2


def test_dropped_obligations_lose_their_edge_and_do_not_decide(tmp_path):
    d = Discharger(outcomes={"too strong": "refuted"})
    store, log, backend = build(
        tmp_path,
        {"obligations": compiler(obligations("x", "too strong"),
                                 obligations("x")),
         "critic": {"verdict": "insufficient",
                    "counterexample_world": "NECESSITY VIOLATION: w",
                    "missing_condition": "REMOVE OR WEAKEN: too strong"}}, d)

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the claim"))

    # the over-strong obligation was refuted speculatively, but the
    # reviewed decomposition dropped it: the claim verifies on x alone
    assert store.status(cid) == "verified"
    assert {o["text"] for o in store.obligation_children(cid)} == {"x"}
    # the node keeps its verdict as a fact about its own statement
    node = [r for r in store.db.execute(
        "SELECT status FROM claims WHERE text='too strong'")]
    assert node == [("refuted",)]


def test_a_pass_uses_the_speculative_results_as_they_are(tmp_path):
    d = Discharger()
    store, log, backend = build(
        tmp_path, {"obligations": obligations("x", "z"), "critic": PASS}, d)
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the claim"))
    assert store.status(cid) == "verified"
    assert sorted(d.seen) == ["x", "z"]
    assert [c["role"] for c in backend.calls] == ["obligations", "critic"]


def test_the_review_really_overlaps_the_checks(tmp_path):
    d = Discharger()
    d.gate.clear()          # discharges block until the critic has spoken
    order = []

    def critic(variables):
        order.append("critic")
        d.gate.set()
        return PASS

    store, log, backend = build(
        tmp_path, {"obligations": obligations("x"), "critic": critic}, d)
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the claim"))
    assert store.status(cid) == "verified"
    assert order == ["critic"] and d.seen == ["x"]


def test_critic_fail_closed_leaves_no_edges(tmp_path):
    d = Discharger()
    store, log, backend = build(
        tmp_path,
        {"obligations": obligations("x"), "critic": ATTACK}, d)
    # a final keeps the full game (critic_rounds_nonfinal applies to
    # non-finals only); make the non-final fail closed by seating no
    # advisory budget
    from verifier.engine.kernel import Kernel as K
    kernel = [cb.__self__ for cb in store._trigger_callbacks
              if isinstance(getattr(cb, "__self__", None), K)][0]
    kernel.critic_rounds_nonfinal = None
    kernel.critic_rounds = 1

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the claim"))

    assert store.status(cid) == "unverified"
    assert store.obligation_children(cid) == []
    assert d.seen == ["x"]          # the speculative check did run


def test_off_by_default_keeps_the_sequential_order(tmp_path):
    d = Discharger()
    d.gate.clear()
    seen_before_critic = []

    def critic(variables):
        seen_before_critic.extend(d.seen)
        d.gate.set()
        return PASS

    store, log, backend = build(
        tmp_path, {"obligations": obligations("x"), "critic": critic}, d,
        speculative=False)
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the claim"))
    assert store.status(cid) == "verified"
    assert seen_before_critic == [] and d.seen == ["x"]
    assert [e for e in log.replay() if e["type"] == "speculative_discharge"] == []
