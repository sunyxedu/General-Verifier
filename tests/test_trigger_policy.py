"""When verification fires: at commit (default) vs at first use (control arm).

Fixture-backed and instrument-free — the point under test is scheduling, so the
obligations role is a fixture and the discharger is a stub that records when it
was asked to do work.
"""
from __future__ import annotations

import os

from verifier.core.events import EventLog
from verifier.core.model import Claim, CLAIM_ASSERTION, CLAIM_EFFECT
from verifier.core.store import (
    ClaimStore, TRIGGER_ON_COMMIT, TRIGGER_ON_FIRST_DEPENDENT,
)
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend

import pytest


# one obligation per claim, derived from its text: obligation nodes are
# content-addressed, so a fixture that emits the same text for every claim
# would make every claim share (and inherit) one verdict
COMPILE = lambda v: {"obligations": [{"text": f"O of: {v['claim_text']}"}]}  # noqa: E731


class StubDischarger:
    """Records every discharge and answers with a fixed outcome."""

    def __init__(self, outcome="holds", witness=None):
        self.outcome = outcome
        self.witness = witness
        self.seen: list[str] = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.seen.append(obligation.text)
        outcome = (self.outcome(row) if callable(self.outcome)
                   else self.outcome)
        return DischargeResult(outcome=outcome, witness=self.witness,
                               artifact="", instrument="stub")


def build(tmp_path, trigger, outcome="holds", witness=None):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=trigger)
    backend = FixtureBackend({"obligations": COMPILE})
    discharger = StubDischarger(outcome, witness)
    Kernel(store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
           critic_role=None, discharger=discharger)
    return store, discharger


def test_on_commit_verifies_before_anything_depends_on_it(tmp_path):
    store, discharger = build(tmp_path, TRIGGER_ON_COMMIT)

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the tree is clean"))

    assert store.status(cid) == "verified"
    assert discharger.seen == ["O of: the tree is clean"]


def test_on_first_dependent_waits_for_a_dependent(tmp_path):
    store, discharger = build(tmp_path, TRIGGER_ON_FIRST_DEPENDENT)

    a = store.add_claim(Claim(CLAIM_ASSERTION, "the tree is clean"))
    assert store.status(a) == "unverified"
    assert discharger.seen == []

    store.add_claim(Claim(CLAIM_ASSERTION, "so the merge is done", deps=(a,)))
    assert store.status(a) == "verified"


def test_on_commit_refutes_before_the_next_claim_is_written(tmp_path):
    """The whole point: the solver learns at commit time, so a dependent claim
    is never written on top of a premise that is already false."""
    store, _ = build(tmp_path, TRIGGER_ON_COMMIT,
                     outcome="refuted", witness="line 3, not line 1")

    a = store.add_claim(Claim(CLAIM_ASSERTION, "master rewrote line 1"))
    assert store.status(a) == "refuted"

    # anything built on it afterwards fails closed rather than being checked
    b = store.add_claim(Claim(CLAIM_ASSERTION, "so resolve it this way", deps=(a,)))
    assert store.status(b) == "unverified"
    assert store.evidence_for(b)[-1]["error"] == "depends on refuted claim"


def test_effects_stay_lazy_under_on_commit(tmp_path):
    """Effect claims are the ledger — hundreds per run, and only interesting
    once a claim leans on one."""
    store, _ = build(tmp_path, TRIGGER_ON_COMMIT)

    e = store.add_claim(Claim(CLAIM_EFFECT, '{"event_seq":3,"tool":"Bash"}'))
    assert store.status(e) == "unverified"


def test_revision_after_an_immediate_refutation(tmp_path):
    """The loop the rule is meant to tighten: say it, get refuted on the spot,
    revise, get verified — with nothing ever built on the bad version."""
    refute_line_1 = lambda row: (                                   # noqa: E731
        "refuted" if "line 1" in row["text"] else "holds")
    store, _ = build(tmp_path, TRIGGER_ON_COMMIT, outcome=refute_line_1)

    bad = store.add_claim(Claim(CLAIM_ASSERTION, "master rewrote line 1"))
    assert store.status(bad) == "refuted"

    good = store.add_claim(
        Claim(CLAIM_ASSERTION, "master rewrote line 3", revises=bad))
    assert store.status(good) == "verified"
    assert store.status(bad) == "stale"      # superseded, not still refuted


def test_effect_is_checked_when_a_claim_leans_on_it(tmp_path):
    """The other half of the carve-out: a past-tense claim citing effect:<seq>
    still forces the ledger check, at the moment it is committed."""
    from verifier.stream.builder import Builder

    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": COMPILE})
    Kernel(store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
           critic_role=None, discharger=StubDischarger())
    builder = Builder(log, store)

    builder.feed(log.append("tool_call", {"tool": "Bash"}))
    builder.feed(log.append("tool_result", {"stdout": "ok"}))
    effect = builder.labels["effect:2"]
    assert store.status(effect) == "unverified"

    builder.feed(log.append("commit", {
        "label": "A1", "kind": "assertion",
        "text": "as of effect:2, the suite passed", "deps": ["effect:2"]}))

    assert store.status(effect) == "verified"      # ledger check ran
    assert store.status(builder.labels["A1"]) == "verified"


def test_unknown_trigger_policy_is_rejected():
    with pytest.raises(ValueError):
        ClaimStore(trigger="whenever")


# ---- the MCP surface: the verdict and its residue ride on gv.commit ----------

def service(tmp_path, outcome="holds", witness=None):
    from verifier.mcp_server import Service

    discharger = StubDischarger(outcome, witness)

    def factory(sc):
        backend = FixtureBackend({"obligations": COMPILE})
        return Kernel(sc.store, sc.log, backend, sc.artifacts,
                      critic_role=None, discharger=discharger)

    return Service(str(tmp_path / "gv"), kernel_factory=factory), discharger


def test_commit_returns_the_verdict_it_just_produced(tmp_path):
    sc, discharger = service(tmp_path)

    out = sc.commit("A1", "assertion", "the tree is clean", [])

    assert out["status"] == "verified"
    assert discharger.seen == ["O of: the tree is clean"]


def test_commit_hands_back_the_counterexample(tmp_path):
    sc, _ = service(tmp_path, outcome="refuted", witness="line 3, not line 1")

    out = sc.commit("A1", "assertion", "master rewrote line 1", [])

    assert out["status"] == "refuted"
    assert out["obligations"][0]["reason"] == "line 3, not line 1"


def test_a_check_that_blows_up_still_returns_the_claim(tmp_path):
    """Verification now runs inside the commit call, so a crashing role must
    fail closed — not lose the claim the solver just made."""
    from verifier.mcp_server import Service

    def factory(sc):
        return Kernel(sc.store, sc.log, FixtureBackend({}), sc.artifacts,
                      discharger=StubDischarger())   # no compile fixture -> RoleError

    sc = Service(str(tmp_path / "gv"), kernel_factory=factory)
    out = sc.commit("A1", "assertion", "the tree is clean", [])

    assert out["status"] == "unverified"
    assert "obligations" in out["error"]
    assert sc.status("A1")["status"] == "unverified"


def test_commit_on_a_refuted_premise_says_why_it_could_not_be_checked(tmp_path):
    sc, _ = service(tmp_path, outcome="refuted", witness="line 3, not line 1")
    sc.commit("A1", "assertion", "master rewrote line 1", [])

    out = sc.commit("A2", "assertion", "so resolve it this way", ["A1"])

    # no obligation was ever compiled for A2 — without the fail-closed reason
    # the solver would see a bare `unverified` and have nothing to act on
    assert out["status"] == "unverified"
    assert out["error"] == "depends on refuted claim"
    assert out["refuted_deps"]
