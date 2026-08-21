"""The decomposition critic: adversarial second signature on the compiler.

Fixture-backed and instrument-free — the point under test is the loop: a
pass admits the decomposition, an insufficient verdict sends it back for
repair with the attack as feedback, and a counterexample still standing
after the last round fails closed (claim stays unverified).
"""
from __future__ import annotations

import json
import os

from verifier.core.events import EventLog
from verifier.core.model import Claim, CLAIM_ASSERTION
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.agents import CRITIC_AGENT
from verifier.llm.schema import FixtureBackend


COMPILE = lambda v: {"obligations": [{"text": f"O of: {v['claim_text']}"}]}  # noqa: E731

PASS = {"verdict": "pass",
        "counterexample_world": "n/a", "missing_condition": "n/a"}
ATTACK = {"verdict": "insufficient",
          "counterexample_world": "P holds for 2, 3 and 5 but fails at 7",
          "missing_condition": "P must hold for every prime, not a sample"}


class StubDischarger:
    def __init__(self):
        self.seen: list[str] = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.seen.append(obligation.text)
        return DischargeResult(outcome="holds", witness=None,
                               artifact="", instrument="stub")


def build(tmp_path, roles):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend(roles)
    discharger = StubDischarger()
    critic = CRITIC_AGENT if "critic" in roles else None
    Kernel(store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
           critic_role=critic, discharger=discharger)
    return store, log, backend, discharger


def ledger_events(log, etype):
    return [e for e in log.replay() if e["type"] == etype]


def test_pass_on_the_first_round_admits_the_decomposition(tmp_path):
    store, log, backend, discharger = build(
        tmp_path, {"obligations": COMPILE, "critic": PASS})

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the tree is clean"))

    assert store.status(cid) == "verified"
    assert [c["role"] for c in backend.calls] == ["obligations", "critic"]
    assert discharger.seen == ["O of: the tree is clean"]
    # the critic saw the claim, the obligations, and the premises — nothing else
    critic_vars = backend.calls[1]["variables"]
    assert set(critic_vars) == {"claim_text", "obligations_json",
                               "verified_deps"}
    assert json.loads(critic_vars["obligations_json"]) == [
        "O of: the tree is clean"]


def test_a_standing_attack_forces_one_repair_with_the_attack_as_feedback(tmp_path):
    verdicts = iter([ATTACK, PASS])
    store, log, backend, _ = build(
        tmp_path, {"obligations": COMPILE, "critic": lambda v: next(verdicts)})

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "P holds for every prime"))

    assert store.status(cid) == "verified"
    assert [c["role"] for c in backend.calls] == [
        "obligations", "critic", "obligations", "critic"]
    # the repair round carries the counterexample back to the compiler
    feedback = backend.calls[2]["variables"]["critic_feedback"]
    assert ATTACK["counterexample_world"] in feedback
    assert ATTACK["missing_condition"] in feedback
    assert backend.calls[0]["variables"]["critic_feedback"] == ""


def test_a_counterexample_standing_after_the_last_round_fails_closed(tmp_path):
    store, log, backend, discharger = build(
        tmp_path, {"obligations": COMPILE, "critic": ATTACK})

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "P holds for every prime"))

    assert store.status(cid) == "unverified"
    assert discharger.seen == []          # nothing was ever discharged
    ev = store.evidence_for(cid)[-1]
    assert ev["error"] == "decomposition failed critic"
    assert ev["counterexample_world"] == ATTACK["counterexample_world"]
    attacks = ledger_events(log, "decomposition_attacked")
    assert [a["payload"]["round"] for a in attacks] == [0, 1]
    assert all(a["payload"]["verdict"] == "insufficient" for a in attacks)
    # exactly one repair was attempted; the post-deadline repair never ran
    assert [c["role"] for c in backend.calls] == [
        "obligations", "critic", "obligations", "critic"]


def test_disabled_critic_restores_the_old_behavior(tmp_path):
    store, log, backend, _ = build(tmp_path, {"obligations": COMPILE})

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the tree is clean"))

    assert store.status(cid) == "verified"
    assert [c["role"] for c in backend.calls] == ["obligations"]
    assert ledger_events(log, "decomposition_attacked") == []
