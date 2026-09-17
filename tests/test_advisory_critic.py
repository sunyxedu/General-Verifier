"""Advisory critic budget for non-final claims: whole-task acceptance is
adjudicated at the final, so a non-final decomposition gets a bounded
review whose last repair stands instead of failing closed. Finals keep
the full adversarial game; a budget of zero seats no critic for
non-finals; the default (None) is the full game everywhere.
"""
from __future__ import annotations

import os

from verifier.core.events import EventLog
from verifier.core.model import Claim, CLAIM_ASSERTION, CLAIM_FINAL
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.agents import CRITIC_AGENT
from verifier.llm.schema import FixtureBackend


def compile_role(variables):
    repaired = bool(variables["critic_feedback"])
    return {"obligations": [{"text": ("repaired: " if repaired else "")
                             + f"O of: {variables['claim_text']}",
                             "required_basis": "world"}]}


ATTACK = {"verdict": "insufficient",
          "counterexample_world": "SUFFICIENCY GAP: a world where x",
          "missing_condition": "ADD: the missing condition"}
PASS = {"verdict": "pass", "counterexample_world": "n/a",
        "missing_condition": "n/a"}


class StubDischarger:
    def __init__(self):
        self.seen = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.seen.append(obligation.text)
        return DischargeResult(witness_kind="execution", outcome="holds",
                               witness=None, artifact="", instrument="stub",
                               basis="world")


def build(tmp_path, roles, nonfinal):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend(roles)
    discharger = StubDischarger()
    Kernel(store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
           critic_role=CRITIC_AGENT, discharger=discharger,
           critic_rounds=4, critic_rounds_nonfinal=nonfinal)
    return store, log, backend, discharger


def test_one_review_and_the_repair_stands_for_a_non_final(tmp_path):
    store, log, backend, discharger = build(
        tmp_path, {"obligations": compile_role, "critic": ATTACK}, nonfinal=1)

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "x holds"))

    assert store.status(cid) == "verified"
    assert [c["role"] for c in backend.calls] == [
        "obligations", "critic", "obligations"]
    assert discharger.seen == ["repaired: O of: x holds"]
    attacked = [e["payload"] for e in log.replay()
                if e["type"] == "decomposition_attacked"]
    assert attacked[-1]["resolution"].startswith("advisory")


def test_a_pass_needs_no_repair(tmp_path):
    store, log, backend, discharger = build(
        tmp_path, {"obligations": compile_role, "critic": PASS}, nonfinal=1)
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "x holds"))
    assert store.status(cid) == "verified"
    assert [c["role"] for c in backend.calls] == ["obligations", "critic"]


def test_finals_keep_the_full_game(tmp_path):
    store, log, backend, discharger = build(
        tmp_path, {"obligations": compile_role, "critic": ATTACK}, nonfinal=1)

    cid = store.add_claim(Claim(CLAIM_FINAL, "the task is done"))

    # four critic rounds, three repairs, fail closed — exactly as before
    assert store.status(cid) == "unverified"
    roles = [c["role"] for c in backend.calls]
    assert roles.count("critic") == 4 and roles.count("obligations") == 4
    assert discharger.seen == []


def test_zero_budget_seats_no_critic_for_non_finals(tmp_path):
    store, log, backend, discharger = build(
        tmp_path, {"obligations": compile_role, "critic": ATTACK}, nonfinal=0)
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "x holds"))
    assert store.status(cid) == "verified"
    assert [c["role"] for c in backend.calls] == ["obligations"]


def test_no_budget_means_the_full_game_everywhere(tmp_path):
    store, log, backend, discharger = build(
        tmp_path, {"obligations": compile_role, "critic": ATTACK},
        nonfinal=None)
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "x holds"))
    assert store.status(cid) == "unverified"
    assert [c["role"] for c in backend.calls].count("critic") == 4
