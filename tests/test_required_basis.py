"""The evidence floor (required_basis) is mechanical, not advisory.

world-floor obligations state what is TRUE; the solver's own record can
never settle them. Enforced at three gates: method filtering in the
binding game, the discharge backstop in the kernel, and the completion
backstop (a final verified on all-record evidence is the laundering
signature).
"""
from __future__ import annotations

import os

from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_RECORD, BASIS_WORLD, Claim, CLAIM_ASSERTION, CLAIM_FINAL,
    Obligation,
)
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.engine.game import BindingGame
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend


def compile_fixture(obligations):
    return lambda v: {"obligations": obligations}


class BasisDischarger:
    def __init__(self, basis=BASIS_WORLD, outcome="holds"):
        self.basis = basis
        self.outcome = outcome
        self.seen = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.seen.append(obligation.text)
        basis = (self.basis(obligation.text) if callable(self.basis)
                 else self.basis)
        return DischargeResult(outcome=self.outcome, witness=None,
                               artifact="", instrument="stub", basis=basis)


def build(tmp_path, obligations, basis=BASIS_WORLD):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": compile_fixture(obligations)})
    discharger = BasisDischarger(basis)
    Kernel(store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
           critic_role=None, discharger=discharger)
    return store, discharger


def test_floor_joins_obligation_identity():
    world = Obligation("c", "the count is 42", BASIS_WORLD)
    record = Obligation("c", "the count is 42", BASIS_RECORD)
    assert world.id != record.id


def test_game_filters_methods_below_the_floor(tmp_path):
    """A record method never executes against a world-floor obligation."""
    oracle = {"methods": [{"procedure": "grep the ledger", "artifact": "log",
                           "level": "basic", "basis": "record"}]}
    game = BindingGame(
        FixtureBackend({"oracles": oracle}),
        ArtifactStore(os.path.join(tmp_path, "art")),
        EventLog(os.path.join(tmp_path, "events.jsonl")),
        BasisDischarger(),
    )

    result = game.run({"id": "c1", "kind": "final", "text": "t"},
                      Obligation("c1", "the count is 42", BASIS_WORLD))

    assert result.outcome == "failed"
    assert "required basis" in result.witness


def test_holds_below_the_floor_fails_closed(tmp_path):
    """Direct-discharge backstop: record evidence cannot settle a
    world-floor obligation, and the failure is on the record."""
    store, _ = build(
        tmp_path,
        [{"text": "the count is 42", "required_basis": "world"}],
        basis=BASIS_RECORD)

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the count is 42"))

    assert store.status(cid) == "unverified"
    (node,) = store.obligation_children(cid)
    assert node["status"] == "unverified"
    ev = store.evidence_for(node["id"])[-1]
    assert ev["outcome"] == "failed" and "below the" in ev["witness"]


def test_all_record_final_is_not_verified(tmp_path):
    """The laundering signature: every check passed, none ever left the
    solver's own account of events."""
    store, _ = build(
        tmp_path,
        [{"text": "the ledger shows the run", "required_basis": "record"}],
        basis=BASIS_RECORD)

    cid = store.add_claim(Claim(CLAIM_FINAL, "the task is done"))

    assert store.status(cid) == "unverified"
    assert store.evidence_for(cid)[-1]["error"] == \
        "no world-basis evidence for a completion claim"


def test_final_with_world_evidence_verifies(tmp_path):
    per_text = lambda text: (                                      # noqa: E731
        BASIS_RECORD if "ledger" in text else BASIS_WORLD)
    store, _ = build(
        tmp_path,
        [{"text": "recomputation gives 42", "required_basis": "world"},
         {"text": "the ledger shows the run", "required_basis": "record"}],
        basis=per_text)

    cid = store.add_claim(Claim(CLAIM_FINAL, "the task is done"))

    assert store.status(cid) == "verified"


def test_inheritance_below_the_floor_is_refused(tmp_path):
    """A record-established claim cannot settle a world-floor obligation
    through same-text inheritance — the obligation is discharged fresh."""
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    fixtures = {
        "the count is 42":
            {"obligations": [{"text": "the count is 42",
                              "required_basis": "record"}]},
        "the answer is correct":
            {"obligations": [{"text": "the count is 42",
                              "required_basis": "world"}]},
    }
    backend = FixtureBackend(
        {"obligations": lambda v: fixtures[v["claim_text"]]})
    per_text = lambda text: BASIS_RECORD                           # noqa: E731
    discharger = BasisDischarger(BASIS_WORLD)
    kernel = Kernel(store, log, backend,
                    ArtifactStore(os.path.join(tmp_path, "art")),
                    critic_role=None, discharger=discharger)

    # claim A settles "the count is 42" at record level (self-restating
    # obligation discharged with record evidence)
    discharger.basis = BASIS_RECORD
    a = store.add_claim(Claim(CLAIM_ASSERTION, "the count is 42"))
    assert store.get(a)["basis"] == BASIS_RECORD

    # claim B demands the same text at world floor: inheritance from A is
    # refused, the obligation is discharged fresh at world basis
    discharger.basis = BASIS_WORLD
    b = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is correct"))
    assert store.status(b) == "verified"
    assert store.get(b)["basis"] == BASIS_WORLD
    assert discharger.seen.count("the count is 42") == 2
