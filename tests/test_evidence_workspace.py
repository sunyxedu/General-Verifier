"""The evidence workspace: durable building material + assumed premises.

Two layers, two rules. The WORKSPACE (state_dir/workspace) is durable
building material shared across every check of a task — it survives
scratch-workdir cleanup and is where a Lean project grows. Borrowing from
it is free only when the instrument re-establishes the borrowed
contribution; relying on a premise's content instead must be DECLARED in
assumed_premises, is legal only inside the claim's own dependency
closure (the conditional fold already charges those premises), and a
node settled under such assumptions must not transfer to a claim that
does not share them.
"""
from __future__ import annotations

import json
import os

from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_WORLD, Claim, CLAIM_ASSERTION, Obligation,
)
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import AgentDischarger, DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend


COMPILE = lambda v: {"obligations": [{"text": f"O of: {v['claim_text']}", "required_basis": "record"}]}  # noqa: E731


class StubDischarger:
    """Per-obligation-text queue of results; records calls."""

    def __init__(self, spec=None):
        self.spec = spec or {}   # text -> list of dicts (popped in order)
        self.seen = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.seen.append(obligation.text)
        queue = self.spec.get(obligation.text)
        item = queue.pop(0) if queue else {}
        return DischargeResult(
            outcome=item.get("outcome", "holds"),
            witness=item.get("witness"),
            artifact="", instrument="stub",
            basis=item.get("basis", BASIS_WORLD),
            assumed_premises=tuple(item.get("assumed_premises", ())))


def build(tmp_path, compile_fixture=COMPILE, discharger=None):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": compile_fixture})
    Kernel(store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
           critic_role=None, discharger=discharger or StubDischarger())
    return store


# the workspace layer

def test_workspace_survives_scratch_cleanup(tmp_path):
    state = tmp_path / "state"
    state.mkdir()

    def discharge(variables):
        with open(os.path.join(variables["workspace_dir"], "proof.lean"),
                  "w", encoding="utf-8") as handle:
            handle.write("theorem t : 1 + 1 = 2 := rfl")
        with open(os.path.join(variables["workdir"], "scratch.txt"),
                  "w", encoding="utf-8") as handle:
            handle.write("scratch")
        return {"outcome": "holds", "witness": "", "artifact": "proof.lean",
                "assumed_premises": []}

    discharger = AgentDischarger(
        FixtureBackend({"discharge": discharge}),
        ArtifactStore(os.path.join(tmp_path, "art")),
        workdir_base=os.path.join(tmp_path, "work"),
        cleanup=True,
        state_dir=str(state),
    )
    obligation = Obligation("c1", "the lemma holds")
    discharger.discharge(
        {"id": "c1", "kind": "assertion", "text": "the lemma holds"},
        obligation)

    assert not os.path.exists(
        os.path.join(tmp_path, "work", obligation.id[:12]))
    workspace_file = os.path.join(str(state), "workspace", "proof.lean")
    assert os.path.exists(workspace_file)
    assert discharger.workspace_dir == os.path.join(str(state), "workspace")


def test_discharge_sees_premises_and_prior_evidence_index(tmp_path):
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    premise = store.add_claim(Claim(CLAIM_ASSERTION, "the decode gives 11"))
    node = store.add_obligation(
        premise, Obligation(premise, "the decode output parses"))
    store.record_evidence(node, {"outcome": "holds", "artifact": "d1gest"})
    store.set_verified(node, basis=BASIS_WORLD)
    dependent = store.add_claim(
        Claim(CLAIM_ASSERTION, "the file is correct", deps=(premise,)))

    backend = FixtureBackend({"discharge": {
        "outcome": "holds", "witness": "", "artifact": "",
        "assumed_premises": []}})
    discharger = AgentDischarger(
        backend, ArtifactStore(os.path.join(tmp_path, "art")),
        workdir_base=os.path.join(tmp_path, "work"),
        state_dir=str(tmp_path), store=store)
    discharger.discharge(
        store.get(dependent),
        Obligation(dependent, "file content equals the decode"))

    variables = backend.calls[-1]["variables"]
    assert variables["workspace_dir"] == os.path.join(
        str(tmp_path), "workspace")
    index = json.loads(variables["workspace_index"])
    assert index["premises"] == [
        {"id": premise, "text": "the decode gives 11",
         "status": "unverified"}]
    assert index["prior_evidence"] == [
        {"obligation": "the decode output parses", "status": "verified",
         "basis": BASIS_WORLD, "strength": None, "artifact": "d1gest"}]


# the assumed-premises gates

def test_assumed_premise_inside_closure_is_recorded_and_legal(tmp_path):
    discharger = StubDischarger()
    store = build(tmp_path, discharger=discharger)
    premise = store.add_claim(Claim(CLAIM_ASSERTION, "the premise"))
    discharger.spec["O of: built on it"] = [
        {"assumed_premises": [premise]}]
    claim = store.add_claim(
        Claim(CLAIM_ASSERTION, "built on it", deps=(premise,)))

    assert store.status(claim) == "verified"
    (node,) = store.obligation_children(claim)
    assert store.evidence_for(node["id"])[-1]["assumed_premises"] \
        == [premise]


def test_assumed_premise_outside_closure_fails_closed(tmp_path):
    discharger = StubDischarger()
    store = build(tmp_path, discharger=discharger)
    stranger = store.add_claim(Claim(CLAIM_ASSERTION, "an unrelated claim"))
    discharger.spec["O of: stands alone"] = [
        {"assumed_premises": [stranger]}]
    claim = store.add_claim(Claim(CLAIM_ASSERTION, "stands alone"))

    assert store.status(claim) == "unverified"
    (node,) = store.obligation_children(claim)
    ev = store.evidence_for(node["id"])[-1]
    assert ev["outcome"] == "failed"
    assert "outside the claim's dependency closure" in ev["witness"]


def test_node_settled_under_assumptions_does_not_transfer(tmp_path):
    shared = lambda v: {"obligations": [                           # noqa: E731
        {"text": "the count is 42", "required_basis": "record"}]}
    fixtures = {"the premise": COMPILE,
                "count claim": shared, "another claim": shared}
    discharger = StubDischarger()
    store = build(
        tmp_path,
        compile_fixture=lambda v: fixtures[v["claim_text"]](v),
        discharger=discharger)

    premise = store.add_claim(Claim(CLAIM_ASSERTION, "the premise"))
    discharger.spec["the count is 42"] = [
        {"assumed_premises": [premise]},   # first: under assumption
        {},                                # second: stands on its own
    ]
    first = store.add_claim(
        Claim(CLAIM_ASSERTION, "count claim", deps=(premise,)))
    assert store.status(first) == "verified"

    # no premise edge to `the premise`: the settled node must not carry
    # its assumption across — a fresh discharge runs instead
    second = store.add_claim(Claim(CLAIM_ASSERTION, "another claim"))
    assert store.status(second) == "verified"
    assert discharger.seen.count("the count is 42") == 2


def test_final_admits_no_assumed_premises(tmp_path):
    from verifier.core.model import CLAIM_FINAL

    discharger = StubDischarger()
    store = build(tmp_path, discharger=discharger)
    premise = store.add_claim(Claim(CLAIM_ASSERTION, "the premise"))
    discharger.spec["O of: the answer is 42"] = [
        {"assumed_premises": [premise]}]
    final = store.add_claim(
        Claim(CLAIM_FINAL, "the answer is 42", deps=(premise,)))

    # the final is exempt from the premise fold, so nothing is ever in
    # force for its contract: the assumption has nowhere to be charged
    # and the discharge fails closed instead
    assert store.status(final) == "unverified"
    (node,) = store.obligation_children(final)
    ev = store.evidence_for(node["id"])[-1]
    assert ev["outcome"] == "failed"
    assert "admits no assumed premises" in ev["witness"]


def test_workspace_index_marks_final_premises_not_assumable(tmp_path):
    from verifier.core.model import CLAIM_FINAL

    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    premise = store.add_claim(Claim(CLAIM_ASSERTION, "the premise"))
    final = store.add_claim(
        Claim(CLAIM_FINAL, "the answer is 42", deps=(premise,)))
    discharger = AgentDischarger(
        FixtureBackend({"discharge": {
            "outcome": "holds", "witness": "", "artifact": "",
            "assumed_premises": []}}),
        ArtifactStore(os.path.join(tmp_path, "art")),
        workdir_base=os.path.join(tmp_path, "work"),
        state_dir=str(tmp_path), store=store)

    discharger.discharge(
        store.get(final), Obligation(final, "the delivered answer is 42"))

    backend = discharger.backend
    index = json.loads(backend.calls[-1]["variables"]["workspace_index"])
    assert index["premises"] == [
        {"note": "completion claim: no premise is assumable; "
                 "declare nothing"}]
