"""Reuse must preserve the proposition, its assumptions, and its witness."""
import threading

import pytest

from verifier.core.events import EventLog
from verifier.core.model import Claim, CLAIM_ASSERTION, CLAIM_FINAL
from verifier.core.store import ClaimStore, TRIGGER_LAZY
from verifier.engine.certificate import request_certificate
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend


def build(tmp_path, contracts, discharge, *, parallel=1):
    store = ClaimStore(trigger=TRIGGER_LAZY)
    log = EventLog(str(tmp_path / "events.jsonl"))
    artifacts = ArtifactStore(str(tmp_path / "artifacts"))

    class Discharger:
        def discharge(self, row, obligation, method=None, tag=""):
            return discharge(row, obligation)

    kernel = Kernel(
        store, log, FixtureBackend({"obligations": lambda v: {
            "obligations": contracts[v["claim_text"]]}}),
        artifacts, critic_role=None, discharger=Discharger(),
        parallel_obligations=parallel)
    return store, kernel, log, artifacts


def result(outcome, **kwargs):
    return DischargeResult(
        outcome=outcome, witness="retained finding", artifact="",
        instrument="fixture", **{"basis": "world",
                                 "witness_kind": "execution", **kwargs})


@pytest.mark.parametrize("first_outcome", ["holds", "refuted", "unobtainable"])
def test_failed_retry_cannot_launder_a_conditional_verdict(tmp_path, first_outcome):
    shared = [{"text": "shared condition", "required_basis": "world"}]
    calls = []

    def discharge(row, _obligation):
        calls.append(row["text"])
        if row["text"] == "conditional":
            return result(first_outcome, assumed_premises=(premise,))
        return result("failed")

    store, kernel, log, artifacts = build(tmp_path, {
        "premise": [{"text": "premise check", "required_basis": "world"}],
        "conditional": shared, "retry": shared, "final": shared,
    }, discharge)
    premise = store.add_claim(Claim(CLAIM_ASSERTION, "premise"))
    conditional = store.add_claim(Claim(
        CLAIM_ASSERTION, "conditional", deps=(premise,)))
    retry = store.add_claim(Claim(CLAIM_ASSERTION, "retry"))
    final = store.add_claim(Claim(CLAIM_FINAL, "final"))

    kernel.verify(conditional)
    kernel.verify(retry)
    cert = request_certificate(store, kernel, log, artifacts, final)

    assert cert["verdict"] == "unverified"
    assert calls.count("final") == 1
    first_node, = store.obligation_children(conditional)
    retry_node, = store.obligation_children(retry)
    final_node, = cert["slice"][0]["obligations"]
    assert first_node["id"] != retry_node["id"]
    assert retry_node["status"] == final_node["status"] == "unverified"
    assert store.effective(first_node["id"])["open_premises"] == [premise]
    assert store.evidence_for(first_node["id"])[-1]["outcome"] == first_outcome


@pytest.mark.parametrize("inherit_from_claim", [False, True])
@pytest.mark.parametrize("target_allows_absence", [False, True])
def test_absence_refutation_cannot_cross_a_presence_contract(
        tmp_path, inherit_from_claim, target_allows_absence):
    source_text = "same statement" if inherit_from_claim else "source"
    source_check = "presence check" if inherit_from_claim else "same statement"
    source_obligation = {
        "text": source_check, "required_basis": "world",
        "refutable_by_absence": True,
    }
    calls = []

    def discharge(row, _obligation):
        calls.append(row["text"])
        return (result("refuted", witness_kind="absence")
                if row["text"] == source_text else result("failed"))

    store, kernel, _, _ = build(tmp_path, {
        source_text: [source_obligation],
        "target": [{"text": "same statement", "required_basis": "world",
                    "refutable_by_absence": target_allows_absence}],
    }, discharge)
    source = store.add_claim(Claim(CLAIM_ASSERTION, source_text))
    target = store.add_claim(Claim(CLAIM_FINAL, "target"))
    assert kernel.verify(source) == "refuted"
    assert kernel.verify(target) == (
        "refuted" if target_allows_absence else "unverified")
    assert calls == ([source_text] if target_allows_absence
                     else [source_text, "target"])


def test_parallel_refutation_keeps_its_own_basis_and_strength(tmp_path):
    barrier = threading.Barrier(2)

    def discharge(row, obligation):
        if row["text"] == "target":
            return result("failed")
        barrier.wait(timeout=5)
        if obligation.text == "counterexample":
            return result("refuted", basis="record", strength="decisive")
        return result("holds", basis="world", strength="bounded")

    store, kernel, _, _ = build(tmp_path, {"source": [
        {"text": "counterexample", "required_basis": "record"},
        {"text": "unrelated success", "required_basis": "world"},
    ], "target": [{"text": "source", "required_basis": "world"}]},
        discharge, parallel=2)
    source = store.add_claim(Claim(CLAIM_ASSERTION, "source"))

    assert kernel.verify(source) == "refuted"
    row = store.get(source)
    assert (row["basis"], row["strength"]) == ("record", "decisive")
    target = store.add_claim(Claim(CLAIM_FINAL, "target"))
    assert kernel.verify(target) == "unverified"


def test_shared_premise_scope_reuses_one_conditional_check(tmp_path):
    shared = [{"text": "shared condition", "required_basis": "world"}]
    calls = []

    def discharge(row, _obligation):
        calls.append(row["text"])
        return result("holds", assumed_premises=(premise,))

    store, kernel, _, _ = build(tmp_path, {"first": shared, "second": shared},
                                discharge)
    premise = store.add_claim(Claim(CLAIM_ASSERTION, "premise"))
    first = store.add_claim(Claim(CLAIM_ASSERTION, "first", deps=(premise,)))
    second = store.add_claim(Claim(CLAIM_ASSERTION, "second", deps=(premise,)))

    assert kernel.verify(first) == kernel.verify(second) == "verified"
    assert calls == ["first"]
    node, = store.obligation_children(first)
    assert store.obligation_children(second)[0]["id"] == node["id"]
    assert store.effective(node["id"])["effective_status"] == "unverified"
    store.set_verified(premise, basis="world")
    assert store.effective(node["id"])["effective_status"] == "verified"
    store.add_claim(Claim(CLAIM_ASSERTION, "revised premise", revises=premise))
    assert store.status(node["id"]) == "stale"


@pytest.mark.parametrize("premise_basis,premise_strength,expected", [
    ("record", "decisive", "unverified"),
    ("world", "bounded", "unverified"),
    ("world", "analytic", "unverified"),
    ("world", "decisive", "refuted"),
])
def test_inherited_refutation_preserves_its_premises_evidence_limits(
        tmp_path, premise_basis, premise_strength, expected):
    calls = []

    def discharge(row, _obligation):
        calls.append(row["text"])
        if row["text"] == "premise":
            return result("holds", basis=premise_basis, strength=premise_strength)
        if row["text"] == "source":
            return result("refuted", assumed_premises=(premise,), strength="decisive")
        return result("failed")

    store, kernel, log, artifacts = build(tmp_path, {
        "premise": [{"text": "premise check", "required_basis": "record"}],
        "source": [{"text": "counterexample check", "required_basis": "world"}],
        "final": [{"text": "source", "required_basis": "world"}],
    }, discharge)
    premise = store.add_claim(Claim(CLAIM_ASSERTION, "premise"))
    source = store.add_claim(Claim(CLAIM_ASSERTION, "source", deps=(premise,)))
    final = store.add_claim(Claim(CLAIM_FINAL, "final"))
    assert kernel.verify(premise) == "verified"
    assert kernel.verify(source) == "refuted"
    assert store.get(source)["basis"] == "world"
    effective = store.effective(source)
    assert effective["effective_basis"] == premise_basis
    assert effective["effective_strength"] == premise_strength
    cert = request_certificate(store, kernel, log, artifacts, final)
    assert cert["verdict"] == expected
    assert ("final" in calls) == (expected == "unverified")
