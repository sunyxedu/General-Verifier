"""Conditional local verdicts + the read-time effective fold.

A stored status is LOCAL: does the statement hold ASSUMING every premise
(dep) is true — so verification never blocks on premise order. What a
consumer relies on is the EFFECTIVE verdict: the weakest link along the
premise chain, a refuted node poisoning everything downstream, basis and
strength folded as weakest links. Nothing is stored; the fold happens at
read time (store.effective), so a premise settled later changes the
read-out without rewrites.

Consequences under test here:
  * the compiler receives unsettled premises as ASSUMED premises and
    compiles the residual on top of them;
  * a refuted premise chain skips (wasted) local verification and reads
    out effectively refuted, transitively;
  * finals are exempt (their contract is compiled independently);
  * inheritance transfers only effective verdicts — a conditionally
    verified source, or a conditional/premise-poisoned refutation,
    settles nothing elsewhere;
  * the certificate exports both verdicts per slice entry.
"""
from __future__ import annotations

import os

from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_RECORD, BASIS_WORLD, Claim, CLAIM_ASSERTION, CLAIM_FINAL,
    STRENGTH_ANALYTIC, STRENGTH_DECISIVE,
)
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.engine.certificate import request_certificate
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend


COMPILE = lambda v: {"obligations": [{"text": f"O of: {v['claim_text']}", "required_basis": "record"}]}  # noqa: E731


class PerTextDischarger:
    """Outcome/basis/strength keyed by obligation text; records calls."""

    def __init__(self, spec=None, default=("holds", BASIS_WORLD, None)):
        self.spec = spec or {}
        self.default = default
        self.seen = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.seen.append(obligation.text)
        outcome, basis, strength = self.spec.get(
            obligation.text, self.default)
        return DischargeResult(
            witness_kind="execution",
            outcome=outcome,
            witness="w" if outcome == "refuted" else None,
            artifact="", instrument="stub", basis=basis, strength=strength)


def build(tmp_path, compile_fixture=COMPILE, discharger=None):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": compile_fixture})
    artifacts = ArtifactStore(os.path.join(tmp_path, "art"))
    discharger = discharger or PerTextDischarger()
    kernel = Kernel(store, log, backend, artifacts, critic_role=None,
                    discharger=discharger)
    return store, backend, discharger, kernel, log, artifacts


# the fold itself

def test_open_premise_caps_effective_status(tmp_path):
    discharger = PerTextDischarger(
        {"O of: the premise": ("failed", BASIS_WORLD, None)})
    store, _, _, _, _, _ = build(tmp_path, discharger=discharger)

    a = store.add_claim(Claim(CLAIM_ASSERTION, "the premise"))
    b = store.add_claim(Claim(CLAIM_ASSERTION, "the conclusion", deps=(a,)))

    assert store.status(a) == "unverified"
    assert store.status(b) == "verified"          # local: given the premise
    effective = store.effective(b)
    assert effective["effective_status"] == "unverified"
    assert effective["open_premises"] == [a]
    # capped: the local basis describes a verdict the chain does not deliver
    assert effective["effective_basis"] is None


def test_refuted_premise_poisons_transitively_and_skips_verification(tmp_path):
    discharger = PerTextDischarger(
        {"O of: the premise": ("refuted", BASIS_WORLD, None)})
    store, _, discharger, _, _, _ = build(tmp_path, discharger=discharger)

    a = store.add_claim(Claim(CLAIM_ASSERTION, "the premise"))
    b = store.add_claim(Claim(CLAIM_ASSERTION, "built on it", deps=(a,)))
    c = store.add_claim(Claim(CLAIM_ASSERTION, "built on that", deps=(b,)))

    assert store.status(a) == "refuted"
    # local verification was skipped as waste, not adjudicated
    assert store.status(b) == "unverified"
    assert store.evidence_for(b)[-1]["error"].startswith(
        "premise chain refuted")
    assert "O of: built on it" not in discharger.seen
    for cid in (b, c):
        effective = store.effective(cid)
        assert effective["effective_status"] == "refuted"
        assert effective["premise_refuted"] == a


def test_final_is_exempt_from_the_premise_fold(tmp_path):
    discharger = PerTextDischarger(
        {"O of: the premise": ("refuted", BASIS_WORLD, None)})
    store, _, _, _, _, _ = build(tmp_path, discharger=discharger)

    a = store.add_claim(Claim(CLAIM_ASSERTION, "the premise"))
    f = store.add_claim(Claim(CLAIM_FINAL, "the answer is 42", deps=(a,)))

    assert store.status(f) == "verified"
    effective = store.effective(f)
    assert effective["effective_status"] == "verified"
    assert effective["premise_refuted"] is None


def test_effective_basis_and_strength_fold_along_the_chain(tmp_path):
    discharger = PerTextDischarger(
        {"O of: the solver reported 42": (
            "holds", BASIS_RECORD, STRENGTH_ANALYTIC)},
        default=("holds", BASIS_WORLD, STRENGTH_DECISIVE))
    store, _, _, _, _, _ = build(tmp_path, discharger=discharger)

    a = store.add_claim(Claim(CLAIM_ASSERTION, "the solver reported 42"))
    b = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 42", deps=(a,)))

    assert store.get(b)["basis"] == BASIS_WORLD       # local: own obligations
    assert store.get(b)["strength"] == STRENGTH_DECISIVE
    effective = store.effective(b)
    assert effective["effective_status"] == "verified"
    assert effective["effective_basis"] == BASIS_RECORD
    assert effective["effective_strength"] == STRENGTH_ANALYTIC


# the compiler's premise context

def test_unsettled_premise_reaches_the_compiler_as_assumed(tmp_path):
    discharger = PerTextDischarger(
        {"O of: the premise": ("failed", BASIS_WORLD, None)})
    store, backend, _, _, _, _ = build(tmp_path, discharger=discharger)

    a = store.add_claim(Claim(CLAIM_ASSERTION, "the premise"))
    store.add_claim(Claim(CLAIM_ASSERTION, "the conclusion", deps=(a,)))

    variables = backend.calls[-1]["variables"]
    assert variables["verified_deps"] == "[]"
    assert "[assumed premise — status: unverified] the premise" \
        in variables["pending_deps"]


# inheritance gates

def test_conditionally_verified_source_does_not_transfer(tmp_path):
    fixtures = {
        "the premise": COMPILE,
        "the count is 42": COMPILE,
        "the answer is correct": lambda v: {"obligations": [
            {"text": "the count is 42", "required_basis": "record"}]},
    }
    discharger = PerTextDischarger(
        {"O of: the premise": ("failed", BASIS_WORLD, None)})
    store, _, discharger, _, _, _ = build(
        tmp_path,
        compile_fixture=lambda v: fixtures[v["claim_text"]](v),
        discharger=discharger)

    a = store.add_claim(Claim(CLAIM_ASSERTION, "the premise"))
    b = store.add_claim(Claim(CLAIM_ASSERTION, "the count is 42", deps=(a,)))
    assert store.status(b) == "verified"                    # conditional
    assert store.effective(b)["effective_status"] == "unverified"

    c = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is correct"))
    assert store.status(c) == "verified"
    (node,) = store.obligation_children(c)
    # settled by fresh discharge, not by graph inheritance
    assert store.evidence_for(node["id"])[-1]["instrument"] == "stub"
    assert discharger.seen.count("the count is 42") == 1


def test_effective_settled_source_transfers_with_chain_basis(tmp_path):
    fixtures = {
        "the solver reported 42": COMPILE,
        "the count is 42": COMPILE,
        "the answer is correct": lambda v: {"obligations": [
            {"text": "the count is 42", "required_basis": "record"}]},
    }
    discharger = PerTextDischarger(
        {"O of: the solver reported 42": ("holds", BASIS_RECORD, None)})
    store, _, discharger, _, _, _ = build(
        tmp_path,
        compile_fixture=lambda v: fixtures[v["claim_text"]](v),
        discharger=discharger)

    a = store.add_claim(Claim(CLAIM_ASSERTION, "the solver reported 42"))
    b = store.add_claim(Claim(CLAIM_ASSERTION, "the count is 42", deps=(a,)))
    assert store.get(b)["basis"] == BASIS_WORLD             # local
    assert store.effective(b)["effective_basis"] == BASIS_RECORD

    c = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is correct"))
    assert store.status(c) == "verified"
    (node,) = store.obligation_children(c)
    ev = store.evidence_for(node["id"])[-1]
    # inherited — carrying the chain's EFFECTIVE basis, not the local one
    assert ev["instrument"] == "graph"
    assert ev["basis"] == BASIS_RECORD
    assert store.get(c)["basis"] == BASIS_RECORD


def test_chain_record_basis_cannot_settle_world_floor_by_inheritance(
        tmp_path):
    fixtures = {
        "the solver reported 42": COMPILE,
        "the count is 42": COMPILE,
        "the answer is correct": lambda v: {"obligations": [
            {"text": "the count is 42", "required_basis": "world"}]},
    }
    discharger = PerTextDischarger(
        {"O of: the solver reported 42": ("holds", BASIS_RECORD, None)})
    store, _, discharger, _, _, _ = build(
        tmp_path,
        compile_fixture=lambda v: fixtures[v["claim_text"]](v),
        discharger=discharger)

    a = store.add_claim(Claim(CLAIM_ASSERTION, "the solver reported 42"))
    b = store.add_claim(Claim(CLAIM_ASSERTION, "the count is 42", deps=(a,)))
    assert store.get(b)["basis"] == BASIS_WORLD             # local world...
    assert store.effective(b)["effective_basis"] == BASIS_RECORD  # ...but not
    # in effect: the world-floor obligation must be discharged fresh
    c = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is correct"))
    (node,) = store.obligation_children(c)
    assert store.evidence_for(node["id"])[-1]["instrument"] == "stub"
    assert discharger.seen.count("the count is 42") == 1


def test_conditional_refutation_does_not_transfer(tmp_path):
    fixtures = {
        "the premise": COMPILE,
        "the count is 42": COMPILE,
        "the answer is correct": lambda v: {"obligations": [
            {"text": "the count is 42", "required_basis": "record"}]},
    }
    discharger = PerTextDischarger(
        {"O of: the premise": ("failed", BASIS_WORLD, None),
         "O of: the count is 42": ("refuted", BASIS_WORLD, None)})
    store, _, discharger, _, _, _ = build(
        tmp_path,
        compile_fixture=lambda v: fixtures[v["claim_text"]](v),
        discharger=discharger)

    a = store.add_claim(Claim(CLAIM_ASSERTION, "the premise"))
    b = store.add_claim(Claim(CLAIM_ASSERTION, "the count is 42", deps=(a,)))
    assert store.status(b) == "refuted"                     # conditional
    assert store.effective(b)["open_premises"] == [a]

    # the conditional refutation must not refute an unrelated claim's
    # obligation; the fresh check settles it on its own
    c = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is correct"))
    assert store.status(c) == "verified"
    assert discharger.seen.count("the count is 42") == 1


# certificate export

def test_certificate_exports_both_verdicts_per_entry(tmp_path):
    discharger = PerTextDischarger(
        {"O of: the premise": ("failed", BASIS_WORLD, None)})
    store, _, _, kernel, log, artifacts = build(
        tmp_path, discharger=discharger)

    a = store.add_claim(Claim(CLAIM_ASSERTION, "the premise"))
    b = store.add_claim(Claim(CLAIM_ASSERTION, "the conclusion", deps=(a,)))
    f = store.add_claim(Claim(CLAIM_FINAL, "the answer is 42", deps=(b,)))
    cert = request_certificate(store, kernel, log, artifacts, f)

    by_text = {e["text"]: e for e in cert["slice"]}
    entry = by_text["the conclusion"]
    assert entry["status"] == "verified"
    assert entry["effective_status"] == "unverified"
    assert entry["open_premises"] == [a]
    assert by_text["the answer is 42"]["effective_status"] == "verified"
    assert cert["verdict"] == "verified"


# the mechanical residual filter

def test_premise_restating_obligation_is_residualized_out(tmp_path):
    fixtures = {
        "the count is 42": COMPILE,
        "the file is right": lambda v: {"obligations": [
            {"text": "the count is 42", "required_basis": "record"},
            {"text": "the file contains the count",
             "required_basis": "record"},
        ]},
    }
    discharger = PerTextDischarger()
    store, _, discharger, _, log, _ = build(
        tmp_path,
        compile_fixture=lambda v: fixtures[v["claim_text"]](v),
        discharger=discharger)

    premise = store.add_claim(Claim(CLAIM_ASSERTION, "the count is 42"))
    claim = store.add_claim(
        Claim(CLAIM_ASSERTION, "the file is right", deps=(premise,)))

    assert store.status(claim) == "verified"
    # the verbatim premise restatement was dropped, not re-verified: the
    # premise edge already charges it through the read-time fold
    texts = [o["text"] for o in store.obligation_children(claim)]
    assert texts == ["the file contains the count"]
    events = [e for e in log.replay()
              if e["type"] == "obligations_residualized"]
    assert events[-1]["payload"]["dropped"] == ["the count is 42"]


def test_filter_never_empties_the_decomposition(tmp_path):
    fixtures = {
        "the count is 42": COMPILE,
        "restates only": lambda v: {"obligations": [
            {"text": "the count is 42", "required_basis": "record"}]},
    }
    discharger = PerTextDischarger()
    store, _, discharger, _, log, _ = build(
        tmp_path,
        compile_fixture=lambda v: fixtures[v["claim_text"]](v),
        discharger=discharger)

    premise = store.add_claim(Claim(CLAIM_ASSERTION, "the count is 42"))
    claim = store.add_claim(
        Claim(CLAIM_ASSERTION, "restates only", deps=(premise,)))

    # dropping the only obligation would fail the compile closed, so the
    # restatement is kept (and settles by inheritance from the premise)
    assert [o["text"] for o in store.obligation_children(claim)] \
        == ["the count is 42"]
    assert not [e for e in log.replay()
                if e["type"] == "obligations_residualized"]
