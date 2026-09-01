"""The axioms enforced as mechanisms rather than prompt text.

  independence — a delivered artifact standing as authority makes a
                 holds circular; the kernel refuses it;
  witness      — a refutation needs a positive, retained observation;
                 absence refutes only a compiled presence requirement;
  necessity    — a should-clause violation lowers a claim, never refutes;
  ties         — within a verdict band only contrary signals rank;
                 equal keys are an explicit tie (argmax_set);
  world        — the delivered world is addressed in the task's scheme.
"""
from __future__ import annotations

import json
import os

from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_WORLD, Claim, CLAIM_ASSERTION, CLAIM_FINAL, Obligation,
    SEVERITY_SHOULD, witness_admissible, circular_inputs,
)
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.core.world import load_world_view
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend, RoleError, validate
from verifier.llm.agents import to_json_schema
from verifier import selection, offline


class Discharger:
    """Per-obligation-text scripted results."""

    def __init__(self, script):
        self.script = script

    def discharge(self, row, obligation, method=None, tag=""):
        outcome, extra = self.script[obligation.text]
        return DischargeResult(outcome=outcome, witness=extra.get("witness"),
                               artifact="", instrument="stub",
                               basis=BASIS_WORLD, **{
                                   k: v for k, v in extra.items()
                                   if k in ("witness_kind", "inputs")})


def build(tmp_path, obligations, script):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": lambda v: {
        "obligations": obligations}})
    Kernel(store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
           critic_role=None, discharger=Discharger(script))
    return store


# ---- witness axiom ----------------------------------------------------------

def test_witness_admissibility_table():
    assert witness_admissible("execution", False)
    assert witness_admissible("world_state", False)
    assert witness_admissible("record_contradiction", False)
    assert not witness_admissible("absence", False)
    assert witness_admissible("absence", True)
    assert not witness_admissible("none", True)
    assert not witness_admissible(None, True)
    assert not witness_admissible("vibes", True)


def test_absence_does_not_refute_an_ordinary_obligation(tmp_path):
    store = build(
        tmp_path,
        [{"text": "version 1.0 was used", "required_basis": "world"}],
        {"version 1.0 was used": ("refuted", {
            "witness": "no tool output prints the version",
            "witness_kind": "absence"})})
    cid = store.add_claim(Claim(CLAIM_FINAL, "done"))
    assert store.status(cid) == "unverified"
    (node,) = store.obligation_children(cid)
    assert node["status"] == "unverified"
    ev = store.evidence_for(node["id"])[-1]
    assert ev["outcome"] == "failed"
    assert "inadmissible" in ev["witness"]
    assert ev["witness_kind"] == "absence"


def test_absence_refutes_a_compiled_presence_requirement(tmp_path):
    store = build(
        tmp_path,
        [{"text": "the deliverable exists", "required_basis": "world",
          "refutable_by_absence": True}],
        {"the deliverable exists": ("refuted", {
            "witness": "no such artifact under any captured root",
            "witness_kind": "absence"})})
    cid = store.add_claim(Claim(CLAIM_FINAL, "done"))
    assert store.status(cid) == "refuted"


def test_undeclared_witness_kind_reads_as_none(tmp_path):
    store = build(
        tmp_path,
        [{"text": "x holds", "required_basis": "world"}],
        {"x holds": ("refuted", {"witness": "it does not"})})
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "x"))
    assert store.status(cid) == "unverified"


# ---- independence axiom ------------------------------------------------------

def test_circular_inputs_detects_delivered_authority():
    assert circular_inputs([
        {"ref": "/app/result.txt", "provenance": "delivered",
         "role": "subject"},
        {"ref": "/app/db.sqlite", "provenance": "delivered",
         "role": "authority"},
        {"ref": "/app/data.txt", "provenance": "task_given",
         "role": "authority"},
    ]) == ["/app/db.sqlite"]
    assert circular_inputs(None) == []


def test_holds_grounded_on_delivered_authority_settles_nothing(tmp_path):
    store = build(
        tmp_path,
        [{"text": "recovered.json matches the database",
          "required_basis": "world"}],
        {"recovered.json matches the database": ("holds", {
            "inputs": [
                {"ref": "/app/recovered.json", "provenance": "delivered",
                 "role": "subject"},
                {"ref": "/app/db.sqlite", "provenance": "delivered",
                 "role": "authority"}]})})
    cid = store.add_claim(Claim(CLAIM_FINAL, "the db is recovered"))
    assert store.status(cid) == "unverified"
    (node,) = store.obligation_children(cid)
    ev = store.evidence_for(node["id"])[-1]
    assert ev["outcome"] == "failed"
    assert "circular" in ev["witness"]


def test_holds_from_task_given_authority_verifies(tmp_path):
    store = build(
        tmp_path,
        [{"text": "result.txt names the fifth-ranked document",
          "required_basis": "world"}],
        {"result.txt names the fifth-ranked document": ("holds", {
            "inputs": [
                {"ref": "/app/result.txt", "provenance": "delivered",
                 "role": "subject"},
                {"ref": "/app/data.txt", "provenance": "task_given",
                 "role": "input"},
                {"ref": "recomputed ranking", "provenance": "verifier",
                 "role": "authority"}]})})
    cid = store.add_claim(Claim(CLAIM_FINAL, "done"))
    assert store.status(cid) == "verified"


# ---- necessity axiom (severity) ------------------------------------------------

def test_should_violation_lowers_but_never_refutes(tmp_path):
    store = build(
        tmp_path,
        [{"text": "the output is correct", "required_basis": "world"},
         {"text": "only the listed libraries are imported",
          "required_basis": "world", "severity": "should",
          "clause": "You can only use these libraries"}],
        {"the output is correct": ("holds", {}),
         "only the listed libraries are imported": ("refuted", {
             "witness": "import sys", "witness_kind": "world_state"})})
    cid = store.add_claim(Claim(CLAIM_FINAL, "done"))
    assert store.status(cid) == "partial"
    kids = {k["text"]: k for k in store.obligation_children(cid)}
    assert kids["only the listed libraries are imported"]["status"] \
        == "refuted"
    assert kids["only the listed libraries are imported"]["severity"] \
        == SEVERITY_SHOULD
    assert kids["only the listed libraries are imported"]["clause"] \
        == "You can only use these libraries"
    ev = store.evidence_for(cid)[-1]
    assert ev["contrary_signals"] == 1


def test_must_violation_still_refutes(tmp_path):
    store = build(
        tmp_path,
        [{"text": "the output is correct", "required_basis": "world",
          "severity": "must"}],
        {"the output is correct": ("refuted", {
            "witness": "expected 5, got 4", "witness_kind": "execution"})})
    cid = store.add_claim(Claim(CLAIM_FINAL, "done"))
    assert store.status(cid) == "refuted"


def test_omitted_severity_defaults_to_must(tmp_path):
    """The schema gate rejects an unknown severity outright; an OMITTED
    one is read in the fail-closed direction (must)."""
    store = build(
        tmp_path,
        [{"text": "x", "required_basis": "world"}],
        {"x": ("refuted", {"witness": "no", "witness_kind": "execution"})})
    cid = store.add_claim(Claim(CLAIM_FINAL, "done"))
    assert store.status(cid) == "refuted"
    (node,) = store.obligation_children(cid)
    assert node["severity"] == "must"


def test_obligation_rejects_unknown_severity():
    try:
        Obligation("c", "x", BASIS_WORLD, severity="maybe")
    except ValueError:
        return
    raise AssertionError("unknown severity accepted")


def test_severity_is_not_part_of_node_identity():
    a = Obligation("c", "x", BASIS_WORLD, severity="must")
    b = Obligation("c", "x", BASIS_WORLD, severity="should")
    assert a.id == b.id


# ---- ranking and ties ----------------------------------------------------------

def F(verdict, contrary=0):
    return {"feature_schema_version": offline.FEATURE_SCHEMA_VERSION,
            "unscoreable": False, "verdict": verdict,
            "final_obligations": {"verified": 1},
            "final_obligations_total": 1,
            "final_contrary_signals": contrary}


def test_contrary_signals_rank_inside_a_band_and_ties_are_explicit():
    rows = [{"answer": 1, "features": F("partial", 1)},
            {"answer": 2, "features": F("partial", 0)},
            {"answer": 3, "features": F("partial", 0)},
            {"answer": 4, "features": F("unverified", 0)}]
    assert selection.argmax(rows)["answer"] == 2
    assert [r["answer"] for r in selection.argmax_set(rows)] == [2, 3]
    assert selection.scalar(F("partial", 0)) > selection.scalar(F("partial", 3))
    assert selection.scalar(F("unverified", 0)) < selection.scalar(F("partial", 9))
    assert selection.argmax_set([]) == []


# ---- schema optional keys ------------------------------------------------------

def test_optional_schema_keys():
    schema = {"outcome": ("holds", "refuted"), "witness_kind?": ("absence",)}
    validate(schema, {"outcome": "holds"})
    validate(schema, {"outcome": "holds", "witness_kind": "absence"})
    try:
        validate(schema, {"outcome": "holds", "witness_kind": "vibes"})
    except RoleError:
        pass
    else:
        raise AssertionError("optional key was not validated when present")
    js = to_json_schema(schema)
    assert js["required"] == ["outcome"]
    assert set(js["properties"]) == {"outcome", "witness_kind"}


# ---- world view -----------------------------------------------------------------

def test_legacy_manifest_is_a_filesystem_world(tmp_path):
    os.makedirs(os.path.join(tmp_path, "world", "app"))
    with open(os.path.join(tmp_path, "world_manifest.json"), "w") as f:
        json.dump({"captured_roots": ["/app"]}, f)
    view = load_world_view(str(tmp_path), live_world=False)
    assert view.mode == "captured" and view.scheme == "filesystem"
    assert view.resolve("/app/out.txt") == os.path.join(
        str(tmp_path), "world", "app/out.txt")
    assert view.resolve("/etc/passwd") is None     # outside captured roots
    desc = view.describe()
    assert desc["captured_roots"] == ["/app"]
    assert "absence proves nothing" in desc["address_mapping"]


def test_explicit_artifact_map_for_a_non_filesystem_world(tmp_path):
    os.makedirs(os.path.join(tmp_path, "world"))
    with open(os.path.join(tmp_path, "world_manifest.json"), "w") as f:
        json.dump({"scheme": "answer",
                   "artifacts": [{"address": "final answer",
                                  "host": "answer.txt"}]}, f)
    view = load_world_view(str(tmp_path), live_world=False)
    assert view.scheme == "answer"
    assert view.resolve("final answer") == os.path.join(
        str(tmp_path), "world", "answer.txt")
    assert view.describe()["artifacts"][0]["address"] == "final answer"


def test_no_world_is_live_or_unavailable(tmp_path):
    assert load_world_view(str(tmp_path), live_world=True).mode == "live"
    assert load_world_view(str(tmp_path), live_world=False).mode \
        == "unavailable"
