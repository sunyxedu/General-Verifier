"""Mechanical acceptance projection; no models, task inputs or network calls."""

from __future__ import annotations

import copy
import json

import pytest

from verifier.core.events import EventLog
from verifier.core.model import Claim, Evidence, Obligation, canonical, content_hash, weakest_strength
from verifier.core.store import ClaimStore
from verifier.engine.certificate import request_certificate
from verifier.execution.instruments import ArtifactStore
from verifier.profiles.cyber.acceptance import (
    ACCEPTANCE_SCOPE, SCHEMA, AcceptanceError, build_candidate_acceptance,
)


class AlreadySettledKernel:
    parallel_claims = 1

    def verify(self, claim_id):
        raise AssertionError("the projection must never settle claims or call a model")


def full_fixture(tmp_path, specifications=None):
    """Use the unmodified production GV certificate exporter, not a lookalike."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    specifications = specifications or [
        ("must", "decisive", "world", "world"),
        ("should", "bounded", "world", "world"),
    ]
    binding = {
        "certificate_scope": "candidate", "profile_id": content_hash("synthetic-profile"),
        "task_sha256": content_hash("synthetic-task"), "delivery_pending": True,
        "verification_epoch": "synthetic-epoch",
        "subject": {"candidate_id": content_hash("synthetic-candidate"),
                    "sha256": content_hash("synthetic-candidate"), "size": 7,
                    "blob_path": "/synthetic/private/blob", "address": "/synthetic/final",
                    "address_source": "host_required"},
        "observations": [],
    }
    binding["context_id"] = content_hash(binding)
    metadata = {"profile": "cyber", "profile_id": binding["profile_id"],
                "certificate_scope": "candidate", "task_sha256": binding["task_sha256"],
                "context_id": binding["context_id"], "binding": binding, "delivery_pending": True}
    store = ClaimStore(str(tmp_path / "claims.sqlite"), trigger="lazy")
    artifacts = ArtifactStore(str(tmp_path / "artifacts"))
    log = EventLog(str(tmp_path / "events.jsonl"))
    final = Claim("final", canonical({"schema": "gv-cyber-candidate-v1", "binding": binding}))
    store.add_claim(final)
    for index, (severity, strength, basis, floor) in enumerate(specifications):
        node = Obligation(final.id, f"synthetic condition {index}", floor,
                          severity=severity, clause=f"synthetic clause {index}",
                          verification_context=binding["context_id"])
        store.add_obligation(final.id, node)
        artifact = artifacts.put({"synthetic": index, "audit": "controller retained"})
        evidence = {
            "obligation": node.text, "required_basis": floor, "instrument": "synthetic instrument",
            "outcome": "holds", "basis": basis, "strength": strength,
            "bridges": [] if strength == "decisive" else ["synthetic audited bridge"],
            "assumed_premises": [], "artifact": artifact, "witness": "synthetic retained witness",
            "witness_kind": "execution", "severity": severity, "clause": node.clause,
            "inputs": [{"ref": "/synthetic/authority", "provenance": "task_given", "role": "authority"}],
        }
        evidence["evidence_id"] = Evidence(node.id, evidence["instrument"], "holds",
                                          {"artifact": artifact}, evidence["witness"]).id
        store.record_evidence(node.id, evidence)
        store.set_verified(node.id, basis=basis, strength=strength)
    strength = weakest_strength([item[1] for item in specifications])
    basis = "world" if all(item[2] == "world" for item in specifications) else "record"
    store.set_verified(final.id, basis=basis, strength=strength)
    exported = request_certificate(store, AlreadySettledKernel(), log, artifacts,
                                   final.id, profile_metadata=metadata)
    full = artifacts.get(exported["certificate_id"])
    assert content_hash(full) == exported["certificate_id"]
    return full, binding


def node(full, index=0):
    return full["slice"][-1]["obligations"][index]


def evidence(full, index=0):
    return node(full, index)["evidence"][-1]


def test_bound_advice_does_not_demote_independent_must_acceptance(tmp_path):
    full, binding = full_fixture(tmp_path)
    before_full, before_binding = copy.deepcopy(full), copy.deepcopy(binding)
    result = build_candidate_acceptance(full, binding)
    assert full["verdict"] == "supported"
    assert full["verdict_strength"] == "bounded"
    assert result["schema"] == SCHEMA
    assert result["verdict"] == "verified"
    assert result["verdict_strength"] == "decisive"
    assert result["full_certificate_id"] == content_hash(full)
    assert result["full_verdict"] == "supported"
    assert result["final"] == full["final"]
    assert result["profile_metadata"]["binding"] == binding
    assert result["profile_metadata"]["acceptance_scope"] == ACCEPTANCE_SCOPE
    assert result["must_evidence"][0]["artifact_id"] == evidence(full)["artifact"]
    assert result["advisory_evidence"][0]["strength"] == "bounded"
    assert full == before_full and binding == before_binding
    result["profile_metadata"]["binding"]["subject"]["size"] = 999
    assert binding == before_binding


@pytest.mark.parametrize("advice_strength", ["decisive", "bounded", "analytic"])
def test_all_holding_advisory_tiers_preserve_the_full_audit(tmp_path, advice_strength):
    full, binding = full_fixture(tmp_path, [("must", "decisive", "world", "world"),
                                            ("should", advice_strength, "world", "world")])
    result = build_candidate_acceptance(full, binding)
    assert result["advisory_evidence"][0]["strength"] == advice_strength
    assert result["full_certificate_id"] == content_hash(full)


def test_record_advice_cannot_upgrade_record_must(tmp_path):
    full, binding = full_fixture(tmp_path, [("must", "decisive", "world", "world"),
                                            ("should", "decisive", "record", "record")])
    assert build_candidate_acceptance(full, binding)["verdict"] == "verified"
    full, binding = full_fixture(tmp_path / "second", [("must", "decisive", "record", "record"),
                                                       ("should", "decisive", "world", "world")])
    with pytest.raises(AcceptanceError, match="must_not_decisive_world"):
        build_candidate_acceptance(full, binding)


@pytest.mark.parametrize("state", ["refuted", "partial", "unverified", "stale", None, "unknown"])
@pytest.mark.parametrize("index", [0, 1])
def test_any_unsettled_or_refuted_obligation_fails_closed(tmp_path, state, index):
    full, binding = full_fixture(tmp_path)
    node(full, index)["status"] = state
    with pytest.raises(AcceptanceError, match="obligation_not_verified"):
        build_candidate_acceptance(full, binding)


@pytest.mark.parametrize("verdict", ["partial", "refuted", "unverified", "unknown", None])
def test_noneligible_full_verdict_never_authorizes(tmp_path, verdict):
    full, binding = full_fixture(tmp_path)
    full["verdict"] = verdict
    with pytest.raises(AcceptanceError, match="full_verdict_not_eligible"):
        build_candidate_acceptance(full, binding)


@pytest.mark.parametrize("strength", ["bounded", "analytic"])
def test_supported_security_must_is_not_promoted(tmp_path, strength):
    full, binding = full_fixture(tmp_path, [("must", strength, "world", "world")])
    with pytest.raises(AcceptanceError, match="must_not_decisive_world"):
        build_candidate_acceptance(full, binding)


def test_at_least_one_must_is_required(tmp_path):
    full, binding = full_fixture(tmp_path, [("should", "decisive", "world", "world")])
    with pytest.raises(AcceptanceError, match="missing_must_obligation"):
        build_candidate_acceptance(full, binding)


@pytest.mark.parametrize("mutation,reason", [
    (lambda f: node(f).update(severity="unknown"), "invalid_severity"),
    (lambda f: node(f).update(strength=None), "invalid_obligation_strength"),
    (lambda f: node(f).update(strength="strong"), "invalid_obligation_strength"),
    (lambda f: node(f).update(basis="unknown"), "invalid_obligation_basis"),
    (lambda f: evidence(f).pop("required_basis"), "invalid_evidence_floor"),
    (lambda f: evidence(f).update(required_basis="unknown"), "invalid_evidence_floor"),
    (lambda f: evidence(f).update(outcome="failed"), "evidence_not_holds"),
    (lambda f: evidence(f).update(outcome="unobtainable"), "evidence_not_holds"),
    (lambda f: evidence(f).update(basis="record"), "evidence_grade_mismatch"),
    (lambda f: evidence(f).update(strength="bounded"), "evidence_grade_mismatch"),
    (lambda f: evidence(f).update(severity="should"), "evidence_statement_mismatch"),
    (lambda f: evidence(f).update(clause="different"), "evidence_statement_mismatch"),
    (lambda f: evidence(f).update(assumed_premises=["foreign premise"]), "assumed_premises"),
    (lambda f: evidence(f).pop("assumed_premises"), "assumed_premises"),
    (lambda f: evidence(f).update(bridges="not typed"), "invalid_bridges"),
    (lambda f: evidence(f).update(bridges=["undischarged assumption"]), "decisive_must_has_bridges"),
    (lambda f: evidence(f).update(inputs=[]), "invalid_evidence_inputs"),
    (lambda f: evidence(f).update(inputs=[{"ref":"synthetic","provenance":"unknown","role":"authority"}]), "invalid_evidence_inputs"),
    (lambda f: evidence(f).update(inputs=[{"ref":"synthetic","provenance":"delivered","role":"authority"}]), "delivered_authority"),
    (lambda f: evidence(f).update(evidence_id=content_hash("unrelated")), "evidence_identity_mismatch"),
    (lambda f: node(f).update(id=content_hash("other epoch")), "obligation_context_mismatch"),
    (lambda f: node(f).update(refutable_by_absence=1), "invalid_absence_rule"),
    (lambda f: node(f).update(evidence=[]), "missing_settled_evidence"),
    (lambda f: evidence(f).pop("evidence_id"), "invalid_evidence_identity"),
    (lambda f: f.update(verdict_strength="decisive"), "full_strength_mismatch"),
    (lambda f: f.update(verdict="verified"), "full_verdict_strength_mismatch"),
    (lambda f: f["slice"][-1].update(strength="decisive"), "full_strength_mismatch"),
    (lambda f: f["slice"][-1].update(basis="record"), "full_basis_mismatch"),
    (lambda f: f["slice"][-1].update(status="stale"), "final_not_verified"),
    (lambda f: f["slice"][-1].update(kind="assertion"), "final_statement_mismatch"),
    (lambda f: f["slice"][-1].update(text="solver substituted statement"), "final_statement_mismatch"),
    (lambda f: f.update(final=content_hash("another final")), "final_identity_mismatch"),
    (lambda f: f["slice"].append(copy.deepcopy(f["slice"][-1])), "ambiguous_final"),
    (lambda f: f["slice"][-1]["obligations"].append(copy.deepcopy(node(f))), "duplicate_obligation"),
])
def test_malformed_or_misbound_evidence_rejects(tmp_path, mutation, reason):
    full, binding = full_fixture(tmp_path)
    mutation(full)
    with pytest.raises(AcceptanceError, match=reason):
        build_candidate_acceptance(full, binding)


def test_world_floor_for_advice_is_not_relaxed(tmp_path):
    full, binding = full_fixture(tmp_path, [("must", "decisive", "world", "world"),
                                            ("should", "bounded", "record", "world")])
    with pytest.raises(AcceptanceError, match="evidence_below_floor"):
        build_candidate_acceptance(full, binding)


@pytest.mark.parametrize("field,value", [("verification_epoch", "other epoch"),
                                          ("task_sha256", content_hash("different task")),
                                          ("profile_id", content_hash("different profile"))])
def test_changed_host_identity_does_not_reuse_old_cert(tmp_path, field, value):
    full, binding = full_fixture(tmp_path)
    binding = copy.deepcopy(binding)
    binding[field] = value
    binding["context_id"] = content_hash({k:v for k,v in binding.items() if k != "context_id"})
    with pytest.raises(AcceptanceError, match="binding_mismatch"):
        build_candidate_acceptance(full, binding)


@pytest.mark.parametrize("field,value", [("size", True), ("address_source", "solver authority")])
def test_binding_type_and_authority_are_strict(tmp_path, field, value):
    full, binding = full_fixture(tmp_path)
    binding["subject"][field] = value
    binding["context_id"] = content_hash({k:v for k,v in binding.items() if k != "context_id"})
    with pytest.raises(AcceptanceError):
        build_candidate_acceptance(full, binding)


def test_errors_do_not_echo_private_or_candidate_strings(tmp_path):
    full, binding = full_fixture(tmp_path)
    secret = "not-a-real-secret--synthetic-do-not-echo"
    full["verdict"] = secret
    with pytest.raises(AcceptanceError) as error:
        build_candidate_acceptance(full, binding)
    assert str(error.value) == error.value.reason_code == "full_verdict_not_eligible"
    assert secret not in str(error.value) and "/synthetic/" not in str(error.value)


def test_projection_is_deterministic_and_roundtrips_json(tmp_path):
    full, binding = full_fixture(tmp_path)
    first = build_candidate_acceptance(full, binding)
    second = build_candidate_acceptance(json.loads(json.dumps(full)), json.loads(json.dumps(binding)))
    assert first == second and content_hash(first) == content_hash(second)


@pytest.mark.parametrize("full,binding", [(None, {}), ({}, None), ([], []), ({}, {})])
def test_invalid_top_level_shapes_only_raise_safe_typed_errors(full, binding):
    with pytest.raises(AcceptanceError):
        build_candidate_acceptance(full, binding)
