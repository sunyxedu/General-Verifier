"""Mechanical acceptance projection; no models, task inputs or network calls."""

from __future__ import annotations

import copy
import json

import pytest

from verifier.core.events import EventLog
from verifier.core.model import Claim, Evidence, Obligation, canonical, content_hash, weakest_strength
from verifier.core.store import ClaimStore
from verifier.engine.certificate import request_independent_certificate
from verifier.engine.kernel import Kernel
from verifier.execution.instruments import ArtifactStore
from verifier.profiles.cyber.acceptance import (
    ACCEPTANCE_SCOPE, SCHEMA, AcceptanceError, build_candidate_acceptance,
)
from verifier.profiles.cyber.phases import (
    AUTHORIZATION_CONTRACT_LOCUS, authorization_verification_context,
)


class NoModelBackend:
    def complete(self, *args, **kwargs):
        raise AssertionError("the acceptance fixture must never call a model")


def full_fixture(tmp_path, specifications=None, *, with_observation=False):
    """Use the host-independent production exporter, not a FINAL lookalike."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    specifications = specifications or [
        ("must", "decisive", "world", "world"),
        ("should", "bounded", "world", "world"),
    ]
    binding = {
        "certificate_scope": "candidate", "phase": "candidate_authorization",
        "profile_id": content_hash("synthetic-profile"),
        "task_sha256": content_hash("synthetic-task"), "delivery_pending": True,
        "verification_epoch": "synthetic-epoch",
        "capability_revision": content_hash("synthetic-capabilities"),
        "subject": {"candidate_id": content_hash("synthetic-candidate"),
                    "sha256": content_hash("synthetic-candidate"), "size": 7,
                    "blob_path": "/synthetic/private/blob", "address": "/synthetic/private/blob",
                    "address_source": "immutable_candidate"},
        "observations": [],
    }
    binding["delivery_contract"] = {
        "kind": "single_file_bytes", "destination": "/synthetic/final",
        "destination_source": "host_required",
        "source_candidate_id": binding["subject"]["candidate_id"],
        "source_sha256": binding["subject"]["sha256"], "source_size": 7,
        "physical_postconditions": ["exists", "regular_file", "exact_bytes"],
        "writer": "host_finalizer",
    }
    retained_effect = Claim("effect", "synthetic controller-retained evaluator invocation")
    if with_observation:
        binding["observations"] = [{
            "candidate_id": binding["subject"]["candidate_id"],
            "artifact_id": content_hash("synthetic-evaluator-artifact"),
            "artifact_path": "/synthetic/retained/observation.json",
            "effect_ref": "effect:1", "effect_claim_id": retained_effect.id,
            "event_seq": 1, "event_digest": content_hash("synthetic-event"),
            "evaluator_id": content_hash("synthetic-evaluator"),
            "environment": "synthetic-environment", "authority_scope": "candidate_signal",
        }]
    binding["context_id"] = content_hash(binding)
    metadata = {"profile": "cyber", "profile_id": binding["profile_id"],
                "certificate_scope": "candidate", "phase": "candidate_authorization",
                "task_sha256": binding["task_sha256"],
                "context_id": binding["context_id"], "binding": binding, "delivery_pending": True}
    store = ClaimStore(str(tmp_path / "claims.sqlite"), trigger="lazy")
    artifacts = ArtifactStore(str(tmp_path / "artifacts"))
    log = EventLog(str(tmp_path / "events.jsonl"))
    if with_observation:
        store.add_claim(retained_effect)
        store.record_evidence(retained_effect.id, {"outcome": "holds", "instrument": "ledger",
                                                  "digest": binding["observations"][0]["event_digest"]})
        store.set_verified(retained_effect.id, basis="record")
    final = Claim("assertion", canonical({"schema": "gv-cyber-candidate-authorization-v2", "binding": binding}),
                  tuple(item["effect_claim_id"] for item in binding["observations"]))
    store.add_claim(final)
    kernel = Kernel(store, log, NoModelBackend(), artifacts,
                    context_id_provider=lambda claim_id: binding["context_id"],
                    independent_root_provider=lambda claim_id: {
                        "scope": "candidate_authorization",
                        "contract_locus": AUTHORIZATION_CONTRACT_LOCUS,
                    } if claim_id == final.id else None)
    assert kernel.is_independent_root(final.id)  # Pin host authority BEFORE settlement.
    for index, (severity, strength, basis, floor) in enumerate(specifications):
        node = Obligation(final.id, f"synthetic condition {index}", floor,
                          severity=severity, clause=f"synthetic clause {index}",
                          verification_context=authorization_verification_context(binding))
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
    exported = request_independent_certificate(store, kernel, log, artifacts,
                                   final.id, profile_metadata=metadata)
    full = artifacts.get(exported["certificate_id"])
    assert content_hash(full) == exported["certificate_id"]
    return full, binding


def node(full, index=0):
    return next(item for item in root(full)["obligations"]
                if item["text"] == f"synthetic condition {index}")


def root(full):
    return next(item for item in full["slice"] if item["id"] == full["root"])


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
    assert result["root"] == full["root"]
    assert result["root_scope"] == "candidate_authorization"
    assert "final" not in result and "final" not in full
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
    (lambda f: f["slice"][-1].update(status="stale"), "root_not_verified"),
    (lambda f: f["slice"][-1].update(kind="final"), "root_statement_mismatch"),
    (lambda f: f["slice"][-1].update(text="solver substituted statement"), "root_statement_mismatch"),
    (lambda f: f.update(root=content_hash("another root")), "root_identity_mismatch"),
    (lambda f: f["slice"].append(copy.deepcopy(f["slice"][-1])), "ambiguous_root"),
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


@pytest.mark.parametrize("mutation", [
    lambda f: f.update(schema="gv-cyber-candidate-acceptance-v1"),
    lambda f: f.pop("schema"),
    lambda f: f.update(final=f["root"]),
    lambda f: f.update(root_scope="task_completion"),
    lambda f: root(f).update(kind="final"),
    lambda f: root(f).update(text="an ordinary solver assertion"),
])
def test_legacy_completion_and_ordinary_assertion_never_become_authorization(tmp_path, mutation):
    full, binding = full_fixture(tmp_path)
    mutation(full)
    with pytest.raises(AcceptanceError):
        build_candidate_acceptance(full, binding)


@pytest.mark.parametrize("target", ["full", "root"])
@pytest.mark.parametrize("mutation", [
    lambda source: source.update(effective_status="partial"),
    lambda source: source.update(effective_status="refuted"),
    lambda source: source.update(effective_status="stale"),
    lambda source: source.update(open_premises=[content_hash("open premise")]),
    lambda source: source.pop("open_premises"),
    lambda source: source.update(premise_refuted=content_hash("poisoning premise")),
    lambda source: source.pop("premise_refuted"),
    lambda source: source.pop("effective_basis"),
    lambda source: source.pop("effective_strength"),
])
def test_effective_root_and_full_state_are_required_without_open_or_poisoned_premises(tmp_path, target, mutation):
    full, binding = full_fixture(tmp_path)
    mutation(full if target == "full" else root(full))
    with pytest.raises(AcceptanceError):
        build_candidate_acceptance(full, binding)


@pytest.mark.parametrize("mutation", [
    lambda n: n.update(premise_scope=[content_hash("borrowed premise")]),
    lambda n: n.pop("premise_scope"),
    lambda n: n.update(verification_context=content_hash("old context")),
    lambda n: n.pop("verification_context"),
])
def test_obligations_require_persisted_independent_context_and_empty_premise_scope(tmp_path, mutation):
    full, binding = full_fixture(tmp_path)
    mutation(node(full))
    with pytest.raises(AcceptanceError):
        build_candidate_acceptance(full, binding)


@pytest.mark.parametrize("mutation", [
    lambda b: b.update(phase="task_completion"),
    lambda b: b.update(delivery_pending=False),
    lambda b: b["subject"].update(address="/synthetic/final", address_source="host_required"),
    lambda b: b["subject"].update(blob_path="/synthetic/other", address="/synthetic/other"),
    lambda b: b["subject"].update(sha256=content_hash("different bytes")),
    lambda b: b["delivery_contract"].update(destination="/synthetic/other"),
    lambda b: b["delivery_contract"].update(writer="solver"),
    lambda b: b["delivery_contract"].update(physical_postconditions=["exists"]),
    lambda b: b["delivery_contract"].update(physical_postconditions=["exists", "regular_file", "exact_bytes", "security_success"]),
    lambda b: b["delivery_contract"].update(source_size=True),
    lambda b: b["delivery_contract"].update(kind="deploy_script"),
    lambda b: b["delivery_contract"].update(extra="undeclared host action"),
    lambda b: b.update(authorization_certificate_id=content_hash("stolen authorization")),
])
def test_phase_subject_and_delivery_plan_changes_cannot_replay_old_authorization(tmp_path, mutation):
    full, binding = full_fixture(tmp_path)
    binding = copy.deepcopy(binding)
    mutation(binding)
    binding["context_id"] = content_hash({key: value for key, value in binding.items() if key != "context_id"})
    with pytest.raises(AcceptanceError):
        build_candidate_acceptance(full, binding)


def test_authenticated_record_effect_is_retained_without_upgrading_its_world_basis(tmp_path):
    full, binding = full_fixture(tmp_path, with_observation=True)
    result = build_candidate_acceptance(full, binding)
    assert len(full["slice"]) == 2
    assert root(full)["deps"] == [binding["observations"][0]["effect_claim_id"]]
    assert next(entry for entry in full["slice"] if entry["kind"] == "effect")["basis"] == "record"
    assert result["full_certificate_id"] == content_hash(full)
    assert "final" not in result


@pytest.mark.parametrize("mutation", [
    lambda f, dep: f["slice"].remove(dep),
    lambda f, dep: f["slice"].append(copy.deepcopy(dep)),
    lambda f, dep: dep.update(status="partial"),
    lambda f, dep: dep.update(effective_status="refuted"),
    lambda f, dep: dep.update(open_premises=[content_hash("unknown premise")]),
    lambda f, dep: dep.update(premise_refuted=content_hash("bad premise")),
    lambda f, dep: dep.update(kind="assertion"),
    lambda f, dep: dep.update(deps=[content_hash("extra dependency")]),
    lambda f, dep: dep.update(text="substituted retained event"),
    lambda f, dep: dep.update(effective_basis="world"),
    lambda f, dep: dep.update(evidence=[]),
    lambda f, dep: dep["evidence"][-1].update(outcome="failed"),
    lambda f, dep: root(f).update(deps=[]),
    lambda f, dep: root(f).pop("deps"),
])
def test_complete_effect_closure_cannot_hide_invalid_or_foreign_support(tmp_path, mutation):
    full, binding = full_fixture(tmp_path, with_observation=True)
    dep = next(entry for entry in full["slice"] if entry["kind"] == "effect")
    mutation(full, dep)
    with pytest.raises(AcceptanceError):
        build_candidate_acceptance(full, binding)


def test_unexpected_extra_slice_claim_cannot_be_filtered_away(tmp_path):
    full, binding = full_fixture(tmp_path)
    extra = copy.deepcopy(root(full))
    extra["id"] = content_hash("foreign claim")
    extra["status"] = "refuted"
    full["slice"].append(extra)
    with pytest.raises(AcceptanceError, match="incomplete_authorization_slice"):
        build_candidate_acceptance(full, binding)


def test_full_certificate_identity_prevents_acceptance_replay_after_evidence_change(tmp_path):
    full, binding = full_fixture(tmp_path)
    first = build_candidate_acceptance(full, binding)
    changed = copy.deepcopy(full)
    evidence(changed)["artifact"] = content_hash("new retained artifact")
    item = evidence(changed)
    item["evidence_id"] = Evidence(node(changed)["id"], item["instrument"], "holds",
                                   {"artifact": item["artifact"]}, item["witness"]).id
    second = build_candidate_acceptance(changed, binding)
    assert first["root"] == second["root"]
    assert first["full_certificate_id"] != second["full_certificate_id"]
    assert content_hash(first) != content_hash(second)


@pytest.mark.parametrize("mutation", [
    lambda metadata: metadata.update(delivery_pending=1),
    lambda metadata: metadata["binding"].update(delivery_pending=1),
    lambda metadata: metadata["binding"]["subject"].update(size=7.0),
    lambda metadata: metadata["binding"]["delivery_contract"].update(source_size=7.0),
])
def test_profile_metadata_uses_typed_identity_not_python_loose_equality(tmp_path, mutation):
    full, binding = full_fixture(tmp_path)
    full = copy.deepcopy(full)
    mutation(full["profile_metadata"])
    with pytest.raises(AcceptanceError, match="binding_mismatch"):
        build_candidate_acceptance(full, binding)
