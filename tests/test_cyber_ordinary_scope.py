"""Ordinary cyber claims retain GV residual scope, not candidate authority.

Deterministic seats check the actual host/Kernel inputs and authority gates.
They do not claim to measure how a model will decompose a proposition.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_cyber_service import RecordingBackend, registered, service_fixture
from verifier.profiles.cyber.prompts import ANSWER_BLIND, SCOPE, SYSTEMS


ORDINARY_STATEMENT = "The named public source rejects values outside its declared input domain."
CLAIMED_BINDING = "A host-generated immutable binding maps label synthetic to the fixed task subject and destination."
TASK = "Establish the required security condition. Preserve the required normal behavior."
PRIVATE_BINDING_FIELDS = {"subject", "binding", "delivery_contract", "delivery_pending",
                          "authorization_root_id", "authorization_certificate_id",
                          "receipt_artifact_id", "delivery_id"}


def scoped_backend(*, outcome="holds"):
    backend = RecordingBackend(outcome=outcome)

    def obligations(variables):
        context = json.loads(variables["profile_context"])
        if context["phase"] == "ordinary_claim":
            assert variables["claim_kind"] != "final"
            controlling = variables["claim_text"]
        else:
            controlling = json.loads(variables["task_context"])["text"]
        return {"obligations": [{"text": controlling, "required_basis": "world",
                                 "severity": "must", "clause": controlling}]}

    backend.fixtures["obligations"] = obligations
    return backend


def register_candidates(service, project, count):
    if count:
        first = registered(service)
        Path(project, "another.bin").write_bytes(b"a distinct synthetic candidate")
        second = service.cyber_register_candidate("another.bin", "another-label")
        assert "error" not in second and first["candidate_id"] != second["candidate_id"]
    assert len(service.cyber_candidates()["candidates"]) == count


def compiler_and_critic(backend, statement):
    return [call for call in backend.calls if call["role"] in ("obligations", "critic")
            and call["variables"]["claim_text"] == statement]


@pytest.mark.parametrize("candidate_count", [0, 2])
def test_ordinary_assertion_has_explicit_residual_scope_without_candidate_authority(tmp_path, candidate_count):
    service, backend, project, _ = service_fixture(tmp_path, backend=scoped_backend(), task=TASK)
    register_candidates(service, project, candidate_count)
    result = service.commit("ordinary-source-fact", "assertion", ORDINARY_STATEMENT, [])
    assert result["status"] == "verified"
    calls = compiler_and_critic(backend, ORDINARY_STATEMENT)
    assert [call["role"] for call in calls] == ["obligations", "critic"]
    for call in calls:
        variables = call["variables"]
        context = json.loads(variables["profile_context"])
        assert context["phase"] == "ordinary_claim" and context["claim_kind"] == "assertion"
        assert context["certificate_scope"] == "task"
        assert not PRIVATE_BINDING_FIELDS.intersection(context)
        assert "this claim is NOT a completion claim" in variables["contract_locus"]
        assert "CANDIDATE AUTHORIZATION CONTRACT" not in variables["contract_locus"]
        if call["role"] == "obligations":
            assert "withheld" not in variables["graph"]
        assert "withheld" not in variables["dependency_referent_context"]
        assert json.loads(variables["verified_deps"]) == []
        assert json.loads(variables["pending_deps"]) == []
    assert not service.kernel.is_independent_root(result["claim_id"])
    assert result["claim_id"] not in service._cyber_bindings
    assert [node["text"] for node in service.store.obligation_children(result["claim_id"])] == [ORDINARY_STATEMENT]
    assert "error" in service.certificate("ordinary-source-fact")
    assert not any(event["type"] in ("independent_certificate", "cyber_materialized")
                   for event in service.log.replay())


def test_ordinary_claim_keeps_real_dependency_referents_and_residual_premises(tmp_path):
    service, backend, _, _ = service_fixture(tmp_path, backend=scoped_backend(), task=TASK)
    premise = service.commit("source-premise", "assertion", ORDINARY_STATEMENT, [])
    child_statement = "The same named source preserves the declared normal input behavior."
    child = service.commit("source-child", "assertion", child_statement, ["source-premise"])
    assert child["status"] == "verified"
    for call in compiler_and_critic(backend, child_statement):
        variables = call["variables"]
        assert json.loads(variables["profile_context"])["phase"] == "ordinary_claim"
        assert any(ORDINARY_STATEMENT in dep for dep in json.loads(variables["verified_deps"]))
        assert premise["claim_id"] in variables["dependency_referent_context"]
        assert "withheld" not in variables["dependency_referent_context"]
    assert service.store.parents(child["claim_id"]) == [premise["claim_id"]]
    assert not service.kernel.is_independent_root(child["claim_id"])


@pytest.mark.parametrize("candidate_count", [0, 2])
def test_explicit_binding_proposition_is_not_excused_or_granted_authority(tmp_path, candidate_count):
    service, backend, project, _ = service_fixture(tmp_path,
        backend=scoped_backend(outcome="unobtainable"), task=TASK)
    register_candidates(service, project, candidate_count)
    result = service.commit("asserted-binding", "assertion", CLAIMED_BINDING, [])
    assert result["status"] != "verified"
    nodes = service.store.obligation_children(result["claim_id"])
    assert len(nodes) == 1 and nodes[0]["text"] == CLAIMED_BINDING
    assert nodes[0]["severity"] == "must" and nodes[0]["status"] != "verified"
    for call in compiler_and_critic(backend, CLAIMED_BINDING):
        context = json.loads(call["variables"]["profile_context"])
        assert context["phase"] == "ordinary_claim" and not PRIVATE_BINDING_FIELDS.intersection(context)
    assert not service.kernel.is_independent_root(result["claim_id"])
    assert result["claim_id"] not in service._cyber_bindings


def test_unbound_public_final_still_has_original_independent_whole_task_contract(tmp_path):
    service, backend, _, _ = service_fixture(tmp_path, backend=scoped_backend(), task=TASK)
    service.commit("source-premise", "assertion", ORDINARY_STATEMENT, [])
    statement = "Only the source fact has been shown."
    final = service.commit("ordinary-public-final", "final", statement, ["source-premise"])
    assert final["status"] == "verified"
    for call in compiler_and_critic(backend, statement):
        variables = call["variables"]
        context = json.loads(variables["profile_context"])
        assert context["phase"] == "ordinary_final" and context["claim_kind"] == "final"
        assert not PRIVATE_BINDING_FIELDS.intersection(context)
        assert "completion claim: the controlling proposition is the authoritative task contract" in variables["contract_locus"]
        assert json.loads(variables["task_context"])["text"] == TASK
        assert json.loads(variables["verified_deps"]) == json.loads(variables["pending_deps"]) == []
        if call["role"] == "obligations":
            assert "withheld" in variables["graph"]
        assert "withheld" in variables["dependency_referent_context"]
    assert [node["text"] for node in service.store.obligation_children(final["claim_id"])] == [TASK]
    certificate = service.certificate("ordinary-public-final")
    artifact = service.artifacts.get(certificate["certificate_id"])
    assert artifact["final"] == final["claim_id"] and "root" not in artifact
    assert not service.kernel.is_independent_root(final["claim_id"])
    assert not any(event["type"] == "cyber_materialized" for event in service.log.replay())


def test_true_authorization_and_completion_keep_host_binding_and_entire_task(tmp_path):
    service, backend, project, _ = service_fixture(tmp_path, backend=scoped_backend(), task=TASK)
    register_candidates(service, project, 2)
    candidate = service.cyber_candidates()["candidates"][0]
    service.commit("ordinary-source-fact", "assertion", ORDINARY_STATEMENT, [])
    backend.calls.clear()
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result["task_completed"] is True, result
    for call in (call for call in backend.calls if call["role"] in ("obligations", "critic")):
        variables = call["variables"]
        context = json.loads(variables["profile_context"])
        assert context["phase"] in ("candidate_authorization", "task_completion")
        assert "claim_kind" not in context  # Exact phase binding schema stays unchanged.
        assert context["subject"]["candidate_id"] == candidate["candidate_id"]
        assert json.loads(variables["verified_deps"]) == json.loads(variables["pending_deps"]) == []
        if call["role"] == "obligations":
            assert "withheld" in variables["graph"]
        assert "withheld" in variables["dependency_referent_context"]
        assert json.loads(variables["task_context"])["text"] == TASK
        if context["phase"] == "candidate_authorization":
            assert context["subject"]["address"] == candidate["blob_path"]
            assert "CANDIDATE AUTHORIZATION CONTRACT" in variables["contract_locus"]
        else:
            assert context["subject"]["address"] == str(project / "answer.bin")
            assert context["delivery_pending"] is False
    for claim_id in (result["authorization_root_id"], result["final_claim_id"]):
        assert [node["text"] for node in service.store.obligation_children(claim_id)] == [TASK]
    assert service.kernel.is_independent_root(result["authorization_root_id"])
    assert not service.kernel.is_independent_root(result["final_claim_id"])


def test_scope_prompt_replaces_unconditional_candidate_focus_with_bounded_role_rules():
    assert "missing candidate binding adds no requirement unless asserted" in SCOPE
    assert "Host binding, when present, supplies identity only" in ANSWER_BLIND
    critic = SYSTEMS["critic"]
    assert "Only host authorization/completion bindings select the exact candidate" in critic
    assert "ordinary claims use their statement/dependency referents" in critic
    assert "task instance and exact candidate" not in critic
    assert all(len(system.split()) < 1500 for system in SYSTEMS.values())
