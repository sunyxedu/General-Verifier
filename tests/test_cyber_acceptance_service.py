"""Real GV role/binding/certificate delivery using deterministic fixture seats.

No model, benchmark API, task solution or private evaluator is involved.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import pytest

from test_cyber_service import JUDGE_YES, RecordingBackend, registered, service_fixture
from verifier.core.model import content_hash
from verifier.profiles.cyber.evaluator import EvaluatorSpec
from verifier.profiles.cyber.evidence import SeatCapabilities


def mixed_backend(*, advice_refuted=False, must_strength="decisive", completion_supported=False):
    backend = RecordingBackend()
    backend.fixtures["obligations"] = {"obligations": [
        {"text": "synthetic required condition", "required_basis": "world", "severity": "must",
         "clause": "Establish the required condition."},
        {"text": "synthetic advisory condition", "required_basis": "world", "severity": "should",
         "clause": "Prefer the advisory condition."},
    ]}

    def oracles(variables):
        phase = json.loads(variables["profile_context"])["phase"]
        advice = variables["obligation_text"] == "synthetic advisory condition"
        if phase == "candidate_authorization":
            strength = ("decisive" if advice_refuted else "bounded") if advice else must_strength
        else:
            assert phase == "task_completion"
            strength = "bounded" if completion_supported and advice else "decisive"
        return {"methods": [{"procedure": "synthetic independent instrument",
                             "artifact": "synthetic observation", "level": "synthetic",
                             "basis": "world", "strength": strength, "bridges": []}]}

    original = backend.fixtures["discharge"]

    def discharge(variables):
        result = original(variables)
        if advice_refuted and variables["obligation_text"] == "synthetic advisory condition":
            result["outcome"] = "refuted"
        return result

    def judger(variables):
        result = dict(JUDGE_YES)
        if variables["declared_strength"] != "decisive":
            result["degradation"] = "justified"
        return result

    backend.fixtures.update(oracles=oracles, discharge=discharge, judger=judger)
    return backend


def delivered(tmp_path, *, evaluator_spec=None):
    service, backend, project, state = service_fixture(
        tmp_path, backend=mixed_backend(), evaluator_spec=evaluator_spec)
    candidate = registered(service)
    if evaluator_spec is not None:
        assert "artifact_id" in service.cyber_evaluate_candidate(candidate["candidate_id"])
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result.get("materialized") is True and result.get("task_completed") is True, result
    return service, backend, project, state, candidate, result


def full(service, result):
    return service.artifacts.get(result["authorization_full_certificate_id"])


def authorization_root(service, result):
    return next(entry for entry in full(service, result)["slice"]
                if entry["id"] == result["authorization_root_id"])


def completion(service, result):
    return service.artifacts.get(result["certificate_id"])


def tamper(service, artifact_id):
    artifact = service.artifacts.get(artifact_id)
    artifact["synthetic_tamper"] = "a fixture mutation"
    Path(service.artifacts.root, artifact_id + ".json").write_text(json.dumps(artifact), encoding="utf-8")


def test_holding_bounded_advice_gets_two_authenticated_scopes_after_all_roles(tmp_path):
    service, backend, project, _, candidate, result = delivered(tmp_path)
    assert full(service, result)["verdict"] == "supported"
    assert "full_verdict" not in result and "full_certificate_id" not in result
    assert result["authorization_certificate_id"] != result["authorization_full_certificate_id"]
    acceptance = service.artifacts.get(result["authorization_certificate_id"])
    assert content_hash(acceptance) == result["authorization_certificate_id"]
    assert acceptance["full_certificate_id"] == result["authorization_full_certificate_id"]
    assert acceptance["verdict"] == result["verdict"] == "verified"
    assert acceptance["profile_metadata"]["binding"]["subject"]["candidate_id"] == candidate["candidate_id"]
    assert acceptance["root"] == result["authorization_root_id"] and "final" not in acceptance
    assert acceptance["profile_metadata"]["phase"] == "candidate_authorization"
    final = completion(service, result)
    assert final["final"] == result["final_claim_id"] and "root" not in final
    assert final["profile_metadata"]["phase"] == "task_completion"
    assert final["verdict"] == "verified" and final["verdict_strength"] == "decisive"
    assert result["certificate_id"] != result["authorization_certificate_id"]
    roles = [call["role"] for call in backend.calls]
    assert roles.count("obligations") == roles.count("critic") == 2
    assert roles.count("oracles") == roles.count("discharge") == roles.count("judger") == 4
    assert (project / "answer.bin").read_bytes() == (project / "proposed.bin").read_bytes()
    assert result["official_score"] is None
    before = len(backend.calls)
    assert service._authenticate_candidate_certificate(result["authorization_certificate_id"], result["authorization_root_id"],
                                                     result["authorization_full_certificate_id"])[0] == acceptance
    assert len(backend.calls) == before


def test_advisory_refutation_stays_fail_closed(tmp_path):
    service, _, project, _ = service_fixture(tmp_path, backend=mixed_backend(advice_refuted=True))
    candidate = registered(service)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result["verdict"] == "partial" and result["materialized"] is False
    assert "full_certificate_id" not in result and not (project / "answer.bin").exists()


def test_bounded_must_cannot_be_promoted_by_holding_advice(tmp_path):
    service, _, project, _ = service_fixture(tmp_path, backend=mixed_backend(must_strength="bounded"))
    candidate = registered(service)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result["verdict"] == "supported" and result["materialized"] is False
    assert not (project / "answer.bin").exists()


def test_supported_whole_task_completion_cannot_use_authorization_projection(tmp_path):
    from verifier.profiles.cyber.acceptance import AcceptanceError, build_candidate_acceptance

    service, backend, project, _ = service_fixture(tmp_path,
        backend=mixed_backend(completion_supported=True))
    candidate = registered(service)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result["phase"] == "task_completion" and result["verdict"] == "supported", result
    assert result["task_completed"] is False and result["verified"] is False
    assert result["materialized"] is False and result["output_rolled_back"] is True
    assert not (project / "answer.bin").exists()
    assert not any(event["type"] == "cyber_materialized" for event in service.log.replay())
    authorization = service.artifacts.get(result["authorization_certificate_id"])
    assert authorization["verdict"] == "verified"
    assert authorization["root"] == result["authorization_root_id"] and "final" not in authorization
    final = completion(service, result)
    assert final["verdict"] == "supported" and final["verdict_strength"] == "bounded"
    assert final["final"] == result["final_claim_id"]
    with pytest.raises(AcceptanceError, match="invalid_binding_scope"):
        build_candidate_acceptance(final, service._cyber_bindings[result["final_claim_id"]])
    roles = [call["role"] for call in backend.calls]
    assert roles.count("obligations") == roles.count("critic") == 2
    assert roles.count("oracles") == roles.count("discharge") == roles.count("judger") == 4


def test_new_hash_top_verdict_cannot_promote_supported_whole_task_final(tmp_path):
    service, backend, project, _ = service_fixture(tmp_path,
        backend=mixed_backend(completion_supported=True))
    candidate = registered(service)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result["phase"] == "task_completion" and result["verdict"] == "supported"
    forged = completion(service, result)
    assert forged["final"] == result["final_claim_id"]
    forged.update(verdict="verified", verdict_strength="decisive")
    forged_id = service.artifacts.put(forged)
    assert forged_id != result["certificate_id"]
    payload = next(event["payload"] for event in service.log.replay()
                   if event["type"] == "cyber_candidate_delivered")
    payload = {**payload, "schema": "gv-cyber-completed-delivery-v2",
               "certificate_id": forged_id, "final_claim_id": result["final_claim_id"]}
    calls = len(backend.calls)
    with pytest.raises(ValueError, match="verdict differs from the settled root"):
        service._authenticate_completion_certificate(payload)
    assert len(backend.calls) == calls and not (project / "answer.bin").exists()
    assert service.artifacts.get(result["authorization_certificate_id"])["verdict"] == "verified"


def test_two_certificate_repeat_is_idempotent_without_reverification(tmp_path):
    service, backend, _, _, candidate, first = delivered(tmp_path)
    calls = len(backend.calls)
    repeat = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert repeat["materialized"] is True
    assert repeat["certificate_id"] == first["certificate_id"]
    assert repeat["final_claim_id"] == first["final_claim_id"]
    assert repeat["authorization_certificate_id"] == first["authorization_certificate_id"]
    assert repeat["authorization_full_certificate_id"] == first["authorization_full_certificate_id"]
    assert repeat["authorization_root_id"] == first["authorization_root_id"]
    assert len(backend.calls) == calls


def test_same_receipt_cannot_replay_completion_for_a_new_delivery_transaction(tmp_path):
    from verifier.profiles.cyber.phases import completion_statement, make_completion_binding

    service, backend, project, _, candidate, result = delivered(tmp_path)
    old_binding = service._cyber_bindings[result["final_claim_id"]]
    old_receipt = service.artifacts.get(old_binding["receipt_artifact_id"])
    new_delivery_id = uuid.uuid4().hex
    assert new_delivery_id != old_binding["delivery_id"]
    new_binding = make_completion_binding(service._cyber_bindings[result["authorization_root_id"]],
        authorization_root_id=result["authorization_root_id"],
        authorization_certificate_id=result["authorization_certificate_id"],
        receipt_artifact_id=old_binding["receipt_artifact_id"], delivery_id=new_delivery_id,
        verification_epoch=old_binding["verification_epoch"])
    # Even if the OS later reuses every receipt identity, a new host transaction
    # must not hit the old settled completion. No filesystem metadata is mocked.
    assert service.artifacts.get(new_binding["receipt_artifact_id"]) == old_receipt
    assert new_binding["context_id"] != old_binding["context_id"]
    assert completion_statement(new_binding) != completion_statement(old_binding)
    assert service._adjudication_key(new_binding) != service._adjudication_key(old_binding)
    calls = len(backend.calls)
    with pytest.raises(ValueError, match="matching host delivery transaction"):
        service._validate_binding(new_binding)
    payload = next(event["payload"] for event in service.log.replay()
                   if event["type"] == "cyber_materialized")
    with pytest.raises(ValueError, match="different authorization"):
        service._authenticate_completion_certificate({**payload, "delivery_id": new_delivery_id})
    assert len(backend.calls) == calls
    assert (project / "answer.bin").read_bytes() == (project / "proposed.bin").read_bytes()
    assert service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")["certificate_id"] == result["certificate_id"]
    assert len(backend.calls) == calls


@pytest.mark.parametrize("which", ["authorization_certificate_id", "authorization_full_certificate_id", "certificate_id"])
@pytest.mark.parametrize("restart", [False, True])
def test_tampering_either_certificate_rejects_receipt_even_after_restart(tmp_path, which, restart):
    service, backend, project, _, candidate, first = delivered(tmp_path)
    tamper(service, first[which])
    if restart:
        service, backend, _, _ = service_fixture(tmp_path, backend=mixed_backend())
    calls = len(backend.calls)
    repeat = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert repeat["materialized"] is False and "error" in repeat
    assert len(backend.calls) == calls and (project / "answer.bin").exists()


def test_authentication_requires_designated_full_link(tmp_path):
    service, _, _, _, _, result = delivered(tmp_path)
    with pytest.raises(ValueError, match="designated full certificate"):
        service._authenticate_candidate_certificate(result["authorization_certificate_id"], result["authorization_root_id"])


@pytest.mark.parametrize("scope", ["authorization", "completion"])
def test_current_evidence_append_cannot_be_hidden_by_the_retained_certificate(tmp_path, scope):
    service, backend, _, _, candidate, first = delivered(tmp_path)
    subject = authorization_root(service, first) if scope == "authorization" else next(
        entry for entry in completion(service, first)["slice"] if entry["id"] == first["final_claim_id"])
    service.store.record_evidence(subject["obligations"][0]["id"], {"synthetic_new_evidence": True})
    calls = len(backend.calls)
    repeat = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert repeat["materialized"] is False and "error" in repeat
    assert len(backend.calls) == calls


@pytest.mark.parametrize("scope", ["authorization", "completion"])
def test_current_obligation_status_change_invalidates_both_scopes(tmp_path, scope):
    service, backend, _, _, candidate, first = delivered(tmp_path)
    subject = authorization_root(service, first) if scope == "authorization" else next(
        entry for entry in completion(service, first)["slice"] if entry["id"] == first["final_claim_id"])
    service.store.db.execute("UPDATE claims SET status='stale' WHERE id=?", (subject["obligations"][0]["id"],))
    service.store.db.commit()
    calls = len(backend.calls)
    repeat = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert repeat["materialized"] is False and "error" in repeat
    assert len(backend.calls) == calls


def test_new_host_observation_cannot_reuse_two_scope_authorization(tmp_path):
    spec = EvaluatorSpec((os.path.realpath(sys.executable), "-I", "-c", "print('synthetic observation')", "{candidate}"))
    service, backend, _, _, candidate, _ = delivered(tmp_path, evaluator_spec=spec)
    assert "artifact_id" in service.cyber_evaluate_candidate(candidate["candidate_id"])
    calls = len(backend.calls)
    repeat = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert repeat["materialized"] is False and "error" in repeat
    assert len(backend.calls) == calls


def test_wrong_destination_does_not_rebind_a_certified_candidate(tmp_path):
    service, backend, project, _ = service_fixture(tmp_path, backend=mixed_backend(), final_destination="answer.bin")
    candidate = registered(service)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "different.bin")
    assert result["materialized"] is False and "error" in result
    assert backend.calls == [] and not (project / "different.bin").exists()


@pytest.mark.parametrize("explicit_retry", [False, True])
def test_materialization_fault_reuses_completed_adjudication_on_retry(tmp_path, monkeypatch, explicit_retry):
    service, backend, project, _ = service_fixture(tmp_path, backend=mixed_backend())
    candidate = registered(service)
    original = service.candidates.materialize

    def unavailable(lease, address):
        raise OSError("synthetic exact destination temporarily unavailable")

    monkeypatch.setattr(service.candidates, "materialize", unavailable)
    first = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert first["materialized"] is False and "error" in first
    assert first["phase"] == "candidate_authorization" and first["task_completed"] is False
    calls = len(backend.calls)
    monkeypatch.setattr(service.candidates, "materialize", original)
    repeat = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin", retry=explicit_retry)
    assert repeat["materialized"] is True
    assert repeat["authorization_root_id"] == first["authorization_root_id"]
    assert repeat["authorization_certificate_id"] == first["certificate_id"]
    assert "final_claim_id" not in first
    assert len(backend.calls) == calls + 8 and (project / "answer.bin").exists()
    assert repeat["task_completed"] is True


@pytest.mark.parametrize("scope", ["authorization", "completion"])
def test_mutated_nested_executor_snapshot_invalidates_acceptance_receipt(tmp_path, scope):
    service, backend, _, _, candidate, first = delivered(tmp_path)
    if scope == "authorization":
        acceptance = service.artifacts.get(first["authorization_certificate_id"])
        artifact_id = acceptance["must_evidence"][0]["artifact_id"]
    else:
        final = next(entry for entry in completion(service, first)["slice"]
                     if entry["id"] == first["final_claim_id"])
        artifact_id = final["obligations"][0]["evidence"][-1]["artifact"]
    binding_game = service.artifacts.get(artifact_id)
    executor_id = binding_game["bound"]["discharge_artifact"]
    tamper(service, executor_id)
    calls = len(backend.calls)
    repeat = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert repeat["materialized"] is False and "error" in repeat
    assert len(backend.calls) == calls


@pytest.mark.parametrize("restart", [False, True])
def test_receipt_artifact_drift_cannot_select_completed_authorization(tmp_path, restart):
    service, backend, project, _, candidate, first = delivered(tmp_path)
    if restart:
        service, backend, _, _ = service_fixture(tmp_path, backend=mixed_backend())
    receipt_id = service._cyber_bindings[first["final_claim_id"]]["receipt_artifact_id"]
    tamper(service, receipt_id)
    calls = len(backend.calls)
    repeat = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert repeat["materialized"] is False and repeat["task_completed"] is False
    assert "error" in repeat and len(backend.calls) == calls
    assert (project / "answer.bin").exists()


def test_graph_change_during_materialization_cannot_append_delivery_authorization(tmp_path, monkeypatch):
    service, _, project, _ = service_fixture(tmp_path, backend=mixed_backend())
    candidate = registered(service)
    original = service.candidates.materialize

    def mutate_graph_after_write(lease, address):
        receipt = original(lease, address)
        final = next(reversed(service._cyber_bindings))
        node_id = service.store.obligation_children(final)[0]["id"]
        service.store.record_evidence(node_id, {"synthetic_changed_during_delivery": True})
        return receipt

    monkeypatch.setattr(service.candidates, "materialize", mutate_graph_after_write)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result["materialized"] is False and "error" in result
    assert not any(event["type"] == "cyber_materialized" for event in service.log.replay())
    assert not (project / "answer.bin").exists()


def test_candidate_identity_drift_at_lease_exit_cannot_leave_authorized_output(tmp_path, monkeypatch):
    service, _, project, _ = service_fixture(tmp_path, backend=mixed_backend())
    candidate = registered(service)
    original = service.candidates.materialize

    def change_identity_after_write(lease, address):
        receipt = original(lease, address)
        info = os.stat(lease.candidate["blob_path"])
        os.utime(lease.candidate["blob_path"], ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
        return receipt

    monkeypatch.setattr(service.candidates, "materialize", change_identity_after_write)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result["materialized"] is False and "error" in result
    assert not any(event["type"] == "cyber_materialized" for event in service.log.replay())
    assert not (project / "answer.bin").exists()


def test_receipt_event_failure_rolls_back_and_retries_delivery_only(tmp_path, monkeypatch):
    service, backend, project, _ = service_fixture(tmp_path, backend=mixed_backend())
    candidate = registered(service)
    original = service.log.append

    def unavailable(kind, payload):
        if kind == "cyber_materialized":
            raise OSError("synthetic durable receipt write failure")
        return original(kind, payload)

    monkeypatch.setattr(service.log, "append", unavailable)
    first = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert first["materialized"] is False and "error" in first
    assert not (project / "answer.bin").exists()
    assert not any(event["type"] == "cyber_materialized" for event in service.log.replay())
    first_delivery_id = service._cyber_bindings[first["final_claim_id"]]["delivery_id"]
    calls = len(backend.calls)
    monkeypatch.setattr(service.log, "append", original)
    second = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin", retry=True)
    assert second["materialized"] is True and second["task_completed"] is True
    assert second["final_claim_id"] != first["final_claim_id"]
    assert second["authorization_root_id"] == first["authorization_root_id"]
    assert second["authorization_certificate_id"] == first["authorization_certificate_id"]
    assert second["authorization_full_certificate_id"] == first["authorization_full_certificate_id"]
    assert service._cyber_bindings[second["final_claim_id"]]["delivery_id"] != first_delivery_id
    assert len(backend.calls) == calls + 8


def test_replaced_canonical_file_is_not_authorized_or_deleted_by_rollback(tmp_path, monkeypatch):
    service, _, project, _ = service_fixture(tmp_path, backend=mixed_backend())
    candidate = registered(service)
    original = service.candidates.materialize
    replacement = b"synthetic unrelated replacement"

    def replace_after_write(lease, address):
        receipt = original(lease, address)
        Path(address).unlink()  # Only the freshly created fixture output.
        Path(address).write_bytes(replacement)
        return receipt

    monkeypatch.setattr(service.candidates, "materialize", replace_after_write)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result["materialized"] is False and "error" in result
    assert not any(event["type"] == "cyber_materialized" for event in service.log.replay())
    assert (project / "answer.bin").read_bytes() == replacement


@pytest.mark.parametrize("restart", [False, True])
def test_changed_host_capability_revision_cannot_reauthorize_old_scope(tmp_path, restart):
    service, _, _, _, candidate, first = delivered(tmp_path)
    previous = service._cyber_bindings[first["authorization_root_id"]]["capability_revision"]
    if restart:
        service, _, _, _ = service_fixture(tmp_path, backend=mixed_backend())
    service._seat_capabilities = SeatCapabilities(read_retained_artifacts=True)
    assert service._evidence_manifest("obligations")["capability_revision"] != previous
    with pytest.raises(ValueError, match="capabilit"):
        service._authenticate_candidate_certificate(first["authorization_certificate_id"], first["authorization_root_id"],
                                                    first["authorization_full_certificate_id"])


@pytest.mark.parametrize("restart", [False, True])
def test_sidecar_current_receipt_authenticates_both_scopes_without_roles(tmp_path, restart):
    pytest.importorskip("tomllib")
    from experiments.cybergym.sidecar import delivery_receipt

    service, backend, _, _ = service_fixture(tmp_path, backend=mixed_backend(), final_destination="answer.bin")
    candidate = registered(service)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result["materialized"] is True
    if restart:
        service, backend, _, _ = service_fixture(tmp_path, backend=mixed_backend(), final_destination="answer.bin")
    calls = len(backend.calls)
    receipt = delivery_receipt(service)
    assert receipt["certificate_id"] == result["certificate_id"]
    assert receipt["final_claim_id"] == result["final_claim_id"]
    assert receipt["authorization_certificate_id"] == result["authorization_certificate_id"]
    assert receipt["authorization_full_certificate_id"] == result["authorization_full_certificate_id"]
    assert receipt["authorization_root_id"] == result["authorization_root_id"]
    assert receipt["candidate_id"] == candidate["candidate_id"]
    assert len(backend.calls) == calls
