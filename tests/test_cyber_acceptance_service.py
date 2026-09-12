"""Real GV role/binding/certificate delivery using deterministic fixture seats.

No model, benchmark API, task solution or private evaluator is involved.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from test_cyber_service import JUDGE_YES, RecordingBackend, registered, service_fixture
from verifier.core.model import content_hash
from verifier.profiles.cyber.evaluator import EvaluatorSpec
from verifier.profiles.cyber.evidence import SeatCapabilities


def mixed_backend(*, advice_refuted=False, must_strength="decisive"):
    backend = RecordingBackend()
    backend.fixtures["obligations"] = {"obligations": [
        {"text": "synthetic required condition", "required_basis": "world", "severity": "must",
         "clause": "Establish the required condition."},
        {"text": "synthetic advisory condition", "required_basis": "world", "severity": "should",
         "clause": "Prefer the advisory condition."},
    ]}

    def oracles(variables):
        strength = ("decisive" if advice_refuted else "bounded") \
            if variables["obligation_text"] == "synthetic advisory condition" else must_strength
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
    assert result.get("materialized") is True, result
    return service, backend, project, state, candidate, result


def full(service, result):
    return service.artifacts.get(result["full_certificate_id"])


def tamper(service, artifact_id):
    artifact = service.artifacts.get(artifact_id)
    artifact["synthetic_tamper"] = "a fixture mutation"
    Path(service.artifacts.root, artifact_id + ".json").write_text(json.dumps(artifact), encoding="utf-8")


def test_holding_bounded_advice_gets_two_authenticated_scopes_after_all_roles(tmp_path):
    service, backend, project, _, candidate, result = delivered(tmp_path)
    assert result["full_verdict"] == full(service, result)["verdict"] == "supported"
    assert result["certificate_id"] != result["full_certificate_id"]
    acceptance = service.artifacts.get(result["certificate_id"])
    assert content_hash(acceptance) == result["certificate_id"]
    assert acceptance["full_certificate_id"] == result["full_certificate_id"]
    assert acceptance["verdict"] == result["verdict"] == "verified"
    assert acceptance["profile_metadata"]["binding"]["subject"]["candidate_id"] == candidate["candidate_id"]
    roles = [call["role"] for call in backend.calls]
    assert roles.count("obligations") == roles.count("critic") == 1
    assert roles.count("oracles") == roles.count("discharge") == roles.count("judger") == 2
    assert (project / "answer.bin").read_bytes() == (project / "proposed.bin").read_bytes()
    assert result["official_score"] is None
    before = len(backend.calls)
    assert service._authenticate_candidate_certificate(result["certificate_id"], result["final_claim_id"],
                                                     result["full_certificate_id"])[0] == acceptance
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


def test_two_certificate_repeat_is_idempotent_without_reverification(tmp_path):
    service, backend, _, _, candidate, first = delivered(tmp_path)
    calls = len(backend.calls)
    repeat = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert repeat["materialized"] is True
    assert repeat["certificate_id"] == first["certificate_id"]
    assert repeat["full_certificate_id"] == first["full_certificate_id"]
    assert len(backend.calls) == calls


@pytest.mark.parametrize("which", ["certificate_id", "full_certificate_id"])
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
        service._authenticate_candidate_certificate(result["certificate_id"], result["final_claim_id"])


def test_current_evidence_append_cannot_be_hidden_by_the_retained_certificate(tmp_path):
    service, backend, _, _, candidate, first = delivered(tmp_path)
    final = next(entry for entry in full(service, first)["slice"] if entry["id"] == first["final_claim_id"])
    service.store.record_evidence(final["obligations"][0]["id"], {"synthetic_new_evidence": True})
    calls = len(backend.calls)
    repeat = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert repeat["materialized"] is False and "error" in repeat
    assert len(backend.calls) == calls


def test_current_obligation_status_change_invalidates_both_scopes(tmp_path):
    service, backend, _, _, candidate, first = delivered(tmp_path)
    final = next(entry for entry in full(service, first)["slice"] if entry["id"] == first["final_claim_id"])
    service.store.db.execute("UPDATE claims SET status='stale' WHERE id=?", (final["obligations"][0]["id"],))
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
    calls = len(backend.calls)
    monkeypatch.setattr(service.candidates, "materialize", original)
    repeat = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin", retry=explicit_retry)
    assert repeat["materialized"] is True
    assert repeat["final_claim_id"] == first["final_claim_id"]
    assert len(backend.calls) == calls and (project / "answer.bin").exists()


def test_mutated_nested_executor_snapshot_invalidates_acceptance_receipt(tmp_path):
    service, backend, _, _, candidate, first = delivered(tmp_path)
    acceptance = service.artifacts.get(first["certificate_id"])
    binding_game = service.artifacts.get(acceptance["must_evidence"][0]["artifact_id"])
    executor_id = binding_game["bound"]["discharge_artifact"]
    tamper(service, executor_id)
    calls = len(backend.calls)
    repeat = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert repeat["materialized"] is False and "error" in repeat
    assert len(backend.calls) == calls


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
    calls = len(backend.calls)
    monkeypatch.setattr(service.log, "append", original)
    second = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin", retry=True)
    assert second["materialized"] is True and second["final_claim_id"] == first["final_claim_id"]
    assert len(backend.calls) == calls


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
    previous = service._cyber_bindings[first["final_claim_id"]]["capability_revision"]
    if restart:
        service, _, _, _ = service_fixture(tmp_path, backend=mixed_backend())
    service._seat_capabilities = SeatCapabilities(read_retained_artifacts=True)
    assert service._evidence_manifest("obligations")["capability_revision"] != previous
    with pytest.raises(ValueError, match="capabilit"):
        service._authenticate_candidate_certificate(first["certificate_id"], first["final_claim_id"],
                                                    first["full_certificate_id"])


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
    assert receipt["full_certificate_id"] == result["full_certificate_id"]
    assert receipt["candidate_id"] == candidate["candidate_id"]
    assert len(backend.calls) == calls
