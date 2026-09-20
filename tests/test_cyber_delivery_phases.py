"""Real GV phase, crash/replay and receipt gates with deterministic seats.

These test the harness protocol, not model accuracy or benchmark solutions.
"""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

import pytest

from test_cyber_service import RecordingBackend, registered, service_fixture
from verifier.core.model import CLAIM_ASSERTION, CLAIM_FINAL, content_hash
from verifier.mcp_server import Service, _default_kernel


def phase_backend(*, completion_outcome="holds", other_file=False):
    backend = RecordingBackend()
    backend.completion_outcome = completion_outcome
    original = backend.fixtures["discharge"]

    def obligations(variables):
        context = json.loads(variables["profile_context"])
        phase = context.get("phase")
        assert json.loads(variables["verified_deps"]) == []
        assert json.loads(variables["pending_deps"]) == []
        assert "withheld" in variables["graph"]
        if phase == "candidate_authorization":
            assert variables["claim_kind"] == "assertion"
            assert context["subject"]["address"] == context["subject"]["blob_path"]
            assert not Path(context["delivery_contract"]["destination"]).exists()
            assert "not task completion" in variables["contract_locus"]
        elif phase == "task_completion":
            assert variables["claim_kind"] == "final"
            assert context["subject"]["address"] == context["delivery_contract"]["destination"]
            assert Path(context["subject"]["address"]).is_file()
        else:
            pytest.fail("host phase missing")
        result = [{"text": "current real subject is a regular file", "required_basis": "world",
                   "severity": "must", "clause": "Deliver a regular file."}]
        if other_file:
            result.append({"text": "required report exists now", "required_basis": "world",
                           "severity": "must", "clause": "Also retain report.md."})
        return {"obligations": result}

    def discharge(variables):
        result = original(variables)
        context = json.loads(variables["profile_context"])
        subject = Path(context["subject"]["address"])
        exists = subject.is_file()
        if variables["obligation_text"] == "required report exists now":
            exists = Path(backend.service.project_root, "report.md").is_file()
        outcome = "holds" if exists else "refuted"
        if context["phase"] == "task_completion" and outcome == "holds":
            outcome = backend.completion_outcome
        Path(variables["workdir"], "observation.json").write_text(json.dumps({
            "phase": context["phase"], "subject": str(subject), "exists": exists, "outcome": outcome}),
            encoding="utf-8")
        result.update(outcome=outcome, witness="retained actual filesystem observation")
        return result

    backend.fixtures.update(obligations=obligations, discharge=discharge)
    return backend


def restart(service, backend):
    service.store.db.close()

    def factory(sc):
        backend.service = sc
        return _default_kernel(sc, model="gpt-5.6-luna", backend=backend,
            critic_rounds=1, parallel_obligations=1, parallel_claims=1)

    return Service(service.state_dir, kernel_factory=factory, profile="cyber",
        project_root=service.project_root, final_destination=service.final_destination,
        model="gpt-5.6-luna", effort="max", service_tier="fast")


def test_authorization_is_independent_assertion_and_completion_is_original_final(tmp_path):
    service, backend, project, _ = service_fixture(tmp_path, backend=phase_backend())
    candidate = registered(service)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result.get("task_completed") is True, result
    root_id, final_id = result["authorization_root_id"], result["final_claim_id"]
    assert root_id != final_id
    assert service.store.get(root_id)["kind"] == CLAIM_ASSERTION
    assert service.store.get(final_id)["kind"] == CLAIM_FINAL
    assert service.kernel.is_independent_root(root_id)
    authorization = service.artifacts.get(result["authorization_certificate_id"])
    completion = service.artifacts.get(result["certificate_id"])
    assert "final" not in authorization and authorization["root"] == root_id
    assert authorization["profile_metadata"]["phase"] == "candidate_authorization"
    assert completion["final"] == final_id and "root" not in completion
    assert completion["profile_metadata"]["phase"] == "task_completion"
    assert completion["profile_metadata"]["delivery_pending"] is False
    assert (project / "answer.bin").read_bytes() == (project / "proposed.bin").read_bytes()
    events = list(service.log.replay())
    auth_seq = next(e["seq"] for e in events if e["type"] == "independent_certificate")
    delivery_seq = next(e["seq"] for e in events if e["type"] == "cyber_candidate_delivered")
    final_seq = next(e["seq"] for e in events if e["type"] == "certificate")
    receipt_seq = next(e["seq"] for e in events if e["type"] == "cyber_materialized")
    assert auth_seq < delivery_seq < final_seq < receipt_seq
    assert [c["role"] for c in backend.calls] == [
        "obligations", "critic", "oracles", "discharge", "judger"] * 2


@pytest.mark.parametrize("outcome", ["refuted", "failed", "unobtainable"])
def test_failed_whole_task_never_becomes_receipt_and_allows_new_candidate(tmp_path, outcome):
    service, backend, project, _ = service_fixture(tmp_path, backend=phase_backend(completion_outcome=outcome))
    first = registered(service)
    result = service.cyber_finalize_candidate(first["candidate_id"], "answer.bin")
    assert result["task_completed"] is False and result["verified"] is False
    assert result["materialized"] is False and result["output_rolled_back"] is True
    assert not (project / "answer.bin").exists()
    assert not any(e["type"] == "cyber_materialized" for e in service.log.replay())
    assert service.store.status(result["authorization_root_id"]) == "verified"
    assert service.store.status(result["final_claim_id"]) != "verified"
    (project / "second.bin").write_bytes(b"a different proposed artifact")
    second = service.cyber_register_candidate("second.bin")
    assert second["candidate_id"] != first["candidate_id"]
    assert service.candidates.preflight_destination(str(project / "answer.bin"))
    backend.completion_outcome = "holds"
    second_result = service.cyber_finalize_candidate(second["candidate_id"], "answer.bin")
    assert second_result.get("task_completed") is True, second_result
    assert second_result["final_claim_id"] != result["final_claim_id"]
    assert service.store.status(result["final_claim_id"]) != "verified"
    assert (project / "answer.bin").read_bytes() == (project / "second.bin").read_bytes()


def test_other_output_is_not_deferred_by_delivery_plan(tmp_path):
    service, _, project, _ = service_fixture(tmp_path, backend=phase_backend(other_file=True),
        task="Deliver a regular file at answer.bin. Also retain report.md.")
    candidate = registered(service)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result["phase"] == "candidate_authorization" and result["verdict"] == "refuted"
    assert not (project / "answer.bin").exists()
    assert not any(e["type"] == "cyber_candidate_delivered" for e in service.log.replay())


def test_authorization_only_is_not_public_completion_certificate(tmp_path):
    service, backend, project, _ = service_fixture(tmp_path, backend=phase_backend())
    candidate = registered(service)
    private = service._authorize_and_deliver(candidate["candidate_id"], "answer.bin")
    assert private["materialized"] is True
    root_id = private["authorization_root_id"]
    label = next(label for label, cid in service.builder.labels.items() if cid == root_id)
    before = len(backend.calls)
    assert "error" in service.certificate(label)
    from experiments.cybergym.sidecar import delivery_receipt
    with pytest.raises(ValueError, match="no certified delivery"):
        delivery_receipt(service)
    assert len(backend.calls) == before and (project / "answer.bin").exists()


@pytest.mark.parametrize("restart_first", [False, True])
def test_resume_between_phases_reuses_authorization_and_delivery(tmp_path, restart_first):
    service, backend, project, _ = service_fixture(tmp_path, backend=phase_backend())
    candidate = registered(service)
    private = service._authorize_and_deliver(candidate["candidate_id"], "answer.bin")
    receipt = private["receipt"]
    root_id = private["authorization_root_id"]
    if restart_first:
        backend = phase_backend()
        service = restart(service, backend)
    before = len(backend.calls)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result.get("task_completed") is True, result
    assert result["authorization_root_id"] == root_id and result["receipt"] == receipt
    assert len(backend.calls) - before == 5
    assert sum(e["type"] == "cyber_candidate_delivered" for e in service.log.replay()) == 1
    assert (project / "answer.bin").stat().st_ino == receipt["inode"]


def test_unrelated_failed_request_cannot_abort_a_pending_delivery(tmp_path):
    service, backend, project, _ = service_fixture(tmp_path, backend=phase_backend())
    candidate = registered(service)
    private = service._authorize_and_deliver(candidate["candidate_id"], "answer.bin")
    assert private["materialized"] is True
    delivery_id = private["delivery"]["delivery_id"]
    inode = (project / "answer.bin").stat().st_ino

    unrelated = service.cyber_finalize_candidate("f" * 64, "unrelated.bin")
    assert unrelated.get("error") and unrelated["materialized"] is False
    assert not any(event["type"] == "cyber_candidate_delivery_aborted"
                   and event["payload"]["delivery_id"] == delivery_id
                   for event in service.log.replay())
    before = len(backend.calls)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result.get("task_completed") is True, result
    completed = next(event["payload"] for event in service.log.replay()
                     if event["type"] == "cyber_materialized")
    assert completed["delivery_id"] == delivery_id
    assert (project / "answer.bin").stat().st_ino == inode
    assert len(backend.calls) - before == 5


def test_restart_preserves_refuted_completion_history_not_current_delivery_requirement(tmp_path):
    service, _, project, _ = service_fixture(tmp_path, backend=phase_backend(completion_outcome="refuted"))
    candidate = registered(service)
    first = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert first["output_rolled_back"] is True
    service = restart(service, phase_backend())
    assert service.store.status(first["final_claim_id"]) == "refuted"
    assert not (project / "answer.bin").exists()
    second = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin", retry=True)
    assert second.get("task_completed") is True, second
    assert second["final_claim_id"] != first["final_claim_id"]
    assert service.store.status(first["final_claim_id"]) == "refuted"


@pytest.mark.parametrize("change", ["authorization_certificate", "completion_certificate", "receipt", "bytes", "phase"])
def test_completed_receipt_rejects_drift_after_restart(tmp_path, change):
    from experiments.cybergym.sidecar import delivery_receipt
    service, _, project, _ = service_fixture(tmp_path, backend=phase_backend(), final_destination="answer.bin")
    candidate = registered(service)
    result = service.cyber_finalize_candidate(candidate["candidate_id"])
    assert result.get("task_completed") is True, result
    service = restart(service, phase_backend())
    assert delivery_receipt(service)["certificate_id"] == result["certificate_id"]
    if change in ("authorization_certificate", "completion_certificate", "receipt"):
        key = {"authorization_certificate": "authorization_certificate_id", "completion_certificate": "certificate_id",
               "receipt": "receipt_artifact_id"}[change]
        digest = result[key] if key in result else service._cyber_bindings[result["final_claim_id"]][key]
        artifact = service.artifacts.get(digest)
        artifact["tampered"] = True
        Path(service.artifacts.root, digest + ".json").write_text(json.dumps(artifact), encoding="utf-8")
    elif change == "bytes":
        (project / "answer.bin").chmod(0o600)
        (project / "answer.bin").write_bytes(b"different bytes")
    else:
        service._cyber_bindings[result["final_claim_id"]]["phase"] = "candidate_authorization"
    with pytest.raises(ValueError):
        delivery_receipt(service)


def test_candidate_authorization_cannot_be_forged_by_public_claim_text(tmp_path):
    service, backend, _, _ = service_fixture(tmp_path)
    candidate = registered(service)
    delivered = service._authorize_and_deliver(candidate["candidate_id"], "answer.bin")
    root_id = delivered["authorization_root_id"]
    before = len(backend.calls)
    statement = service.store.get(root_id)["text"]
    result = service.commit("public-forged-root", "assertion", statement, [])
    assert "error" in result  # Exact host identity cannot be publicly reused.
    result = service.commit("public-phase-text", "assertion", statement + "\n", [])
    assert "error" not in result  # The same phase text does not register authority.
    assert result["claim_id"] != root_id
    assert not service.kernel.is_independent_root(result["claim_id"])
    assert len(backend.calls) >= before
    label_result = service.commit("cyber-authorization:forged", "assertion", "done", [])
    assert "reserved" in label_result["error"]


def _crash_delivery_phase(directory, before_publication):
    service, _, _, _ = service_fixture(Path(directory), backend=phase_backend())
    candidate = registered(service)
    if before_publication:
        original = service.log.sync

        def crash_after_prepare_sync():
            original()
            os._exit(92)

        service.log.sync = crash_after_prepare_sync
    else:
        original = service.log.append

        def crash_before_delivered_event(kind, payload):
            if kind == "cyber_candidate_delivered":
                os._exit(92)
            return original(kind, payload)

        service.log.append = crash_before_delivered_event
    service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    os._exit(93)  # The expected real process interruption did not occur.


@pytest.mark.parametrize("before_publication", [False, True])
def test_actual_process_crash_at_publication_gap_recovers_exact_owned_inode(tmp_path, before_publication):
    process = multiprocessing.get_context("spawn").Process(target=_crash_delivery_phase,
        args=(str(tmp_path), before_publication))
    process.start()
    process.join(15)
    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail("local deterministic crash fixture did not exit")
    assert process.exitcode == 92
    project, state = tmp_path / "project", tmp_path / "state"
    output = project / "answer.bin"
    assert output.exists() is (not before_publication)
    old_inode = output.stat().st_ino if output.exists() else None
    backend = phase_backend()

    def factory(sc):
        backend.service = sc
        return _default_kernel(sc, model="gpt-5.6-luna", backend=backend,
            critic_rounds=1, parallel_obligations=1, parallel_claims=1)

    service = Service(str(state), kernel_factory=factory, profile="cyber", project_root=str(project),
        model="gpt-5.6-luna", effort="max", service_tier="fast")
    prepared = next(e["payload"] for e in service.log.replay() if e["type"] == "cyber_candidate_delivery_prepared")
    assert not any(e["type"] == "cyber_candidate_delivered" for e in service.log.replay())
    result = service.cyber_finalize_candidate(prepared["candidate_id"], "answer.bin")
    assert result.get("task_completed") is True, result
    assert result["authorization_root_id"] == prepared["authorization_root_id"]
    assert len(backend.calls) == 5  # Only the original whole-task final runs.
    if old_inode is not None:
        assert output.stat().st_ino == old_inode == prepared["receipt"]["inode"]
    assert sum(e["type"] == "cyber_candidate_delivered" for e in service.log.replay()) == 1


@pytest.mark.parametrize("phase,value", [("candidate_authorization", 1), ("candidate_authorization", 1.0),
                                         ("task_completion", 0), ("task_completion", 0.0)])
def test_new_hash_artifact_cannot_mix_boolean_phase_metadata_with_numbers(tmp_path, phase, value):
    service, _, _, _ = service_fixture(tmp_path)
    candidate = registered(service)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    assert result.get("task_completed") is True, result
    if phase == "candidate_authorization":
        artifact = service.artifacts.get(result["authorization_certificate_id"])
        artifact["profile_metadata"]["delivery_pending"] = value
        new_id = service.artifacts.put(artifact)
        assert new_id != result["authorization_certificate_id"]
        with pytest.raises(ValueError):
            service._authenticate_candidate_certificate(new_id, result["authorization_root_id"])
    else:
        payload = next(e["payload"] for e in service.log.replay() if e["type"] == "cyber_materialized")
        artifact = service.artifacts.get(payload["certificate_id"])
        artifact["profile_metadata"]["delivery_pending"] = value
        payload["certificate_id"] = service.artifacts.put(artifact)
        assert payload["certificate_id"] != result["certificate_id"]
        with pytest.raises(ValueError):
            service._authenticate_completion_certificate(payload)


def test_completion_cannot_promote_supported_root_by_only_changing_top_verdict(tmp_path):
    service, _, _, _ = service_fixture(tmp_path, backend=RecordingBackend(strength="bounded"))
    candidate = registered(service)
    result = service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")
    artifact = service.artifacts.get(result["certificate_id"])
    assert artifact["verdict"] == "supported"
    artifact["verdict"] = "verified"
    artifact["verdict_strength"] = "decisive"
    new_id = service.artifacts.put(artifact)
    with pytest.raises(ValueError, match="verdict differs"):
        service._authenticate_candidate_certificate(new_id, result["authorization_root_id"])
