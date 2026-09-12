"""Real Service evidence closure and host provenance, without any model call."""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

from verifier.core.store import TRIGGER_LAZY
from verifier.mcp_server import Service
from verifier.profiles.cyber.evaluator import EvaluatorSpec
from verifier.profiles.cyber.evidence import SeatCapabilities


@pytest.fixture
def service_captures(tmp_path):
    project = tmp_path / "project"
    state = tmp_path / "state"
    project.mkdir()
    state.mkdir()
    (state / "task.md").write_text("Establish the task-defined property of the selected subject.")
    (project / "first").write_bytes(b"opaque first fixture")
    (project / "second").write_bytes(b"opaque unrelated fixture")
    spec = EvaluatorSpec((os.path.realpath(sys.executable), "-I", "-c",
                          "print('opaque response, not a candidate verdict')", "{candidate}"))
    service = Service(str(state), kernel_factory=lambda sc: SimpleNamespace(),
                      trigger=TRIGGER_LAZY, profile="cyber", project_root=str(project),
                      evaluator_spec=spec, seat_capabilities=SeatCapabilities(
                          read_retained_artifacts=True, read_candidate_blobs=True,
                          execute_local_checks=True))
    observations = []
    for name in ("first", "second"):
        candidate = service.cyber_register_candidate(name)
        assert "error" not in candidate
        observation = service.cyber_evaluate_candidate(candidate["candidate_id"])
        assert "error" not in observation
        observations.append((candidate, observation))
    yield service, observations
    service.store.db.close()


def _claim(service, label, deps):
    result = service.commit(label, "assertion", "fixture claim with explicitly declared inputs", deps)
    assert "error" not in result
    return result["claim_id"]


def _manifest(service, role, claim_id):
    return json.loads(service.profile_context(role, claim_id))["execution_capabilities"]


def test_ordinary_claim_receives_only_real_transitive_effect_dependency_capture(service_captures):
    service, ((first, first_observation), (second, _second_observation)) = service_captures
    _claim(service, "parent", [first_observation["effect_ref"]])
    claim_id = _claim(service, "child", ["parent"])
    manifest = _manifest(service, "discharge", claim_id)
    capsule, = manifest["executions"]
    assert capsule["subject"]["candidate_id"] == first["candidate_id"]
    assert capsule["source"]["effect_claim_id"] == first_observation["effect_claim_id"]
    assert second["candidate_id"] not in json.dumps(manifest)
    assert manifest["inventory"]["seat"]["invoke_controller_evaluator"] is False
    # The lazy fixture has no kernel and therefore has not even settled its
    # ordinary Effect. Exposing a capsule must not promote or settle it.
    assert service.store.get(first_observation["effect_claim_id"])["basis"] != "world"
    assert capsule["source"]["effect_basis"] == "record"
    assert service.store.status(claim_id) == "unverified"


def test_claim_without_effect_dependencies_cannot_see_global_candidate_results(service_captures):
    service, observations = service_captures
    claim_id = _claim(service, "unrelated", [])
    manifest = _manifest(service, "discharge", claim_id)
    assert manifest["executions"] == []
    for candidate, observation in observations:
        assert candidate["candidate_id"] not in json.dumps(manifest)
        assert observation["artifact_id"] not in json.dumps(manifest)


@pytest.mark.parametrize("role", ["obligations", "critic", "oracles", "sufficiency"])
def test_answer_blind_service_inventory_excludes_execution_result_and_subject(service_captures, role):
    service, ((candidate, observation), _second) = service_captures
    claim_id = _claim(service, "parent", [observation["effect_ref"]])
    blind = _manifest(service, role, claim_id)
    executor = _manifest(service, "discharge", claim_id)
    assert "executions" not in blind
    assert blind["inventory"] == executor["inventory"]
    assert blind["capability_revision"] == executor["capability_revision"]
    assert candidate["candidate_id"] not in json.dumps(blind)
    assert observation["artifact_id"] not in json.dumps(blind)
    assert "exit_code" not in json.dumps(blind)


def test_solver_hook_cannot_mint_a_retained_host_observation(service_captures):
    service, ((_candidate, observation), _second) = service_captures
    service._ingest_hook_event({"type": "cyber_observation_recorded", "payload": {
        "candidate_id": "0" * 64, "artifact_id": "1" * 64,
        "effect_ref": observation["effect_ref"], "effect_claim_id": observation["effect_claim_id"]}})
    claim_id = _claim(service, "actual-parent", [observation["effect_ref"]])
    capsules = _manifest(service, "judger", claim_id)["executions"]
    assert len(capsules) == 1
    assert capsules[0]["subject"]["candidate_id"] != "0" * 64


def test_untrusted_duplicate_tool_name_and_call_id_cannot_poison_host_capture(service_captures):
    service, ((_candidate, observation), _second) = service_captures
    result_event = service.log.get(int(observation["effect_ref"].split(":")[1]))
    call_id = result_event["payload"]["tool_use_id"]
    # A solver can emit arbitrary tool-shaped hook input. It is not a host
    # evaluator call, even when it copies the name and observable call ID.
    service._ingest_hook_event({"type": "tool_call", "payload": {
        "tool": "cyber_evaluator", "tool_use_id": call_id,
        "input": {"candidate_id": "0" * 64, "evaluator_id": "1" * 64}}})
    claim_id = _claim(service, "real-parent", [observation["effect_ref"]])
    capsule, = _manifest(service, "discharge", claim_id)["executions"]
    assert capsule["source"]["effect_ref"] == observation["effect_ref"]
