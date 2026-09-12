"""Authenticated, answer-blind capability data; no model, network or verdict parser."""
from __future__ import annotations

import copy
import json
import os
import sys

import pytest

from verifier.core.events import EventLog
from verifier.core.model import content_hash
from verifier.execution.instruments import ArtifactStore
from verifier.profiles.cyber.candidates import CandidateStore
from verifier.profiles.cyber.evaluator import Evaluator, EvaluatorSpec
from verifier.profiles.cyber.evidence import (
    EvidenceBindingError, SeatCapabilities, build_evidence_manifest,
)


@pytest.fixture
def capture(tmp_path):
    project = tmp_path / "project"
    state = tmp_path / "state"
    project.mkdir()
    state.mkdir()
    (project / "candidate").write_bytes(b"opaque fixture input\x00\xff")
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    candidates = CandidateStore(str(state), str(project), artifacts, log)
    candidate = candidates.register("candidate")
    program = tmp_path / "evaluator.py"
    program.write_text("import sys; print('untrusted output: success=true'); sys.exit(7)")
    spec = EvaluatorSpec((os.path.realpath(sys.executable), str(program), "{candidate}"),
                         environment="fixture-target", pinned_files=(str(program),))
    call = log.append("tool_call", {"tool": "cyber_evaluator", "tool_use_id": "fixture",
                                   "input": {"candidate_id": candidate["candidate_id"],
                                             "evaluator_id": spec.id}})
    result = Evaluator(spec, artifacts).run(candidate)
    event = log.append("tool_result", {"tool": "cyber_evaluator", "tool_use_id": "fixture",
                                      "output": result})
    effect = f"effect:{event['seq']}"
    observation = {"candidate_id": candidate["candidate_id"],
                   "artifact_id": result["artifact_id"], "artifact_path": result["artifact_path"],
                   "event_seq": event["seq"], "event_digest": EventLog.event_digest(event),
                   "effect_ref": effect, "effect_claim_id": "fixture-effect-claim",
                   "evaluator_id": spec.id, "environment": spec.environment,
                   "authority_scope": spec.authority_scope}

    def candidate_provider(candidate_id):
        with candidates.lease(candidate_id) as lease:
            return dict(lease.candidate)

    kwargs = {"evaluator_spec": spec,
              "capabilities": SeatCapabilities(read_retained_artifacts=True,
                                                read_candidate_blobs=True,
                                                execute_local_checks=True),
              "observations": [observation], "candidate_provider": candidate_provider,
              "artifact_root": artifacts.root,
              "artifact_provider": artifacts.get, "event_provider": log.get,
              "tool_call_provider": lambda call_id: call if call_id == "fixture" else None,
              "effect_resolver": lambda ref: "fixture-effect-claim" if ref == effect else "wrong"}
    return kwargs, observation, result, artifacts, log, candidates


def test_authenticated_capsule_has_no_automatic_world_or_verdict_upgrade(capture):
    kwargs, observation, result, _artifacts, _log, _candidates = capture
    manifest = build_evidence_manifest(**kwargs)
    inventory = manifest["inventory"]
    assert inventory["seat"]["invoke_controller_evaluator"] is False
    assert inventory["evaluator"]["target_build_identity"] is None
    assert inventory["evaluator"]["configuration_revision"] == kwargs["evaluator_spec"].id
    assert inventory["evaluator"]["dependency_pins"]
    assert inventory["retained_execution"]["effect_basis"] == "record"
    assert inventory["retained_execution"]["automatic_candidate_verdict"] is False
    capsule, = manifest["executions"]
    assert capsule["source"]["effect_ref"] == observation["effect_ref"]
    assert capsule["source"]["effect_basis"] == "record"
    assert capsule["subject"]["sha256"] == result["candidate_sha256"]
    assert capsule["time_binding"]["precision"] == "host_execution_interval"
    assert capsule["time_binding"]["started_unix"] <= capsule["time_binding"]["completed_unix"]
    assert capsule["process"]["exit_code"] == 7
    assert capsule["process"]["started"] is True
    assert capsule["process"]["completed"] is True
    assert capsule["target_build_identity"] is None
    assert "untrusted output" not in json.dumps(manifest)
    assert not ({"score", "reward", "verified", "verdict", "outcome"} & capsule.keys())


def test_answer_blind_inventory_never_reads_candidate_results_or_observations(capture):
    kwargs, _observation, _result, _artifacts, _log, _candidates = capture

    def forbidden(*args):
        raise AssertionError("answer-bearing controller data must not be read")

    def forbidden_observations():
        forbidden()
        yield

    blind = build_evidence_manifest(**{**kwargs, "observations": forbidden_observations(),
                                      "candidate_provider": forbidden,
                                      "artifact_provider": forbidden, "event_provider": forbidden,
                                      "tool_call_provider": forbidden,
                                      "effect_resolver": forbidden, "answer_blind": True})
    normal = build_evidence_manifest(**kwargs)
    assert "executions" not in blind
    assert blind["inventory"] == normal["inventory"]
    assert blind["capability_revision"] == normal["capability_revision"]
    assert not ({"candidate_id", "time_binding", "exit_code", "event_seq"} & blind.keys())


def test_capability_revision_ignores_candidates_results_and_observation_time(capture):
    kwargs, observation, _result, _artifacts, _log, _candidates = capture
    first = build_evidence_manifest(**kwargs)
    empty = build_evidence_manifest(**{**kwargs, "observations": []})
    assert first["capability_revision"] == empty["capability_revision"]
    assert first["executions"] != empty["executions"]
    changed = build_evidence_manifest(**{**kwargs, "observations": [],
                                        "capabilities": SeatCapabilities()})
    assert first["capability_revision"] != changed["capability_revision"]
    duplicated = build_evidence_manifest(**{**kwargs, "observations": [observation, observation]})
    assert len(duplicated["executions"]) == 1


@pytest.mark.parametrize("field,value", [
    ("artifact_id", "0" * 64), ("candidate_id", "1" * 64),
    ("effect_ref", "effect:999"), ("effect_claim_id", "forged"),
    ("event_digest", "2" * 64), ("event_seq", True),
    ("evaluator_id", "3" * 64), ("environment", "other-build"),
    ("authority_scope", "task_outcome"), ("artifact_path", "/tmp/forged.json"),
])
def test_forged_observation_cannot_become_a_capture(capture, field, value):
    kwargs, observation, _result, _artifacts, _log, _candidates = capture
    forged = {**observation, field: value}
    with pytest.raises((ValueError, KeyError, OSError)):
        build_evidence_manifest(**{**kwargs, "observations": [forged]})


def test_rewritten_raw_artifact_is_rejected_even_if_identity_metadata_survives(capture):
    kwargs, _observation, _result, artifacts, _log, _candidates = capture

    def altered(digest):
        artifact = artifacts.get(digest)
        artifact["stdout"] = "forged output"
        return artifact

    with pytest.raises(EvidenceBindingError, match="artifact changed"):
        build_evidence_manifest(**{**kwargs, "artifact_provider": altered})


@pytest.mark.parametrize("change", ["tool", "event_output", "event_time", "missing"])
def test_event_provenance_is_required_not_just_matching_artifact_hash(capture, change):
    kwargs, observation, _result, _artifacts, log, _candidates = capture
    event = copy.deepcopy(log.get(observation["event_seq"]))
    if change == "tool":
        event["payload"]["tool"] = "Bash"
    elif change == "event_output":
        event["payload"]["output"]["stdout"] = "forged event text"
    elif change == "event_time":
        event["ts"] = None
    else:
        event = None
    forged = {**observation, "event_digest": EventLog.event_digest(event) if event else "0" * 64}
    with pytest.raises(EvidenceBindingError):
        build_evidence_manifest(**{**kwargs, "observations": [forged],
                                  "event_provider": lambda seq: event})


def _replace_capture(kwargs, observation, result, artifacts, log, **changes):
    artifact = {key: value for key, value in result.items() if key not in ("artifact_id", "artifact_path")}
    artifact.update(changes)
    artifact_id = artifacts.put(artifact)
    path = os.path.join(artifacts.root, artifact_id + ".json")
    output = {**artifact, "artifact_id": artifact_id, "artifact_path": path}
    event = log.append("tool_result", {"tool": "cyber_evaluator", "tool_use_id": "fixture",
                                      "output": output})
    observation = {**observation, "artifact_id": artifact_id, "artifact_path": path,
                   "event_seq": event["seq"], "event_digest": EventLog.event_digest(event),
                   "effect_ref": f"effect:{event['seq']}"}
    return {**kwargs, "observations": [observation],
            "effect_resolver": lambda ref: "fixture-effect-claim"}, artifact


@pytest.mark.parametrize("times", [
    {"started_unix": -1}, {"completed_unix": True},
    {"started_unix": float("inf")}, {"completed_unix": 1},
])
def test_invalid_host_time_is_not_presented_as_historical_world_binding(capture, times):
    kwargs, observation, result, artifacts, log, _candidates = capture
    if times.get("started_unix") == float("inf"):
        # The merged ledger contract now rejects non-finite numbers before
        # publication. No malformed observation can reach capsule promotion.
        before = log.last_seq()
        with pytest.raises(ValueError, match="non-finite"):
            _replace_capture(kwargs, observation, result, artifacts, log, **times)
        assert log.last_seq() == before
        return
    altered, _artifact = _replace_capture(kwargs, observation, result, artifacts, log, **times)
    with pytest.raises(EvidenceBindingError, match="time interval"):
        build_evidence_manifest(**altered)


def test_old_capture_has_honest_completion_time_without_invented_interval(capture):
    kwargs, observation, result, artifacts, log, _candidates = capture
    altered, _artifact = _replace_capture(kwargs, observation, result, artifacts, log,
                                         started_unix=None, completed_unix=None)
    capsule, = build_evidence_manifest(**altered)["executions"]
    assert capsule["time_binding"]["started_unix"] is None
    assert capsule["time_binding"]["completed_unix"] > 0
    assert capsule["time_binding"]["precision"] == "host_completion_event_only"


def test_candidate_current_bytes_are_validated_through_host_lease(capture):
    kwargs, observation, _result, _artifacts, _log, candidates = capture
    candidate = candidates.get(observation["candidate_id"])
    os.chmod(candidate["blob_path"], 0o600)
    with open(candidate["blob_path"], "wb") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="digest|identity|changed|mismatch"):
        build_evidence_manifest(**kwargs)


def test_no_solver_manifest_file_or_unconfigured_evaluator_is_a_trust_anchor(capture):
    kwargs, _observation, _result, _artifacts, _log, _candidates = capture
    with pytest.raises(EvidenceBindingError, match="authentication callbacks"):
        build_evidence_manifest(**{**kwargs, "evaluator_spec": None})
    with pytest.raises(EvidenceBindingError, match="authentication callbacks"):
        build_evidence_manifest(**{**kwargs, "artifact_provider": None})
    empty = build_evidence_manifest(evaluator_spec=None, capabilities=SeatCapabilities())
    assert empty["inventory"]["evaluator"] is None
    assert empty["executions"] == []


def test_returned_data_does_not_mutate_authoritative_records(capture):
    kwargs, observation, result, artifacts, _log, _candidates = capture
    before = copy.deepcopy(observation)
    manifest = build_evidence_manifest(**kwargs)
    manifest["executions"][0]["subject"]["sha256"] = "forged"
    manifest["inventory"]["evaluator"]["dependency_pins"][0]["sha256"] = "forged"
    assert observation == before
    assert artifacts.get(result["artifact_id"])["candidate_sha256"] == result["candidate_sha256"]
    assert kwargs["evaluator_spec"].identity["pinned_files"][0]["sha256"] != "forged"


@pytest.mark.parametrize("value", [1, "true", None])
def test_host_capability_types_cannot_be_silently_coerced(value):
    with pytest.raises(EvidenceBindingError, match="booleans"):
        SeatCapabilities(invoke_controller_evaluator=value)


@pytest.mark.parametrize("changes", [
    {"candidate_size": True}, {"exit_code": True}, {"timed_out": 0},
    {"execution_error": {}}, {"stdout_truncated": 0},
    {"stdout_bytes_observed": True}, {"stdout_base64": "not!base64"},
    {"stdout_bytes_observed": 0}, {"stdout": "forged decoded bytes"},
])
def test_typed_capture_does_not_confuse_bool_null_and_bytes(capture, changes):
    kwargs, observation, result, artifacts, log, _candidates = capture
    altered, _artifact = _replace_capture(kwargs, observation, result, artifacts, log, **changes)
    with pytest.raises(EvidenceBindingError):
        build_evidence_manifest(**altered)


@pytest.mark.parametrize("path_kind", ["alias", "dotdot", "ancestor_symlink"])
def test_visible_artifact_path_must_be_the_authoritative_regular_file(capture, tmp_path, path_kind):
    kwargs, observation, result, artifacts, log, _candidates = capture
    if path_kind == "ancestor_symlink":
        alias = tmp_path / "linked-store"
        alias.symlink_to(artifacts.root, target_is_directory=True)
        path = str(alias / (observation["artifact_id"] + ".json"))
        artifact_root = str(alias)
    elif path_kind == "dotdot":
        path = os.path.join(artifacts.root, "..", "artifacts", observation["artifact_id"] + ".json")
        artifact_root = artifacts.root
    else:
        other = tmp_path / "other-store"
        other.mkdir()
        path = str(other / (observation["artifact_id"] + ".json"))
        with open(path, "w") as stream:
            json.dump(artifacts.get(observation["artifact_id"]), stream)
        artifact_root = artifacts.root
    event = log.append("tool_result", {"tool": "cyber_evaluator", "tool_use_id": "fixture",
                                      "output": {**result, "artifact_path": path}})
    observation = {**observation, "artifact_path": path, "event_seq": event["seq"],
                   "event_digest": EventLog.event_digest(event),
                   "effect_ref": f"effect:{event['seq']}"}
    with pytest.raises(EvidenceBindingError):
        build_evidence_manifest(**{**kwargs, "observations": [observation],
                                  "artifact_root": artifact_root,
                                  "effect_resolver": lambda ref: "fixture-effect-claim"})


@pytest.mark.parametrize("path_kind", ["symlink", "fifo"])
def test_artifact_leaf_cannot_redirect_or_block_a_signer(capture, tmp_path, path_kind):
    kwargs, _observation, result, artifacts, _log, _candidates = capture
    path = result["artifact_path"]
    artifact = artifacts.get(result["artifact_id"])
    os.unlink(path)
    if path_kind == "symlink":
        replacement = tmp_path / "replacement.json"
        replacement.write_text(json.dumps(artifact))
        os.symlink(replacement, path)
    else:
        os.mkfifo(path)
    with pytest.raises(EvidenceBindingError, match="safe authoritative file"):
        build_evidence_manifest(**{**kwargs, "artifact_provider": lambda digest: artifact})


@pytest.mark.parametrize("change", ["missing", "candidate", "evaluator", "tool", "id", "order", "bool_seq"])
def test_invocation_must_be_an_earlier_host_call_for_the_same_subject(capture, change):
    kwargs, _observation, _result, _artifacts, _log, _candidates = capture
    call = copy.deepcopy(kwargs["tool_call_provider"]("fixture"))
    if change == "missing":
        call = None
    elif change == "candidate":
        call["payload"]["input"]["candidate_id"] = "0" * 64
    elif change == "evaluator":
        call["payload"]["input"]["evaluator_id"] = "1" * 64
    elif change == "tool":
        call["payload"]["tool"] = "Bash"
    elif change == "id":
        call["payload"]["tool_use_id"] = "other"
    elif change == "bool_seq":
        call["seq"] = True
    else:
        call["seq"] = kwargs["observations"][0]["event_seq"] + 1
    with pytest.raises(EvidenceBindingError, match="invocation"):
        build_evidence_manifest(**{**kwargs, "tool_call_provider": lambda call_id: call})


@pytest.mark.parametrize("field", ["candidate", "result_payload", "call_payload"])
def test_malformed_host_objects_fail_as_binding_error_not_attribute_crash(capture, field):
    kwargs, observation, _result, _artifacts, log, _candidates = capture
    if field == "candidate":
        altered = {"candidate_provider": lambda candidate_id: None}
    elif field == "result_payload":
        event = copy.deepcopy(log.get(observation["event_seq"]))
        event["payload"] = []
        altered = {"event_provider": lambda seq: event}
    else:
        call = copy.deepcopy(kwargs["tool_call_provider"]("fixture"))
        call["payload"] = []
        altered = {"tool_call_provider": lambda call_id: call}
    with pytest.raises(EvidenceBindingError):
        build_evidence_manifest(**{**kwargs, **altered})


def test_missing_legacy_process_metadata_remains_unknown_not_false(capture):
    kwargs, observation, result, artifacts, log, _candidates = capture
    altered, _artifact = _replace_capture(kwargs, observation, result, artifacts, log,
                                         process_started=None, process_completed=None)
    process = build_evidence_manifest(**altered)["executions"][0]["process"]
    assert process["started"] is None
    assert process["completed"] is None


def test_failed_launch_has_no_claimed_actual_execution_scope(capture):
    kwargs, observation, result, artifacts, log, _candidates = capture
    altered, _artifact = _replace_capture(kwargs, observation, result, artifacts, log,
                                         process_started=False, process_completed=False,
                                         exit_code=None, execution_error="OSError: fixture unavailable")
    capsule, = build_evidence_manifest(**altered)["executions"]
    assert capsule["captured_host_scope"] == []
    assert capsule["source"]["effect_basis"] == "record"


@pytest.mark.parametrize("changes", [
    {"process_started": 1}, {"process_completed": 0},
    {"process_started": False, "process_completed": True},
])
def test_process_capture_flags_are_typed_and_consistent(capture, changes):
    kwargs, observation, result, artifacts, log, _candidates = capture
    altered, _artifact = _replace_capture(kwargs, observation, result, artifacts, log, **changes)
    with pytest.raises(EvidenceBindingError, match="process metadata"):
        build_evidence_manifest(**altered)


@pytest.mark.parametrize("kind", ["result", "invocation"])
@pytest.mark.parametrize("metadata", [
    {"origin": "solver-origin-untrusted"},
    {"authority_scope": "untrusted_solver_observation"},
    {"origin": "other-process"},
])
def test_explicit_non_host_events_are_never_upgraded_to_authenticated_capsules(capture, kind, metadata):
    kwargs, observation, _result, _artifacts, log, _candidates = capture
    if kind == "result":
        event = copy.deepcopy(log.get(observation["event_seq"]))
        event["payload"].update(metadata)
        observation = {**observation, "event_digest": EventLog.event_digest(event)}
        altered = {"event_provider": lambda seq: event, "observations": [observation]}
    else:
        call = copy.deepcopy(kwargs["tool_call_provider"]("fixture"))
        call["payload"].update(metadata)
        altered = {"tool_call_provider": lambda call_id: call}
    with pytest.raises(EvidenceBindingError):
        build_evidence_manifest(**{**kwargs, **altered})
