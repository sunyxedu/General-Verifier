"""Real host command evidence through ordinary GV seats and both certificates.

Harmless localhost fixtures only: no model, benchmark, Docker, vulnerability,
or fabricated tool event. The deterministic discharge/Judger seats parse the
authenticated retained artifact, not a solver Bash string. This exercises the
ordinary agent-discharge route; it does not claim that record.program can
JSON-decode nested stdout, nor that fixture success measures model accuracy.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

import pytest

from test_cyber_service import RecordingBackend, registered, service_fixture
from test_cybergym_submission_execution import queue, response, submission
from experiments.cybergym import evaluator
from verifier.core.events import EventLog
from verifier.core.model import CLAIM_ASSERTION, CLAIM_EFFECT, CLAIM_FINAL, canonical, content_hash
from verifier.profiles.cyber.evaluator import EvaluatorSpec


RECORD = "a successful host-delegated task-command submission of these exact bytes is retained"
CONTENT = "the current phase subject has exactly the task-given harmless bytes"


class RecordGap(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise RecordGap(reason)


def trusted_host_origin(payload):
    require(payload.get("origin") == "verifier_controller"
            and payload.get("authority_scope") == "host_evaluator_execution",
            "untrusted_host_origin")


class SubmissionEvidenceBackend(RecordingBackend):
    """Two complete MUSTs; neither is inferred from the other's verdict."""

    def __init__(self, spec, configuration, script):
        super().__init__()
        self.spec = spec
        self.configuration = configuration
        self.script = script
        # Identity expectations are captured before execution from host pins,
        # not reread from the current world to reconstruct a historical act.
        self.pins = {item["path"]: item["sha256"] for item in spec.identity["pinned_files"]}
        self.audits = []
        self.judger_audits = []

        def obligations(variables):
            task = json.loads(json.loads(variables["task_context"])["text"])
            assert task["submission"] == RECORD
            assert task["destination"] == "answer.bin"
            assert json.loads(variables["profile_context"])["phase"] in (
                "candidate_authorization", "task_completion")
            return {"obligations": [
                {"text": RECORD, "required_basis": "record", "severity": "must", "clause": RECORD},
                {"text": CONTENT, "required_basis": "world", "severity": "must", "clause": CONTENT},
            ]}

        def oracles(variables):
            record = variables["obligation_text"] == RECORD
            return {"methods": [{
                "procedure": (
                    "Read the claim's frozen ledger and retained controller artifact; authenticate the "
                    "host call/result, evaluator/candidate and pins; parse its child execution and "
                    "check actual post-spawn completion, exact Bash argv/cwd, same bytes, no errors "
                    "and complete valid response. Missing history is unobtainable. Never rerun it."
                    if record else
                    "Read the actual phase subject bytes, retain their independent digest, then "
                    "compare only at the final comparator with the task-given exact bytes; wrong "
                    "content is a world-state counterexample. Do not use target exit as the verdict."),
                "artifact": "observation.json with the observed facts and source identities",
                "level": "independent fixture observation", "basis": "record" if record else "world",
                "strength": "decisive", "bridges": [],
            }]}

        def discharge(variables):
            observed = self.inspect(variables)
            self.audits.append(observed)
            Path(variables["workdir"], "observation.json").write_text(json.dumps(observed), encoding="utf-8")
            return {"outcome": observed["outcome"], "witness": json.dumps(observed),
                    "artifact": "observation.json", "assumed_premises": [],
                    "witness_kind": ("world_state" if observed["outcome"] == "refuted" else
                                     "absence" if observed["outcome"] == "unobtainable" else "execution"),
                    "inputs": self.inputs(variables)}

        def judger(variables):
            # Fresh second signer reopens the real executor snapshot and
            # independently re-reads the SAME frozen historical record.
            evidence = json.loads(variables["evidence"])
            artifact = json.loads(Path(evidence["artifact_path"]).read_text())
            assert content_hash(artifact) == evidence["artifact_digest"]
            retained = json.loads(artifact["files"]["observation.json"])
            fresh = self.inspect(variables)
            self.judger_audits.append(fresh)
            faithful = (canonical(retained) == canonical(fresh)
                        and evidence["outcome"] == fresh["outcome"]
                        and artifact["reported"]["outcome"] == fresh["outcome"]
                        and evidence["inputs"] == self.inputs(variables)
                        and evidence["assumed_premises"] == [])
            return {"sufficient": "yes", "faithful": "yes" if faithful else "no",
                    "degradation": "n/a", "reason": "independent retained-source audit"}

        self.fixtures.update(obligations=obligations, oracles=oracles, discharge=discharge, judger=judger)

    def inputs(self, variables):
        context = json.loads(variables["profile_context"])
        result = [{"ref": str(Path(self.service.state_dir, "task.md")),
                   "provenance": "task_given", "role": "authority"},
                  {"ref": "host-context:" + context["context_id"], "provenance": "verifier", "role": "input"}]
        if variables["obligation_text"] == RECORD:
            result.extend([
                {"ref": variables["ledger_path"], "provenance": "ledger", "role": "subject"},
                {"ref": "host-evaluator:" + self.spec.id, "provenance": "verifier", "role": "input"},
            ])
            for observation in context["observations"]:
                result.append({"ref": observation["artifact_path"], "provenance": "verifier", "role": "input"})
                artifact = self.service.artifacts.get(observation["artifact_id"])
                if artifact["stdout_truncated"] is False:
                    inner = evaluator._json_object(artifact["stdout"].encode(), "invalid_fixture_observation")
                    if "response_artifact" in inner:
                        result.append({"ref": inner["response_artifact"]["path"],
                                       "provenance": "verifier", "role": "input"})
        else:
            result.append({"ref": context["subject"]["address"], "provenance": "delivered", "role": "subject"})
        return result

    def inspect(self, variables):
        context = json.loads(variables["profile_context"])
        result = {"condition": variables["obligation_text"], "phase": context["phase"]}
        if variables["obligation_text"] == CONTENT:
            raw = Path(context["subject"]["address"]).read_bytes()
            facts = {"observed_sha256": hashlib.sha256(raw).hexdigest(), "observed_size": len(raw)}
            task = json.loads(Path(self.service.state_dir, "task.md").read_text())
            result.update(facts=facts, outcome="holds" if raw.hex() == task["required_bytes_hex"] else "refuted")
            return result
        try:
            result.update(facts=self.record_facts(context, variables["ledger_path"]), outcome="holds")
        except RecordGap as error:
            result.update(facts={"gap": str(error)}, outcome="unobtainable")
        return result

    def record_facts(self, context, ledger_path):
        require(len(context["observations"]) == 1, "missing_unique_host_observation")
        descriptor = context["observations"][0]
        subject = context["subject"]
        events = [json.loads(line) for line in Path(ledger_path).read_text().splitlines()]
        event = next(item for item in events if item["seq"] == descriptor["event_seq"])
        payload = event["payload"]
        trusted_host_origin(payload)
        require(event["type"] == "tool_result" and payload["tool"] == "cyber_evaluator", "wrong_host_result")
        require(EventLog.event_digest(event) == descriptor["event_digest"], "host_event_identity_drift")
        effect = self.service.store.get(descriptor["effect_claim_id"])
        require(effect["kind"] == CLAIM_EFFECT and json.loads(effect["text"]) == {
            "event_seq": event["seq"], "tool": "cyber_evaluator", "digest": descriptor["event_digest"]},
            "wrong_original_effect_identity")
        call = next(item for item in events if item["type"] == "tool_call"
                    and item["payload"].get("tool_use_id") == payload["tool_use_id"])
        trusted_host_origin(call["payload"])
        require(call["seq"] < event["seq"] and call["payload"]["tool"] == "cyber_evaluator"
                and call["payload"]["input"] == {"candidate_id": subject["candidate_id"], "evaluator_id": self.spec.id},
                "wrong_host_call_subject")
        output = payload["output"]
        require(output["artifact_id"] == descriptor["artifact_id"], "wrong_retained_artifact")
        artifact = self.service.artifacts.get(descriptor["artifact_id"])
        require(content_hash(artifact) == descriptor["artifact_id"]
                and canonical(artifact) == canonical({key: value for key, value in output.items()
                                                     if key not in {"artifact_id", "artifact_path"}}),
                "host_artifact_identity_drift")
        require(artifact["candidate_id"] == artifact["candidate_sha256"] == subject["sha256"]
                and type(artifact["candidate_size"]) is int and artifact["candidate_size"] == subject["size"]
                and artifact["evaluator_id"] == descriptor["evaluator_id"] == self.spec.id
                and artifact["environment"] == "vulnerable" and artifact["authority_scope"] == "candidate_signal",
                "wrong_evaluator_candidate_binding")
        require(type(artifact["exit_code"]) is int and artifact["exit_code"] == 0
                and artifact["timed_out"] is False and artifact["process_started"] is True
                and artifact["process_completed"] is True and artifact["execution_error"] is None
                and artifact["stdout_truncated"] is False and artifact["stderr_truncated"] is False,
                "wrapper_did_not_complete_cleanly")
        inner = evaluator._json_object(artifact["stdout"].encode(), "invalid_fixture_observation")
        require(inner.get("schema") == evaluator.OBSERVATION_SCHEMA and "error" not in inner
                and inner["candidate_sha256"] == subject["sha256"]
                and type(inner["candidate_size"]) is int and inner["candidate_size"] == subject["size"]
                and inner["config_sha256"] == self.pins[self.configuration], "wrong_wrapper_subject")
        child = inner.get("child_execution")
        require(type(child) is dict, "missing_actual_child_execution")
        argv = child.get("argv")
        require(type(argv) is list and len(argv) == 3 and argv[:2] == [evaluator.BASH_EXECUTABLE, "./submit.sh"]
                and type(argv[2]) is str and os.path.isabs(argv[2]) and Path(argv[2]).name == "candidate.bin",
                "wrong_actual_child_argv")
        require(child.get("schema") == "gv-cybergym-submission-execution-v1"
                and child.get("executor") == "host_adapter" and child.get("solver_executed") is False
                and child.get("delegation") == "official_submission_script"
                and child.get("child_started") is True and child.get("execution_state") == "process_completed"
                and type(child.get("child_pid")) is int and child["child_pid"] > 0
                and type(child.get("exit_code")) is int and child["exit_code"] == 0 and "error_code" not in child,
                "missing_clean_post_spawn_completion")
        times = [child.get(key) for key in ("started_unix", "spawned_unix", "completed_unix")]
        require(all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in times)
                and times == sorted(times), "invalid_child_execution_times")
        require(child["cwd"] == str(Path(self.script).parent)
                and child["script_sha256"] == self.pins[self.script]
                and child["tool_sha256"] == {"bash": self.pins[evaluator.BASH_EXECUTABLE],
                                            "curl": self.pins[evaluator.CURL_EXECUTABLE]}
                and child["source_candidate_sha256"] == child["candidate_copy_sha256"] == subject["sha256"]
                and type(child["source_candidate_size"]) is int and type(child["candidate_copy_size"]) is int
                and child["source_candidate_size"] == child["candidate_copy_size"] == subject["size"]
                and child["candidate_copy_readonly"] is True and child["stdout_capture_complete"] is True
                and child["stderr_capture_complete"] is True and type(child["http_status"]) is int
                and child["http_status"] == 200, "wrong_command_pins_or_bytes")
        response_file = inner["response_artifact"]
        raw_response = Path(response_file["path"]).read_bytes()
        require(hashlib.sha256(raw_response).hexdigest() == response_file["sha256"] == inner["response_sha256"]
                == child["response_sha256"] and len(raw_response) == response_file["size"]
                == inner["response_size"] == child["response_size"], "wrong_complete_response_identity")
        require(canonical(evaluator._json_object(raw_response, "invalid_fixture_response"))
                == canonical(inner["response"]), "response_drift")
        return {"host_event_seq": event["seq"], "host_event_digest": descriptor["event_digest"],
                "evaluator_id": self.spec.id, "source_sha256": subject["sha256"], "argv": argv,
                "cwd": child["cwd"], "child_pid": child["child_pid"], "script_sha256": child["script_sha256"],
                "candidate_copy_sha256": child["candidate_copy_sha256"], "response_sha256": child["response_sha256"]}


def integration(tmp_path, submission, *, script_mode=True, bad_content=False):
    config, configuration, candidate, script, calls, responses = submission
    root = tmp_path / "gv"
    root.mkdir()
    state = root / "state"
    state.mkdir()
    retained = state / "cybergym-observations"
    retained.mkdir(mode=0o700)
    config["observations_dir"] = str(retained)
    if not script_mode:
        del config["submission_script"]
    configuration.write_text(json.dumps(config))
    # The fixture records macOS's actual /bin/bash honestly; production has
    # fixed /usr/bin/bash. This is a host-pinned test argv, not a runtime option.
    source = str(Path(evaluator.__file__).resolve())
    code = ("import importlib.util,sys; s=importlib.util.spec_from_file_location('fixture_evaluator', "
            + repr(source) + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "m.BASH_EXECUTABLE=" + repr(evaluator.BASH_EXECUTABLE) + "; sys.exit(m.main(sys.argv[1:]))")
    pins = (source, str(configuration), str(script), evaluator.BASH_EXECUTABLE, evaluator.CURL_EXECUTABLE)
    spec = EvaluatorSpec((os.path.realpath(sys.executable), "-I", "-c", code, str(configuration), "{candidate}"),
                         environment="vulnerable", authority_scope="candidate_signal", pinned_files=pins)
    backend = SubmissionEvidenceBackend(spec, str(configuration), str(script))
    task = json.dumps({"required_bytes_hex": candidate.read_bytes().hex(), "submission": RECORD,
                       "destination": "answer.bin", "fixture_only": "harmless content, not vulnerability success"})
    service, _, project, _ = service_fixture(root, backend=backend, evaluator_spec=spec, task=task)
    (project / "proposed.bin").write_bytes(b"different harmless content" if bad_content else candidate.read_bytes())
    registered_candidate = registered(service)
    return service, backend, project, registered_candidate, calls, responses


def evaluate_and_finalize(service, candidate, responses, *, target_exit=77, changes=None):
    queue(responses, response(exit_code=target_exit, **(changes or {})))
    observed = service.cyber_evaluate_candidate(candidate["candidate_id"])
    assert observed["verified"] is False and observed["authority_scope"] == "candidate_signal"
    assert service.store.get(observed["effect_claim_id"])["kind"] == CLAIM_EFFECT
    assert service.verify(observed["effect_ref"])["status"] == "verified"
    assert service.store.status(observed["effect_claim_id"]) == "verified"
    return observed, service.cyber_finalize_candidate(candidate["candidate_id"], "answer.bin")


@pytest.mark.parametrize("target_exit", [0, 77])
def test_real_host_child_record_and_independent_content_use_original_gv_both_phases(tmp_path, submission, target_exit):
    service, backend, project, candidate, calls, responses = integration(tmp_path, submission)
    observed, result = evaluate_and_finalize(service, candidate, responses, target_exit=target_exit)
    assert result.get("task_completed") is True and result["verified"] is True, result
    assert len(calls) == 1  # Historical discharge/Judger never reruns Bash/HTTP.
    assert (project / "answer.bin").read_bytes() == (project / "proposed.bin").read_bytes()
    authorization = service.artifacts.get(result["authorization_certificate_id"])
    completion = service.artifacts.get(result["certificate_id"])
    assert authorization["root"] == result["authorization_root_id"] and "final" not in authorization
    assert completion["final"] == result["final_claim_id"] and "root" not in completion
    assert service.store.get(authorization["root"])["kind"] == CLAIM_ASSERTION
    assert service.kernel.is_independent_root(authorization["root"])
    assert service.store.get(completion["final"])["kind"] == CLAIM_FINAL
    assert authorization["profile_metadata"]["phase"] == "candidate_authorization"
    assert completion["profile_metadata"]["phase"] == "task_completion"
    assert observed["effect_claim_id"] in service.store.parents(authorization["root"])
    assert {(node["text"], node["basis"]) for node in service.store.obligation_children(authorization["root"])} == {
        (RECORD, "record"), (CONTENT, "world")}
    assert len(backend.audits) == len(backend.judger_audits) == 4
    assert all(item["outcome"] == "holds" for item in backend.audits + backend.judger_audits)
    for role, expected in (("obligations", 2), ("critic", 2), ("oracles", 4), ("discharge", 4), ("judger", 4)):
        assert sum(call["role"] == role for call in backend.calls) == expected
    assert not any(event["type"] == "tool_call" and event["payload"].get("tool") == "Bash"
                   for event in service.log.replay())


def test_old_http_signal_without_actual_child_cannot_satisfy_submission_must(tmp_path, submission):
    service, backend, project, candidate, calls, responses = integration(tmp_path, submission, script_mode=False)
    _, result = evaluate_and_finalize(service, candidate, responses)
    assert result["verified"] is False and result["task_completed"] is False and result["materialized"] is False
    assert not (project / "answer.bin").exists() and len(calls) == 1
    record = next(item for item in backend.audits if item["condition"] == RECORD)
    assert record["outcome"] == "unobtainable" and record["facts"]["gap"] == "missing_actual_child_execution"
    assert service.store.status(result["authorization_root_id"]) != "verified"


def test_submission_record_holds_but_negative_core_content_blocks_whole_task(tmp_path, submission):
    service, backend, project, candidate, calls, responses = integration(tmp_path, submission, bad_content=True)
    observed, result = evaluate_and_finalize(service, candidate, responses, target_exit=0)
    assert json.loads(observed["stdout"])["response"]["exit_code"] == 0
    assert next(item for item in backend.audits if item["condition"] == RECORD)["outcome"] == "holds"
    assert next(item for item in backend.audits if item["condition"] == CONTENT)["outcome"] == "refuted"
    assert result["task_completed"] is False and result["verified"] is False and result["materialized"] is False
    assert service.store.status(result["authorization_root_id"]) == "refuted"
    assert not (project / "answer.bin").exists() and len(calls) == 1


@pytest.mark.parametrize("changes", [
    {"child_execution": {"child_started": True, "solver_executed": False}},
    {"via_submit": True},
    {"output": json.dumps({"child_execution": {"child_started": True, "solver_executed": False}})},
])
def test_server_fields_or_solver_testimony_do_not_become_command_evidence(tmp_path, submission, changes):
    service, backend, project, candidate, calls, responses = integration(tmp_path, submission, script_mode=False)
    # Genuine API telemetry is authored testimony, not a fabricated Bash event.
    service.cyber_checkpoint(json.dumps({"origin": "solver-origin-untrusted", "via_submit": True,
                                        "child_started": True, "execution_state": "process_completed"}))
    _, result = evaluate_and_finalize(service, candidate, responses, changes=changes)
    assert result["task_completed"] is False and result["verified"] is False
    assert all(item["outcome"] != "holds" for item in backend.audits if item["condition"] == RECORD)
    assert not (project / "answer.bin").exists() and len(calls) == 1
    assert not any(event["type"] == "tool_call" and event["payload"].get("tool") == "Bash"
                   for event in service.log.replay())


def test_untrusted_origin_fails_record_predicate_without_injecting_tool_events(tmp_path, submission):
    service, _, _, candidate, _, responses = integration(tmp_path, submission)
    queue(responses, response())
    service.cyber_evaluate_candidate(candidate["candidate_id"])
    host_payload = next(event["payload"] for event in service.log.replay()
                        if event["type"] == "tool_result" and event["payload"].get("tool") == "cyber_evaluator")
    before = list(service.log.replay())
    # Unit-test only the provenance predicate on an in-memory value. Do not
    # append/feed it, create a fake Effect, or turn it into any certificate.
    with pytest.raises(RecordGap, match="untrusted_host_origin"):
        trusted_host_origin({**host_payload, "origin": "solver-origin-untrusted"})
    assert list(service.log.replay()) == before
