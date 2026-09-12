from __future__ import annotations

import base64
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from verifier.core.events import EventLog
from verifier.execution.instruments import ArtifactStore
from verifier.profiles.cyber.candidates import CandidateStore
from verifier.profiles.cyber.evaluator import (
    MAX_OUTPUT_BYTES, Evaluator, EvaluatorError, EvaluatorSpec,
)

PYTHON = os.path.realpath(sys.executable)


@pytest.fixture
def subject(tmp_path):
    project, state = tmp_path / "project", tmp_path / "state"
    project.mkdir()
    state.mkdir()
    artifacts = ArtifactStore(str(state / "artifacts"))
    candidates = CandidateStore(str(state), str(project), artifacts,
                                EventLog(str(state / "events.jsonl")))
    path = project / "input"
    path.write_bytes(b"\x00binary\xff")
    return candidates.register(str(path)), artifacts


def evaluator(subject, code, **kwargs):
    candidate, artifacts = subject
    spec = EvaluatorSpec((PYTHON, "-c", code, "{candidate}"), **kwargs)
    return Evaluator(spec, artifacts), candidate


def test_observation_is_bound_exactly_without_success_inference(subject):
    runner, candidate = evaluator(subject, "import sys;print('verified: score=1');"
                                 "sys.stderr.write('AddressSanitizer: fake text\\n')",
                                 environment="custom-build-digest")
    observation = runner.run(candidate)
    assert observation["exit_code"] == 0
    assert observation["candidate_sha256"] == candidate["sha256"]
    assert observation["candidate_size"] == candidate["size"]
    assert observation["evaluator_id"] == runner.spec.id
    assert observation["authority_scope"] == "candidate_signal"
    assert observation["environment"] == "custom-build-digest"
    assert observation["started_unix"] <= observation["completed_unix"] <= time.time()
    assert observation["process_started"] is True
    assert observation["process_completed"] is True
    assert not ({"verified", "success", "score", "reward", "outcome"} & observation.keys())
    artifact = runner.artifacts.get(observation["artifact_id"])
    assert artifact["stdout"] == observation["stdout"]
    assert base64.b64decode(artifact["stdout_base64"]) == b"verified: score=1\n"


def test_environment_credentials_and_home_are_scrubbed(subject, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")
    monkeypatch.setenv("CUSTOM_AUTH_TOKEN", "private-test-token")
    monkeypatch.setenv("HOME", "/secret/home")
    runner, candidate = evaluator(subject, "import json,os; print(json.dumps(dict(os.environ)))")
    observation = runner.run(candidate)
    env = json.loads(observation["stdout"])
    assert "OPENAI_API_KEY" not in env and "CUSTOM_AUTH_TOKEN" not in env
    assert env["HOME"] != "/secret/home"
    assert env["HOME"] == env["TMPDIR"]


def test_candidate_substitution_is_literal_and_no_shell(subject):
    runner, candidate = evaluator(subject, "import pathlib,sys;sys.stdout.buffer.write("
                                 "pathlib.Path(sys.argv[1]).read_bytes())")
    output = runner.run(candidate)
    assert base64.b64decode(output["stdout_base64"]) == b"\x00binary\xff"
    assert not output["stdout_truncated"]


def test_failed_process_start_is_a_capture_attempt_not_completed_execution(subject, monkeypatch):
    runner, candidate = evaluator(subject, "print('not reached')")

    def unavailable(*args, **kwargs):
        raise OSError("fixture launch unavailable")

    monkeypatch.setattr("verifier.profiles.cyber.evaluator.subprocess.Popen", unavailable)
    observation = runner.run(candidate)
    assert observation["process_started"] is False
    assert observation["process_completed"] is False
    assert observation["exit_code"] is None
    assert observation["execution_error"] is not None


@pytest.mark.parametrize("argv", [(), (PYTHON, "-V"),
                                      (PYTHON, "{candidate}", "{candidate}"),
                                      (PYTHON, "prefix{candidate}"),
                                      ("python", "{candidate}")])
def test_invalid_command_spec_rejected(argv):
    with pytest.raises(EvaluatorError):
        EvaluatorSpec(argv)


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_invalid_timeout_rejected(timeout):
    with pytest.raises(EvaluatorError, match="timeout"):
        EvaluatorSpec((PYTHON, "{candidate}"), timeout=timeout)


def test_default_has_no_timeout_and_scope_is_explicit(subject):
    runner, candidate = evaluator(subject, "import time;time.sleep(.06); print('done')",
                                 authority_scope="task_outcome")
    assert runner.spec.timeout is None
    assert runner.run(candidate)["authority_scope"] == "task_outcome"
    with pytest.raises(EvaluatorError, match="scope"):
        EvaluatorSpec((PYTHON, "{candidate}"), authority_scope="trusted-success")


def test_two_output_streams_are_drained_with_bounded_retention(subject):
    runner, candidate = evaluator(subject, "import os;os.write(1,b'A'*200000);"
                                 "os.write(2,b'B'*210000)")
    observation = runner.run(candidate)
    assert observation["exit_code"] == 0
    assert len(base64.b64decode(observation["stdout_base64"])) == MAX_OUTPUT_BYTES
    assert len(base64.b64decode(observation["stderr_base64"])) == MAX_OUTPUT_BYTES
    assert observation["stdout_bytes_observed"] == 200000
    assert observation["stderr_bytes_observed"] == 210000
    assert observation["stdout_truncated"] and observation["stderr_truncated"]


def test_timeout_kills_entire_owned_process_group(subject, tmp_path):
    marker = tmp_path / "child-survived"
    code = ("import os,time,pathlib;pid=os.fork();"
            f"time.sleep(.35) if pid==0 else time.sleep(10);"
            f"pathlib.Path({str(marker)!r}).write_text('survived')")
    runner, candidate = evaluator(subject, code, timeout=.08)
    started = time.monotonic()
    observation = runner.run(candidate)
    assert observation["timed_out"]
    assert observation["exit_code"] == -signal.SIGKILL
    assert time.monotonic() - started < 2
    time.sleep(.4)
    assert not marker.exists()


def test_escaped_descendant_pipe_writer_cannot_block_drain(subject):
    code = ("import os,time;pid=os.fork();"
            "(os.setsid(),time.sleep(5)) if pid==0 else "
            "(print(pid,flush=True),os._exit(0))")
    runner, candidate = evaluator(subject, code)
    started = time.monotonic()
    observation = runner.run(candidate)
    child_pid = int(observation["stdout"].strip())
    try:
        assert observation["exit_code"] == 0
        assert time.monotonic() - started < 2
        assert observation["stdout_truncated"] or observation["stderr_truncated"]
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_executable_and_explicit_script_pins_reject_mutation(subject, tmp_path):
    program = tmp_path / "evaluator.py"
    program.write_text("print('first')")
    candidate, artifacts = subject
    spec = EvaluatorSpec((PYTHON, str(program), "{candidate}"),
                         pinned_files=(str(program),))
    assert spec.identity["pinned_files"][0]["sha256"]
    program.write_text("print('changed')")
    with pytest.raises(EvaluatorError, match="dependency pin"):
        Evaluator(spec, artifacts).run(candidate)


def test_direct_executable_pin_rejects_mutation(subject, tmp_path):
    program = tmp_path / "evaluator"
    program.write_text("#!/bin/sh\nprintf first\\n\n")
    program.chmod(0o700)
    spec = EvaluatorSpec((str(program), "{candidate}"))
    program.write_text("#!/bin/sh\nprintf changed\\n\n")
    with pytest.raises(EvaluatorError, match="executable pin"):
        Evaluator(spec, subject[1]).run(subject[0])


def test_candidate_tampering_before_or_during_run_is_rejected(subject):
    runner, candidate = evaluator(subject, "import os,pathlib,sys;os.chmod(sys.argv[1],0o600);"
                                 "pathlib.Path(sys.argv[1]).write_bytes(b'tampered')")
    with pytest.raises(EvaluatorError, match="digest mismatch"):
        runner.run(candidate)
    neutral, _ = evaluator(subject, "print('not run')")
    with pytest.raises(EvaluatorError, match="digest mismatch"):
        neutral.run(candidate)
