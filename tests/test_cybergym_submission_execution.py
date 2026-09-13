"""Real harmless bash/curl and localhost HTTP fixtures; no models or Docker."""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from experiments.cybergym import evaluator


def response(task="fixture-task", **changes):
    return {"task_id": task, "exit_code": 77, "output": "harmless fixture", "poc_id": "fixture-record", **changes}


@pytest.fixture
def submission(tmp_path, monkeypatch):
    # Production always uses /usr/bin/bash. macOS has only /bin/bash; this
    # isolated fixture records the real overridden argv rather than inventing
    # /usr/bin/bash evidence. No runtime configuration override is provided.
    if not Path(evaluator.BASH_EXECUTABLE).exists():
        monkeypatch.setattr(evaluator, "BASH_EXECUTABLE", "/bin/bash")
    calls, responses = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append((self.path, dict(self.headers), body))
            status, payload, headers = responses.pop(0)
            if status is None:
                return  # Genuine curl transport error: owned peer closes with no HTTP reply.
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(payload)
            except BrokenPipeError:
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script_parent = tmp_path / "private-submission"
    script_parent.mkdir(mode=0o700)
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    script = script_parent / "submit.sh"
    config = {"schema": evaluator.CONFIG_SCHEMA, "server_url": f"http://127.0.0.1:{server.server_port}",
              "task_id": "fixture-task", "agent_id": "fixture-agent", "checksum": "private-checksum",
              "require_flag": False, "submission_script": str(script), "observations_dir": str(evidence)}
    script.write_bytes(evaluator.render_submission_script(config))
    script.chmod(0o400)
    script_parent.chmod(0o500)
    candidate = tmp_path / "candidate"
    candidate.write_bytes(b"\x00\xffharmless fixture bytes\r\n")
    candidate.chmod(0o400)
    config_path = tmp_path / "configuration.json"
    config_path.write_text(json.dumps(config))
    try:
        yield config, config_path, candidate, script, calls, responses
    finally:
        script_parent.chmod(0o700)
        server.shutdown()
        server.server_close()
        thread.join()


def queue(responses, value, status=200, headers=None):
    raw = value if type(value) is bytes else json.dumps(value).encode()
    responses.append((status, raw, headers or {}))
    return raw


def test_real_official_bash_submission_is_host_execution_not_target_verdict(submission):
    config, config_path, candidate, script, calls, responses = submission
    raw = queue(responses, response())
    observation = evaluator.evaluate(str(config_path), str(candidate))
    assert len(calls) == 1 and calls[0][0] == "/submit-vul"
    assert candidate.read_bytes() in calls[0][2]
    assert config["checksum"].encode() in calls[0][2]
    record = observation["child_execution"]
    assert record["argv"][:2] == [evaluator.BASH_EXECUTABLE, "./submit.sh"]
    copied = record["argv"][2]
    assert os.path.isabs(copied) and Path(copied).name == "candidate.bin"
    assert copied != str(candidate) and not Path(copied).exists()
    assert record["cwd"] == str(script.parent)
    assert type(record["child_pid"]) is int and record["child_pid"] > 0
    assert record["started_unix"] <= record["completed_unix"]
    assert record["child_started"] is True and record["execution_state"] == "process_completed"
    assert record["duration_seconds"] >= 0
    assert record["exit_code"] == 0 and observation["response"]["exit_code"] == 77
    assert record["http_status"] == 200 and record["executor"] == "host_adapter"
    assert record["delegation"] == "official_submission_script" and record["solver_executed"] is False
    assert record["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert record["candidate_copy_sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert record["stdout_capture_complete"] is True and record["stdout_size"] > len(raw)
    assert record["response_sha256"] == observation["response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["stderr_size"] > 0 and len(record["stderr_sha256"]) == 64  # curl's progress, no raw text.
    assert Path(observation["response_artifact"]["path"]).read_bytes() == raw
    assert not {"success", "verified", "reward", "score", "target_success"} & record.keys()
    public = json.dumps(observation)
    assert config["checksum"] not in public and config["server_url"] not in public


def test_private_copy_excludes_curl_form_grammar(submission):
    _config, config_path, candidate, _script, calls, responses = submission
    hazardous = candidate.with_name("poc;type=text,second@file")
    candidate.rename(hazardous)
    queue(responses, response())
    observed = evaluator.evaluate(str(config_path), str(hazardous))
    assert hazardous.read_bytes() in calls[0][2]
    assert Path(observed["child_execution"]["argv"][2]).name == "candidate.bin"


def test_clear_environment_and_private_curlrc_ignore_credentials_and_proxies(submission, monkeypatch, tmp_path):
    _config, config_path, candidate, _script, calls, responses = submission
    hostile = tmp_path / "hostile-home"
    hostile.mkdir()
    (hostile / ".curlrc").write_text('url = "http://127.0.0.1:1/private"\nlocation\nheader = "X-Secret: credential"\n')
    for key in ("HOME", "CURL_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, str(hostile))
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(key, "http://user:credential@127.0.0.1:1")
    monkeypatch.setenv("BASH_ENV", str(hostile / ".curlrc"))
    monkeypatch.setenv("OPENAI_API_KEY", "credential")
    monkeypatch.setenv("TMPDIR", str(hostile))
    real = evaluator.subprocess.Popen
    observed_arguments = []
    def popen(*args, **kwargs):
        observed_arguments.append(kwargs)
        return real(*args, **kwargs)
    monkeypatch.setattr(evaluator.subprocess, "Popen", popen)
    queue(responses, response())
    observed = evaluator.evaluate(str(config_path), str(candidate))
    assert len(calls) == 1 and "X-Secret" not in calls[0][1]
    environment = observed_arguments[0]["env"]
    assert set(environment) == {"PATH", "HOME", "CURL_HOME", "LC_ALL"}
    assert environment["PATH"] == "/usr/bin:/bin" and environment["HOME"] == environment["CURL_HOME"]
    assert str(hostile) not in environment["HOME"]
    assert "timeout" not in observed_arguments[0]
    assert observed["child_execution"]["transport_hardening"]["inherited_environment"] is False


@pytest.mark.parametrize("update", [{"exit_code": True}, {"exit_code": 77.0}, {"task_id": "other"}, {"poc_id": ""},
                                    {"output": {}}, {"child_execution": {"solver_executed": True}},
                                    {"solver_executed": False}, {"via_submit": True},
                                    {"flag": "unexpected"}, {"fix_exit_code": 0}])
def test_response_validation_cannot_overwrite_execution_provenance(submission, update):
    _config, config_path, candidate, _script, calls, responses = submission
    queue(responses, response(**update))
    with pytest.raises(evaluator.AdapterError, match="invalid_response") as failure:
        evaluator.evaluate(str(config_path), str(candidate))
    assert len(calls) == 1 and failure.value.child_execution["exit_code"] == 0
    assert failure.value.child_execution["solver_executed"] is False
    assert failure.value.child_execution["http_status"] == 200


@pytest.mark.parametrize("status,code", [(302, "unexpected_redirect"), (307, "unexpected_redirect"),
                                        (404, "unexpected_http_status"), (500, "unexpected_http_status")])
def test_http_status_is_separate_from_bash_exit_and_no_redirect(submission, status, code):
    config, config_path, candidate, _script, calls, responses = submission
    queue(responses, response(), status, {"Location": config["server_url"] + "/submit-fix"})
    with pytest.raises(evaluator.AdapterError, match=code) as failure:
        evaluator.evaluate(str(config_path), str(candidate))
    assert [call[0] for call in calls] == ["/submit-vul"]
    assert failure.value.child_execution["exit_code"] == 0
    assert failure.value.child_execution["http_status"] == status


def test_bounded_response_retains_only_counts_hashes_on_error(submission, monkeypatch, capsys):
    config, config_path, candidate, _script, calls, responses = submission
    monkeypatch.setattr(evaluator, "MAX_RESPONSE_BYTES", 128)
    queue(responses, response(output=config["checksum"] * 1000))
    assert evaluator.main([str(config_path), str(candidate)]) == 1
    console = capsys.readouterr()
    result = json.loads(console.out)
    assert result["error"] == "response_too_large" and result["response_truncated"] is True
    assert result["child_execution"]["stdout_capture_complete"] is False
    assert len(calls) == 1 and config["checksum"] not in console.out + console.err


@pytest.mark.parametrize("change", ["printf", "endpoint", "task", "append"])
def test_arbitrary_or_wrong_official_script_is_rejected_before_execution(submission, change):
    _config, config_path, candidate, script, calls, _responses = submission
    raw = script.read_bytes()
    replacement = {"printf": b"#!/bin/bash\nprintf '{\"task_id\":\"fixture-task\"}'",
                   "endpoint": raw.replace(b"/submit-vul", b"/submit-fix"),
                   "task": raw.replace(b"fixture-task", b"other-task"), "append": raw + b"\ntrue"}[change]
    script.chmod(0o600)
    script.write_bytes(replacement)
    script.chmod(0o400)
    with pytest.raises(evaluator.AdapterError, match="submission_script_mismatch"):
        evaluator.evaluate(str(config_path), str(candidate))
    assert not calls


@pytest.mark.parametrize("field,value", [("checksum", "x';printf fabricated;'"), ("agent_id", 'x"}'),
                                        ("task_id", "x\ntrue"), ("server_url", "http://host;true"),
                                        ("server_url", "http://host/"), ("server_url", "http://$(true)")])
def test_script_mode_rejects_template_injection_and_double_slash_before_submit(submission, field, value):
    config, config_path, candidate, _script, calls, _responses = submission
    config[field] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises(evaluator.AdapterError, match="invalid_submission_configuration"):
        evaluator.evaluate(str(config_path), str(candidate))
    assert not calls


@pytest.mark.parametrize("kind", ["leaf_symlink", "parent_symlink", "hardlink", "writable_parent", "writable_script"])
def test_script_storage_boundary_and_pins_fail_closed(submission, tmp_path, kind):
    config, config_path, candidate, script, calls, _responses = submission
    if kind == "leaf_symlink":
        script.parent.chmod(0o700)
        source = tmp_path / "template-copy"
        source.write_bytes(script.read_bytes())
        source.chmod(0o400)
        script.unlink()
        script.symlink_to(source)
        script.parent.chmod(0o500)
    elif kind == "parent_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(script.parent, target_is_directory=True)
        config["submission_script"] = str(alias / "submit.sh")
    elif kind == "hardlink":
        os.link(script, tmp_path / "other-link")
    elif kind == "writable_parent":
        script.parent.chmod(0o700)
    else:
        script.chmod(0o600)
    config_path.write_text(json.dumps(config))
    with pytest.raises(evaluator.AdapterError, match="unsafe_submission_script"):
        evaluator.evaluate(str(config_path), str(candidate))
    assert not calls


def test_candidate_is_pinned_before_and_after_actual_execution(submission, monkeypatch):
    _config, config_path, candidate, _script, calls, responses = submission
    real = evaluator._child_output
    def changed(*args, **kwargs):
        result = real(*args, **kwargs)
        candidate.chmod(0o600)
        candidate.write_bytes(b"changed after actual upload")
        candidate.chmod(0o400)
        return result
    monkeypatch.setattr(evaluator, "_child_output", changed)
    queue(responses, response())
    with pytest.raises(evaluator.AdapterError, match="candidate_changed") as failure:
        evaluator.evaluate(str(config_path), str(candidate))
    assert len(calls) == 1 and failure.value.child_execution["child_pid"] > 0


def test_real_curl_transport_failure_records_actual_outer_exit_without_raw_error(submission, capsys):
    config, config_path, candidate, _script, calls, responses = submission
    queue(responses, b"", status=None)
    assert evaluator.main([str(config_path), str(candidate)]) == 1
    console = capsys.readouterr()
    value = json.loads(console.out)
    record = value["child_execution"]
    assert value["error"] == "transport_error" and record["exit_code"] != 0
    assert record["child_started"] is True and record["child_pid"] > 0
    assert record["execution_state"] == "process_completed" and record["stderr_size"] > 0
    assert len(calls) == 1 and config["checksum"] not in console.out + console.err
    assert config["server_url"] not in console.out + console.err


def test_spawn_error_does_not_fabricate_started_child_or_leak_private_exception(submission, monkeypatch, capsys):
    config, config_path, candidate, _script, calls, _responses = submission
    def fail(*_args, **_kwargs):
        raise OSError(config["checksum"] + config["server_url"])
    monkeypatch.setattr(evaluator.subprocess, "Popen", fail)
    assert evaluator.main([str(config_path), str(candidate)]) == 1
    console = capsys.readouterr()
    value = json.loads(console.out)
    record = value["child_execution"]
    assert value["error"] == "submission_execution_error"
    assert record["child_started"] is False and record["execution_state"] == "spawn_failed"
    assert "child_pid" not in record and "exit_code" not in record
    assert record["stdout_size"] == record["stderr_size"] == 0
    assert not calls and config["checksum"] not in console.out + console.err
    assert config["server_url"] not in console.out + console.err


@pytest.mark.parametrize("changed_dependency", ["script", "copy", "tool_identity"])
def test_actual_execution_rejects_changed_post_pins(submission, monkeypatch, changed_dependency):
    _config, config_path, candidate, script, calls, responses = submission
    real = evaluator._child_output
    def changed(argv, *args, **kwargs):
        output = real(argv, *args, **kwargs)
        if changed_dependency in ("script", "copy"):
            path = script if changed_dependency == "script" else Path(argv[2])
            path.chmod(0o600)
            path.write_bytes(path.read_bytes() + b"changed")
            path.chmod(0o400)
        else:
            pinned = evaluator._pinned_bytes
            def changed_tool(path, *arguments, **options):
                raw, identity = pinned(path, *arguments, **options)
                if path == evaluator.CURL_EXECUTABLE:
                    identity = (*identity[:4], identity[4] + 1, *identity[5:])
                return raw, identity
            monkeypatch.setattr(evaluator, "_pinned_bytes", changed_tool)
        return output
    monkeypatch.setattr(evaluator, "_child_output", changed)
    queue(responses, response())
    with pytest.raises(evaluator.AdapterError) as failure:
        evaluator.evaluate(str(config_path), str(candidate))
    assert str(failure.value) == {"script": "submission_script_mismatch", "copy": "submission_candidate_copy_changed",
                                  "tool_identity": "submission_dependency_changed"}[changed_dependency]
    assert len(calls) == 1 and failure.value.child_execution["child_started"] is True


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "writable"])
def test_unpinned_source_candidate_never_reaches_bash(submission, tmp_path, kind):
    _config, config_path, candidate, _script, calls, _responses = submission
    if kind == "symlink":
        alias = tmp_path / "candidate-alias"
        alias.symlink_to(candidate)
        candidate = alias
    elif kind == "hardlink":
        os.link(candidate, tmp_path / "candidate-link")
    else:
        candidate.chmod(0o600)
    with pytest.raises(evaluator.AdapterError, match="invalid_candidate"):
        evaluator.evaluate(str(config_path), str(candidate))
    assert not calls


def test_check_config_never_runs_poc_or_network_and_has_static_errors(submission, monkeypatch, capsys):
    config, config_path, candidate, script, calls, _responses = submission
    monkeypatch.setattr(evaluator.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("no child on static preflight"))
    assert evaluator.main(["--check-config", str(config_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"schema": evaluator.OBSERVATION_SCHEMA, "configuration_valid": True}
    assert not calls
    config["server_url"] += "/"
    config_path.write_text(json.dumps(config))
    assert evaluator.main(["--check-config", str(config_path)]) == 1
    console = capsys.readouterr()
    assert json.loads(console.out)["error"] == "invalid_submission_configuration"
    assert config["checksum"] not in console.out + console.err


def test_static_check_rejects_unavailable_or_symlink_tools(submission, monkeypatch, tmp_path):
    config, _config_path, _candidate, _script, calls, _responses = submission
    monkeypatch.setattr(evaluator, "BASH_EXECUTABLE", str(tmp_path / "absent-bash"))
    with pytest.raises(evaluator.AdapterError, match="submission_tool_unavailable"):
        evaluator.check_submission_configuration(config)
    alias = tmp_path / "curl-alias"
    alias.symlink_to("/usr/bin/curl")
    monkeypatch.setattr(evaluator, "BASH_EXECUTABLE", str(alias))
    with pytest.raises(evaluator.AdapterError, match="submission_tool_unavailable"):
        evaluator.check_submission_configuration(config)
    assert not calls


def test_default_http_abi_is_unchanged_and_does_not_run_script(submission, monkeypatch):
    config, config_path, candidate, _script, calls, responses = submission
    del config["submission_script"]
    config["server_url"] += "/"
    config["checksum"] = "quotes'\"$; remain safe HTTP metadata"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(evaluator.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("default must remain HTTP"))
    raw = queue(responses, response())
    observed = evaluator.evaluate(str(config_path), str(candidate))
    assert "child_execution" not in observed and observed["response"] == response()
    assert calls[0][0] == "/submit-vul"
    assert observed["response_sha256"] == hashlib.sha256(raw).hexdigest()


def test_cli_error_keeps_raw_stderr_and_private_authority_out_of_console(submission, capsys):
    config, config_path, candidate, _script, calls, responses = submission
    queue(responses, (config["checksum"] + config["server_url"]).encode(), status=404)
    assert evaluator.main([str(config_path), str(candidate)]) == 1
    console = capsys.readouterr()
    value = json.loads(console.out)
    assert value["error"] == "unexpected_http_status" and value["child_execution"]["http_status"] == 404
    assert value["child_execution"]["stderr_size"] > 0
    assert config["checksum"] not in console.out + console.err and config["server_url"] not in console.out + console.err
    assert len(calls) == 1
