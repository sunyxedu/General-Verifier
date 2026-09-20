from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from experiments.cybergym import evaluator


@pytest.fixture
def adapter(tmp_path):
    calls, responses = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append((self.path, dict(self.headers), body))
            status, payload, headers = responses.pop(0)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    config = {"schema": evaluator.CONFIG_SCHEMA,
              "server_url": f"http://127.0.0.1:{server.server_port}",
              "task_id": "masked-fixture-id", "agent_id": "fixture-agent",
              "checksum": "private-checksum-sentinel", "require_flag": False,
              "observations_dir": str(evidence)}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    candidate_path = tmp_path / "candidate"
    candidate_path.write_bytes(b"\x00\xffraw-input\r\n")
    try:
        yield config, config_path, candidate_path, calls, responses
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def response(exit_code=77, output="fixture diagnostic"):
    return {"task_id": "masked-fixture-id", "exit_code": exit_code,
            "output": output, "poc_id": "fixture-record"}


def queue(responses, value, status=200, headers=None):
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    responses.append((status, raw, headers or {}))
    return raw


@pytest.mark.parametrize("exit_code", [0, 77, 300])
def test_exact_candidate_and_full_vulnerable_feedback_are_observations(adapter, exit_code):
    config, config_path, candidate_path, calls, responses = adapter
    value = response(exit_code, "diagnostic\n#0 fixture_entry\nverified score=1")
    raw = queue(responses, value)
    observed = evaluator.evaluate(str(config_path), str(candidate_path))
    assert len(calls) == 1 and calls[0][0] == "/submit-vul"
    assert candidate_path.read_bytes() in calls[0][2]
    assert config["checksum"].encode() in calls[0][2]
    assert b"X-API-Key" not in repr(calls[0][1]).encode()
    assert observed["candidate_sha256"] == hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    assert observed["candidate_size"] == candidate_path.stat().st_size
    assert observed["response"] == value
    assert observed["authority_scope"] == "candidate_signal"
    assert observed["environment"] == "vulnerable"
    assert not {"verified", "success", "score", "reward", "outcome"} & observed.keys()
    assert config["checksum"] not in json.dumps(observed)
    assert config["server_url"] not in json.dumps(observed)
    assert Path(observed["response_artifact"]["path"]).read_bytes() == raw


def test_repeated_candidates_remain_online_vulnerable_only(adapter):
    _config, config_path, candidate_path, calls, responses = adapter
    queue(responses, response(0))
    first = evaluator.evaluate(str(config_path), str(candidate_path))
    candidate_path.write_bytes(b"different candidate")
    queue(responses, response(77))
    second = evaluator.evaluate(str(config_path), str(candidate_path))
    assert first["candidate_sha256"] != second["candidate_sha256"]
    assert [call[0] for call in calls] == ["/submit-vul", "/submit-vul"]


def test_complete_long_response_is_retained_without_projection(adapter):
    _config, config_path, candidate_path, _calls, responses = adapter
    raw = queue(responses, response(output="full-diagnostic-line\n" * 10000))
    observed = evaluator.evaluate(str(config_path), str(candidate_path))
    artifact = observed["response_artifact"]
    assert artifact["size"] > 64 * 1024
    assert artifact["sha256"] == hashlib.sha256(raw).hexdigest()
    assert Path(artifact["path"]).read_bytes() == raw
    assert observed["response"]["output"].endswith("full-diagnostic-line\n")


@pytest.mark.parametrize("update", [
    {"task_id": "another-task"}, {"exit_code": True}, {"exit_code": "77"},
    {"output": {}}, {"poc_id": ""}, {"poc_id": None}, {"fix_exit_code": 0},
    {"flag": "unexpected"},
])
def test_mismatched_or_extended_official_response_is_rejected(adapter, update):
    _config, config_path, candidate_path, calls, responses = adapter
    value = response()
    value.update(update)
    queue(responses, value)
    with pytest.raises(evaluator.AdapterError, match="invalid_response"):
        evaluator.evaluate(str(config_path), str(candidate_path))
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [
    b"not JSON", b"[]", b'{"task_id":"a","task_id":"b"}',
    b'{"task_id":"masked-fixture-id","exit_code":NaN,"output":"","poc_id":"x"}',
])
def test_invalid_or_duplicate_json_is_rejected(adapter, payload):
    _config, config_path, candidate_path, _calls, responses = adapter
    queue(responses, payload)
    with pytest.raises(evaluator.AdapterError, match="invalid_response"):
        evaluator.evaluate(str(config_path), str(candidate_path))


def test_response_size_bound_fails_closed(adapter, monkeypatch):
    _config, config_path, candidate_path, _calls, responses = adapter
    monkeypatch.setattr(evaluator, "MAX_RESPONSE_BYTES", 20)
    queue(responses, response())
    with pytest.raises(evaluator.AdapterError, match="response_too_large"):
        evaluator.evaluate(str(config_path), str(candidate_path))


def test_redirect_cannot_reach_private_or_other_endpoint(adapter):
    config, config_path, candidate_path, calls, responses = adapter
    queue(responses, b"", 307, {"Location": config["server_url"] + "/verify-agent-pocs"})
    with pytest.raises(evaluator.AdapterError, match="unexpected_redirect"):
        evaluator.evaluate(str(config_path), str(candidate_path))
    assert len(calls) == 1


@pytest.mark.parametrize("server_url", [
    "file:///private/config", "http://host/submit-fix", "http://user:password@host",
    "http://host?checksum=private", "http://host#fragment", "http://host:invalid",
])
def test_config_cannot_change_endpoint_or_embed_credentials(adapter, server_url):
    config, config_path, candidate_path, calls, _responses = adapter
    config["server_url"] = server_url
    config_path.write_text(json.dumps(config))
    with pytest.raises(evaluator.AdapterError, match="invalid_configuration"):
        evaluator.evaluate(str(config_path), str(candidate_path))
    assert not calls


def test_cli_error_does_not_echo_http_body_or_private_config(adapter):
    config, config_path, candidate_path, calls, responses = adapter
    queue(responses, ("private-checksum-sentinel " + config["server_url"]).encode(), 400)
    process = subprocess.run([sys.executable, evaluator.__file__, str(config_path),
                              str(candidate_path)], capture_output=True, text=True)
    assert process.returncode == 1 and len(calls) == 1
    assert json.loads(process.stdout)["error"] == "transport_error"
    assert "private-checksum-sentinel" not in process.stdout + process.stderr
    assert config["server_url"] not in process.stdout + process.stderr


def test_no_application_or_inherited_socket_timeout(adapter, monkeypatch):
    _config, config_path, candidate_path, _calls, responses = adapter
    actual_build = evaluator.urllib.request.build_opener
    observed_timeouts = []

    def build(*handlers):
        opener = actual_build(*handlers)
        actual_open = opener.open

        def open_request(request, timeout):
            observed_timeouts.append(timeout)
            return actual_open(request, timeout=timeout)

        opener.open = open_request
        return opener

    monkeypatch.setattr(evaluator.urllib.request, "build_opener", build)
    queue(responses, response())
    evaluator.evaluate(str(config_path), str(candidate_path))
    assert observed_timeouts == [None]


def test_missing_and_oversized_candidates_fail_before_upload(adapter, monkeypatch):
    _config, config_path, candidate_path, calls, _responses = adapter
    monkeypatch.setattr(evaluator, "MAX_CANDIDATE_BYTES", 2)
    with pytest.raises(evaluator.AdapterError, match="invalid_candidate"):
        evaluator.evaluate(str(config_path), str(candidate_path))
    with pytest.raises(evaluator.AdapterError, match="invalid_candidate"):
        evaluator.evaluate(str(config_path), str(candidate_path) + "-missing")
    assert not calls


def test_symlink_evidence_directory_is_refused(adapter, tmp_path):
    config, config_path, candidate_path, _calls, responses = adapter
    alias = tmp_path / "evidence-alias"
    alias.symlink_to(config["observations_dir"], target_is_directory=True)
    config["observations_dir"] = str(alias)
    config_path.write_text(json.dumps(config))
    queue(responses, response())
    with pytest.raises(evaluator.AdapterError, match="evidence_storage_error"):
        evaluator.evaluate(str(config_path), str(candidate_path))
