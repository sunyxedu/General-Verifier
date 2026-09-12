"""Offline preflight protocol/lifecycle fixtures; Docker/model calls are mocked."""
from __future__ import annotations

import copy
import io
import json
import subprocess
import time
from types import SimpleNamespace

import pytest

from experiments.cybergym import preflight, sidecar
from experiments.cybergym.model_config import EFFORT, MODEL, SERVICE_TIER
from verifier.llm.codex_backend import _OFFLINE_PROFILE
from verifier.profiles.cyber.prompts import SYSTEMS

CID = "a" * 64
MARKER = "b" * 32


def report_fixture():
    return {"schema": "gv-cybergym-preflight-v1", "image": preflight.IMAGE,
            "roles": [{"role": role, "typed_config_accepted": True,
                       "effective_context": dict(preflight.VALUES),
                       "generated_permissions": {"profile": _OFFLINE_PROFILE,
                           "network_enabled": False, "answer_blind": role in ("obligations", "critic"),
                           "generated_permission_arguments_sha256": "c" * 64}}
                      for role in SYSTEMS],
            "negative_controls": [{"case": key, "typed_rejected": True}
                                  for key in (*preflight.VALUES, "duplicate_grant")],
            "model_calls": 0, "sandbox_enforcement_tested": False}


def docker_fixture(monkeypatch, report=None, labels=None, fail_start=False):
    calls = []
    report = report if report is not None else report_fixture()
    labels = labels if labels is not None else {preflight.LABEL: MARKER, "gv.purpose": preflight.PURPOSE}
    monkeypatch.setattr(preflight.uuid, "uuid4", lambda: SimpleNamespace(hex=MARKER))

    def docker(args, timeout=5):
        calls.append((args, timeout))
        if args[0] == "create":
            return subprocess.CompletedProcess(args, 0, (CID + "\n").encode(), b"")
        if args[0] == "start":
            if fail_start:
                raise preflight.PreflightError("docker_command_failed")
            return subprocess.CompletedProcess(args, 0, json.dumps(report).encode(), b"")
        if args[0] == "inspect":
            return subprocess.CompletedProcess(args, 0, json.dumps(labels).encode(), b"")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(preflight, "_docker", docker)
    return calls


def test_fresh_pinned_noauth_networkless_container_and_exact_cleanup(monkeypatch):
    calls = docker_fixture(monkeypatch)
    report = preflight.run()
    assert report["model_calls"] == 0 and report["sandbox_enforcement_tested"] is False
    command = calls[0][0]
    assert preflight.IMAGE in command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--user") + 1] == "1000:1000"
    assert "--read-only" in command
    mounts = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--mount"]
    assert len(mounts) == 1 and mounts[0].endswith(",target=/gv,readonly")
    assert "auth" not in mounts[0] and "solver-home" not in mounts[0]
    assert ("inspect", CID) == (calls[-3][0][0], calls[-3][0][-1])
    assert calls[-2][0] == ["stop", "--time", "1", CID]
    assert calls[-1][0] == ["rm", CID]
    assert not any(value in ("prune", "rm -f") for args, _ in calls for value in args)


def test_failed_fixture_still_cleans_only_owned_container(monkeypatch):
    calls = docker_fixture(monkeypatch, fail_start=True)
    with pytest.raises(preflight.PreflightError, match="docker_command_failed"):
        preflight.run()
    assert calls[-1][0] == ["rm", CID]


@pytest.mark.parametrize("labels", [{}, {preflight.LABEL: MARKER, "gv.purpose": "other"},
                                    {preflight.LABEL: "other", "gv.purpose": preflight.PURPOSE}])
def test_ownership_mismatch_never_stops_or_removes_a_container(monkeypatch, labels):
    calls = docker_fixture(monkeypatch, labels=labels)
    with pytest.raises(preflight.PreflightError, match="cleanup_failed"):
        preflight.run()
    assert not any(args[0] in ("stop", "rm") for args, _ in calls)


@pytest.mark.parametrize("mutation", ["duplicate_role", "accepted_negative", "bool_calls", "extra_payload",
                                      "supported_context", "missing_case", "bad_permissions"])
def test_report_is_complete_typed_and_only_safelisted_metadata(monkeypatch, mutation):
    report = report_fixture()
    if mutation == "duplicate_role":
        report["roles"][-1] = copy.deepcopy(report["roles"][0])
    elif mutation == "accepted_negative":
        report["negative_controls"][0]["typed_rejected"] = False
    elif mutation == "bool_calls":
        report["model_calls"] = False
    elif mutation == "extra_payload":
        report["raw_output"] = "private fixture content"
    elif mutation == "supported_context":
        report["roles"][0]["effective_context"]["model_context_window"] = 272_000
    elif mutation == "missing_case":
        report["negative_controls"].pop()
    else:
        report["roles"][0]["generated_permissions"]["network_enabled"] = True
    calls = docker_fixture(monkeypatch, report=report)
    with pytest.raises(preflight.PreflightError):
        preflight.run()
    assert calls[-1][0] == ["rm", CID]


@pytest.mark.parametrize("method", ["thread/start", "turn/start", "command/exec", "account/read", "model/list"])
def test_protocol_allowlist_cannot_start_threads_models_or_commands(method):
    process = SimpleNamespace(stdin=io.BytesIO())
    conversation = preflight._Conversation.__new__(preflight._Conversation)
    conversation.process = process
    with pytest.raises(preflight.PreflightError, match="forbidden_protocol_method"):
        conversation.send(method, {}, 0)
    assert process.stdin.getvalue() == b""


def protocol_fixture(monkeypatch, *, negative=None, config=None, reject=True):
    records = []
    process = SimpleNamespace(wait=lambda timeout=None: 0)
    monkeypatch.setattr(preflight.Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(preflight.subprocess, "Popen", lambda command, **kwargs:
                        (records.append(("command", command, kwargs)) or process))

    class Conversation:
        def __init__(self, process, deadline):
            key = negative if negative in preflight.VALUES else "filesystem"
            self.buffers = {"stderr": (f"invalid type: expected integer for {key}").encode()}

        def send(self, method, params, request_id=None):
            records.append(("rpc", method, params, request_id))

        def read(self, request_id):
            if negative and reject:
                return None
            if request_id == 0:
                return {"id": 0, "result": {}}
            effective = {**preflight.VALUES, "model": MODEL,
                         "model_reasoning_effort": EFFORT, "service_tier": SERVICE_TIER}
            if config is not None:
                effective = config
            return {"id": 1, "result": {"config": effective}}

        def close(self):
            records.append(("close",))

    monkeypatch.setattr(preflight, "_Conversation", Conversation)
    return records


def test_valid_case_only_uses_initialize_and_config_read(monkeypatch):
    records = protocol_fixture(monkeypatch)
    monkeypatch.setattr(preflight, "_permission_projection", lambda args, role: {"fixture": True})
    result = preflight._one_case("discharge", [], time.monotonic() + 1)
    assert result["effective_context"] == preflight.VALUES
    assert [record[1] for record in records if record[0] == "rpc"] == [
        "initialize", "initialized", "config/read"]
    command = next(record for record in records if record[0] == "command")
    assert command[2]["env"]["CODEX_HOME"].startswith("/preflight/codex-home/")
    assert "OPENAI_API_KEY" not in command[2]["env"]


@pytest.mark.parametrize("key", [*preflight.VALUES, "duplicate_grant"])
def test_negative_controls_must_be_typed_rejections_not_any_process_failure(monkeypatch, key):
    records = protocol_fixture(monkeypatch, negative=key)
    args = ["-c", 'permissions.gv_offline.filesystem={"/state/work"="write"}']
    result = preflight._one_case("discharge", args, time.monotonic() + 1, key)
    assert result == {"case": key, "typed_rejected": True}
    assert "app-server" in records[0][1]
    protocol_fixture(monkeypatch, negative=key, reject=False)
    with pytest.raises(preflight.PreflightError, match="negative_control_was_accepted"):
        preflight._one_case("discharge", args, time.monotonic() + 1, key)


@pytest.mark.parametrize("role", tuple(SYSTEMS))
def test_projection_inspects_each_actual_sandbox_backend_role(tmp_path, role):
    # /state is the synthetic container path; host fixture paths must not be
    # mistaken for the pinned-container assertions in _permission_projection.
    backend = sidecar.SandboxedVerifierBackend(cwd=str(tmp_path))
    args = backend._execution_policy_args(role, {"world_mode": "live"})
    translated = [value.replace(str(tmp_path), "/state") for value in args]
    projection = preflight._permission_projection(translated, role)
    assert projection["network_enabled"] is False
    assert projection["answer_blind"] is (role in ("obligations", "critic"))
    assert projection["profile"] == _OFFLINE_PROFILE


@pytest.mark.parametrize("raw,key,expected", [
    (b"invalid type expected i64 model_context_window", "model_context_window", True),
    (b"duplicate key permissions.gv_offline.filesystem", "filesystem", True),
    (b"Docker socket unavailable", "filesystem", False),
    (b"unexpected argument --ignore-user-config", "model_context_window", False),
])
def test_typed_error_classifier_rejects_unrelated_infrastructure_errors(raw, key, expected):
    assert preflight._typed_error(raw, key) is expected


def test_failed_container_exposes_only_fixed_case_error_not_cli_stderr(monkeypatch):
    calls = docker_fixture(monkeypatch)
    original = preflight._docker

    def docker(args, timeout=5):
        if args[0] == "start":
            return subprocess.CompletedProcess(args, 1, b"", b"private raw noise\n"
                b"cybergym-preflight role_extract_initialize_not_accepted\n")
        return original(args, timeout)

    monkeypatch.setattr(preflight, "_docker", docker)
    with pytest.raises(preflight.PreflightError, match="^pinned_role_extract_initialize_not_accepted$"):
        preflight.run()
    assert calls[-1][0] == ["rm", CID]
