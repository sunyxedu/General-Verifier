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
                           "network_enabled": False, "answer_blind": role in preflight.ANSWER_BLIND_ROLES,
                           "generated_permission_arguments_sha256": "c" * 64}}
                      for role in SYSTEMS],
            "negative_controls": [{"case": key, "typed_rejected": True}
                                  for key in (*preflight.VALUES, "duplicate_grant")],
            "model_calls": 0, "sandbox_enforcement_tested": False,
            "omitted_exec_only_flags": ["--ignore-user-config", "--ignore-rules", "--ephemeral"]}


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
                                      "supported_context", "missing_case", "bad_permissions", "hidden_omission"])
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
    elif mutation == "bad_permissions":
        report["roles"][0]["generated_permissions"]["network_enabled"] = True
    else:
        report["omitted_exec_only_flags"] = []
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


@pytest.mark.parametrize("inherited_strict", [False, True])
def test_valid_case_only_uses_initialize_and_config_read(monkeypatch, inherited_strict):
    records = protocol_fixture(monkeypatch)
    args = ["--ignore-user-config", "--ignore-rules", "--ephemeral", "-c", 'approval_policy="never"']
    if inherited_strict:
        args = ["--strict-config", *args]
    inspected = []
    monkeypatch.setattr(preflight, "_permission_projection", lambda original, role:
                        (inspected.append(original) or {"fixture": True}))
    result = preflight._one_case("discharge", args, time.monotonic() + 1)
    assert result["effective_context"] == preflight.VALUES
    assert [record[1] for record in records if record[0] == "rpc"] == [
        "initialize", "initialized", "config/read"]
    command = next(record for record in records if record[0] == "command")
    assert not any(flag in command[1] for flag in preflight.EXEC_ONLY_FLAGS)
    assert command[1].count("--strict-config") == 1
    assert 'approval_policy="never"' in command[1]
    assert inspected == [args] and all(flag in args for flag in preflight.EXEC_ONLY_FLAGS)
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
    assert projection["answer_blind"] is (role in preflight.ANSWER_BLIND_ROLES)
    assert projection["profile"] == _OFFLINE_PROFILE
    if role in preflight.ANSWER_BLIND_ROLES:
        assert backend._execution_cwd(role, {"world_mode": "live"}) == str(tmp_path / "_blind" / role)


@pytest.mark.parametrize("role", sorted(preflight.ANSWER_BLIND_ROLES))
@pytest.mark.parametrize("path", ["/state/record", "/state/source_evidence"])
def test_answer_blind_projection_rejects_new_evidence_directory_leak(tmp_path, role, path):
    backend = sidecar.SandboxedVerifierBackend(cwd=str(tmp_path))
    args = [value.replace(str(tmp_path), "/state") for value in
            backend._execution_policy_args(role, {"world_mode": "live"})]
    index = next(i for i, value in enumerate(args) if ".filesystem=" in value)
    prefix, table = args[index].split("=", 1)
    grants = preflight.tomllib.loads("grants=" + table)["grants"]
    assert preflight._explicit_path_grant(grants, path, root_default=True) == "deny"
    grants[path] = "read"
    args[index] = prefix + "={" + ",".join(json.dumps(key) + "=" + json.dumps(value)
                                          for key, value in grants.items()) + "}"
    with pytest.raises(preflight.PreflightError, match="answer_blind_state_grant_outside_whitelist"):
        preflight._permission_projection(args, role)


@pytest.mark.parametrize("role", sorted(preflight.ANSWER_BLIND_ROLES))
@pytest.mark.parametrize("path,grant,error", [
    ("/state", "deny", "answer_blind_state_not_default_denied"),
    ("/state", "read", "answer_blind_state_not_default_denied"),
    ("/state/task.md", None, "answer_blind_authority_not_read_only"),
    ("/state/baseline", "write", "answer_blind_authority_not_read_only"),
    ("/state/oracle_catalog", None, "answer_blind_authority_not_read_only"),
    ("/state/future_metadata", "read", "answer_blind_state_grant_outside_whitelist"),
    ("/state/future_metadata", "write", "answer_blind_state_grant_outside_whitelist"),
    ("/state/_blind/foreign", "write", "answer_blind_state_grant_outside_whitelist"),
    ("/state/baseline/private", "read", "answer_blind_state_grant_outside_whitelist"),
    ("/state/_blind/oracles/extra", "write", "answer_blind_state_grant_outside_whitelist"),
])
def test_blind_whitelist_rejects_missing_authority_and_new_metadata_access(tmp_path, role, path, grant, error):
    backend = sidecar.SandboxedVerifierBackend(cwd=str(tmp_path))
    args = [value.replace(str(tmp_path), "/state") for value in
            backend._execution_policy_args(role, {"world_mode": "live"})]
    index = next(i for i, value in enumerate(args) if ".filesystem=" in value)
    prefix, table = args[index].split("=", 1)
    grants = preflight.tomllib.loads("grants=" + table)["grants"]
    if grant is None:
        grants.pop(path)
    else:
        grants[path] = grant
    args[index] = prefix + "={" + ",".join(json.dumps(key) + "=" + json.dumps(value)
                                          for key, value in grants.items()) + "}"
    with pytest.raises(preflight.PreflightError, match="^" + error + "$"):
        preflight._permission_projection(args, role)


@pytest.mark.parametrize("cwd", ["/state", "/state/work", "/state/_blind/foreign"])
def test_inside_rejects_actual_blind_backend_cwd_before_any_protocol_call(monkeypatch, cwd):
    class Backend:
        effort, service_tier = EFFORT, SERVICE_TIER

        def __init__(self, **kwargs):
            pass

        def _execution_policy_args(self, *args):
            return ["--ignore-user-config"]

        def _execution_cwd(self, *args):
            return cwd

    monkeypatch.setattr(sidecar, "SandboxedVerifierBackend", Backend)
    monkeypatch.setattr(preflight.Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(preflight, "_permission_projection", lambda *args: {})
    monkeypatch.setattr(preflight, "_one_case", lambda *args: pytest.fail("Protocol must not start"))
    with pytest.raises(preflight.PreflightError, match="^answer_blind_execution_cwd_not_isolated$"):
        preflight.inside()


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


def test_usage_failure_is_classified_without_echoing_cli_output(monkeypatch):
    protocol_fixture(monkeypatch, negative="model_context_window")
    original = preflight._Conversation

    class Conversation(original):
        def __init__(self, process, deadline):
            super().__init__(process, deadline)
            self.buffers["stderr"] = b"error: unexpected argument '--ignore-user-config' found\n"

    monkeypatch.setattr(preflight, "_Conversation", Conversation)
    with pytest.raises(preflight.PreflightError, match="app_server_rejects_exec_only_ignore_user_config"):
        preflight._one_case("obligations", ["--ignore-user-config"], time.monotonic() + 1)


def test_pinned_help_has_no_mounts_and_cleans_exact_verified_id(monkeypatch):
    calls = docker_fixture(monkeypatch)
    original = preflight._docker

    def docker(args, timeout=5):
        if args[0] == "start":
            calls.append((args, timeout))
            return subprocess.CompletedProcess(args, 0, b"--strict-config --stdio", b"")
        return original(args, timeout)

    monkeypatch.setattr(preflight, "_docker", docker)
    report = preflight.cli_help()
    assert report["app_server_flags"] == ["--stdio", "--strict-config"]
    assert report["app_server_help_lists_ignore_user_config"] is False
    assert report["app_server_help_lists_strict_config"] is True
    assert "--mount" not in calls[0][0]
    assert calls[-1][0] == ["rm", CID]


@pytest.mark.parametrize("raw,command,rpc_error,expected", [
    (b"error unexpected argument '--enable' found private noise", ["--enable", "fast_mode"], None,
     "initialize_unexpected_argument_enable"),
    (b"error unexpected argument '--private-unprovided' found", ["--enable"], None,
     None),
    (b"error: the argument '--strict-config' cannot be used multiple times",
     ["--strict-config", "--strict-config"], None,
     "initialize_duplicate_argument_strict_config"),
    (b"invalid type for permissions.gv-offline.filesystem; private noise",
     ["-c", "permissions.gv-offline.filesystem={}"], None,
     "initialize_invalid_type_permissions_gv_offline_filesystem"),
    (b"unknown field private_field", ["-c", "model=\"gpt-5.6-luna\""], None,
     "initialize_unknown_field"),
    (b"Permission denied at /private/fixture/auth-like-file", [], None,
     "initialize_permission_denied"),
    (b"private RPC message", [], {"code": -32602, "message": "private message"},
     "initialize_rpc_error_n32602"),
])
def test_failure_features_never_echo_untrusted_details(raw, command, rpc_error, expected):
    result = preflight._failure_code(raw, "initialize", command, rpc_error)
    if expected is None:
        assert result.startswith("initialize_not_accepted_b")
    else:
        assert result == expected
    assert "private" not in result and "message" not in result
    assert "/" not in result and "noise" not in result
