"""CyberGym transport/permission fixtures: no model or live evaluator calls."""
import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from experiments.cybergym import sidecar
from verifier.core.model import content_hash
from verifier.llm.agents import ALL_AGENT_ROLES
from verifier.llm.codex_backend import CodexBackend
from verifier.llm.schema import FixtureBackend, RoleError
from verifier.mcp_server import Service, _default_kernel


TOKEN = "fixture-only-" + "x" * 48
BYTES = b"immutable fixture candidate\x00\xff"


def args_fixture(tmp_path):
    state, project = tmp_path / "state", tmp_path / "task"
    state.mkdir()
    project.mkdir()
    (state / "task.md").write_text("Establish the task-defined security condition.")
    adapter = tmp_path / "adapter.py"
    adapter.write_text("print('fixture')\n")
    config = tmp_path / "config.json"
    config.write_text("{}")
    return SimpleNamespace(state_dir=state, project_root=project,
        cyber_evaluator_command_json=json.dumps([
            str(Path(sys.executable).resolve()), str(adapter), "{candidate}"]),
        cyber_evaluator_environment="vulnerable",
        cyber_evaluator_pin=[str(adapter), str(config)])


def certified_service(tmp_path):
    args = args_fixture(tmp_path)
    proposed = args.project_root / "proposed.bin"
    proposed.write_bytes(BYTES)

    def discharge(v):
        Path(v["workdir"], "evidence.json").write_text("{\"outcome\":\"holds\"}")
        return {"outcome": "holds", "witness": "bound fixture observation",
                "artifact": "evidence.json", "assumed_premises": [],
                "witness_kind": "execution", "inputs": [{
                    "ref": str(args.state_dir / "task.md"),
                    "provenance": "task_given", "role": "authority"}]}

    backend = FixtureBackend({
        "obligations": {"obligations": [{"text": "The complete task-defined security condition holds.",
            "required_basis": "world", "severity": "must"}]},
        "critic": {"verdict": "pass", "counterexample_world": "", "missing_condition": ""},
        "oracles": {"methods": [{"procedure": "execute independent bound fixture",
            "artifact": "evidence.json", "level": "independent", "basis": "world",
            "strength": "decisive", "bridges": []}]},
        "discharge": discharge,
        "judger": {"sufficient": "yes", "faithful": "yes", "degradation": "n/a",
            "reason": "fixture establishes the fixed task condition"},
    })
    service = Service(str(args.state_dir), profile="cyber", project_root=str(args.project_root),
        model=sidecar.MODEL, seats="codex", effort="max", service_tier="fast",
        final_destination="final_poc", kernel_factory=lambda sc: _default_kernel(
            sc, model=sidecar.MODEL, backend=backend, critic_rounds=1,
            parallel_obligations=1, parallel_claims=1))
    candidate = service.cyber_register_candidate("proposed.bin")
    result = service.cyber_finalize_candidate(candidate["candidate_id"])
    assert result.get("materialized") is True, result
    return service, args, backend, candidate, result


def request(app, path, *, method="GET", body=b"", headers=None):
    sent = []
    pending = [{"type": "http.request", "body": body}]
    async def receive():
        return pending.pop(0) if pending else {"type": "http.disconnect"}
    async def send(message):
        sent.append(message)
    asyncio.run(app({"type": "http", "path": path, "method": method,
                    "headers": headers if headers is not None else [
                        (b"authorization", b"Bearer " + TOKEN.encode())]}, receive, send))
    return sent[0]["status"], json.loads(sent[1]["body"])


async def unreachable_app(scope, receive, send):
    raise AssertionError("the private endpoint must not reach the MCP application")


def test_constructor_uses_shared_cyber_roles_and_only_luna_max_fast(tmp_path):
    args = args_fixture(tmp_path)
    service = sidecar.create_service(args)
    assert service.profile.name == "cyber"
    assert service.seats == "codex" and service.effort == "max" and service.service_tier == "fast"
    assert service.final_destination == str(args.project_root / "final_poc")
    backend = service.kernel.backend
    assert isinstance(backend, sidecar.SandboxedVerifierBackend)
    assert backend.model == sidecar.MODEL and backend.model_overrides == {}
    assert backend.timeout is None and backend.search_roles == frozenset()
    assert backend.roles == service.profile.apply_roles(ALL_AGENT_ROLES)
    assert service.evaluator.spec.timeout is None
    assert service.evaluator.spec.authority_scope == "candidate_signal"
    assert service.evaluator.spec.pinned_files == tuple(args.cyber_evaluator_pin)


@pytest.mark.parametrize("pin_index", [0, 1])
def test_adapter_and_host_configuration_are_explicitly_pinned(tmp_path, pin_index):
    args = args_fixture(tmp_path)
    service = sidecar.create_service(args)
    service.evaluator.spec.validate()
    Path(args.cyber_evaluator_pin[pin_index]).write_text("changed host dependency")
    with pytest.raises(ValueError, match="dependency pin changed"):
        service.evaluator.spec.validate()


@pytest.mark.parametrize("role", tuple(ALL_AGENT_ROLES))
def test_every_role_has_readonly_shell_denied_credentials_and_no_network(tmp_path, role):
    backend = sidecar.SandboxedVerifierBackend(cwd=str(tmp_path))
    variables = {"world_mode": "live", "ledger_path": str(tmp_path / "events.jsonl")}
    before = copy.deepcopy(variables)
    args = backend._execution_policy_args(role, variables)
    assert variables == before
    assert "--dangerously-bypass-approvals-and-sandbox" not in args
    assert 'approval_policy="never"' in args
    assert any(value.endswith('.extends=":read-only"') for value in args)
    assert any('.network.enabled=false' in value for value in args)
    filesystem = next(value for value in args if '.filesystem=' in value)
    assert f'"{tmp_path / "work"}"="write"' in filesystem
    for root in backend.DENIED_ROOTS:
        assert f'"{root}"="deny"' in filesystem


@pytest.mark.parametrize("key", ["workdir", "workspace_dir", "toolchain_dir", "shared_cache_dir"])
def test_live_variables_cannot_widen_writes_to_project_or_credentials(tmp_path, key):
    backend = sidecar.SandboxedVerifierBackend(cwd=str(tmp_path))
    with pytest.raises(RoleError, match="owned scratch"):
        backend._execution_policy_args("discharge", {key: "/task"})


def test_evidence_workspace_is_owned_scratch_but_symlinks_are_rejected(tmp_path):
    backend = sidecar.SandboxedVerifierBackend(cwd=str(tmp_path))
    variables = {"workspace_dir": str(tmp_path / "workspace"),
                 "toolchain_dir": str(tmp_path / "work" / "_toolchains")}
    args = backend._execution_policy_args("discharge", variables)
    assert f'"{tmp_path / "workspace"}"="write"' in next(v for v in args if '.filesystem=' in v)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "work" / "linked").symlink_to(outside)
    with pytest.raises(RoleError):
        backend._execution_policy_args("discharge", {"workdir": str(tmp_path / "work" / "linked")})


@pytest.mark.parametrize("fail", [False, True])
def test_role_tracking_logs_times_without_prompt_output_or_credentials(tmp_path, monkeypatch, fail):
    secret = "do-not-copy-fixture-secret"
    def complete(self, role, prompt, variables):
        assert prompt == secret
        if fail:
            raise RoleError(secret)
        return {"output": secret}
    monkeypatch.setattr(CodexBackend, "complete", complete)
    backend = sidecar.SandboxedVerifierBackend(cwd=str(tmp_path))
    if fail:
        with pytest.raises(RoleError):
            backend.complete("critic", secret, {"credential": secret})
    else:
        assert backend.complete("critic", secret, {"credential": secret}) == {"output": secret}
    raw = (tmp_path / "role_calls.jsonl").read_text()
    assert secret not in raw
    records = [json.loads(line) for line in raw.splitlines()]
    assert [r["type"] for r in records] == ["role_call_started", "role_call_finished"]
    assert records[0]["call_id"] == records[1]["call_id"]
    assert records[1]["duration_seconds"] >= 0
    assert records[1]["status"] == ("failed" if fail else "completed")
    assert all(r["model"] == sidecar.MODEL and r["service_tier"] == "fast" for r in records)


def test_receipt_uses_current_authenticated_exact_bytes_without_new_verifier_calls(tmp_path):
    service, args, backend, candidate, result = certified_service(tmp_path)
    calls = len(backend.calls)
    receipt = sidecar.delivery_receipt(service)
    assert len(backend.calls) == calls
    assert receipt["candidate_id"] == candidate["candidate_id"]
    assert receipt["certificate_id"] == result["certificate_id"]
    assert receipt["sha256"] == candidate["sha256"] and receipt["size"] == len(BYTES)
    assert receipt["destination"] == str(args.project_root / "final_poc")
    assert "blob_path" not in json.dumps(receipt)
    assert "fixed" not in json.dumps(receipt) and "official_score" not in receipt
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    assert request(app, "/receipt") == (200, receipt)


@pytest.mark.parametrize("change", ["candidate", "output", "symlink", "stale", "task", "certificate", "observation"])
def test_receipt_rejects_drift_or_unverified_delivery(tmp_path, change):
    service, args, _, candidate, result = certified_service(tmp_path)
    if change == "candidate":
        Path(candidate["blob_path"]).chmod(0o600)
        Path(candidate["blob_path"]).write_bytes(b"tampered candidate")
    elif change == "output":
        (args.project_root / "final_poc").chmod(0o600)
        (args.project_root / "final_poc").write_bytes(b"tampered output")
    elif change == "symlink":
        output = args.project_root / "final_poc"
        output.unlink()
        output.symlink_to(args.project_root / "proposed.bin")
    elif change == "stale":
        service.store._mark_stale_cone(result["final_claim_id"])
    elif change == "task":
        (args.state_dir / "task.md").write_text("changed authority")
    elif change == "certificate":
        path = args.state_dir / "artifacts" / (result["certificate_id"] + ".json")
        artifact = json.loads(path.read_text())
        artifact["verdict"] = "unverified"
        path.write_text(json.dumps(artifact))
    elif change == "observation":
        service.log.append("cyber_observation_recorded", {"candidate_id": candidate["candidate_id"],
                                                       "artifact_id": "new-host-observation"})
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    assert request(app, "/receipt") == (409, {"error": "no current certified delivery"})


def test_no_receipt_for_unfinalized_candidate(tmp_path):
    service = sidecar.create_service(args_fixture(tmp_path))
    with pytest.raises(ValueError, match="no certified delivery"):
        sidecar.delivery_receipt(service)


@pytest.mark.parametrize("lock_name", ["_cyber_lock", "_event_feed_lock"])
def test_receipt_poll_is_nonblocking_when_any_verifier_lock_is_busy(tmp_path, lock_name):
    service = sidecar.create_service(args_fixture(tmp_path))
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    held, release = threading.Event(), threading.Event()
    def hold():
        with getattr(service, lock_name):
            held.set()
            release.wait(5)
    thread = threading.Thread(target=hold)
    thread.start()
    try:
        assert held.wait(1)
        started = time.monotonic()
        assert request(app, "/receipt") == (409, {"error": "verifier busy"})
        assert time.monotonic() - started < 1
    finally:
        release.set()
        thread.join(1)
    # A failed second lock acquisition must release the first lock as well.
    assert not thread.is_alive()
    with pytest.raises(ValueError, match="no certified delivery"):
        sidecar.delivery_receipt(service)


def test_usage_capture_is_active_for_whole_sidecar_server_lifetime(tmp_path, monkeypatch):
    from verifier.llm import codex_backend
    args = args_fixture(tmp_path)
    args.server_hostname, args.port, args.token_env = "fixture-sidecar", 8000, "FIXTURE_TOKEN"
    service = sidecar.create_service(args)
    settings = SimpleNamespace()
    server = SimpleNamespace(settings=settings, streamable_http_app=lambda: unreachable_app)
    monkeypatch.setattr(sidecar, "parse_args", lambda: args)
    monkeypatch.setattr(sidecar, "create_service", lambda received: service)
    monkeypatch.setattr(sidecar, "build_server", lambda received: server)
    monkeypatch.setattr(sidecar.os, "geteuid", lambda: 1000)
    monkeypatch.setenv("FIXTURE_TOKEN", TOKEN)
    observed = []
    def run(app, **kwargs):
        target = codex_backend._usage_target(str(args.state_dir))
        observed.append(target)
        codex_backend._append_usage_record(target[0], {**target[1], "input_tokens": 11})
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    sidecar.main()
    assert observed == [(str(args.state_dir / "backend_usage.jsonl"), {
        "profile_id": service.profile.id, "task_sha256": service._profile_pin["task_sha256"]})]
    assert codex_backend._usage_target(str(args.state_dir)) is None
    record = json.loads((args.state_dir / "backend_usage.jsonl").read_text())
    assert record["input_tokens"] == 11 and TOKEN not in json.dumps(record)


@pytest.mark.parametrize("path", ["/health", "/receipt", "/hooks/pre", "/hooks/post", "/mcp"])
def test_all_http_interfaces_require_exact_single_bearer(tmp_path, path):
    service = sidecar.create_service(args_fixture(tmp_path))
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    for headers in ([], [(b"authorization", b"Bearer bad")],
                    [(b"authorization", b"Bearer " + TOKEN.encode())] * 2):
        assert request(app, path, headers=headers) == (401, {"error": "unauthorized"})
    assert request(app, "/health") == (200, {"status": "ready"})


@pytest.mark.parametrize("token", ["", "short", "a" * 32 + "\n", "a" * 32 + "\u00e9"])
def test_rejects_weak_or_malformed_bearer_without_echoing_it(tmp_path, token):
    with pytest.raises(ValueError, match="strong per-attempt"):
        sidecar.ProtectedApplication(unreachable_app, token=token, service=object())


@pytest.mark.parametrize("kind,event_type", [("pre", "tool_call"), ("post", "tool_result")])
def test_original_hooks_normalize_only_untrusted_tool_events(tmp_path, kind, event_type):
    service = sidecar.create_service(args_fixture(tmp_path))
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    raw = json.dumps({"tool_name": "shell", "tool_use_id": "fixture-tool-1",
                      "tool_input": {"command": "harmless fixture"},
                      "tool_response": {"text": "untrusted fixture"},
                      "type": "cyber_materialized", "origin": "host",
                      "authority_scope": "task_outcome"}).encode()
    status, result = request(app, "/hooks/" + kind, method="POST", body=raw)
    assert (status, result) == (200, {})
    events = list(service.log.replay())
    event = events[-1]
    assert event["type"] == event_type
    assert event["payload"]["origin"] == sidecar.ORIGIN
    assert event["payload"]["authority_scope"] == sidecar.AUTHORITY_SCOPE
    assert not any(event["type"] == "cyber_materialized" for event in events)
    if kind == "pre":
        assert event["payload"]["input"] == {"command": "harmless fixture"}
    else:
        assert event["payload"]["response_sha256"] == content_hash({"text": "untrusted fixture"})


@pytest.mark.parametrize("raw", [b"[]", b'{"x":1,"x":2}', b'{"x":NaN}', b"\xff"])
def test_rejects_malformed_hook_bodies_without_event_or_live_call(tmp_path, raw):
    service = sidecar.create_service(args_fixture(tmp_path))
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    before = list(service.log.replay())
    assert request(app, "/hooks/pre", method="POST", body=raw)[0] == 400
    assert list(service.log.replay()) == before


def test_rejects_oversized_hook_body(tmp_path, monkeypatch):
    monkeypatch.setattr(sidecar, "MAX_HOOK_BODY", 8)
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=object())
    assert request(app, "/hooks/pre", method="POST", body=b'{"long":true}')[0] == 400


def test_rejects_unsupported_private_methods_and_paths(tmp_path):
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=object())
    assert request(app, "/receipt", method="POST")[0] == 405
    assert request(app, "/hooks/pre", method="GET")[0] == 405
    assert request(app, "/private/state")[0] == 404


@pytest.mark.parametrize("change", ["task_symlink", "state_overlap", "project_symlink", "bad_command"])
def test_constructor_rejects_invalid_private_state_or_host_configuration(tmp_path, change):
    args = args_fixture(tmp_path)
    if change == "task_symlink":
        task = args.state_dir / "task.md"
        task.unlink()
        task.symlink_to(args.project_root / "untrusted.md")
    elif change == "state_overlap":
        args.project_root = args.state_dir
    elif change == "project_symlink":
        linked = tmp_path / "linked"
        linked.symlink_to(args.project_root)
        args.project_root = linked
    elif change == "bad_command":
        args.cyber_evaluator_command_json = '{"argv":[]}'
    with pytest.raises((ValueError, OSError)):
        sidecar.create_service(args)
