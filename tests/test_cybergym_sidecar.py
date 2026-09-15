"""CyberGym transport/permission fixtures: no model or live evaluator calls."""
import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import threading
import time
import tomllib
from types import SimpleNamespace

import pytest

from experiments.cybergym import model_config, sidecar
from experiments.cybergym.policy import SearchPolicyEvidenceError
from verifier.core.model import content_hash
from verifier.llm.agents import ALL_AGENT_ROLES, AgentBackend
from verifier.llm.codex_backend import CodexBackend
from verifier.llm.schema import FixtureBackend, RoleError
from verifier.mcp_server import Service, _default_kernel


TOKEN = "fixture-only-" + "x" * 48
BYTES = b"immutable fixture candidate\x00\xff"


@pytest.fixture(autouse=True)
def prohibit_real_model_cli(monkeypatch):
    original = sidecar.subprocess.Popen
    class GuardedPopen(original):
        def __init__(self, command, *args, **kwargs):
            if not isinstance(command, str) and any(isinstance(part, str) and os.path.basename(part) == "codex" for part in command):
                raise AssertionError("real Codex invocation is forbidden in these fixtures")
            super().__init__(command, *args, **kwargs)
    monkeypatch.setattr(sidecar.subprocess, "Popen", GuardedPopen)


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


def test_constructor_uses_shared_cyber_roles_and_the_pinned_claude_seats(tmp_path):
    args = args_fixture(tmp_path)
    service = sidecar.create_service(args)
    assert service.profile.name == "cyber"
    assert service.seats == "sdk" and service.effort is None and service.service_tier is None
    assert service.final_destination == str(args.project_root / "final_poc")
    backend = service.kernel.backend
    assert isinstance(backend, sidecar.ClaudeVerifierBackend)
    assert backend.model == sidecar.MODEL and backend.transcripts == []
    assert backend.message_observer is not None
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
    assert "--ignore-user-config" in args
    assert 'approval_policy="never"' in args
    assert any(value.endswith('.extends=":read-only"') for value in args)
    assert any('.network.enabled=false' in value for value in args)
    context = [value for value in args
               if value.startswith(("model_context_window=", "model_auto_compact_token_limit="))]
    assert context == [value for value in model_config.context_cli_args() if value != "-c"]
    filesystem = next(value for value in args if '.filesystem=' in value)
    grants = tomllib.loads("grants=" + filesystem.split("=", 1)[1])["grants"]
    def effective(path):
        return next((grants[str(parent)] for parent in (path, *path.parents)
                     if str(parent) in grants), grants[":root"])
    assert effective(tmp_path / "work") == ("deny" if role in ("obligations", "critic", "oracles", "sufficiency") else "write")
    assert effective(tmp_path / "verifier-home") == "deny"
    if role in ("obligations", "critic", "oracles", "sufficiency"):
        assert str(tmp_path) not in grants and grants[":root"] == "deny"
        for authority in ("task.md", "baseline", "oracle_catalog"):
            assert grants[str(tmp_path / authority)] == "read"
        for hidden in ("cybergym-observations", "hook_feed.jsonl", "artifacts", "cyber",
                       "workspace", "record", "source_evidence"):
            assert effective(tmp_path / hidden) == "deny"
            assert str(tmp_path / hidden) not in grants
        assert effective(tmp_path / "future-private-transport") == "deny"
        assert effective(tmp_path / "_blind" / "another-role") == "deny"
        assert grants[str(tmp_path / "_blind" / role)] == "write"
        assert grants["/task"] == "deny"
        env_arg = next(value for value in args if value.startswith("shell_environment_policy.set="))
        env = tomllib.loads(env_arg)["shell_environment_policy"]["set"]
        assert env["HOME"] == str(tmp_path / "_blind" / role / "home")
        assert Path(env["TMPDIR"]).is_dir()
        assert backend._execution_cwd(role, variables) == str(tmp_path / "_blind" / role)
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
    assert all(r["model"] == sidecar.MODEL and r["service_tier"] == sidecar.SERVICE_TIER
               for r in records)
    assert all(r["model_context_window"] == sidecar.CONTEXT_WINDOW
               and r["model_auto_compact_token_limit"] == sidecar.AUTO_COMPACT_TOKEN_LIMIT
               for r in records)


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


def test_claude_backend_writes_usage_rows_without_the_codex_ledger(tmp_path, monkeypatch):
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
        # The claude backend appends usage rows per role call; no process-wide
        # Codex capture is installed for the server lifetime.
        observed.append(codex_backend._usage_target(str(args.state_dir)))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    sidecar.main()
    assert observed == [None]
    assert not (args.state_dir / "backend_usage.jsonl").exists()


@pytest.mark.parametrize("path", ["/health", "/receipt", "/policy", "/policy/seal", "/hooks/pre", "/hooks/post", "/mcp"])
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


def clean_search_policy():
    return {"schema": "gv-search-policy-v1", "web_search": "disabled", "violation": False,
            "violation_source": None, "event_type": None, "sealed": False, "evidence_complete": True}


class ToolUseBlock:
    """SDK-shaped tool_use block (claude_agent_sdk.types.ToolUseBlock)."""
    def __init__(self, name):
        self.name = name


class ServerToolUseBlock(ToolUseBlock):
    """SDK-shaped server_tool_use block (claude_agent_sdk.types.ServerToolUseBlock)."""


def assistant_message(*blocks):
    return SimpleNamespace(content=list(blocks))


def observe_web_search(backend):
    backend._observe_message(assistant_message(ToolUseBlock("WebSearch")))


def policy_envelope(app, policy):
    return {"schema": "gv-search-policy-response-v1", "instance_id": app.policy_instance_id, "policy": policy}


@pytest.mark.parametrize("event_type", ["item.started", "item.updated", "item.completed"])
def test_actual_backend_cli_phase_latches_search_before_failed_diagnostic_logging(tmp_path, event_type):
    (tmp_path / "role_calls.jsonl").mkdir()  # Deterministic disk path failure, not a fake observer.
    backend = sidecar.SandboxedVerifierBackend(cwd=str(tmp_path))
    assert backend.search_policy_guard.snapshot() == clean_search_policy()
    backend._phase("a" * 32, 1, "cli_event", event_type=event_type, item_kind="web_search")
    assert backend._diagnostic_failures == 1
    expected = {**clean_search_policy(), "violation": True,
                "violation_source": "verifier", "event_type": event_type}
    assert backend.search_policy_guard.snapshot() == expected
    backend._phase("b" * 32, 1, "cli_event", event_type="item.completed", item_kind="command_execution")
    assert backend.search_policy_guard.snapshot() == expected
    assert TOKEN not in json.dumps(expected)


@pytest.mark.parametrize("phase,event_type,item_kind", [
    ("heartbeat", "item.completed", "web_search"),
    ("cli_event", "turn.completed", "web_search"),
    ("cli_event", "item.completed", "mcp_tool_call"),
    ("cli_event", "item.updated", "command_execution"),
    ("cli_event", "item.completed", "web_search_preview"),
])
def test_backend_nonmatching_typed_phase_does_not_latch_search(tmp_path, phase, event_type, item_kind):
    backend = sidecar.SandboxedVerifierBackend(cwd=str(tmp_path))
    backend._phase("a" * 32, 1, phase, event_type=event_type, item_kind=item_kind)
    backend.search_policy_guard.observe_backend_phase({"text": "item.completed web_search", "query": "fixture prose"})
    assert backend.search_policy_guard.snapshot() == clean_search_policy()


def test_service_factory_and_repeated_kernel_instances_share_one_private_guard(tmp_path):
    service = sidecar.create_service(args_fixture(tmp_path))
    first = service.kernel.backend
    second = sidecar.kernel_factory(service).backend
    assert first is not second
    assert first.search_policy_guard is second.search_policy_guard is service.search_policy_guard
    observe_web_search(first)
    assert second.search_policy_guard.snapshot()["violation"] is True


def test_direct_service_kernel_factory_also_installs_one_shared_guard(tmp_path):
    args = args_fixture(tmp_path)
    service = Service(str(args.state_dir), profile="cyber", project_root=str(args.project_root),
                      model=sidecar.MODEL, seats="sdk", kernel_factory=sidecar.kernel_factory)
    second = sidecar.kernel_factory(service).backend
    assert service.kernel.backend.search_policy_guard is second.search_policy_guard is service.search_policy_guard


def test_guard_is_available_during_early_service_kernel_construction(tmp_path, monkeypatch):
    observed = []
    def early_constructor(state_dir, *, kernel_factory, **kwargs):
        service = SimpleNamespace(state_dir=state_dir, project_root=kwargs["project_root"],
                                  seat_runtime=kwargs["seat_runtime"])
        service.kernel = kernel_factory(service)
        observed.append(service.kernel.backend.search_policy_guard)
        return service
    monkeypatch.setattr(sidecar, "Service", early_constructor)
    monkeypatch.setattr(sidecar, "_default_kernel", lambda service, **kwargs: SimpleNamespace(backend=kwargs["backend"]))
    service = sidecar.create_service(args_fixture(tmp_path))
    assert len(observed) == 1 and observed[0] is service.search_policy_guard


def test_two_services_and_standalone_backends_do_not_share_latches(tmp_path):
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = sidecar.create_service(args_fixture(first_root))
    second = sidecar.create_service(args_fixture(second_root))
    standalone = sidecar.SandboxedVerifierBackend(cwd=str(tmp_path))
    assert len({id(first.search_policy_guard), id(second.search_policy_guard), id(standalone.search_policy_guard)}) == 3
    observe_web_search(first.kernel.backend)
    assert first.search_policy_guard.snapshot()["violation"] is True
    assert second.search_policy_guard.snapshot() == standalone.search_policy_guard.snapshot() == clean_search_policy()


@pytest.mark.parametrize("lock_name", ["_cyber_lock", "_event_feed_lock"])
def test_authenticated_policy_snapshot_does_not_wait_for_verifier_graph_locks(tmp_path, lock_name):
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
        assert request(app, "/policy") == (200, policy_envelope(app, clean_search_policy()))
        assert time.monotonic() - started < 1
    finally:
        release.set()
        thread.join(1)
    assert not thread.is_alive()


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "OPTIONS"])
def test_policy_endpoint_is_authenticated_get_only(tmp_path, method):
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=object())
    assert request(app, "/policy", method=method) == (405, {"error": "method not allowed"})
    assert request(app, "/policy", method=method, headers=[]) == (401, {"error": "unauthorized"})


@pytest.mark.parametrize("service", [object(), SimpleNamespace(search_policy_guard=None),
                                      SimpleNamespace(search_policy_guard=object())])
def test_policy_endpoint_missing_or_invalid_guard_fails_closed(service):
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    assert request(app, "/policy") == (503, {"error": "search policy unavailable"})


def test_policy_snapshot_reports_latch_without_becoming_a_gv_verdict(tmp_path):
    service = sidecar.create_service(args_fixture(tmp_path))
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    before = list(service.log.replay())
    observe_web_search(service.kernel.backend)
    status, response = request(app, "/policy")
    assert status == 200 and response == policy_envelope(app, {**clean_search_policy(), "violation": True,
        "violation_source": "verifier", "event_type": "item.completed"})
    snapshot = response["policy"]
    assert "verdict" not in snapshot and "score" not in snapshot and "query" not in snapshot
    assert list(service.log.replay()) == before


def test_policy_response_generation_is_stable_per_app_and_changes_on_replacement(tmp_path):
    service = sidecar.create_service(args_fixture(tmp_path))
    first = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    initial = request(first, "/policy")
    assert initial == request(first, "/policy") == (200, policy_envelope(first, clean_search_policy()))
    assert len(first.policy_instance_id) == 32 and set(first.policy_instance_id) <= set("0123456789abcdef")
    second = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    assert second.policy_instance_id != first.policy_instance_id
    assert request(second, "/policy")[1]["instance_id"] == second.policy_instance_id


def test_active_complete_prevents_seal_until_entire_backend_call_returns(tmp_path, monkeypatch):
    service = sidecar.create_service(args_fixture(tmp_path))
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    held, release = threading.Event(), threading.Event()
    result, errors = [], []
    def complete(self, role, prompt, variables):
        held.set()
        assert release.wait(5)  # This owned synthetic fixture only, not a role deadline.
        return {"ok": True}
    monkeypatch.setattr(AgentBackend, "complete", complete)
    def invoke():
        try:
            result.append(service.kernel.backend.complete("critic", "fixture", {}))
        except Exception as error:
            errors.append(error)
    worker = threading.Thread(target=invoke)
    worker.start()
    try:
        assert held.wait(1)
        assert request(app, "/policy/seal", method="POST") == (200, policy_envelope(app, clean_search_policy()))
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive() and not errors and result == [{"ok": True}]
    sealed = {**clean_search_policy(), "sealed": True}
    assert request(app, "/policy/seal", method="POST") == (200, policy_envelope(app, sealed))
    assert request(app, "/policy") == (200, policy_envelope(app, sealed))


@pytest.mark.parametrize("state", ["queued", "running"])
def test_pending_verification_job_prevents_premature_policy_seal(tmp_path, monkeypatch, state):
    service = sidecar.create_service(args_fixture(tmp_path))
    monkeypatch.setattr(service, "verification_jobs", lambda: {"schema": "gv-verification-progress-v1",
        "jobs": [{"state": state, "query": "do-not-copy-fixture-metadata"}]})
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    assert request(app, "/policy/seal", method="POST") == (200, policy_envelope(app, clean_search_policy()))


def test_terminal_jobs_seal_irreversibly_and_new_complete_never_calls_super(tmp_path, monkeypatch):
    service = sidecar.create_service(args_fixture(tmp_path))
    monkeypatch.setattr(service, "verification_jobs", lambda: {"schema": "gv-verification-progress-v1",
        "jobs": [{"state": state} for state in ("completed", "failed", "interrupted")]})
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    sealed = {**clean_search_policy(), "sealed": True}
    assert request(app, "/policy/seal", method="POST") == (200, policy_envelope(app, sealed))
    assert request(app, "/policy/seal", method="POST") == (200, policy_envelope(app, sealed))
    def forbidden(*args, **kwargs):
        pytest.fail("sealed backend must never enter the model complete method")
    monkeypatch.setattr(AgentBackend, "complete", forbidden)
    with pytest.raises(SearchPolicyEvidenceError, match="after_policy_seal"):
        service.kernel.backend.complete("critic", "fixture", {})
    assert service.search_policy_guard.snapshot() == sealed


@pytest.mark.parametrize("failed_log_call", [1, 2])
def test_diagnostic_exception_always_releases_active_execution(tmp_path, monkeypatch, failed_log_call):
    backend = sidecar.SandboxedVerifierBackend(cwd=str(tmp_path))
    calls = []
    def diagnostic(record):
        calls.append(record["type"])
        if len(calls) == failed_log_call:
            raise RuntimeError("synthetic diagnostic unavailable")
    monkeypatch.setattr(backend, "_diagnostic", diagnostic)
    monkeypatch.setattr(CodexBackend, "complete", lambda *_: {"ok": True})
    with pytest.raises(RuntimeError, match="synthetic diagnostic unavailable"):
        backend.complete("critic", "fixture", {})
    assert backend.search_policy_guard.try_seal() is True


@pytest.mark.parametrize("progress", [None, {}, {"schema": "wrong", "jobs": []},
    {"schema": "gv-verification-progress-v1", "jobs": None},
    {"schema": "gv-verification-progress-v1", "jobs": [None]},
    {"schema": "gv-verification-progress-v1", "jobs": [{"state": True}]},
    {"schema": "gv-verification-progress-v1", "jobs": [{"state": "unknown"}]}])
def test_invalid_verification_progress_cannot_authorize_seal(tmp_path, monkeypatch, progress):
    service = sidecar.create_service(args_fixture(tmp_path))
    monkeypatch.setattr(service, "verification_jobs", lambda: progress)
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    assert request(app, "/policy/seal", method="POST") == (503, {"error": "search policy seal unavailable"})
    assert service.search_policy_guard.snapshot() == clean_search_policy()


def test_late_pending_job_after_irreversible_seal_fails_closed(tmp_path, monkeypatch):
    service = sidecar.create_service(args_fixture(tmp_path))
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=service)
    assert request(app, "/policy/seal", method="POST")[1]["policy"]["sealed"] is True
    monkeypatch.setattr(service, "verification_jobs", lambda: {"schema": "gv-verification-progress-v1", "jobs": [{"state": "queued"}]})
    assert request(app, "/policy/seal", method="POST") == (503, {"error": "search policy seal unavailable"})
    assert service.search_policy_guard.snapshot()["sealed"] is True


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE"])
def test_policy_seal_is_authenticated_post_only(method):
    app = sidecar.ProtectedApplication(unreachable_app, token=TOKEN, service=object())
    assert request(app, "/policy/seal", method=method) == (405, {"error": "method not allowed"})
    assert request(app, "/policy/seal", method=method, headers=[]) == (401, {"error": "unauthorized"})
    assert request(app, "/policy/seal", method="POST", headers=[]) == (401, {"error": "unauthorized"})
    assert request(app, "/policy/seal", method="POST") == (503, {"error": "search policy unavailable"})


def test_real_python_oversize_cli_event_marks_policy_evidence_incomplete_without_raw_log(tmp_path, monkeypatch):
    backend = sidecar.SandboxedVerifierBackend(cwd=str(tmp_path))
    backend.phase_heartbeat_seconds = 0.02
    program = """
import json,sys
print(json.dumps({'type':'item.completed','item':{'type':'web_search','query':'do-not-retain-synthetic-query'*100000}}),flush=True)
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'synthetic final'}}),flush=True)
print(json.dumps({'type':'turn.completed','usage':{}}),flush=True)
"""
    def synthetic_complete(self, role, prompt, variables):
        import subprocess
        with open(os.devnull, "rb") as devnull:
            process = self._run_observed_exec([sys.executable, "-u", "-c", program], str(tmp_path), devnull, "a" * 32, 1)
        assert isinstance(process, subprocess.CompletedProcess) and process.returncode == 0
        return {"ok": True}
    monkeypatch.setattr(CodexBackend, "complete", synthetic_complete)
    assert backend.complete("critic", "fixture", {}) == {"ok": True}
    snapshot = backend.search_policy_guard.snapshot()
    assert snapshot["evidence_complete"] is False and snapshot["violation"] is False
    raw = (tmp_path / "role_calls.jsonl").read_text()
    records = [json.loads(line) for line in raw.splitlines()]
    returned = next(record for record in records if record.get("phase") == "exec_returned")
    assert returned["unparsed_stdout_lines"] > 0 and returned["reader_failed"] is False
    assert returned["last_cli_event"] == "turn.completed"
    assert "do-not-retain-synthetic-query" not in raw and "synthetic final" not in raw
    assert backend.search_policy_guard.try_seal() is True


def claude_sdk_fixture(monkeypatch, transcript, *, fail=False):
    """Mock the SDK layer: complete appends terminal transcript rows."""
    def complete(self, role, prompt, variables):
        if transcript is not None:
            self.transcripts.append(transcript)
        if fail:
            raise RoleError("sdk fixture failure")
        return {"ok": True}
    monkeypatch.setattr(AgentBackend, "complete", complete)


CLAUDE_USAGE = {"input_tokens": 100, "cache_creation_input_tokens": 30,
    "cache_read_input_tokens": 20, "output_tokens": 10,
    "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
    "output_tokens_details": {"thinking_tokens": 7}}


@pytest.mark.parametrize("fail", [False, True])
def test_claude_backend_role_tracking_and_usage_rows_share_one_call(tmp_path, monkeypatch, fail):
    claude_sdk_fixture(monkeypatch, {"role": "critic", "usage": CLAUDE_USAGE}, fail=fail)
    backend = sidecar.ClaudeVerifierBackend(cwd=str(tmp_path))
    if fail:
        with pytest.raises(RoleError):
            backend.complete("critic", "fixture prompt", {})
    else:
        assert backend.complete("critic", "fixture prompt", {}) == {"ok": True}
    records = [json.loads(line) for line in (tmp_path / "role_calls.jsonl").read_text().splitlines()]
    assert [row["type"] for row in records] == ["role_call_started", "role_call_finished"]
    assert records[0]["call_id"] == records[1]["call_id"]
    assert records[1]["status"] == ("failed" if fail else "completed")
    assert all(row["model"] == sidecar.MODEL and row["service_tier"] is None
               and row["model_context_window"] == sidecar.CONTEXT_WINDOW
               and row["model_auto_compact_token_limit"] is None for row in records)
    usage = [json.loads(line) for line in (tmp_path / "backend_usage.jsonl").read_text().splitlines()]
    assert len(usage) == 1
    row = usage[0]
    assert row["call_id"] == records[0]["call_id"] and row["role"] == "critic"
    assert row["model"] == sidecar.MODEL and row["service_tier"] is None
    assert row["returncode"] == (1 if fail else 0)
    assert type(row["started_at_unix_ns"]) is int and type(row["finished_at_unix_ns"]) is int
    assert row["started_at_unix_ns"] <= row["finished_at_unix_ns"]
    assert row["duration_seconds"] >= 0
    assert row["input_tokens"] == 100 and row["cached_input_tokens"] == 50
    assert row["output_tokens"] == 10
    assert row["reasoning_tokens"] == 0 and row["reasoning_output_tokens"] == 7
    assert "fixture prompt" not in json.dumps(records) + json.dumps(usage)


@pytest.mark.parametrize("usage", [None, "opaque", {}, {"input_tokens": -1, "output_tokens": "x"}])
def test_claude_backend_usage_row_is_defensive_about_missing_counters(tmp_path, monkeypatch, usage):
    claude_sdk_fixture(monkeypatch, {"role": "critic", "usage": usage})
    backend = sidecar.ClaudeVerifierBackend(cwd=str(tmp_path))
    assert backend.complete("critic", "fixture", {}) == {"ok": True}
    row = json.loads((tmp_path / "backend_usage.jsonl").read_text())
    assert row["input_tokens"] == 0 and row["cached_input_tokens"] == 0
    assert row["output_tokens"] == 0 and row["reasoning_output_tokens"] == 0


def test_claude_backend_without_new_transcript_writes_no_usage_row(tmp_path, monkeypatch):
    claude_sdk_fixture(monkeypatch, None)
    backend = sidecar.ClaudeVerifierBackend(cwd=str(tmp_path))
    assert backend.complete("critic", "fixture", {}) == {"ok": True}
    assert not (tmp_path / "backend_usage.jsonl").exists()
    assert backend._diagnostic_failures == 0


def test_claude_backend_usage_disk_failure_never_decides_a_call(tmp_path, monkeypatch):
    (tmp_path / "backend_usage.jsonl").mkdir()  # Deterministic disk path failure.
    claude_sdk_fixture(monkeypatch, {"role": "critic", "usage": CLAUDE_USAGE})
    backend = sidecar.ClaudeVerifierBackend(cwd=str(tmp_path))
    assert backend.complete("critic", "fixture", {}) == {"ok": True}
    assert backend._diagnostic_failures == 1
    assert (tmp_path / "role_calls.jsonl").is_file()


@pytest.mark.parametrize("block", [ToolUseBlock("WebSearch"), ToolUseBlock("WebFetch"),
                                   ServerToolUseBlock("WebSearch")])
def test_claude_assistant_tool_use_blocks_latch_verifier_search(tmp_path, block):
    backend = sidecar.ClaudeVerifierBackend(cwd=str(tmp_path))
    backend._observe_message(assistant_message(block))
    snapshot = backend.search_policy_guard.snapshot()
    assert snapshot["violation"] is True
    assert snapshot["violation_source"] == "verifier" and snapshot["event_type"] == "item.completed"


@pytest.mark.parametrize("message", [
    assistant_message(ToolUseBlock("Read")),
    assistant_message(),
    SimpleNamespace(content="plain text"),
    SimpleNamespace(subtype="init", data={}),
    {"type": "system", "subtype": "init"},
])
def test_claude_non_search_messages_do_not_latch(tmp_path, message):
    backend = sidecar.ClaudeVerifierBackend(cwd=str(tmp_path))
    backend._observe_message(message)
    assert backend.search_policy_guard.snapshot() == clean_search_policy()


def test_claude_stream_error_marks_policy_evidence_incomplete(tmp_path):
    backend = sidecar.ClaudeVerifierBackend(cwd=str(tmp_path))
    backend._observe_message({"stream_error": True, "observed_messages": 3})
    snapshot = backend.search_policy_guard.snapshot()
    assert snapshot["evidence_complete"] is False and snapshot["violation"] is False
    backend._observe_message({"stream_error": False})
    assert backend.search_policy_guard.snapshot()["evidence_complete"] is False


def test_claude_message_observation_never_crashes_a_role_call(tmp_path):
    backend = sidecar.ClaudeVerifierBackend(cwd=str(tmp_path))

    class Exploding:
        @property
        def content(self):
            raise RuntimeError("synthetic sdk message failure")

    backend._observe_message(Exploding())
    assert backend.search_policy_guard.snapshot() == clean_search_policy()


def test_claude_backend_requires_a_real_search_policy_guard(tmp_path):
    with pytest.raises(TypeError, match="search policy guard"):
        sidecar.ClaudeVerifierBackend(cwd=str(tmp_path), search_policy_guard=object())
