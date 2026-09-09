"""The typed-adapter Judger transport is a no-tools transport, not a judge."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from verifier.core.events import EventLog
from verifier.core.model import BASIS_WORLD, Obligation
from verifier.engine import game as game_module
from verifier.engine.game import Attempt, BindingGame
from verifier.engine.typed_judger_packet import (
    TYPED_JUDGER_PACKET_PROTOCOL,
    TypedAdapterJudgerPacket,
    build_typed_adapter_judger_packet,
)
from verifier.execution.adapters import (
    AdapterRoutingDischarger,
    default_adapter_registry,
)
from verifier.execution.instruments import ArtifactStore
from verifier.llm import codex_backend as cb
from verifier.llm import typed_judger_backend as tjb
from verifier.llm.agents import AgentRole
from verifier.llm.codex_backend import (
    CodexBackend,
    _typed_judger_attempt_scope,
)
from verifier.llm.schema import RoleError
from verifier.llm.typed_judger_backend import (
    TYPED_JUDGER_POLICY_IDENTITY,
    _TypedJudgerCodexBackend,
    TypedJudgerTransportError,
    restricted_typed_judger_backend,
    typed_judger_policy_args,
    validate_typed_judger_stream,
)
from experiments.terminal_bench.observe_codex_attempts import (
    RUN_CODEX_PROCESS_SEAM,
    observe_codex_attempts,
)


YES = {
    "sufficient": "yes",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "fresh independent review",
}
NO = {
    "sufficient": "no",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "the method does not cover the whole obligation",
}


class _NoAgentFallback:
    def discharge(self, *_args, **_kwargs):
        raise AssertionError("fixture must use the deterministic adapter")


def _fixture(tmp_path):
    state = tmp_path / "state"
    state.mkdir(parents=True)
    (state / "task.md").write_text(
        "The delivered marker must exist.", encoding="utf-8")
    target = tmp_path / "marker.txt"
    target.write_text("alpha", encoding="utf-8")
    log = EventLog(str(state / "events.jsonl"))
    artifacts = ArtifactStore(str(state / "artifacts"))
    obligation = Obligation(
        "claim-1", "the delivered marker exists as a regular file",
        BASIS_WORLD, clause="marker must exist")
    method = {
        "procedure": "inspect the exact delivered address and file kind",
        "artifact": "a content-addressed adapter evidence envelope",
        "level": "deterministic",
        "basis": BASIS_WORLD,
        "strength": "decisive",
        "bridges": [],
        "adapter": "filesystem.exists",
        "adapter_version": 1,
        "adapter_arguments": json.dumps({
            "address": str(target), "kind": "file"}, sort_keys=True),
        "adapter_inputs": [{
            "ref": str(target), "provenance": "delivered",
            "role": "subject",
        }],
    }
    route = AdapterRoutingDischarger(
        _NoAgentFallback(), default_adapter_registry(), artifacts, log,
        str(state), live_world=True)
    result = route.discharge(
        {"id": obligation.claim_id, "kind": "final"}, obligation, method)
    packet = build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result,
        "(none — decisive method)")
    assert type(packet) is TypedAdapterJudgerPacket
    return state, target, log, artifacts, route, obligation, method, result, packet


def _stream(*middle):
    return "\n".join(json.dumps(event) for event in (
        {"type": "thread.started", "thread_id": "fresh-thread"},
        {"type": "turn.started"}, *middle,
        {"type": "turn.completed", "usage": {}},
    ))


def test_policy_denies_every_external_and_tool_channel(tmp_path, monkeypatch):
    args = typed_judger_policy_args(str(tmp_path))
    assert {"--strict-config", "--ignore-user-config", "--ignore-rules",
            "--ephemeral"} <= set(args)
    assert "--dangerously-bypass-approvals-and-sandbox" not in args
    values = [args[index + 1] for index, item in enumerate(args)
              if item == "-c"]
    config = tomllib.loads("\n".join(values))
    policy = config["permissions"]["gv-authority"]
    assert policy["filesystem"][":root"] == "deny"
    assert set(policy["filesystem"].values()) <= {"read", "deny"}
    assert policy["network"]["enabled"] is False
    assert config["tools"] == {"web_search": False}
    assert config["mcp_servers"] == {}
    for feature in (
        "shell_tool", "unified_exec", "code_mode_host", "view_image",
        "apps", "plugins", "remote_plugin", "browser_use",
        "browser_use_external", "browser_use_full_cdp_access",
        "computer_use", "in_app_browser", "image_generation",
        "enable_mcp_apps", "plugin_sharing", "tool_call_mcp_elicitation",
        "in_app_local_automation", "recommended_plugins",
    ):
        assert config["features"][feature] is False

    canonical_policy = json.dumps(
        typed_judger_policy_args("/__gv_typed_judger_scratch__"),
        ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    assert TYPED_JUDGER_POLICY_IDENTITY == hashlib.sha256(
        canonical_policy.encode("utf-8")).hexdigest()
    runtime_backend = _TypedJudgerCodexBackend()
    original_runtime_identity = runtime_backend.runtime_instruction_identity
    monkeypatch.setattr(tjb, "TYPED_JUDGER_POLICY_IDENTITY", "0" * 64)
    assert runtime_backend.runtime_instruction_identity \
        != original_runtime_identity


def test_stream_accepts_reasoning_and_message_sequence_but_rejects_tools():
    assert validate_typed_judger_stream(_stream(
        {"type": "item.completed", "item": {
            "id": "r", "type": "reasoning", "text": "private"}},
        {"type": "item.completed", "item": {
            "id": "m1", "type": "agent_message", "text": "progress"}},
        {"type": "item.completed", "item": {
            "id": "m2", "type": "agent_message", "text": "{}"}},
    )) == "{}"
    for item_type in (
        "command_execution", "file_change", "mcp_tool_call", "web_search",
        "image_generation", "collab_tool_call", "future_tool", None,
    ):
        with pytest.raises(TypedJudgerTransportError, match="tool"):
            validate_typed_judger_stream(_stream(
                {"type": "item.completed", "item": {
                    "id": "x", "type": item_type}},
            ))

    with pytest.raises(TypedJudgerTransportError, match="malformed"):
        validate_typed_judger_stream("\n".join((
            '{"type":"thread.started","type":"thread.started",'
            '"thread_id":"x"}',
            '{"type":"turn.started"}',
            '{"type":"item.completed","item":{"id":"m",'
            '"type":"agent_message","text":"{}"}}',
            '{"type":"turn.completed","usage":{}}',
        )))
    with pytest.raises(TypedJudgerTransportError, match="malformed"):
        validate_typed_judger_stream("\n".join((
            '{"type":"thread.started","thread_id":"x","extra":1e999}',
            '{"type":"turn.started"}',
            '{"type":"item.completed","item":{"id":"m",'
            '"type":"agent_message","text":"{}"}}',
            '{"type":"turn.completed","usage":{}}',
        )))


@pytest.mark.parametrize("middle", [
    (
        {"type": "item.completed", "item": {
            "id": "m1", "type": "agent_message", "text": "progress"}},
        {"type": "item.completed", "item": {
            "id": "r", "type": "reasoning", "text": "late"}},
    ),
    (
        {"type": "item.completed", "item": {
            "id": "m1", "type": "agent_message", "text": "progress"}},
        {"type": "item.completed", "item": {
            "id": "m2", "type": "agent_message", "text": "  "}},
    ),
    (
        {"type": "item.completed", "item": {
            "id": "same", "type": "agent_message", "text": "one"}},
        {"type": "item.completed", "item": {
            "id": "same", "type": "agent_message", "text": "two"}},
    ),
    (
        {"type": "item.started", "item": {
            "id": "m1", "type": "agent_message"}},
        {"type": "item.completed", "item": {
            "id": "m2", "type": "agent_message", "text": "two"}},
    ),
])
def test_message_sequence_rejects_ambiguity_after_first_message(middle):
    with pytest.raises(TypedJudgerTransportError):
        validate_typed_judger_stream(_stream(*middle))


def test_context_preserves_seat_identity_and_uses_fresh_scratch(tmp_path):
    backend = CodexBackend(
        cwd=str(tmp_path), model="gpt-5.6-luna;judger=gpt-5.6-luna",
        effort="medium", service_tier="standard", timeout=321,
        compact_inspection=True, judger_prefetch=True)
    runtime_identities = []
    execution_cwds = []
    with restricted_typed_judger_backend(backend) as restricted:
        assert type(restricted) is _TypedJudgerCodexBackend
        assert restricted.roles == backend.roles
        assert restricted.roles["judger"] is backend.roles["judger"]
        assert restricted.model == backend.model
        assert restricted.model_overrides == backend.model_overrides
        assert restricted.effort == backend.effort
        assert restricted.service_tier == backend.service_tier
        assert restricted._model_for("judger") == backend._model_for("judger")
        assert restricted.timeout == backend.timeout
        assert restricted.cwd == backend.cwd
        assert restricted._execution_cwd("judger", {}) != backend.cwd
        assert os.path.isdir(restricted._execution_cwd("judger", {}))
        assert restricted.judger_prefetch is False
        runtime_identities.append(restricted.runtime_instruction_identity)
        execution_cwds.append(restricted._execution_cwd("judger", {}))

    with restricted_typed_judger_backend(backend) as restricted:
        runtime_identities.append(restricted.runtime_instruction_identity)
        execution_cwds.append(restricted._execution_cwd("judger", {}))
    assert len(set(execution_cwds)) == 2
    assert len(set(runtime_identities)) == 1

    class Child(CodexBackend):
        pass

    child = Child(cwd=str(tmp_path))
    with restricted_typed_judger_backend(child) as unchanged:
        assert unchanged is child


def test_operator_bound_scratch_is_exact_and_stable(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    scratch = tmp_path / "typed-judger-scratch"
    scratch.mkdir(mode=0o500)
    backend = CodexBackend(
        cwd=str(state), restricted_judger_cwd=str(scratch))

    with restricted_typed_judger_backend(backend) as first:
        assert first._execution_cwd("judger", {}) == str(scratch)
    with restricted_typed_judger_backend(backend) as second:
        assert second._execution_cwd("judger", {}) == str(scratch)


@pytest.mark.parametrize("defect", ["nonempty", "permissions", "symlink"])
def test_operator_bound_scratch_rejects_untrusted_directory(
        defect, tmp_path):
    scratch = tmp_path / "typed-judger-scratch"
    if defect == "symlink":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        scratch.symlink_to(target, target_is_directory=True)
    else:
        scratch.mkdir(mode=0o500)
        if defect == "nonempty":
            scratch.chmod(0o700)
            (scratch / "unexpected").write_text("x")
            scratch.chmod(0o500)
        else:
            scratch.chmod(0o755)

    with pytest.raises(RoleError, match="restricted Judger cwd"):
        CodexBackend(cwd=str(tmp_path), restricted_judger_cwd=str(scratch))


def test_operator_bound_scratch_rejects_inode_replacement(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    scratch = tmp_path / "typed-judger-scratch"
    scratch.mkdir(mode=0o500)
    backend = CodexBackend(
        cwd=str(state), restricted_judger_cwd=str(scratch))

    with pytest.raises(TypedJudgerTransportError, match="scratch changed"):
        with restricted_typed_judger_backend(backend):
            scratch.rename(tmp_path / "old-scratch")
            scratch.mkdir(mode=0o500)


def test_operator_bound_scratch_rejects_post_call_file(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    scratch = tmp_path / "typed-judger-scratch"
    scratch.mkdir(mode=0o500)
    backend = CodexBackend(
        cwd=str(state), restricted_judger_cwd=str(scratch))

    with pytest.raises(TypedJudgerTransportError, match="scratch changed"):
        with restricted_typed_judger_backend(backend):
            scratch.chmod(0o700)
            (scratch / "unexpected").write_text("x")
            scratch.chmod(0o500)
    scratch.chmod(0o700)


def test_scratch_revalidation_does_not_mask_cancellation(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    scratch = tmp_path / "typed-judger-scratch"
    scratch.mkdir(mode=0o500)
    backend = CodexBackend(
        cwd=str(state), restricted_judger_cwd=str(scratch))

    with pytest.raises(KeyboardInterrupt):
        with restricted_typed_judger_backend(backend):
            scratch.chmod(0o700)
            raise KeyboardInterrupt


def test_bound_process_enters_open_directory_not_replacement(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    scratch = tmp_path / "typed-judger-scratch"
    scratch.mkdir(mode=0o500)
    backend = CodexBackend(
        cwd=str(state), restricted_judger_cwd=str(scratch))
    bound = backend._restricted_judger_execution_cwd()
    original = tmp_path / "original-scratch"
    scratch.rename(original)
    scratch.mkdir(mode=0o500)

    with open(os.devnull) as devnull:
        completed = cb._run_codex_process(
            ["/bin/pwd", "-P"], cwd=bound, stdin=devnull, timeout=5)

    assert completed.returncode == 0
    assert completed.stdout.strip() == str(original)


def test_bound_scratch_must_be_disjoint_from_candidate_root(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir(mode=0o700)
    scratch = candidate / "typed-judger-scratch"
    scratch.mkdir(mode=0o500)

    with pytest.raises(RoleError, match="must be disjoint"):
        CodexBackend(
            cwd=str(candidate), restricted_judger_cwd=str(scratch))


def test_restricted_call_appends_complete_inert_packet_only(tmp_path, monkeypatch):
    state, _target, *_middle, packet = _fixture(tmp_path)
    backend = CodexBackend(
        cwd=str(state), model="gpt-5.6-luna", effort="medium",
        service_tier="standard", judger_prefetch=True)
    seen = {}

    def fake_complete(self, role_name, prompt, variables):
        seen.update({
            "self": self, "role": role_name, "prompt": prompt,
            "variables": variables,
        })
        return dict(YES)

    monkeypatch.setattr(CodexBackend, "complete", fake_complete)
    with restricted_typed_judger_backend(backend) as restricted:
        result = restricted.complete("judger", "ORIGINAL PROMPT", {
            "_typed_adapter_judger_packet": packet,
            "_judger_evidence_packet": object(),
        })
    assert result == YES
    assert seen["role"] == "judger"
    assert seen["prompt"].startswith("ORIGINAL PROMPT")
    assert seen["prompt"].count(packet.render()) == 1
    assert "No prior Judger verdict" in seen["prompt"]
    assert "_judger_evidence_packet" not in seen["variables"]
    assert "_typed_adapter_judger_packet" not in seen["variables"]


def test_typed_output_file_must_equal_last_message_in_sequence(
    tmp_path, monkeypatch,
):
    state, _target, *_middle, packet = _fixture(tmp_path)
    role = AgentRole("judger", "fixed prompt", {"ok": bool}, system="fixed")
    backend = CodexBackend(
        roles={"judger": role}, cwd=str(state), model="gpt-5.6-luna")

    def run(command, **_kwargs):
        Path(command[command.index("-o") + 1]).write_text(
            '{"ok":true}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout=_stream(
            {"type": "item.completed", "item": {
                "id": "m1", "type": "agent_message",
                "text": '{"ok":true}'}},
            {"type": "item.completed", "item": {
                "id": "m2", "type": "agent_message",
                "text": '{"ok":false}'}},
        ))

    monkeypatch.setattr(cb, "_run_codex_process", run)
    with restricted_typed_judger_backend(backend) as restricted:
        with pytest.raises(RoleError, match="disagrees"):
            restricted.complete("judger", "fixed prompt", {
                "_typed_adapter_judger_packet": packet,
            })


def test_typed_message_sequence_does_not_bypass_usage_protocol(
    tmp_path, monkeypatch,
):
    state, _target, *_middle, packet = _fixture(tmp_path)
    role = AgentRole("judger", "fixed prompt", {"ok": bool}, system="fixed")
    backend = CodexBackend(
        roles={"judger": role}, cwd=str(state), model="gpt-5.6-luna")

    def run(command, **_kwargs):
        response = '{"ok":true}'
        Path(command[command.index("-o") + 1]).write_text(
            response, encoding="utf-8")
        stream = "\n".join(json.dumps(event) for event in (
            {"type": "thread.started", "thread_id": "missing-usage"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {
                "id": "m1", "type": "agent_message", "text": "progress"}},
            {"type": "item.completed", "item": {
                "id": "m2", "type": "agent_message", "text": response}},
            {"type": "turn.completed"},
        ))
        return SimpleNamespace(returncode=0, stderr="", stdout=stream)

    monkeypatch.setattr(cb, "_run_codex_process", run)
    usage_path = tmp_path / "usage.jsonl"
    with cb.codex_usage_ledger(str(usage_path), {str(state): {}}):
        with restricted_typed_judger_backend(backend) as restricted:
            with pytest.raises(RoleError, match="usage"):
                restricted.complete("judger", "fixed prompt", {
                    "_typed_adapter_judger_packet": packet,
                })
    row = json.loads(usage_path.read_text())
    assert row["attempt_status"] == "telemetry_error"
    assert row["usage_complete"] is False


def test_typed_output_file_requires_strict_utf8(tmp_path, monkeypatch):
    state, _target, *_middle, packet = _fixture(tmp_path)
    role = AgentRole("judger", "fixed prompt", {"ok": str}, system="fixed")
    backend = CodexBackend(
        roles={"judger": role}, cwd=str(state), model="gpt-5.6-luna")

    def run(command, **_kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(b'{"ok":"\xff"}')
        return SimpleNamespace(returncode=0, stderr="", stdout=_stream(
            {"type": "item.completed", "item": {
                "id": "m", "type": "agent_message", "text": '{"ok":"�"}'}},
        ))

    monkeypatch.setattr(cb, "_run_codex_process", run)
    with restricted_typed_judger_backend(backend) as restricted:
        with pytest.raises(RoleError, match="not valid UTF-8"):
            restricted.complete("judger", "fixed prompt", {
                "_typed_adapter_judger_packet": packet,
            })


def test_native_and_observer_attempts_bind_one_packet_without_leaking(
    tmp_path, monkeypatch,
):
    state, _target, *_middle, packet = _fixture(tmp_path)
    role = AgentRole("judger", "fixed prompt", {"ok": bool}, system="fixed")
    backend = CodexBackend(
        roles={"judger": role}, cwd=str(state), model="gpt-5.6-luna",
        effort="medium", service_tier="standard")
    calls = []

    def run(command, **kwargs):
        calls.append((list(command), kwargs))
        response = '{"ok":true}'
        Path(command[command.index("-o") + 1]).write_text(
            response, encoding="utf-8")
        messages = ([{"type": "item.completed", "item": {
            "id": "progress", "type": "agent_message",
            "text": "working"}}] if len(calls) == 1 else [])
        stream = "\n".join(json.dumps(event) for event in (
            {"type": "thread.started",
             "thread_id": f"typed-audit-{len(calls)}"},
            {"type": "turn.started"},
            *messages,
            {"type": "item.completed", "item": {
                "id": "answer", "type": "agent_message",
                "text": response}},
            {"type": "turn.completed", "usage": {
                "input_tokens": 10, "cached_input_tokens": 2,
                "output_tokens": 3, "reasoning_output_tokens": 1}},
        ))
        return SimpleNamespace(returncode=0, stdout=stream, stderr="")

    monkeypatch.setattr(cb, "_run_codex_process", run)
    native_path = tmp_path / "native.jsonl"
    observer_path = tmp_path / "observer.jsonl"
    context = {str(state): {"fixture": "typed-transport"}}
    with observe_codex_attempts(
            cb, observer_path, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd=context):
        with cb.codex_usage_ledger(str(native_path), context):
            with restricted_typed_judger_backend(backend) as restricted:
                expected_runtime_identity = (
                    restricted.runtime_instruction_identity)
                assert restricted.complete("judger", "fixed prompt", {
                    "_typed_adapter_judger_packet": packet,
                }) == {"ok": True}
            # A later call on the original backend must return to explicit
            # defaults; no instance field or thread-global state is reused.
            assert backend.complete("judger", "fixed prompt", {}) == {
                "ok": True}

    native = [json.loads(line) for line in native_path.read_text().splitlines()]
    observer_rows = [json.loads(line)
                     for line in observer_path.read_text().splitlines()]
    observed = [row for row in observer_rows
                if row["record_type"] == "codex_cli_attempt_finished"]
    assert len(native) == len(observed) == 2
    expected_sha = hashlib.sha256(packet.render().encode("utf-8")).hexdigest()
    for row in (native[0], observed[0]):
        assert row["typed_judger_transport"] is True
        assert row["typed_judger_protocol"] == TYPED_JUDGER_PACKET_PROTOCOL
        assert row["typed_judger_packet_sha256"] == expected_sha
    assert native[0]["runtime_instruction_identity"] \
        == observed[0]["runtime_instruction_identity"]
    assert native[0]["runtime_instruction_identity"] \
        == expected_runtime_identity
    assert native[0]["runtime_instruction_identity"] \
        != native[1]["runtime_instruction_identity"]
    for row in (native[1], observed[1]):
        assert row["typed_judger_transport"] is False
        assert row["typed_judger_protocol"] is None
        assert row["typed_judger_packet_sha256"] is None


def test_attempt_identity_is_context_local_and_nesting_fails_closed():
    backend = CodexBackend()
    barrier = threading.Barrier(2)

    def scoped(digit):
        digest = digit * 64
        with _typed_judger_attempt_scope("protocol-" + digit, digest):
            barrier.wait(timeout=5)
            inside = (
                backend.typed_judger_transport,
                backend.typed_judger_protocol,
                backend.typed_judger_packet_sha256,
            )
            with pytest.raises(RoleError, match="nested"):
                with _typed_judger_attempt_scope("nested", "f" * 64):
                    pass
        outside = (
            backend.typed_judger_transport,
            backend.typed_judger_protocol,
            backend.typed_judger_packet_sha256,
        )
        return inside, outside

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(scoped, ("a", "b")))
    assert {inside for inside, _outside in results} == {
        (True, "protocol-a", "a" * 64),
        (True, "protocol-b", "b" * 64),
    }
    assert all(outside == (False, None, None)
               for _inside, outside in results)
    assert (
        backend.typed_judger_transport,
        backend.typed_judger_protocol,
        backend.typed_judger_packet_sha256,
    ) == (False, None, None)


def _game(fixture, backend, *, retries=1):
    state, _target, log, artifacts, route, obligation, method, result, _packet = \
        fixture
    game = BindingGame(
        backend, artifacts, log, route, live_world=True,
        judger_error_retries=retries, typed_adapter_judger_packet=True)
    attempt = Attempt(
        method, discharge_artifact=result.artifact,
        discharge_outcome=result.outcome)
    return game, obligation, method, result, attempt


def test_default_off_codex_path_is_exactly_the_existing_backend(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path)
    state, _target, log, artifacts, route, obligation, method, result, _packet = \
        fixture
    backend = CodexBackend(cwd=str(state), model="gpt-5.6-luna")
    calls = []

    def fake_complete(self, role_name, prompt, variables, **kwargs):
        calls.append(type(self))
        return dict(YES)

    def forbidden(_backend):
        raise AssertionError("default-off run selected the restricted transport")

    monkeypatch.setattr(CodexBackend, "_complete", fake_complete)
    monkeypatch.setattr(
        game_module, "restricted_typed_judger_backend", forbidden)
    game = BindingGame(
        backend, artifacts, log, route, live_world=True,
        typed_adapter_judger_packet=False)
    attempt = Attempt(
        method, discharge_artifact=result.artifact,
        discharge_outcome=result.outcome)
    assert game._judge(obligation, method, result, attempt) == YES
    assert calls == [CodexBackend]


def test_semantic_no_is_final_and_never_falls_back_to_toolful(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    state = fixture[0]
    backend = CodexBackend(cwd=str(state), model="gpt-5.6-luna")
    calls = []

    def fake_complete(self, role_name, prompt, variables, **kwargs):
        calls.append((type(self), dict(variables)))
        return dict(NO if type(self) is _TypedJudgerCodexBackend else YES)

    monkeypatch.setattr(CodexBackend, "_complete", fake_complete)
    game, obligation, method, result, attempt = _game(fixture, backend)
    assert game._judge(obligation, method, result, attempt) == NO
    assert [kind for kind, _variables in calls] == [_TypedJudgerCodexBackend]


def test_tool_attempt_uses_only_existing_retry_for_toolful_fallback(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path)
    backend = CodexBackend(cwd=str(fixture[0]), model="gpt-5.6-luna")
    calls = []

    def fake_complete(self, role_name, prompt, variables, **kwargs):
        calls.append((type(self), dict(variables)))
        if type(self) is _TypedJudgerCodexBackend:
            raise TypedJudgerTransportError(
                "typed Judger emitted a tool or unknown event")
        return dict(YES)

    monkeypatch.setattr(CodexBackend, "_complete", fake_complete)
    game, obligation, method, result, attempt = _game(fixture, backend)
    assert game._judge(obligation, method, result, attempt) == YES
    # AgentRole retains its pre-existing one schema/role retry inside the
    # first Judger call; BindingGame then spends its one existing call retry.
    assert [kind for kind, _variables in calls] == [
        _TypedJudgerCodexBackend, _TypedJudgerCodexBackend, CodexBackend]
    assert "_typed_adapter_judger_packet" not in calls[-1][1]


def test_pre_call_replay_failure_falls_back_without_accepting_packet(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path)
    state, target = fixture[:2]
    packet = fixture[-1]
    backend = CodexBackend(cwd=str(state), model="gpt-5.6-luna")
    calls = []

    def stale_packet(*_args, **_kwargs):
        target.write_text("changed after packet construction", encoding="utf-8")
        return packet

    def fake_complete(self, role_name, prompt, variables, **kwargs):
        calls.append(type(self))
        return dict(YES)

    monkeypatch.setattr(
        game_module, "build_typed_adapter_judger_packet", stale_packet)
    monkeypatch.setattr(CodexBackend, "_complete", fake_complete)
    game, obligation, method, result, attempt = _game(fixture, backend)
    assert game._judge(obligation, method, result, attempt) == YES
    # The stale packet is rejected before a restricted model process starts;
    # the next outer call is the unchanged toolful Judger.
    assert calls == [CodexBackend]


def test_no_retry_budget_fails_closed_after_restricted_transport_error(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path)
    backend = CodexBackend(cwd=str(fixture[0]), model="gpt-5.6-luna")

    def failed_transport(self, role_name, prompt, variables, **kwargs):
        raise TypedJudgerTransportError("synthetic restricted failure")

    monkeypatch.setattr(CodexBackend, "_complete", failed_transport)
    game, obligation, method, result, attempt = _game(
        fixture, backend, retries=0)
    verdict = game._judge(obligation, method, result, attempt)
    assert verdict["sufficient"] == "unknown"
    assert verdict["faithful"] == "no"
    assert verdict["_infra"] is True
    assert (
        backend.typed_judger_transport,
        backend.typed_judger_protocol,
        backend.typed_judger_packet_sha256,
    ) == (False, None, None)
