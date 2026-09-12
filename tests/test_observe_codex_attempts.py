"""Local tests for the cross-version Codex attempt observer."""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from experiments.terminal_bench.observe_codex_attempts import (
    LEGACY_SUBPROCESS_SEAM,
    OBSERVER_SCHEMA,
    RUN_CODEX_PROCESS_SEAM,
    observe_codex_attempts,
)
from experiments.terminal_bench.analyze_codex_observer import (
    analyze_file,
    analyze_rows,
)


def _command(message='{"ok":true}'):
    return [
        "codex", "exec", "--model", "gpt-5.6-luna",
        "-c", 'service_tier="default"', "--disable", "fast_mode",
        "-c", "model_reasoning_effort=medium", "-o", "/output", "--",
        message,
    ]


def _stream(*, input_tokens=100, cached=60, cache_write=3,
            output=12, reasoning=7):
    return "\n".join(json.dumps(event) for event in (
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {
            "id": "message-1", "type": "agent_message",
            "text": '{"ok":true}',
        }},
        {"type": "turn.completed", "usage": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached,
            "cache_write_input_tokens": cache_write,
            "output_tokens": output,
            "reasoning_output_tokens": reasoning,
        }},
    )) + "\n"


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _by_type(path: Path, record_type: str):
    return [row for row in _rows(path)
            if row["record_type"] == record_type]


def _state(tmp_path, name="state"):
    path = tmp_path / name
    path.mkdir()
    return path


def _module(tmp_path, seam, run):
    module = types.ModuleType("fake_codex_backend_" + seam.replace("-", "_"))
    module.subprocess = types.SimpleNamespace(
        run=run, TimeoutExpired=subprocess.TimeoutExpired)
    if seam == RUN_CODEX_PROCESS_SEAM:
        module._run_codex_process = run

    class CodexBackend:
        def __init__(self, cwd):
            self.cwd = str(cwd)
            self.model = "gpt-5.6-luna"
            self.effort = "medium"
            self.service_tier = "standard"

        def _model_for(self, _role):
            return self.model

        def complete(self, role_name, prompt, variables):
            command = variables.get("command", _command(prompt))
            if seam == RUN_CODEX_PROCESS_SEAM:
                proc = module._run_codex_process(
                    command, cwd=self.cwd, stdin=variables.get("stdin"),
                    timeout=variables.get("timeout", 11))
            else:
                proc = module.subprocess.run(
                    command, cwd=self.cwd, stdin=variables.get("stdin"),
                    capture_output=True, text=True,
                    timeout=variables.get("timeout", 11))
            if variables.get("schema_failure"):
                raise ValueError("sensitive schema output must not be logged")
            return variables.get("return", proc)

    module.CodexBackend = CodexBackend
    return module


@pytest.mark.parametrize("seam", [
    LEGACY_SUBPROCESS_SEAM, RUN_CODEX_PROCESS_SEAM,
])
def test_success_records_content_free_attempt_and_preserves_result(
        tmp_path, seam):
    state = _state(tmp_path)
    sentinel = subprocess.CompletedProcess(
        _command(), 0, stdout=_stream(), stderr="private stderr")
    seen = []

    def run(*args, **kwargs):
        seen.append((args, kwargs))
        return sentinel

    module = _module(state, seam, run)
    backend = module.CodexBackend(state)
    ledger = tmp_path / "observer.jsonl"
    context = {str(state): {"task": "benign", "trial": "candidate-1"}}
    with observe_codex_attempts(
            module, ledger, seam=seam, context_by_cwd=context):
        result = backend.complete("judger", "private prompt", {})
    assert result is sentinel
    assert len(seen) == 1
    if seam == LEGACY_SUBPROCESS_SEAM:
        assert seen[0][0] == (_command("private prompt"),)
        assert seen[0][1] == {
            "cwd": str(state), "stdin": None,
            "capture_output": True, "text": True, "timeout": 11,
        }
    else:
        assert seen[0][0] == (_command("private prompt"),)
        assert seen[0][1] == {
            "cwd": str(state), "stdin": None, "timeout": 11,
        }

    rows = _rows(ledger)
    assert [row["record_type"] for row in rows] == [
        "codex_logical_call_started", "codex_cli_attempt_started",
        "codex_cli_attempt_finished", "codex_logical_call_finished",
    ]
    assert all(row["schema"] == OBSERVER_SCHEMA for row in rows)
    terminal = rows[2]
    assert terminal["terminal_status"] == "returned"
    assert terminal["returncode"] == 0
    assert terminal["usage_complete"] is True
    assert terminal["usage_status"] == "complete"
    assert terminal["input_tokens"] == 100
    assert terminal["cached_input_tokens"] == 60
    assert terminal["uncached_input_tokens"] == 40
    assert terminal["cache_write_input_tokens"] == 3
    assert terminal["output_tokens"] == 12
    assert terminal["reasoning_tokens"] == 7
    assert terminal["command_model"] == "gpt-5.6-luna"
    assert terminal["command_effort"] == "medium"
    assert terminal["command_service_tier"] == "standard"
    assert terminal["runtime_instruction_identity"] is None
    assert terminal["typed_judger_transport"] is False
    assert terminal["typed_judger_protocol"] is None
    assert terminal["typed_judger_packet_sha256"] is None
    assert terminal["context"] == context[str(state)]
    encoded = ledger.read_text()
    assert "private prompt" not in encoded
    assert "private stderr" not in encoded
    assert '{"ok":true}' not in encoded


@pytest.mark.parametrize("seam", [
    LEGACY_SUBPROCESS_SEAM, RUN_CODEX_PROCESS_SEAM,
])
def test_schema_failure_is_logical_raise_not_process_reclassification(
        tmp_path, seam):
    state = _state(tmp_path)
    proc = subprocess.CompletedProcess(_command(), 0, _stream(), "")
    module = _module(state, seam, lambda *args, **kwargs: proc)
    ledger = tmp_path / "observer.jsonl"
    with observe_codex_attempts(
            module, ledger, seam=seam,
            context_by_cwd={str(state): {"trial": "one"}}):
        with pytest.raises(ValueError, match="sensitive schema"):
            module.CodexBackend(state).complete(
                "judger", "secret", {"schema_failure": True})
    attempt = _by_type(ledger, "codex_cli_attempt_finished")[0]
    logical = _by_type(ledger, "codex_logical_call_finished")[0]
    assert attempt["terminal_status"] == "returned"
    assert attempt["usage_complete"] is True
    assert logical["terminal_status"] == "raised"
    assert logical["error_kind"] == "ValueError"
    assert "sensitive" not in ledger.read_text()


def test_provider_retry_processes_are_counted_without_semantic_label(tmp_path):
    state = _state(tmp_path)
    responses = iter([
        subprocess.CompletedProcess(_command(), 1, "", "capacity private"),
        subprocess.CompletedProcess(_command(), 0, _stream(), ""),
    ])
    module = types.ModuleType("fake_retry_backend")
    module.subprocess = types.SimpleNamespace(
        TimeoutExpired=subprocess.TimeoutExpired)

    def process(cmd, *, cwd, stdin, timeout):
        return next(responses)

    module._run_codex_process = process

    class CodexBackend:
        model = "gpt-5.6-luna"
        effort = "medium"
        service_tier = "standard"

        def __init__(self, cwd):
            self.cwd = str(cwd)

        def complete(self, role_name, prompt, variables):
            for _ in range(2):
                proc = module._run_codex_process(
                    _command(prompt), cwd=self.cwd, stdin=None, timeout=4)
                if proc.returncode == 0:
                    return proc
            raise RuntimeError

    module.CodexBackend = CodexBackend
    ledger = tmp_path / "observer.jsonl"
    with observe_codex_attempts(
            module, ledger, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd={str(state): {"trial": "retry"}}):
        result = CodexBackend(state).complete("oracles", "hidden", {})
    assert result.returncode == 0
    attempts = _by_type(ledger, "codex_cli_attempt_finished")
    assert [row["attempt_index"] for row in attempts] == [1, 2]
    assert [row["returncode"] for row in attempts] == [1, 0]
    assert [row["terminal_status"] for row in attempts] == [
        "returned", "returned"]
    assert [row["usage_complete"] for row in attempts] == [False, True]
    assert _by_type(
        ledger, "codex_logical_call_finished")[0]["attempt_count"] == 2
    assert "capacity private" not in ledger.read_text()


@pytest.mark.parametrize(("error", "status", "started"), [
    (subprocess.TimeoutExpired(_command(), 1, output=b"partial"),
     "timeout", True),
    (OSError("private launch path"), "launch_error", False),
    (KeyboardInterrupt(), "interrupted", None),
])
def test_process_exceptions_preserve_identity_and_record_unknown_usage(
        tmp_path, error, status, started):
    state = _state(tmp_path)
    def run(*args, **kwargs):
        raise error

    module = _module(state, RUN_CODEX_PROCESS_SEAM, run)
    ledger = tmp_path / "observer.jsonl"
    with observe_codex_attempts(
            module, ledger, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd={str(state): {"trial": "failure"}}):
        with pytest.raises(BaseException) as caught:
            module.CodexBackend(state).complete("critic", "secret", {})
    assert caught.value is error
    attempt = _by_type(ledger, "codex_cli_attempt_finished")[0]
    assert attempt["terminal_status"] == status
    assert attempt["process_started"] is started
    assert attempt["usage_complete"] is False
    assert all(attempt[field] is None for field in (
        "input_tokens", "cached_input_tokens", "uncached_input_tokens",
        "cache_write_input_tokens", "output_tokens", "reasoning_tokens",
    ))
    if str(error):
        assert str(error) not in ledger.read_text()


def test_timeout_cannot_certify_a_complete_looking_partial_stream(tmp_path):
    state = _state(tmp_path)
    error = subprocess.TimeoutExpired(
        _command(), 1, output=_stream().encode("utf-8"))
    module = _module(
        state, RUN_CODEX_PROCESS_SEAM,
        lambda *args, **kwargs: (_ for _ in ()).throw(error))
    ledger = tmp_path / "observer.jsonl"
    with observe_codex_attempts(
            module, ledger, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd={str(state): {"trial": "timeout"}}):
        with pytest.raises(subprocess.TimeoutExpired):
            module.CodexBackend(state).complete("judger", "private", {})
    row = _by_type(ledger, "codex_cli_attempt_finished")[0]
    assert row["usage_status"] == "partial"
    assert row["usage_complete"] is False
    assert row["input_tokens"] is None


@pytest.mark.parametrize(("stdout", "status"), [
    ("", "unknown"),
    ('{"type":"thread.started","thread_id":"x"}\n', "partial"),
    (_stream(input_tokens="100"), "invalid"),
    (_stream(input_tokens=50, cached=60), "invalid"),
    ('{"type":"thread.started","thread_id":"x","thread_id":"y"}\n',
     "invalid"),
])
def test_missing_or_invalid_usage_never_becomes_zero(tmp_path, stdout, status):
    state = _state(tmp_path)
    proc = subprocess.CompletedProcess(_command(), 0, stdout, "")
    module = _module(state, RUN_CODEX_PROCESS_SEAM,
                     lambda *args, **kwargs: proc)
    ledger = tmp_path / f"{status}.jsonl"
    with observe_codex_attempts(
            module, ledger, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd={str(state): {"trial": status}}):
        module.CodexBackend(state).complete("judger", "private", {})
    row = _by_type(ledger, "codex_cli_attempt_finished")[0]
    assert row["usage_status"] == status
    assert row["usage_complete"] is False
    for field in (
        "input_tokens", "cached_input_tokens", "uncached_input_tokens",
        "cache_write_input_tokens", "output_tokens", "reasoning_tokens",
    ):
        assert row[field] is None


def test_nested_duplicate_in_non_usage_event_does_not_hide_terminal_usage(
        tmp_path):
    state = _state(tmp_path)
    lines = _stream().splitlines()
    lines.insert(2, (
        '{"type":"item.completed","item":{"id":"tool-1",'
        '"type":"command_execution","metadata":{"value":1,"value":2}}}'
    ))
    proc = subprocess.CompletedProcess(
        _command(), 0, "\n".join(lines) + "\n", "")
    module = _module(state, RUN_CODEX_PROCESS_SEAM,
                     lambda *args, **kwargs: proc)
    ledger = tmp_path / "nested-duplicate.jsonl"

    with observe_codex_attempts(
            module, ledger, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd={str(state): {"trial": "nested-duplicate"}}):
        module.CodexBackend(state).complete("judger", "private", {})

    row = _by_type(ledger, "codex_cli_attempt_finished")[0]
    assert row["usage_status"] == "complete"
    assert row["input_tokens"] == 100


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_unicode_string_separators_do_not_split_native_jsonl(
        tmp_path, separator):
    state = _state(tmp_path)
    lines = _stream().split("\n")
    lines.insert(2, json.dumps({
        "type": "item.completed",
        "item": {
            "id": "tool-1",
            "type": "command_execution",
            "aggregated_output": "before" + separator + "after",
        },
    }, ensure_ascii=False))
    proc = subprocess.CompletedProcess(
        _command(), 0, "\n".join(lines), "")
    module = _module(state, RUN_CODEX_PROCESS_SEAM,
                     lambda *args, **kwargs: proc)
    ledger = tmp_path / "unicode-separator.jsonl"

    with observe_codex_attempts(
            module, ledger, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd={str(state): {"trial": "unicode-separator"}}):
        module.CodexBackend(state).complete("judger", "private", {})

    row = _by_type(ledger, "codex_cli_attempt_finished")[0]
    assert row["usage_status"] == "complete"
    assert row["input_tokens"] == 100


def test_duplicate_top_level_type_keeps_usage_incomplete(tmp_path):
    state = _state(tmp_path)
    lines = _stream().splitlines()
    lines.insert(2, (
        '{"type":"turn.completed","type":"item.completed",'
        '"item":{"id":"tool-1","type":"command_execution"},'
        '"usage":{"input_tokens":1,"cached_input_tokens":0,'
        '"output_tokens":1,"reasoning_output_tokens":0}}'
    ))
    proc = subprocess.CompletedProcess(
        _command(), 0, "\n".join(lines) + "\n", "")
    module = _module(state, RUN_CODEX_PROCESS_SEAM,
                     lambda *args, **kwargs: proc)
    ledger = tmp_path / "duplicate-type.jsonl"

    with observe_codex_attempts(
            module, ledger, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd={str(state): {"trial": "duplicate-type"}}):
        module.CodexBackend(state).complete("judger", "private", {})

    row = _by_type(ledger, "codex_cli_attempt_finished")[0]
    assert row["usage_status"] == "invalid"
    assert row["usage_complete"] is False
    assert row["input_tokens"] is None


def test_threaded_calls_keep_role_and_candidate_context_separate(tmp_path):
    first = tmp_path / "candidate-a"
    second = tmp_path / "candidate-b"
    first.mkdir()
    second.mkdir()
    barrier = threading.Barrier(2)

    def run(cmd, *, cwd, stdin, timeout):
        barrier.wait(timeout=2)
        thread = "thread-a" if Path(cwd) == first else "thread-b"
        return subprocess.CompletedProcess(
            cmd, 0, _stream().replace("thread-1", thread), "")

    module = _module(tmp_path, RUN_CODEX_PROCESS_SEAM, run)
    ledger = tmp_path / "observer.jsonl"
    contexts = {
        str(first): {"trial": "a"},
        str(second): {"trial": "b"},
    }
    with observe_codex_attempts(
            module, ledger, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd=contexts):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(module.CodexBackend(first).complete,
                            "critic", "one", {}),
                pool.submit(module.CodexBackend(second).complete,
                            "judger", "two", {}),
            ]
            [future.result() for future in futures]
    attempts = _by_type(ledger, "codex_cli_attempt_finished")
    assert len(attempts) == 2
    assert {(row["context"]["trial"], row["role"], row["thread_id"])
            for row in attempts} == {
        ("a", "critic", "thread-a"),
        ("b", "judger", "thread-b"),
    }
    assert len({row["call_id"] for row in attempts}) == 2


def test_context_is_copied_and_unknown_cwd_is_unattributed(tmp_path):
    state = _state(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    proc = subprocess.CompletedProcess(_command(), 0, _stream(), "")
    module = _module(state, RUN_CODEX_PROCESS_SEAM,
                     lambda *args, **kwargs: proc)
    context = {str(state): {"trial": "before"}}
    ledger = tmp_path / "observer.jsonl"
    with observe_codex_attempts(
            module, ledger, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd=context):
        context[str(state)]["trial"] = "after"
        module.CodexBackend(state).complete("critic", "one", {})
        module.CodexBackend(other).complete("critic", "two", {})
    calls = _by_type(ledger, "codex_logical_call_started")
    assert calls[0]["context"] == {"trial": "before"}
    assert calls[0]["attribution"] == "attributed"
    assert calls[1]["context"] == {}
    assert calls[1]["attribution"] == "unattributed"


def test_non_codex_subprocess_is_unchanged_and_not_recorded(tmp_path):
    sentinel = object()
    seen = []

    def run(*args, **kwargs):
        seen.append((args, kwargs))
        return sentinel

    module = _module(tmp_path, LEGACY_SUBPROCESS_SEAM, run)
    ledger = tmp_path / "observer.jsonl"
    original_subprocess = module.subprocess
    with observe_codex_attempts(
            module, ledger, seam=LEGACY_SUBPROCESS_SEAM,
            context_by_cwd={}):
        result = module.subprocess.run(
            ["python3", "-V"], check=False, cwd=str(tmp_path))
    assert result is sentinel
    assert seen == [((["python3", "-V"],), {
        "check": False, "cwd": str(tmp_path)})]
    assert ledger.read_bytes() == b""
    assert module.subprocess is original_subprocess


def test_install_is_exclusive_and_restores_exact_objects(tmp_path):
    proc = subprocess.CompletedProcess(_command(), 0, _stream(), "")
    module = _module(tmp_path, RUN_CODEX_PROCESS_SEAM,
                     lambda *args, **kwargs: proc)
    original_complete = module.CodexBackend.complete
    original_process = module._run_codex_process
    first = tmp_path / "first.jsonl"
    with observe_codex_attempts(
            module, first, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd={}):
        with pytest.raises(RuntimeError, match="already installed"):
            with observe_codex_attempts(
                    module, tmp_path / "second.jsonl",
                    seam=RUN_CODEX_PROCESS_SEAM, context_by_cwd={}):
                pass
        assert not (tmp_path / "second.jsonl").exists()
    assert module.CodexBackend.complete is original_complete
    assert module._run_codex_process is original_process


def test_summary_binds_closed_ledger_and_integrates_with_accounting_gate(
        tmp_path):
    state = _state(tmp_path)
    proc = subprocess.CompletedProcess(_command(), 0, _stream(), "")
    module = _module(state, RUN_CODEX_PROCESS_SEAM,
                     lambda *args, **kwargs: proc)
    ledger = tmp_path / "observer.jsonl"
    contexts = {str(state.resolve()): {"trial": "one"}}
    with observe_codex_attempts(
            module, ledger, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd=contexts) as session:
        module.CodexBackend(state).complete("judger", "private", {})
        assert session.summary()["closed"] is False
    summary = session.summary()
    assert summary == {
        "schema": "gv-codex-attempt-observer-summary-v1",
        "observer_id": session.observer_id,
        "seam": RUN_CODEX_PROCESS_SEAM,
        "logical_calls_started": 1,
        "logical_calls_finished": 1,
        "attempts_started": 1,
        "attempts_finished": 1,
        "write_failures": 0,
        "active_calls": 0,
        "restoration_failed": False,
        "closed": True,
        "ledger_path": str(ledger.resolve()),
        "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "ledger_bytes": len(ledger.read_bytes()),
    }
    report = analyze_rows(
        _rows(ledger), context_by_cwd=contexts,
        model="gpt-5.6-luna", effort="medium", service_tier="standard",
        session_summary=summary,
    )
    assert report["accounting_complete"] is True
    assert report["attempt_count"] == 1


def test_ledger_must_be_fresh_and_outside_candidate_state(tmp_path):
    state = _state(tmp_path)
    module = _module(
        state, RUN_CODEX_PROCESS_SEAM,
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", ""))
    existing = tmp_path / "existing.jsonl"
    existing.write_text("old\n")
    for ledger in (existing, state / "candidate-readable.jsonl"):
        with pytest.raises((FileExistsError, ValueError)):
            with observe_codex_attempts(
                    module, ledger, seam=RUN_CODEX_PROCESS_SEAM,
                    context_by_cwd={str(state): {"trial": "one"}}):
                pass


def test_observer_write_failure_does_not_change_backend_result(tmp_path):
    state = _state(tmp_path)
    sentinel = subprocess.CompletedProcess(_command(), 0, _stream(), "")
    module = _module(state, RUN_CODEX_PROCESS_SEAM,
                     lambda *args, **kwargs: sentinel)
    ledger = tmp_path / "observer.jsonl"
    with observe_codex_attempts(
            module, ledger, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd={str(state): {"trial": "one"}}) as session:
        ledger.unlink()
        ledger.write_text("replacement")
        result = module.CodexBackend(state).complete(
            "judger", "private", {})
    assert result is sentinel
    assert session.summary()["write_failures"] > 0


def test_restoration_failure_does_not_mask_body_exception(tmp_path):
    state = _state(tmp_path)
    proc = subprocess.CompletedProcess(_command(), 0, _stream(), "")
    original = lambda *args, **kwargs: proc
    module = _module(state, RUN_CODEX_PROCESS_SEAM, original)
    ledger = tmp_path / "observer.jsonl"
    body_error = ValueError("body failure")
    with pytest.raises(ValueError) as caught:
        with observe_codex_attempts(
                module, ledger, seam=RUN_CODEX_PROCESS_SEAM,
                context_by_cwd={str(state): {"trial": "one"}}) as session:
            module._run_codex_process = lambda *args, **kwargs: proc
            raise body_error
    assert caught.value is body_error
    assert session.summary()["restoration_failed"] is True
    assert session.summary()["closed"] is True


def test_observer_does_not_call_stateful_model_hook(tmp_path):
    state = _state(tmp_path)
    module = types.ModuleType("stateful_model_backend")
    module.subprocess = types.SimpleNamespace(
        TimeoutExpired=subprocess.TimeoutExpired)
    module._run_codex_process = lambda cmd, *, cwd, stdin, timeout: (
        subprocess.CompletedProcess(cmd, 0, _stream(), ""))

    class CodexBackend:
        model = "gpt-5.6-luna"
        effort = "medium"
        service_tier = "standard"

        def __init__(self):
            self.cwd = str(state)
            self.model_hook_calls = 0

        def _model_for(self, role):
            self.model_hook_calls += 1
            return self.model

        def complete(self, role, prompt, variables):
            self._model_for(role)
            return module._run_codex_process(
                _command(prompt), cwd=self.cwd, stdin=None, timeout=1)

    module.CodexBackend = CodexBackend
    backend = CodexBackend()
    with observe_codex_attempts(
            module, tmp_path / "observer.jsonl",
            seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd={str(state): {"trial": "one"}}):
        backend.complete("judger", "private", {})
    assert backend.model_hook_calls == 1


@pytest.mark.parametrize("extra", [
    ["--model=other"],
    ["--config", "other.toml"],
    ["-c", "model=other"],
])
def test_actual_profile_rejects_ambiguous_overrides_before_prompt(
        tmp_path, extra):
    state = _state(tmp_path)
    command = _command()
    command[2:2] = extra
    # Prompt-looking tokens after the boundary must not be parsed as options.
    command[-1] = "--model other -c service_tier=fast"
    proc = subprocess.CompletedProcess(command, 0, _stream(), "")
    module = _module(state, RUN_CODEX_PROCESS_SEAM,
                     lambda *args, **kwargs: proc)
    with observe_codex_attempts(
            module, tmp_path / "observer.jsonl",
            seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd={str(state): {"trial": "one"}}):
        module.CodexBackend(state).complete(
            "judger", "private", {"command": command})
    row = _by_type(
        tmp_path / "observer.jsonl", "codex_cli_attempt_finished")[0]
    assert row["command_model"] is None
    assert row["command_effort"] is None
    assert row["command_service_tier"] is None


def test_actual_profile_never_parses_prompt_tokens_as_options(tmp_path):
    state = _state(tmp_path)
    command = _command("--model other --config secret.toml --enable fast_mode")
    proc = subprocess.CompletedProcess(command, 0, _stream(), "")
    module = _module(state, RUN_CODEX_PROCESS_SEAM,
                     lambda *args, **kwargs: proc)
    ledger = tmp_path / "observer.jsonl"
    with observe_codex_attempts(
            module, ledger, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd={str(state): {"trial": "one"}}):
        module.CodexBackend(state).complete(
            "judger", "private", {"command": command})
    row = _by_type(ledger, "codex_cli_attempt_finished")[0]
    assert (row["command_model"], row["command_effort"],
            row["command_service_tier"]) == (
        "gpt-5.6-luna", "medium", "standard")


def test_current_backend_and_native_usage_capture_integrate_without_process(
        tmp_path, monkeypatch):
    from types import SimpleNamespace
    from verifier.llm import codex_backend as current

    state = _state(tmp_path)
    observer_ledger = tmp_path / "observer.jsonl"
    native_ledger = tmp_path / "native-usage.jsonl"
    contexts = {str(state.resolve()): {
        "job": "local", "task": "benign", "trial": "one",
        "config": "test", "rep": "r1",
    }}

    def fake_process(cmd, *, cwd, stdin, timeout):
        output = Path(cmd[cmd.index("-o") + 1])
        output.write_text('{"ok":true}', encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd, 0, _stream(cache_write=0), "")

    monkeypatch.setattr(current, "_run_codex_process", fake_process)
    backend = current.CodexBackend(
        roles={"critic": SimpleNamespace(system="", schema={"ok": bool})},
        cwd=str(state), model="gpt-5.6-luna", effort="medium",
        service_tier="standard",
    )
    with observe_codex_attempts(
            current, observer_ledger, seam=RUN_CODEX_PROCESS_SEAM,
            context_by_cwd=contexts) as session:
        with current.codex_usage_ledger(
                str(native_ledger), context_by_cwd=contexts):
            assert backend.complete("critic", "inert local test", {}) == {
                "ok": True}
    report = analyze_file(
        observer_ledger, context_by_cwd=contexts,
        model="gpt-5.6-luna", effort="medium", service_tier="standard",
        session_summary=session.summary(),
    )
    assert report["accounting_complete"] is True
    assert report["ledger_identity_complete"] is True
    assert report["attempt_count"] == 1


@pytest.mark.parametrize("bad_context", [
    {"/tmp/state": {"nested": {"candidate": "bytes"}}},
    {"/tmp/state": {"large": "x" * 513}},
    {"/tmp/state": {"nan": float("nan")}},
])
def test_context_rejects_candidate_shaped_or_unbounded_values(
        tmp_path, bad_context):
    module = _module(
        tmp_path, RUN_CODEX_PROCESS_SEAM,
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", ""))
    with pytest.raises(ValueError):
        with observe_codex_attempts(
                module, tmp_path / "observer.jsonl",
                seam=RUN_CODEX_PROCESS_SEAM,
                context_by_cwd=bad_context):
            pass


def test_invalid_install_surface_fails_before_patching(tmp_path):
    module = types.ModuleType("missing_codex_surface")
    module.CodexBackend = type("CodexBackend", (), {"complete": lambda: None})
    with pytest.raises(TypeError, match="_run_codex_process"):
        with observe_codex_attempts(
                module, tmp_path / "observer.jsonl",
                seam=RUN_CODEX_PROCESS_SEAM, context_by_cwd={}):
            pass
