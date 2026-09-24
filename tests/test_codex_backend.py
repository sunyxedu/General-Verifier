"""CodexBackend contracts, JSON recovery, and local process-group cleanup.

Real model calls are kept out of this unit suite.
"""
import base64
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
import tomllib
from types import SimpleNamespace

import pytest

from verifier.llm.codex_backend import (
    CodexBackend,
    _run_codex_process,
    codex_usage_ledger,
    extract_json,
    service_tier_cli_args,
)
from verifier.llm.schema import RoleError


PNG_A = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII=")
PNG_B = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAusB9Y9ZepAAAAAASUVORK5CYII=")


def _usage_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _cli_config(args):
    values = [args[index + 1] for index, arg in enumerate(args)
              if arg == "-c"]
    return tomllib.loads("\n".join(values))


def _complete_usage_stream(thread_id="thread-1"):
    events = [
        {"type": "thread.started", "thread_id": thread_id},
        {"type": "item.completed", "item": {
            "type": "agent_message", "text": '{"ok": true}',
        }},
        {"type": "turn.completed", "usage": {
            "input_tokens": 100,
            "cached_input_tokens": 60,
            "output_tokens": 12,
            "reasoning_output_tokens": 7,
        }},
    ]
    return "\n".join(json.dumps(event) for event in events) + "\n"


def _assert_unknown_usage(row):
    assert row["usage_complete"] is False
    for field in (
        "input_tokens", "cached_input_tokens", "uncached_input_tokens",
        "cache_write_input_tokens", "output_tokens", "reasoning_tokens",
    ):
        assert row[field] is None


def _process_group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(pgid, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_group_exists(pgid):
            return True
        time.sleep(0.02)
    return not _process_group_exists(pgid)


def _sleeper_tree_command(marker, parent_sleep):
    child_code = "import time; time.sleep(30)"
    wrapper_code = (
        "import os,pathlib,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[2]],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
        "stderr=subprocess.DEVNULL)\n"
        "pathlib.Path(sys.argv[1]).write_text("
        "f'{os.getpid()} {child.pid}')\n"
        "time.sleep(float(sys.argv[3]))\n"
    )
    return [
        sys.executable, "-c", wrapper_code,
        str(marker), child_code, str(parent_sleep),
    ]


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "killpg"),
    reason="POSIX process-group containment",
)
@pytest.mark.parametrize(
    ("parent_sleep", "timeout", "times_out"),
    [(30, 0.3, True), (0, 2, False)],
)
def test_codex_runner_leaves_no_same_group_child(
        tmp_path, parent_sleep, timeout, times_out):
    marker = tmp_path / "process-tree.txt"
    command = _sleeper_tree_command(marker, parent_sleep)
    pgid = None
    try:
        with open(os.devnull) as devnull:
            if times_out:
                with pytest.raises(subprocess.TimeoutExpired):
                    _run_codex_process(
                        command, cwd=str(tmp_path), stdin=devnull,
                        timeout=timeout)
            else:
                completed = _run_codex_process(
                    command, cwd=str(tmp_path), stdin=devnull,
                    timeout=timeout)
                assert completed.returncode == 0
        pgid, child_pid = map(int, marker.read_text().split())
        assert child_pid != pgid
        assert _wait_for_process_group_exit(pgid), (
            f"process group {pgid} survived its Codex invocation")
    finally:
        if pgid is not None and _process_group_exists(pgid):
            os.killpg(pgid, signal.SIGKILL)


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "killpg"),
    reason="POSIX process-group containment",
)
def test_codex_runner_cleans_group_when_communicate_is_interrupted(
        tmp_path, monkeypatch):
    marker = tmp_path / "interrupted-tree.txt"
    command = _sleeper_tree_command(marker, 30)
    real_popen = subprocess.Popen

    def interrupted_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)

        def interrupt(*_args, **_kwargs):
            deadline = time.monotonic() + 2
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.exists()
            raise KeyboardInterrupt

        proc.communicate = interrupt
        return proc

    monkeypatch.setattr(
        "verifier.llm.codex_backend.subprocess.Popen", interrupted_popen)
    pgid = None
    try:
        with open(os.devnull) as devnull:
            with pytest.raises(KeyboardInterrupt):
                _run_codex_process(
                    command, cwd=str(tmp_path), stdin=devnull, timeout=2)
        pgid, _child_pid = map(int, marker.read_text().split())
        assert _wait_for_process_group_exit(pgid), (
            f"process group {pgid} survived interrupted communicate")
    finally:
        if pgid is not None and _process_group_exists(pgid):
            os.killpg(pgid, signal.SIGKILL)


def _captured_state(tmp_path, files):
    world = tmp_path / "world"
    app = world / "app"
    app.mkdir(parents=True)
    for relative, content in files.items():
        path = world / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = tmp_path / "world_manifest.json"
    manifest.write_text(json.dumps({"captured_roots": ["/app"]}))
    return world, manifest


def _image_variables(world, manifest, *, role="discharge", reference=None,
                     world_mode="captured", basis="world"):
    variables = {
        "obligation_text": reference or "inspect the delivered result",
        "procedure": "audit the referenced captured-world input",
        "promised_artifact": "a complete audit report",
        "world_mode": world_mode,
        "world_path": str(world),
        "world_manifest": str(manifest),
    }
    variables["declared_basis" if role == "discharge" else "basis"] = basis
    return variables


def test_extract_json_tolerates_fences_and_prose():
    text = 'Sure! Here it is:\n```json\n{"verdict": "pass", "n": [1, 2]}\n```'
    assert extract_json(text) == {"verdict": "pass", "n": [1, 2]}


def test_extract_json_skips_broken_braces():
    assert extract_json('{oops} then {"ok": true}') == {"ok": True}


def test_extract_json_fails_closed():
    with pytest.raises(RoleError):
        extract_json("no json here")


def test_unknown_role_is_a_role_error():
    with pytest.raises(RoleError, match="unknown agent role"):
        CodexBackend().complete("nope", "prompt", {})


def test_failed_process_cannot_return_a_valid_seat_answer(tmp_path, monkeypatch):
    backend = CodexBackend(roles={"critic": SimpleNamespace(system="", schema={"ok": bool})},
                           cwd=str(tmp_path))

    def fail_after_output(cmd, **kwargs):
        with open(cmd[cmd.index("-o") + 1], "w") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(returncode=1, stderr="execution failed",
                               stdout='{"type":"thread.started","thread_id":"t1"}\n')

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fail_after_output)
    with pytest.raises(RoleError, match="codex exec failed"):
        backend.complete("critic", "prompt", {})


def test_retry_cannot_reuse_an_earlier_attempts_output(tmp_path, monkeypatch):
    backend = CodexBackend(roles={"critic": SimpleNamespace(system="", schema={"ok": bool})},
                           cwd=str(tmp_path))
    attempts = []

    def retry_without_output(cmd, **kwargs):
        attempts.append(cmd)
        if len(attempts) == 1:
            with open(cmd[cmd.index("-o") + 1], "w") as handle:
                json.dump({"ok": True}, handle)
            return SimpleNamespace(returncode=1, stderr="", stdout=(
                '{"type":"thread.started","thread_id":"t1"}\n'
                '{"type":"turn.failed","error":{"message":"at capacity"}}\n'))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", retry_without_output)
    monkeypatch.setattr("verifier.llm.codex_backend.time.sleep", lambda _: None)
    with pytest.raises(RoleError, match="produced no output"):
        backend.complete("critic", "prompt", {})
    assert len(attempts) == 2


def test_unknown_or_config_no_thread_failure_does_not_backoff(
    tmp_path, monkeypatch,
):
    usage_path = tmp_path / "seat-usage.jsonl"
    backend = CodexBackend(
        roles={"critic": SimpleNamespace(system="", schema={"ok": bool})},
        cwd=str(tmp_path),
    )
    calls = []
    sleeps = []

    def config_failure(_cmd, **_kwargs):
        calls.append(1)
        return SimpleNamespace(
            returncode=1, stdout="", stderr=(
                "Error loading config.toml: unknown configuration field "
                "tools.view_image in -c/--config override"),
        )

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", config_failure)
    monkeypatch.setattr(
        "verifier.llm.codex_backend.time.sleep", sleeps.append)
    with codex_usage_ledger(
        str(usage_path), {str(tmp_path): {"trial": "trial-config"}},
    ):
        with pytest.raises(RoleError, match="codex exec failed"):
            backend.complete("critic", "prompt", {})

    assert len(calls) == 1
    assert sleeps == []
    row, = _usage_rows(usage_path)
    assert row["attempt_status"] == "nonzero_exit"
    assert row["usage_status"] == "unknown"
    _assert_unknown_usage(row)


@pytest.mark.parametrize("stderr", [
    "HTTP status 503 from provider",
    "upstream service temporarily unavailable",
])
def test_explicit_no_thread_provider_failure_retries(
    tmp_path, monkeypatch, stderr,
):
    backend = CodexBackend(
        roles={"critic": SimpleNamespace(system="", schema={"ok": bool})},
        cwd=str(tmp_path),
    )
    calls = []

    def transient_then_success(cmd, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr=stderr)
        with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process",
        transient_then_success)
    monkeypatch.setattr("verifier.llm.codex_backend.time.sleep", lambda _: None)

    assert backend.complete("critic", "prompt", {}) == {"ok": True}
    assert len(calls) == 2


def test_auth_failure_saying_try_again_does_not_retry(tmp_path, monkeypatch):
    backend = CodexBackend(
        roles={"critic": SimpleNamespace(system="", schema={"ok": bool})},
        cwd=str(tmp_path),
    )
    calls = []

    def auth_failure(_cmd, **_kwargs):
        calls.append(1)
        return SimpleNamespace(
            returncode=1, stderr="", stdout=json.dumps({
                "type": "error",
                "message": "Authentication failed; login and try again",
            }),
        )

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", auth_failure)
    with pytest.raises(RoleError, match="codex exec failed"):
        backend.complete("critic", "prompt", {})

    assert len(calls) == 1


def test_usage_ledger_records_exact_repeated_seat_calls(
    tmp_path, monkeypatch,
):
    usage_path = tmp_path / "seat-usage.jsonl"
    role = SimpleNamespace(system="", schema={"ok": bool})
    backend = CodexBackend(
        roles={"critic": role}, cwd=str(tmp_path),
        model="gpt-5.6-terra", effort="medium",
        service_tier="fast",
    )
    invocations = []

    def fake_run(cmd, **_kwargs):
        index = len(invocations) + 1
        invocations.append(cmd)
        output = cmd[cmd.index("-o") + 1]
        with open(output, "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        events = [
            {"type": "thread.started", "thread_id": f"thread-{index}"},
            {"type": "item.completed", "item": {
                "type": "agent_message", "text": '{"ok": true}',
            }},
            {"type": "turn.completed", "usage": {
                "input_tokens": 100 + index,
                "cached_input_tokens": 60,
                "cache_write_input_tokens": 0,
                "output_tokens": 12,
                "reasoning_output_tokens": 7,
            }},
        ]
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(json.dumps(event) for event in events) + "\n",
            stderr="",
        )

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    context = {
        "job": "rec-codex-terra",
        "task": "task-a",
        "trial": "trial-a",
        "config": "config-a",
        "rep": "r1",
    }
    with codex_usage_ledger(
        str(usage_path), {str(tmp_path): context},
    ):
        assert backend.complete("critic", "secret prompt", {}) == {"ok": True}
        assert backend.complete("critic", "secret prompt", {}) == {"ok": True}

    assert all("--json" in command for command in invocations)
    for command in invocations:
        assert 'service_tier="fast"' in command
        assert command[command.index("--enable") + 1] == "fast_mode"
    rows = _usage_rows(usage_path)
    assert len(rows) == 2
    assert [row["thread_id"] for row in rows] == ["thread-1", "thread-2"]
    assert len({row["call_id"] for row in rows}) == 2
    assert len({row["attempt_id"] for row in rows}) == 2
    assert [row["input_tokens"] for row in rows] == [101, 102]
    assert [row["uncached_input_tokens"] for row in rows] == [41, 42]
    for row in rows:
        assert {key: row[key] for key in context} == context
        assert row["role"] == "critic"
        assert row["model"] == "gpt-5.6-terra"
        assert row["effort"] == "medium"
        assert row["service_tier"] == "fast"
        assert row["cached_input_tokens"] == 60
        assert row["output_tokens"] == 12
        assert row["reasoning_tokens"] == 7
        assert row["record_type"] == "codex_cli_attempt"
        assert row["attempt_schema_version"] == 1
        assert row["attempt_index"] == 1
        assert row["started"] is True
        assert row["attempt_status"] == "succeeded"
        assert row["returncode"] == 0
        assert row["usage_status"] == "complete"
        assert row["usage_complete"] is True
        assert row["usage_error"] is None
        assert row["error_kind"] is None
        assert row["error"] is None
        assert row["duration_ns"] >= 0
        assert row["duration_seconds"] == row["duration_ns"] / 1_000_000_000
        assert "secret prompt" not in json.dumps(row)


def test_oracle_gets_compile_boundary_instead_of_execution_inspection(
        tmp_path, monkeypatch):
    role = SimpleNamespace(system="seat", schema={"ok": bool})
    backend = CodexBackend(
        roles={"oracles": role, "discharge": role}, cwd=str(tmp_path),
        compact_inspection=True,
    )
    instructions = {}

    def fake_run(cmd, **_kwargs):
        role_name = "oracles" if "ORACLE COMPILE/EXECUTE BOUNDARY" in cmd[-1] \
            else "discharge"
        instructions[role_name] = cmd[-1]
        output = cmd[cmd.index("-o") + 1]
        with open(output, "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(
            returncode=0,
            stdout=_complete_usage_stream(f"thread-{role_name}"),
            stderr="",
        )

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    assert backend.complete("oracles", "prompt", {}) == {"ok": True}
    assert backend.complete("discharge", "prompt", {}) == {"ok": True}

    assert "ORACLE COMPILE/EXECUTE BOUNDARY" in instructions["oracles"]
    assert "CUSTOM METHOD COMPLETENESS" in instructions["oracles"]
    assert "LOCAL EVIDENCE INSPECTION" not in instructions["oracles"]
    assert "LOCAL EVIDENCE INSPECTION" in instructions["discharge"]
    assert "PLAN-FIRST EXECUTION" in instructions["discharge"]
    assert "ORACLE COMPILE/EXECUTE BOUNDARY" not in instructions["discharge"]
    assert "PLAN-FIRST EXECUTION" not in instructions["oracles"]


def test_codex_explicitly_rejects_compact_source_evidence_for_every_role_mode():
    backend = CodexBackend()

    for role_name in (
            "obligations", "critic", "oracles", "discharge", "judger",
            "extract"):
        for world_mode in (
                "live", "captured", "unavailable", "authority-only"):
            assert backend.allows_compact_source_evidence(
                role_name, {"world_mode": world_mode}) is False


def test_service_tier_args_explicitly_override_user_config():
    assert service_tier_cli_args("standard") == [
        "-c", 'service_tier="default"', "--disable", "fast_mode",
    ]
    assert service_tier_cli_args("fast") == [
        "-c", 'service_tier="fast"', "--enable", "fast_mode",
    ]
    with pytest.raises(ValueError, match="service tier"):
        service_tier_cli_args("turbo")


def test_usage_ledger_counts_schema_failed_call_once(tmp_path, monkeypatch):
    """Usage belongs to the backend invocation even when its final answer is
    malformed.  BindingGame records the corresponding Judger error event; it
    must not add a second synthetic token row."""
    usage_path = tmp_path / "seat-usage.jsonl"
    role = SimpleNamespace(system="", schema={"ok": bool})
    backend = CodexBackend(
        roles={"judger": role}, cwd=str(tmp_path),
        model="gpt-5.6-terra", effort="medium",
    )

    def fake_run(cmd, **_kwargs):
        output = cmd[cmd.index("-o") + 1]
        with open(output, "w", encoding="utf-8") as handle:
            handle.write("malformed final answer")
        events = [
            {"type": "thread.started", "thread_id": "thread-bad-json"},
            {"type": "item.completed", "item": {
                "type": "agent_message", "text": "malformed final answer",
            }},
            {"type": "turn.completed", "usage": {
                "input_tokens": 321,
                "cached_input_tokens": 256,
                "output_tokens": 17,
                "reasoning_output_tokens": 9,
            }},
        ]
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(json.dumps(event) for event in events) + "\n",
            stderr="",
        )

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    with codex_usage_ledger(
        str(usage_path), {str(tmp_path): {"trial": "trial-bad-json"}},
    ):
        with pytest.raises(RoleError, match="no JSON object"):
            backend.complete("judger", "prompt", {})

    rows = _usage_rows(usage_path)
    assert len(rows) == 1
    assert rows[0]["thread_id"] == "thread-bad-json"
    assert rows[0]["role"] == "judger"
    assert rows[0]["attempt_status"] == "schema_error"
    assert rows[0]["usage_status"] == "complete"
    assert rows[0]["usage_complete"] is True
    assert rows[0]["input_tokens"] == 321
    assert rows[0]["output_tokens"] == 17


@pytest.mark.parametrize(
    ("terminal_event", "message", "usage_status"),
    [
        (None, "turn.completed usage", "partial"),
        ({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 60,
                "output_tokens": 12,
                # reasoning_output_tokens is required for exact accounting.
            },
        }, "invalid reasoning_output_tokens", "invalid"),
        ({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 101,
                "output_tokens": 12,
                "reasoning_output_tokens": 7,
            },
        }, "cached_input_tokens greater than input_tokens", "invalid"),
        ({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 60,
                "output_tokens": 12,
                "reasoning_output_tokens": "usage-secret",
            },
        }, "invalid reasoning_output_tokens", "invalid"),
    ],
)
def test_usage_ledger_rejects_missing_or_partial_usage(
    tmp_path, monkeypatch, terminal_event, message, usage_status,
):
    usage_path = tmp_path / "seat-usage.jsonl"
    role = SimpleNamespace(system="", schema={"ok": bool})
    backend = CodexBackend(roles={"critic": role}, cwd=str(tmp_path))

    def fake_run(cmd, **_kwargs):
        output = cmd[cmd.index("-o") + 1]
        with open(output, "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        events = [{"type": "thread.started", "thread_id": "thread-1"}]
        if terminal_event is not None:
            events.append(terminal_event)
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(json.dumps(event) for event in events) + "\n",
            stderr="",
        )

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    with codex_usage_ledger(
        str(usage_path), {str(tmp_path): {"trial": "trial-a"}},
    ):
        with pytest.raises(RoleError, match=message):
            backend.complete("critic", "prompt", {})

    rows = _usage_rows(usage_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["attempt_status"] == "telemetry_error"
    assert row["usage_status"] == usage_status
    assert row["usage_error"] == {
        "partial": "usage telemetry was incomplete",
        "invalid": "usage telemetry failed validation",
    }[usage_status]
    assert row["started"] is True
    assert row["returncode"] == 0
    assert row["thread_id"] == "thread-1"
    _assert_unknown_usage(row)
    assert "usage-secret" not in json.dumps(row)


def test_usage_ledger_records_internal_provider_retry(tmp_path, monkeypatch):
    usage_path = tmp_path / "seat-usage.jsonl"
    role = SimpleNamespace(system="", schema={"ok": bool})
    backend = CodexBackend(roles={"critic": role}, cwd=str(tmp_path))
    attempts = []

    def fake_run(cmd, **_kwargs):
        attempts.append(cmd)
        if len(attempts) == 1:
            events = [
                {"type": "thread.started", "thread_id": "thread-retry"},
                {"type": "turn.failed", "error": {
                    "message": "service temporarily unavailable",
                }},
            ]
            return SimpleNamespace(
                returncode=1, stderr="sensitive provider detail",
                stdout="\n".join(json.dumps(event) for event in events),
            )
        with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(
            returncode=0, stderr="",
            stdout=_complete_usage_stream("thread-success"),
        )

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    monkeypatch.setattr("verifier.llm.codex_backend.time.sleep", lambda _: None)
    with codex_usage_ledger(
        str(usage_path), {str(tmp_path): {"trial": "trial-retry"}},
    ):
        assert backend.complete("critic", "secret prompt", {}) == {"ok": True}

    rows = _usage_rows(usage_path)
    assert len(rows) == 2
    assert [row["attempt_index"] for row in rows] == [1, 2]
    assert [row["attempt_status"] for row in rows] == [
        "provider_retry", "succeeded"]
    assert len({row["attempt_id"] for row in rows}) == 2
    assert len({row["call_id"] for row in rows}) == 1
    assert rows[0]["usage_status"] == "partial"
    _assert_unknown_usage(rows[0])
    assert rows[1]["usage_complete"] is True
    serialized = json.dumps(rows)
    assert "secret prompt" not in serialized
    assert "sensitive provider detail" not in serialized


def test_usage_ledger_records_exhausted_provider_retry(tmp_path, monkeypatch):
    usage_path = tmp_path / "seat-usage.jsonl"
    role = SimpleNamespace(system="", schema={"ok": bool})
    backend = CodexBackend(roles={"critic": role}, cwd=str(tmp_path))

    def fake_run(_cmd, **_kwargs):
        return SimpleNamespace(
            returncode=1, stderr="", stdout=(
                '{"type":"thread.started","thread_id":"retry"}\n'
                '{"type":"turn.failed","error":'
                '{"message":"at capacity"}}\n'),
        )

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    monkeypatch.setattr("verifier.llm.codex_backend.TRANSIENT_RETRIES", 1)
    monkeypatch.setattr("verifier.llm.codex_backend.time.sleep", lambda _: None)
    with codex_usage_ledger(
        str(usage_path), {str(tmp_path): {"trial": "trial-exhausted"}},
    ):
        with pytest.raises(RoleError, match="transient codex failure persisted"):
            backend.complete("critic", "prompt", {})

    rows = _usage_rows(usage_path)
    assert [row["attempt_status"] for row in rows] == [
        "provider_retry", "provider_exhausted"]
    assert [row["attempt_index"] for row in rows] == [1, 2]
    assert len({row["call_id"] for row in rows}) == 1
    for row in rows:
        _assert_unknown_usage(row)


def test_usage_ledger_records_nonzero_exit_without_fake_tokens(
    tmp_path, monkeypatch,
):
    usage_path = tmp_path / "seat-usage.jsonl"
    role = SimpleNamespace(system="", schema={"ok": bool})
    backend = CodexBackend(roles={"critic": role}, cwd=str(tmp_path))

    def fake_run(_cmd, **_kwargs):
        return SimpleNamespace(
            returncode=2, stderr="sensitive stderr",
            stdout='{"type":"thread.started","thread_id":"failed"}\n',
        )

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    with codex_usage_ledger(
        str(usage_path), {str(tmp_path): {"trial": "trial-nonzero"}},
    ):
        with pytest.raises(RoleError, match="codex exec failed"):
            backend.complete("critic", "prompt", {})

    row, = _usage_rows(usage_path)
    assert row["attempt_status"] == "nonzero_exit"
    assert row["started"] is True
    assert row["returncode"] == 2
    assert row["usage_status"] == "partial"
    _assert_unknown_usage(row)
    assert "sensitive stderr" not in json.dumps(row)


def test_usage_ledger_timeout_never_trusts_partial_terminal_usage(
    tmp_path, monkeypatch,
):
    usage_path = tmp_path / "seat-usage.jsonl"
    role = SimpleNamespace(system="", schema={"ok": bool})
    backend = CodexBackend(roles={"critic": role}, cwd=str(tmp_path))

    def fake_run(cmd, **_kwargs):
        error = subprocess.TimeoutExpired(
            cmd, 1, output=_complete_usage_stream("thread-timeout"))
        error.stdout = error.output
        error._gv_process_started = True
        raise error

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    with codex_usage_ledger(
        str(usage_path), {str(tmp_path): {"trial": "trial-timeout"}},
    ):
        with pytest.raises(RoleError, match="timed out"):
            backend.complete("critic", "prompt", {})

    row, = _usage_rows(usage_path)
    assert row["attempt_status"] == "timeout"
    assert row["started"] is True
    assert row["returncode"] is None
    assert row["thread_id"] == "thread-timeout"
    assert row["usage_status"] == "partial"
    _assert_unknown_usage(row)


def test_usage_ledger_distinguishes_launch_failure(tmp_path, monkeypatch):
    usage_path = tmp_path / "seat-usage.jsonl"
    role = SimpleNamespace(system="", schema={"ok": bool})
    backend = CodexBackend(roles={"critic": role}, cwd=str(tmp_path))

    def fake_run(_cmd, **_kwargs):
        raise FileNotFoundError("sensitive executable path")

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    with codex_usage_ledger(
        str(usage_path), {str(tmp_path): {"trial": "trial-launch"}},
    ):
        with pytest.raises(FileNotFoundError):
            backend.complete("critic", "prompt", {})

    row, = _usage_rows(usage_path)
    assert row["attempt_status"] == "launch_error"
    assert row["started"] is False
    assert row["returncode"] is None
    assert row["usage_status"] == "unknown"
    _assert_unknown_usage(row)
    assert "sensitive executable path" not in json.dumps(row)


def test_usage_ledger_records_started_process_interruption(
    tmp_path, monkeypatch,
):
    usage_path = tmp_path / "seat-usage.jsonl"
    role = SimpleNamespace(system="", schema={"ok": bool})
    backend = CodexBackend(roles={"critic": role}, cwd=str(tmp_path))

    def fake_run(_cmd, **_kwargs):
        error = KeyboardInterrupt()
        error._gv_process_started = True
        raise error

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    with codex_usage_ledger(
        str(usage_path), {str(tmp_path): {"trial": "trial-interrupt"}},
    ):
        with pytest.raises(KeyboardInterrupt):
            backend.complete("critic", "prompt", {})

    row, = _usage_rows(usage_path)
    assert row["attempt_status"] == "interrupted"
    assert row["started"] is True
    assert row["usage_status"] == "unknown"
    _assert_unknown_usage(row)


def test_usage_ledger_records_output_processing_interruption(
    tmp_path, monkeypatch,
):
    usage_path = tmp_path / "seat-usage.jsonl"
    role = SimpleNamespace(system="", schema={"ok": bool})
    backend = CodexBackend(roles={"critic": role}, cwd=str(tmp_path))

    def fake_run(cmd, **_kwargs):
        with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(
            returncode=0, stderr="", stdout=_complete_usage_stream())

    def interrupt_extract(_text, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    monkeypatch.setattr(
        "verifier.llm.codex_backend.extract_json", interrupt_extract)
    with codex_usage_ledger(
        str(usage_path), {str(tmp_path): {"trial": "trial-postprocess"}},
    ):
        with pytest.raises(KeyboardInterrupt):
            backend.complete("critic", "prompt", {})

    row, = _usage_rows(usage_path)
    assert row["attempt_status"] == "interrupted"
    assert row["started"] is True
    assert row["usage_status"] == "complete"
    assert row["usage_complete"] is True
    assert row["input_tokens"] == 100


def test_capability_hooks_force_json_and_validate_before_return(
    tmp_path, monkeypatch,
):
    class RestrictedBackend(CodexBackend):
        def _execution_cwd(self, role_name, variables):
            return str(tmp_path / "empty-scratch")

        def _search_enabled(self, role_name):
            return False

        def _requires_json(self):
            return True

        def _validate_attempt_stream(self, stdout):
            raise RoleError("stream policy rejected tool activity")

    (tmp_path / "empty-scratch").mkdir()
    role = SimpleNamespace(system="", schema={"ok": bool})
    backend = RestrictedBackend(roles={"critic": role}, cwd=str(tmp_path))
    observed = {}

    def fake_run(cmd, **kwargs):
        observed["cmd"] = cmd
        observed["cwd"] = kwargs["cwd"]
        with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(
            returncode=0, stderr="", stdout=_complete_usage_stream())

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    with pytest.raises(RoleError, match="stream policy rejected"):
        backend.complete("critic", "prompt", {})

    assert "--json" in observed["cmd"]
    assert "tools.web_search=true" not in observed["cmd"]
    assert observed["cwd"] == str(tmp_path / "empty-scratch")


def test_stream_policy_failure_has_one_output_error_attempt(
    tmp_path, monkeypatch,
):
    usage_path = tmp_path / "seat-usage.jsonl"

    class RestrictedBackend(CodexBackend):
        def _validate_attempt_stream(self, stdout):
            raise RoleError("sensitive stream contents")

    role = SimpleNamespace(system="", schema={"ok": bool})
    backend = RestrictedBackend(roles={"critic": role}, cwd=str(tmp_path))

    def fake_run(cmd, **_kwargs):
        with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(
            returncode=0, stderr="", stdout=_complete_usage_stream())

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    with codex_usage_ledger(
        str(usage_path), {str(tmp_path): {"trial": "trial-policy"}},
    ):
        with pytest.raises(RoleError, match="sensitive stream contents"):
            backend.complete("critic", "prompt", {})

    row, = _usage_rows(usage_path)
    assert row["attempt_status"] == "output_error"
    assert row["error_kind"] == "attempt_stream_policy"
    assert row["usage_complete"] is True
    assert "sensitive stream contents" not in json.dumps(row)


@pytest.mark.parametrize(
    "role", ["obligations", "critic", "oracles", "discharge", "judger"])
@pytest.mark.parametrize("world_mode", ["captured", "unavailable"])
def test_all_offline_seats_get_isolated_execution(
    tmp_path, role, world_mode,
):
    backend = CodexBackend(cwd=str(tmp_path))
    args = backend._execution_policy_args(
        role, {"world_mode": world_mode})

    assert "--dangerously-bypass-approvals-and-sandbox" not in args
    assert "--sandbox" not in args  # a permission profile replaces it
    assert {"--ignore-user-config", "--ignore-rules", "--strict-config",
            "--ephemeral"} <= set(args)
    assert 'default_permissions="gv-offline"' in args
    assert 'approval_policy="never"' in args
    assert 'permissions.gv-offline.extends=":read-only"' in args
    # Network is deliberately ON: pinned instrument toolchains (grpcio-tools,
    # protoc, build deps) must be installable inside the profile; a DNS-dead
    # sandbox converts those checks into structural unobtainables.
    assert "permissions.gv-offline.network.enabled=true" in args
    assert 'shell_environment_policy.inherit="core"' in args

    config = _cli_config(args)
    policy = config["permissions"]["gv-offline"]
    filesystem = policy["filesystem"]
    work = os.path.join(str(tmp_path), "work")
    assert filesystem[":root"] == "deny"
    assert filesystem[":minimal"] == "read"
    assert filesystem[str(tmp_path)] == "read"
    assert filesystem[work] == "write"
    for denied in (
        "~/.cache", "~/Library/Caches", "~/.huggingface", "~/.codex",
    ):
        assert filesystem[denied] == "deny"

    assert config["project_doc_max_bytes"] == 0
    assert config["mcp_servers"] == {}
    assert config["features"]["skip_host_skill_discovery"] is True
    for feature in (
        "apps", "plugins", "multi_agent", "memories", "browser_use",
        "computer_use", "in_app_browser", "workspace_dependencies",
        "hooks", "skill_search", "image_generation", "shell_snapshot",
    ):
        assert config["features"][feature] is False
    # Offline evidence still needs no-Node code-mode orchestration and native
    # inspection of generated renderings. Neither is disabled by this policy.
    assert config["features"].get("code_mode_host") is not False
    assert config["features"].get("view_image") is not False
    assert config["tools"]["web_search"] is (role in {
        "obligations", "critic", "oracles",
    })

    home = os.path.join(work, "_offline_home")
    env = next(
        arg for arg in args
        if arg.startswith("shell_environment_policy.set=")
    )
    assert f'HOME={json.dumps(home)}' in env
    assert f'XDG_CACHE_HOME={json.dumps(os.path.join(home, ".cache"))}' in env
    assert f'TMPDIR={json.dumps(os.path.join(home, "tmp"))}' in env
    for key in (
        "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_XET_CACHE",
        "HF_ASSETS_CACHE",
    ):
        assert f"{key}=" in env
    assert os.path.isdir(home)
    assert os.path.isdir(os.path.join(home, "tmp"))
    assert os.path.isdir(os.path.join(home, ".cache", "huggingface", "hub"))


@pytest.mark.parametrize(
    ("role", "world_mode"),
    [
        ("discharge", "live"),
        ("judger", "live"),
        ("obligations", "live"),
        ("critic", "live"),
        ("oracles", "live"),
    ],
)
def test_live_world_seats_retain_original_yolo_policy(
    tmp_path, role, world_mode,
):
    args = CodexBackend(cwd=str(tmp_path))._execution_policy_args(
        role, {"world_mode": world_mode})

    assert args == ["--dangerously-bypass-approvals-and-sandbox"]


def test_offline_extract_gets_isolated_execution(tmp_path):
    args = CodexBackend(cwd=str(tmp_path))._execution_policy_args(
        "extract", {})

    assert "--dangerously-bypass-approvals-and-sandbox" not in args
    assert 'permissions.gv-offline.extends=":read-only"' in args
    assert _cli_config(args)["tools"]["web_search"] is True


def test_offline_execution_can_build_in_the_durable_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    args = CodexBackend(cwd=str(tmp_path))._execution_policy_args(
        "discharge", {"world_mode": "captured", "workspace_dir": str(workspace)})
    filesystem = _cli_config(args)["permissions"]["gv-offline"]["filesystem"]
    assert filesystem[str(workspace)] == "write"
    assert workspace.is_dir()
    assert filesystem[str(tmp_path)] == "read"


def test_offline_execution_rejects_a_symlinked_workspace(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = tmp_path / "workspace"
    workspace.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RoleError, match="symlink path components"):
        CodexBackend(cwd=str(tmp_path))._execution_policy_args(
            "discharge", {"world_mode": "captured", "workspace_dir": str(workspace)})


def test_offline_execution_rejects_workspace_outside_state(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    workspace = tmp_path / "other" / "workspace"
    workspace.mkdir(parents=True)
    with pytest.raises(RoleError, match="escapes the state workspace"):
        CodexBackend(cwd=str(state))._execution_policy_args(
            "discharge", {
                "world_mode": "captured", "workspace_dir": str(workspace),
            })


def test_offline_execution_rejects_workdir_outside_work(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    outside = tmp_path / "other"
    outside.mkdir()
    with pytest.raises(RoleError, match="workdir escapes"):
        CodexBackend(cwd=str(state))._execution_policy_args(
            "discharge", {
                "world_mode": "captured", "workdir": str(outside),
            })


@pytest.mark.parametrize("key", ["toolchain_dir", "shared_cache_dir"])
def test_offline_execution_rejects_shared_root_containing_state(tmp_path, key):
    state = tmp_path / "scoring" / "candidate" / "gv"
    state.mkdir(parents=True)
    root = tmp_path / "scoring"
    kwargs = {
        ("offline_toolchain_dir" if key == "toolchain_dir"
         else "offline_shared_cache_dir"): str(root),
    }
    expected = ("overlaps the isolated state" if key == "toolchain_dir"
                else "overlaps the isolated state|must be disjoint")
    with pytest.raises(RoleError, match=expected):
        CodexBackend(cwd=str(state), **kwargs)._execution_policy_args(
            "discharge", {
                "world_mode": "captured", key: str(root),
            })


def test_offline_execution_rejects_symlink_in_state_path(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(RoleError, match="symlink path components"):
        CodexBackend(cwd=str(alias))._execution_policy_args(
            "critic", {"world_mode": "captured"})


def test_offline_execution_keeps_state_cwd_but_only_work_is_writable(
    tmp_path, monkeypatch,
):
    role = SimpleNamespace(system="", schema={"ok": bool})
    backend = CodexBackend(roles={"critic": role}, cwd=str(tmp_path))
    observed = {}

    def fake_run(cmd, **kwargs):
        observed["cwd"] = kwargs["cwd"]
        output = cmd[cmd.index("-o") + 1]
        with open(output, "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    assert backend.complete(
        "critic", "return ok", {"world_mode": "captured"},
    ) == {"ok": True}

    # task.md and baseline/ are harness-relative authoritative inputs, so the
    # seat must run at the state root.  The custom permission profile grants
    # writes only to state/work; cwd itself remains read-only.
    assert observed["cwd"] == str(tmp_path)


@pytest.mark.parametrize("link_name", ["work", "work/_offline_home"])
def test_offline_execution_rejects_writable_path_symlinks(
    tmp_path, link_name,
):
    state = tmp_path / "state"
    state.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = state / link_name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)

    backend = CodexBackend(cwd=str(state))
    with pytest.raises(RoleError, match="real directory"):
        backend._execution_policy_args(
            "critic", {"world_mode": "captured"})


def test_offline_evidence_can_derive_state_from_ledger(tmp_path):
    ledger = tmp_path / "events.jsonl"
    args = CodexBackend()._execution_policy_args(
        "judger",
        {"world_mode": "unavailable", "ledger_path": str(ledger)},
    )

    env = next(
        arg for arg in args
        if arg.startswith("shell_environment_policy.set=")
    )
    expected = os.path.join(str(tmp_path), "work", "_offline_home")
    assert f'HOME={json.dumps(expected)}' in env


def test_offline_evidence_without_a_state_fails_closed():
    with pytest.raises(RoleError, match="needs a state cwd"):
        CodexBackend()._execution_policy_args(
            "discharge", {"world_mode": "captured"})


def test_world_image_bytes_are_shared_by_discharge_and_judger(
    tmp_path, monkeypatch,
):
    world, manifest = _captured_state(
        tmp_path, {"app/board.png": PNG_A})
    backend = CodexBackend(cwd=str(tmp_path))
    calls = []
    payloads = [
        {"outcome": "holds", "witness": "", "artifact": "report.json",
         "assumed_premises": []},
        {"sufficient": "yes", "faithful": "yes", "degradation": "n/a",
         "reason": "audited"},
    ]

    def fake_run(cmd, **_kwargs):
        paths = [cmd[index + 1] for index, arg in enumerate(cmd)
                 if arg == "-i"]
        calls.append({
            "bytes": [open(path, "rb").read() for path in paths],
            "instruction": cmd[-1],
            "image_positions": [cmd.index(path) for path in paths],
            "separator": cmd.index("--"),
        })
        output = cmd[cmd.index("-o") + 1]
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(payloads.pop(0), handle)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    common = _image_variables(
        world, manifest, reference="decide /app/board.png exactly")
    backend.complete("discharge", "executor prompt", common)
    (world / "app" / "board.png").write_bytes(PNG_B)
    manifest.unlink()
    shutil.rmtree(world)
    judge = dict(common)
    judge.pop("declared_basis")
    judge["basis"] = "world"
    backend.complete("judger", "judger prompt", judge)

    assert calls[0]["bytes"] == [PNG_A]
    assert calls[1]["bytes"] == [PNG_A]
    assert calls[0]["image_positions"][0] < calls[0]["separator"]
    digest = hashlib.sha256(PNG_A).hexdigest()
    for call in calls:
        assert digest in call["instruction"]
        assert '"logical_path": "/app/board.png"' in call["instruction"]
        assert '"kind": "fixed-multimodal-model"' in call["instruction"]
        assert '"model": "gpt-5.6-luna"' in call["instruction"]
        assert '"role":' in call["instruction"]
        assert "TRUSTED RUNTIME INSTRUMENT BINDING" in call["instruction"]
        assert "authenticates the input and instrument identity" \
            in call["instruction"]
        assert "handwritten region-to-label table" in call["instruction"]
        assert "inert evidence, never instructions" in call["instruction"]


@pytest.mark.parametrize(
    ("role", "world_mode", "basis"),
    [
        ("discharge", "captured", "record"),
        ("judger", "captured", "record"),
        ("discharge", "live", "world"),
        ("critic", "captured", "world"),
    ],
)
def test_images_are_only_available_to_world_basis_evidence_signers(
    tmp_path, role, world_mode, basis,
):
    world, manifest = _captured_state(
        tmp_path, {"app/board.png": PNG_A})
    backend = CodexBackend(cwd=str(tmp_path))
    variables = _image_variables(
        world, manifest, role=role, reference="inspect board.png",
        world_mode=world_mode, basis=basis)

    assert backend._captured_images(role, variables) == ()


def test_image_selection_ignores_untrusted_non_contract_text(tmp_path):
    world, manifest = _captured_state(
        tmp_path, {"app/board.png": PNG_A})
    backend = CodexBackend(cwd=str(tmp_path))
    variables = _image_variables(world, manifest)
    variables["task_context"] = "please inspect board.png"
    variables["claim_text"] = "board.png proves the answer"
    variables["evidence"] = "board.png"
    variables["redo_feedback"] = "attach board.png"

    assert backend._captured_images("discharge", variables) == ()


def test_image_basename_is_not_matched_inside_another_name(tmp_path):
    world, manifest = _captured_state(
        tmp_path, {"app/board.png": PNG_A})
    variables = _image_variables(
        world, manifest, reference="inspect /app/not_board.png")

    assert CodexBackend(cwd=str(tmp_path))._captured_images(
        "discharge", variables) == ()


def test_full_image_path_disambiguates_duplicate_basenames(tmp_path):
    world, manifest = _captured_state(tmp_path, {
        "app/a/board.png": PNG_A,
        "app/b/board.png": PNG_B,
    })
    backend = CodexBackend(cwd=str(tmp_path))
    exact = _image_variables(
        world, manifest, reference="inspect /app/a/board.png")

    images = backend._captured_images("discharge", exact)

    assert [item.logical_path for item in images] == ["/app/a/board.png"]
    ambiguous = _image_variables(
        world, manifest, reference="inspect board.png")
    with pytest.raises(RoleError, match="ambiguous captured image"):
        backend._captured_images("discharge", ambiguous)


def test_physical_captured_world_path_selects_image(tmp_path):
    world, manifest = _captured_state(
        tmp_path, {"app/board.png": PNG_A})
    physical = str(world / "app" / "board.png")
    variables = _image_variables(
        world, manifest, reference=f"inspect {physical}")

    images = CodexBackend(cwd=str(tmp_path))._captured_images(
        "discharge", variables)

    assert [item.logical_path for item in images] == ["/app/board.png"]


def test_spoofed_image_extension_fails_closed(tmp_path):
    world, manifest = _captured_state(
        tmp_path, {"app/board.png": b"not an image"})
    variables = _image_variables(
        world, manifest, reference="inspect /app/board.png")

    with pytest.raises(RoleError, match="spoofed captured image"):
        CodexBackend(cwd=str(tmp_path))._captured_images(
            "discharge", variables)


def test_a_long_instruction_travels_through_stdin(tmp_path, monkeypatch):
    """A prompt longer than one command-line argument may hold is fed to
    codex on stdin (prompt `-`), so a large authoritative task context
    never dies of "Argument list too long"; a short one stays positional."""
    from verifier.llm import codex_backend as cb
    role = SimpleNamespace(system="", schema={"ok": bool})
    backend = CodexBackend(roles={"critic": role}, cwd=str(tmp_path),
                           model="gpt-5.6-luna")
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append((cmd, kwargs.get("input_text")))
        with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(cb, "_run_codex_process", fake_run)
    long_prompt = "x" * (cb.ARGV_INSTRUCTION_LIMIT + 10)
    assert backend.complete("critic", long_prompt, {}) == {"ok": True}
    cmd, fed = seen[-1]
    assert cmd[-2:] == ["--", "-"]
    assert fed is not None and long_prompt in fed

    assert backend.complete("critic", "short prompt", {}) == {"ok": True}
    cmd, fed = seen[-1]
    assert cmd[-2] == "--" and "short prompt" in cmd[-1] and fed is None
