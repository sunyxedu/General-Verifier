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
import threading
from concurrent.futures import ThreadPoolExecutor
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
from verifier.llm import codex_backend as cb


@pytest.fixture(autouse=True)
def forbid_real_codex_invocation(monkeypatch):
    """Fail before spawning a model CLI if a process mock is misplaced.

    Local Python process-group fixtures remain real. Test-specific Codex
    interceptors may replace this wrapper but call it only with Python.
    """
    original = subprocess.Popen

    def guarded(cmd, *args, **kwargs):
        if not isinstance(cmd, str) and any(
                isinstance(part, str) and os.path.basename(part) == "codex" for part in cmd):
            raise AssertionError("real Codex invocation is forbidden in this unit suite")
        return original(cmd, *args, **kwargs)

    monkeypatch.setattr("verifier.llm.codex_backend.subprocess.Popen", guarded)


def _observed_synthetic_backend(tmp_path, monkeypatch, program, observer):
    """A real local Python child replaces Codex; never a model/API call."""
    original = subprocess.Popen
    commands = []

    def synthetic(cmd, **kwargs):
        commands.append(cmd)
        assert "--json" in cmd
        assert "timeout" not in kwargs
        out = cmd[cmd.index("-o") + 1]
        return original([sys.executable, "-u", "-c", program, out], **kwargs)

    monkeypatch.setattr("verifier.llm.codex_backend.subprocess.Popen", synthetic)
    backend = CodexBackend(roles={"critic": SimpleNamespace(system="", schema={"ok": bool})},
                           cwd=str(tmp_path), timeout=None, phase_observer=observer)
    backend.phase_heartbeat_seconds = 0.02
    return backend, commands


def test_observed_exec_drains_both_large_pipes_and_redacts_payloads(tmp_path, monkeypatch):
    records = []
    secret = "synthetic-secret-never-log"
    program = r'''
import json,os,sys,time
with open(sys.argv[1],"w") as f: json.dump({"ok":True},f)
def emit(value): os.write(1,(json.dumps(value)+"\n").encode())
emit({"type":"thread.started","thread_id":"synthetic-secret-never-log"})
emit({"type":"turn.started"})
emit({"type":"item.started","item":{"id":"synthetic-secret-never-log","type":"command_execution","command":"synthetic-secret-never-log"}})
os.write(2,b"synthetic-secret-never-log"*130000)
emit({"type":"item.completed","item":{"id":"synthetic-secret-never-log","type":"command_execution","aggregated_output":"synthetic-secret-never-log"*130000}})
time.sleep(.07)
emit({"type":"turn.completed","usage":{}})
'''
    backend, commands = _observed_synthetic_backend(tmp_path, monkeypatch, program, records.append)
    assert backend.complete("critic", secret, {}) == {"ok": True}
    assert len(commands) == 1
    assert secret not in json.dumps(records)
    returned = next(row for row in records if row["phase"] == "exec_returned")
    assert returned["stdout_bytes"] > 3_000_000 and returned["stderr_bytes"] > 3_000_000
    assert returned["reader_failed"] is False
    assert any(row["phase"] == "exec_heartbeat" for row in records)
    assert any(row.get("wait_category") == "cli_activity_unknown" for row in records)
    assert returned["active_tool_items"] == 0 and returned["item_state_complete"] is False
    assert any(row.get("item_kind") == "command_execution" for row in records)
    assert any(row["phase"] == "exec_process_exited" for row in records)
    assert all("thread_id" not in row and "command" not in row and "message" not in row for row in records)
    assert all(row["role"] == "critic" for row in records)
    assert backend._phase_roles == {}


@pytest.mark.parametrize("command,category", [("gcc -c source.c", "compiler"),
    ("/usr/bin/clang++ -c source.cc", "compiler"), ("cmake -S . -B build", "build"),
    ("ninja -C build", "build"), ("python3 -m pytest", "test"), ("ctest", "test"),
    ("python3 script.py", "other"), ("bash -lc 'gcc source.c'", "other"),
    ("cd work && make", "other"), ("./gcc", "other"), ("/state/work/gcc", "other"),
    ("gcc source.c; ./test", "other"), ("echo gcc", "other")])
def test_command_metadata_categories_are_conservative_and_body_free(command, category):
    value = cb._command_metadata({"command": command, "status": "completed", "exit_code": 0,
                                  "aggregated_output": "汉字"})
    assert value["command_category"] == category
    assert value["command_sha256"] == hashlib.sha256(command.encode()).hexdigest()
    assert value["command_output_utf8_bytes"] == 6 and value["command_exit_code"] == 0
    assert "command" not in value and "aggregated_output" not in value and "input_bytes" not in value


def test_command_metadata_wrong_types_unknown_exit_and_truncation(monkeypatch):
    value = cb._command_metadata({"command": ["private"], "exit_code": True,
                                  "status": ["private"], "aggregated_output": b"private"})
    assert value == {"command_metadata_available": True, "command_category": "other", "command_status": "unknown"}
    monkeypatch.setattr(cb, "_COMMAND_TEXT_LIMIT", 3)
    value = cb._command_metadata({"command": "汉字", "aggregated_output": "\ud800"})
    assert "command_sha256" not in value and "command_output_utf8_bytes" not in value
    assert value["command_metadata_truncated"] is True


def test_command_stats_distinct_completions_repeat_unknown_exit_and_boundedness(monkeypatch):
    stats = cb._CommandStats()
    known = cb._command_metadata({"command": "gcc fixture.c", "exit_code": 0})
    assert stats.completed("one", known) is False
    assert stats.completed("one", known) is True
    stats.completed("two", cb._command_metadata({"command": "gcc fixture.c", "status": "failed"}))
    stats.completed("three", cb._command_metadata({"command": "make", "exit_code": 127}))
    value = stats.snapshot()
    assert value["completed_items"] == 3 and value["fingerprinted_items"] == 3
    assert value["unique_fingerprints"] == 2 and value["repeated_fingerprint_items"] == 1
    assert value["max_fingerprint_repetitions"] == 2 and value["duplicate_completions"] == 1
    assert value["success_exits"] == value["unknown_exits"] == value["nonzero_exits"] == 1
    monkeypatch.setattr(cb, "_COMMAND_ITEM_LIMIT", 1)
    monkeypatch.setattr(cb, "_COMMAND_COUNTER_LIMIT", 1)
    limited = cb._CommandStats()
    limited.completed("one", known)
    limited.completed("two", known)
    limited.completed("three", known)
    assert len(limited.seen) == len(limited.fingerprints) == 1
    value = limited.snapshot()
    assert value["partial"] and value["saturated"] and value["truncated"] and value["untracked_completions"] == 1
    assert value["completed_items"] == 1 and value["repeated_fingerprint_items"] == 0


def test_missing_command_anonymous_completion_and_failed_status_without_exit_are_unknown():
    stats = cb._CommandStats()
    metadata = cb._command_metadata({"status": "failed"})
    stats.completed("known-id", metadata)
    stats.completed(None, metadata)
    value = stats.snapshot()
    assert value["completed_items"] == 1 and value["untracked_completions"] == 1
    assert value["unknown_exits"] == 1 and value["nonzero_exits"] == 0
    assert value["fingerprinted_items"] == 0 and value["partial"]


def test_observed_command_metadata_dedups_completed_lifecycle_and_preserves_ephemeral(tmp_path, monkeypatch):
    records = []
    program = r'''
import json,os,sys
with open(sys.argv[1],"w") as f: json.dump({"ok":True},f)
def emit(event): os.write(1,(json.dumps(event)+"\n").encode())
emit({"type":"thread.started","thread_id":"private-id"})
emit({"type":"turn.started"})
for i in range(2):
 emit({"type":"item.started","item":{"type":"command_execution","id":str(i),"command":"gcc -c private-source.c"}})
 event={"type":"item.completed","item":{"type":"command_execution","id":str(i),"status":"completed","exit_code":0,"aggregated_output":"private-output"}}
 emit(event)
 if i==0: emit(event)
emit({"type":"turn.completed"})
'''
    backend, commands = _observed_synthetic_backend(tmp_path, monkeypatch, program, records.append)
    assert backend.complete("critic", "private-prompt", {"world_mode": "captured"}) == {"ok": True}
    assert "--ephemeral" in commands[0]
    returned = next(row for row in records if row["phase"] == "exec_returned")
    stats = returned["command_stats"]
    assert stats["available"] and stats["scope"] == "cli_attempt_prefix"
    assert stats["completed_items"] == 2 and stats["repeated_fingerprint_items"] == 1
    assert stats["duplicate_completions"] == 1 and stats["partial"]
    completed = [row for row in records if row.get("event_type") == "item.completed"]
    assert all(row["command_output_utf8_bytes"] == 14 for row in completed)
    assert all(row["command_category"] == "compiler" for row in completed
               if not row["command_duplicate_completion"])
    assert "command_sha256" not in next(row for row in completed if row["command_duplicate_completion"])
    assert "private-" not in json.dumps(records)
    assert all(not any(key in row for key in ("command", "aggregated_output", "input_bytes", "prompt_bytes", "thread_id", "item_id")) for row in records)


def test_observed_exec_handles_chunked_utf8_and_no_output_file(tmp_path, monkeypatch):
    records = []
    program = r'''
import json,os,sys,time
value={"type":"item.completed","item":{"type":"agent_message","text":json.dumps({"ok":True,"note":"汉字"},ensure_ascii=False)}}
raw=(json.dumps(value,ensure_ascii=False)+"\n").encode()
for index in range(0,len(raw),3):
 os.write(1,raw[index:index+3]);time.sleep(.001)
'''
    backend, _ = _observed_synthetic_backend(tmp_path, monkeypatch, program, records.append)
    assert backend.complete("critic", "synthetic prompt", {})["ok"] is True
    assert any(row.get("item_kind") == "agent_message" for row in records)
    assert "汉字" not in json.dumps(records, ensure_ascii=False)


def test_observed_bad_json_is_only_a_metadata_gap_with_valid_final_file(tmp_path, monkeypatch):
    records = []
    program = r'''
import json,os,sys
with open(sys.argv[1],"w") as f: json.dump({"ok":True},f)
os.write(1,b"not json\n{broken\n{\"type\":[]}\n{\"type\":\"item.started\",\"item\":{\"type\":[]}}\n")
'''
    backend, _ = _observed_synthetic_backend(tmp_path, monkeypatch, program, records.append)
    assert backend.complete("critic", "synthetic prompt", {}) == {"ok": True}
    assert backend._phase_roles == {}
    assert next(row for row in records if row["phase"] == "exec_returned")["unparsed_stdout_lines"] == 2


def test_phase_roles_stay_bound_to_parallel_call_ids_and_are_cleaned(tmp_path, monkeypatch):
    records = []
    roles = {name: SimpleNamespace(system="", schema={"ok": bool})
             for name in ("critic", "judger")}
    backend = CodexBackend(roles=roles, cwd=str(tmp_path), timeout=None,
                           phase_observer=records.append)
    barrier = threading.Barrier(2)
    expected = {}
    expected_lock = threading.Lock()

    def simultaneous(cmd, cwd, devnull, call_id, attempt):
        name = "critic" if cmd[-1].startswith("critic fixture") else "judger"
        with expected_lock:
            expected[call_id] = name
        barrier.wait(timeout=2)  # A fixture synchronization guard, not a role deadline.
        backend._phase(call_id, attempt, "exec_dispatch")
        backend._phase(call_id, attempt, "exec_returned")
        with open(cmd[cmd.index("-o") + 1], "w") as output:
            json.dump({"ok": True}, output)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(backend, "_run_observed_exec", simultaneous)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(backend.complete, name, name + " fixture", {})
                   for name in roles]
        assert [future.result() for future in pending] == [{"ok": True}, {"ok": True}]
    assert len(expected) == 2 and len(records) == 4
    assert all(record["role"] == expected[record["call_id"]] for record in records)
    assert backend._phase_roles == {}


def test_failed_observed_call_cleans_role_and_unknown_is_not_guessed(tmp_path, monkeypatch):
    records = []
    backend, _ = _observed_synthetic_backend(tmp_path, monkeypatch, 'print("not a final response")',
                                             records.append)
    with pytest.raises(RoleError, match="produced no output"):
        backend.complete("critic", "synthetic fixture", {})
    assert backend._phase_roles == {}
    assert all(record["role"] == "critic" for record in records)
    backend._phase("unassociated", 0, "exec_dispatch")
    assert records[-1]["role"] == "unknown"


def test_observer_write_failure_cannot_change_verdict(tmp_path, monkeypatch):
    def fail(record):
        raise OSError("synthetic-private-path-or-credential-do-not-echo")

    program = 'import json,sys; json.dump({"ok":True},open(sys.argv[1],"w")); print("{\\\"type\\\":\\\"turn.started\\\"}")'
    backend, _ = _observed_synthetic_backend(tmp_path, monkeypatch, program, fail)
    assert backend.complete("critic", "synthetic prompt", {}) == {"ok": True}


def test_exited_cli_with_inherited_open_pipe_has_visible_drain_wait(tmp_path, monkeypatch):
    records=[]
    program=r'''
import json,subprocess,sys
with open(sys.argv[1],"w") as f: json.dump({"ok":True},f)
subprocess.Popen([sys.executable,"-c","import time;time.sleep(.12)"],start_new_session=True)
'''
    backend,_=_observed_synthetic_backend(tmp_path,monkeypatch,program,records.append)
    assert backend.complete("critic","synthetic task",{}) == {"ok":True}
    drain=[row for row in records if row["phase"]=="pipe_drain_heartbeat"]
    assert drain and all(row["process_running"] is False for row in drain)
    assert all(row["wait_category"]=="pipe_drain_unknown" for row in drain)
    phases=[row["phase"] for row in records]
    assert phases.index("exec_process_exited") < phases.index("pipe_drain_heartbeat") < phases.index("exec_returned")


def test_observed_backend_retries_and_backoff_are_safe_metadata(tmp_path,monkeypatch,capsys):
    records=[]
    backend=CodexBackend(roles={"critic":SimpleNamespace(system="",schema={"ok":bool})},
                         cwd=str(tmp_path),timeout=None,phase_observer=records.append)
    calls=[]

    def transient_then_success(cmd,cwd,devnull,call_id,attempt):
        calls.append(attempt)
        if attempt==0:
            return SimpleNamespace(returncode=1,stderr="",stdout=(
                '{"type":"thread.started","thread_id":"synthetic"}\n'
                '{"type":"turn.failed","error":{"message":"at capacity synthetic-secret-do-not-log"}}\n'))
        with open(cmd[cmd.index("-o")+1],"w") as f: json.dump({"ok":True},f)
        return SimpleNamespace(returncode=0,stdout="",stderr="")

    monkeypatch.setattr(backend,"_run_observed_exec",transient_then_success)
    monkeypatch.setattr("verifier.llm.codex_backend.time.sleep",lambda _:None)
    assert backend.complete("critic","synthetic",{}) == {"ok":True}
    assert calls==[0,1]
    phases=[row["phase"] for row in records]
    assert phases==["backend_transient_failure","backend_backoff_started","backend_backoff_finished"]
    assert records[1]["wait_seconds"]==15 and records[1]["retry_number"]==1
    assert "synthetic-secret-do-not-log" not in json.dumps(records)+capsys.readouterr().err


def test_observed_thread_event_cannot_masquerade_as_final_answer(tmp_path, monkeypatch):
    program = 'print(\'{"type":"thread.started","thread_id":"synthetic"}\')'
    backend, _ = _observed_synthetic_backend(tmp_path, monkeypatch, program, lambda _: None)
    with pytest.raises(RoleError, match="produced no output"):
        backend.complete("critic", "synthetic prompt", {})


def test_observed_backend_keeps_atomic_role_wire_lowering(tmp_path, monkeypatch):
    from verifier.llm.agents import AgentRole

    program = 'import json,sys;json.dump({"wrapped":{"ok":True}},open(sys.argv[1],"w"))'
    records = []
    backend, _ = _observed_synthetic_backend(tmp_path, monkeypatch, program, records.append)
    role = AgentRole("critic", "synthetic fixture", {"ok": bool},
                     wire_schema={"wrapped": {"ok": bool}},
                     wire_lower=lambda payload: payload["wrapped"])
    backend.roles = {"critic": role}
    assert backend.uses_role_wire_schema is True
    assert role.call(backend, {}) == {"ok": True}
    assert records and all(record["role"] == "critic" for record in records)


def test_observed_backend_preserves_stricter_attempt_stream_policy(tmp_path, monkeypatch):
    program = 'import json,sys;json.dump({"ok":True},open(sys.argv[1],"w"));print("synthetic policy marker")'
    backend, _ = _observed_synthetic_backend(tmp_path, monkeypatch, program, lambda _: None)
    inspected = []

    def reject(stdout):
        inspected.append(stdout)
        raise RoleError("synthetic stricter role policy rejection")

    monkeypatch.setattr(backend, "_validate_attempt_stream", reject)
    with pytest.raises(RoleError, match="stricter role policy rejection"):
        backend.complete("critic", "synthetic fixture", {})
    assert inspected == ["synthetic policy marker\n"]
    assert backend._phase_roles == {}


@pytest.mark.skipif(os.name != "posix", reason="POSIX fd-bound launch contract")
def test_observed_launch_uses_bound_cwd_fd_not_replaced_path(tmp_path):
    from verifier.llm.codex_backend import _BoundExecutionDirectory

    original = tmp_path / "bound"
    original.mkdir()
    descriptor = os.open(original, os.O_RDONLY | os.O_DIRECTORY)
    moved = tmp_path / "moved"
    records = []
    backend = CodexBackend(cwd=str(tmp_path), timeout=None, phase_observer=records.append)
    try:
        cwd = _BoundExecutionDirectory(str(original), descriptor)
        original.rename(moved)
        original.mkdir()  # A distinct fixture directory at the old pathname.
        with open(os.devnull) as devnull:
            result = backend._run_observed_exec(
                [sys.executable, "-I", "-c", "import os;print(os.getcwd())"],
                cwd, devnull, "synthetic-bound-cwd", 0)
        assert result.returncode == 0 and result.stdout.strip() == str(moved)
        assert str(moved) not in json.dumps(records)
        assert os.fstat(descriptor).st_ino == moved.stat().st_ino
    finally:
        os.close(descriptor)


def test_completed_turn_is_not_retried_for_an_earlier_recovered_error(tmp_path, monkeypatch):
    commands = []
    backend = CodexBackend(roles={"critic":SimpleNamespace(system="",schema={"ok":bool})},cwd=str(tmp_path))

    def recover(cmd, **kwargs):
        commands.append(cmd)
        with open(cmd[cmd.index("-o")+1],"w") as f: json.dump({"ok":True},f)
        return SimpleNamespace(returncode=0,stderr="",stdout=(
            '{"type":"thread.started","thread_id":"synthetic"}\n'
            '{"type":"error","message":"at capacity"}\n'
            '{"type":"turn.completed","usage":{}}\n'))

    monkeypatch.setattr("verifier.llm.codex_backend._run_codex_process",recover)
    assert backend.complete("critic","synthetic",{}) == {"ok":True}
    assert len(commands) == 1


def test_exhausted_codex_infrastructure_does_not_repeat_generation(tmp_path, monkeypatch):
    from verifier.llm.agents import AgentRole
    from verifier.llm.codex_backend import TRANSIENT_RETRIES, TransientBackendError
    calls = []
    role = AgentRole("critic", "{statement}", {"ok":bool})
    backend = CodexBackend(roles={"critic":role},cwd=str(tmp_path))

    def capacity(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=1,stderr="",stdout=(
            '{"type":"thread.started","thread_id":"synthetic"}\n'
            '{"type":"turn.failed","error":{"message":"at capacity"}}\n'))

    monkeypatch.setattr("verifier.llm.codex_backend._run_codex_process",capacity)
    monkeypatch.setattr("verifier.llm.codex_backend.time.sleep",lambda _:None)
    with pytest.raises(TransientBackendError):
        role.call(backend,{"statement":"synthetic original task"})
    assert len(calls) == TRANSIENT_RETRIES+1
    assert all("previous attempt failed" not in cmd[-1] for cmd in calls)


def test_true_schema_generation_errors_still_retry_once():
    from verifier.llm.agents import AgentRole
    role = AgentRole("critic", "synthetic task", {"ok":bool})

    class Backend:
        def __init__(self): self.prompts=[]
        def complete(self, name, prompt, variables):
            self.prompts.append(prompt)
            return {"ok":"wrong type"} if len(self.prompts)==1 else {"ok":True}

    backend=Backend()
    assert role.call(backend,{}) == {"ok":True}
    assert len(backend.prompts)==2 and "previous attempt failed" in backend.prompts[1]


def test_sdk_exhaustion_preserves_the_same_non_generation_contract(monkeypatch):
    from verifier.llm.agents import AgentBackend, AgentRole, ExhaustedInfrastructureError, _TransientSDKError
    role=AgentRole("critic","synthetic task",{"ok":bool})
    backend=AgentBackend(roles={"critic":role})
    backend.transient_delays=(0,0,0)
    prompts=[]

    async def fail(seat,prompt):
        prompts.append(prompt)
        raise _TransientSDKError("synthetic transient transport")

    monkeypatch.setattr(backend,"_attempt",fail)
    with pytest.raises(ExhaustedInfrastructureError): role.call(backend,{})
    assert prompts == ["synthetic task"]*3


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
        "marker=pathlib.Path(sys.argv[1])\n"
        "pending=marker.with_name(marker.name+'.pending')\n"
        "pending.write_text("
        "f'{os.getpid()} {child.pid}')\n"
        "os.replace(pending,marker)\n"
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

    def interrupt_extract(_text):
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
