"""Seat and check processes inside a small task envelope: memory room is
read right and shared fairly, failures are classified by what caused them,
and nothing a seat or check starts outlives it or its output bound."""
from __future__ import annotations

import collections
import contextlib
import json
import os
import re
import ssl
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

import verifier.execution.scratch_budget as sb
import verifier.llm.codex_backend as cb
from verifier.core.model import Obligation
from verifier.execution import resource_share as rs
from verifier.execution.discharge import AgentDischarger, _CappedReader
from verifier.execution.instruments import ArtifactStore
from verifier.llm import agents
from verifier.llm import direct_backend as db
from verifier.llm.agents import ALL_AGENT_ROLES, AgentRole, SeatResourceKilled
from verifier.llm.schema import FixtureBackend, RoleError

MB = 1 << 20


# ---------------------------------------------------------------- room


def test_room_counts_all_unmapped_file_cache_as_free_cgroup_v2(tmp_path):
    (tmp_path / "memory.max").write_text(f"{2048 * MB}\n")
    (tmp_path / "memory.current").write_text(f"{1900 * MB}\n")
    (tmp_path / "memory.stat").write_text(
        f"anon {1400 * MB}\nactive_file {350 * MB}\n"
        f"inactive_file {150 * MB}\nfile_mapped {100 * MB}\n")
    # used 1900 - reclaimable (350 + 150 - 100) = 1500 in use
    assert rs.envelope_room(str(tmp_path)) == 548 * MB


def test_room_counts_all_unmapped_file_cache_as_free_cgroup_v1(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "memory.limit_in_bytes").write_text(str(1024 * MB))
    (mem / "memory.usage_in_bytes").write_text(str(900 * MB))
    (mem / "memory.stat").write_text(
        f"total_active_file {200 * MB}\ntotal_inactive_file {100 * MB}\n"
        f"total_mapped_file {50 * MB}\nactive_file {999 * MB}\n")
    assert rs.envelope_room(str(tmp_path)) == 374 * MB


def test_room_tolerates_missing_cache_counters(tmp_path):
    (tmp_path / "memory.max").write_text(f"{1024 * MB}\n")
    (tmp_path / "memory.current").write_text(f"{900 * MB}\n")
    (tmp_path / "memory.stat").write_text(f"file_mapped {400 * MB}\n")
    # mapped beyond the (absent) cache never makes the in-use figure grow
    assert rs.envelope_room(str(tmp_path)) == 124 * MB


@pytest.fixture
def fresh_slots(monkeypatch):
    monkeypatch.setattr(rs, "_admitted", collections.deque())
    monkeypatch.setattr(rs, "_queue", collections.deque())
    monkeypatch.setattr(rs, "_running", 0)
    monkeypatch.setattr(rs, "ROOM_POLL_SECONDS", 0.02)
    monkeypatch.setenv("GV_SEAT_ROOM_MB", "512")


def test_waiting_launches_are_admitted_in_arrival_order(fresh_slots,
                                                         monkeypatch):
    monkeypatch.setattr(rs, "STARTUP_SECONDS", 0.0)
    opened = threading.Event()
    order, release = [], threading.Event()

    def room():
        return 8192 * MB if opened.is_set() else 0

    def holder():
        with rs.launch_slot(room):
            release.wait(5)

    def waiter(name):
        with rs.launch_slot(room):
            order.append(name)

    first = threading.Thread(target=holder)
    first.start()
    time.sleep(0.05)
    names = [f"w{i}" for i in range(6)]
    threads = []
    for name in names:
        thread = threading.Thread(target=waiter, args=(name,))
        thread.start()
        threads.append(thread)
        time.sleep(0.013)
    time.sleep(0.1)
    opened.set()
    for thread in threads:
        thread.join(5)
    release.set()
    first.join(5)
    assert order == names


def test_an_exited_launch_gives_its_startup_share_back(fresh_slots,
                                                      monkeypatch):
    monkeypatch.setattr(rs, "STARTUP_SECONDS", 30.0)
    room = lambda: 1100 * MB  # noqa: E731 - room for two seats' shares
    with rs.launch_slot(room):
        pass                                   # started and already gone
    entered = threading.Event()
    release = threading.Event()

    def holder():
        with rs.launch_slot(room):
            entered.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    assert entered.wait(2)
    started = time.monotonic()
    with rs.launch_slot(room):
        pass
    release.set()
    thread.join(5)
    # only the running holder's share is held: 1100 - 512 leaves room
    assert time.monotonic() - started < 1.0


def test_a_wait_records_why(fresh_slots, monkeypatch):
    monkeypatch.setattr(rs, "STARTUP_SECONDS", 0.0)
    free = {"bytes": 100 * MB}
    release = threading.Event()
    entered = threading.Event()

    def holder():
        with rs.launch_slot(lambda: free["bytes"]):
            entered.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    assert entered.wait(2)
    rs.take_room_wait_detail()
    threading.Timer(0.15, lambda: free.update(bytes=4096 * MB)).start()
    with rs.launch_slot(lambda: free["bytes"]):
        pass
    detail = rs.take_room_wait_detail()
    release.set()
    thread.join(5)
    assert detail["seconds"] >= 0.1
    assert detail["reason"].startswith("no room: 100 MB free")
    assert "512 MB needed" in detail["reason"]
    assert rs.take_room_wait_detail() == {"seconds": 0.0, "reason": None}


# ------------------------------------------------------ escaped children


def test_escaped_members_are_those_a_group_kill_misses(monkeypatch):
    leader = 100
    table = {
        100: (1, 100, 100),     # the seat
        101: (100, 100, 100),   # same group: killpg takes it
        102: (100, 102, 100),   # setpgid: same session, own group
        103: (100, 103, 103),   # setsid under the seat
        104: (103, 103, 103),   # its child
        105: (1, 105, 100),     # orphaned session member in its own group
        200: (1, 200, 200),     # unrelated
    }
    monkeypatch.setattr(rs, "_proc_table", lambda: table)
    assert rs.escaped_members(leader) == [102, 103, 104, 105]


@pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs /proc")
def test_a_setsid_descendant_is_killed_with_the_seat(tmp_path):
    pidfile = tmp_path / "pid"
    child = ("import os,sys,time\n"
             "if os.fork() == 0:\n"
             "    os.setsid()\n"
             f"    open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
             "    time.sleep(60)\n"
             "time.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, "-c", child],
                            start_new_session=True)
    try:
        deadline = time.time() + 5
        while not pidfile.exists() or not pidfile.read_text():
            assert time.time() < deadline
            time.sleep(0.02)
        escaped = int(pidfile.read_text())
        cb._kill_process_group(proc)
        proc.wait(5)
        deadline = time.time() + 5
        while time.time() < deadline:
            state = rs._read(f"/proc/{escaped}/stat")
            if state is None or state.rsplit(")", 1)[1].split()[0] == "Z":
                break
            time.sleep(0.02)
        else:
            pytest.fail("the setsid descendant survived")
    finally:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, 9)


def test_a_scratch_watch_never_stops_after_it_exits(monkeypatch):
    stops = []
    walking = threading.Event()

    def slow_exceeds(_path, _cap):
        walking.set()
        time.sleep(0.3)
        return True

    monkeypatch.setattr(sb, "exceeds", slow_exceeds)
    watch = sb.ScratchWatch("/nowhere", 1, lambda: stops.append(1), poll=0.01)
    with watch:
        assert walking.wait(2)
    time.sleep(0.5)
    assert stops == [] and watch.exceeded is False


# ------------------------------------------------- codex classification

HEADER = ("Reading additional input from stdin...\nOpenAI Codex v0.156.1\n"
          "--------\nworkdir: /gv-state\nmodel: m\nsession id: s\n--------\n"
          "user\nthe prompt\n")


def test_capacity_error_after_a_transcript_that_mentions_missing_files():
    stderr = (HEADER + "exec\n/bin/bash -lc 'cat x'\n"
              "cat: x: No such file or directory\npermission denied\n"
              "ERROR: Selected model is at capacity. Please try a different "
              "model.\n")
    assert cb._transient_failure_reason("", 1, stderr)


def test_a_seats_own_http_status_output_is_not_a_provider_failure():
    stderr = (HEADER + "exec\ncurl -si http://localhost:8080\n"
              "HTTP/1.1 503 Service Unavailable\n 503 \n"
              "ERROR: stream disconnected unexpectedly\n")
    assert cb._transient_failure_reason("", 1, stderr) == ""


def test_codex_own_auth_error_after_the_header_stays_permanent():
    stderr = (HEADER + "2026-09-24T06:22:09.977398Z ERROR codex_api::endpoint:"
              " failed to connect to websocket: HTTP error: 401 Unauthorized\n"
              "ERROR: service unavailable\n")
    assert cb._transient_failure_reason("", 1, stderr) == ""


def test_a_failure_before_any_session_is_read_whole():
    assert cb._transient_failure_reason(
        "", 1, "Error loading config.toml: unknown configuration field") == ""
    assert cb._transient_failure_reason(
        "", 1, "HTTP status 503 from provider")


def _backend(tmp_path, schema=None):
    role = SimpleNamespace(system="", schema=schema or {"ok": bool})
    return cb.CodexBackend(roles={"critic": role}, cwd=str(tmp_path))


def test_a_capacity_failure_behind_a_transcript_is_retried(tmp_path,
                                                          monkeypatch):
    calls = []

    def run(cmd, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr=(
                HEADER + "no such file or directory\n"
                "ERROR: Selected model is at capacity.\n"))
        with open(cmd[cmd.index("-o") + 1], "w") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cb, "_run_codex_process", run)
    monkeypatch.setattr(cb.time, "sleep", lambda _: None)
    assert _backend(tmp_path).complete("critic", "p", {}) == {"ok": True}
    assert len(calls) == 2


def test_a_nul_byte_never_reaches_the_command_line(tmp_path, monkeypatch):
    seen = []

    def run(cmd, **_kwargs):
        seen.append(cmd[-1])
        with open(cmd[cmd.index("-o") + 1], "w") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cb, "_run_codex_process", run)
    assert _backend(tmp_path).complete(
        "critic", "binary \0 evidence", {}) == {"ok": True}
    assert "\0" not in seen[0] and "binary \\x00 evidence" in seen[0]


def test_a_nul_argument_fails_as_a_role_error_before_any_room_wait(
        monkeypatch):
    def no_room():
        raise AssertionError("waited for room")

    monkeypatch.setattr(rs, "launch_slot", no_room)
    with pytest.raises(RoleError, match="NUL"):
        cb._run_codex_process(["codex", "a\0b"], cwd=None,
                              stdin=subprocess.DEVNULL, timeout=5)


def test_a_signal_death_is_a_resource_failure_not_a_bad_answer(
        tmp_path, monkeypatch):
    def run(cmd, **_kwargs):
        return SimpleNamespace(returncode=-9, stdout="", stderr="transcript")

    monkeypatch.setattr(cb, "_run_codex_process", run)
    with pytest.raises(SeatResourceKilled, match="signal 9"):
        _backend(tmp_path).complete("critic", "p", {})


def test_the_answer_is_the_last_object_and_the_one_that_validates():
    text = ('I will return {"draft": 1} shaped like {"ok": "maybe"} then\n'
            '{"ok": true, "nested": {"x": 1}}')
    assert cb.extract_json(text) == {"ok": True, "nested": {"x": 1}}
    text = '{"ok": true} and a trailing note {"note": 1}'
    assert cb.extract_json(text) == {"note": 1}
    assert cb.extract_json(
        text, accept=lambda obj: cb.validate({"ok": bool}, obj)) == {"ok": True}


# --------------------------------------------------------- role retries


def _role():
    return AgentRole(name="critic", template="judge {x}",
                     schema={"ok": bool})


def test_a_killed_seat_is_retried_without_model_feedback():
    prompts = []

    class Backend:
        def complete(self, role_name, prompt, variables):
            prompts.append(prompt)
            if len(prompts) == 1:
                raise SeatResourceKilled("critic: killed by signal 9")
            return {"ok": True}

    assert _role().call(Backend(), {"x": "it"}) == {"ok": True}
    assert prompts == ["judge it", "judge it"]


# ------------------------------------------------------------- direct


class _Transport:
    name, model = "fake", "fake-model"

    def __init__(self, answers):
        self.answers = list(answers)

    def complete(self, **_kwargs):
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer, {"input_tokens": 1, "output_tokens": 1}


def _screener_vars(role):
    return {name: "" for name in re.findall(r"{(\w+)}", role.template)}


def test_a_raw_tls_error_is_retried_and_on_record(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "TRANSIENT_DELAYS", (0, 0, 0))
    transport = _Transport([
        ssl.SSLError("[SSL: UNEXPECTED_EOF_WHILE_READING] unexpected eof"),
        {"confidence": "low", "grounds": "g", "reason": "r"}])
    direct = db.DirectBackend(transport, cwd=str(tmp_path))
    role = ALL_AGENT_ROLES["screener"]
    assert role.call(direct, _screener_vars(role))["confidence"] == "low"
    row = [json.loads(line) for line in open(tmp_path / "role_calls.jsonl")][-1]
    assert row["attempts"] == 2
    assert "UNEXPECTED_EOF" in row["attempt_errors"][0]
    assert row["attempt_errors"][1] is None


def test_a_failed_attempt_is_on_record_even_when_not_retried(tmp_path):
    transport = _Transport([RoleError("direct call failed: HTTP 400: bad")] * 2)
    direct = db.DirectBackend(transport, cwd=str(tmp_path))
    role = ALL_AGENT_ROLES["screener"]
    with pytest.raises(RoleError):
        role.call(direct, _screener_vars(role))
    rows = [json.loads(line) for line in open(tmp_path / "role_calls.jsonl")]
    assert rows[0]["attempts"] == 1
    assert "HTTP 400" in rows[0]["attempt_errors"][0]


@pytest.mark.parametrize("transport_class", [
    db.AnthropicMessagesTransport, db.OpenAIResponsesTransport,
    db.OpenAIChatTransport])
def test_a_truncated_ok_body_is_a_transport_failure(monkeypatch,
                                                    transport_class):
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b'{"content": [')),
        **kw))
    transport = transport_class("key", "m", base_url="https://x.test")
    with pytest.raises(db.TransientTransportError, match="unparseable"):
        transport.complete(system="s", prompt="p", schema={"type": "object"})


# ------------------------------------------------------------ scripts


def _discharger(tmp_path):
    return AgentDischarger(FixtureBackend({}),
                           ArtifactStore(str(tmp_path / "art")),
                           workdir_base=str(tmp_path / "work"),
                           state_dir=str(tmp_path))


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    state = rs._read(f"/proc/{pid}/stat")
    return not (state and state.rsplit(")", 1)[1].split()[0] == "Z")


def test_what_a_script_leaves_running_is_killed_and_does_not_hold_it(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GV_SCRIPT_TIMEOUT", "30")
    pidfile = tmp_path / "bg.pid"
    method = {"procedure": "p", "basis": "record", "strength": "decisive",
              "script": f"sleep 60 &\necho $! > {pidfile}\necho done\n"}
    started = time.time()
    run = _discharger(tmp_path).run_script(
        {"id": "c1", "text": "claim"}, Obligation("c1", "o"), method, tag="bg")
    assert time.time() - started < 15
    assert run.timed_out is False and run.exit_code == 0
    assert "done" in run.stdout
    pid = int(pidfile.read_text())
    deadline = time.time() + 5
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)


def test_a_scripts_room_wait_is_not_its_run_time(tmp_path, monkeypatch):
    @contextlib.contextmanager
    def slow_slot():
        time.sleep(0.4)
        yield

    monkeypatch.setattr(rs, "launch_slot", slow_slot)
    method = {"procedure": "p", "basis": "record", "strength": "decisive",
              "script": "true\n"}
    run = _discharger(tmp_path).run_script(
        {"id": "c1", "text": "claim"}, Obligation("c1", "o"), method, tag="w")
    assert run.room_wait >= 0.4
    assert run.seconds < run.room_wait


def test_script_output_is_capped_while_it_is_read():
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import sys\nsys.stdout.write('h' * 1000 + 'x' * 5_000_000 "
         "+ 't' * 1000)"],
        stdout=subprocess.PIPE)
    reader = _CappedReader(proc.stdout, cap=4000)
    proc.wait(10)
    reader.join(10)
    text = reader.text()
    assert text.startswith("h" * 1000) and text.endswith("t" * 1000)
    assert "bytes dropped" in text and len(text) < 4100
    proc.stdout.close()
