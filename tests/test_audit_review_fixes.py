"""Regressions for what a review of the audit fixes found: the stop hook
does not wait on a dead server, stays inside its time budget and replays
only what it has not seen, skips failed commits; the server exits at once
when its client is gone; the stable ledger read excludes cross-process
appends; a hook line whose ledger append succeeded is consumed; a support
claim reuses a node a final settled; a draft prewarm queues behind the
review; the seat-failure scan holds no node lock; direct and process seats
share one cap without queueing behind memory; the newest status change
wins however listeners are ordered."""
from __future__ import annotations

import collections
import fcntl
import importlib.util
import io
import json
import os
import subprocess
import threading
import time

import pytest

from verifier import mcp_server
from verifier.core.events import EventLog
from verifier.core.model import Claim, CLAIM_ASSERTION, CLAIM_FINAL
from verifier.core.store import TRIGGER_LAZY
from verifier.execution import resource_share as rs
from verifier.llm import agents
from verifier.mcp_server import HOOK_FEED, SERVER_LOCK, Service

from test_kernel_lifecycle_audit import Gate, build

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "gv_stop_hook_review", os.path.join(HERE, "hooks", "stop.py"))
stop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stop)
MB = 1 << 20


def _final(label, text):
    return {"kind": "final", "label": label, "text": text, "deps": []}


def _pending_state(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    log = EventLog(str(state / "events.jsonl"))
    log.append("server_started", {"pid": 1})
    commit = log.append("commit", _final("F", "done"))
    fid = stop.latest_final([commit])["id"]
    log.append("verify_request", {"target": fid, "cause": "certificate"})
    return state, log


def _run_hook(state, monkeypatch):
    monkeypatch.setenv("GV_STATE_DIR", str(state))
    monkeypatch.setenv("GV_GOAL", "partial")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    stop.main()
    return json.loads(out.getvalue()) if out.getvalue() else None


# 1. a request no live server can answer is a verifier failure

def test_a_request_with_no_live_server_is_stopped_not_running(tmp_path):
    state, log = _pending_state(tmp_path)
    events = list(log.replay())
    # no lock file: liveness cannot be told, the check counts as running
    assert stop.final_state(events, "partial", str(state))[0] == "running"
    (state / SERVER_LOCK).write_text("")
    assert stop.final_state(events, "partial", str(state))[0] == "stopped"
    with open(state / SERVER_LOCK, "a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert stop.final_state(events, "partial", str(state))[0] == "running"


def test_the_hook_does_not_wait_on_a_dead_server(tmp_path, monkeypatch):
    state, _ = _pending_state(tmp_path)
    (state / SERVER_LOCK).write_text("")
    monkeypatch.setattr(stop, "POLL_SECONDS", 30.0)
    started = time.monotonic()
    out = _run_hook(state, monkeypatch)
    assert time.monotonic() - started < 5
    assert out["decision"] == "block"
    assert "exited while the check was running" in out["reason"]
    record = json.loads((state / stop.BLOCKS_FILE).read_text())
    assert record["waits"] == [] and record["blocks"][0]["state"] == "stopped"


# 2. the whole hook ends inside its budget; replays are incremental

def test_polling_ends_within_the_hook_budget(tmp_path, monkeypatch):
    state, _ = _pending_state(tmp_path)
    monkeypatch.setattr(stop, "POLL_SECONDS", 30.0)
    monkeypatch.setattr(stop, "POLL_INTERVAL", 0.05)
    monkeypatch.setattr(stop, "HOOK_BUDGET_SECONDS", 0.5)
    started = time.monotonic()
    out = _run_hook(state, monkeypatch)
    assert time.monotonic() - started < 2
    assert "GOAL PENDING" in out["reason"]


def test_the_builder_replays_only_new_events(tmp_path):
    state, log = _pending_state(tmp_path)
    judge = stop._Judge(str(state), "partial")
    assert judge.read()[0] == "running"
    replay = judge.replay
    seen = replay.seen
    log.append("discharge", {"obligation": "o", "outcome": "ran"})
    assert judge.read()[0] == "running"
    assert judge.replay is replay and replay.seen == seen + 1
    # a commit_failed naming a commit the replay accepted replays afresh
    commit = next(e for e in log.replay() if e["type"] == "commit")
    log.append("commit_failed", {"commit": commit["seq"], "error": "x"})
    assert judge.read()[0] == "none"
    assert judge.replay is not replay


def test_the_ledger_reader_parses_only_appended_complete_lines(tmp_path):
    state, log = _pending_state(tmp_path)
    ledger = stop._Ledger(str(state))
    assert ledger.refresh() and len(ledger.events) == 3
    assert not ledger.refresh()
    with open(state / "events.jsonl", "ab") as f:
        f.write(b'{"seq": 4, "type": "x"')           # torn
    assert not ledger.refresh()
    with open(state / "events.jsonl", "ab") as f:
        f.write(b', "payload": {}}\n')
    assert ledger.refresh() and ledger.events[-1]["seq"] == 4


# 4. a failed commit is never the latest final

def test_latest_final_skips_commits_named_failed():
    events = [
        {"seq": 1, "type": "commit", "payload": _final("F1", "first")},
        {"seq": 2, "type": "commit", "payload": _final("F2", "second")},
        {"seq": 3, "type": "commit_failed", "payload": {"commit": 2}},
    ]
    assert stop.latest_final(events)["label"] == "F1"


# 3. the server exits at once when its client is gone

class _Stub:
    def __init__(self, state):
        self.log = EventLog(str(state / "events.jsonl"))
        self.server_lock = mcp_server.hold_server_lock(str(state))
        self._drain_lock = threading.Lock()
        self.saved = False

    def _save_feed_offset(self):
        self.saved = True


def test_terminate_kills_launches_releases_the_lock_and_exits(tmp_path):
    state = tmp_path / "state"
    svc = _Stub(state)
    svc.log.append("server_started", {"pid": 1})
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    rs.track_launched(proc.pid)
    exits = []
    try:
        mcp_server.terminate(svc, exit=exits.append)
        assert proc.wait(timeout=5) == -9
    finally:
        rs.untrack_launched(proc.pid)
        if proc.poll() is None:
            proc.kill()
        svc.log._lock.release()
    assert exits == [0] and svc.saved
    # the state is free for the restarted server at once
    handle = mcp_server.hold_server_lock(str(state))
    handle.close()


def test_the_server_terminates_when_the_client_is_gone(tmp_path, monkeypatch):
    ended = []

    class FakeServer:
        def run(self):
            pass

    monkeypatch.setattr(mcp_server, "Service",
                        lambda *a, **k: type("S", (), {
                            "record_start": lambda self, l: None,
                            "start_background": lambda self: False})())
    monkeypatch.setattr(mcp_server, "build_server", lambda s: FakeServer())
    monkeypatch.setattr(mcp_server, "terminate", ended.append)
    monkeypatch.setattr("sys.argv", ["gv", "--state-dir", str(tmp_path)])
    monkeypatch.setenv("GIT_CONFIG_COUNT", "0")
    mcp_server.main()
    assert len(ended) == 1


# 5. the stable read waits for another process's append

def test_the_stable_read_takes_the_writers_lock(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("session_start", {})
    done = threading.Event()
    with open(tmp_path / "events.jsonl", "a+b") as writer:
        fcntl.flock(writer.fileno(), fcntl.LOCK_EX)
        thread = threading.Thread(
            target=lambda: (log.source_projection(), done.set()), daemon=True)
        thread.start()
        assert not done.wait(0.3)          # blocked behind the writer
        fcntl.flock(writer.fileno(), fcntl.LOCK_UN)
    assert done.wait(5)


# 6. a hook line in the ledger is consumed; a bad id is rejected

def _service(state):
    class Idle:
        def __init__(self, sc):
            self.sc = sc

        def verify(self, claim_id, force=False):
            return self.sc.store.status(claim_id)
    return Service(str(state), trigger=TRIGGER_LAZY, kernel_factory=Idle)


def _feed(state, *events):
    with open(os.path.join(state, HOOK_FEED), "a", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _count(svc, etype):
    return sum(1 for e in svc.log.replay() if e["type"] == etype)


def test_a_line_whose_append_succeeded_is_not_appended_again(tmp_path):
    svc = _service(tmp_path)

    def broken(event):
        raise RuntimeError("wiring failed")
    svc.builder.feed = broken
    _feed(tmp_path, {"type": "tool_call", "payload": {
        "tool": "Bash", "tool_use_id": "u1", "input": {}}})
    svc.drain_hooks()
    svc.drain_hooks()
    assert _count(svc, "tool_call") == 1


def test_a_non_string_tool_use_id_is_rejected_not_fatal(tmp_path):
    svc = _service(tmp_path)
    _feed(tmp_path,
          {"type": "tool_call", "payload": {"tool": "Bash",
                                            "tool_use_id": ["x"]}},
          {"type": "tool_result", "payload": {"tool": "Bash",
                                              "tool_use_id": {"a": 1}}},
          {"type": "tool_call", "payload": {"tool": "Bash",
                                            "tool_use_id": "ok"}})
    svc.drain_hooks()
    assert _count(svc, "hook_feed_rejected") == 2
    assert _count(svc, "tool_call") == 1
    # an older ledger holding such an event still starts
    svc.log.append("tool_call", {"tool": "Bash", "tool_use_id": ["y"]})
    _service(tmp_path)


def test_the_newest_status_change_wins_out_of_order(tmp_path):
    svc = _service(tmp_path)
    svc._observe({"type": "status_change", "seq": 9,
                  "payload": {"claim": "c", "status": "refuted"}})
    svc._observe({"type": "status_change", "seq": 7,
                  "payload": {"claim": "c", "status": "unverified"}})
    assert svc._last_change["c"] == (9, "refuted")


# 7. a support claim reuses a node a final settled

def test_a_support_claim_reuses_a_node_a_final_settled(tmp_path):
    gate = Gate(block=False)
    store, log, _, kernel = build(tmp_path, gate)
    final = store.add_claim(Claim(CLAIM_FINAL, "the answer is 15"))
    assert kernel.verify(final) == "verified"
    support = store.add_claim(Claim(CLAIM_ASSERTION, "the file is right"))
    kernel.verify(support)
    assert len(gate.calls) == 1
    assert not [e for e in log.replay()
                if e["type"] == "dedup_hit" and e["payload"].get("refused")]


# 9. the seat-failure lookup reads no ledger under the node lock

def test_seat_failures_are_noted_as_appended(tmp_path):
    store, log, _, kernel = build(tmp_path, Gate(block=False))
    since = log.last_seq()
    log.append("judger_call_failed", {"obligation": "o1", "will_retry": True})
    log.append("methods_enumerated", {"obligation": "o2", "error": "x",
                                      "prewarm": True})
    log.append("judger_call_failed", {"obligation": "o3"})
    log.replay = None                       # never scanned
    assert not kernel._seat_failure("o1", since)
    assert not kernel._seat_failure("o2", since)
    assert kernel._seat_failure("o3", since)
    assert not kernel._seat_failure("o3", log.last_seq())


# 8. a draft prewarm's launches queue behind the review's

@pytest.fixture
def fresh_slots(monkeypatch):
    monkeypatch.setattr(rs, "_admitted", collections.deque())
    monkeypatch.setattr(rs, "_queue", collections.deque())
    monkeypatch.setattr(rs, "_running", 0)
    monkeypatch.setattr(rs, "ROOM_POLL_SECONDS", 0.02)
    monkeypatch.setattr(rs, "STARTUP_SECONDS", 0.0)
    monkeypatch.setenv("GV_SEAT_ROOM_MB", "512")


def _admission_order(until):
    opened, release = threading.Event(), threading.Event()
    order = []

    def room():
        return 8192 * MB if opened.is_set() else 0

    def holder():
        with rs.launch_slot(room):
            release.wait(5)

    def waiter(name, warm):
        if warm:
            with rs.warmup_launches(until=until):
                with rs.launch_slot(room):
                    order.append(name)
        else:
            with rs.launch_slot(room):
                order.append(name)

    first = threading.Thread(target=holder)
    first.start()
    time.sleep(0.05)
    threads = []
    for name, warm in (("warm", True), ("review", False)):
        t = threading.Thread(target=waiter, args=(name, warm))
        t.start()
        threads.append(t)
        time.sleep(0.05)
    release.set()
    first.join(5)
    opened.set()
    for t in threads:
        t.join(5)
    return order


def test_a_warmup_launch_waits_behind_a_later_ordinary_one(fresh_slots):
    assert _admission_order(until=lambda: False) == ["review", "warm"]


def test_a_warmup_queues_in_arrival_order_once_the_review_is_done(
        fresh_slots):
    assert _admission_order(until=lambda: True) == ["warm", "review"]


# 10. one seat cap; a seat waiting for memory holds no permit

def test_a_seat_waiting_for_room_gives_its_permit_back(fresh_slots,
                                                       monkeypatch):
    monkeypatch.setenv("GV_MAX_CONCURRENT_SEATS", "1")
    gate = agents._SeatGate()
    opened, holding = threading.Event(), threading.Event()
    release = threading.Event()
    room = lambda: 8192 * MB if opened.is_set() else 0  # noqa: E731

    def running_process_seat():
        with rs.launch_slot(room):
            holding.set()
            release.wait(5)

    def waiting_process_seat():
        with gate.slot():
            with rs.launch_slot(room):
                pass

    first = threading.Thread(target=running_process_seat)
    first.start()
    assert holding.wait(2)
    waiting = threading.Thread(target=waiting_process_seat)
    waiting.start()
    time.sleep(0.1)
    entered = threading.Event()

    def direct_seat():
        with gate.slot():
            entered.set()
    direct = threading.Thread(target=direct_seat)
    direct.start()
    assert entered.wait(2)       # not behind the seat waiting for memory
    release.set()
    opened.set()
    for t in (first, waiting, direct):
        t.join(5)
    assert not waiting.is_alive()
    # the cap is one pool for every seat
    assert not hasattr(agents, "DIRECT_SEAT_GATE")
