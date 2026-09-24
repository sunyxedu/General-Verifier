"""The verifier's processes give way to the solver and the task inside the
task's resource envelope: first to be killed when memory runs out, lower CPU
priority, and a new seat process waits for room unless none is running."""
from __future__ import annotations

import collections
import os
import subprocess
import sys
import threading
import time

import pytest

from verifier.execution import resource_share
from verifier.execution.resource_share import (
    envelope_room, launch_slot, take_room_wait, yield_process)

MB = 1 << 20


def cgroup_v2(root, limit, current, inactive_file):
    root.mkdir(exist_ok=True)
    (root / "memory.max").write_text(f"{limit}\n")
    (root / "memory.current").write_text(f"{current}\n")
    (root / "memory.stat").write_text(
        f"anon {current - inactive_file}\ninactive_file {inactive_file}\n"
        "active_file 0\n")
    return str(root)


def test_room_is_the_limit_less_what_is_in_use_counting_idle_cache_free(tmp_path):
    root = cgroup_v2(tmp_path / "cg", 2048 * MB, 1900 * MB, 100 * MB)
    assert envelope_room(root) == 248 * MB


def test_an_unbounded_envelope_has_no_room_to_wait_for(tmp_path):
    (tmp_path / "cg").mkdir()
    (tmp_path / "cg" / "memory.max").write_text("max\n")
    assert envelope_room(str(tmp_path / "cg")) is None
    assert envelope_room(str(tmp_path / "nowhere")) is None


def test_cgroup_v1_is_read_too(tmp_path):
    mem = tmp_path / "cg" / "memory"
    mem.mkdir(parents=True)
    (mem / "memory.limit_in_bytes").write_text(str(1024 * MB))
    (mem / "memory.usage_in_bytes").write_text(str(900 * MB))
    (mem / "memory.stat").write_text(f"total_inactive_file {100 * MB}\n")
    assert envelope_room(str(tmp_path / "cg")) == 224 * MB
    (mem / "memory.limit_in_bytes").write_text(str(1 << 62))
    assert envelope_room(str(tmp_path / "cg")) is None


@pytest.fixture(autouse=True)
def fresh_admissions(monkeypatch):
    monkeypatch.setattr(resource_share, "_admitted", collections.deque())


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(resource_share, "ROOM_POLL_SECONDS", 0.05)
    monkeypatch.setattr(resource_share, "STARTUP_SECONDS", 0.0)
    monkeypatch.setenv("GV_SEAT_ROOM_MB", "512")


def test_a_launch_waits_for_room_while_another_runs(quick):
    room = {"free": 100 * MB}
    first_done = threading.Event()
    order = []

    def first():
        with launch_slot(lambda: room["free"]):
            order.append("first in")
            time.sleep(0.3)
            order.append("first out")
        first_done.set()

    def second():
        time.sleep(0.05)
        with launch_slot(lambda: room["free"]):
            order.append("second in")

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    # no room while the first ran: the second entered only after it left
    assert order == ["first in", "first out", "second in"]


def test_room_lets_launches_run_side_by_side(quick):
    inside = []
    barrier = threading.Barrier(2, timeout=2)

    def seat():
        with launch_slot(lambda: 4096 * MB):
            inside.append(1)
            barrier.wait()

    threads = [threading.Thread(target=seat) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert len(inside) == 2


def test_with_none_running_a_launch_never_waits(quick):
    take_room_wait()
    with launch_slot(lambda: 0):
        pass
    assert take_room_wait() < 0.05


def test_launches_admitted_moments_ago_hold_their_share(monkeypatch):
    monkeypatch.setattr(resource_share, "ROOM_POLL_SECONDS", 0.05)
    monkeypatch.setattr(resource_share, "STARTUP_SECONDS", 0.4)
    monkeypatch.setenv("GV_SEAT_ROOM_MB", "512")
    entered = []
    release = threading.Event()

    def seat(name):
        with launch_slot(lambda: 1100 * MB):
            entered.append((name, time.monotonic()))
            release.wait(3)

    a = threading.Thread(target=seat, args=("a",))
    a.start()
    time.sleep(0.05)
    b = threading.Thread(target=seat, args=("b",))
    b.start()
    time.sleep(0.05)
    c = threading.Thread(target=seat, args=("c",))
    c.start()
    time.sleep(0.2)
    # 1100 MB shows free, but a and b started moments ago and their memory
    # has not shown yet: c waits for their start-up to pass
    assert [n for n, _ in entered] == ["a", "b"]
    deadline = time.monotonic() + 2
    while len(entered) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert [n for n, _ in entered] == ["a", "b", "c"]
    release.set()
    for t in (a, b, c):
        t.join(2)


def test_a_launched_process_yields_cpu_and_memory(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        yield_process(proc.pid)
        assert os.getpriority(os.PRIO_PROCESS, proc.pid) >= resource_share.LAUNCHED_NICENESS
        adj = f"/proc/{proc.pid}/oom_score_adj"
        if os.path.exists(adj):
            assert open(adj).read().strip() == str(resource_share.LAUNCHED_OOM_SCORE_ADJ)
    finally:
        proc.kill()
        proc.wait()
