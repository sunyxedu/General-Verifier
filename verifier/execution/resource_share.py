"""The verifier's processes run inside the task's own resource envelope — on
Terminal-Bench a container with, typically, one CPU and 2 GB — beside the
solver and the task's processes, and they come last. Measured 2026-09-24:
eight concurrent seat processes (~170 MB each) held a 2 GB container at 92%
of its limit, and the kernel's out-of-memory killer, which takes the largest
process, took the verifier's server (the solver's gv calls then failed with
"Transport closed") and then the solver's own session. None of this depends
on the task:

- every seat and check process the verifier launches is the kernel's first
  choice when the envelope runs out of memory, and the verifier's server the
  next: under pressure the verifier loses a check, never the solver's session
  or a service the task runs;
- those processes run at a lower CPU priority, so a check beside the
  solver's build does not slow it;
- a new seat process starts only while the envelope has room for one, or
  when none of the verifier's is running, so a check always progresses.
"""
from __future__ import annotations

import collections
import contextlib
import os
import threading
import time
from typing import Optional

LAUNCHED_OOM_SCORE_ADJ = 1000
SERVER_OOM_SCORE_ADJ = 500
LAUNCHED_NICENESS = 10
# one seat process with what it runs, plus slack for the solver's next step
SEAT_ROOM_MB_DEFAULT = 512
# a started process's memory shows in the envelope's counters only after
# its start-up; until then its share is held back from the room
STARTUP_SECONDS = 15.0
ROOM_POLL_SECONDS = 2.0

_cond = threading.Condition()
_running = 0
_admitted = collections.deque()   # admission times within STARTUP_SECONDS
_waited = threading.local()


def seat_room_bytes() -> int:
    """Room a new seat process needs (GV_SEAT_ROOM_MB; <= 0 disables the
    wait)."""
    try:
        return int(os.environ.get("GV_SEAT_ROOM_MB",
                                  SEAT_ROOM_MB_DEFAULT)) << 20
    except ValueError:
        return SEAT_ROOM_MB_DEFAULT << 20


def _read(path: str) -> Optional[str]:
    try:
        with open(path, encoding="ascii") as f:
            return f.read().strip()
    except (OSError, ValueError):
        return None


def _stat(path: str, key: str) -> int:
    for line in (_read(path) or "").splitlines():
        name, _, value = line.partition(" ")
        if name == key and value.isdigit():
            return int(value)
    return 0


def envelope_room(root: str = "/sys/fs/cgroup") -> Optional[int]:
    """Bytes the memory envelope can still give: its limit less what is in
    use, inactive file cache counted as free. None without a bounded
    envelope (a host, an unlimited container)."""
    limit = _read(os.path.join(root, "memory.max"))
    if limit is not None:              # cgroup v2
        if not limit.isdigit():
            return None
        used = int(_read(os.path.join(root, "memory.current")) or 0)
        cache = _stat(os.path.join(root, "memory.stat"), "inactive_file")
    else:                              # cgroup v1
        limit = _read(os.path.join(root, "memory", "memory.limit_in_bytes"))
        if limit is None or not limit.isdigit() or int(limit) >= 1 << 60:
            return None
        used = int(_read(os.path.join(root, "memory",
                                      "memory.usage_in_bytes")) or 0)
        cache = _stat(os.path.join(root, "memory", "memory.stat"),
                      "total_inactive_file")
    return int(limit) - max(0, used - cache)


def _write_oom_score_adj(pid: int, value: int) -> None:
    try:
        with open(f"/proc/{pid}/oom_score_adj", "w", encoding="ascii") as f:
            f.write(str(value))
    except OSError:
        pass


def yield_process(pid: int) -> None:
    """Mark a launched process — and so what it forks later, which inherits
    both settings — as the envelope's first to give way."""
    try:
        os.setpriority(os.PRIO_PROCESS, pid, LAUNCHED_NICENESS)
    except (AttributeError, OSError):
        pass
    _write_oom_score_adj(pid, LAUNCHED_OOM_SCORE_ADJ)


def yield_server() -> None:
    """The verifier's server gives way before the solver and the task, after
    its own seats."""
    _write_oom_score_adj(os.getpid(), SERVER_OOM_SCORE_ADJ)


def take_room_wait() -> float:
    """Seconds this thread's launches waited for room since the last take."""
    seconds = getattr(_waited, "seconds", 0.0)
    _waited.seconds = 0.0
    return seconds


@contextlib.contextmanager
def launch_slot(room=envelope_room):
    """Held while one launched seat or check process runs; entered once the
    envelope has room for it, or at once when none of ours is running."""
    global _running
    need = seat_room_bytes()
    started = time.monotonic()
    with _cond:
        while _running > 0 and need > 0:
            now = time.monotonic()
            while _admitted and now - _admitted[0] > STARTUP_SECONDS:
                _admitted.popleft()
            free = room()
            if free is None or free - need * len(_admitted) >= need:
                break
            _cond.wait(ROOM_POLL_SECONDS)
        _running += 1
        _admitted.append(time.monotonic())
    _waited.seconds = round(getattr(_waited, "seconds", 0.0)
                            + time.monotonic() - started, 3)
    try:
        yield
    finally:
        with _cond:
            _running -= 1
            _cond.notify_all()
