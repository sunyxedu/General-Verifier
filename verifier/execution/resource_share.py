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
- (they keep their CPU share: at a lower priority, 38 of 50 checks of a
  video task timed out on one CPU while the solver ran its own OCR, and a
  check that cannot finish is pure waste;)
- a new seat process starts only while the envelope has room for one, or
  when none of the verifier's is running, so a check always progresses;
  launches waiting for room are admitted in arrival order.
"""
from __future__ import annotations

import collections
import contextlib
import itertools
import os
import signal
import threading
import time
from typing import Optional

LAUNCHED_OOM_SCORE_ADJ = 1000
SERVER_OOM_SCORE_ADJ = 500
LAUNCHED_NICENESS = 0      # CPU shared fairly (see above); >0 deprioritizes
# one seat process with what it runs, plus slack for the solver's next step
SEAT_ROOM_MB_DEFAULT = 512
# a started process's memory shows in the envelope's counters only after
# its start-up; until then its share is held back from the room
STARTUP_SECONDS = 15.0
ROOM_POLL_SECONDS = 2.0

_cond = threading.Condition()
_running = 0
# (admission time, ticket) of launches within STARTUP_SECONDS
_admitted = collections.deque()
_queue = collections.deque()      # tickets of waiting launches, in arrival order
_tickets = itertools.count()
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


def _stats(path: str) -> dict:
    """A memory.stat file's counters by name (absent keys stay absent)."""
    stats = {}
    for line in (_read(path) or "").splitlines():
        name, _, value = line.partition(" ")
        if value.isdigit():
            stats[name] = int(value)
    return stats


def envelope_room(root: str = "/sys/fs/cgroup") -> Optional[int]:
    """Bytes the memory envelope can still give: its limit less what is in
    use, reclaimable file cache counted as free. None without a bounded
    envelope (a host, an unlimited container).

    Reclaimable is the file cache on both LRU lists less what is mapped into
    a process: a container's page cache is mostly the images and libraries
    the task and the seats have read, the kernel drops it under pressure
    whether the kernel last saw it active or not, and only mapped pages
    (running code) are held. Counting the inactive list alone left ~350 MB
    of cache counted as used in a 2 GB container, so a second seat rarely
    fitted though the kernel would have made room for it."""
    limit = _read(os.path.join(root, "memory.max"))
    if limit is not None:              # cgroup v2
        if not limit.isdigit():
            return None
        used = int(_read(os.path.join(root, "memory.current")) or 0)
        stat_path = os.path.join(root, "memory.stat")
        names = ("active_file", "inactive_file", "file_mapped")
    else:                              # cgroup v1
        limit = _read(os.path.join(root, "memory", "memory.limit_in_bytes"))
        if limit is None or not limit.isdigit() or int(limit) >= 1 << 60:
            return None
        used = int(_read(os.path.join(root, "memory",
                                      "memory.usage_in_bytes")) or 0)
        stat_path = os.path.join(root, "memory", "memory.stat")
        names = ("total_active_file", "total_inactive_file",
                 "total_mapped_file")
    stats = _stats(stat_path)
    active, inactive, mapped = (stats.get(name, 0) for name in names)
    cache = max(0, active + inactive - mapped)
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
    if LAUNCHED_NICENESS:
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
    return take_room_wait_detail()["seconds"]


def take_room_wait_detail() -> dict:
    """This thread's launch waits since the last take: seconds, and why the
    longest-waiting launch waited (None when none had to)."""
    detail = {"seconds": getattr(_waited, "seconds", 0.0),
              "reason": getattr(_waited, "reason", None)}
    _waited.seconds, _waited.reason, _waited.longest = 0.0, None, 0.0
    return detail


def _mb(value: int) -> int:
    return int(value) >> 20


@contextlib.contextmanager
def launch_slot(room=envelope_room):
    """Held while one launched seat or check process runs; entered once the
    envelope has room for it, or at once when none of ours is running.

    Waiters are admitted in arrival order: seats and checks are threads of
    the verifier's one server process, and with every waiter polling the
    room, whichever poll came first after room appeared won — a method
    synthesis call waited 38 minutes while later arrivals went ahead. Only
    the head of the queue asks for room; the others wait behind it."""
    global _running
    need = seat_room_bytes()
    started = time.monotonic()
    reason = None
    with _cond:
        ticket = next(_tickets)
        if need > 0:
            _queue.append(ticket)
            try:
                while True:
                    if _queue[0] != ticket:
                        reason = reason or "queued behind earlier launches"
                    elif _running == 0:
                        break
                    else:
                        now = time.monotonic()
                        while _admitted and \
                                now - _admitted[0][0] > STARTUP_SECONDS:
                            _admitted.popleft()
                        free = room()
                        held = need * len(_admitted)
                        if free is None or free - held >= need:
                            break
                        reason = (
                            f"no room: {_mb(free)} MB free"
                            + (f" less {_mb(held)} MB held for launches "
                               "still starting" if held else "")
                            + f", {_mb(need)} MB needed, "
                            f"{_running} running")
                    _cond.wait(ROOM_POLL_SECONDS)
            finally:
                _queue.remove(ticket)
                _cond.notify_all()
        _running += 1
        _admitted.append((time.monotonic(), ticket))
    waited = time.monotonic() - started
    _waited.seconds = round(getattr(_waited, "seconds", 0.0) + waited, 3)
    if reason and waited >= getattr(_waited, "longest", 0.0):
        _waited.reason, _waited.longest = reason, waited
    try:
        yield
    finally:
        with _cond:
            _running -= 1
            # a process that already exited holds no memory: its start-up
            # share goes back to the room at once, not after STARTUP_SECONDS
            for entry in [e for e in _admitted if e[1] == ticket]:
                _admitted.remove(entry)
            _cond.notify_all()


def _proc_table() -> dict:
    """pid -> (ppid, process group, session) for every visible process, from
    /proc (Linux); empty elsewhere."""
    table = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return table
    for name in names:
        if not name.isdigit():
            continue
        raw = _read(f"/proc/{name}/stat")
        if not raw or ")" not in raw:
            continue
        # the command name may hold spaces and parentheses: fields resume
        # after its last ')' as "state ppid pgrp session ..."
        fields = raw.rsplit(")", 1)[1].split()
        try:
            table[int(name)] = (int(fields[1]), int(fields[2]),
                                int(fields[3]))
        except (IndexError, ValueError):
            continue
    return table


def escaped_members(leader: int) -> list:
    """Processes a launched process's group kill would miss: members of its
    session that moved to their own process group (setpgid), and live
    descendants that left the session too (setsid) while still parented
    under it. The leader is a session leader (start_new_session), so its
    pid names the session."""
    table = _proc_table()
    members = {pid for pid, (_ppid, pgrp, sid) in table.items()
               if sid == leader and pgrp != leader}
    children = collections.defaultdict(list)
    for pid, (ppid, _pgrp, _sid) in table.items():
        children[ppid].append(pid)
    stack = [leader, *members]
    seen = set(stack)
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    seen.discard(leader)
    return sorted(pid for pid in seen
                  if pid != os.getpid() and table[pid][1] != leader)


def kill_escaped(leader: int) -> None:
    """SIGKILL what escaped_members finds; killpg covers the rest."""
    for pid in escaped_members(leader):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
