"""A check's scratch directory has a size budget while the check runs.

A check copies what it exercises into its working directory and runs it
there. Nothing bounded how much a running check could write: one executor
copied a whole root filesystem into its scratch directory, read the random
device as if it were a file, and wrote 60 GB until the host disk was full,
which took every other trial on the machine down with it. The workdir is
cleaned after each check, but only after; this watches it DURING the run
and stops the check when it outgrows the budget (GV_SCRATCH_CAP_GB, default
6; 0 disables). The check then fails like any other execution failure, and
its scratch is cleaned as usual.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Optional

POLL_SECONDS = 10.0


def scratch_cap_bytes() -> int:
    try:
        gb = float(os.environ.get("GV_SCRATCH_CAP_GB", "6"))
    except ValueError:
        gb = 6.0
    return int(gb * (1 << 30)) if gb > 0 else 0


def exceeds(path: str, cap: int) -> bool:
    """Whether the regular files under path total more than cap bytes
    (symlinks are not followed; the walk stops as soon as the cap is passed)."""
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            if total > cap:
                                return True
                    except OSError:
                        continue
        except OSError:
            continue
    return False


class ScratchWatch:
    """Polls a directory while a process runs; calls stop() once when the
    directory outgrows cap. Use as a context manager around the wait."""

    def __init__(self, path: Optional[str], cap: int, stop: Callable[[], None],
                 poll: Optional[float] = None):
        self.path, self.cap, self._stop_process = path, cap, stop
        self.poll = POLL_SECONDS if poll is None else poll
        self.exceeded = False
        self._done = threading.Event()
        # held while stopping: once __exit__ has taken it, no stop follows
        # (the process may be reaped and its pid reused by then)
        self._stopping = threading.Lock()
        self._thread = None

    def __enter__(self):
        if self.path and self.cap > 0:
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="gv-scratch-watch")
            self._thread.start()
        return self

    def _run(self):
        while not self._done.wait(self.poll):
            try:
                if exceeds(self.path, self.cap):
                    with self._stopping:
                        if self._done.is_set():
                            return
                        self.exceeded = True
                        self._stop_process()
                    return
            except Exception:  # noqa: BLE001 - a watcher never breaks the check
                continue

    def __exit__(self, *exc):
        with self._stopping:
            self._done.set()
        if self._thread is not None:
            # a walk in progress sees _done before it could stop anything;
            # waiting for it only keeps a stale watcher from outliving the run
            self._thread.join(self.poll)
        return False
