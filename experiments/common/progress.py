"""Harbor-style live progress for experiment loops, drawn by rich (the
same machinery harbor itself uses), so the look matches exactly:

    <finished-run result lines scroll up here>
    12/60 Mean: 0.729 ━━━━━━━━━╺━━━━━━━━━━━━━ 0:14:30 0:52:10
       ⠸ fm-bmo-optimization#0 0:03:21
       ⠸ fm-elliptic-curves-galois#2 0:01:05

Overall bar on top, in-flight runs below it with spinners and per-run
elapsed time, completed lines printed above the live region. Degrades
gracefully on non-TTY output (plain log lines, one final bar render).

API: Board(total) · begin(name) -> handle · advance(handle, metric) ·
log(line) · close(). `metric` folds into the running mean (e.g. 1.0 for
a solved run).
"""
import sys
import threading

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)


def tty_console():
    """(console, separate) — a Console on the controlling terminal.

    separate=True means it is /dev/tty, DISTINCT from the piped stdio: the
    live bar draws on screen while plain lines keep flowing into the pipe
    (tee'd logs stay free of ANSI). (None, False) when no terminal exists
    at all (cron), in which case callers fall back to plain lines."""
    if sys.stderr.isatty():
        return Console(file=sys.stderr), False
    try:
        return Console(file=open("/dev/tty", "w")), True
    except OSError:
        return None, False


class Board:
    def __init__(self, total: int, console: Console | None = None):
        self.console = console or Console()
        self.overall = Progress(
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("Mean: {task.fields[mean]:.3f}"),
            BarColumn(bar_width=40),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console)
        self.jobs = Progress(
            TextColumn("  "),
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            console=self.console)
        self._task = self.overall.add_task("", total=total, mean=0.0)
        self._lock = threading.Lock()
        self._done = 0
        self._metric = 0.0
        # redirect_* MUST stay off: rich's default hijacks the process's
        # stdout/stderr into this console while live — with the board on
        # /dev/tty that would silently drain the tee'd pipes (TSV, logs)
        # onto the screen instead of into the log file
        self._live = Live(Group(self.overall, self.jobs),
                          console=self.console, refresh_per_second=4,
                          redirect_stdout=False, redirect_stderr=False)
        self._live.start()

    def begin(self, name: str):
        """One unit of work started: a spinner row with its own clock."""
        return self.jobs.add_task(name, total=None)

    def log(self, line: str) -> None:
        """Print a full line above the live region (plain on non-TTY)."""
        self.console.print(line, markup=False, highlight=False)

    def advance(self, handle=None, metric: float = 0.0) -> None:
        """One unit finished; drops its spinner row, folds in the metric."""
        with self._lock:
            self._done += 1
            self._metric += float(metric)
            self.overall.update(self._task, completed=self._done,
                                mean=self._metric / self._done)
        if handle is not None:
            try:
                self.jobs.remove_task(handle)
            except KeyError:
                pass

    def close(self) -> None:
        self._live.stop()
