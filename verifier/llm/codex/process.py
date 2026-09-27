"""Launching one ``codex exec`` process in its own POSIX process group,
bounded by a timeout and the executor's scratch budget, and killing the
whole group afterwards."""
from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys

from ..schema import RoleError, SeatScratchExceeded
from .paths import _BoundExecutionDirectory


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill processes in the dedicated POSIX group led by ``proc``.

    ``start_new_session=True`` makes the child's pid the process-group id.
    Calling killpg after the leader has exited is intentional: detached
    same-group descendants may still be alive. A descendant that moved to
    its own process group or session is found through /proc where there is
    one (resource_share.kill_escaped) and killed first, while its parents
    still link it to the seat.
    """
    from ...execution.resource_share import kill_escaped
    kill_escaped(proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        # Non-conforming platforms are handled by the direct-process fallback.
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _run_codex_process(cmd, *, cwd, stdin, timeout, input_text=None,
                       watch_dir=None, watch_cap=0):
    """Run one CLI attempt and clean its dedicated POSIX process group.
    With ``input_text`` the prompt is written to the process's stdin (the
    command then names `-` as its prompt); otherwise stdin is the given
    closed handle — codex waits forever on an open, silent stdin."""
    bound_cwd = (cwd if isinstance(cwd, _BoundExecutionDirectory) else None)
    if bound_cwd is not None and os.name != "posix":
        raise OSError(
            errno.ENOTSUP, "fd-bound Codex cwd requires a POSIX host")
    if input_text is not None:
        stdin = subprocess.PIPE
    run_kwargs = {
        "cwd": cwd,
        "stdin": stdin,
        "capture_output": True,
        "text": True,
        "timeout": timeout,
    }
    if input_text is not None:
        run_kwargs.pop("stdin")
        run_kwargs["input"] = input_text
    # Checked before waiting for room: Popen refuses an argument holding a
    # NUL byte with a ValueError, which no retry path recognises, and two
    # judger calls met it only after 526 s and 688 s in the room queue.
    if any("\0" in str(arg) for arg in cmd):
        raise RoleError("codex command line holds a NUL byte")
    if os.name != "posix" or not hasattr(os, "killpg"):
        return subprocess.run(cmd, **run_kwargs)
    from ...execution.resource_share import launch_slot
    with launch_slot():
        return _run_codex_group(cmd, cwd=cwd, stdin=stdin, timeout=timeout,
                                input_text=input_text, bound_cwd=bound_cwd,
                                watch_dir=watch_dir, watch_cap=watch_cap)


def _run_codex_group(cmd, *, cwd, stdin, timeout, input_text, bound_cwd,
                     watch_dir, watch_cap):
    launch_cmd = cmd
    launch_cwd = cwd
    launch_kwargs = {}
    if bound_cwd is not None:
        # Popen's cwd parameter resolves a pathname after our policy checks.
        # The tiny isolated launcher instead enters the already-open directory
        # and immediately replaces itself with Codex.  Closing the inherited
        # descriptor before exec prevents it becoming an ambient capability.
        launch_cmd = [
            sys.executable, "-I", "-c",
            "import os,sys;f=int(sys.argv[1]);os.fchdir(f);os.close(f);"
            "os.execvp(sys.argv[2],sys.argv[2:])",
            str(bound_cwd.descriptor), *cmd,
        ]
        launch_cwd = None
        launch_kwargs["pass_fds"] = (bound_cwd.descriptor,)
    proc = subprocess.Popen(
        launch_cmd, cwd=launch_cwd, stdin=stdin,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True, **launch_kwargs,
    )
    from ...execution.resource_share import (
        track_launched, untrack_launched, yield_process)
    track_launched(proc.pid)
    yield_process(proc.pid)
    from ...execution.scratch_budget import ScratchWatch
    watch = ScratchWatch(watch_dir, watch_cap, lambda: _kill_process_group(proc))
    try:
        try:
            try:
                with watch:
                    stdout, stderr = proc.communicate(input=input_text,
                                                      timeout=timeout)
                if watch.exceeded:
                    raise SeatScratchExceeded(
                        f"the check's scratch directory outgrew its "
                        f"{watch_cap >> 30} GB budget and the check was stopped "
                        f"(copy only what the check exercises; never a whole "
                        f"filesystem or a device file)")
            except subprocess.TimeoutExpired as error:
                _kill_process_group(proc)
                stdout, stderr = proc.communicate()
                # Match subprocess.run's useful TimeoutExpired payload.
                error.output = stdout
                error.stdout = stdout
                error.stderr = stderr
                raise
            return subprocess.CompletedProcess(
                cmd, proc.returncode, stdout=stdout, stderr=stderr)
        finally:
            # A successful wrapper may leave a same-group child after closing
            # inherited pipes. This finally also covers interruption and
            # communicate() errors, not just normal completion and timeout.
            _kill_process_group(proc)
            untrack_launched(proc.pid)
            if proc.poll() is None:
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
    except BaseException as error:
        # The backend uses this private marker to distinguish a launched call
        # interrupted by its caller from a failure before Popen returned.
        try:
            setattr(error, "_gv_process_started", True)
        except (AttributeError, TypeError):
            pass
        raise
