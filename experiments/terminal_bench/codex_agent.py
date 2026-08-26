"""Harbor Codex agent with explicit tier and post-run runtime capture.

Harbor 0.20's built-in Codex wrapper does not yet expose Codex's
``service_tier`` and ``fast_mode`` CLI settings.  The subclass also takes a
task-agnostic process/network snapshot after the solver returns and before the
official verifier runs, so offline verification is not limited to file state.
"""
import shlex
from pathlib import Path

from harbor.agents.installed.base import CliFlag
from harbor.agents.installed.codex import Codex


RUNTIME_SNAPSHOT_ROOT = "/gv-runtime-snapshot"
_REMOTE_PYTHON_COLLECTOR = "/tmp/gv-runtime-snapshot-collector.py"
_REMOTE_SHELL_COLLECTOR = "/tmp/gv-runtime-snapshot-collector.sh"


class TieredCodex(Codex):
    """Stock Harbor Codex plus reproducible tier and runtime evidence."""

    CLI_FLAGS = [
        *Codex.CLI_FLAGS,
        CliFlag(
            "service_tier",
            cli="-c",
            type="enum",
            choices=["default", "fast"],
            default="default",
            format='-c service_tier="{value}"',
        ),
        CliFlag(
            "fast_mode",
            cli="--enable",
            type="enum",
            choices=["enable", "disable"],
            default="disable",
            format="--{value} fast_mode",
        ),
    ]

    async def _capture_runtime_snapshot(self, environment) -> None:
        """Best-effort harness observation; never convert solver success to failure."""
        local_python_collector = Path(__file__).with_name("runtime_snapshot.py")
        local_shell_collector = Path(__file__).with_name("runtime_snapshot.sh")
        stdout_path = "/tmp/gv-runtime-snapshot.stdout"
        stderr_path = "/tmp/gv-runtime-snapshot.stderr"
        try:
            await environment.upload_file(
                local_python_collector.as_posix(), _REMOTE_PYTHON_COLLECTOR)
            await environment.upload_file(
                local_shell_collector.as_posix(), _REMOTE_SHELL_COLLECTOR)
            command = (
                f"rm -rf -- {shlex.quote(RUNTIME_SNAPSHOT_ROOT)}\n"
                "if command -v python3 >/dev/null 2>&1; then\n"
                "  gv_snapshot_python=$(command -v python3)\n"
                "elif command -v python >/dev/null 2>&1; then\n"
                "  gv_snapshot_python=$(command -v python)\n"
                "else\n"
                "  gv_snapshot_python=\n"
                "fi\n"
                "if [ -n \"$gv_snapshot_python\" ]; then\n"
                f"  \"$gv_snapshot_python\" "
                f"{shlex.quote(_REMOTE_PYTHON_COLLECTOR)} "
                f"--output {shlex.quote(RUNTIME_SNAPSHOT_ROOT)} "
                f">{shlex.quote(stdout_path)} 2>{shlex.quote(stderr_path)}\n"
                "  gv_snapshot_rc=$?\n"
                "else\n"
                f"  /bin/sh {shlex.quote(_REMOTE_SHELL_COLLECTOR)} "
                f"--output {shlex.quote(RUNTIME_SNAPSHOT_ROOT)} "
                f">{shlex.quote(stdout_path)} 2>{shlex.quote(stderr_path)}\n"
                "  gv_snapshot_rc=$?\n"
                "fi\n"
                f"mkdir -p {shlex.quote(RUNTIME_SNAPSHOT_ROOT)}\n"
                f"printf '%s\\n' \"$gv_snapshot_rc\" >"
                f"{shlex.quote(RUNTIME_SNAPSHOT_ROOT + '/collector.returncode')}\n"
                f"test ! -f {shlex.quote(stdout_path)} || cp "
                f"{shlex.quote(stdout_path)} "
                f"{shlex.quote(RUNTIME_SNAPSHOT_ROOT + '/collector.stdout')}\n"
                f"test ! -f {shlex.quote(stderr_path)} || cp "
                f"{shlex.quote(stderr_path)} "
                f"{shlex.quote(RUNTIME_SNAPSHOT_ROOT + '/collector.stderr')}\n"
                f"rm -f -- {shlex.quote(_REMOTE_PYTHON_COLLECTOR)} "
                f"{shlex.quote(_REMOTE_SHELL_COLLECTOR)} "
                f"{shlex.quote(stdout_path)} {shlex.quote(stderr_path)}\n"
                "exit 0"
            )
            await self.exec_as_root(environment, command=command)
        except Exception as exc:  # snapshot absence stays visible in manifest
            self.logger.warning("Post-run runtime snapshot failed: %s", exc)

    async def run(self, instruction, environment, context) -> None:
        try:
            await super().run(instruction, environment, context)
        finally:
            await self._capture_runtime_snapshot(environment)
