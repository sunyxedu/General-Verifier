"""Pinned, host-configured evaluators produce observations, never verdicts."""
from __future__ import annotations

import base64
import hashlib
import os
import selectors
import signal
import stat
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ...core.model import content_hash
from ...execution.instruments import ArtifactStore
from .candidates import CandidateError, _identity, _read_fd, _regular_fd

MAX_OUTPUT_BYTES = 64 * 1024  # independently bounded stdout and stderr


class EvaluatorError(ValueError):
    """An evaluator configuration or subject failed its mechanical binding."""


def _file_pin(path: str, executable: bool = False) -> tuple[str, tuple[int, ...]]:
    if not isinstance(path, str) or not os.path.isabs(path):
        raise EvaluatorError("evaluator executable must be an absolute path")
    try:
        fd = _regular_fd(os.path.sep, path)
        try:
            info = os.fstat(fd)
            if executable and not info.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                raise EvaluatorError("evaluator executable is not executable")
            before = _identity(info)
            digest = hashlib.sha256()
            while chunk := os.read(fd, 65536):
                digest.update(chunk)
            if before != _identity(os.fstat(fd)):
                raise EvaluatorError("evaluator executable changed while being pinned")
            return digest.hexdigest(), before
        finally:
            os.close(fd)
    except (OSError, CandidateError) as exc:
        raise EvaluatorError("evaluator executable is unavailable or contains a symlink") from exc


@dataclass(frozen=True)
class EvaluatorSpec:
    argv: tuple[str, ...]
    environment: str = "host"
    authority_scope: str = "candidate_signal"
    timeout: float | None = None
    pinned_files: tuple[str, ...] = ()
    _executable_sha256: str = field(init=False, repr=False)
    _executable_identity: tuple[int, ...] = field(init=False, repr=False)
    _file_pins: tuple[tuple[str, str, tuple[int, ...]], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (not isinstance(self.argv, tuple) or not self.argv
                or any(not isinstance(v, str) or not v or "\x00" in v for v in self.argv)
                or self.argv.count("{candidate}") != 1
                or any("{candidate}" in v and v != "{candidate}" for v in self.argv)):
            raise EvaluatorError("argv requires exactly one standalone {candidate} token")
        if not isinstance(self.environment, str) or not self.environment or len(self.environment) > 4096:
            raise EvaluatorError("evaluator requires a host-supplied environment label")
        if self.authority_scope not in ("candidate_signal", "task_outcome"):
            raise EvaluatorError("unknown evaluator authority scope")
        if (self.timeout is not None
                and (type(self.timeout) not in (int, float)
                     or not 0 < self.timeout < float("inf"))):
            raise EvaluatorError("timeout must be positive and finite, or None")
        if (not isinstance(self.pinned_files, tuple)
                or any(not isinstance(v, str) or not os.path.isabs(v) for v in self.pinned_files)
                or len(set(self.pinned_files)) != len(self.pinned_files)):
            raise EvaluatorError("pinned_files must contain unique absolute host file paths")
        digest, identity = _file_pin(self.argv[0], executable=True)
        object.__setattr__(self, "_executable_sha256", digest)
        object.__setattr__(self, "_executable_identity", identity)
        object.__setattr__(self, "_file_pins", tuple(
            (path, *_file_pin(path)) for path in self.pinned_files))

    @property
    def identity(self) -> dict[str, Any]:
        return {"schema": "cyber-evaluator-v1", "argv": list(self.argv),
                "executable_sha256": self._executable_sha256,
                "pinned_files": [{"path": path, "sha256": digest}
                                 for path, digest, _ in self._file_pins],
                "environment": self.environment,
                "authority_scope": self.authority_scope, "timeout": self.timeout}

    @property
    def id(self) -> str:
        return content_hash(self.identity)

    def validate(self) -> None:
        digest, identity = _file_pin(self.argv[0], executable=True)
        if digest != self._executable_sha256 or identity != self._executable_identity:
            raise EvaluatorError("evaluator executable pin changed")
        for path, digest, identity in self._file_pins:
            if _file_pin(path) != (digest, identity):
                raise EvaluatorError("evaluator declared dependency pin changed")


class Evaluator:
    def __init__(self, spec: EvaluatorSpec, artifacts: ArtifactStore):
        self.spec, self.artifacts = spec, artifacts

    def _candidate_identity(self, candidate: dict[str, Any]) -> tuple[int, ...]:
        digest, size, path = (candidate.get("sha256"), candidate.get("size"),
                              candidate.get("blob_path"))
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or candidate.get("candidate_id") != digest
                or type(size) is not int or not isinstance(path, str)
                or not os.path.isabs(path) or os.path.basename(path) != digest):
            raise EvaluatorError("invalid evaluator candidate metadata")
        try:
            artifact_id = candidate["artifact_id"]
            if (not isinstance(artifact_id, str) or len(artifact_id) != 64
                    or any(c not in "0123456789abcdef" for c in artifact_id)):
                raise EvaluatorError("invalid candidate artifact identity")
            artifact = self.artifacts.get(artifact_id)
            expected = {k: v for k, v in candidate.items() if k != "artifact_id"}
            if content_hash(artifact) != artifact_id or artifact != expected:
                raise EvaluatorError("candidate artifact identity mismatch")
            fd = _regular_fd(os.path.sep, path)
            try:
                identity = _identity(os.fstat(fd))
                data = _read_fd(fd)
                if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                    raise EvaluatorError("candidate digest mismatch")
                return identity
            finally:
                os.close(fd)
        except (OSError, KeyError, CandidateError) as exc:
            raise EvaluatorError("evaluator candidate cannot be safely verified") from exc

    @staticmethod
    def _kill_group(process: subprocess.Popen) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def run(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Execute with no inherited credentials and record bounded raw output.

        Exit codes and output strings remain observations even when the host has
        explicitly supplied a task-outcome evaluator. No generic success parser
        or score is introduced here. The Service attaches the ordinary Effect.
        """
        self.spec.validate()
        candidate_identity = self._candidate_identity(candidate)
        argv = [candidate["blob_path"] if v == "{candidate}" else v
                for v in self.spec.argv]
        started = time.monotonic()
        captured = {"stdout": bytearray(), "stderr": bytearray()}
        totals = {"stdout": 0, "stderr": 0}
        incomplete = {"stdout": False, "stderr": False}
        timed_out = False
        execution_error = None
        exit_code = None
        process = None
        with tempfile.TemporaryDirectory(prefix="gv-evaluator-") as temporary:
            env = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C",
                   "TMPDIR": temporary, "TMP": temporary, "TEMP": temporary,
                   "HOME": temporary, "XDG_CONFIG_HOME": temporary}
            try:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           env=env, cwd=temporary, start_new_session=True)
                selector = selectors.DefaultSelector()
                assert process.stdout is not None and process.stderr is not None
                for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                    os.set_blocking(pipe.fileno(), False)
                    selector.register(pipe, selectors.EVENT_READ, name)
                exited_at = None
                killed_at = None
                try:
                    while selector.get_map():
                        now = time.monotonic()
                        if (self.spec.timeout is not None and killed_at is None
                                and now - started >= self.spec.timeout):
                            timed_out = True
                            self._kill_group(process)
                            killed_at = now
                        if process.poll() is not None and exited_at is None:
                            exited_at = now
                        # A descendant may retain either pipe after the leader
                        # exits, or even escape its process group. Closing the
                        # pipes after a short drain prevents an unbounded wait.
                        drain_since = killed_at if killed_at is not None else exited_at
                        if drain_since is not None and now - drain_since >= 0.25:
                            self._kill_group(process)
                            for key in list(selector.get_map().values()):
                                incomplete[key.data] = True
                                selector.unregister(key.fileobj)
                                key.fileobj.close()
                            break
                        for key, _ in selector.select(0.025):
                            try:
                                chunk = os.read(key.fileobj.fileno(), 65536)
                            except BlockingIOError:
                                continue
                            if not chunk:
                                selector.unregister(key.fileobj)
                                key.fileobj.close()
                                continue
                            name = key.data
                            totals[name] += len(chunk)
                            remaining = MAX_OUTPUT_BYTES - len(captured[name])
                            captured[name].extend(chunk[:remaining])
                    # Pipes can close while the leader keeps running. Wait for
                    # the leader as well, still respecting an optional timeout.
                    while process.poll() is None:
                        if (self.spec.timeout is not None
                                and time.monotonic() - started >= self.spec.timeout):
                            timed_out = True
                            self._kill_group(process)
                            break
                        time.sleep(0.025)
                    exit_code = process.wait(timeout=1)
                finally:
                    selector.close()
                    for pipe in (process.stdout, process.stderr):
                        if not pipe.closed:
                            pipe.close()
            except (OSError, subprocess.TimeoutExpired) as exc:
                execution_error = f"{type(exc).__name__}: {exc}"
            finally:
                if process is not None and process.poll() is None:
                    self._kill_group(process)
                    try:
                        exit_code = process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
        if self._candidate_identity(candidate) != candidate_identity:
            raise EvaluatorError("candidate path identity changed during evaluation")
        self.spec.validate()
        observation: dict[str, Any] = {
            "schema": "cyber-evaluator-observation-v1", "execution_id": uuid.uuid4().hex,
            "evaluator_id": self.spec.id, "candidate_id": candidate["candidate_id"],
            "candidate_sha256": candidate["sha256"], "candidate_size": candidate["size"],
            "environment": self.spec.environment, "authority_scope": self.spec.authority_scope,
            "exit_code": exit_code, "timed_out": timed_out,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "execution_error": execution_error,
        }
        for name in ("stdout", "stderr"):
            raw = bytes(captured[name])
            observation[name] = raw.decode("utf-8", errors="replace")
            observation[name + "_base64"] = base64.b64encode(raw).decode("ascii")
            observation[name + "_bytes_observed"] = totals[name]
            observation[name + "_truncated"] = incomplete[name] or totals[name] > len(raw)
        artifact_id = self.artifacts.put(observation)
        return {**observation, "artifact_id": artifact_id,
                "artifact_path": os.path.join(self.artifacts.root, artifact_id + ".json")}
