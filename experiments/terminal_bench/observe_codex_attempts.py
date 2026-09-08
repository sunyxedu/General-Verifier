"""Non-semantic Codex CLI attempt observer for cross-version evaluations.

This module is an evaluation shim, not a verifier backend.  It wraps the
existing backend entry point and one existing process-call seam without
changing commands, retries, parsing, recovery, exceptions, or results.  The
ledger is deliberately content-free: stdout is represented only by its byte
count, SHA-256, and strictly validated native usage counters.

The shim does not install an execution sandbox or otherwise equalize runtime
permissions.  An evaluation that needs a common isolation policy must pin and
audit that policy separately.
"""
from __future__ import annotations

import contextvars
import copy
import functools
import hashlib
import json
import math
import os
import stat
import subprocess
import threading
import time
import types
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Sequence


OBSERVER_SCHEMA = "gv-codex-attempt-observer-v1"
OBSERVER_SUMMARY_SCHEMA = "gv-codex-attempt-observer-summary-v1"
LEGACY_SUBPROCESS_SEAM = "legacy-subprocess-run"
RUN_CODEX_PROCESS_SEAM = "run-codex-process"
_SEAMS = frozenset({LEGACY_SUBPROCESS_SEAM, RUN_CODEX_PROCESS_SEAM})
_USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)
_MAX_CONTEXTS = 10_000
_MAX_CONTEXT_FIELDS = 32
_MAX_CONTEXT_KEY_CHARS = 64
_MAX_CONTEXT_STRING_CHARS = 512
_MAX_ERROR_KIND_CHARS = 128
_MAX_LEDGER_BYTES = 64 * 1024 * 1024

_INSTALL_LOCK = threading.Lock()
_INSTALLED: "weakref.WeakKeyDictionary[types.ModuleType, object]" = (
    weakref.WeakKeyDictionary()
)


@dataclass
class _CallFrame:
    call_id: str
    role: Optional[str]
    model: Optional[str]
    effort: Optional[str]
    service_tier: Optional[str]
    runtime_instruction_identity: Optional[str]
    typed_judger_transport: Any
    typed_judger_protocol: Any
    typed_judger_packet_sha256: Any
    state_dir: Optional[str]
    attribution: str
    context: Mapping[str, Any]
    started_at_unix_ns: int
    started_at_monotonic_ns: int
    attempt_count: int = 0


@dataclass(frozen=True)
class _Usage:
    status: str
    complete: bool
    thread_id: Optional[str]
    input_tokens: Optional[int]
    cached_input_tokens: Optional[int]
    uncached_input_tokens: Optional[int]
    cache_write_input_tokens: Optional[int]
    output_tokens: Optional[int]
    reasoning_tokens: Optional[int]


class ObservationSession:
    """Runtime handle yielded by :func:`observe_codex_attempts`.

    ``write_failures`` is diagnostic only.  Observation failures never alter
    the wrapped backend; the downstream completeness audit must reject a
    missing start/terminal pair.
    """

    def __init__(
        self,
        ledger_path: str | os.PathLike[str],
        seam: str,
        context_by_cwd: Mapping[str | os.PathLike[str], Mapping[str, Any]],
    ) -> None:
        raw_ledger_path = os.path.abspath(os.fspath(ledger_path))
        if os.path.lexists(raw_ledger_path):
            raise FileExistsError("observer ledger must be a fresh path")
        self.ledger_path = os.path.realpath(raw_ledger_path)
        self.seam = seam
        self.observer_id = uuid.uuid4().hex
        self.write_failures = 0
        self.restoration_failed = False
        self.closed = False
        self._ledger_sha256: Optional[str] = None
        self._ledger_bytes: Optional[int] = None
        self._write_lock = threading.Lock()
        self._summary_lock = threading.Lock()
        self._event_counts = {
            "codex_logical_call_started": 0,
            "codex_logical_call_finished": 0,
            "codex_cli_attempt_started": 0,
            "codex_cli_attempt_finished": 0,
        }
        self._active_calls: set[str] = set()
        self._calls: contextvars.ContextVar[Optional[_CallFrame]] = (
            contextvars.ContextVar(
                f"gv_codex_observer_{self.observer_id}", default=None)
        )
        self._context_by_cwd = _copy_contexts(context_by_cwd)
        for state_dir in self._context_by_cwd:
            if _path_is_within(state_dir, self.ledger_path):
                raise ValueError(
                    "observer ledger must be outside every candidate state")
        self._ledger_identity = self._create_fresh_ledger()

    def _create_fresh_ledger(self) -> tuple[int, int]:
        parent = os.path.dirname(self.ledger_path) or "."
        os.makedirs(parent, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.ledger_path, flags, 0o600)
        try:
            opened = os.fstat(fd)
            if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                    or opened.st_uid != os.geteuid()):
                raise OSError(
                    "observer ledger must be a private owned regular file")
            os.fchmod(fd, 0o600)
            os.fsync(fd)
            return opened.st_dev, opened.st_ino
        finally:
            os.close(fd)

    def _note_failure(self) -> None:
        with self._summary_lock:
            self.write_failures += 1

    def _set_call_active(self, call_id: str, active: bool) -> None:
        with self._summary_lock:
            if active:
                self._active_calls.add(call_id)
            else:
                self._active_calls.discard(call_id)

    def summary(self) -> dict[str, Any]:
        """Return the out-of-band lifecycle totals required by the gate."""
        with self._summary_lock:
            counts = dict(self._event_counts)
            return {
                "schema": OBSERVER_SUMMARY_SCHEMA,
                "observer_id": self.observer_id,
                "seam": self.seam,
                "logical_calls_started": counts[
                    "codex_logical_call_started"],
                "logical_calls_finished": counts[
                    "codex_logical_call_finished"],
                "attempts_started": counts["codex_cli_attempt_started"],
                "attempts_finished": counts["codex_cli_attempt_finished"],
                "write_failures": self.write_failures,
                "active_calls": len(self._active_calls),
                "restoration_failed": self.restoration_failed,
                "closed": self.closed,
                "ledger_path": self.ledger_path,
                "ledger_sha256": self._ledger_sha256,
                "ledger_bytes": self._ledger_bytes,
            }

    def _close(self, restoration_failed: bool) -> None:
        """Seal the ledger identity after the wrappers are uninstalled."""
        with self._summary_lock:
            self.restoration_failed = restoration_failed
        try:
            with self._write_lock:
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(self.ledger_path, flags)
                try:
                    opened = os.fstat(fd)
                    if (not stat.S_ISREG(opened.st_mode)
                            or opened.st_nlink != 1
                            or opened.st_uid != os.geteuid()
                            or (opened.st_dev, opened.st_ino)
                            != self._ledger_identity
                            or opened.st_size > _MAX_LEDGER_BYTES):
                        raise OSError("observer ledger cannot be sealed")
                    chunks = []
                    total = 0
                    while True:
                        chunk = os.read(fd, 1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > _MAX_LEDGER_BYTES:
                            raise OSError("observer ledger exceeds size cap")
                        chunks.append(chunk)
                    after = os.fstat(fd)
                    if ((after.st_dev, after.st_ino, after.st_size,
                         after.st_mtime_ns)
                            != (opened.st_dev, opened.st_ino, opened.st_size,
                                opened.st_mtime_ns)
                            or total != opened.st_size):
                        raise OSError("observer ledger changed while sealing")
                finally:
                    os.close(fd)
            raw = b"".join(chunks)
            with self._summary_lock:
                self._ledger_sha256 = hashlib.sha256(raw).hexdigest()
                self._ledger_bytes = len(raw)
        except BaseException:
            self._note_failure()
        finally:
            with self._summary_lock:
                self.closed = True

    def _base(self, record_type: str) -> dict[str, Any]:
        return {
            "schema": OBSERVER_SCHEMA,
            "record_type": record_type,
            "observer_id": self.observer_id,
            "seam": self.seam,
        }

    def _append(self, record: Mapping[str, Any]) -> None:
        record_type = record.get("record_type")
        with self._summary_lock:
            if record_type in self._event_counts:
                self._event_counts[record_type] += 1
        try:
            encoded = (json.dumps(
                dict(record), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n").encode("utf-8")
            parent = os.path.dirname(self.ledger_path) or "."
            with self._write_lock:
                os.makedirs(parent, exist_ok=True)
                flags = os.O_WRONLY | os.O_APPEND
                flags |= getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(self.ledger_path, flags, 0o600)
                try:
                    opened = os.fstat(fd)
                    if (not stat.S_ISREG(opened.st_mode)
                            or opened.st_nlink != 1
                            or opened.st_uid != os.geteuid()
                            or (opened.st_dev, opened.st_ino)
                            != self._ledger_identity):
                        raise OSError("observer ledger identity changed")
                    written = os.write(fd, encoded)
                    if written != len(encoded):
                        raise OSError("short observer ledger write")
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except BaseException:
            # The shim must be observational.  A missing row makes the later
            # accounting audit incomplete; it must not change a role result.
            self._note_failure()

    @staticmethod
    def _frame_fields(frame: _CallFrame) -> dict[str, Any]:
        return {
            "call_id": frame.call_id,
            "role": frame.role,
            "model": frame.model,
            "effort": frame.effort,
            "service_tier": frame.service_tier,
            "runtime_instruction_identity": frame.runtime_instruction_identity,
            "typed_judger_transport": frame.typed_judger_transport,
            "typed_judger_protocol": frame.typed_judger_protocol,
            "typed_judger_packet_sha256": (
                frame.typed_judger_packet_sha256),
            "state_dir": frame.state_dir,
            "attribution": frame.attribution,
            "context": dict(frame.context),
        }

    def _new_frame(self, backend: Any, role_name: Any) -> _CallFrame:
        state_dir = _canonical_path(getattr(backend, "cwd", None))
        context = self._context_by_cwd.get(state_dir, {})
        attribution = "attributed" if state_dir in self._context_by_cwd \
            else "unattributed"
        # Never call backend hooks merely to collect metadata. The actual
        # command row independently records the model passed to the CLI.
        model = (backend.model
                 if isinstance(getattr(backend, "model", None), str)
                 else None)
        now_unix = time.time_ns()
        now_mono = time.monotonic_ns()
        return _CallFrame(
            call_id=uuid.uuid4().hex,
            role=role_name if isinstance(role_name, str) else None,
            model=model,
            effort=(getattr(backend, "effort", None)
                    if isinstance(getattr(backend, "effort", None), str)
                    else None),
            service_tier=(getattr(backend, "service_tier", None)
                          if isinstance(
                              getattr(backend, "service_tier", None), str)
                          else None),
            runtime_instruction_identity=(
                getattr(backend, "runtime_instruction_identity", None)
                if isinstance(getattr(
                    backend, "runtime_instruction_identity", None), str)
                else None),
            typed_judger_transport=getattr(
                backend, "typed_judger_transport", False),
            typed_judger_protocol=getattr(
                backend, "typed_judger_protocol", None),
            typed_judger_packet_sha256=getattr(
                backend, "typed_judger_packet_sha256", None),
            state_dir=state_dir,
            attribution=attribution,
            context=context,
            started_at_unix_ns=now_unix,
            started_at_monotonic_ns=now_mono,
        )

    def complete_wrapper(self, original):
        @functools.wraps(original)
        def wrapped(backend, role_name: str, prompt: str, variables: dict):
            try:
                frame = self._new_frame(backend, role_name)
                token = self._calls.set(frame)
            except BaseException:
                # Even an observer implementation failure may only create an
                # incomplete ledger; it may not prevent the underlying call.
                self._note_failure()
                return original(backend, role_name, prompt, variables)
            self._set_call_active(frame.call_id, True)
            self._append({
                **self._base("codex_logical_call_started"),
                **self._frame_fields(frame),
                "started_at_unix_ns": frame.started_at_unix_ns,
                "started_at_monotonic_ns": frame.started_at_monotonic_ns,
            })
            error: Optional[BaseException] = None
            try:
                return original(backend, role_name, prompt, variables)
            except BaseException as caught:
                error = caught
                raise
            finally:
                try:
                    finished_unix = time.time_ns()
                    finished_mono = time.monotonic_ns()
                    self._append({
                        **self._base("codex_logical_call_finished"),
                        **self._frame_fields(frame),
                        "attempt_count": frame.attempt_count,
                        "terminal_status": (
                            "returned" if error is None else "raised"),
                        "error_kind": (_error_kind(error)
                                       if error is not None else None),
                        "started_at_unix_ns": frame.started_at_unix_ns,
                        "finished_at_unix_ns": finished_unix,
                        "duration_ns": max(
                            0,
                            finished_mono - frame.started_at_monotonic_ns),
                    })
                except BaseException:
                    self._note_failure()
                finally:
                    self._set_call_active(frame.call_id, False)
                    self._calls.reset(token)

        return wrapped

    def process_wrapper(self, original):
        @functools.wraps(original)
        def wrapped(cmd, *, cwd, stdin, timeout):
            try:
                observed = _is_codex_exec(cmd)
            except BaseException:
                self._note_failure()
                observed = False
            if not observed:
                return original(cmd, cwd=cwd, stdin=stdin, timeout=timeout)
            return self._observe_process(
                cmd, cwd,
                lambda: original(cmd, cwd=cwd, stdin=stdin, timeout=timeout),
            )

        return wrapped

    def subprocess_run_wrapper(self, original):
        @functools.wraps(original)
        def wrapped(*popenargs, **kwargs):
            try:
                cmd = _subprocess_command(popenargs, kwargs)
                observed = _is_codex_exec(cmd)
            except BaseException:
                self._note_failure()
                observed = False
            if not observed:
                return original(*popenargs, **kwargs)
            return self._observe_process(
                cmd, kwargs.get("cwd"),
                lambda: original(*popenargs, **kwargs),
            )

        return wrapped

    def _orphan_frame(self, cwd: Any, command: Any) -> _CallFrame:
        actual = _command_profile(command)
        now_unix = time.time_ns()
        now_mono = time.monotonic_ns()
        return _CallFrame(
            call_id=uuid.uuid4().hex,
            role=None,
            model=actual[0],
            effort=None,
            service_tier=None,
            runtime_instruction_identity=None,
            typed_judger_transport=False,
            typed_judger_protocol=None,
            typed_judger_packet_sha256=None,
            state_dir=_canonical_path(cwd),
            attribution="unattributed",
            context={},
            started_at_unix_ns=now_unix,
            started_at_monotonic_ns=now_mono,
        )

    def _observe_process(self, command: Any, cwd: Any, invoke):
        try:
            frame = self._calls.get()
            if frame is None:
                frame = self._orphan_frame(cwd, command)
            frame.attempt_count += 1
            attempt_index = frame.attempt_count
            attempt_id = uuid.uuid4().hex
            started_unix = time.time_ns()
            started_mono = time.monotonic_ns()
            command_sha = _command_sha256(command)
            command_model, command_effort, command_tier = _command_profile(
                command)
            common = {
                **self._frame_fields(frame),
                "attempt_id": attempt_id,
                "attempt_index": attempt_index,
                "command_sha256": command_sha,
                "command_model": command_model,
                "command_effort": command_effort,
                "command_service_tier": command_tier,
                "cwd": _canonical_path(cwd),
            }
            self._append({
                **self._base("codex_cli_attempt_started"),
                **common,
                "started_at_unix_ns": started_unix,
                "started_at_monotonic_ns": started_mono,
            })
        except BaseException:
            self._note_failure()
            return invoke()

        result = None
        error: Optional[BaseException] = None
        try:
            result = invoke()
            return result
        except BaseException as caught:
            error = caught
            raise
        finally:
            try:
                finished_unix = time.time_ns()
                finished_mono = time.monotonic_ns()
                stdout = _result_stdout(result, error)
                stdout_sha, stdout_bytes, stdout_text = _stdout_identity(
                    stdout)
                # A timeout or interruption may expose a plausible completed
                # event only as partial output. Never certify that as the
                # complete cost of the launched process.
                usage = (_parse_usage(stdout_text) if error is None else
                         _incomplete_usage(
                             "partial" if stdout_text else "unknown"))
                terminal_status, process_started = _process_terminal(
                    result, error)
                self._append({
                    **self._base("codex_cli_attempt_finished"),
                    **common,
                    "terminal_status": terminal_status,
                    "process_started": process_started,
                    "returncode": _returncode(result),
                    "error_kind": (_error_kind(error)
                                   if error is not None else None),
                    "started_at_unix_ns": started_unix,
                    "finished_at_unix_ns": finished_unix,
                    "duration_ns": max(0, finished_mono - started_mono),
                    "stdout_sha256": stdout_sha,
                    "stdout_bytes": stdout_bytes,
                    "usage_status": usage.status,
                    "usage_complete": usage.complete,
                    "thread_id": usage.thread_id,
                    "input_tokens": usage.input_tokens,
                    "cached_input_tokens": usage.cached_input_tokens,
                    "uncached_input_tokens": usage.uncached_input_tokens,
                    "cache_write_input_tokens": (
                        usage.cache_write_input_tokens),
                    "output_tokens": usage.output_tokens,
                    "reasoning_tokens": usage.reasoning_tokens,
                })
            except BaseException:
                # In particular, never replace an original process exception
                # with an observer parser or filesystem exception.
                self._note_failure()


class _SubprocessProxy:
    """Replace one module binding without mutating the stdlib module."""

    def __init__(self, original: Any, run) -> None:
        self._original = original
        self.run = run

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


@contextmanager
def observe_codex_attempts(
    module: types.ModuleType,
    ledger_path: str | os.PathLike[str],
    *,
    seam: str,
    context_by_cwd: Mapping[
        str | os.PathLike[str], Mapping[str, Any]
    ],
) -> Iterator[ObservationSession]:
    """Temporarily observe native Codex attempts made by ``module``.

    ``legacy-subprocess-run`` targets the original-main backend, whose retry
    loop called its module-local ``subprocess.run`` directly.
    ``run-codex-process`` targets the current backend's private process seam.
    No process launcher is replaced: each wrapper delegates to the exact
    original callable with the exact arguments it received.
    """
    if not isinstance(module, types.ModuleType):
        raise TypeError("module must be an imported Python module")
    if seam not in _SEAMS:
        raise ValueError(f"unsupported Codex observer seam: {seam!r}")
    backend = getattr(module, "CodexBackend", None)
    original_complete = getattr(backend, "complete", None)
    if not isinstance(backend, type) or not callable(original_complete):
        raise TypeError("module must expose CodexBackend.complete")

    seam_owner: Any
    seam_attribute: str
    original_seam: Any
    installed_seam: Any
    if seam == RUN_CODEX_PROCESS_SEAM:
        seam_owner = module
        seam_attribute = "_run_codex_process"
        original_seam = getattr(module, seam_attribute, None)
        if not callable(original_seam):
            raise TypeError("current seam requires module._run_codex_process")
    else:
        seam_owner = module
        seam_attribute = "subprocess"
        original_seam = getattr(module, seam_attribute, None)
        original_run = getattr(original_seam, "run", None)
        if not callable(original_run):
            raise TypeError("legacy seam requires module-local subprocess.run")

    marker = object()
    with _INSTALL_LOCK:
        if module in _INSTALLED:
            raise RuntimeError("Codex attempt observer is already installed")
        _INSTALLED[module] = marker
    try:
        session = ObservationSession(
            ledger_path, seam=seam, context_by_cwd=context_by_cwd)
        complete_wrapper = session.complete_wrapper(original_complete)
        if seam == RUN_CODEX_PROCESS_SEAM:
            installed_seam = session.process_wrapper(original_seam)
        else:
            installed_seam = _SubprocessProxy(
                original_seam, session.subprocess_run_wrapper(original_run))
        complete_set = seam_set = False
        try:
            with _INSTALL_LOCK:
                if _INSTALLED.get(module) is not marker:
                    raise RuntimeError("Codex observer installation changed")
                setattr(backend, "complete", complete_wrapper)
                complete_set = True
                setattr(seam_owner, seam_attribute, installed_seam)
                seam_set = True
        except BaseException:
            if complete_set and getattr(
                    backend, "complete", None) is complete_wrapper:
                setattr(backend, "complete", original_complete)
            if seam_set and getattr(
                    seam_owner, seam_attribute, None) is installed_seam:
                setattr(seam_owner, seam_attribute, original_seam)
            raise
    except BaseException:
        with _INSTALL_LOCK:
            if _INSTALLED.get(module) is marker:
                del _INSTALLED[module]
        raise
    restore_error = None
    body_error = None
    try:
        yield session
    except BaseException as caught:
        body_error = caught
        raise
    finally:
        with _INSTALL_LOCK:
            complete_unchanged = getattr(
                backend, "complete", None) is complete_wrapper
            seam_unchanged = getattr(
                seam_owner, seam_attribute, None) is installed_seam
            if complete_unchanged:
                setattr(backend, "complete", original_complete)
            if seam_unchanged:
                setattr(seam_owner, seam_attribute, original_seam)
            if _INSTALLED.get(module) is marker:
                del _INSTALLED[module]
            if not complete_unchanged or not seam_unchanged:
                restore_error = RuntimeError(
                    "observed Codex module changed while shim was installed")
        session._close(restoration_failed=restore_error is not None)
        if restore_error is not None and body_error is None:
            raise restore_error


def _copy_contexts(
    contexts: Mapping[str | os.PathLike[str], Mapping[str, Any]],
) -> dict[Optional[str], Mapping[str, Any]]:
    if not isinstance(contexts, Mapping) or len(contexts) > _MAX_CONTEXTS:
        raise ValueError("context_by_cwd must be a bounded mapping")
    result: dict[Optional[str], Mapping[str, Any]] = {}
    for raw_cwd, raw_context in contexts.items():
        cwd = _canonical_path(raw_cwd)
        if cwd is None or cwd in result:
            raise ValueError("context_by_cwd paths must be unique filesystem paths")
        context = copy.deepcopy(raw_context)
        if (not isinstance(context, Mapping)
                or len(context) > _MAX_CONTEXT_FIELDS):
            raise ValueError("observer context must be a bounded mapping")
        clean: dict[str, Any] = {}
        for key, value in context.items():
            if (not isinstance(key, str) or not key
                    or len(key) > _MAX_CONTEXT_KEY_CHARS):
                raise ValueError("observer context keys must be short strings")
            if isinstance(value, str):
                if len(value) > _MAX_CONTEXT_STRING_CHARS:
                    raise ValueError("observer context strings are too long")
            elif isinstance(value, bool) or value is None:
                pass
            elif isinstance(value, int):
                if abs(value) > 10**18:
                    raise ValueError("observer context integer is too large")
            elif isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError("observer context float is not finite")
            else:
                raise ValueError("observer context values must be JSON scalars")
            clean[key] = value
        # Round-trip now so objects cannot change their representation later.
        json.dumps(clean, ensure_ascii=False, allow_nan=False)
        result[cwd] = clean
    return result


def _canonical_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return os.path.realpath(os.path.abspath(os.fspath(value)))
    except (TypeError, ValueError, OSError):
        return None


def _path_is_within(root: str, candidate: str) -> bool:
    try:
        return os.path.commonpath((root, candidate)) == root
    except ValueError:
        return False


def _subprocess_command(popenargs: Sequence[Any], kwargs: Mapping[str, Any]):
    if popenargs:
        return popenargs[0]
    return kwargs.get("args")


def _command_strings(command: Any) -> Optional[list[str]]:
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        return None
    values = []
    for value in command:
        try:
            item = os.fspath(value)
        except TypeError:
            return None
        if isinstance(item, bytes):
            try:
                item = item.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if not isinstance(item, str):
            return None
        values.append(item)
    return values


def _is_codex_exec(command: Any) -> bool:
    values = _command_strings(command)
    return bool(
        values and len(values) >= 2
        and os.path.basename(values[0]) == "codex"
        and values[1] == "exec"
    )


def _command_sha256(command: Any) -> Optional[str]:
    values = _command_strings(command)
    if values is None:
        return None
    digest = hashlib.sha256()
    digest.update(b"gv-argv-length-prefixed-utf8-v1\0")
    try:
        for value in values:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    except (UnicodeEncodeError, OverflowError):
        return None
    return digest.hexdigest()


def _unique_option(values: Sequence[str], option: str) -> Optional[str]:
    found = []
    for index, item in enumerate(values[:-1]):
        if item == option:
            found.append(values[index + 1])
    return found[0] if len(found) == 1 and found[0] else None


def _unique_config(values: Sequence[str], key: str) -> Optional[str]:
    found = []
    for index, item in enumerate(values[:-1]):
        if item != "-c":
            continue
        assignment = values[index + 1]
        if assignment.startswith(key + "="):
            found.append(assignment[len(key) + 1:])
    if len(found) != 1 or not found[0]:
        return None
    value = found[0]
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    return value or None


def _command_profile(
    command: Any,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    values = _command_strings(command)
    if values is None:
        return None, None, None
    try:
        boundary = values.index("--")
    except ValueError:
        return None, None, None
    options = values[:boundary]
    # The pinned backends use only these canonical split forms. Refuse to
    # guess precedence if another spelling or another config source could
    # override them. Tokens after ``--`` are prompt data, not CLI options.
    if any(item == "-m" or item.startswith("--model=")
           or item == "--config"
           or item.startswith("--config=") for item in options):
        return None, None, None
    if any(item == "-c" and index + 1 < len(options)
           and options[index + 1].startswith("model=")
           for index, item in enumerate(options)):
        return None, None, None
    model = _unique_option(options, "--model")
    effort = _unique_config(options, "model_reasoning_effort")
    raw_tier = _unique_config(options, "service_tier")
    enables = [options[index + 1]
               for index, item in enumerate(options[:-1])
               if item == "--enable"]
    disables = [options[index + 1]
                for index, item in enumerate(options[:-1])
                if item == "--disable"]
    tier = None
    if raw_tier == "fast" and enables.count("fast_mode") == 1 \
            and "fast_mode" not in disables:
        tier = "fast"
    elif raw_tier == "default" and disables.count("fast_mode") == 1 \
            and "fast_mode" not in enables:
        tier = "standard"
    return model, effort, tier


def _stdout_identity(
    stdout: Any,
) -> tuple[Optional[str], Optional[int], Optional[str]]:
    if stdout is None:
        raw = b""
        text = ""
    elif isinstance(stdout, bytes):
        raw = stdout
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return hashlib.sha256(raw).hexdigest(), len(raw), None
    elif isinstance(stdout, str):
        try:
            raw = stdout.encode("utf-8")
        except UnicodeEncodeError:
            return None, None, None
        text = stdout
    else:
        return None, None, None
    return hashlib.sha256(raw).hexdigest(), len(raw), text


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _reject_constant(_value):
    raise ValueError("non-finite JSON value")


class _ObjectPairs(list):
    """Distinguish a decoded JSON object from an ordinary JSON array."""


def _preserve_object_pairs(pairs):
    return _ObjectPairs(pairs)


def _usage_event_type(raw: str) -> Optional[str]:
    """Read one unambiguous top-level event type without policing payloads.

    Usage is carried only by ``thread.started`` and ``turn.completed``.  Codex
    tool-result payloads are unrelated to those counters and may themselves
    contain duplicate object fields.  Preserve their pairs so a nested
    duplicate cannot erase otherwise valid terminal accounting, while still
    rejecting an ambiguous top-level ``type`` that could hide another usage
    event.  The two accounting events are decoded strictly below.
    """
    event = json.loads(
        raw, object_pairs_hook=_preserve_object_pairs,
        parse_constant=_reject_constant,
    )
    if not isinstance(event, _ObjectPairs):
        raise ValueError("Codex JSONL event is not an object")
    event_types = [value for key, value in event if key == "type"]
    if len(event_types) > 1:
        raise ValueError("Codex JSONL event has duplicate type fields")
    if not event_types or not isinstance(event_types[0], str):
        return None
    return event_types[0]


def _parse_usage(text: Optional[str]) -> _Usage:
    empty = _incomplete_usage("unknown" if text == "" else "invalid")
    if text is None:
        return empty
    if not text.strip():
        return empty
    threads = []
    completions = []
    try:
        for raw in text.splitlines():
            if not raw.strip():
                continue
            event_type = _usage_event_type(raw)
            if event_type not in {"thread.started", "turn.completed"}:
                continue
            event = json.loads(
                raw, object_pairs_hook=_unique_json_object,
                parse_constant=_reject_constant,
            )
            if event_type == "thread.started":
                thread_id = event.get("thread_id")
                if not isinstance(thread_id, str) or not thread_id:
                    return empty
                threads.append(thread_id)
            else:
                completions.append(event.get("usage"))
    except (ValueError, TypeError, RecursionError):
        return empty
    if len(threads) != 1 or len(completions) != 1:
        return _Usage(
            status="partial", complete=False,
            thread_id=threads[0] if len(threads) == 1 else None,
            input_tokens=None, cached_input_tokens=None,
            uncached_input_tokens=None, cache_write_input_tokens=None,
            output_tokens=None, reasoning_tokens=None,
        )
    raw_usage = completions[0]
    if not isinstance(raw_usage, dict):
        return empty
    counts: dict[str, int] = {}
    for field in _USAGE_FIELDS:
        value = raw_usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return empty
        counts[field] = value
    if counts["cached_input_tokens"] > counts["input_tokens"]:
        return empty
    cache_write = raw_usage.get("cache_write_input_tokens")
    if cache_write is not None and (
            isinstance(cache_write, bool)
            or not isinstance(cache_write, int) or cache_write < 0):
        return empty
    return _Usage(
        status="complete", complete=True, thread_id=threads[0],
        input_tokens=counts["input_tokens"],
        cached_input_tokens=counts["cached_input_tokens"],
        uncached_input_tokens=(
            counts["input_tokens"] - counts["cached_input_tokens"]),
        cache_write_input_tokens=cache_write,
        output_tokens=counts["output_tokens"],
        reasoning_tokens=counts["reasoning_output_tokens"],
    )


def _incomplete_usage(status: str) -> _Usage:
    return _Usage(
        status=status, complete=False, thread_id=None,
        input_tokens=None, cached_input_tokens=None,
        uncached_input_tokens=None, cache_write_input_tokens=None,
        output_tokens=None, reasoning_tokens=None,
    )


def _result_stdout(result: Any, error: Optional[BaseException]) -> Any:
    if result is not None:
        return getattr(result, "stdout", None)
    if isinstance(error, subprocess.TimeoutExpired):
        return (getattr(error, "stdout", None)
                or getattr(error, "output", None))
    return None


def _returncode(result: Any) -> Optional[int]:
    value = getattr(result, "returncode", None) if result is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) \
        else None


def _process_terminal(
    result: Any, error: Optional[BaseException],
) -> tuple[str, Optional[bool]]:
    if error is None:
        return "returned", True
    if isinstance(error, subprocess.TimeoutExpired):
        return "timeout", bool(getattr(error, "_gv_process_started", True))
    if isinstance(error, OSError):
        started = bool(getattr(error, "_gv_process_started", False))
        return ("interrupted" if started else "launch_error"), started
    marker = getattr(error, "_gv_process_started", None)
    return "interrupted", marker if isinstance(marker, bool) else None


def _error_kind(error: BaseException) -> str:
    return type(error).__name__[:_MAX_ERROR_KIND_CHARS]
