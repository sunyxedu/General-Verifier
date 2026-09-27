"""One logical ``codex exec`` call: its CLI attempts and its answer.

Attempts repeat only while the provider fails transiently (``failures``);
every attempt that ends leaves exactly one ``codex_cli_attempt`` row in an
active usage ledger (``usage``), and the answer is accepted only after the
seat's stream policy, the accounting/response agreement and the role's
schema have all passed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Dict, Optional

from ..retry import Backoff, RetryRule, retry_transient
from ..schema import RoleError, SeatResourceKilled, SeatTimeout, validate
from . import failures
from .failures import (
    TransientBackendError,
    _codex_own_stderr,
    _transient_failure_reason,
)
from .stream import _parse_exec_telemetry, _stream_text, extract_json
from .usage import (
    _AttemptUsage,
    _attempt_usage,
    _record_codex_attempt,
    _usage_error_code,
)


class _ProviderTransient(Exception):
    """One attempt the provider failed transiently: the request never got
    a completed turn, so it is retried (internal to the retry loop)."""

    def __init__(self, reason: str, attempt: int, row: Dict[str, Any]):
        super().__init__(reason)
        self.reason = reason
        self.attempt = attempt
        self.row = row


class CodexInvocation:
    """The attempts of one seat call's ``cmd`` launched in ``cwd``.

    ``validate_stream``, ``bound_response`` and ``recover_response`` are
    the seat's stream policy (the runtime's hooks); ``usage_target`` is the
    active usage ledger and the call's scoring context, or None."""

    def __init__(self, *, role_name: str, cmd: list, cwd: Any,
                 instruction: str, via_stdin: bool, timeout: int,
                 watch: Dict[str, Any], output_path: str, usage_target,
                 attempt_base: Dict[str, Any],
                 validate_stream: Callable[[str], None],
                 bound_response: Callable[[str], Optional[str]],
                 recover_response: Callable[[str], Optional[str]]):
        self.role_name = role_name
        self.cmd = cmd
        self.cwd = cwd
        self.instruction = instruction
        self.via_stdin = via_stdin
        self.timeout = timeout
        self.watch = watch
        self.output_path = output_path
        self.usage_target = usage_target
        self.attempt_base = attempt_base
        self._validate_stream = validate_stream
        self._bound_response = bound_response
        self._recover_response = recover_response

    def _record(self, **fields) -> _AttemptUsage:
        return _record_codex_attempt(
            self.usage_target, base=self.attempt_base, **fields)

    def run(self, launch: Callable[..., Any],
            sleep: Callable[[float], Any]) -> Dict[str, Any]:
        """The completed attempt, provider capacity flaps waited out
        through ``failures.TRANSIENT_BACKOFF_SECONDS``."""
        retries = failures.TRANSIENT_RETRIES
        attempts = iter(range(retries + 1))

        def on_failure(error: _ProviderTransient, wait: Optional[float]):
            self._record(
                attempt_status=("provider_exhausted" if wait is None
                                else "provider_retry"),
                error_kind="provider_transient",
                error="Codex provider reported a transient failure",
                **error.row,
            )
            if wait is not None:
                print(f"[codex-backend] transient failure ({error.reason}); "
                      f"retry {error.attempt + 1}/{retries} "
                      f"in {wait}s", file=sys.stderr, flush=True)

        def exhausted(error: _ProviderTransient) -> BaseException:
            return TransientBackendError(
                f"{self.role_name}: transient codex failure persisted "
                f"through {retries} retries: {error.reason}")

        return retry_transient(
            lambda: self._attempt(next(attempts), launch),
            (RetryRule((_ProviderTransient,),
                       Backoff(failures.TRANSIENT_BACKOFF_SECONDS,
                               retries=retries),
                       exhausted=exhausted, on_failure=on_failure),),
            sleep=sleep)

    def _attempt(self, attempt: int, launch) -> Dict[str, Any]:
        # Each attempt must produce its own final message. A failed
        # process can leave a valid-looking JSON file behind.
        with open(self.output_path, "w", encoding="utf-8"):
            pass
        attempt_id = uuid.uuid4().hex
        started_at_unix_ns = time.time_ns()
        started_at_monotonic_ns = time.monotonic_ns()
        try:
            with open(os.devnull) as devnull:
                # stdin MUST be closed: codex otherwise waits forever.
                proc = launch(
                    self.cmd, cwd=self.cwd, stdin=devnull,
                    timeout=self.timeout,
                    **({"input_text": self.instruction} if self.via_stdin
                       else {}),
                    **self.watch)
        except subprocess.TimeoutExpired as error:
            finished_at_unix_ns = time.time_ns()
            duration_ns = (time.monotonic_ns()
                           - started_at_monotonic_ns)
            partial = (getattr(error, "stdout", None)
                       or getattr(error, "output", None) or "")
            self._record(
                attempt_id=attempt_id, attempt_index=attempt + 1,
                started=bool(getattr(
                    error, "_gv_process_started", True)),
                attempt_status="timeout", returncode=None,
                started_at_unix_ns=started_at_unix_ns,
                finished_at_unix_ns=finished_at_unix_ns,
                duration_ns=duration_ns, stdout=partial,
                force_usage_incomplete=True,
                usage_incomplete_reason=(
                    "process timed out; partial CLI output cannot "
                    "establish complete usage"),
                error_kind="timeout",
                error="Codex CLI exceeded the configured timeout",
            )
            raise SeatTimeout(
                f"{self.role_name}: codex exec timed out after "
                f"{self.timeout}s") from error
        except OSError as error:
            finished_at_unix_ns = time.time_ns()
            duration_ns = (time.monotonic_ns()
                           - started_at_monotonic_ns)
            started = bool(getattr(
                error, "_gv_process_started", False))
            self._record(
                attempt_id=attempt_id, attempt_index=attempt + 1,
                started=started,
                attempt_status=("interrupted" if started
                                else "launch_error"),
                returncode=None,
                started_at_unix_ns=started_at_unix_ns,
                finished_at_unix_ns=finished_at_unix_ns,
                duration_ns=duration_ns,
                force_usage_incomplete=True,
                usage_incomplete_reason=(
                    "process did not produce trusted terminal usage"),
                error_kind=("process_cleanup" if started
                            else "process_launch"),
                error=("Codex CLI process cleanup failed" if started
                       else "Codex CLI process could not be started"),
            )
            raise
        except BaseException as error:
            finished_at_unix_ns = time.time_ns()
            duration_ns = (time.monotonic_ns()
                           - started_at_monotonic_ns)
            self._record(
                attempt_id=attempt_id, attempt_index=attempt + 1,
                started=bool(getattr(
                    error, "_gv_process_started", False)),
                attempt_status="interrupted", returncode=None,
                started_at_unix_ns=started_at_unix_ns,
                finished_at_unix_ns=finished_at_unix_ns,
                duration_ns=duration_ns,
                force_usage_incomplete=True,
                usage_incomplete_reason=(
                    "process was interrupted before trusted terminal "
                    "usage was recorded"),
                error_kind=type(error).__name__,
                error="Codex CLI attempt was interrupted",
            )
            raise
        finished_at_unix_ns = time.time_ns()
        duration_ns = (time.monotonic_ns()
                       - started_at_monotonic_ns)
        stdout = _stream_text(proc.stdout)
        attempt_usage = _attempt_usage(stdout)
        reason = _transient_failure_reason(
            stdout, proc.returncode, _stream_text(proc.stderr))
        if reason:
            raise _ProviderTransient(reason, attempt, dict(
                attempt_id=attempt_id, attempt_index=attempt + 1,
                started=True, returncode=proc.returncode,
                started_at_unix_ns=started_at_unix_ns,
                finished_at_unix_ns=finished_at_unix_ns,
                duration_ns=duration_ns, stdout=stdout,
            ))
        if proc.returncode < 0:
            # Killed by a signal we did not send (our timeout and
            # scratch budget raise before this): in a 2 GB envelope,
            # the kernel's OOM killer. Nothing the model did wrong,
            # so no "fix it" feedback; the role's retry launches
            # again through the room queue.
            self._record(
                attempt_id=attempt_id, attempt_index=attempt + 1,
                started=True, attempt_status="nonzero_exit",
                returncode=proc.returncode,
                started_at_unix_ns=started_at_unix_ns,
                finished_at_unix_ns=finished_at_unix_ns,
                duration_ns=duration_ns, stdout=stdout,
                error_kind="signal_killed",
                error="Codex CLI was killed by a signal",
            )
            raise SeatResourceKilled(
                f"{self.role_name}: codex exec was killed by signal "
                f"{-proc.returncode} (not a timeout; most likely "
                f"out of memory)")
        if proc.returncode != 0:
            self._record(
                attempt_id=attempt_id, attempt_index=attempt + 1,
                started=True, attempt_status="nonzero_exit",
                returncode=proc.returncode,
                started_at_unix_ns=started_at_unix_ns,
                finished_at_unix_ns=finished_at_unix_ns,
                duration_ns=duration_ns, stdout=stdout,
                error_kind="nonzero_exit",
                error="Codex CLI exited with a nonzero status",
            )
            raise RoleError(
                f"{self.role_name}: codex exec failed "
                f"(rc={proc.returncode}, stderr="
                f"{_codex_own_stderr(_stream_text(proc.stderr))[-300:]!r})")
        try:
            self._validate_stream(stdout)
            validated_stream_response = self._bound_response(stdout)
            if (validated_stream_response is not None
                    and (not isinstance(validated_stream_response, str)
                         or not validated_stream_response.strip())):
                raise RoleError(
                    "Codex stream policy returned an invalid response")
            validated_stream_response = (
                validated_stream_response.strip()
                if isinstance(validated_stream_response, str) else None)
            response_without_usage = None
            # Billing extraction deliberately ignores unrelated item
            # payloads. Before accepting a role response, retain the
            # pre-existing full-stream parse and exact final-message
            # recovery semantics.
            if attempt_usage.telemetry is not None:
                response_telemetry = _parse_exec_telemetry(stdout)
                if (response_telemetry.thread_id
                        != attempt_usage.telemetry.thread_id
                        or response_telemetry.usage
                        != attempt_usage.telemetry.usage):
                    raise RoleError(
                        "Codex accounting and response streams disagree")
                attempt_usage = _AttemptUsage(
                    telemetry=response_telemetry,
                    thread_id=response_telemetry.thread_id,
                    status="complete",
                    error=None,
                )
            usage_code = _usage_error_code(attempt_usage.error) or ""
            if (self.usage_target is not None and attempt_usage.telemetry is None
                    and (usage_code.startswith("invalid_counter:")
                         or usage_code == "cached_exceeds_input")):
                recovered = self._recover_response(stdout)
                if validated_stream_response is not None \
                        and recovered != validated_stream_response:
                    raise RoleError(
                        "Codex stream validation and recovery disagree")
                response_without_usage = recovered
        except RoleError:
            self._record(
                attempt_id=attempt_id, attempt_index=attempt + 1,
                started=True, attempt_status="output_error",
                returncode=proc.returncode,
                started_at_unix_ns=started_at_unix_ns,
                finished_at_unix_ns=finished_at_unix_ns,
                duration_ns=duration_ns, stdout=stdout,
                error_kind="attempt_stream_policy",
                error="Codex CLI event stream violated role policy",
            )
            raise
        except BaseException as error:
            self._record(
                attempt_id=attempt_id, attempt_index=attempt + 1,
                started=True, attempt_status="interrupted",
                returncode=proc.returncode,
                started_at_unix_ns=started_at_unix_ns,
                finished_at_unix_ns=finished_at_unix_ns,
                duration_ns=duration_ns, stdout=stdout,
                error_kind=type(error).__name__,
                error="Codex CLI event-stream validation interrupted",
            )
            raise
        if self.usage_target is not None and attempt_usage.telemetry is None \
                and response_without_usage is None:
            self._record(
                attempt_id=attempt_id, attempt_index=attempt + 1,
                started=True, attempt_status="telemetry_error",
                returncode=proc.returncode,
                started_at_unix_ns=started_at_unix_ns,
                finished_at_unix_ns=finished_at_unix_ns,
                duration_ns=duration_ns, stdout=stdout,
                error_kind="usage_protocol",
                error="Codex CLI usage telemetry was incomplete",
            )
            raise RoleError(attempt_usage.error or
                            "codex exec usage telemetry incomplete")
        return {
            "attempt_id": attempt_id,
            "attempt_index": attempt + 1,
            "returncode": proc.returncode,
            "started_at_unix_ns": started_at_unix_ns,
            "finished_at_unix_ns": finished_at_unix_ns,
            "duration_ns": duration_ns,
            "stdout": stdout,
            "usage": attempt_usage,
            "validated_stream_response": validated_stream_response,
            "response_without_usage": response_without_usage,
            "proc": proc,
        }

    def payload(self, completed: Dict[str, Any], role: Any) -> Any:
        """The role's payload from the completed attempt's final output,
        with the call's terminal ledger row."""
        proc = completed["proc"]
        # A no-tools session requires JSON even without a
        # usage ledger. Preserve its parsed final agent message as the
        # safe fallback when the CLI's -o file is unexpectedly empty.
        telemetry = completed["usage"].telemetry
        attempt_fields = {
            key: completed[key] for key in (
                "attempt_id", "attempt_index", "returncode",
                "started_at_unix_ns", "finished_at_unix_ns",
                "duration_ns", "stdout")
        }
        terminal_recorded = False
        try:
            text = ""
            try:
                if os.path.exists(self.output_path):
                    with open(self.output_path, encoding="utf-8",
                              errors="strict") as f:
                        text = f.read().strip()
            except (OSError, UnicodeError) as error:
                self._record(
                    started=True, attempt_status="output_error",
                    error_kind=("output_read" if isinstance(error, OSError)
                                else "output_encoding"),
                    error=("Codex CLI output file could not be read"
                           if isinstance(error, OSError) else
                           "Codex CLI output was not strict UTF-8"),
                    **attempt_fields,
                )
                terminal_recorded = True
                if isinstance(error, UnicodeError):
                    raise RoleError(
                        "codex final output is not valid UTF-8") from error
                raise
            validated_stream_response = completed[
                "validated_stream_response"]
            if validated_stream_response is not None:
                if not text or text != validated_stream_response:
                    self._record(
                        started=True, attempt_status="output_error",
                        error_kind="output_stream_mismatch",
                        error=("Final output file was absent or disagreed "
                               "with the policy-validated final message"),
                        **attempt_fields,
                    )
                    terminal_recorded = True
                    raise RoleError(
                        "codex final output is missing or disagrees with "
                        "its completed stream")
                text = validated_stream_response
            response_without_usage = completed["response_without_usage"]
            if response_without_usage is not None:
                if not text or text != response_without_usage:
                    self._record(
                        started=True, attempt_status="output_error",
                        error_kind="output_stream_mismatch",
                        error="Final output file was absent or disagreed with completed agent message",
                        **attempt_fields,
                    )
                    terminal_recorded = True
                    raise RoleError("codex final output is missing or disagrees with its completed stream")
                text = response_without_usage
            if not text:
                text = (telemetry.last_message if telemetry is not None
                        else completed["stdout"].strip())
            if not text:
                self._record(
                    started=True, attempt_status="output_error",
                    error_kind="empty_output",
                    error="Codex CLI produced no final role output",
                    **attempt_fields,
                )
                terminal_recorded = True
                raise RoleError(
                    f"{self.role_name}: codex exec produced no output "
                    f"(rc={proc.returncode}, "
                    f"stderr={proc.stderr[-300:]!r})")
            payload = extract_json(text, accept=lambda candidate: validate(
                getattr(role, "wire_schema", None) or role.schema,
                candidate))
            validate(
                getattr(role, "wire_schema", None) or role.schema,
                payload,
            )
        except RoleError:
            if terminal_recorded:
                raise
            self._record(
                started=True, attempt_status="schema_error",
                error_kind="role_schema",
                error="Final role output did not match its JSON schema",
                **attempt_fields,
            )
            terminal_recorded = True
            raise
        except BaseException as error:
            if not terminal_recorded:
                self._record(
                    started=True, attempt_status="interrupted",
                    error_kind=type(error).__name__,
                    error="Role output processing was interrupted",
                    **attempt_fields,
                )
            raise
        self._record(
            started=True, attempt_status="succeeded",
            **attempt_fields,
        )
        return payload
