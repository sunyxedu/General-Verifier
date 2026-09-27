"""How one seat call runs: a chain of stages from the role to the runtime.

``AgentRole.call(runtime, variables)`` runs ``SEAT_RUNNER``. Each concern of
a seat call is one stage, outermost first:

  FeedbackRetry            the role's span; two attempts, the first one's
                           RoleError fed back into the second's prompt
  OutputValidation         the model-facing wire schema lowered, then the
                           role's schema checked
  SourceEvidencePostcheck  after the seat is released, the source-evidence
                           view the call read is unchanged
  SeatGate                 one process-wide in-flight permit per call
  LateBinding              variables bound after queueing: the
                           source-evidence view and the lazy Oracle catalog
  dispatch                 the prompt rendered and the runtime called

Below ``dispatch`` the chain continues inside the runtime boundary as
runtime decorators (``runtime.RuntimeDecorator``): ``RoutingRuntime`` here
seats each role on the runtime it needs, ``TypedJudgerSeat`` admits a typed
Judger packet in front of a restricted session. A runtime itself waits out
its transient transport failures (``retry``) and writes one receipt per call
(``receipts``).
"""
from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Set

from ..config import max_concurrent_seats
from ..engine.source_evidence import (
    SourceEvidenceView,
    refresh_source_evidence_variables,
)
from .model_spans import (
    role_attempt, role_span, seat_acquired, seat_released, seat_wait_started,
)
from .runtime import RuntimeDecorator, capabilities_of
from .schema import RoleError, SeatResourceKilled, SeatTimeout, validate


def _seat_limit() -> int:
    """Process-wide cap on in-flight seat calls, whatever the nesting of
    trial x claim x obligation parallelism above it. Env
    GV_MAX_CONCURRENT_SEATS (default 12); <= 0 disables the cap."""
    return max_concurrent_seats()


class _SeatGate:
    """A bounded semaphore sized lazily from the env, shared by every role
    in the process; ``slot()`` yields one in-flight permit. Every seat —
    a direct HTTP call or a launched process — holds one, so the cap bounds
    them all together; a seat whose process waits for the envelope's
    memory gives its permit back while it waits (resource_share
    hold_permit_while_waiting), so seats that need no memory of their own
    never queue behind seats that are waiting for memory."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sem: Optional[threading.BoundedSemaphore] = None
        self._size = 0

    def _current(self) -> Optional[threading.BoundedSemaphore]:
        limit = _seat_limit()
        if limit <= 0:
            return None
        with self._lock:
            if self._sem is None or self._size != limit:
                self._sem = threading.BoundedSemaphore(limit)
                self._size = limit
            return self._sem

    @contextlib.contextmanager
    def slot(self):
        sem = self._current()
        if sem is None:
            yield
            return
        from ..execution.resource_share import hold_permit_while_waiting
        sem.acquire()
        try:
            with hold_permit_while_waiting(sem.release, sem.acquire):
                yield
        finally:
            sem.release()


SEAT_GATE = _SeatGate()


@dataclass
class SeatCall:
    """One logical seat call: the role, the runtime it runs on, the
    caller's variables, and what one attempt leaves for the next."""
    role: Any
    runtime: Any
    variables: Dict[str, Any]
    # the previous attempt's error, as fed back into the next prompt
    feedback: str = ""
    last_error: Optional[RoleError] = None
    # a lazy Oracle catalog that failed once is inlined from then on
    oracle_catalog_force_baseline: bool = False


@dataclass
class SeatAttempt:
    """One attempt of a seat call and the variables bound for it (the
    caller's own until a binding stage replaces them)."""
    call: SeatCall
    index: int
    variables: Dict[str, Any] = field(default_factory=dict)
    source_view: Any = None


class FeedbackRetry:
    """Two attempts (schema error fed back once), then RoleError. A spent
    budget is not retried and a killed process is retried without
    feedback."""

    def run(self, call: SeatCall,
            attempt: Callable[[SeatAttempt], Any]) -> Any:
        name = call.role.name
        # Start before acquiring SEAT_GATE: queueing is part of end-to-end
        # role latency, while each schema retry remains separately visible.
        with role_span(name):
            for attempt_index in range(1, 3):
                try:
                    with role_attempt(attempt_index):
                        payload = attempt(SeatAttempt(
                            call, attempt_index, variables=call.variables))
                    return payload
                except SeatTimeout:
                    # a spent time budget is not something the retry's
                    # feedback can fix: rerunning the same prompt spends it
                    # again (a 30-minute executor timeout became an hour)
                    raise
                except SeatResourceKilled as err:
                    # retried as it was asked: "fix it" would send the model
                    # looking for a fault in an answer it never gave
                    call.last_error = err
                except RoleError as err:
                    call.last_error = err
                    call.feedback = (
                        f"\n\n[previous attempt failed; fix it and retry] {err}"
                    )
            raise RoleError(f"{name}: failed twice: {call.last_error}")


class OutputValidation:
    """The payload in the role's contract: a runtime that answers in the
    role's wire schema has it checked and lowered first."""

    def run(self, attempt: SeatAttempt, proceed) -> Any:
        payload = proceed(attempt)
        role, runtime = attempt.call.role, attempt.call.runtime
        use_wire = (
            role.wire_schema is not None
            and capabilities_of(runtime).uses_role_wire_schema is True
        )
        if use_wire:
            validate(role.wire_schema, payload)
            if role.wire_lower is None:
                raise RoleError(
                    f"{role.name}: wire schema has no lowerer",
                )
            payload = role.wire_lower(payload)
        validate(role.schema, payload)
        return payload


class SourceEvidencePostcheck:
    """The source-evidence view the call read is still the published one."""

    def run(self, attempt: SeatAttempt, proceed) -> Any:
        payload = proceed(attempt)
        source_view = attempt.source_view
        if (type(source_view) is SourceEvidenceView
                and not source_view.revalidate()):
            raise RoleError(
                f"{attempt.call.role.name}: source evidence changed during audit")
        return payload


class SeatGate:
    """The call holds one permit of ``gate`` while it runs."""

    def __init__(self, gate: _SeatGate):
        self.gate = gate

    def run(self, attempt: SeatAttempt, proceed) -> Any:
        wait_started = seat_wait_started()
        with self.gate.slot():
            acquired = seat_acquired(wait_started)
            try:
                return proceed(attempt)
            finally:
                seat_released(acquired)


class LateBinding:
    """Bind after queueing: a view captured before the seat gate could go
    stale while waiting and waste a complete paid backend call."""

    def run(self, attempt: SeatAttempt, proceed) -> Any:
        call = attempt.call
        runtime = capabilities_of(call.runtime)
        attempt_variables = self._bind_source_evidence(call, runtime)
        catalog_snapshot = None
        if call.role.name == "oracles":
            attempt_variables, catalog_snapshot = self._bind_oracle_catalog(
                call, runtime, attempt_variables)
        source_view = attempt_variables.get("_source_evidence_view")
        if (type(source_view) is SourceEvidenceView
                and not source_view.revalidate()):
            # The optimization is optional. A race in
            # publication must expose the original
            # full ledger, not consume a model retry.
            attempt_variables = dict(attempt_variables)
            attempt_variables["ledger_path"] = \
                attempt_variables["_mixed_ledger_path"]
            attempt_variables.pop("_source_evidence_view", None)
            source_view = None
        attempt.variables = attempt_variables
        attempt.source_view = source_view
        catalog_changed = False
        try:
            payload = proceed(attempt)
        finally:
            if catalog_snapshot is not None:
                catalog_changed = not catalog_snapshot.revalidate()
                catalog_snapshot.close()
                if catalog_changed:
                    call.oracle_catalog_force_baseline = True
        if catalog_changed:
            raise RoleError(
                "oracles: adapter catalog changed "
                "during audit")
        return payload

    @staticmethod
    def _bind_source_evidence(call: SeatCall, runtime) -> Dict[str, Any]:
        variables = call.variables
        attempt_variables = dict(variables)
        source_view = attempt_variables.get("_source_evidence_view")
        mixed_ledger_path = attempt_variables.get("_mixed_ledger_path")
        compact_source_allowed = None
        if type(source_view) is SourceEvidenceView:
            try:
                compact_source_allowed = \
                    runtime.allows_compact_source_evidence(
                        call.role.name,
                        attempt_variables,
                    )
            except Exception:
                # An absent or faulty optional
                # capability cannot change the
                # established evidence transport.
                compact_source_allowed = None
        valid_mixed_path = False
        if (type(source_view) is SourceEvidenceView
                and isinstance(mixed_ledger_path, str)
                and os.path.isabs(mixed_ledger_path)):
            try:
                valid_mixed_path = (
                    mixed_ledger_path
                    == os.path.abspath(source_view._log.path))
            except (AttributeError, OSError, TypeError, ValueError):
                valid_mixed_path = False
        if compact_source_allowed is False and valid_mixed_path:
            # Some runtimes mutate filesystem metadata
            # merely by reading a directory. Respect
            # an explicit opt-out before paying for a
            # call whose immutable-view postcheck could
            # therefore never succeed.
            attempt_variables["ledger_path"] = mixed_ledger_path
            attempt_variables.pop("_source_evidence_view", None)
            return attempt_variables
        return refresh_source_evidence_variables(variables)

    @staticmethod
    def _bind_oracle_catalog(call: SeatCall, runtime,
                             attempt_variables: Dict[str, Any]):
        from ..execution.adapters import OracleCatalogView
        catalog_snapshot = None
        catalog_view = attempt_variables.get("adapter_catalog")
        if type(catalog_view) is OracleCatalogView:
            try:
                permits_lazy_catalog = runtime.allows_lazy_oracle_catalog(
                    call.role.name,
                    attempt_variables,
                ) is True
            except Exception:
                permits_lazy_catalog = False
            if (call.oracle_catalog_force_baseline
                    or not permits_lazy_catalog):
                attempt_variables = dict(attempt_variables)
                attempt_variables["adapter_catalog"] = \
                    catalog_view.fallback_catalog()
            else:
                catalog_snapshot = catalog_view.snapshot()
                if catalog_snapshot is None:
                    call.oracle_catalog_force_baseline = True
                    attempt_variables = dict(attempt_variables)
                    attempt_variables["adapter_catalog"] = \
                        catalog_view.fallback_catalog()
        return attempt_variables, catalog_snapshot


def dispatch(attempt: SeatAttempt) -> Any:
    """The chain's end: the rendered prompt, the previous attempt's error
    appended, handed to the runtime."""
    call = attempt.call
    prompt = call.role.render(attempt.variables) + call.feedback
    return call.runtime.complete(call.role.name, prompt, attempt.variables)


class SeatRunner:
    """Runs a seat call through ``retry`` and then, per attempt, through
    ``stages`` (outermost first) to ``dispatch``."""

    def __init__(self, stages: Sequence[Any],
                 retry: Optional[FeedbackRetry] = None,
                 terminal: Callable[[SeatAttempt], Any] = dispatch):
        self.stages = tuple(stages)
        self.retry = retry or FeedbackRetry()
        self.terminal = terminal

    def call(self, role: Any, runtime: Any, variables: Dict[str, Any]) -> Any:
        return self.retry.run(SeatCall(role, runtime, variables),
                              self._attempt)

    def _attempt(self, attempt: SeatAttempt, index: int = 0) -> Any:
        if index == len(self.stages):
            return self.terminal(attempt)
        return self.stages[index].run(
            attempt, lambda bound: self._attempt(bound, index + 1))


SEAT_RUNNER = SeatRunner((
    OutputValidation(),
    SourceEvidencePostcheck(),
    SeatGate(SEAT_GATE),
    LateBinding(),
))


class _SeatOfRecord:
    """An attribute read through to the routing runtime's agent side (the
    seat of record); assigning it on the routing runtime shadows it."""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return getattr(instance.agent, self.name)


class RoutingRuntime(RuntimeDecorator):
    """Seats each role on the runtime it needs: tool-free roles on the
    direct model call, tool-using roles on the agent runtime. The role
    registry and every declared capability are the agent runtime's, which
    remains the seat of record for anything with a palette; so is the
    other seat state a caller reads (model, effort, cwd, identities)."""

    RUNTIME_IDENTITY = "verifier.llm.direct_backend.RoutingBackend"

    model = _SeatOfRecord()
    model_overrides = _SeatOfRecord()
    effort = _SeatOfRecord()
    service_tier = _SeatOfRecord()
    permission_mode = _SeatOfRecord()
    cwd = _SeatOfRecord()
    transcripts = _SeatOfRecord()

    def __init__(self, direct, agent, direct_roles: Iterable[str]):
        self.direct = direct
        self.agent = agent
        self.direct_roles: Set[str] = set(direct_roles)
        self.uses_role_wire_schema = True

    @property
    def seat_of_record(self):
        return self.agent

    @property
    def roles(self) -> Dict[str, Any]:
        return self.agent.roles

    @roles.setter
    def roles(self, value: Dict[str, Any]) -> None:
        self.agent.roles = value
        self.direct.roles = dict(value)

    def routes_direct(self, role_name: str) -> bool:
        return role_name in self.direct_roles

    def seat_for(self, role_name: str):
        # A direct seat runs on the direct provider only — never
        # silently on the agent runtime, which is a different model
        # call with a different cost and latency profile. Outages and
        # exhausted credit are waited out inside the direct call.
        if self.routes_direct(role_name):
            return self.direct
        return self.agent

    def seat_runtime(self, role_names: Iterable[str]):
        names = list(role_names)
        if all(self.routes_direct(name) for name in names):
            return self.direct
        return self.agent

    def allows_lazy_oracle_catalog(self, role_name: str,
                                   variables: Dict[str, Any]) -> bool:
        if self.routes_direct(role_name):
            return False
        return bool(capabilities_of(self.agent).allows_lazy_oracle_catalog(
            role_name, variables))

    def captured_images(self, role_name: str, variables: Dict[str, Any]):
        return capabilities_of(self.agent).captured_images(role_name, variables)

    def reuse_identity(self) -> Dict[str, Any]:
        return capabilities_of(self.agent).reuse_identity()
