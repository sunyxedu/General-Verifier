"""Agent runtimes and the capabilities they declare.

An agent runtime runs a seat: ``complete(role_name, prompt, variables)``
returns the role's JSON payload, with ``roles`` the registry it resolves
role names in. The shipped runtimes are the Claude Code runtime (the Agent
SDK), the Codex runtime (the Codex CLI), the direct runtime (one model
request per seat), the routing runtime that seats each role on one of two
others, and the fixture runtime tests replay answers from.

What a caller may ask a runtime beyond ``complete`` is declared here, with
the answer a runtime that does not declare it gives. The engine and the
seats ask through these members only; ``capabilities_of`` reads a duck-typed
backend (a test double, an experiment's wrapper) with the same defaults.

A ``RuntimeDecorator`` stands in front of the runtime that is its seat of
record: the stages of a seat call below the role (dispatch of each seat to
the runtime it needs, admission of a restricted call) are decorators.

The boundary members (``supports_policy``, ``no_tools_runtime``,
``no_tools_identity``, ``judger_scratch``, ``replays_fixtures``) are not
inherited: a subclass may change any hook a boundary relies on, so it must
declare the boundary itself.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, Iterable, Optional, Protocol, Tuple

from ..config import RunOptions
from .policy import NoToolsSession, ToolPolicy


class AgentRuntime(Protocol):
    """What runs a seat."""
    roles: Dict[str, Any]

    def complete(self, role_name: str, prompt: str,
                 variables: Dict[str, Any]) -> Any: ...


class BoundScratch(Protocol):
    """An operator-bound empty directory a restricted session runs in."""
    path: str

    def execution_dir(self) -> str:
        """The directory to launch in, re-validated first."""

    def revalidate(self) -> str:
        """Raise RoleError unless the directory is still the bound, empty
        one; return its path."""


_BOUNDARY_MEMBERS = ("supports_policy", "no_tools_runtime",
                     "no_tools_identity", "judger_scratch", "replays_fixtures")

# The seat settings a runtime's reuse keys are made of, in key order.
REUSE_IDENTITY_FIELDS = ("model", "model_overrides", "effort", "service_tier",
                         "permission_mode", "runtime_instruction_identity")


def _present_identity_fields(backend: Any) -> Dict[str, Any]:
    """The reuse-identity fields ``backend`` happens to have: the identity
    of a backend that declares none (a fixture a test configured, a
    duck-typed wrapper)."""
    return {name: getattr(backend, name) for name in REUSE_IDENTITY_FIELDS
            if hasattr(backend, name)}


class RuntimeCapabilities:
    """The declared members of an agent runtime, with their defaults."""

    # The runtime's name in reuse keys (method cache, contract cache),
    # fixed across renames and moves; read from the class itself only.
    RUNTIME_IDENTITY: ClassVar[Optional[str]] = None

    # the role's model-facing wire schema is what this runtime returns
    uses_role_wire_schema: bool = False
    # a runtime that replays fixed answers: a deterministic seam with no
    # tools to use, admitted to the authority-only boundary as it is
    replays_fixtures: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in _BOUNDARY_MEMBERS:
            if name not in vars(cls):
                setattr(cls, name, vars(RuntimeCapabilities)[name])

    @property
    def run_options(self) -> RunOptions:
        """The run options this runtime renders instructions for."""
        return RunOptions()

    def allows_lazy_oracle_catalog(self, role_name: str,
                                   variables: Dict[str, Any]) -> bool:
        """The Oracle may read catalog entries through locators instead of
        receiving the whole catalog inlined."""
        return False

    def allows_compact_source_evidence(
            self, role_name: str, variables: Dict[str, Any]) -> Optional[bool]:
        """True/False: the source-evidence view is (not) safe to hand this
        runtime; None: no opinion (the view is refreshed and used)."""
        return None

    def routes_direct(self, role_name: str) -> bool:
        """The role runs as a single direct model request."""
        return False

    def captured_images(self, role_name: str,
                        variables: Dict[str, Any]) -> Tuple[Any, ...]:
        """The raw image inputs this runtime attaches to the call."""
        return ()

    def validate_offline_tool_scope(self, state_dir: str,
                                    toolchain_dir: Optional[str],
                                    shared_cache_dir: Optional[str]) -> None:
        """Raise unless the offline service and this runtime share one
        operator-owned tool scope."""

    def check_service_tier(self, service_tier: str) -> None:
        """Raise ValueError unless the runtime runs on this service tier."""

    def seat_runtime(self, role_names: Iterable[str]) -> AgentRuntime:
        """The runtime that seats every one of these roles."""
        return self

    def supports_policy(self, policy: ToolPolicy) -> bool:
        """A fresh session of this runtime can honour ``policy``."""
        return False

    def no_tools_runtime(self, roles: Dict[str, Any],
                         session: NoToolsSession) -> AgentRuntime:
        """A fresh runtime of this kind whose seats (``roles``) run under
        ``session``; only called when ``supports_policy`` holds."""
        raise NotImplementedError

    def no_tools_identity(self, policy: ToolPolicy,
                          scratch_placeholder: str) -> Dict[str, Any]:
        """What a candidate-independent session of this runtime records
        about the runtime itself, the scratch path shown as a placeholder."""
        return {}

    def judger_scratch(self) -> Optional[BoundScratch]:
        """The operator-bound directory for a restricted Judger session, or
        None (a fresh temporary directory is used)."""
        return None

    def reuse_identity(self) -> Dict[str, Any]:
        """The seat settings (``REUSE_IDENTITY_FIELDS``) that distinguish
        this runtime's answers in reuse keys. A runtime that does not
        declare them is read for the ones it has."""
        return _present_identity_fields(self)


class RuntimeDecorator(RuntimeCapabilities):
    """A runtime in front of ``seat_of_record``, the runtime whose role
    registry, run options, source-evidence opinion and instruction identity
    it carries. ``complete`` hands each call to ``seat_for(role_name)``."""

    @property
    def seat_of_record(self) -> Any:
        raise NotImplementedError

    def seat_for(self, role_name: str) -> Any:
        """The runtime that runs this seat."""
        return self.seat_of_record

    @property
    def roles(self) -> Dict[str, Any]:
        return self.seat_of_record.roles

    @roles.setter
    def roles(self, value: Dict[str, Any]) -> None:
        self.seat_of_record.roles = value

    @property
    def runtime_instruction_identity(self) -> str:
        return self.seat_of_record.runtime_instruction_identity

    @property
    def run_options(self) -> RunOptions:
        return capabilities_of(self.seat_of_record).run_options

    def allows_compact_source_evidence(self, role_name, variables):
        return capabilities_of(
            self.seat_of_record).allows_compact_source_evidence(
                role_name, variables)

    def complete(self, role_name: str, prompt: str,
                 variables: Dict[str, Any]) -> Any:
        return self.seat_for(role_name).complete(role_name, prompt, variables)


class _DuckCapabilities(RuntimeCapabilities):
    """A backend that does not derive from RuntimeCapabilities, read with
    the defaults: members it has are used, the rest take the default."""

    def __init__(self, backend: Any):
        self._backend = backend

    def _member(self, name: str):
        member = getattr(self._backend, name, None)
        return member if callable(member) else None

    @property
    def uses_role_wire_schema(self) -> bool:
        return getattr(self._backend, "uses_role_wire_schema", False) is True

    @property
    def run_options(self) -> RunOptions:
        return RunOptions(
            compact_inspection=getattr(
                self._backend, "compact_inspection", False) is True,
            judger_prefetch=getattr(
                self._backend, "judger_prefetch", False) is True)

    def allows_lazy_oracle_catalog(self, role_name, variables):
        member = self._member("allows_lazy_oracle_catalog")
        return member(role_name, variables) if member else False

    def allows_compact_source_evidence(self, role_name, variables):
        member = self._member("allows_compact_source_evidence")
        return member(role_name, variables) if member else None

    def routes_direct(self, role_name):
        member = self._member("routes_direct")
        return member(role_name) if member else False

    def captured_images(self, role_name, variables):
        member = self._member("captured_images")
        return member(role_name, variables) if member else ()

    def validate_offline_tool_scope(self, state_dir, toolchain_dir,
                                    shared_cache_dir):
        member = self._member("validate_offline_tool_scope")
        if member:
            member(state_dir, toolchain_dir, shared_cache_dir)

    def seat_runtime(self, role_names):
        return self._backend

    def reuse_identity(self):
        return _present_identity_fields(self._backend)


def capabilities_of(backend: Any) -> RuntimeCapabilities:
    """The declared capabilities of ``backend``."""
    if isinstance(backend, RuntimeCapabilities):
        return backend
    return _DuckCapabilities(backend)


def runtime_identity_type(backend: Any) -> str:
    """The runtime's name in reuse keys: its declared ``RUNTIME_IDENTITY``,
    else its class path."""
    declared = vars(type(backend)).get("RUNTIME_IDENTITY")
    if isinstance(declared, str):
        return declared
    return f"{type(backend).__module__}.{type(backend).__qualname__}"
