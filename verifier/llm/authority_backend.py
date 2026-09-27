"""Candidate-independent, tool-free Compiler/Critic sessions.

Nothing here replaces the Oracle or the independent Judger. Unknown backends
do not qualify. Codex additionally uses an empty execution directory, explicit
capability restrictions, and an allowlisted JSON event stream: unexpected tool
activity invalidates the generation instead of entering a shared contract.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

from .agents import AgentRole
from .codex.policy import (
    DISABLED_CODE_MODE_NOTICE,
    no_tools_policy_args,
    validate_no_tools_stream,
)
from .codex.stream import _completed_response_without_usage
from .policy import NO_TOOLS, NoToolsSession
from .runtime import capabilities_of
from .schema import AuthorityBackendUnavailable


AUTHORITY_BOUNDARY_VERSION = "gv-authority-no-tools-v1"
# the session's policy: nothing to use but the inlined prompt; a runtime
# whose palette is a hard boundary qualifies without a stream audit
AUTHORITY_POLICY = NO_TOOLS
_FIXTURE_SCOPES = weakref.WeakKeyDictionary()
_FIXTURE_SCOPES_LOCK = threading.Lock()
AUTHORITY_ONLY_NOTE = (
    "\nAUTHORITY-ONLY COMPILATION BOUNDARY: Only the inlined task and pre-task "
    "baseline metadata are available. You have no tools, candidate answer, "
    "candidate files, solver ledger, or graph. Compile the complete task "
    "contract for an arbitrary current candidate. Preserve every quantifier, "
    "modality, source-relative term, process constraint, and output binding. "
    "Use task-named logical paths and 'the current candidate final response' "
    "where the task calls for an answer rather than a file; do not invent its "
    "value. Candidate-dependent references are parameters for the later "
    "Oracle, not facts available here. An unavailable referenced authority "
    "must remain source-relative, never be guessed or silently weakened. "
    "Resolve relative dates only from authoritative timestamps; the runtime "
    "clock is not task authority. Preserve unanchored time references for "
    "the later Oracle instead of substituting the compilation time. "
    "Do not attempt to read local files, use tools, or search the web."
)

def authority_policy_args(scratch_dir: str) -> list[str]:
    """The Codex CLI boundary of the authority policy in ``scratch_dir``."""
    return no_tools_policy_args(scratch_dir, AUTHORITY_POLICY)


def authority_session(scratch_dir: str) -> NoToolsSession:
    """One authority-only session in the empty ``scratch_dir``: only a
    complete, tool-free single turn is accepted."""
    return NoToolsSession(
        policy=AUTHORITY_POLICY,
        scratch=lambda: scratch_dir,
        validate_stream=validate_no_tools_stream,
        # The capability validator runs first. Permit only its exact known
        # startup notice, not arbitrary errors, tool activity or a new event
        # dialect. The recovery parser still checks unique item IDs.
        recover_response=partial(
            _completed_response_without_usage,
            permitted_startup_notice=DISABLED_CODE_MODE_NOTICE))


@dataclass(frozen=True)
class AuthorityBackendSession:
    backend: Any
    obligations_role: Any
    critic_role: Any
    identity: dict

    def revalidate(self):
        if self.identity["calendar_epoch"] != _calendar_epoch():
            raise AuthorityBackendUnavailable(
                "authority runtime calendar changed during compilation")


def _calendar_epoch():
    return {
        "utc_date": time.strftime("%Y-%m-%d", time.gmtime()),
        "local_date": time.strftime("%Y-%m-%d", time.localtime()),
        "timezone": list(time.tzname),
    }


@contextmanager
def authority_compilation_backend(backend, obligations_role, critic_role):
    """Create a private role registry; never mutate a shared candidate backend."""
    from ..engine.method_cache import _backend_identity

    if any(type(role) is not AgentRole
           for role in (obligations_role, critic_role)):
        raise AuthorityBackendUnavailable("unsupported authority-only role")
    roles = [replace(role, tools=[], system=role.system + AUTHORITY_ONLY_NOTE)
             for role in (obligations_role, critic_role)]
    registry = {role.name: role for role in roles}
    identity = {"boundary": AUTHORITY_BOUNDARY_VERSION,
                "backend": _backend_identity(backend),
                # Codex's environment message carries a date/timezone. A
                # process surviving midnight must not reuse yesterday's
                # interpretation of a relative-date authority.
                "calendar_epoch": _calendar_epoch()}
    runtime = capabilities_of(backend)
    # A fixture runtime is an explicit deterministic test seam (declared by
    # the fixture class itself, never inherited), not a promise that
    # arbitrary duck-typed backends cannot inspect candidates.
    if runtime.replays_fixtures:
        # Object ids can be reused after GC while a cache entry survives.
        with _FIXTURE_SCOPES_LOCK:
            if backend not in _FIXTURE_SCOPES:
                _FIXTURE_SCOPES[backend] = uuid.uuid4().hex
            identity["fixture_scope"] = _FIXTURE_SCOPES[backend]
        yield AuthorityBackendSession(backend, *roles, identity)
        return
    # a routing runtime seats the compiler and critic on its direct
    # transport; the authority-only session runs on that transport alone
    # (a single model call sees only its prompt: nothing to inspect)
    seat = capabilities_of(runtime.seat_runtime(
        (obligations_role.name, critic_role.name)))
    if not seat.supports_policy(AUTHORITY_POLICY):
        raise AuthorityBackendUnavailable("unsupported authority-only backend")
    identity.update(seat.no_tools_identity(
        AUTHORITY_POLICY, "/AUTHORITY_SCRATCH"))
    with tempfile.TemporaryDirectory(prefix="gv-authority-") as scratch:
        scratch = os.path.realpath(scratch)
        isolated = seat.no_tools_runtime(registry, authority_session(scratch))
        yield AuthorityBackendSession(isolated, *roles, identity)
