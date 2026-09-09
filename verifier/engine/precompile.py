"""Compile the candidate-blind final contract concurrently with extraction.

The final claim's contract is compiled through the authority-only boundary:
its Compiler and Critic see the task authority and nothing about the
candidate.  Nothing in that compilation depends on extraction, yet offline
scoring ran them one after the other on the critical path.  This module
runs the same compile-and-critique in a thread while the Extractor works,
buffering the events it would have appended, and lets the kernel adopt the
result only when the kernel's own configuration and authority produce the
identical contract key.  Anything else falls back to the unchanged inline
compile, so adoption can never change what is verified.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.model import CLAIM_FINAL, content_hash
from ..llm.model_spans import model_scope

PRECOMPILE_LOCUS = "final_task"


@dataclass
class PrecompileOutcome:
    key: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    events: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    bypass_reason: Optional[str] = None
    error: Optional[BaseException] = None
    elapsed_seconds: float = 0.0


class _BufferLog:
    """Receives the appends the kernel's critique loop would make."""

    def __init__(self, path: str):
        self.path = path
        self.events: List[Tuple[str, Dict[str, Any]]] = []

    def append(self, type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.events.append((type, dict(payload)))
        return {"type": type, "payload": payload}


class _BufferStore:
    def __init__(self):
        self.evidence: List[Dict[str, Any]] = []

    def record_evidence(self, claim_id: str, evidence: Dict[str, Any]) -> None:
        self.evidence.append(dict(evidence))


class _CritiqueContext:
    """The slice of Kernel state the critique loop reads."""

    def __init__(self, log, store, critic_rounds, append_only_deltas,
                 append_only_delta_protocol):
        self.log = log
        self.store = store
        self.critic_rounds = critic_rounds
        self.append_only_deltas = append_only_deltas
        self.append_only_delta_protocol = append_only_delta_protocol

    def __getattr__(self, name):
        # Helpers of the kernel's critique loop (append-guard validation and
        # delta merging) are borrowed unchanged.  Look them up on the class
        # dictionary so a staticmethod stays unbound: ``getattr(Kernel, ...)``
        # would already have unwrapped it into a plain function and binding
        # that to this context would shift its arguments.
        from .kernel import Kernel
        for klass in Kernel.__mro__:
            if name in vars(klass):
                raw = vars(klass)[name]
                if isinstance(raw, staticmethod):
                    return raw.__func__
                if isinstance(raw, classmethod):
                    return raw.__func__.__get__(Kernel, type(Kernel))
                if callable(raw):
                    return raw.__get__(self, _CritiqueContext)
                return raw
        raise AttributeError(name)


class PrecompiledFinalContract:
    """A running or finished concurrent authority-only compilation."""

    def __init__(self, state_dir: str, backend, obligations_role, critic_role,
                 critic_rounds: int, append_only_deltas: bool,
                 append_only_delta_protocol: Optional[str],
                 contract_cache=None):
        self.state_dir = os.path.abspath(state_dir)
        # With the process-local contract cache the thread compiles only
        # when nobody has published or started this key: the kernel's own
        # lookup is then a miss that adopts and publishes this result.
        self.contract_cache = contract_cache
        self.backend = backend
        self.obligations_role = obligations_role
        self.critic_role = critic_role
        self.critic_rounds = critic_rounds
        self.append_only_deltas = append_only_deltas
        self.append_only_delta_protocol = append_only_delta_protocol
        self.pseudo_claim_id = content_hash({
            "precompiled_final_contract": self.state_dir})
        self._outcome = PrecompileOutcome()
        self._thread = threading.Thread(
            target=self._run, name="gv-precompile-final", daemon=True)
        self._adopted = False

    @classmethod
    def start(cls, *args, **kwargs) -> "PrecompiledFinalContract":
        handle = cls(*args, **kwargs)
        handle._thread.start()
        return handle

    def _run(self) -> None:
        import time
        from ..llm.authority_backend import (
            AuthorityBackendUnavailable, authority_compilation_backend,
        )
        from ..llm.schema import validate
        from .authority_view import (
            AuthorityViewUnavailable, build_authority_compilation_view,
        )
        from .contract_cache import contract_cache_key
        from .kernel import Kernel

        started = time.monotonic()
        outcome = self._outcome
        row = {"id": self.pseudo_claim_id, "kind": CLAIM_FINAL,
               "text": "(precompiled final contract)"}
        buffer = _BufferLog(os.path.join(self.state_dir, "events.jsonl"))
        store = _BufferStore()
        try:
            with model_scope(claim_id=self.pseudo_claim_id,
                             claim_kind=CLAIM_FINAL, locus=PRECOMPILE_LOCUS):
                view = build_authority_compilation_view(self.state_dir)
                with authority_compilation_backend(
                        self.backend, self.obligations_role,
                        self.critic_role) as session:
                    variables = Kernel._authority_contract_variables(view)
                    outcome.key = contract_cache_key(
                        view=view, variables=variables,
                        backend_identity=session.identity,
                        obligations_role=session.obligations_role,
                        critic_role=session.critic_role,
                        critic_rounds=self.critic_rounds,
                        append_only_deltas=self.append_only_deltas,
                        append_only_delta_protocol=
                            self.append_only_delta_protocol)
                    if self.contract_cache is not None:
                        entry, in_flight = self.contract_cache.peek(
                            outcome.key)
                        if entry is not None:
                            outcome.bypass_reason = (
                                "contract cache already holds this key")
                            return
                        if in_flight:
                            outcome.bypass_reason = (
                                "contract compilation for this key is "
                                "already in flight")
                            return
                    with model_scope(stage="contract_compile", round=0):
                        payload = session.obligations_role.call(
                            session.backend, variables)
                    context = _CritiqueContext(
                        buffer, store, self.critic_rounds,
                        self.append_only_deltas,
                        self.append_only_delta_protocol)
                    payload = Kernel._critique(
                        context, row, payload, variables,
                        backend=session.backend,
                        obligations_role=session.obligations_role,
                        critic_role=session.critic_role)
                    view.revalidate(self.state_dir)
                    session.revalidate()
                    if payload is not None:
                        validate(session.obligations_role.schema, payload)
                    outcome.payload = payload
        except (AuthorityViewUnavailable, AuthorityBackendUnavailable) as error:
            outcome.bypass_reason = str(error)[:300]
        except BaseException as error:  # adoption re-runs inline instead
            outcome.error = error
        finally:
            outcome.events = buffer.events
            outcome.evidence = store.evidence
            outcome.elapsed_seconds = time.monotonic() - started

    def wait(self) -> PrecompileOutcome:
        self._thread.join()
        return self._outcome

    @property
    def finished(self) -> bool:
        return not self._thread.is_alive()
