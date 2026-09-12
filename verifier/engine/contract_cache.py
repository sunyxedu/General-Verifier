"""Exact, process-local reuse of Critic-approved final contracts.

Only authority-only compilation may publish here. Entries contain requirements,
not candidate observations, methods, evidence, or verdicts. No semantic matching
is performed; a different key always requires a new compilation.
"""
from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from ..core.model import content_hash
from .method_cache import _role_identity


CONTRACT_CACHE_SCHEMA = "gv-authority-final-contract-v2"


def contract_cache_key(*, view, variables, backend_identity,
                       obligations_role, critic_role, critic_rounds,
                       append_only_deltas=False,
                       append_only_delta_protocol=None) -> str:
    return content_hash({
        "schema": CONTRACT_CACHE_SCHEMA,
        "authority": view.identity,
        "variables": variables,
        "backend": backend_identity,
        "obligations_role": _role_identity(obligations_role),
        "critic_role": _role_identity(critic_role),
        "critic_rounds": critic_rounds,
        "append_only_repair": {
            "enabled": append_only_deltas,
            "protocol": append_only_delta_protocol,
        },
    })


@dataclass(frozen=True)
class CompiledFinalContract:
    payload: dict[str, Any]
    source_digest: str

    @classmethod
    def from_payload(cls, payload):
        detached = copy.deepcopy(payload)
        return cls(detached, content_hash(detached))

    def detached(self):
        return CompiledFinalContract(copy.deepcopy(self.payload), self.source_digest)


@dataclass
class ContractLease:
    cache: "CompiledFinalContractCache"
    key: str
    entry: Optional[CompiledFinalContract] = None
    leader: bool = False
    waited: bool = False

    def finish(self, entry=None):
        if self.leader:
            self.cache._finish(self.key, entry)
            self.leader = False


class CompiledFinalContractCache:
    """Bounded LRU and single flight; an unsuccessful fill wakes all waiters."""

    def __init__(self, max_entries=256):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) \
                or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self._flights = {}

    def acquire(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return ContractLease(self, key, entry.detached())
            flight = self._flights.get(key)
            if flight is None:
                self._flights[key] = threading.Event()
                return ContractLease(self, key, leader=True)
        flight.wait()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return ContractLease(self, key, entry.detached(), waited=True)
        # A failed leader does not serialize every follower's full retry.
        return ContractLease(self, key, waited=True)

    def _finish(self, key, entry):
        with self._lock:
            flight = self._flights.pop(key, None)
            try:
                if entry is not None:
                    self._publish(key, entry)
            finally:
                if flight is not None:
                    flight.set()

    def _publish(self, key, entry):
        self._entries[key] = entry.detached()
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def peek(self, key):
        """Return ``(entry, in_flight)`` without joining the single flight.

        A concurrent authority-only precompile asks this before spending a
        Compiler/Critic run: a published entry or an open flight for the
        same key means the kernel's own lookup will already be served.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                return entry.detached(), False
            return None, key in self._flights

    def publish_if_absent(self, key, entry):
        with self._lock:
            published = key not in self._entries
            if published:
                self._publish(key, entry)
            return self._entries[key].detached(), published

    def clear(self):
        with self._lock:
            self._entries.clear()

    def __len__(self):
        with self._lock:
            return len(self._entries)


_DEFAULT_COMPILED_FINAL_CONTRACT_CACHE = CompiledFinalContractCache()


def default_compiled_final_contract_cache():
    return _DEFAULT_COMPILED_FINAL_CONTRACT_CACHE
