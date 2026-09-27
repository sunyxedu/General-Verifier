"""Process-local reuse of independently accepted typed methods.

The cache never reuses a candidate verdict or evidence as the current result.
A hit supplies MethodIRs and the outcome directions for which an independent
Judger accepted their sufficiency.  Holds and refuted signatures are stored
separately; neither direction is inferred from the other.  The current
candidate must run an adapter again and produce a new, mechanically validated
evidence envelope matching the exact method and one accepted direction before
BindingGame can settle anything.

Keeping this cache in memory is a security boundary: executor workspaces can
write shared package caches, so a disk cache would need a separate trusted
writer/authentication protocol.  Process-local entries are both sufficient
for parallel offline Best-of-N scoring and useful for repeated online
revisions in one verifier service.
"""
from __future__ import annotations

import copy
import json
import os
import stat
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..core.model import OUTCOME_HOLDS, OUTCOME_REFUTED, content_hash
from ..core.world import MODE_LIVE, load_world_view
from ..llm.runtime import runtime_identity_type


_IDENTITY_FILE_CAP = 128 * 1024 * 1024


def _read_fixed_bytes(path: str) -> Optional[bytes]:
    """Read a stable regular authority file without following links."""
    if not os.path.lexists(path):
        return None
    expected = os.lstat(path)
    if (not stat.S_ISREG(expected.st_mode)
            or expected.st_size > _IDENTITY_FILE_CAP):
        raise ValueError("cache identity input is not a bounded regular file")
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_NONBLOCK", 0))
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if ((before.st_dev, before.st_ino)
                != (expected.st_dev, expected.st_ino)
                or not stat.S_ISREG(before.st_mode)
                or before.st_size > _IDENTITY_FILE_CAP):
            raise ValueError("cache identity input changed before opening")
        chunks = []
        total = 0
        while True:
            block = os.read(fd, min(1024 * 1024,
                                    _IDENTITY_FILE_CAP + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > _IDENTITY_FILE_CAP:
                raise ValueError("cache identity input exceeds size limit")
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field)
               for field in fields):
            raise ValueError("cache identity input changed while reading")
    finally:
        os.close(fd)
    return b"".join(chunks)


def _read_fixed_json(path: str) -> Any:
    data = _read_fixed_bytes(path)
    if data is None:
        return {"absent": True}
    return json.loads(data.decode("utf-8"))


def _authority_identity(state_dir: str) -> Dict[str, Any]:
    # task.md is text, but use the same stable descriptor and read discipline.
    task_path = os.path.join(state_dir, "task.md")
    task_data = _read_fixed_bytes(task_path)
    task = task_data.decode("utf-8") if task_data is not None else None

    git = _read_fixed_json(os.path.join(state_dir, "baseline", "git.json"))
    manifest = _read_fixed_json(
        os.path.join(state_dir, "baseline", "manifest.json"))
    if isinstance(manifest, dict):
        # The captured source root differs between candidate sandboxes; the
        # file map is the actual authority content.
        manifest = {key: value for key, value in manifest.items()
                    if key != "root"}
    return {"task_text": task, "baseline_git": git,
            "baseline_manifest": manifest}


def _role_identity(role) -> Dict[str, Any]:
    def normalize(value):
        if isinstance(value, type):
            return f"{value.__module__}.{value.__qualname__}"
        if isinstance(value, dict):
            return {str(key): normalize(item)
                    for key, item in sorted(
                        value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    return normalize({
        "name": getattr(role, "name", ""),
        "template": getattr(role, "template", ""),
        "schema": getattr(role, "schema", {}),
        "tools": getattr(role, "tools", []),
        "model": getattr(role, "model", None),
        "max_turns": getattr(role, "max_turns", None),
        "system": getattr(role, "system", ""),
    })


def _backend_identity(backend) -> Optional[Dict[str, Any]]:
    identity: Dict[str, Any] = {
        "type": runtime_identity_type(backend),
    }
    found = False
    for name in ("model", "model_overrides", "effort", "service_tier",
                 "permission_mode", "runtime_instruction_identity"):
        if hasattr(backend, name):
            value = getattr(backend, name)
            if isinstance(value, dict):
                value = dict(sorted(value.items()))
            if value is None or isinstance(value, (str, int, float, bool, dict)):
                identity[name] = value
                found = True
    # FixtureBackend is a deterministic test/replay seam.  Unknown custom
    # backends without a stable semantic identity simply do not get reuse.
    if not found and type(backend).__name__ != "FixtureBackend":
        return None
    return identity


def build_method_cache_key(
    *, state_dir: str, row: Dict[str, Any], obligation, backend,
    oracles_role, judger_role, live_world: bool, adapter_catalog: str,
    append_only_deltas: bool = False,
    append_only_delta_protocol: Optional[str] = None,
) -> Optional[str]:
    """Build an exact semantic key, or disable reuse on uncertain input."""
    if row.get("kind") != "final" or obligation.premise_scope:
        return None
    backend_identity = _backend_identity(backend)
    if backend_identity is None:
        return None
    try:
        authority = _authority_identity(os.path.abspath(state_dir))
        if (not live_world and authority["task_text"] is None
                and authority["baseline_git"] == {"absent": True}
                and authority["baseline_manifest"] == {"absent": True}):
            # With no task authority at all, two captured states may be
            # unrelated tasks that happen to phrase a claim identically.
            return None
        world = load_world_view(state_dir, live_world=live_world)
        world_identity: Dict[str, Any] = {
            "mode": world.mode,
            "scheme": world.scheme,
            "captured_roots": list(world.captured_roots),
            # Host paths are candidate-local.  Logical addresses and mapping
            # shape decide applicability; the adapter resolves them afresh.
            "artifact_addresses": sorted(
                str(item.get("address", "")) for item in world.artifacts),
            "note": world.note,
        }
        if world.mode == MODE_LIVE:
            # Typed live methods contain absolute addresses.  Never move one
            # between verifier states/tasks.
            world_identity["state_scope"] = os.path.realpath(state_dir)
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        return None

    return content_hash({
        "schema": "gv-accepted-method-cache-key-v2",
        # Method applicability is obligation-scoped, but experiment modes
        # must never silently share process-local reuse across different
        # compiler repair semantics.
        "compiler_repair": {
            "append_only_deltas": append_only_deltas,
            "protocol": append_only_delta_protocol,
        },
        # Parent-claim text is explicitly untrusted context in the Oracle
        # protocol.  Exact obligation + controlling clause are the semantic
        # boundary, which lets equivalent final claims from different
        # candidates share one accepted method.
        "claim_kind": row.get("kind"),
        "obligation": {
            "text": obligation.text,
            "required_basis": obligation.required_basis,
            "severity": obligation.severity,
            "clause": obligation.clause,
            "refutable_by_absence": obligation.refutable_by_absence,
            "premise_scope": list(obligation.premise_scope),
        },
        "authority": authority,
        "world": world_identity,
        "adapter_catalog": adapter_catalog,
        "oracles_role": _role_identity(oracles_role),
        "judger_role": _role_identity(judger_role),
        "backend": backend_identity,
    })


@dataclass(frozen=True)
class AcceptedMethod:
    """A direction-scoped sufficiency signature, never a candidate answer."""

    method: Dict[str, Any]
    source_binding_artifact: str
    source_discharge_artifact: str
    source_outcome: str
    audit: Dict[str, Any]

    def detached(self) -> "AcceptedMethod":
        # Method dictionaries originate at an LLM boundary.  Do not let one
        # BindingGame mutate the object another game has trusted.
        return AcceptedMethod(
            method=copy.deepcopy(self.method),
            source_binding_artifact=self.source_binding_artifact,
            source_discharge_artifact=self.source_discharge_artifact,
            source_outcome=self.source_outcome,
            audit=copy.deepcopy(self.audit),
        )


@dataclass
class _Flight:
    done: threading.Event


@dataclass
class MethodCacheLease:
    """One key/direction lookup and any cold-fill flight it owns.

    ``entry`` preserves the old single-entry API.  ``entries`` is the complete
    detached direction set for a key-wide lookup and contains at most one
    independently accepted signature for each substantive outcome.
    """

    cache: "AcceptedMethodCache"
    key: str
    entry: Optional[AcceptedMethod] = None
    entries: tuple[AcceptedMethod, ...] = ()
    direction: Optional[str] = None
    leader: bool = False
    waited: bool = False

    def finish(self, entry: Optional[AcceptedMethod] = None) -> None:
        if self.leader:
            self.cache._finish(self.key, self.direction, entry)
            self.leader = False


class AcceptedMethodCache:
    """Bounded LRU of two independently certified outcome directions.

    The LRU limit counts semantic keys, not direction slots.  Empty keys and a
    missing direction each single-flight independently.  If a leader cannot
    publish an eligible signature, followers bypass the flight and run
    normally; they do not serialize a chain of failed fills.
    """

    def __init__(self, max_entries: int = 2048):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: OrderedDict[
            str, Dict[str, AcceptedMethod]
        ] = OrderedDict()
        self._flights: Dict[tuple[str, Optional[str]], _Flight] = {}

    @staticmethod
    def _detached_entries(bucket) -> tuple[AcceptedMethod, ...]:
        return tuple(entry.detached() for entry in bucket.values())

    def acquire(
        self, key: str, direction: Optional[str] = None,
    ) -> MethodCacheLease:
        """Acquire the whole key, or one missing outcome-direction slot.

        A key-wide lookup starts the original cold-key flight only when no
        direction exists.  A direction lookup starts a separate flight only
        when that exact independently approved direction is absent.
        """
        if direction is not None and direction not in (
                OUTCOME_HOLDS, OUTCOME_REFUTED):
            raise ValueError("direction must be holds or refuted")
        flight_key = (key, direction)
        with self._lock:
            bucket = self._entries.get(key)
            if bucket and (direction is None or direction in bucket):
                self._entries.move_to_end(key)
                entries = self._detached_entries(bucket)
                entry = (entries[0] if direction is None
                         else bucket[direction].detached())
                return MethodCacheLease(
                    self, key, entry=entry, entries=entries,
                    direction=direction)
            flight = self._flights.get(flight_key)
            if flight is None:
                self._flights[flight_key] = _Flight(threading.Event())
                return MethodCacheLease(
                    self, key, direction=direction, leader=True)

        flight.done.wait()
        with self._lock:
            bucket = self._entries.get(key)
            if bucket and (direction is None or direction in bucket):
                self._entries.move_to_end(key)
                entries = self._detached_entries(bucket)
                entry = (entries[0] if direction is None
                         else bucket[direction].detached())
                return MethodCacheLease(
                    self, key, entry=entry, entries=entries,
                    direction=direction, waited=True)
        return MethodCacheLease(
            self, key, direction=direction, waited=True)

    def _touch_and_bound(self, key: str) -> None:
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def _publish_direction(
        self, key: str, entry: AcceptedMethod,
    ) -> tuple[AcceptedMethod, bool]:
        direction = entry.source_outcome
        if direction not in (OUTCOME_HOLDS, OUTCOME_REFUTED):
            raise ValueError("accepted method has no substantive direction")
        bucket = self._entries.setdefault(key, {})
        stored = bucket.get(direction)
        published = stored is None
        if stored is None:
            stored = bucket[direction] = entry.detached()
        self._touch_and_bound(key)
        return stored.detached(), published

    def _finish(
        self, key: str, direction: Optional[str],
        entry: Optional[AcceptedMethod],
    ) -> None:
        flight_key = (key, direction)
        with self._lock:
            flight = self._flights.pop(flight_key, None)
            try:
                if entry is not None and (
                        direction is None
                        or entry.source_outcome == direction):
                    self._publish_direction(key, entry)
            finally:
                # Even a malformed direct caller must not strand followers.
                # BindingGame only supplies schema-derived builtin values,
                # but the cache owns its own concurrency invariant.
                if flight is not None:
                    flight.done.set()

    def publish_if_absent(
        self, key: str, entry: AcceptedMethod,
    ) -> tuple[AcceptedMethod, bool]:
        """Publish only when this independently audited direction is absent.

        Followers deliberately stop waiting when the original leader cannot
        fill. They may then run independently; the first successful signature
        in each direction should still benefit later candidates.
        """
        with self._lock:
            return self._publish_direction(key, entry)

    def clear(self) -> None:
        """Test/administrative reset; never interrupts an active fill."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_DEFAULT_ACCEPTED_METHOD_CACHE = AcceptedMethodCache()


def default_accepted_method_cache() -> AcceptedMethodCache:
    return _DEFAULT_ACCEPTED_METHOD_CACHE
