"""Typed, read-only execution methods proposed by the oracle.

The obligation compiler still decides *what* must be established and the
oracle still decides *how*.  An oracle method may optionally bind its
procedure to one of the adapters in this module.  The adapter then replaces
only the executor seat; the independent judger continues to audit method
sufficiency and execution fidelity.

Adapters are deliberately narrow.  There is no arbitrary-command adapter:
unknown methods, malformed MethodIR, unsupported world mappings, and adapter
failures all fall back to the existing agent executor.  That makes the fast
path an optimization rather than a new source of verifier verdicts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..core.events import EventLog
from ..core.model import (
    BASIS_RECORD,
    BASIS_WORLD,
    INPUT_ROLES,
    OUTCOME_HOLDS,
    OUTCOME_REFUTED,
    PROVENANCES,
    PROVENANCE_DELIVERED,
    PROVENANCE_LEDGER,
    PROVENANCE_PUBLIC,
    PROVENANCE_TASK_GIVEN,
    ROLE_SUBJECT,
    WITNESS_ABSENCE,
    WITNESS_NONE,
    WITNESS_RECORD_CONTRADICTION,
    WITNESS_WORLD_STATE,
    Obligation,
    canonical,
    content_hash,
)
from ..core.world import (
    MODE_CAPTURED,
    MODE_LIVE,
    SCHEME_FILESYSTEM,
    WorldView,
    load_world_view,
)
from .discharge import DischargeResult
from .instruments import ArtifactStore


ADAPTER_SCHEMA = "gv-method-adapter-v1"
EVIDENCE_SCHEMA = "gv-adapter-evidence-v1"
_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ADAPTER_FIELDS = (
    "adapter", "adapter_version", "adapter_arguments", "adapter_inputs",
)
MAX_ARGUMENT_CHARS = 65_536
MAX_DECLARED_INPUTS = 32
MAX_INPUT_REF_CHARS = 4_096
MAX_RETAINED_VALUE_CHARS = 16_384
WORLD_FILE_PROVENANCES = frozenset((
    PROVENANCE_DELIVERED, PROVENANCE_TASK_GIVEN, PROVENANCE_PUBLIC,
))


class AdapterNotApplicable(ValueError):
    """The method cannot safely use the fast path; execute it normally."""


def _no_json_constants(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _object_without_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _strict_object(raw: str) -> Dict[str, Any]:
    if len(raw) > MAX_ARGUMENT_CHARS:
        raise AdapterNotApplicable(
            f"adapter_arguments exceeds {MAX_ARGUMENT_CHARS} characters")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_no_json_constants,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AdapterNotApplicable(
            f"adapter_arguments is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise AdapterNotApplicable("adapter_arguments must encode an object")
    return value


@dataclass(frozen=True)
class AdapterInvocation:
    """The typed portion of one otherwise unchanged oracle method."""

    name: str
    version: int
    arguments: Dict[str, Any]
    inputs: Tuple[Dict[str, str], ...]

    @classmethod
    def from_method(cls, method: Mapping[str, Any]) \
            -> Optional["AdapterInvocation"]:
        present = [field in method for field in _ADAPTER_FIELDS]
        if not any(present):
            return None
        if not all(present):
            missing = [field for field, seen in zip(_ADAPTER_FIELDS, present)
                       if not seen]
            raise AdapterNotApplicable(
                "partial adapter MethodIR; missing " + ", ".join(missing))

        name = method["adapter"]
        version = method["adapter_version"]
        raw_arguments = method["adapter_arguments"]
        raw_inputs = method["adapter_inputs"]
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise AdapterNotApplicable("invalid adapter name")
        if isinstance(version, bool) or not isinstance(version, int) \
                or version < 1:
            raise AdapterNotApplicable("adapter_version must be a positive integer")
        if not isinstance(raw_arguments, str):
            raise AdapterNotApplicable("adapter_arguments must be a JSON string")
        if not isinstance(raw_inputs, list):
            raise AdapterNotApplicable("adapter_inputs must be a list")
        if len(raw_inputs) > MAX_DECLARED_INPUTS:
            raise AdapterNotApplicable(
                f"adapter_inputs exceeds {MAX_DECLARED_INPUTS} entries")

        inputs = []
        for index, item in enumerate(raw_inputs):
            if not isinstance(item, dict) or set(item) != {
                    "ref", "provenance", "role"}:
                raise AdapterNotApplicable(
                    f"adapter_inputs[{index}] has the wrong shape")
            if not isinstance(item["ref"], str) or not item["ref"] \
                    or len(item["ref"]) > MAX_INPUT_REF_CHARS:
                raise AdapterNotApplicable(
                    f"adapter_inputs[{index}].ref must be non-empty and at "
                    f"most {MAX_INPUT_REF_CHARS} characters")
            if item["provenance"] not in PROVENANCES:
                raise AdapterNotApplicable(
                    f"adapter_inputs[{index}] has invalid provenance")
            if item["role"] not in INPUT_ROLES:
                raise AdapterNotApplicable(
                    f"adapter_inputs[{index}] has invalid role")
            inputs.append({key: item[key]
                           for key in ("ref", "provenance", "role")})
        return cls(name, version, _strict_object(raw_arguments), tuple(inputs))

    def public(self) -> Dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "name": self.name,
            "version": self.version,
            "arguments": self.arguments,
            "inputs": list(self.inputs),
        }


def adapter_method_view(method: Mapping[str, Any]) -> str:
    """Stable rendering for the executor and judger prompts."""
    try:
        invocation = AdapterInvocation.from_method(method)
    except AdapterNotApplicable as error:
        return json.dumps({"invalid": str(error)}, ensure_ascii=False,
                          sort_keys=True)
    if invocation is None:
        return "(agent procedure; no typed adapter binding)"
    return json.dumps(invocation.public(), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


@dataclass(frozen=True)
class AdapterExecution:
    outcome: str
    witness: Optional[Any]
    witness_kind: str
    observation: Dict[str, Any]
    consumed: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class ResolvedAddress:
    address: str
    host_path: str
    mode: str
    scheme: str


class AdapterContext:
    """Read-only access to the exact world and ledger being verified."""

    def __init__(self, state_dir: str, live_world: bool, log: EventLog):
        self.state_dir = os.path.abspath(state_dir)
        self.live_world = live_world
        self.log = log
        self.world = load_world_view(self.state_dir, live_world=live_world)

    def resolve(self, address: str) -> ResolvedAddress:
        if not isinstance(address, str) or not address:
            raise AdapterNotApplicable("world address must be a non-empty string")
        view = self.world
        if view.mode == MODE_LIVE:
            if view.scheme != SCHEME_FILESYSTEM or not os.path.isabs(address):
                raise AdapterNotApplicable(
                    "live adapters require an absolute filesystem address")
            return ResolvedAddress(address, os.path.normpath(address),
                                   view.mode, view.scheme)
        if view.mode != MODE_CAPTURED:
            raise AdapterNotApplicable("the delivered world is unavailable")
        host = view.resolve(address)
        if host is None:
            raise AdapterNotApplicable(
                "address is outside the captured world boundary")

        # Explicit artifact mappings are authoritative and may intentionally
        # live outside host_root.  The implicit filesystem mapping may not
        # escape its captured root through '..' or a symlinked parent.
        explicit = any(item.get("address") == address for item in view.artifacts)
        if not explicit:
            root = os.path.realpath(view.host_root)
            target = os.path.realpath(host)
            try:
                inside = os.path.commonpath((root, target)) == root
            except ValueError:
                inside = False
            if not inside:
                raise AdapterNotApplicable(
                    "resolved address escapes the captured world")
        return ResolvedAddress(address, os.path.abspath(host),
                               view.mode, view.scheme)


class MethodAdapter:
    """Base contract implemented by every certified adapter."""

    name = ""
    version = 1
    basis = BASIS_WORLD
    summary = ""
    arguments: Dict[str, str] = {}

    @property
    def identity(self) -> str:
        return f"{self.name}@{self.version}"

    def catalog_entry(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "basis": self.basis,
            "read_only": True,
            "summary": self.summary,
            "arguments": self.arguments,
            "input_contract": (
                "declare every source consumed; the primary address/event "
                "must appear as role=subject"),
        }

    def validate(self, invocation: AdapterInvocation,
                 method: Mapping[str, Any]) -> None:
        if invocation.version != self.version:
            raise AdapterNotApplicable(
                f"adapter version {invocation.version} is unavailable")
        if method.get("basis") != self.basis:
            raise AdapterNotApplicable(
                f"adapter requires basis={self.basis}")

    @staticmethod
    def exact_arguments(arguments: Mapping[str, Any], required: Iterable[str],
                        optional: Iterable[str] = ()) -> None:
        required, optional = set(required), set(optional)
        keys = set(arguments)
        if not required <= keys or not keys <= required | optional:
            missing = sorted(required - keys)
            extra = sorted(keys - required - optional)
            detail = []
            if missing:
                detail.append("missing=" + ",".join(missing))
            if extra:
                detail.append("unknown=" + ",".join(extra))
            raise AdapterNotApplicable(
                "adapter argument shape mismatch" +
                (": " + "; ".join(detail) if detail else ""))

    @staticmethod
    def require_subject(invocation: AdapterInvocation, ref: str,
                        provenances: Optional[Iterable[str]] = None) -> None:
        allowed = set(provenances) if provenances is not None else None
        for item in invocation.inputs:
            if item["ref"] == ref and item["role"] == ROLE_SUBJECT \
                    and (allowed is None or item["provenance"] in allowed):
                return
        suffix = (f" with provenance in {sorted(allowed)!r}"
                  if allowed is not None else "")
        raise AdapterNotApplicable(
            f"adapter inputs must declare subject {ref!r}{suffix}")

    def execute(self, invocation: AdapterInvocation, obligation: Obligation,
                context: AdapterContext) -> AdapterExecution:
        raise NotImplementedError


def _file_kind(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "other"


def _file_identity(resolved: ResolvedAddress, info: os.stat_result,
                   digest: Optional[str] = None) -> Dict[str, Any]:
    result = {
        "ref": resolved.address,
        "mode": resolved.mode,
        "scheme": resolved.scheme,
        "kind": _file_kind(info.st_mode),
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }
    if digest is not None:
        result["sha256"] = digest
    return result


def _same_open_file(before: os.stat_result, after: os.stat_result) -> bool:
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    return all(getattr(before, field) == getattr(after, field)
               for field in fields)


def _open_regular(resolved: ResolvedAddress):
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(resolved.host_path, flags)
    except OSError as error:
        raise AdapterNotApplicable(f"cannot open world artifact: {error}") from error
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise AdapterNotApplicable("adapter requires a regular, non-symlink file")
    return fd, info


class FileExistsAdapter(MethodAdapter):
    name = "filesystem.exists"
    basis = BASIS_WORLD
    summary = "Decide whether one world address exists with the requested kind."
    arguments = {"address": "string", "kind": "any|file|directory"}

    def execute(self, invocation, obligation, context):
        self.exact_arguments(invocation.arguments, ("address",), ("kind",))
        address = invocation.arguments["address"]
        expected_kind = invocation.arguments.get("kind", "any")
        if not isinstance(address, str) or expected_kind not in {
                "any", "file", "directory"}:
            raise AdapterNotApplicable("invalid filesystem.exists arguments")
        self.require_subject(invocation, address, WORLD_FILE_PROVENANCES)
        resolved = context.resolve(address)
        try:
            info = os.lstat(resolved.host_path)
        except FileNotFoundError:
            observation = {"address": address, "exists": False,
                           "world_mode": resolved.mode}
            if not obligation.refutable_by_absence:
                raise AdapterNotApplicable(
                    "absence cannot settle this obligation; the agent "
                    "executor must interpret the missing address")
            return AdapterExecution(
                OUTCOME_REFUTED,
                {"reason": "required world address is absent",
                 "address": address},
                WITNESS_ABSENCE, observation, ())
        except OSError as error:
            raise AdapterNotApplicable(f"cannot inspect world address: {error}") \
                from error
        actual_kind = _file_kind(info.st_mode)
        observation = {"address": address, "exists": True,
                       "actual_kind": actual_kind,
                       "expected_kind": expected_kind,
                       "identity": _file_identity(resolved, info)}
        consumed = (_file_identity(resolved, info),)
        if expected_kind == "any" or actual_kind == expected_kind:
            return AdapterExecution(
                OUTCOME_HOLDS, None, WITNESS_NONE,
                observation, consumed)
        return AdapterExecution(
            OUTCOME_REFUTED,
            {"reason": "world address has the wrong artifact kind",
             "address": address, "expected": expected_kind,
             "actual": actual_kind},
            WITNESS_WORLD_STATE, observation, consumed)


class FileSha256Adapter(MethodAdapter):
    name = "filesystem.sha256"
    basis = BASIS_WORLD
    summary = "Compare a regular world file's bytes with an expected SHA-256."
    arguments = {"address": "string", "expected_sha256": "64 lowercase hex"}

    def execute(self, invocation, obligation, context):
        self.exact_arguments(invocation.arguments,
                             ("address", "expected_sha256"))
        address = invocation.arguments["address"]
        expected = invocation.arguments["expected_sha256"]
        if not isinstance(address, str) or not isinstance(expected, str) \
                or not _SHA256.fullmatch(expected):
            raise AdapterNotApplicable("invalid filesystem.sha256 arguments")
        self.require_subject(invocation, address, WORLD_FILE_PROVENANCES)
        resolved = context.resolve(address)
        try:
            fd, info = _open_regular(resolved)
        except AdapterNotApplicable:
            if not os.path.lexists(resolved.host_path):
                observation = {"address": address, "exists": False,
                               "world_mode": resolved.mode}
                if obligation.refutable_by_absence:
                    return AdapterExecution(
                        OUTCOME_REFUTED,
                        {"reason": "required hashed file is absent",
                         "address": address},
                        WITNESS_ABSENCE, observation, ())
                raise AdapterNotApplicable(
                    "absence cannot settle this obligation; the agent "
                    "executor must interpret the missing file")
            raise
        digest = hashlib.sha256()
        try:
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(fd)
        except OSError as error:
            raise AdapterNotApplicable(f"cannot hash world file: {error}") from error
        finally:
            os.close(fd)
        if not _same_open_file(info, after):
            raise AdapterNotApplicable("world file changed while it was hashed")
        actual = digest.hexdigest()
        identity = _file_identity(resolved, info, actual)
        observation = {"address": address, "expected_sha256": expected,
                       "actual_sha256": actual, "identity": identity}
        if actual == expected:
            return AdapterExecution(
                OUTCOME_HOLDS, None, WITNESS_NONE,
                observation, (identity,))
        return AdapterExecution(
            OUTCOME_REFUTED,
            {"reason": "world file digest differs", "address": address,
             "expected_sha256": expected, "actual_sha256": actual},
            WITNESS_WORLD_STATE, observation, (identity,))


def _read_bounded_file(resolved: ResolvedAddress,
                       limit: int) -> Tuple[bytes, Dict[str, Any]]:
    fd, info = _open_regular(resolved)
    try:
        chunks = []
        total = 0
        while total <= limit:
            chunk = os.read(fd, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
    except OSError as error:
        raise AdapterNotApplicable(f"cannot read world file: {error}") from error
    finally:
        os.close(fd)
    if not _same_open_file(info, after):
        raise AdapterNotApplicable("world file changed while it was read")
    if len(data) > limit:
        raise AdapterNotApplicable(
            f"world file exceeds adapter limit of {limit} bytes")
    digest = hashlib.sha256(data).hexdigest()
    return data, _file_identity(resolved, info, digest)


def _bounded_json(value: Any) -> Any:
    rendered = canonical(value)
    if len(rendered) <= MAX_RETAINED_VALUE_CHARS:
        return value
    return {
        "truncated": True,
        "canonical_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "canonical_chars": len(rendered),
        "type": type(value).__name__,
    }


def _json_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise AdapterNotApplicable("pointer must be an RFC 6901 JSON pointer")
    current = value
    for raw in pointer[1:].split("/"):
        if re.search(r"~(?:[^01]|$)", raw):
            raise AdapterNotApplicable("pointer contains an invalid escape")
        part = raw.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(current, list):
                if not part.isdigit() or (len(part) > 1 and part[0] == "0"):
                    raise KeyError(part)
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                raise KeyError(part)
        except (KeyError, IndexError):
            raise LookupError(pointer) from None
    return current


class JsonValueAdapter(MethodAdapter):
    name = "document.json-value"
    basis = BASIS_WORLD
    summary = "Parse one world file and compare a JSON Pointer value exactly."
    arguments = {
        "address": "string", "pointer": "RFC 6901 string",
        "expected": "any JSON value",
    }
    max_bytes = 8 * 1024 * 1024

    def execute(self, invocation, obligation, context):
        self.exact_arguments(invocation.arguments,
                             ("address", "pointer", "expected"))
        address = invocation.arguments["address"]
        pointer = invocation.arguments["pointer"]
        if not isinstance(address, str) or not isinstance(pointer, str):
            raise AdapterNotApplicable("invalid document.json-value arguments")
        self.require_subject(invocation, address, WORLD_FILE_PROVENANCES)
        resolved = context.resolve(address)
        try:
            data, identity = _read_bounded_file(resolved, self.max_bytes)
        except AdapterNotApplicable:
            if not os.path.lexists(resolved.host_path):
                observation = {"address": address, "exists": False,
                               "world_mode": resolved.mode}
                if obligation.refutable_by_absence:
                    return AdapterExecution(
                        OUTCOME_REFUTED,
                        {"reason": "required JSON artifact is absent",
                         "address": address},
                        WITNESS_ABSENCE, observation, ())
                raise AdapterNotApplicable(
                    "absence cannot settle this obligation; the agent "
                    "executor must interpret the missing document")
            raise
        try:
            parsed = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_no_json_constants,
            )
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            observation = {"address": address, "identity": identity,
                           "parse_error": str(error)}
            return AdapterExecution(
                OUTCOME_REFUTED,
                {"reason": "world artifact is not strict UTF-8 JSON",
                 "address": address, "error": str(error)},
                WITNESS_WORLD_STATE, observation, (identity,))
        try:
            actual = _json_pointer(parsed, pointer)
        except LookupError:
            observation = {"address": address, "identity": identity,
                           "pointer": pointer, "pointer_present": False}
            if obligation.refutable_by_absence:
                return AdapterExecution(
                    OUTCOME_REFUTED,
                    {"reason": "required JSON pointer is absent",
                     "address": address, "pointer": pointer},
                    WITNESS_ABSENCE, observation, (identity,))
            raise AdapterNotApplicable(
                "absence cannot settle this obligation; the agent executor "
                "must interpret the missing document member")
        expected = invocation.arguments["expected"]
        equal = canonical(actual) == canonical(expected)
        observation = {"address": address, "identity": identity,
                       "pointer": pointer, "pointer_present": True,
                       "expected": _bounded_json(expected),
                       "actual": _bounded_json(actual),
                       "equal": equal}
        if equal:
            return AdapterExecution(
                OUTCOME_HOLDS, None, WITNESS_NONE,
                observation, (identity,))
        return AdapterExecution(
            OUTCOME_REFUTED,
            {"reason": "JSON value differs", "address": address,
             "pointer": pointer, "expected": _bounded_json(expected),
             "actual": _bounded_json(actual)},
            WITNESS_WORLD_STATE, observation, (identity,))


def _contains(actual: Any, expected: Any, path: str = "$.payload") \
        -> Sequence[Dict[str, Any]]:
    """Return exact subset mismatches; an empty list means containment."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [{"path": path, "expected": _bounded_json(expected),
                     "actual": _bounded_json(actual)}]
        out = []
        for key, value in expected.items():
            child = f"{path}.{key}"
            if key not in actual:
                out.append({"path": child, "expected": _bounded_json(value),
                            "actual": "<missing>"})
            else:
                out.extend(_contains(actual[key], value, child))
        return out
    if canonical(actual) == canonical(expected):
        return []
    return [{"path": path, "expected": _bounded_json(expected),
             "actual": _bounded_json(actual)}]


class LedgerEventAdapter(MethodAdapter):
    name = "ledger.event"
    basis = BASIS_RECORD
    summary = "Match one immutable ledger event's type and payload subset."
    arguments = {
        "event_seq": "positive integer", "event_type": "optional string",
        "payload_contains": "optional JSON object",
    }

    def execute(self, invocation, obligation, context):
        self.exact_arguments(invocation.arguments, ("event_seq",),
                             ("event_type", "payload_contains"))
        seq = invocation.arguments["event_seq"]
        expected_type = invocation.arguments.get("event_type")
        expected_payload = invocation.arguments.get("payload_contains", {})
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise AdapterNotApplicable("event_seq must be a positive integer")
        if expected_type is not None and not isinstance(expected_type, str):
            raise AdapterNotApplicable("event_type must be a string")
        if not isinstance(expected_payload, dict):
            raise AdapterNotApplicable("payload_contains must be an object")
        ref = f"event:{seq}"
        self.require_subject(invocation, ref, (PROVENANCE_LEDGER,))
        event = context.log.get(seq)
        if event is None:
            observation = {"event_seq": seq, "exists": False}
            if obligation.refutable_by_absence:
                return AdapterExecution(
                    OUTCOME_REFUTED,
                    {"reason": "required ledger event is absent", "ref": ref},
                    WITNESS_ABSENCE, observation, ())
            raise AdapterNotApplicable(
                "absence cannot settle this obligation; the agent executor "
                "must interpret the missing ledger event")
        mismatches = list(_contains(event.get("payload"), expected_payload))
        if expected_type is not None and event.get("type") != expected_type:
            mismatches.insert(0, {
                "path": "$.type", "expected": expected_type,
                "actual": event.get("type"),
            })
        digest = EventLog.event_digest(event)
        consumed = ({"ref": ref, "event_digest": digest,
                     "event_type": event.get("type")},)
        observation = {"event_seq": seq, "exists": True,
                       "event_digest": digest, "mismatches": mismatches}
        if not mismatches:
            return AdapterExecution(
                OUTCOME_HOLDS, None, WITNESS_NONE,
                observation, consumed)
        return AdapterExecution(
            OUTCOME_REFUTED,
            {"reason": "retained ledger event contradicts the expected record",
             "ref": ref, "mismatches": mismatches},
            WITNESS_RECORD_CONTRADICTION, observation, consumed)


class AdapterRegistry:
    """Versioned registry and evidence-envelope builder."""

    def __init__(self, adapters: Iterable[MethodAdapter] = ()):
        self._adapters: Dict[Tuple[str, int], MethodAdapter] = {}
        for adapter in adapters:
            key = (adapter.name, adapter.version)
            if key in self._adapters:
                raise ValueError(f"duplicate adapter {adapter.identity}")
            if not _NAME.fullmatch(adapter.name):
                raise ValueError(f"invalid registered adapter name {adapter.name!r}")
            self._adapters[key] = adapter

    def catalog(self) -> str:
        entries = [adapter.catalog_entry()
                   for _, adapter in sorted(self._adapters.items())]
        return json.dumps(entries, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))

    def execute(self, method: Mapping[str, Any], row: Mapping[str, Any],
                obligation: Obligation, context: AdapterContext,
                artifacts: ArtifactStore) -> Optional[DischargeResult]:
        invocation = AdapterInvocation.from_method(method)
        if invocation is None:
            return None
        adapter = self._adapters.get((invocation.name, invocation.version))
        if adapter is None:
            raise AdapterNotApplicable(
                f"adapter {invocation.name}@{invocation.version} is not registered")
        adapter.validate(invocation, method)
        execution = adapter.execute(invocation, obligation, context)
        payload = {
            "schema": EVIDENCE_SCHEMA,
            "instrument": f"adapter:{adapter.identity}",
            "claim": row.get("id", ""),
            "obligation": {
                "id": obligation.id,
                "text": obligation.text,
                "required_basis": obligation.required_basis,
                "refutable_by_absence": obligation.refutable_by_absence,
            },
            "method": {
                "procedure": method.get("procedure", ""),
                "artifact": method.get("artifact", ""),
                "basis": method.get("basis", ""),
                "strength": method.get("strength", ""),
                "bridges": method.get("bridges", []),
                "adapter": invocation.public(),
            },
            "execution": {
                "outcome": execution.outcome,
                "witness": execution.witness,
                "witness_kind": execution.witness_kind,
                "declared_inputs": list(invocation.inputs),
                "consumed": list(execution.consumed),
                "observation": execution.observation,
            },
        }
        digest = artifacts.put(payload)
        return DischargeResult(
            outcome=execution.outcome,
            witness=execution.witness,
            artifact=digest,
            instrument=f"adapter:{adapter.identity}",
            witness_kind=execution.witness_kind,
            inputs=invocation.inputs,
        )

    def validate_evidence(
        self,
        method: Mapping[str, Any],
        row: Mapping[str, Any],
        obligation: Obligation,
        result: DischargeResult,
        artifacts: ArtifactStore,
    ) -> Tuple[bool, str]:
        """Authenticate the narrow evidence envelope without an LLM.

        This is intentionally stricter than the ordinary Judger path.  It is
        used only when reusing a previously Judger-approved MethodIR, so every
        semantic field must match the fresh call and the artifact filename
        must still match its content.  A mismatch merely disables reuse; the
        full Oracle/executor/Judger path remains available.
        """
        try:
            invocation = AdapterInvocation.from_method(method)
            if invocation is None:
                return False, "method has no typed adapter binding"
            adapter = self._adapters.get((invocation.name, invocation.version))
            if adapter is None:
                return False, "typed adapter is no longer registered"
            adapter.validate(invocation, method)
            expected_instrument = f"adapter:{adapter.identity}"
            if result.instrument != expected_instrument:
                return False, "result instrument does not match MethodIR"
            if tuple(result.inputs) != invocation.inputs:
                return False, "result inputs do not match MethodIR"
            if result.assumed_premises:
                return False, "adapter evidence unexpectedly assumes premises"

            payload = artifacts.get(result.artifact)
            if content_hash(payload) != result.artifact:
                return False, "adapter artifact digest does not match content"
            if set(payload) != {
                    "schema", "instrument", "claim", "obligation", "method",
                    "execution"}:
                return False, "adapter evidence envelope has the wrong shape"
            if payload.get("schema") != EVIDENCE_SCHEMA \
                    or payload.get("instrument") != expected_instrument \
                    or payload.get("claim") != row.get("id", ""):
                return False, "adapter evidence identity mismatch"
            if payload.get("obligation") != {
                "id": obligation.id,
                "text": obligation.text,
                "required_basis": obligation.required_basis,
                "refutable_by_absence": obligation.refutable_by_absence,
            }:
                return False, "adapter evidence obligation mismatch"
            if payload.get("method") != {
                "procedure": method.get("procedure", ""),
                "artifact": method.get("artifact", ""),
                "basis": method.get("basis", ""),
                "strength": method.get("strength", ""),
                "bridges": method.get("bridges", []),
                "adapter": invocation.public(),
            }:
                return False, "adapter evidence method mismatch"

            execution = payload.get("execution")
            if not isinstance(execution, dict) or set(execution) != {
                    "outcome", "witness", "witness_kind", "declared_inputs",
                    "consumed", "observation"}:
                return False, "adapter execution record has the wrong shape"
            if (execution.get("outcome") != result.outcome
                    or execution.get("witness") != result.witness
                    or execution.get("witness_kind") != result.witness_kind
                    or execution.get("declared_inputs")
                    != list(invocation.inputs)):
                return False, "adapter execution record does not match result"
            consumed = execution.get("consumed")
            if not isinstance(consumed, list) or not all(
                    isinstance(item, dict) for item in consumed):
                return False, "adapter consumed-input record is invalid"
            declared_refs = {item["ref"] for item in invocation.inputs}
            if any(item.get("ref") not in declared_refs for item in consumed):
                return False, "adapter consumed an undeclared input"
            if not isinstance(execution.get("observation"), dict):
                return False, "adapter observation is not an object"
        except Exception as error:
            return False, f"adapter evidence validation failed: {error}"
        return True, "mechanically authenticated adapter evidence"


def default_adapter_registry() -> AdapterRegistry:
    return AdapterRegistry((
        FileExistsAdapter(),
        FileSha256Adapter(),
        JsonValueAdapter(),
        LedgerEventAdapter(),
    ))


class AdapterRoutingDischarger:
    """Try a typed adapter, otherwise preserve the existing executor path."""

    def __init__(self, fallback, registry: AdapterRegistry,
                 artifacts: ArtifactStore, log: EventLog,
                 state_dir: str, live_world: bool = True):
        self.fallback = fallback
        self.registry = registry
        self.artifacts = artifacts
        self.log = log
        self.context = AdapterContext(state_dir, live_world, log)

    def adapter_catalog(self) -> str:
        return self.registry.catalog()

    def discharge_adapter_only(self, row, obligation, method):
        """Run only the typed fast path for an accepted-method cache hit.

        Returning ``(None, reason)`` tells BindingGame to use its complete old
        path.  In particular this never turns an inapplicable cached method
        into an agent execution that could accidentally inherit the cached
        sufficiency signature.
        """
        try:
            result = self.registry.execute(
                method, row, obligation, self.context, self.artifacts)
        except Exception as error:
            return None, str(error)
        if result is None:
            return None, "cached method has no typed adapter binding"
        valid, reason = self.registry.validate_evidence(
            method, row, obligation, result, self.artifacts)
        if not valid:
            return None, reason
        self.log.append("adapter_discharge", {
            "claim": row["id"], "obligation": obligation.id,
            "adapter": result.instrument.removeprefix("adapter:"),
            "outcome": result.outcome, "artifact": result.artifact,
            "accepted_method_reuse": True,
        })
        return result, ""

    def discharge(self, row, obligation, method=None, tag=""):
        method = method or {}
        # A judger-requested redo exists to repair execution fidelity.  A
        # deterministic adapter would reproduce identical evidence, so let
        # the existing executor consume the feedback instead.
        if method.get("redo_feedback"):
            self._fallback_event(row, obligation, method,
                                 "judger redo requires the agent executor")
            return self.fallback.discharge(row, obligation, method, tag)
        try:
            result = self.registry.execute(
                method, row, obligation, self.context, self.artifacts)
        except Exception as error:
            self._fallback_event(row, obligation, method, str(error))
            fallback_method = {**method, "adapter_fallback_reason": str(error)}
            return self.fallback.discharge(
                row, obligation, fallback_method, tag)
        if result is None:
            return self.fallback.discharge(row, obligation, method, tag)
        self.log.append("adapter_discharge", {
            "claim": row["id"], "obligation": obligation.id,
            "adapter": result.instrument.removeprefix("adapter:"),
            "outcome": result.outcome, "artifact": result.artifact,
        })
        return result

    def _fallback_event(self, row, obligation, method, reason):
        if not any(field in method for field in _ADAPTER_FIELDS):
            return
        self.log.append("adapter_fallback", {
            "claim": row["id"], "obligation": obligation.id,
            "adapter": str(method.get("adapter", ""))[:80],
            "reason": reason[:500],
        })
