"""Typed, read-only execution methods proposed by the oracle.

The obligation compiler still decides *what* must be established and the
oracle still decides *how*.  An oracle method may optionally bind its
procedure to one of the adapters in this module.  The adapter then replaces
only the executor seat; the independent judger continues to audit method
sufficiency and execution fidelity.

Adapters are deliberately narrow.  There is no arbitrary-command executor:
unknown methods, malformed MethodIR, unsupported world mappings, and adapter
failures all fall back to the existing agent executor.  That makes the fast
path an optimization rather than a new source of verifier verdicts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

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
    WITNESS_EXECUTION,
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
from .browser import (
    BROWSER_DOCUMENT_POLICY,
    BROWSER_NETWORK_POLICY,
    BROWSER_OBSERVATION_PROFILE,
    BROWSER_RESPONSE_SCHEMA,
    BrowserObserver,
    configured_browser_observer,
)
from .instruments import ArtifactStore


ADAPTER_SCHEMA = "gv-method-adapter-v1"
EVIDENCE_SCHEMA = "gv-adapter-evidence-v1"
FIXED_INPUT_BINDING = "fixed-v1"
_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ADAPTER_FIELDS = (
    "adapter", "adapter_version", "adapter_arguments", "adapter_inputs",
)
MAX_ARGUMENT_CHARS = 65_536
MAX_DECLARED_INPUTS = 32
MAX_INPUT_REF_CHARS = 4_096
MAX_RETAINED_VALUE_CHARS = 16_384
MAX_JSON_PREDICATES = 16
MAX_JSON_SHAPE_PROPERTIES = 64
MAX_JSON_PROPERTY_NAME_CHARS = 1_024
MAX_RETAINED_JSON_FAILURES = 64
MAX_JSON_DEPTH = 128
MAX_JSON_NODES = 100_000
MAX_JSON_INTEGER_DIGITS = 4_096
MAX_LEDGER_EVENTS = 20_000
MAX_LEDGER_BYTES = 32 * 1024 * 1024
MAX_LEDGER_VALUE_BYTES = 1024 * 1024
MAX_LEDGER_VALUE_BYTES_TOTAL = 8 * 1024 * 1024
MAX_LEDGER_SCOPE_PREDICATES = 16
MAX_LEDGER_TOOL_NAMES = 32
MAX_LEDGER_TOOL_NAME_CHARS = 256
MAX_RETAINED_LEDGER_VALUES = 64
MAX_BROWSER_ASSERTIONS = 16
MAX_BROWSER_WAIT_MS = 5_000
WORLD_FILE_PROVENANCES = frozenset((
    PROVENANCE_DELIVERED, PROVENANCE_TASK_GIVEN, PROVENANCE_PUBLIC,
))


class AdapterNotApplicable(ValueError):
    """The method cannot safely use the fast path; execute it normally."""


def _no_json_constants(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _bounded_json_int(value: str) -> int:
    if len(value.removeprefix("-")) > MAX_JSON_INTEGER_DIGITS:
        raise AdapterNotApplicable(
            "JSON integer exceeds the predicate resource boundary")
    return int(value)


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
    input_binding: Optional[str] = None

    @classmethod
    def from_method(cls, method: Mapping[str, Any]) \
            -> Optional["AdapterInvocation"]:
        present = [field in method for field in _ADAPTER_FIELDS]
        binding = method.get("adapter_input_binding")
        if "adapter_input_binding" in method and binding != FIXED_INPUT_BINDING:
            raise AdapterNotApplicable("unsupported adapter_input_binding")
        if not any(present) and binding is None:
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
            if (not isinstance(item["provenance"], str)
                    or item["provenance"] not in PROVENANCES):
                raise AdapterNotApplicable(
                    f"adapter_inputs[{index}] has invalid provenance")
            if not isinstance(item["role"], str) or item["role"] not in INPUT_ROLES:
                raise AdapterNotApplicable(
                    f"adapter_inputs[{index}] has invalid role")
            inputs.append({key: item[key]
                           for key in ("ref", "provenance", "role")})
        return cls(name, version, _strict_object(raw_arguments), tuple(inputs), binding)

    def public(self) -> Dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "name": self.name,
            "version": self.version,
            "arguments": self.arguments,
            "inputs": list(self.inputs),
            **({"input_binding": self.input_binding}
               if self.input_binding is not None else {}),
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


@dataclass(frozen=True)
class FixedAdapterInput:
    """A trusted instrument's source slot, never inferred from task text.

    A slot is either literal or a reference prefix followed by one exact
    positive-integer argument. It cannot interpolate paths or execute code.
    """

    ref: str
    provenance: str
    role: str = ROLE_SUBJECT
    integer_argument: Optional[str] = None

    def __post_init__(self):
        if (not isinstance(self.ref, str) or not self.ref
                or len(self.ref) > MAX_INPUT_REF_CHARS
                or not isinstance(self.provenance, str)
                or self.provenance not in PROVENANCES or self.role != ROLE_SUBJECT
                or (self.integer_argument is not None and (
                    not isinstance(self.integer_argument, str)
                    or not self.integer_argument))):
            raise ValueError("invalid verifier-owned fixed input descriptor")

    def public(self):
        return {"ref": self.ref, "provenance": self.provenance, "role": self.role,
                **({"append_positive_integer_argument": self.integer_argument}
                   if self.integer_argument is not None else {})}

    def resolve(self, arguments):
        ref = self.ref
        if self.integer_argument is not None:
            value = arguments.get(self.integer_argument)
            if type(value) is not int or value < 1:
                raise AdapterNotApplicable(
                    f"fixed input requires positive integer {self.integer_argument}")
            ref += str(value)
        if len(ref) > MAX_INPUT_REF_CHARS:
            raise AdapterNotApplicable("fixed input reference is too long")
        return {"ref": ref, "provenance": self.provenance, "role": self.role}


class MethodAdapter:
    """Base contract implemented by every certified adapter."""

    name = ""
    version = 1
    basis = BASIS_WORLD
    summary = ""
    arguments: Dict[str, str] = {}
    # New instruments opt in only after their whole execution contract is
    # reproducible from MethodIR and the current declared inputs.
    accepted_method_reuse = False
    fixed_inputs: Tuple[FixedAdapterInput, ...] = ()

    @property
    def identity(self) -> str:
        return f"{self.name}@{self.version}"

    def catalog_entry(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "basis": self.basis,
            "read_only": True,
            "accepted_method_reuse": self.accepted_method_reuse,
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
    accepted_method_reuse = True
    name = "filesystem.exists"
    basis = BASIS_WORLD
    summary = (
        "Decide whether one world address exists with the requested kind. "
        "Reports address, exists, actual_kind (absent when missing), "
        "expected_kind, and world_mode; does not inspect file contents."
    )
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
                           "actual_kind": "absent",
                           "expected_kind": expected_kind,
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
                       "world_mode": resolved.mode,
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
    accepted_method_reuse = True
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


def _strict_json_document(data: bytes) -> Any:
    return json.loads(
        data.decode("utf-8", errors="strict"),
        object_pairs_hook=_object_without_duplicates,
        parse_constant=_no_json_constants,
        parse_float=_finite_json_float,
        parse_int=_bounded_json_int,
    )


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
    accepted_method_reuse = True
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


_JSON_TYPES = frozenset((
    "null", "boolean", "integer", "number", "string", "array", "object",
))


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise ValueError("JSON value has an unsupported runtime type")


def _matches_json_type(value: Any, expected: str) -> bool:
    # JSON has one number production. ``integer`` is the useful exact subset;
    # ``number`` deliberately includes both decoded integer and float values.
    if expected == "number":
        return type(value) in (int, float)
    return _json_type(value) == expected


def _validate_json_tree(root: Any) -> None:
    """Reject invalid Unicode and bound work before evaluating predicates.

    Parser recursion/resource failures remain adapter-inapplicable.  A parsed
    string containing an unpaired surrogate is substantively invalid UTF-8
    JSON for this adapter, while a merely large/deep valid value falls back.
    """
    stack = [(root, 0)]
    nodes = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise AdapterNotApplicable(
                "JSON document exceeds the predicate resource boundary")
        if isinstance(value, str):
            value.encode("utf-8", errors="strict")
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
        elif isinstance(value, dict):
            for key, item in value.items():
                key.encode("utf-8", errors="strict")
                stack.append((item, depth + 1))
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("non-finite JSON number")
        elif value is not None and type(value) not in (bool, int):
            raise ValueError("unsupported JSON value type")


def _validate_json_property_name(value: Any, field: str) -> str:
    if (not isinstance(value, str)
            or len(value) > MAX_JSON_PROPERTY_NAME_CHARS):
        raise AdapterNotApplicable(
            f"{field} must be a string of at most "
            f"{MAX_JSON_PROPERTY_NAME_CHARS} characters")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise AdapterNotApplicable(f"{field} contains invalid Unicode") from error
    return value


def _validate_json_predicate(
        value: Any, *, allow_presence_only_required: bool = False,
) -> Dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise AdapterNotApplicable(
            "each JSON predicate must be an object with a string kind")
    kind = value["kind"]
    if kind == "valid":
        required = {"kind"}
    elif kind in {"equals", "array.contains"}:
        required = {"kind", "pointer", "expected"}
    elif kind == "array.length":
        required = {"kind", "pointer", "operator", "expected"}
    elif kind == "array.items-object":
        required = {
            "kind", "pointer", "required", "properties",
            "additional_properties",
        }
    else:
        raise AdapterNotApplicable(f"unknown JSON predicate kind {kind!r}")
    if set(value) != required:
        raise AdapterNotApplicable(
            f"JSON predicate {kind!r} has the wrong shape")

    if "pointer" in value:
        pointer = value["pointer"]
        if not isinstance(pointer, str):
            raise AdapterNotApplicable("JSON predicate pointer must be a string")
        try:
            pointer.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise AdapterNotApplicable(
                "JSON predicate pointer contains invalid Unicode") from error
        # Exercise the same parser used during execution. A valid non-root
        # pointer cannot resolve against None and therefore raises LookupError.
        try:
            _json_pointer(None, pointer)
        except LookupError:
            pass

    if kind in {"equals", "array.contains"}:
        try:
            _validate_json_tree(value["expected"])
        except UnicodeEncodeError as error:
            raise AdapterNotApplicable(
                "JSON predicate expected value contains invalid Unicode") \
                from error
    elif kind == "array.length":
        operator = value["operator"]
        expected = value["expected"]
        if not isinstance(operator, str) or operator not in {"eq", "gte", "lte"}:
            raise AdapterNotApplicable(
                "array.length operator must be eq, gte, or lte")
        if type(expected) is not int or expected < 0:
            raise AdapterNotApplicable(
                "array.length expected must be a non-negative integer")
    elif kind == "array.items-object":
        required_keys = value["required"]
        properties = value["properties"]
        additional = value["additional_properties"]
        if (not isinstance(required_keys, list)
                or len(required_keys) > MAX_JSON_SHAPE_PROPERTIES
                or not isinstance(properties, dict)
                or len(properties) > MAX_JSON_SHAPE_PROPERTIES
                or type(additional) is not bool):
            raise AdapterNotApplicable(
                "invalid array.items-object shape declaration")
        checked_required = [
            _validate_json_property_name(item, "required property")
            for item in required_keys
        ]
        if len(checked_required) != len(set(checked_required)):
            raise AdapterNotApplicable(
                "array.items-object required properties must be unique")
        checked_properties = {}
        for key, expected_type in properties.items():
            key = _validate_json_property_name(key, "property name")
            if (not isinstance(expected_type, str)
                    or expected_type not in _JSON_TYPES):
                raise AdapterNotApplicable(
                    f"unsupported JSON property type {expected_type!r}")
            checked_properties[key] = expected_type
        if (allow_presence_only_required
                and len(set(checked_required) | set(checked_properties))
                > MAX_JSON_SHAPE_PROPERTIES):
            raise AdapterNotApplicable(
                "array.items-object declared property union exceeds the "
                "shape property boundary")
        if (not allow_presence_only_required
                and not set(checked_required) <= set(checked_properties)):
            raise AdapterNotApplicable(
                "every required property needs a declared type")
    return dict(value)


class JsonPredicateAdapter(MethodAdapter):
    """Apply a bounded conjunction of closed predicates to strict JSON."""

    name = "document.json-predicate"
    basis = BASIS_WORLD
    # This path is eligible for a compact *fresh* Judger transport, but never
    # for a cached prior verdict/signature.
    accepted_method_reuse = False
    allow_presence_only_required = False
    summary = (
        "Parse one bounded regular world file as strict UTF-8 JSON, rejecting "
        "duplicate keys, invalid Unicode, and non-finite numbers, then apply "
        "1..16 declarative predicates: valid, type-sensitive structural "
        "equals, exact array contains, array length eq/gte/lte, or bounded "
        "array item object shape/type. Predicates are conjoined; no code or "
        "selector language is executed."
    )
    arguments = {
        "address": "string",
        "predicates": (
            "1..16 closed objects. valid={kind}; "
            "equals|array.contains={kind,pointer,expected}; "
            "array.length={kind,pointer,operator:eq|gte|lte,expected:int>=0}; "
            "array.items-object={kind,pointer,required:string[],"
            "properties:{name:null|boolean|integer|number|string|array|object},"
            "additional_properties:boolean}. Equality and containment are "
            "type-sensitive; object key order is ignored and array order is "
            "preserved. number accepts integer or floating-point JSON numbers; "
            "integer accepts only integer JSON numbers."
        ),
    }
    max_bytes = 8 * 1024 * 1024

    def catalog_entry(self) -> Dict[str, Any]:
        entry = super().catalog_entry()
        # The base catalog is intentionally simple. This closed union gives
        # the Oracle an exact machine-readable field contract without adding
        # a general schema language for the adapter to interpret.
        item_shape = {
            "kind": "array.items-object",
            "fields": ["kind", "pointer", "required", "properties",
                       "additional_properties"],
            "property_types": sorted(_JSON_TYPES),
            "pointer_required": True,
            "root_pointer": "",
        }
        if self.allow_presence_only_required:
            item_shape.update({
                "required_semantics": "presence-only",
                "properties_semantics": "type-if-present",
                "allowed_keys_when_closed": "required_union_properties",
            })
        else:
            item_shape["required_must_be_typed"] = True
        entry["predicate_variants"] = [
            {"kind": "valid", "fields": ["kind"]},
            {"kind": "equals",
             "fields": ["kind", "pointer", "expected"]},
            {"kind": "array.contains",
             "fields": ["kind", "pointer", "expected"]},
            {"kind": "array.length",
             "fields": ["kind", "pointer", "operator", "expected"],
             "operators": ["eq", "gte", "lte"]},
            item_shape,
        ]
        entry["resource_limits"] = {
            "file_bytes": self.max_bytes,
            "predicates": MAX_JSON_PREDICATES,
            "shape_properties": MAX_JSON_SHAPE_PROPERTIES,
            "json_depth": MAX_JSON_DEPTH,
            "json_nodes": MAX_JSON_NODES,
            "integer_digits": MAX_JSON_INTEGER_DIGITS,
            "retained_failures": MAX_RETAINED_JSON_FAILURES,
        }
        return entry

    @staticmethod
    def _base_result(index: int, predicate: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "index": index,
            "kind": predicate["kind"],
            **({"pointer": predicate["pointer"]}
               if "pointer" in predicate else {}),
        }

    def _array_items_result(self, index: int, predicate: Mapping[str, Any],
                            actual: Any) -> Dict[str, Any]:
        result = JsonPredicateAdapter._base_result(index, predicate)
        result["actual_type"] = _json_type(actual)
        if not isinstance(actual, list):
            result.update({
                "matched": False,
                "failure_kind": WITNESS_WORLD_STATE,
                "failure_count": 1,
                "failures": [{
                    "reason": "selected value is not an array",
                    "actual_type": result["actual_type"],
                }],
                "failures_truncated": 0,
            })
            return result

        required = tuple(predicate["required"])
        properties = predicate["properties"]
        allow_extra = predicate["additional_properties"]
        failures = []
        failure_count = 0
        saw_positive = False
        saw_absence = False

        def add_failure(failure: Dict[str, Any], *, absence: bool) -> None:
            nonlocal failure_count, saw_positive, saw_absence
            failure_count += 1
            saw_absence = saw_absence or absence
            saw_positive = saw_positive or not absence
            if len(failures) < MAX_RETAINED_JSON_FAILURES:
                failures.append(failure)

        # V1 requires required ⊆ properties, so including both leaves its
        # closed-object behavior unchanged. V2 permits presence-only keys.
        declared_keys = set(required) | set(properties)
        for item_index, item in enumerate(actual):
            if not isinstance(item, dict):
                add_failure({
                    "item_index": item_index,
                    "reason": "array item is not an object",
                    "actual_type": _json_type(item),
                }, absence=False)
                continue
            missing = [key for key in required if key not in item]
            if missing:
                add_failure({
                    "item_index": item_index,
                    "reason": "required properties are absent",
                    "properties": missing,
                }, absence=True)
            if not allow_extra:
                extra = sorted(set(item) - declared_keys)
                if extra:
                    add_failure({
                        "item_index": item_index,
                        "reason": "additional properties are present",
                        "property_count": len(extra),
                        "properties": extra[:MAX_JSON_SHAPE_PROPERTIES],
                        "properties_truncated": max(
                            0, len(extra) - MAX_JSON_SHAPE_PROPERTIES),
                    }, absence=False)
            for key, expected_type in properties.items():
                if key in item and not _matches_json_type(
                        item[key], expected_type):
                    add_failure({
                        "item_index": item_index,
                        "reason": "property has the wrong JSON type",
                        "property": key,
                        "expected_type": expected_type,
                        "actual_type": _json_type(item[key]),
                    }, absence=False)
        result.update({
            "matched": failure_count == 0,
            "array_length": len(actual),
            "items_examined": len(actual),
            "failure_count": failure_count,
            "failures": failures,
            "failures_truncated": failure_count - len(failures),
        })
        if failure_count:
            result["failure_kind"] = (
                WITNESS_WORLD_STATE if saw_positive else WITNESS_ABSENCE)
            result["contains_absence_failure"] = saw_absence
        return result

    def _predicate_result(self, index: int, predicate: Mapping[str, Any],
                          parsed: Any) -> Dict[str, Any]:
        kind = predicate["kind"]
        result = self._base_result(index, predicate)
        if kind == "valid":
            return {**result, "matched": True, "root_type": _json_type(parsed)}
        try:
            actual = _json_pointer(parsed, predicate["pointer"])
        except LookupError:
            return {
                **result,
                "pointer_present": False,
                "matched": False,
                "failure_kind": WITNESS_ABSENCE,
                "reason": "JSON pointer is absent",
            }
        result.update({"pointer_present": True,
                       "actual_type": _json_type(actual)})
        if kind == "equals":
            expected = predicate["expected"]
            matched = canonical(actual) == canonical(expected)
            return {
                **result,
                "expected": _bounded_json(expected),
                "actual": _bounded_json(actual),
                "matched": matched,
                **({"failure_kind": WITNESS_WORLD_STATE} if not matched else {}),
            }
        if kind == "array.contains":
            if not isinstance(actual, list):
                return {
                    **result, "matched": False,
                    "failure_kind": WITNESS_WORLD_STATE,
                    "reason": "selected value is not an array",
                }
            expected = predicate["expected"]
            expected_canonical = canonical(expected)
            match_count = 0
            first_match = None
            for item_index, item in enumerate(actual):
                if canonical(item) == expected_canonical:
                    match_count += 1
                    if first_match is None:
                        first_match = item_index
            return {
                **result,
                "expected": _bounded_json(expected),
                "array_length": len(actual),
                "match_count": match_count,
                "first_match_index": first_match,
                "matched": match_count > 0,
                **({"failure_kind": WITNESS_ABSENCE}
                   if match_count == 0 else {}),
            }
        if kind == "array.length":
            if not isinstance(actual, list):
                return {
                    **result, "matched": False,
                    "failure_kind": WITNESS_WORLD_STATE,
                    "reason": "selected value is not an array",
                }
            operator = predicate["operator"]
            expected = predicate["expected"]
            actual_length = len(actual)
            matched = {
                "eq": actual_length == expected,
                "gte": actual_length >= expected,
                "lte": actual_length <= expected,
            }[operator]
            absence = (operator in {"eq", "gte"}
                       and actual_length < expected)
            return {
                **result,
                "operator": operator,
                "expected_length": expected,
                "actual_length": actual_length,
                "matched": matched,
                **({"failure_kind": (
                    WITNESS_ABSENCE if absence else WITNESS_WORLD_STATE)}
                   if not matched else {}),
            }
        return self._array_items_result(index, predicate, actual)

    def execute(self, invocation, obligation, context):
        self.exact_arguments(invocation.arguments, ("address", "predicates"))
        address = invocation.arguments["address"]
        raw_predicates = invocation.arguments["predicates"]
        if (not isinstance(address, str)
                or not isinstance(raw_predicates, list)
                or not 1 <= len(raw_predicates) <= MAX_JSON_PREDICATES):
            raise AdapterNotApplicable(
                f"document.json-predicate requires 1..{MAX_JSON_PREDICATES} "
                "predicates")
        predicates = [
            _validate_json_predicate(
                item,
                allow_presence_only_required=(
                    self.allow_presence_only_required),
            )
            for item in raw_predicates
        ]
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
            parsed = _strict_json_document(data)
            _validate_json_tree(parsed)
        except AdapterNotApplicable:
            raise
        except (UnicodeDecodeError, UnicodeEncodeError, ValueError,
                json.JSONDecodeError) as error:
            observation = {
                "address": address,
                "identity": identity,
                "strict_json_valid": False,
                "parse_error": str(error)[:2_000],
            }
            return AdapterExecution(
                OUTCOME_REFUTED,
                {"reason": "world artifact is not strict UTF-8 JSON",
                 "address": address, "error": str(error)[:2_000]},
                WITNESS_WORLD_STATE, observation, (identity,))

        results = [self._predicate_result(index, predicate, parsed)
                   for index, predicate in enumerate(predicates)]
        failed = [result for result in results if not result["matched"]]
        retained_document = _bounded_json(parsed)
        observation = {
            "address": address,
            "identity": identity,
            "strict_json_valid": True,
            # Keep one bounded copy, rather than duplicating a selected array
            # in every predicate receipt. Small documents are fully
            # inspectable by the fresh Judger; large ones remain bound by an
            # exact canonical digest and explicit incompleteness marker.
            "document": retained_document,
            "document_complete": retained_document is parsed,
            "predicate_count": len(results),
            "predicates_examined": len(results),
            "results": results,
            "matched": not failed,
        }
        if not failed:
            return AdapterExecution(
                OUTCOME_HOLDS, None, WITNESS_NONE, observation, (identity,))
        positive = any(
            result.get("failure_kind") == WITNESS_WORLD_STATE
            for result in failed)
        if not positive and not obligation.refutable_by_absence:
            raise AdapterNotApplicable(
                "predicate failure proves only absence; the agent executor "
                "must interpret it for this obligation")
        witness_kind = WITNESS_WORLD_STATE if positive else WITNESS_ABSENCE
        return AdapterExecution(
            OUTCOME_REFUTED,
            {"reason": "one or more JSON predicates did not match",
             "address": address, "failed": failed},
            witness_kind, observation, (identity,))


class JsonPredicateAdapterV2(JsonPredicateAdapter):
    """Separate object-member presence from optional type constraints."""

    version = 2
    allow_presence_only_required = True
    summary = (
        "Parse one bounded regular world file as strict UTF-8 JSON, rejecting "
        "duplicate keys, invalid Unicode, and non-finite numbers, then apply "
        "1..16 declarative predicates. Array item object predicates express "
        "required member presence independently from optional member type "
        "constraints; no code or selector language is executed."
    )
    arguments = {
        **JsonPredicateAdapter.arguments,
        "predicates": (
            JsonPredicateAdapter.arguments["predicates"]
            + " For array.items-object, required means presence only; "
              "properties constrains a key only when present; with "
              "additional_properties=false the allowed keys are the union "
              "of required and properties. pointer is always required and "
              "the document root is pointer=\"\"."
        ),
    }


class TextValueAdapter(MethodAdapter):
    """Apply one exact, bounded literal predicate to a UTF-8 document.

    This deliberately does not expose regular expressions, normalization, or
    executable selectors. The Oracle still has to justify that the chosen
    literal predicate settles the obligation, and the independent Judger
    still audits that semantic bridge.
    """

    name = "document.text"
    accepted_method_reuse = True
    basis = BASIS_WORLD
    summary = (
        "Decode one bounded regular world file as strict UTF-8 and apply one "
        "literal predicate: equals, contains, starts_with, or ends_with. "
        "No normalization, regular expression, or command execution occurs."
    )
    arguments = {
        "address": "string",
        "predicate": "equals|contains|starts_with|ends_with",
        "expected": f"literal string of at most {MAX_RETAINED_VALUE_CHARS} characters",
    }
    max_bytes = 8 * 1024 * 1024
    predicates = frozenset(("equals", "contains", "starts_with", "ends_with"))

    def execute(self, invocation, obligation, context):
        self.exact_arguments(
            invocation.arguments, ("address", "predicate", "expected"))
        address = invocation.arguments["address"]
        predicate = invocation.arguments["predicate"]
        expected = invocation.arguments["expected"]
        if (not isinstance(address, str)
                or predicate not in self.predicates
                or not isinstance(expected, str)
                or (predicate != "equals" and not expected)
                or len(expected) > MAX_RETAINED_VALUE_CHARS):
            raise AdapterNotApplicable("invalid document.text arguments")
        try:
            expected.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise AdapterNotApplicable(
                "document.text expected value is not valid UTF-8 text") from error
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
                        {"reason": "required text artifact is absent",
                         "address": address},
                        WITNESS_ABSENCE, observation, ())
                raise AdapterNotApplicable(
                    "absence cannot settle this obligation; the agent "
                    "executor must interpret the missing document")
            raise
        try:
            actual = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            observation = {
                "address": address, "identity": identity,
                "encoding": "utf-8", "decode_error": str(error),
            }
            return AdapterExecution(
                OUTCOME_REFUTED,
                {"reason": "world artifact is not strict UTF-8 text",
                 "address": address, "error": str(error)},
                WITNESS_WORLD_STATE, observation, (identity,))

        if predicate == "equals":
            matched = actual == expected
            first_match = 0 if matched else None
        elif predicate == "contains":
            first_match = actual.find(expected)
            matched = first_match >= 0
            if not matched:
                first_match = None
        elif predicate == "starts_with":
            matched = actual.startswith(expected)
            first_match = 0 if matched else None
        else:
            matched = actual.endswith(expected)
            first_match = len(actual) - len(expected) if matched else None

        observation = {
            "address": address,
            "identity": identity,
            "encoding": "utf-8",
            "predicate": predicate,
            "expected": expected,
            "text_characters": len(actual),
            "matched": matched,
            "first_match_character": first_match,
        }
        if matched:
            return AdapterExecution(
                OUTCOME_HOLDS, None, WITNESS_NONE, observation, (identity,))
        return AdapterExecution(
            OUTCOME_REFUTED,
            {"reason": "UTF-8 text literal predicate did not match",
             "address": address, "predicate": predicate,
             "expected": expected, "actual_sha256": identity["sha256"],
             "actual_characters": len(actual)},
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
    fixed_inputs = (FixedAdapterInput(
        "event:", PROVENANCE_LEDGER, integer_argument="event_seq"),)

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


def _validate_ledger_pointer(pointer: Any, field: str) -> str:
    """Validate every RFC 6901 segment without dereferencing any value."""
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise AdapterNotApplicable(f"{field} must be an RFC 6901 pointer")
    if pointer:
        for raw in pointer[1:].split("/"):
            if re.search(r"~(?:[^01]|$)", raw):
                raise AdapterNotApplicable(
                    f"{field} contains an invalid RFC 6901 escape")
    return pointer


def _bounded_ledger_prefix(log: EventLog, through_seq: int):
    """Yield a validated ledger prefix without reading beyond its byte cap."""
    if through_seq == 0:
        return
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(log.path, flags)
    except OSError as error:
        raise AdapterNotApplicable(
            f"cannot open the declared ledger prefix: {error}") from error
    stream = None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise AdapterNotApplicable("ledger must be a regular file")
        stream = os.fdopen(fd, "rb")
        fd = -1
        expected_seq = 0
        bytes_read = 0
        while expected_seq < through_seq:
            remaining = MAX_LEDGER_BYTES - bytes_read
            raw = stream.readline(remaining + 1)
            if not raw:
                raise AdapterNotApplicable(
                    f"ledger ends before through_seq={through_seq}")
            bytes_read += len(raw)
            if bytes_read > MAX_LEDGER_BYTES:
                raise AdapterNotApplicable(
                    f"ledger prefix exceeds {MAX_LEDGER_BYTES} bytes")
            if not raw.strip():
                continue
            try:
                event = json.loads(
                    raw.decode("utf-8"),
                    object_pairs_hook=_object_without_duplicates,
                    parse_constant=_no_json_constants,
                )
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
                raise AdapterNotApplicable(
                    f"ledger prefix is not strict JSONL: {error}") from error
            if not isinstance(event, dict):
                raise AdapterNotApplicable("ledger event must be an object")
            expected_seq += 1
            seq = event.get("seq")
            if isinstance(seq, bool) or not isinstance(seq, int) \
                    or seq != expected_seq:
                raise AdapterNotApplicable(
                    f"ledger sequence is invalid at event {expected_seq}")
            if not isinstance(event.get("type"), str) or "payload" not in event:
                raise AdapterNotApplicable("ledger event has an invalid shape")
            yield event, bytes_read
    finally:
        if stream is not None:
            stream.close()
        elif fd >= 0:
            os.close(fd)


def _scan_ledger_tool_call_strings(
    invocation: AdapterInvocation,
    context: AdapterContext,
    *,
    pointer: str,
    through_seq: int,
    after_seq: int,
    tool_names: Any,
    payload_equals: Any,
    matches: Callable[[str], Optional[bool]],
    retain_limit: int,
    event_source=None,
) -> Dict[str, Any]:
    """Scan one bounded ledger range for the string-query adapters.

    Predicate interpretation and selector semantics deliberately remain in
    the versioned adapters.  This helper only validates the shared record
    scope and returns authenticated candidates plus bounded representatives.
    """
    for field, value in (("through_seq", through_seq),
                         ("after_seq", after_seq)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AdapterNotApplicable(
                f"{field} must be a non-negative integer")
    if after_seq > through_seq:
        raise AdapterNotApplicable("after_seq must not exceed through_seq")
    if through_seq > MAX_LEDGER_EVENTS:
        raise AdapterNotApplicable(
            f"through_seq exceeds {MAX_LEDGER_EVENTS} events")

    if tool_names is not None:
        if not isinstance(tool_names, list) or not tool_names \
                or len(tool_names) > MAX_LEDGER_TOOL_NAMES \
                or any(not isinstance(item, str) or not item
                       or len(item) > MAX_LEDGER_TOOL_NAME_CHARS
                       for item in tool_names) \
                or len(set(tool_names)) != len(tool_names):
            raise AdapterNotApplicable(
                "tool_names must be a bounded, non-empty unique string list")
        allowed_tools = frozenset(tool_names)
    else:
        allowed_tools = None

    if not isinstance(payload_equals, dict) \
            or len(payload_equals) > MAX_LEDGER_SCOPE_PREDICATES:
        raise AdapterNotApplicable(
            "payload_equals must be a bounded object")
    checked_scope = tuple(
        (_validate_ledger_pointer(scope_pointer, "payload_equals pointer"),
         canonical(scope_expected))
        for scope_pointer, scope_expected in payload_equals.items()
    )

    ref = "ledger:events"
    MethodAdapter.require_subject(
        invocation, ref, (PROVENANCE_LEDGER,))
    prefix_digests = []
    range_digests = []
    candidate_digests = []
    prefix_bytes = 0
    candidate_count = 0
    candidate_value_bytes = 0
    match_count = 0
    first_item = None
    last_item = None
    first_items = []
    matching_items = []
    differing_items = []

    events = (event_source if event_source is not None
              else _bounded_ledger_prefix(context.log, through_seq))
    for event, prefix_bytes in events:
        event_digest = EventLog.event_digest(event)
        prefix_digests.append(event_digest)
        if event["seq"] <= after_seq:
            continue
        range_digests.append(event_digest)
        if event["type"] != "tool_call":
            continue
        payload = event["payload"]
        if not isinstance(payload, dict):
            raise AdapterNotApplicable(
                "in-range tool_call payload must be an object")
        tool = payload.get("tool")
        if not isinstance(tool, str) or not tool \
                or len(tool) > MAX_LEDGER_TOOL_NAME_CHARS:
            raise AdapterNotApplicable(
                "in-range tool_call must retain a bounded tool name")
        if allowed_tools is not None and tool not in allowed_tools:
            continue

        in_scope = True
        for scope_pointer, canonical_expected in checked_scope:
            try:
                scope_actual = _json_pointer(payload, scope_pointer)
            except LookupError:
                in_scope = False
                break
            if canonical(scope_actual) != canonical_expected:
                in_scope = False
                break
        if not in_scope:
            continue

        candidate_digests.append(event_digest)
        try:
            value = _json_pointer(payload, pointer)
        except LookupError as error:
            raise AdapterNotApplicable(
                "an in-scope tool_call lacks value_pointer") from error
        if not isinstance(value, str):
            raise AdapterNotApplicable(
                "an in-scope tool_call value is not a string")
        value_bytes = len(value.encode("utf-8"))
        if value_bytes > MAX_LEDGER_VALUE_BYTES:
            raise AdapterNotApplicable(
                f"in-scope string exceeds {MAX_LEDGER_VALUE_BYTES} bytes")
        candidate_value_bytes += value_bytes
        if candidate_value_bytes > MAX_LEDGER_VALUE_BYTES_TOTAL:
            raise AdapterNotApplicable(
                "in-scope strings exceed the cumulative byte limit")

        item = {
            "seq": event["seq"],
            "tool": tool,
            "value": value,
            "event_digest": event_digest,
        }
        candidate_count += 1
        if first_item is None:
            first_item = item
        last_item = item
        if len(first_items) < retain_limit:
            first_items.append(item)
        matched = matches(value)
        if matched is True:
            match_count += 1
            if len(matching_items) < retain_limit:
                matching_items.append(item)
        elif matched is False and len(differing_items) < retain_limit:
            differing_items.append(item)

    snapshot = {
        "ref": ref,
        "after_seq": after_seq,
        "through_seq": through_seq,
        "prefix_bytes": prefix_bytes,
        "prefix_event_count": len(prefix_digests),
        "prefix_events_digest": content_hash(prefix_digests),
        "range_event_count": len(range_digests),
        "range_events_digest": content_hash(range_digests),
        "candidate_event_count": len(candidate_digests),
        "candidate_events_digest": content_hash(candidate_digests),
        "candidate_string_count": candidate_count,
        "candidate_string_bytes": candidate_value_bytes,
    }
    return {
        "ref": ref,
        "pointer": pointer,
        "through_seq": through_seq,
        "after_seq": after_seq,
        "tool_names": tool_names,
        "payload_equals": payload_equals,
        "candidate_count": candidate_count,
        "match_count": match_count,
        "first_item": first_item,
        "last_item": last_item,
        "first_items": first_items,
        "matching_items": matching_items,
        "differing_items": differing_items,
        "snapshot": snapshot,
    }


_LEDGER_PAYLOAD_POINTER_DESCRIPTION = (
    "RFC 6901 pointer rooted at event.payload, not the outer event: /key "
    "means event.payload['key']; an initial /payload is a real nested key, "
    "never an automatic wrapper prefix. Derive keys from the retained payload."
)


class LedgerToolCallStringAdapter(MethodAdapter):
    """Select exact string fields from scoped, immutable tool-call records."""

    name = "ledger.tool-call-string"
    basis = BASIS_RECORD
    fixed_inputs = (FixedAdapterInput("ledger:events", PROVENANCE_LEDGER),)
    summary = (
        "Within an explicit ledger sequence boundary, select a string at an "
        "RFC 6901 payload pointer from exactly scoped tool_call records and "
        "compare retained values exactly; never parses or executes the string."
    )
    arguments = {
        "selector": (
            "first|last|any|all over a non-empty selection of in-scope "
            "string values"
        ),
        "value_pointer": _LEDGER_PAYLOAD_POINTER_DESCRIPTION,
        "through_seq": "required inclusive ledger watermark, integer >= 0",
        "after_seq": "optional exclusive lower watermark, default 0",
        "expected": (
            "optional exact string; omission means existence and requires "
            "selector=any"
        ),
        "tool_names": "optional non-empty list of exact tool-name strings",
        "payload_equals": (
            "optional object from RFC 6901 pointers rooted at event.payload "
            "to exact JSON values"
        ),
    }

    def catalog_entry(self) -> Dict[str, Any]:
        entry = super().catalog_entry()
        entry["input_contract"] = (
            "declare ref=ledger:events, provenance=ledger, role=subject; "
            "the adapter consumes only (after_seq, through_seq]"
        )
        return entry

    @staticmethod
    def _retained(items: Sequence[Mapping[str, Any]]) -> Sequence[Dict[str, Any]]:
        return [
            {**item, "value": _bounded_json(item["value"])}
            for item in items[:MAX_RETAINED_LEDGER_VALUES]
        ]

    def execute(self, invocation, obligation, context):
        self.exact_arguments(
            invocation.arguments,
            ("selector", "value_pointer", "through_seq"),
            ("after_seq", "expected", "tool_names", "payload_equals"),
        )
        arguments = invocation.arguments
        selector = arguments["selector"]
        pointer = _validate_ledger_pointer(
            arguments["value_pointer"], "value_pointer")
        through_seq = arguments["through_seq"]
        after_seq = arguments.get("after_seq", 0)
        expected_present = "expected" in arguments
        expected = arguments.get("expected")
        tool_names = arguments.get("tool_names")
        payload_equals = arguments.get("payload_equals", {})

        if selector not in {"first", "last", "any", "all"}:
            raise AdapterNotApplicable(
                "invalid ledger.tool-call-string selector")
        if not expected_present and selector != "any":
            raise AdapterNotApplicable(
                "existence queries require selector=any")
        if expected_present and not isinstance(expected, str):
            raise AdapterNotApplicable(
                "ledger.tool-call-string expected must be a string")
        scan = _scan_ledger_tool_call_strings(
            invocation,
            context,
            pointer=pointer,
            through_seq=through_seq,
            after_seq=after_seq,
            tool_names=tool_names,
            payload_equals=payload_equals,
            matches=(lambda value: (
                value == expected if expected_present else None)),
            retain_limit=MAX_RETAINED_LEDGER_VALUES,
        )
        ref = scan["ref"]
        candidate_count = scan["candidate_count"]
        match_count = scan["match_count"]
        first_item = scan["first_item"]
        last_item = scan["last_item"]
        first_items = scan["first_items"]
        matching_items = scan["matching_items"]
        differing_items = scan["differing_items"]
        snapshot = scan["snapshot"]

        if selector == "first":
            selected_count = min(candidate_count, 1)
            selected_source = [first_item] if first_item is not None else []
        elif selector == "last":
            selected_count = min(candidate_count, 1)
            selected_source = [last_item] if last_item is not None else []
        elif selector == "any" and expected_present and match_count:
            selected_count = candidate_count
            selected_source = matching_items
        elif selector == "all" and expected_present \
                and match_count != candidate_count:
            selected_count = candidate_count
            selected_source = differing_items
        else:
            selected_count = candidate_count
            selected_source = first_items
        retained = self._retained(selected_source)
        observation = {
            "selector": selector,
            "value_pointer": pointer,
            "tool_names": tool_names,
            "payload_equals": payload_equals,
            "expected_present": expected_present,
            "expected": _bounded_json(expected) if expected_present else None,
            "selected": retained,
            "selected_count": selected_count,
            "selected_truncated": selected_count > len(retained),
            "snapshot": snapshot,
        }

        if candidate_count == 0:
            if obligation.refutable_by_absence:
                observation["matches"] = False
                return AdapterExecution(
                    OUTCOME_REFUTED,
                    {"reason": "no in-scope string record exists",
                     "selector": selector, "through_seq": through_seq},
                    WITNESS_ABSENCE, observation, (snapshot,))
            raise AdapterNotApplicable(
                "absence cannot settle this obligation; the agent executor "
                "must interpret the empty string selection")

        if not expected_present:
            matches = True
        elif selector == "first":
            matches = first_item["value"] == expected
        elif selector == "last":
            matches = last_item["value"] == expected
        elif selector == "any":
            matches = match_count > 0
        else:
            matches = match_count == candidate_count
        observation["matches"] = matches
        if matches:
            return AdapterExecution(
                OUTCOME_HOLDS, None, WITNESS_NONE,
                observation, (snapshot,))
        if selector == "any":
            if obligation.refutable_by_absence:
                return AdapterExecution(
                    OUTCOME_REFUTED,
                    {"reason": "no in-scope string has the exact value",
                     "expected": _bounded_json(expected),
                     "through_seq": through_seq},
                    WITNESS_ABSENCE, observation, (snapshot,))
            raise AdapterNotApplicable(
                "failure of an existential exact match is absence evidence; "
                "the obligation does not permit absence refutation")
        return AdapterExecution(
            OUTCOME_REFUTED,
            {"reason": "retained string selection contradicts the exact value",
             "selector": selector, "expected": _bounded_json(expected),
             "selected": retained},
            WITNESS_RECORD_CONTRADICTION, observation, (snapshot,))


class LedgerToolCallStringAdapterV2(LedgerToolCallStringAdapter):
    """Evaluate one explicit unary predicate over scoped ledger strings."""

    version = 2
    accepted_method_reuse = False
    summary = (
        "Within an explicit ledger sequence boundary, select strings from one "
        "exactly scoped set of tools and evaluate an explicit exists, equals, "
        "nonempty, or nonblank predicate. Values are never parsed or executed; "
        "the receipt retains a compact witness and digests the complete "
        "bounded prefix."
    )
    arguments = {
        "selector": (
            "first|last|any|all over a non-empty selection of in-scope "
            "string values; exists permits only any"
        ),
        "value_pointer": _LEDGER_PAYLOAD_POINTER_DESCRIPTION,
        "through_seq": "required inclusive ledger watermark, integer >= 0",
        "after_seq": "optional exclusive lower watermark, default 0",
        "tool_names": "optional non-empty list of exact tool-name strings",
        "payload_equals": (
            "optional object from RFC 6901 pointers rooted at event.payload "
            "to exact JSON values"
        ),
        "predicate": (
            "exists|equals|nonempty|nonblank; nonblank uses Unicode str.strip"
        ),
        "expected": "required exact string for equals; forbidden otherwise",
    }

    @staticmethod
    def _matches(predicate: str, value: str, expected: Any) -> bool:
        if predicate == "exists":
            return True
        if predicate == "equals":
            return value == expected
        if predicate == "nonempty":
            return len(value) > 0
        return bool(value.strip())

    @staticmethod
    def _validate_query(
        selector: Any,
        predicate: Any,
        expected_present: bool,
        expected: Any,
    ) -> None:
        if selector not in {"first", "last", "any", "all"}:
            raise AdapterNotApplicable(
                "invalid ledger.tool-call-string selector")
        if predicate not in {"exists", "equals", "nonempty", "nonblank"}:
            raise AdapterNotApplicable(
                "invalid ledger.tool-call-string predicate")
        if predicate == "exists" and selector != "any":
            raise AdapterNotApplicable(
                "exists queries require selector=any")
        if predicate == "equals":
            if not expected_present or not isinstance(expected, str):
                raise AdapterNotApplicable(
                    "equals requires expected to be a string")
        elif expected_present:
            raise AdapterNotApplicable(
                "expected is permitted only for predicate=equals")

    def _execute_scan(
        self,
        obligation: Obligation,
        scan: Mapping[str, Any],
        *,
        selector: str,
        predicate: str,
        expected_present: bool,
        expected: Any,
        scope: Optional[str] = None,
    ) -> AdapterExecution:
        """Apply the unchanged v2 predicate/selector algebra to one scan."""
        candidate_count = scan["candidate_count"]
        match_count = scan["match_count"]
        first_item = scan["first_item"]
        last_item = scan["last_item"]
        matching_items = scan["matching_items"]
        differing_items = scan["differing_items"]
        snapshot = scan["snapshot"]

        if candidate_count == 0:
            selector_domain_count = 0
            selected_source = []
            witness_selection = "none"
        elif selector == "first":
            selector_domain_count = 1
            selected_source = [first_item] if first_item is not None else []
            witness_selection = "first_candidate"
        elif selector == "last":
            selector_domain_count = 1
            selected_source = [last_item] if last_item is not None else []
            witness_selection = "last_candidate"
        elif selector == "any" and match_count:
            selector_domain_count = candidate_count
            selected_source = matching_items
            witness_selection = "first_match"
        elif selector == "all" and match_count != candidate_count:
            selector_domain_count = candidate_count
            selected_source = differing_items
            witness_selection = "first_counterexample"
        else:
            selector_domain_count = candidate_count
            selected_source = matching_items or differing_items
            witness_selection = (
                "first_match" if matching_items else "first_counterexample")

        retained = self._retained(selected_source[:1])
        observation = {
            "predicate": predicate,
            "selector": selector,
            "value_pointer": scan["pointer"],
            "tool_names": scan["tool_names"],
            "payload_equals": scan["payload_equals"],
            "expected_present": expected_present,
            "expected": _bounded_json(expected) if expected_present else None,
            "witness": {
                "predicate": predicate,
                "selection": witness_selection,
                "items": retained,
                "candidate_count": candidate_count,
                "selector_domain_count": selector_domain_count,
                "predicate_match_count": match_count,
            },
            "snapshot": snapshot,
        }
        if scope is not None:
            observation["scope"] = scope

        if candidate_count == 0:
            observation["matches"] = False
            if obligation.refutable_by_absence:
                witness = {
                    "reason": "no in-scope string record exists",
                    "predicate": predicate,
                    "selector": selector,
                    "through_seq": scan["through_seq"],
                }
                if scope is not None:
                    witness["scope"] = scope
                return AdapterExecution(
                    OUTCOME_REFUTED, witness, WITNESS_ABSENCE,
                    observation, (snapshot,))
            raise AdapterNotApplicable(
                "absence cannot settle this obligation; the agent executor "
                "must interpret the empty string selection")

        def predicate_matches(value: str) -> bool:
            return self._matches(predicate, value, expected)

        if selector == "first":
            matches = predicate_matches(first_item["value"])
        elif selector == "last":
            matches = predicate_matches(last_item["value"])
        elif selector == "any":
            matches = match_count > 0
        else:
            matches = match_count == candidate_count
        observation["matches"] = matches
        if matches:
            return AdapterExecution(
                OUTCOME_HOLDS, None, WITNESS_NONE,
                observation, (snapshot,))
        if selector == "any":
            if obligation.refutable_by_absence:
                witness = {
                    "reason": "no in-scope string satisfies the predicate",
                    "predicate": predicate,
                    "through_seq": scan["through_seq"],
                }
                if scope is not None:
                    witness["scope"] = scope
                if expected_present:
                    witness["expected"] = _bounded_json(expected)
                return AdapterExecution(
                    OUTCOME_REFUTED, witness, WITNESS_ABSENCE,
                    observation, (snapshot,))
            raise AdapterNotApplicable(
                "failure of an existential predicate is absence evidence; "
                "the obligation does not permit absence refutation")

        witness = {
            "reason": "retained string selection contradicts the predicate",
            "predicate": predicate,
            "selector": selector,
            "selected": retained,
        }
        if scope is not None:
            witness["scope"] = scope
        if expected_present:
            witness["expected"] = _bounded_json(expected)
        return AdapterExecution(
            OUTCOME_REFUTED, witness, WITNESS_RECORD_CONTRADICTION,
            observation, (snapshot,))

    def execute(self, invocation, obligation, context):
        self.exact_arguments(
            invocation.arguments,
            ("selector", "value_pointer", "through_seq", "predicate"),
            ("after_seq", "expected", "tool_names", "payload_equals"),
        )
        arguments = invocation.arguments
        selector = arguments["selector"]
        pointer = _validate_ledger_pointer(
            arguments["value_pointer"], "value_pointer")
        through_seq = arguments["through_seq"]
        after_seq = arguments.get("after_seq", 0)
        tool_names = arguments.get("tool_names")
        payload_equals = arguments.get("payload_equals", {})
        predicate = arguments["predicate"]
        expected_present = "expected" in arguments
        expected = arguments.get("expected")

        self._validate_query(
            selector, predicate, expected_present, expected)

        def predicate_matches(value: str) -> bool:
            return self._matches(predicate, value, expected)
        scan = _scan_ledger_tool_call_strings(
            invocation,
            context,
            pointer=pointer,
            through_seq=through_seq,
            after_seq=after_seq,
            tool_names=tool_names,
            payload_equals=payload_equals,
            matches=predicate_matches,
            retain_limit=1,
        )
        return self._execute_scan(
            obligation,
            scan,
            selector=selector,
            predicate=predicate,
            expected_present=expected_present,
            expected=expected,
        )


def _finite_ledger_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _ledger_file_identity(info: os.stat_result) -> Tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _open_ledger_parent(path: str) -> Tuple[int, str]:
    """Open every parent component without following a symlink."""
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise AdapterNotApplicable(
            "observed-eof ledger path must be normalized and absolute")
    name = os.path.basename(path)
    if not name or name in {".", ".."}:
        raise AdapterNotApplicable("observed-eof ledger name is invalid")
    flags = (os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
             | getattr(os, "O_CLOEXEC", 0))
    try:
        descriptor = os.open(os.path.sep, flags)
    except OSError as error:
        raise AdapterNotApplicable(
            f"cannot open the observed-eof filesystem root: {error}") from error
    try:
        parent = os.path.dirname(path)
        for component in parent.split(os.path.sep):
            if not component:
                continue
            if component in {".", ".."} or os.path.sep in component:
                raise AdapterNotApplicable(
                    "observed-eof ledger parent is invalid")
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise AdapterNotApplicable(
                    "cannot open an observed-eof ledger parent without "
                    f"following links: {error}") from error
            try:
                if not stat.S_ISDIR(os.fstat(child).st_mode):
                    raise AdapterNotApplicable(
                        "observed-eof ledger parent is not a directory")
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor, name
    except BaseException:
        os.close(descriptor)
        raise


def _read_stable_ledger_eof(log: EventLog) -> Dict[str, Any]:
    """Read and validate one complete bounded ledger observation.

    This establishes only the bytes present at the returned observation time.
    It does not freeze an online ledger or claim that no later append occurs.
    """
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    if any(not hasattr(os, name) for name in required_flags):
        raise AdapterNotApplicable(
            "observed-eof requires directory, nofollow, and nonblocking support")
    if (os.open not in getattr(os, "supports_dir_fd", ())
            or os.stat not in getattr(os, "supports_dir_fd", ())
            or os.stat not in getattr(os, "supports_follow_symlinks", ())):
        raise AdapterNotApplicable(
            "observed-eof requires relative nofollow stat support")
    observation_started_at_unix_ns = time.time_ns()
    observation_started_monotonic_ns = time.monotonic_ns()
    raw_path = os.fspath(log.path)
    if not isinstance(raw_path, str) or ".." in raw_path.split(os.path.sep):
        raise AdapterNotApplicable(
            "observed-eof ledger path must not contain '..' components")
    path = os.path.abspath(raw_path)
    parent_fd, name = _open_ledger_parent(path)
    try:
        expected = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        os.close(parent_fd)
        raise AdapterNotApplicable(
            f"cannot inspect the observed-eof ledger: {error}") from error
    if not stat.S_ISREG(expected.st_mode) or expected.st_nlink != 1:
        os.close(parent_fd)
        raise AdapterNotApplicable(
            "observed-eof ledger must be a private regular file")
    if expected.st_size > MAX_LEDGER_BYTES:
        os.close(parent_fd)
        raise AdapterNotApplicable(
            f"observed-eof ledger exceeds {MAX_LEDGER_BYTES} bytes")

    flags = (os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
             | getattr(os, "O_CLOEXEC", 0))
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        os.close(parent_fd)
        raise AdapterNotApplicable(
            f"cannot open the observed-eof ledger: {error}") from error
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or _ledger_file_identity(before)
                != _ledger_file_identity(expected)):
            raise AdapterNotApplicable(
                "observed-eof ledger changed before reading")
        chunks = []
        total = 0
        while True:
            try:
                chunk = os.read(
                    fd, min(1024 * 1024, MAX_LEDGER_BYTES + 1 - total))
            except OSError as error:
                raise AdapterNotApplicable(
                    f"cannot read the observed-eof ledger: {error}") from error
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_LEDGER_BYTES:
                raise AdapterNotApplicable(
                    f"observed-eof ledger exceeds {MAX_LEDGER_BYTES} bytes")
        raw = b"".join(chunks)
        if raw and not raw.endswith(b"\n"):
            raise AdapterNotApplicable(
                "observed-eof ledger has an incomplete final line")

        events = []
        event_digests = []
        bytes_read = 0
        for raw_line in raw.splitlines(keepends=True):
            bytes_read += len(raw_line)
            if not raw_line.strip():
                raise AdapterNotApplicable(
                    "observed-eof ledger contains a blank JSONL record")
            try:
                event = json.loads(
                    raw_line.decode("utf-8"),
                    object_pairs_hook=_object_without_duplicates,
                    parse_constant=_no_json_constants,
                    parse_float=_finite_ledger_float,
                )
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError,
                    RecursionError, OverflowError) as error:
                raise AdapterNotApplicable(
                    f"observed-eof ledger is not strict UTF-8 JSONL: {error}"
                ) from error
            if not isinstance(event, dict):
                raise AdapterNotApplicable(
                    "observed-eof ledger event must be an object")
            expected_seq = len(events) + 1
            if expected_seq > MAX_LEDGER_EVENTS:
                raise AdapterNotApplicable(
                    f"observed-eof ledger exceeds {MAX_LEDGER_EVENTS} events")
            seq = event.get("seq")
            if (isinstance(seq, bool) or not isinstance(seq, int)
                    or seq != expected_seq
                    or not isinstance(event.get("type"), str)
                    or "payload" not in event):
                raise AdapterNotApplicable(
                    f"observed-eof ledger has invalid event {expected_seq}")
            events.append((event, bytes_read))
            try:
                event_digests.append(EventLog.event_digest(event))
            except (TypeError, ValueError, RecursionError,
                    OverflowError) as error:
                raise AdapterNotApplicable(
                    "observed-eof ledger event cannot be digested"
                ) from error

        after = os.fstat(fd)
        try:
            current = os.stat(
                name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise AdapterNotApplicable(
                f"cannot revalidate the observed-eof ledger: {error}") from error
        identity = _ledger_file_identity(before)
        if (_ledger_file_identity(after) != identity
                or _ledger_file_identity(current) != identity
                or len(raw) != before.st_size):
            raise AdapterNotApplicable(
                "observed-eof ledger changed while it was read")

        rebound_fd, rebound_name = _open_ledger_parent(path)
        try:
            rebound_parent = os.fstat(rebound_fd)
            original_parent = os.fstat(parent_fd)
            rebound_file = os.stat(
                rebound_name, dir_fd=rebound_fd, follow_symlinks=False)
        except OSError as error:
            raise AdapterNotApplicable(
                f"cannot rebind the observed-eof ledger path: {error}") from error
        finally:
            os.close(rebound_fd)
        if ((rebound_parent.st_dev, rebound_parent.st_ino)
                != (original_parent.st_dev, original_parent.st_ino)
                or _ledger_file_identity(rebound_file) != identity):
            raise AdapterNotApplicable(
                "observed-eof ledger path changed while it was read")
        observation_completed_monotonic_ns = time.monotonic_ns()
        observation_completed_at_unix_ns = time.time_ns()
    finally:
        os.close(fd)
        os.close(parent_fd)

    return {
        "events": tuple(events),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "events_digest": content_hash(event_digests),
        "resolved_through_seq": len(events),
        "event_count": len(events),
        "byte_count": len(raw),
        "observation_started_at_unix_ns": observation_started_at_unix_ns,
        "observation_completed_at_unix_ns": observation_completed_at_unix_ns,
        "observation_started_monotonic_ns": observation_started_monotonic_ns,
        "observation_completed_monotonic_ns": (
            observation_completed_monotonic_ns),
        "monotonic_clock_scope": "process-local; not cross-process comparable",
        "file_identity": {
            "path": path,
            "device": before.st_dev,
            "inode": before.st_ino,
            "mode": before.st_mode,
            "links": before.st_nlink,
            "uid": before.st_uid,
            "gid": before.st_gid,
            "bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
        },
    }


class LedgerToolCallStringAdapterV3(LedgerToolCallStringAdapterV2):
    """Evaluate a v2 predicate over a prefix or stable observed EOF."""

    version = 3
    accepted_method_reuse = False
    summary = (
        "Evaluate the v2 string predicates over either an explicit ledger "
        "prefix or one complete, bounded, stable observed ledger file. An "
        "observed EOF is only a point-in-time observation and never proves "
        "future ledger closure; values are never parsed or executed."
    )
    arguments = {
        "scope": "prefix|observed-eof",
        "selector": (
            "first|last|any|all over a non-empty selection of in-scope "
            "string values; exists permits only any"
        ),
        "value_pointer": _LEDGER_PAYLOAD_POINTER_DESCRIPTION,
        "through_seq": (
            "required inclusive integer watermark for scope=prefix; "
            "forbidden for scope=observed-eof"
        ),
        "after_seq": "optional exclusive lower watermark, default 0",
        "tool_names": "optional non-empty list of exact tool-name strings",
        "payload_equals": (
            "optional object from RFC 6901 pointers rooted at event.payload "
            "to exact JSON values"
        ),
        "predicate": (
            "exists|equals|nonempty|nonblank; nonblank uses Unicode str.strip"
        ),
        "expected": "required exact string for equals; forbidden otherwise",
    }

    def catalog_entry(self) -> Dict[str, Any]:
        entry = super().catalog_entry()
        entry["input_contract"] = (
            "declare ref=ledger:events, provenance=ledger, role=subject; "
            "scope=prefix consumes only (after_seq, through_seq], while "
            "scope=observed-eof consumes the complete stable file visible at "
            "one observation time and does not prove future closure"
        )
        return entry

    def execute(self, invocation, obligation, context):
        self.exact_arguments(
            invocation.arguments,
            ("scope", "selector", "value_pointer", "predicate"),
            ("through_seq", "after_seq", "expected", "tool_names",
             "payload_equals"),
        )
        arguments = invocation.arguments
        scope = arguments["scope"]
        through_present = "through_seq" in arguments
        if scope not in {"prefix", "observed-eof"}:
            raise AdapterNotApplicable(
                "invalid ledger.tool-call-string scope")
        if scope == "prefix" and not through_present:
            raise AdapterNotApplicable(
                "scope=prefix requires through_seq")
        if scope == "observed-eof" and through_present:
            raise AdapterNotApplicable(
                "scope=observed-eof forbids through_seq")

        selector = arguments["selector"]
        pointer = _validate_ledger_pointer(
            arguments["value_pointer"], "value_pointer")
        after_seq = arguments.get("after_seq", 0)
        tool_names = arguments.get("tool_names")
        payload_equals = arguments.get("payload_equals", {})
        predicate = arguments["predicate"]
        expected_present = "expected" in arguments
        expected = arguments.get("expected")
        self._validate_query(
            selector, predicate, expected_present, expected)
        self.require_subject(
            invocation, "ledger:events", (PROVENANCE_LEDGER,))

        def predicate_matches(value: str) -> bool:
            return self._matches(predicate, value, expected)

        if scope == "observed-eof":
            eof = _read_stable_ledger_eof(context.log)
            through_seq = eof["resolved_through_seq"]
            event_source = eof["events"]
        else:
            eof = None
            through_seq = arguments["through_seq"]
            event_source = None
        scan = _scan_ledger_tool_call_strings(
            invocation,
            context,
            pointer=pointer,
            through_seq=through_seq,
            after_seq=after_seq,
            tool_names=tool_names,
            payload_equals=payload_equals,
            matches=predicate_matches,
            retain_limit=1,
            event_source=event_source,
        )
        snapshot = scan["snapshot"]
        snapshot["scope"] = scope
        if eof is not None:
            if (snapshot["prefix_event_count"] != eof["event_count"]
                    or snapshot["prefix_bytes"] != eof["byte_count"]
                    or snapshot["prefix_events_digest"]
                    != eof["events_digest"]):
                raise AdapterNotApplicable(
                    "observed-eof scan does not cover the stable ledger")
            snapshot["observed_eof"] = {
                key: eof[key] for key in (
                    "raw_sha256", "events_digest", "resolved_through_seq",
                    "event_count", "byte_count",
                    "observation_started_at_unix_ns",
                    "observation_completed_at_unix_ns",
                    "observation_started_monotonic_ns",
                    "observation_completed_monotonic_ns",
                    "monotonic_clock_scope", "file_identity",
                )
            }
            snapshot["observed_eof"].update({
                "complete_file_at_observation": True,
                "proves_future_closure": False,
                "scope_statement": (
                    "complete stable ledger file at observation time only; "
                    "later appends are outside this observation"
                ),
            })

        return self._execute_scan(
            obligation,
            scan,
            selector=selector,
            predicate=predicate,
            expected_present=expected_present,
            expected=expected,
            scope=scope,
        )


class BrowserHtmlAdapter(MethodAdapter):
    """Evaluate a closed assertion algebra using a verifier-owned browser."""

    name = "browser.html"
    basis = BASIS_WORLD
    summary = (
        "Compare observations of one standalone HTML document during "
        "navigation through page freeze after load+wait_ms in a fixed profile. "
        "External resource requests make the method inapplicable. "
        "The finite observation window proves no behavior outside that window."
    )
    arguments = {
        "address": "string",
        "assertions": (
            "1..16 objects: {kind:dialog.messages,expected:string[]}; "
            "{kind:console.messages,expected:string[],level?:string}; "
            "{kind:selector.count,selector:CSS string,expected:integer>=0}; "
            "{kind:selector.texts,selector:CSS string,expected:string[]}. "
            "Lists are exact ordered values observed within the window."
        ),
        "wait_ms": f"optional integer 0..{MAX_BROWSER_WAIT_MS}",
    }
    max_bytes = 8 * 1024 * 1024

    def __init__(self, observer: BrowserObserver):
        self.observer = observer

    def catalog_entry(self) -> Dict[str, Any]:
        entry = super().catalog_entry()
        entry["backend"] = dict(self.observer.identity)
        return entry

    @staticmethod
    def _assertion(item: Any) -> Dict[str, Any]:
        if not isinstance(item, dict) or not isinstance(item.get("kind"), str):
            raise AdapterNotApplicable("each browser assertion must be an object")
        kind = item["kind"]
        if kind == "dialog.messages":
            required, optional = {"kind", "expected"}, set()
        elif kind == "console.messages":
            required, optional = {"kind", "expected"}, {"level"}
        elif kind in {"selector.count", "selector.texts"}:
            required, optional = {"kind", "selector", "expected"}, set()
        else:
            raise AdapterNotApplicable(f"unknown browser assertion kind {kind!r}")
        if not required <= set(item) or not set(item) <= required | optional:
            raise AdapterNotApplicable("browser assertion has the wrong shape")
        expected = item["expected"]
        if kind in {"dialog.messages", "console.messages", "selector.texts"}:
            if not isinstance(expected, list) or any(
                    not isinstance(value, str) for value in expected):
                raise AdapterNotApplicable(
                    f"{kind} expected must be a string list")
        elif isinstance(expected, bool) or not isinstance(expected, int) \
                or expected < 0:
            raise AdapterNotApplicable(
                "selector.count expected must be a non-negative integer")
        if "selector" in item and (not isinstance(item["selector"], str)
                                   or not 1 <= len(item["selector"]) <= 4096):
            raise AdapterNotApplicable("browser selector must have 1..4096 characters")
        if "level" in item and (not isinstance(item["level"], str)
                                or not item["level"]):
            raise AdapterNotApplicable("console level must be non-empty")
        return dict(item)

    @staticmethod
    def _validated_observation(raw: Mapping[str, Any], digest: str,
                               selectors: Sequence[str], wait_ms: int,
                               backend: Mapping[str, Any]) -> Dict[str, Any]:
        response_fields = {
            "schema", "document_sha256", "network_policy", "engine",
            "engine_version", "dialogs", "console", "page_errors",
            "selectors", "document_policy", "blocked_requests",
            "browser_sha256", "observation_profile", "observation_window",
        }
        if not isinstance(raw, Mapping) or set(raw) != response_fields \
                or raw.get("schema") \
                != BROWSER_RESPONSE_SCHEMA:
            raise AdapterNotApplicable("browser observer returned the wrong schema")
        if raw.get("document_sha256") != digest:
            raise AdapterNotApplicable(
                "browser observer did not bind the observed document digest")
        if raw.get("network_policy") != BROWSER_NETWORK_POLICY:
            raise AdapterNotApplicable(
                "browser observer used a different request-blocking policy")
        if raw.get("document_policy") != BROWSER_DOCUMENT_POLICY:
            raise AdapterNotApplicable(
                "browser observer used a different document boundary")
        if raw.get("browser_sha256") != backend.get("browser_sha256") \
                or not isinstance(raw.get("browser_sha256"), str) \
                or not _SHA256.fullmatch(raw["browser_sha256"]):
            raise AdapterNotApplicable("browser runtime identity mismatch")
        if canonical(raw.get("observation_profile")) \
                != canonical(BROWSER_OBSERVATION_PROFILE):
            raise AdapterNotApplicable("browser observation profile mismatch")
        if raw.get("observation_window") != {
                "start": "navigation", "end": "page-frozen-after-load+wait",
                "wait_ms": wait_ms}:
            raise AdapterNotApplicable("browser observation window mismatch")
        if not isinstance(raw.get("engine"), str) \
                or not raw["engine"] \
                or not isinstance(raw.get("engine_version"), str) \
                or not raw["engine_version"]:
            raise AdapterNotApplicable("browser observer omitted engine identity")
        dialogs, console = raw.get("dialogs"), raw.get("console")
        page_errors, selected = raw.get("page_errors"), raw.get("selectors")
        blocked_requests = raw.get("blocked_requests")
        if not isinstance(dialogs, list) or any(
                not isinstance(item, dict) or set(item) != {"type", "message"}
                or not all(isinstance(item[key], str)
                           for key in ("type", "message"))
                for item in dialogs):
            raise AdapterNotApplicable("browser dialog observations are invalid")
        if not isinstance(console, list) or any(
                not isinstance(item, dict) or set(item) != {"type", "text"}
                or not all(isinstance(item[key], str)
                           for key in ("type", "text"))
                for item in console):
            raise AdapterNotApplicable("browser console observations are invalid")
        if not isinstance(page_errors, list) or any(
                not isinstance(item, str) for item in page_errors):
            raise AdapterNotApplicable("browser page errors are invalid")
        if not isinstance(blocked_requests, list) or any(
                not isinstance(item, dict)
                or set(item) != {"url", "resource_type"}
                or not isinstance(item["url"], str)
                or not isinstance(item["resource_type"], str)
                for item in blocked_requests):
            raise AdapterNotApplicable("browser blocked-request records are invalid")
        if blocked_requests:
            raise AdapterNotApplicable(
                "browser document requested undeclared external resources")
        if not isinstance(selected, dict) or set(selected) != set(selectors):
            raise AdapterNotApplicable("browser selector observations are incomplete")
        for value in selected.values():
            if not isinstance(value, dict) or set(value) != {
                    "count", "texts", "texts_complete"}:
                raise AdapterNotApplicable("browser selector result is invalid")
            if isinstance(value["count"], bool) \
                    or not isinstance(value["count"], int) \
                    or value["count"] < 0 \
                    or not isinstance(value["texts_complete"], bool) \
                    or not isinstance(value["texts"], list) \
                    or any(not isinstance(item, str) for item in value["texts"]):
                raise AdapterNotApplicable("browser selector result is invalid")
            if (len(value["texts"]) > value["count"]
                    or (value["texts_complete"]
                        and len(value["texts"]) != value["count"])):
                raise AdapterNotApplicable("browser selector count/text inconsistency")
        normalized = dict(raw)
        if len(canonical(normalized)) > MAX_RETAINED_VALUE_CHARS:
            raise AdapterNotApplicable("browser observation exceeds evidence limit")
        return normalized

    def execute(self, invocation, obligation, context):
        self.exact_arguments(invocation.arguments, ("address", "assertions"),
                             ("wait_ms",))
        address = invocation.arguments["address"]
        assertions = invocation.arguments["assertions"]
        wait_ms = invocation.arguments.get("wait_ms", 500)
        if not isinstance(address, str) or not address:
            raise AdapterNotApplicable("browser address must be non-empty")
        if not isinstance(assertions, list) or not assertions \
                or len(assertions) > MAX_BROWSER_ASSERTIONS:
            raise AdapterNotApplicable(
                f"browser assertions must contain 1..{MAX_BROWSER_ASSERTIONS} items")
        if isinstance(wait_ms, bool) or not isinstance(wait_ms, int) \
                or not 0 <= wait_ms <= MAX_BROWSER_WAIT_MS:
            raise AdapterNotApplicable(
                f"wait_ms must be an integer from 0 to {MAX_BROWSER_WAIT_MS}")
        checked = [self._assertion(item) for item in assertions]
        selectors = sorted({item["selector"] for item in checked
                            if "selector" in item})
        self.require_subject(invocation, address, WORLD_FILE_PROVENANCES)
        resolved = context.resolve(address)
        try:
            data, identity = _read_bounded_file(resolved, self.max_bytes)
        except AdapterNotApplicable:
            if not os.path.lexists(resolved.host_path) \
                    and obligation.refutable_by_absence:
                observation = {"address": address, "exists": False,
                               "world_mode": resolved.mode}
                return AdapterExecution(
                    OUTCOME_REFUTED,
                    {"reason": "required HTML document is absent",
                     "address": address},
                    WITNESS_ABSENCE, observation, ())
            raise

        request = {"wait_ms": wait_ms, "selectors": selectors}
        try:
            raw = self.observer.observe(data, request)
        except Exception as error:
            raise AdapterNotApplicable(
                f"browser observer could not execute: {error}") from error
        browser = self._validated_observation(
            raw, identity["sha256"], selectors, wait_ms,
            self.observer.identity)
        results = []
        for assertion in checked:
            kind = assertion["kind"]
            if kind == "dialog.messages":
                actual = [item["message"] for item in browser["dialogs"]]
            elif kind == "console.messages":
                level = assertion.get("level")
                actual = [item["text"] for item in browser["console"]
                          if level is None or item["type"] == level]
            elif kind == "selector.count":
                actual = browser["selectors"][assertion["selector"]]["count"]
            else:
                value = browser["selectors"][assertion["selector"]]
                if not value["texts_complete"]:
                    raise AdapterNotApplicable(
                        "browser selector text observation was truncated")
                actual = value["texts"]
            results.append({
                "assertion": assertion,
                "actual": actual,
                "matches": canonical(actual) == canonical(assertion["expected"]),
            })
        matches = all(item["matches"] for item in results)
        observation = {
            "address": address,
            "identity": identity,
            "backend": dict(self.observer.identity),
            "browser": browser,
            "assertions": results,
            "matches": matches,
        }
        if matches:
            return AdapterExecution(
                OUTCOME_HOLDS, None, WITNESS_NONE,
                observation, (identity,))
        failed = [item for item in results if not item["matches"]]
        positive = []
        for item in failed:
            actual, expected = item["actual"], item["assertion"]["expected"]
            if isinstance(actual, int):
                # Extra observed nodes are a positive counter-observation;
                # fewer nodes prove only the absence of a required member.
                is_positive = actual > expected
            else:
                # A proper prefix is silence about the missing suffix. A
                # different observed member or extra event is positive.
                is_positive = len(actual) > len(expected) or any(
                    found != required for found, required in zip(actual, expected))
            if is_positive:
                positive.append(item)
        if not positive:
            if not obligation.refutable_by_absence:
                raise AdapterNotApplicable(
                    "missing browser observations cannot refute this obligation")
            return AdapterExecution(
                OUTCOME_REFUTED,
                {"reason": "required browser observations are absent within the window",
                 "failed": failed},
                WITNESS_ABSENCE, observation, (identity,))
        return AdapterExecution(
            OUTCOME_REFUTED,
            {"reason": "browser observations contradict an exact assertion",
             "failed": positive},
            WITNESS_EXECUTION, observation, (identity,))


class AdapterRegistry:
    """Versioned registry and evidence-envelope builder."""

    def __init__(self, adapters: Iterable[MethodAdapter] = ()):
        self._adapters: Dict[Tuple[str, int], MethodAdapter] = {}
        self._fixed_inputs: Dict[Tuple[str, int], Tuple[FixedAdapterInput, ...]] = {}
        for adapter in adapters:
            key = (adapter.name, adapter.version)
            if key in self._adapters:
                raise ValueError(f"duplicate adapter {adapter.identity}")
            if not _NAME.fullmatch(adapter.name):
                raise ValueError(f"invalid registered adapter name {adapter.name!r}")
            self._adapters[key] = adapter
            # Snapshot this new contract, including for extension adapters.
            # Catalog identity and materialization must not diverge if a
            # class-level descriptor tuple is replaced after registration.
            if any(type(item) is not FixedAdapterInput for item in adapter.fixed_inputs):
                raise ValueError("fixed inputs must be immutable descriptors")
            self._fixed_inputs[key] = tuple(FixedAdapterInput(
                item.ref, item.provenance, item.role, item.integer_argument)
                for item in adapter.fixed_inputs)

    def catalog(self) -> str:
        entries = []
        for key, adapter in sorted(self._adapters.items()):
            entry = adapter.catalog_entry()
            entry.pop("fixed_input_binding", None)
            if self._fixed_inputs[key]:
                entry["fixed_input_binding"] = {
                    "mode": FIXED_INPUT_BINDING,
                    "inputs": [item.public() for item in self._fixed_inputs[key]],
                }
            entries.append(entry)
        return json.dumps(entries, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))

    def bind_inputs(self, method: Mapping[str, Any]) -> Dict[str, Any]:
        """Lower an explicitly opted-in method without repairing model claims.

        No source is opened. Conflicting declarations, unknown instruments,
        and missing semantic arguments remain errors for ordinary fallback.
        """
        if "adapter_input_binding" not in method:
            return dict(method)
        invocation = AdapterInvocation.from_method(method)
        key = (invocation.name, invocation.version)
        adapter = self._adapters.get(key)
        descriptors = self._fixed_inputs.get(key, ())
        if adapter is None or not descriptors:
            raise AdapterNotApplicable("adapter has no verifier-owned fixed inputs")
        MethodAdapter.validate(adapter, invocation, method)
        fixed = [item.resolve(invocation.arguments) for item in descriptors]
        if len({item["ref"] for item in fixed}) != len(fixed):
            raise AdapterNotApplicable("fixed input descriptors have duplicate references")
        inputs = list(invocation.inputs)
        for item in inputs:
            same_ref = [slot for slot in fixed if slot["ref"] == item["ref"]]
            if same_ref and item != same_ref[0]:
                raise AdapterNotApplicable("declared input conflicts with a fixed slot")
            if item["role"] == ROLE_SUBJECT and item not in fixed:
                raise AdapterNotApplicable("undeclared subject in fixed-input method")
        for slot in fixed:
            if inputs.count(slot) > 1:
                raise AdapterNotApplicable("duplicate fixed input declaration")
            if slot not in inputs:
                inputs.append(slot)
        if len(inputs) > MAX_DECLARED_INPUTS:
            raise AdapterNotApplicable("materialized adapter_inputs exceeds entry limit")
        bound = {**method, "adapter_inputs": inputs}
        # Extension validators may require their subject slot. Validate the
        # complete method, not the deliberately unmaterialized model output.
        adapter.validate(AdapterInvocation.from_method(bound), bound)
        return bound

    def _require_bound_inputs(self, method):
        if "adapter_input_binding" in method and self.bind_inputs(method) != method:
            raise AdapterNotApplicable(
                "fixed inputs must be materialized before execution and judgment")

    def supports_method_reuse(self, method: Mapping[str, Any]) -> bool:
        if "adapter_input_binding" in method:
            return False  # This new binding protocol requires fresh review.
        try:
            invocation = AdapterInvocation.from_method(method)
            if invocation is None:
                return False
            adapter = self._adapters.get((invocation.name, invocation.version))
            return bool(adapter and adapter.accepted_method_reuse)
        except (AdapterNotApplicable, TypeError, ValueError):
            return False

    def execute(self, method: Mapping[str, Any], row: Mapping[str, Any],
                obligation: Obligation, context: AdapterContext,
                artifacts: ArtifactStore) -> Optional[DischargeResult]:
        self._require_bound_inputs(method)
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
            self._require_bound_inputs(method)
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


def default_adapter_registry(
    browser_observer: Optional[BrowserObserver] = None,
) -> AdapterRegistry:
    if browser_observer is None:
        browser_observer = configured_browser_observer()
    adapters = [
        FileExistsAdapter(),
        FileSha256Adapter(),
        JsonValueAdapter(),
        JsonPredicateAdapter(),
        JsonPredicateAdapterV2(),
        TextValueAdapter(),
        LedgerEventAdapter(),
        LedgerToolCallStringAdapter(),
        LedgerToolCallStringAdapterV2(),
        LedgerToolCallStringAdapterV3(),
    ]
    if browser_observer is not None:
        adapters.append(BrowserHtmlAdapter(browser_observer))
    return AdapterRegistry(adapters)


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

    def prepare_method(self, method):
        try:
            return self.registry.bind_inputs(method)
        except AdapterNotApplicable:
            # Execution repeats the mechanical check and routes failures to
            # the existing agent executor with its diagnostic, not a verdict.
            return method

    def supports_method_reuse(self, method) -> bool:
        return self.registry.supports_method_reuse(method)

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
            "accepted_method_reuse_attempt": True,
        })
        return result, ""

    def discharge_with_repair(self, row, obligation, method=None, tag="", *,
                              previous_artifact=None):
        # Repair is an explicit harness capability, not adapter JSON. A fresh
        # agent execution still follows; do not rerun the deterministic adapter
        # or import any previous result as an accepted outcome.
        if getattr(self.discharge, "__func__", None) \
                is not AdapterRoutingDischarger.discharge:
            return self.discharge(row, obligation, method=method, tag=tag)
        self._fallback_event(row, obligation, method or {},
                             "judger redo requires the agent executor")
        repair = getattr(self.fallback, "discharge_with_repair", None)
        if callable(repair):
            return repair(row, obligation, method=method, tag=tag,
                          previous_artifact=previous_artifact)
        return self.fallback.discharge(row, obligation, method=method, tag=tag)

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
            **({"adapter_input_binding": method["adapter_input_binding"]}
               if "adapter_input_binding" in method else {}),
        })
        return result

    def _fallback_event(self, row, obligation, method, reason):
        if (not any(field in method for field in _ADAPTER_FIELDS)
                and "adapter_input_binding" not in method):
            return
        self.log.append("adapter_fallback", {
            "claim": row["id"], "obligation": obligation.id,
            "adapter": str(method.get("adapter", ""))[:80],
            "reason": reason[:500],
            **({"adapter_input_binding": method["adapter_input_binding"]}
               if "adapter_input_binding" in method else {}),
        })
