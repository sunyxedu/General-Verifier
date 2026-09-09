"""A bounded, pure evidence-program adapter.

The Oracle may describe a task-specific *method*, but this interpreter is a
task-independent verifier instrument.  It parses a deliberately small Python-
shaped language and evaluates the syntax itself.  It never calls ``eval`` or
``exec`` and exposes no filesystem, process, network, clock, randomness,
imports, attributes, or ambient environment.

Version 1 is holds-direction only and routes every false requirement or error
to the existing agent executor. Version 2 reads a stable, complete, bounded
vector of private regular-file snapshots. A fully validated false requirement
then fails only that exact method; it never refutes the obligation. Malformed
programs, runtime errors, races, and resource boundaries still fall back. A
successful run in either version remains evidence for the fresh independent
Judger, not a verifier verdict shortcut.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import stat
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from ..core.model import (
    BASIS_WORLD,
    OUTCOME_FAILED,
    OUTCOME_HOLDS,
    ROLE_AUTHORITY,
    ROLE_SUBJECT,
    WITNESS_NONE,
    canonical,
    content_hash,
)
from ..core.world import load_world_view
from .adapters import (
    WORLD_FILE_PROVENANCES,
    AdapterExecution,
    AdapterInvocation,
    AdapterNotApplicable,
    MethodAdapter,
    _read_bounded_file,
    _strict_json_document,
    _validate_json_tree,
)


LANGUAGE = "gv-pure-evidence-program-v1"
MAX_SOURCE_CHARS = 8_192
MAX_AST_NODES = 2_048
MAX_AST_DEPTH = 64
MAX_INPUT_FILES = 8
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 32 * 1024 * 1024
MAX_FUEL = 250_000
MAX_LOOP_ITEMS = 100_000
MAX_VALUE_ITEMS = 100_000
MAX_VALUE_BYTES = 8 * 1024 * 1024
MAX_CUMULATIVE_ALLOCATION_BYTES = 128 * 1024 * 1024
MAX_INTEGER_BITS = 4_096
MAX_INTEGER_DECIMAL_CHARS = 1_236
MAX_CALL_DEPTH = 8
MAX_ASSERTIONS = 64
MAX_OBSERVATIONS = 32
MAX_RETAINED_OBSERVATION_CHARS = 4_096
MAX_RECEIPT_BYTES = 48_000
COMPLETE_AUDIT_OBSERVATION_SCHEMA = (
    "gv-pure-evidence-program-complete-audit-v1"
)
WORLD_INPUT_SNAPSHOT_SCHEMA = "gv-world-input-vector-snapshot-v1"

_BUILTINS = frozenset({
    "input_bytes", "input_text", "input_json", "len", "range",
    "be_uint", "be_int", "le_uint", "le_int", "be_float64",
    "le_float64", "utf8", "sha256", "keys", "is_type", "require",
    "observe",
})

_ALLOWED_AST_TYPES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg,
    ast.Assign, ast.AugAssign, ast.Expr, ast.If, ast.For, ast.Return,
    ast.Break, ast.Continue, ast.Pass,
    ast.Constant, ast.Name, ast.List, ast.Tuple, ast.Dict, ast.UnaryOp,
    ast.BinOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Subscript, ast.Call,
    ast.Slice,
    ast.Load, ast.Store,
    ast.Not, ast.UAdd, ast.USub, ast.Invert,
    ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div, ast.Mod,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
)


class _Return(Exception):
    def __init__(self, value: Any, taint: set[int]):
        self.value = value
        self.taint = taint


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _RequirementFalse(Exception):
    """Internal control flow for one fully validated false requirement."""

    def __init__(self, assertion_index: int):
        self.assertion_index = assertion_index


@dataclass(frozen=True)
class _Function:
    name: str
    parameters: tuple[str, ...]
    body: tuple[ast.stmt, ...]


def _source_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _node_digest(node: ast.AST) -> str:
    return hashlib.sha256(
        ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _checked_int(value: int) -> int:
    if type(value) is not int or value.bit_length() > MAX_INTEGER_BITS:
        raise AdapterNotApplicable("evidence program integer exceeds its bound")
    return value


def _tree_shape(root: ast.AST) -> None:
    stack = [(root, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_AST_NODES or depth > MAX_AST_DEPTH:
            raise AdapterNotApplicable("evidence program AST exceeds its bound")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))


def _validate_ast(root: ast.Module) -> None:
    """Reject unsupported syntax everywhere, including unreachable code."""
    functions = [node.name for node in root.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != len(set(functions)):
        raise AdapterNotApplicable("evidence program function identity is invalid")
    callable_names = _BUILTINS | frozenset(functions)
    for node in ast.walk(root):
        if not isinstance(node, _ALLOWED_AST_TYPES):
            raise AdapterNotApplicable(
                f"evidence program syntax {type(node).__name__} is unsupported")
        if isinstance(node, ast.FunctionDef):
            arguments = node.args
            if (node.decorator_list or node.returns is not None
                    or node.type_comment is not None
                    or getattr(node, "type_params", ())
                    or arguments.posonlyargs or arguments.kwonlyargs
                    or arguments.kw_defaults or arguments.defaults
                    or arguments.vararg or arguments.kwarg
                    or any(argument.annotation is not None
                           or getattr(argument, "type_comment", None) is not None
                           for argument in arguments.args)):
                raise AdapterNotApplicable(
                    "evidence program function shape is unsupported")
        if isinstance(node, ast.Assign) and len(node.targets) != 1:
            raise AdapterNotApplicable(
                "evidence program assignment must have one target")
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.For)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if not isinstance(target, ast.Name):
                raise AdapterNotApplicable(
                    "evidence program assignment target is invalid")
            if (not target.id.isascii() or not target.id.isidentifier()
                    or target.id.startswith("_") or target.id in _BUILTINS):
                raise AdapterNotApplicable(
                    "evidence program assignment target is invalid")
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Call):
            raise AdapterNotApplicable(
                "evidence program expression statement must be a call")
        if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name) or node.keywords):
            raise AdapterNotApplicable(
                "evidence program calls must be positional named calls")
        if isinstance(node, ast.Call) and node.func.id not in callable_names:
            raise AdapterNotApplicable(
                "evidence program call target is unsupported")
        if isinstance(node, ast.Name) and (
                not node.id.isascii() or not node.id.isidentifier()
                or node.id.startswith("_")):
            raise AdapterNotApplicable("evidence program name is invalid")
        if isinstance(node, ast.Dict) and any(key is None for key in node.keys):
            raise AdapterNotApplicable(
                "evidence program dictionary unpacking is unsupported")

    def statements(nodes: Sequence[ast.stmt], *, in_function: bool,
                   in_loop: bool, top_level: bool) -> None:
        for node in nodes:
            if isinstance(node, ast.FunctionDef):
                if not top_level:
                    raise AdapterNotApplicable(
                        "evidence program functions must be defined at top level")
                statements(node.body, in_function=True, in_loop=False, top_level=False)
            elif isinstance(node, ast.If):
                statements(node.body, in_function=in_function,
                           in_loop=in_loop, top_level=False)
                statements(node.orelse, in_function=in_function,
                           in_loop=in_loop, top_level=False)
            elif isinstance(node, ast.For):
                statements(node.body, in_function=in_function,
                           in_loop=True, top_level=False)
                statements(node.orelse, in_function=in_function,
                           in_loop=in_loop, top_level=False)
            elif isinstance(node, ast.Return) and not in_function:
                raise AdapterNotApplicable(
                    "evidence program return is outside a function")
            elif isinstance(node, (ast.Break, ast.Continue)) and not in_loop:
                raise AdapterNotApplicable(
                    "evidence program loop control is outside a loop")

    statements(root.body, in_function=False, in_loop=False, top_level=True)


def _strict_equal(left: Any, right: Any) -> bool:
    """Structural equality without Python's bool/int/float coercions."""
    pending = [(left, right)]
    while pending:
        first, second = pending.pop()
        if type(first) is not type(second):
            return False
        if type(first) in (list, tuple):
            if len(first) != len(second):
                return False
            pending.extend(zip(first, second))
        elif type(first) is dict:
            if (len(first) != len(second)
                    or any(key not in second for key in first)):
                return False
            pending.extend((first[key], second[key]) for key in first)
        elif first != second:
            return False
    return True


def _storage_reservation(value: Any) -> int:
    """Conservative retained-size estimate for one acyclic VM value."""
    total = 0
    stack = [value]
    while stack:
        item = stack.pop()
        if item is None or type(item) in (bool, float):
            total += 32
        elif type(item) is int:
            total += 32 + max(1, item.bit_length() // 8)
        elif type(item) is str:
            total += 64 + len(item) * 4
        elif type(item) is bytes:
            total += 64 + len(item)
        elif type(item) in (list, tuple):
            total += 64 + len(item) * 16
            stack.extend(item)
        elif type(item) is dict:
            total += 128 + len(item) * 64
            stack.extend(item.keys())
            stack.extend(item.values())
        elif type(item) is range:
            total += 64
        else:
            raise AdapterNotApplicable(
                "evidence program produced an unsupported value")
        if total > MAX_CUMULATIVE_ALLOCATION_BYTES:
            return total
    return total


def _json_decoder_preflight(data: bytes) -> int:
    """Bound JSON tree size and decoder expansion before ``json.loads``."""
    in_string = False
    escaped = False
    depth = 0
    opens = 0
    commas = 0
    colons = 0
    for byte in data:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x5B, 0x7B):  # [ {
            opens += 1
            depth += 1
            if depth > MAX_AST_DEPTH:
                raise AdapterNotApplicable(
                    "evidence program JSON nesting exceeds its bound")
        elif byte in (0x5D, 0x7D):  # ] }
            depth -= 1
        elif byte == 0x2C:  # comma
            commas += 1
        elif byte == 0x3A:  # colon
            colons += 1
    # For a syntactically valid tree, each non-container child after the
    # first is introduced by a comma; each container is introduced by an
    # opening delimiter. Empty containers only make this estimate larger.
    estimated_nodes = 1 + opens + commas + colons
    if estimated_nodes > MAX_VALUE_ITEMS:
        raise AdapterNotApplicable(
            "evidence program JSON node count exceeds its bound")
    return max(1, len(data)) * 4 + estimated_nodes * 128


def _json_safe(value: Any) -> Any:
    """Validate exact built-in values and return a canonical JSON value.

    Bytes are represented by a digest/length receipt instead of being copied
    into model-visible evidence.  The VM may inspect complete input bytes, but
    observations remain small and inert.
    """
    nodes = 0

    def visit(item: Any, depth: int) -> Any:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_VALUE_ITEMS or depth > MAX_AST_DEPTH:
            raise AdapterNotApplicable("evidence program value exceeds its bound")
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            return _checked_int(item)
        if type(item) is float:
            if not math.isfinite(item):
                raise AdapterNotApplicable("evidence program produced a non-finite number")
            return item
        if type(item) is str:
            raw = item.encode("utf-8", errors="strict")
            if len(raw) > MAX_VALUE_BYTES:
                raise AdapterNotApplicable("evidence program string exceeds its bound")
            return item
        if type(item) is bytes:
            if len(item) > MAX_VALUE_BYTES:
                raise AdapterNotApplicable("evidence program bytes exceed their bound")
            return {
                "bytes": len(item),
                "sha256": hashlib.sha256(item).hexdigest(),
            }
        if type(item) in (list, tuple):
            if len(item) > MAX_VALUE_ITEMS:
                raise AdapterNotApplicable("evidence program collection exceeds its bound")
            return [visit(child, depth + 1) for child in item]
        if type(item) is dict:
            if len(item) > MAX_VALUE_ITEMS or any(type(key) is not str for key in item):
                raise AdapterNotApplicable("evidence program object has invalid keys or size")
            return {key: visit(child, depth + 1) for key, child in item.items()}
        if type(item) is range:
            if len(item) > MAX_LOOP_ITEMS:
                raise AdapterNotApplicable("evidence program range exceeds its bound")
            return list(item)
        raise AdapterNotApplicable("evidence program produced an unsupported value")

    return visit(value, 0)


def _validate_value(value: Any) -> None:
    """Validate a VM value without constructing a mirrored JSON tree."""
    nodes = 0
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_VALUE_ITEMS or depth > MAX_AST_DEPTH:
            raise AdapterNotApplicable("evidence program value exceeds its bound")
        if item is None or type(item) is bool:
            continue
        if type(item) is int:
            _checked_int(item)
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise AdapterNotApplicable(
                    "evidence program produced a non-finite number")
            continue
        if type(item) is str:
            try:
                raw = item.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise AdapterNotApplicable(
                    "evidence program string is invalid Unicode") from error
            if len(raw) > MAX_VALUE_BYTES:
                raise AdapterNotApplicable(
                    "evidence program string exceeds its bound")
            continue
        if type(item) is bytes:
            if len(item) > MAX_VALUE_BYTES:
                raise AdapterNotApplicable(
                    "evidence program bytes exceed their bound")
            continue
        if type(item) in (list, tuple):
            if len(item) > MAX_VALUE_ITEMS:
                raise AdapterNotApplicable(
                    "evidence program collection exceeds its bound")
            stack.extend((child, depth + 1) for child in item)
            continue
        if type(item) is dict:
            if len(item) > MAX_VALUE_ITEMS:
                raise AdapterNotApplicable(
                    "evidence program object exceeds its bound")
            for key, child in item.items():
                if type(key) is not str:
                    raise AdapterNotApplicable(
                        "evidence program object has invalid keys")
                try:
                    key.encode("utf-8", errors="strict")
                except UnicodeEncodeError as error:
                    raise AdapterNotApplicable(
                        "evidence program object key is invalid Unicode") from error
                stack.append((child, depth + 1))
            continue
        if type(item) is range:
            if len(item) > MAX_LOOP_ITEMS:
                raise AdapterNotApplicable(
                    "evidence program range exceeds its bound")
            continue
        raise AdapterNotApplicable(
            "evidence program produced an unsupported value")


def _retained_json_upper_bound(value: Any) -> int:
    """Upper-bound canonical JSON chars without serializing the value."""
    total = 0
    stack = [value]
    while stack:
        item = stack.pop()
        if item is None:
            total += 4
        elif type(item) is bool:
            total += 5
        elif type(item) is int:
            total += len(str(item))
        elif type(item) is float:
            total += 32
        elif type(item) is str:
            total += 2 + len(item) * 6
        elif type(item) is bytes:
            total += 128
        elif type(item) in (list, tuple):
            total += 2 + max(0, len(item) - 1)
            stack.extend(item)
        elif type(item) is dict:
            total += 2 + max(0, len(item) - 1)
            for key, child in item.items():
                total += 3 + len(key) * 6
                stack.append(child)
        elif type(item) is range:
            # Retaining a range would materialize every integer.
            total += (2 + max(0, len(item) - 1)
                      + len(item) * MAX_INTEGER_DECIMAL_CHARS)
        else:
            raise AdapterNotApplicable(
                "evidence program produced an unsupported value")
        if total > MAX_RETAINED_OBSERVATION_CHARS:
            return total
    return total


def _snapshot_stat_identity(info: os.stat_result) -> tuple[int, ...]:
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


def _open_snapshot_parent(path: str) -> tuple[int, str]:
    """Open an absolute parent path one no-follow component at a time."""
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    if any(not hasattr(os, name) for name in required_flags):
        raise AdapterNotApplicable(
            "complete world snapshots require directory, nofollow, and "
            "nonblocking support")
    if (os.open not in getattr(os, "supports_dir_fd", ())
            or os.stat not in getattr(os, "supports_dir_fd", ())
            or os.stat not in getattr(os, "supports_follow_symlinks", ())):
        raise AdapterNotApplicable(
            "complete world snapshots require relative nofollow stat support")
    if not isinstance(path, str) or not os.path.isabs(path) \
            or os.path.normpath(path) != path:
        raise AdapterNotApplicable(
            "complete world snapshot paths must be normalized and absolute")
    name = os.path.basename(path)
    if not name or name in {".", ".."}:
        raise AdapterNotApplicable("complete world snapshot file name is invalid")
    flags = (os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
             | getattr(os, "O_CLOEXEC", 0))
    try:
        descriptor = os.open(os.path.sep, flags)
    except OSError as error:
        raise AdapterNotApplicable(
            f"cannot open the world filesystem root: {error}") from error
    try:
        for component in os.path.dirname(path).split(os.path.sep):
            if not component:
                continue
            if component in {".", ".."} or os.path.sep in component:
                raise AdapterNotApplicable(
                    "complete world snapshot parent is invalid")
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise AdapterNotApplicable(
                    "cannot open a world input parent without following "
                    f"links: {error}") from error
            try:
                if not stat.S_ISDIR(os.fstat(child).st_mode):
                    raise AdapterNotApplicable(
                        "complete world snapshot parent is not a directory")
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor, name
    except BaseException:
        os.close(descriptor)
        raise


@dataclass
class _OpenWorldInput:
    resolved: Any
    path: str
    name: str
    parent_fd: int
    fd: int
    before: os.stat_result
    data: bytes
    identity: Dict[str, Any]


def _snapshot_input_identity(resolved, info: os.stat_result,
                             data: bytes) -> Dict[str, Any]:
    return {
        "ref": resolved.address,
        "mode": resolved.mode,
        "scheme": resolved.scheme,
        "kind": "file",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "file_identity": {
            "device": info.st_dev,
            "inode": info.st_ino,
            "file_mode": info.st_mode,
            "links": info.st_nlink,
            "uid": info.st_uid,
            "gid": info.st_gid,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        },
    }


def _revalidate_open_world_input(item: _OpenWorldInput) -> None:
    expected = _snapshot_stat_identity(item.before)
    try:
        after = os.fstat(item.fd)
        current = os.stat(
            item.name, dir_fd=item.parent_fd, follow_symlinks=False)
    except OSError as error:
        raise AdapterNotApplicable(
            f"cannot revalidate world input {item.resolved.address!r}: "
            f"{error}") from error
    if (_snapshot_stat_identity(after) != expected
            or _snapshot_stat_identity(current) != expected
            or len(item.data) != item.before.st_size):
        raise AdapterNotApplicable(
            f"world input {item.resolved.address!r} changed during snapshot")

    rebound_fd, rebound_name = _open_snapshot_parent(item.path)
    try:
        rebound_parent = os.fstat(rebound_fd)
        original_parent = os.fstat(item.parent_fd)
        rebound_file = os.stat(
            rebound_name, dir_fd=rebound_fd, follow_symlinks=False)
    except OSError as error:
        raise AdapterNotApplicable(
            f"cannot rebind world input {item.resolved.address!r}: "
            f"{error}") from error
    finally:
        os.close(rebound_fd)
    if ((rebound_parent.st_dev, rebound_parent.st_ino)
            != (original_parent.st_dev, original_parent.st_ino)
            or _snapshot_stat_identity(rebound_file) != expected):
        raise AdapterNotApplicable(
            f"world input {item.resolved.address!r} path changed during snapshot")


@contextmanager
def _stable_world_input_vector(context, execution_inputs):
    """Yield complete private bytes for one common stable observation window."""
    try:
        cached_world = canonical(context.world.describe())
        current_world = canonical(load_world_view(
            context.state_dir, live_world=context.live_world).describe())
    except (OSError, TypeError, ValueError, UnicodeError,
            RecursionError, MemoryError, OverflowError) as error:
        raise AdapterNotApplicable(
            f"cannot bind the current world view: {error}") from error
    if current_world != cached_world:
        raise AdapterNotApplicable(
            "world view changed before the complete input snapshot")

    opened: list[_OpenWorldInput] = []
    total = 0
    physical_identities: set[tuple[int, int]] = set()
    try:
        for item in execution_inputs:
            resolved = context.resolve(item["ref"])
            path = os.path.abspath(resolved.host_path)
            parent_fd, name = _open_snapshot_parent(path)
            fd = -1
            try:
                expected = os.stat(
                    name, dir_fd=parent_fd, follow_symlinks=False)
                if not stat.S_ISREG(expected.st_mode) or expected.st_nlink != 1:
                    raise AdapterNotApplicable(
                        "complete world inputs must be private regular files")
                remaining = MAX_TOTAL_INPUT_BYTES - total
                if expected.st_size > MAX_INPUT_BYTES \
                        or expected.st_size > remaining:
                    raise AdapterNotApplicable(
                        "complete world inputs exceed their byte boundary")
                flags = (os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                         | getattr(os, "O_CLOEXEC", 0))
                fd = os.open(name, flags, dir_fd=parent_fd)
                before = os.fstat(fd)
                if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                        or _snapshot_stat_identity(before)
                        != _snapshot_stat_identity(expected)):
                    raise AdapterNotApplicable(
                        "world input changed before its complete snapshot")
                physical = (before.st_dev, before.st_ino)
                if physical in physical_identities:
                    raise AdapterNotApplicable(
                        "world input slots must name distinct physical files")
                physical_identities.add(physical)

                chunks = []
                size = 0
                limit = min(MAX_INPUT_BYTES, remaining)
                while True:
                    chunk = os.read(
                        fd, min(1024 * 1024, limit + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > limit:
                        raise AdapterNotApplicable(
                            "complete world inputs exceed their byte boundary")
                data = b"".join(chunks)
                identity = _snapshot_input_identity(resolved, before, data)
                opened.append(_OpenWorldInput(
                    resolved, path, name, parent_fd, fd, before, data, identity))
                parent_fd = -1
                fd = -1
                total += len(data)
            except OSError as error:
                raise AdapterNotApplicable(
                    f"cannot read complete world input {item['ref']!r}: "
                    f"{error}") from error
            finally:
                if fd >= 0:
                    os.close(fd)
                if parent_fd >= 0:
                    os.close(parent_fd)

        for item in opened:
            _revalidate_open_world_input(item)
        identities = [item.identity for item in opened]
        snapshot = {
            "schema": WORLD_INPUT_SNAPSHOT_SCHEMA,
            "model": (
                "ordered declared-file byte vector stable across one common "
                "observation window"
            ),
            "complete_regular_files": True,
            "distinct_private_files": True,
            "input_count": len(opened),
            "total_bytes": total,
            "input_vector_sha256": content_hash([{
                "slot": index,
                "ref": identity["ref"],
                "bytes": identity["bytes"],
                "sha256": identity["sha256"],
            } for index, identity in enumerate(identities)]),
            "proves_directory_completeness": False,
            "proves_future_closure": False,
        }
        yield [item.data for item in opened], identities, snapshot

        for item in opened:
            _revalidate_open_world_input(item)
        try:
            rebound_world = canonical(load_world_view(
                context.state_dir, live_world=context.live_world).describe())
        except (OSError, TypeError, ValueError, UnicodeError,
                RecursionError, MemoryError, OverflowError) as error:
            raise AdapterNotApplicable(
                f"cannot revalidate the current world view: {error}") from error
        if rebound_world != cached_world:
            raise AdapterNotApplicable(
                "world view changed during the complete input snapshot")
    finally:
        for item in reversed(opened):
            os.close(item.fd)
            os.close(item.parent_fd)


class _Interpreter:
    """Closed evaluator for ``LANGUAGE``; all operations consume fuel."""

    def __init__(self, source: str, inputs: Sequence[bytes],
                 subject_slots: set[int], *, complete_audit: bool = False):
        try:
            self.module = ast.parse(source, mode="exec")
        except (SyntaxError, ValueError, TypeError, MemoryError) as error:
            raise AdapterNotApplicable("evidence program syntax is invalid") from error
        _tree_shape(self.module)
        _validate_ast(self.module)
        if complete_audit and any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "require"
                and len(node.args) != 3
                for node in ast.walk(self.module)):
            raise AdapterNotApplicable(
                "complete evidence programs require three-argument require calls")
        self.source = source
        self.inputs = tuple(inputs)
        self.input_cache: Dict[tuple[str, int], Any] = {}
        self.subject_slots = frozenset(subject_slots)
        self.used_slots: set[int] = set()
        self.fuel = MAX_FUEL
        self.allocated_bytes = 0
        self.call_depth = 0
        self.env: Dict[str, Any] = {}
        self._env_taints: Dict[int, Dict[str, set[int]]] = {id(self.env): {}}
        self._trace_stack: list[set[int]] = []
        self._control_stack: list[set[int]] = []
        self.functions: Dict[str, _Function] = {}
        self.assertions: list[dict] = []
        self.observations: list[dict] = []
        self.complete_audit = complete_audit

    def _burn(self, amount: int = 1) -> None:
        if type(amount) is not int or amount < 1 or self.fuel < amount:
            raise AdapterNotApplicable("evidence program exhausted its fuel")
        self.fuel -= amount

    def _reserve(self, amount: int) -> None:
        """Charge conservative cumulative host allocation before allocating."""
        if (type(amount) is not int or amount < 0
                or amount > MAX_CUMULATIVE_ALLOCATION_BYTES - self.allocated_bytes):
            raise AdapterNotApplicable(
                "evidence program exhausted its allocation budget")
        self.allocated_bytes += amount

    @staticmethod
    def _sequence_reservation(kind: type, length: int) -> int:
        if kind is str:
            # Python may retain up to four bytes per Unicode code point.
            return length * 4
        if kind is bytes:
            return length
        return length * 16

    @staticmethod
    def _truth(value: Any) -> bool:
        if type(value) is not bool:
            raise AdapterNotApplicable("evidence program conditions must be boolean")
        return value

    @staticmethod
    def _name(node: ast.AST) -> str:
        if (not isinstance(node, ast.Name) or not node.id.isascii()
                or not node.id.isidentifier() or node.id.startswith("_")):
            raise AdapterNotApplicable("evidence program assignment target is invalid")
        return node.id

    @staticmethod
    def _assignable_name(node: ast.AST) -> str:
        name = _Interpreter._name(node)
        if name in _BUILTINS:
            raise AdapterNotApplicable("evidence program cannot shadow a builtin")
        return name

    def _mark(self, slots: set[int] | frozenset[int]) -> None:
        for trace in self._trace_stack:
            trace.update(slots)

    def _charge_scan(self, value: Any) -> None:
        """Charge before a potentially linear host-Python operation."""
        stack = [value]
        visited = 0
        while stack:
            item = stack.pop()
            visited += 1
            if visited % 64 == 0:
                self._burn()
            if type(item) in (str, bytes):
                self._burn(max(1, len(item) // 1024))
            elif type(item) in (list, tuple):
                stack.extend(item)
            elif type(item) is dict:
                stack.extend(item.keys())
                stack.extend(item.values())
            elif type(item) is range:
                self._burn(max(1, len(item) // 64))

    @contextmanager
    def _capture(self):
        trace = set().union(*self._control_stack) if self._control_stack else set()
        self._trace_stack.append(trace)
        try:
            yield trace
        finally:
            if self._trace_stack[-1] is not trace:
                raise RuntimeError("evidence program taint stack was corrupted")
            self._trace_stack.pop()

    @contextmanager
    def _control(self, slots: set[int]):
        self._control_stack.append(set(slots))
        try:
            yield
        finally:
            self._control_stack.pop()

    def run(self) -> dict:
        for statement in self.module.body:
            if isinstance(statement, ast.FunctionDef):
                self._define(statement)
        failed_requirement = None
        try:
            for statement in self.module.body:
                if not isinstance(statement, ast.FunctionDef):
                    self._statement(
                        statement, self.env, in_function=False, in_loop=False)
        except _RequirementFalse as failure:
            if not self.complete_audit:
                raise RuntimeError(
                    "legacy evidence program leaked complete-audit control flow")
            failed_requirement = failure.assertion_index
        if not self.assertions:
            raise AdapterNotApplicable("evidence program must make at least one requirement")
        if self.used_slots != set(range(len(self.inputs))):
            raise AdapterNotApplicable("evidence program did not read every declared input slot")
        result = {
            "assertions": self.assertions,
            "observations": self.observations,
            "fuel_limit": MAX_FUEL,
            "fuel_used": MAX_FUEL - self.fuel,
            "allocation_limit_bytes": MAX_CUMULATIVE_ALLOCATION_BYTES,
            "allocation_reserved_bytes": self.allocated_bytes,
            "used_input_slots": sorted(self.used_slots),
        }
        if self.complete_audit:
            result["complete_audit"] = {
                "audit_complete": True,
                "program_outcome": (
                    "require-false" if failed_requirement is not None
                    else "all-requirements-held"
                ),
                "failed_assertion_index": failed_requirement,
                "require_is_terminal": True,
                "obligation_verdict": False,
                "proves_future_closure": False,
            }
        return result

    def _define(self, node: ast.FunctionDef) -> None:
        self._burn()
        if (node.decorator_list or node.returns is not None or node.type_comment is not None
                or node.args.posonlyargs or node.args.kwonlyargs or node.args.kw_defaults
                or node.args.defaults or node.args.vararg or node.args.kwarg):
            raise AdapterNotApplicable("evidence program function shape is unsupported")
        name = self._assignable_name(ast.Name(id=node.name))
        parameters = tuple(argument.arg for argument in node.args.args)
        if (len(parameters) != len(set(parameters))
                or any(not item.isascii() or not item.isidentifier()
                       or item.startswith("_") or item in _BUILTINS
                       for item in parameters)
                or name in self.functions or name in self.env):
            raise AdapterNotApplicable("evidence program function identity is invalid")
        self.functions[name] = _Function(name, parameters, tuple(node.body))

    def _statement(self, node: ast.stmt, env: Dict[str, Any], *,
                   in_function: bool, in_loop: bool) -> None:
        self._burn()
        taints = self._env_taints[id(env)]
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                raise AdapterNotApplicable("evidence program assignment must have one target")
            name = self._assignable_name(node.targets[0])
            with self._capture() as trace:
                env[name] = self._value(self._expr(node.value, env))
            taints[name] = set(trace)
            return
        if isinstance(node, ast.AugAssign):
            name = self._assignable_name(node.target)
            if name not in env:
                raise AdapterNotApplicable("evidence program augmented target is undefined")
            with self._capture() as trace:
                self._mark(taints.get(name, set()))
                value = self._expr(node.value, env)
                env[name] = self._value(self._binary(node.op, env[name], value))
            taints[name] = set(trace)
            return
        if isinstance(node, ast.Expr):
            if not isinstance(node.value, ast.Call):
                raise AdapterNotApplicable("evidence program expression statement must be a call")
            with self._capture():
                self._expr(node.value, env)
            return
        if isinstance(node, ast.If):
            with self._capture() as condition_taint:
                condition = self._truth(self._expr(node.test, env))
            branch = node.body if condition else node.orelse
            with self._control(condition_taint):
                for child in branch:
                    self._statement(child, env, in_function=in_function, in_loop=in_loop)
            return
        if isinstance(node, ast.For):
            name = self._assignable_name(node.target)
            with self._capture() as iterable_taint:
                iterable = self._expr(node.iter, env)
            if type(iterable) not in (list, tuple, range, bytes, str, dict):
                raise AdapterNotApplicable("evidence program loop iterable is unsupported")
            if len(iterable) > MAX_LOOP_ITEMS:
                raise AdapterNotApplicable("evidence program loop exceeds its bound")
            broke = False
            for item in iterable:
                self._burn()
                env[name] = item
                taints[name] = set(iterable_taint)
                try:
                    with self._control(iterable_taint):
                        for child in node.body:
                            self._statement(child, env, in_function=in_function, in_loop=True)
                except _Continue:
                    continue
                except _Break:
                    broke = True
                    break
            if not broke:
                with self._control(iterable_taint):
                    for child in node.orelse:
                        self._statement(child, env, in_function=in_function, in_loop=in_loop)
            return
        if isinstance(node, ast.Return) and in_function:
            with self._capture() as trace:
                value = None if node.value is None else self._value(self._expr(node.value, env))
            raise _Return(value, set(trace))
        if isinstance(node, ast.Break) and in_loop:
            raise _Break()
        if isinstance(node, ast.Continue) and in_loop:
            raise _Continue()
        if isinstance(node, ast.Pass):
            return
        raise AdapterNotApplicable(
            f"evidence program statement {type(node).__name__} is unsupported")

    def _expr(self, node: ast.expr, env: Mapping[str, Any]) -> Any:
        self._burn()
        if isinstance(node, ast.Constant):
            if type(node.value) not in (type(None), bool, int, float, str, bytes):
                raise AdapterNotApplicable("evidence program literal is unsupported")
            return self._value(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                self._mark(self._env_taints[id(env)].get(node.id, set()))
                return env[node.id]
            raise AdapterNotApplicable("evidence program name is undefined")
        if isinstance(node, (ast.List, ast.Tuple)):
            self._reserve(max(1, len(node.elts)) * (
                32 if isinstance(node, ast.Tuple) else 16))
            result = [self._expr(item, env) for item in node.elts]
            return self._value(tuple(result) if isinstance(node, ast.Tuple) else result)
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise AdapterNotApplicable("evidence program dictionary unpacking is unsupported")
            self._reserve(max(1, len(node.keys)) * 64)
            result = {}
            for key_node, value_node in zip(node.keys, node.values):
                key = self._expr(key_node, env)
                if type(key) is not str or key in result:
                    raise AdapterNotApplicable("evidence program object keys must be unique strings")
                result[key] = self._expr(value_node, env)
            return self._value(result)
        if isinstance(node, ast.UnaryOp):
            value = self._expr(node.operand, env)
            if isinstance(node.op, ast.Not):
                return not self._truth(value)
            if type(value) not in (int, float) or type(value) is bool:
                raise AdapterNotApplicable("evidence program numeric unary operand is invalid")
            if isinstance(node.op, ast.UAdd):
                return self._value(+value)
            if isinstance(node.op, ast.USub):
                return self._value(-value)
            if isinstance(node.op, ast.Invert) and type(value) is int:
                return _checked_int(~value)
            raise AdapterNotApplicable("evidence program unary operator is unsupported")
        if isinstance(node, ast.BinOp):
            return self._value(self._binary(
                node.op, self._expr(node.left, env), self._expr(node.right, env)))
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for child in node.values:
                    if not self._truth(self._expr(child, env)):
                        return False
                return True
            if isinstance(node.op, ast.Or):
                for child in node.values:
                    if self._truth(self._expr(child, env)):
                        return True
                return False
            raise AdapterNotApplicable("evidence program boolean operator is unsupported")
        if isinstance(node, ast.Compare):
            left = self._expr(node.left, env)
            for operator, right_node in zip(node.ops, node.comparators):
                right = self._expr(right_node, env)
                if not self._compare(operator, left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self._expr(
                node.body if self._truth(self._expr(node.test, env)) else node.orelse,
                env,
            )
        if isinstance(node, ast.Subscript):
            container = self._expr(node.value, env)
            index = self._slice(node.slice, env)
            if type(container) not in (list, tuple, dict, bytes, str):
                raise AdapterNotApplicable("evidence program subscript target is unsupported")
            if type(container) is dict:
                if type(index) is not str:
                    raise AdapterNotApplicable(
                        "evidence program object index must be a string")
            elif not (type(index) is int or isinstance(index, slice)):
                raise AdapterNotApplicable(
                    "evidence program sequence index must be an integer or slice")
            try:
                if isinstance(index, slice):
                    start, stop, step = index.indices(len(container))
                    sliced_items = len(range(start, stop, step))
                    self._reserve(self._sequence_reservation(
                        type(container), sliced_items))
                value = container[index]
                return self._value(value)
            except (IndexError, KeyError, TypeError, ValueError) as error:
                raise AdapterNotApplicable("evidence program subscript failed") from error
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise AdapterNotApplicable("evidence program calls must be positional named calls")
            args = []
            arg_taints = []
            for item in node.args:
                parents = [set(trace) for trace in self._trace_stack]
                with self._capture() as trace:
                    args.append(self._expr(item, env))
                arg_taints.append(set(trace))
                # Evaluating an argument is not itself a dependency of a call's
                # result.  The function/builtin below propagates only inputs
                # that its actual return value or side effect depends on.
                for parent, snapshot in zip(self._trace_stack, parents):
                    parent.clear()
                    parent.update(snapshot)
            return self._call(node.func.id, args, arg_taints, env, node)
        raise AdapterNotApplicable(
            f"evidence program expression {type(node).__name__} is unsupported")

    def _slice(self, node: ast.expr, env: Mapping[str, Any]) -> Any:
        if isinstance(node, ast.Slice):
            if node.step is not None:
                step = self._expr(node.step, env)
                if type(step) is not int or step == 0 or abs(step) > MAX_LOOP_ITEMS:
                    raise AdapterNotApplicable("evidence program slice step is invalid")
            else:
                step = None
            lower = None if node.lower is None else self._expr(node.lower, env)
            upper = None if node.upper is None else self._expr(node.upper, env)
            if any(value is not None and type(value) is not int
                   for value in (lower, upper)):
                raise AdapterNotApplicable("evidence program slice bound is invalid")
            return slice(lower, upper, step)
        return self._expr(node, env)

    def _binary(self, operator: ast.operator, left: Any, right: Any) -> Any:
        try:
            return self._unchecked_binary(operator, left, right)
        except (ArithmeticError, ValueError) as error:
            raise AdapterNotApplicable(
                "evidence program arithmetic failed") from error

    def _unchecked_binary(self, operator: ast.operator,
                          left: Any, right: Any) -> Any:
        numeric = type(left) in (int, float) and type(right) in (int, float) \
            and type(left) is not bool and type(right) is not bool
        if isinstance(operator, ast.Add):
            if numeric:
                return left + right
            if type(left) is type(right) and type(left) in (str, bytes, list, tuple):
                if len(left) + len(right) > MAX_VALUE_ITEMS \
                        and type(left) in (list, tuple):
                    raise AdapterNotApplicable("evidence program allocation exceeds its bound")
                if type(left) in (str, bytes) and len(left) + len(right) > MAX_VALUE_BYTES:
                    raise AdapterNotApplicable("evidence program allocation exceeds its bound")
                self._reserve(self._sequence_reservation(
                    type(left), len(left) + len(right)))
                return left + right
        if isinstance(operator, ast.Sub) and numeric:
            return left - right
        if isinstance(operator, ast.Mult):
            if numeric:
                if type(left) is int and type(right) is int \
                        and left.bit_length() + right.bit_length() > MAX_INTEGER_BITS + 1:
                    raise AdapterNotApplicable("evidence program multiplication exceeds its bound")
                return left * right
            if type(left) in (str, bytes, list, tuple) and type(right) is int:
                limit = MAX_VALUE_BYTES if type(left) in (str, bytes) else MAX_VALUE_ITEMS
                if right < 0 or (len(left) and right > limit // len(left)):
                    raise AdapterNotApplicable("evidence program repetition exceeds its bound")
                self._reserve(self._sequence_reservation(
                    type(left), len(left) * right))
                return left * right
        if isinstance(operator, ast.FloorDiv) and numeric and right != 0:
            return left // right
        if isinstance(operator, ast.Div) and numeric and right != 0:
            return left / right
        if isinstance(operator, ast.Mod) and numeric and right != 0:
            return left % right
        if type(left) is int and type(right) is int:
            if isinstance(operator, ast.BitAnd):
                return left & right
            if isinstance(operator, ast.BitOr):
                return left | right
            if isinstance(operator, ast.BitXor):
                return left ^ right
            if isinstance(operator, (ast.LShift, ast.RShift)):
                if right < 0 or right > MAX_INTEGER_BITS:
                    raise AdapterNotApplicable("evidence program shift exceeds its bound")
                return left << right if isinstance(operator, ast.LShift) else left >> right
        raise AdapterNotApplicable("evidence program binary operator or operands are unsupported")

    def _compare(self, operator: ast.cmpop, left: Any, right: Any) -> bool:
        try:
            if isinstance(operator, ast.Eq):
                self._charge_scan(left)
                self._charge_scan(right)
                return _strict_equal(left, right)
            if isinstance(operator, ast.NotEq):
                self._charge_scan(left)
                self._charge_scan(right)
                return not _strict_equal(left, right)
            if isinstance(operator, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                if type(left) is not type(right) or type(left) not in (
                        int, float, str, bytes):
                    raise AdapterNotApplicable(
                        "evidence program ordered operands are invalid")
                self._charge_scan(left)
                self._charge_scan(right)
                if isinstance(operator, ast.Lt):
                    return left < right
                if isinstance(operator, ast.LtE):
                    return left <= right
                if isinstance(operator, ast.Gt):
                    return left > right
                return left >= right
            if isinstance(operator, (ast.In, ast.NotIn)):
                self._charge_scan(right)
                if type(right) in (list, tuple):
                    present = any(_strict_equal(left, item) for item in right)
                elif type(right) is dict:
                    if type(left) is not str:
                        raise AdapterNotApplicable(
                            "evidence program object membership key must be a string")
                    present = left in right
                elif type(right) is str:
                    if type(left) is not str:
                        raise AdapterNotApplicable(
                            "evidence program string membership value must be a string")
                    present = left in right
                elif type(right) is bytes:
                    if type(left) not in (int, bytes):
                        raise AdapterNotApplicable(
                            "evidence program bytes membership value is invalid")
                    present = left in right
                elif type(right) is range:
                    if type(left) is not int:
                        raise AdapterNotApplicable(
                            "evidence program range membership value must be an integer")
                    present = left in right
                else:
                    raise AdapterNotApplicable(
                        "evidence program membership container is invalid")
                return present if isinstance(operator, ast.In) else not present
            if isinstance(operator, ast.Is):
                if left is not None and right is not None:
                    raise AdapterNotApplicable(
                        "evidence program identity comparison only supports None")
                return left is None and right is None
            if isinstance(operator, ast.IsNot):
                if left is not None and right is not None:
                    raise AdapterNotApplicable(
                        "evidence program identity comparison only supports None")
                return not (left is None and right is None)
        except (TypeError, ValueError, OverflowError) as error:
            raise AdapterNotApplicable("evidence program comparison failed") from error
        raise AdapterNotApplicable("evidence program comparison is unsupported")

    def _call(self, name: str, args: list[Any], arg_taints: list[set[int]],
              env: Mapping[str, Any], node: ast.Call) -> Any:
        self._burn()
        if name in self.functions:
            function = self.functions[name]
            if len(args) != len(function.parameters) or self.call_depth >= MAX_CALL_DEPTH:
                raise AdapterNotApplicable("evidence program function call exceeds its contract")
            local = dict(zip(function.parameters, args))
            self._env_taints[id(local)] = {
                parameter: set(taint)
                for parameter, taint in zip(function.parameters, arg_taints)
            }
            self.call_depth += 1
            try:
                try:
                    for statement in function.body:
                        self._statement(statement, local, in_function=True, in_loop=False)
                except _Return as returned:
                    self._mark(returned.taint)
                    return returned.value
                return None
            finally:
                self.call_depth -= 1
                self._env_taints.pop(id(local), None)
        return self._builtin(name, args, arg_taints, node)

    def _builtin(self, name: str, args: list[Any], arg_taints: list[set[int]],
                 node: ast.Call) -> Any:
        if name in {"input_bytes", "input_text", "input_json"}:
            if len(args) != 1 or type(args[0]) is not int \
                    or not 0 <= args[0] < len(self.inputs):
                raise AdapterNotApplicable("evidence program input slot is invalid")
            slot = args[0]
            data = self.inputs[slot]
            self._burn(max(1, len(data) // 1024))
            self.used_slots.add(slot)
            self._mark(set(arg_taints[0]) | {slot})
            if name == "input_bytes":
                return data
            cache_key = (name, slot)
            if cache_key in self.input_cache:
                return self.input_cache[cache_key]
            if name == "input_text":
                try:
                    self._reserve(max(1, len(data)) * 4)
                    value = self._value(data.decode("utf-8", errors="strict"))
                    self.input_cache[cache_key] = value
                    return value
                except UnicodeDecodeError as error:
                    raise AdapterNotApplicable("evidence program input is not strict UTF-8") from error
            try:
                # Precharge decoder transients before the host parser, then
                # account the validated retained tree by exact VM value shape.
                precharge = _json_decoder_preflight(data)
                self._reserve(precharge)
                parsed = _strict_json_document(data)
                _validate_json_tree(parsed)
                value = self._value(parsed)
                retained = _storage_reservation(value)
                if retained > precharge:
                    self._reserve(retained - precharge)
                self.input_cache[cache_key] = value
                return value
            except (UnicodeError, ValueError, json.JSONDecodeError, RecursionError) as error:
                raise AdapterNotApplicable("evidence program input is not strict bounded JSON") from error
        if name == "len" and len(args) == 1 and type(args[0]) in (
                str, bytes, list, tuple, dict, range):
            self._mark(set(arg_taints[0]))
            return _checked_int(len(args[0]))
        if name == "range" and 1 <= len(args) <= 3 \
                and all(type(item) is int for item in args):
            try:
                value = range(*args)
                if len(value) > MAX_LOOP_ITEMS:
                    raise AdapterNotApplicable("evidence program range exceeds its bound")
                self._mark(set().union(*arg_taints))
                return value
            except (ValueError, OverflowError) as error:
                raise AdapterNotApplicable("evidence program range is invalid") from error
        if name in {"be_uint", "be_int", "le_uint", "le_int"} and len(args) == 3:
            data, start, length = args
            if (type(data) is not bytes or type(start) is not int or type(length) is not int
                    or length < 1 or length > 512 or start < 0
                    or start + length > len(data)):
                raise AdapterNotApplicable("evidence program integer decode is invalid")
            result = int.from_bytes(
                data[start:start + length],
                "big" if name.startswith("be_") else "little",
                signed=name.endswith("_int"),
            )
            self._mark(set().union(*arg_taints))
            return _checked_int(result)
        if name in {"be_float64", "le_float64"} and len(args) == 2:
            data, start = args
            if type(data) is not bytes or type(start) is not int \
                    or start < 0 or start + 8 > len(data):
                raise AdapterNotApplicable("evidence program float decode is invalid")
            result = struct.unpack(">d" if name == "be_float64" else "<d", data[start:start + 8])[0]
            self._mark(set().union(*arg_taints))
            return self._value(result)
        if name == "utf8" and len(args) == 1 and type(args[0]) is bytes:
            self._burn(max(1, len(args[0]) // 1024))
            try:
                self._reserve(max(1, len(args[0])) * 4)
                self._mark(set(arg_taints[0]))
                return self._value(args[0].decode("utf-8", errors="strict"))
            except UnicodeDecodeError as error:
                raise AdapterNotApplicable("evidence program UTF-8 decode failed") from error
        if name == "sha256" and len(args) == 1 and type(args[0]) is bytes:
            self._burn(max(1, len(args[0]) // 1024))
            self._mark(set(arg_taints[0]))
            return hashlib.sha256(args[0]).hexdigest()
        if name == "keys" and len(args) == 1 and type(args[0]) is dict:
            self._burn(max(1, len(args[0]) // 64))
            self._reserve(max(1, len(args[0])) * 16)
            self._mark(set(arg_taints[0]))
            return list(args[0].keys())
        if name == "is_type" and len(args) == 2 and type(args[1]) is str:
            kinds = {
                "null": type(None), "bool": bool, "int": int, "float": float,
                "number": (int, float), "str": str, "bytes": bytes,
                "list": list, "tuple": tuple, "object": dict,
            }
            expected = kinds.get(args[1])
            if expected is None:
                raise AdapterNotApplicable("evidence program type name is invalid")
            self._mark(set().union(*arg_taints))
            if args[1] == "number":
                return type(args[0]) in expected and type(args[0]) is not bool
            return type(args[0]) is expected
        if name == "require" and 2 <= len(args) <= 3:
            passed, label = args[0], args[1]
            if type(passed) is not bool or type(label) is not str \
                    or not label or len(label.encode("utf-8", errors="strict")) > 256:
                raise AdapterNotApplicable("evidence program requirement is invalid")
            if not passed and not self.complete_audit:
                raise AdapterNotApplicable("evidence program requirement did not hold")
            # Only the predicate, including its control dependencies,
            # authenticates the requirement.  A tainted label or observed
            # value cannot launder a constant-true predicate.
            dependency_slots = set(arg_taints[0])
            subject_dependencies = dependency_slots & self.subject_slots
            if not subject_dependencies:
                raise AdapterNotApplicable(
                    "evidence program requirement is independent of every subject")
            if len(self.assertions) >= MAX_ASSERTIONS:
                raise AdapterNotApplicable("evidence program has too many requirements")
            item = {
                "label": label,
                "expression_sha256": _node_digest(node.args[0]),
                "holds": passed,
                "input_slots": sorted(dependency_slots),
                "subject_slots": sorted(subject_dependencies),
            }
            if len(args) == 3:
                item["observed"] = self._retained(args[2])
                observed_slots = set(arg_taints[2])
                item["observed_input_slots"] = sorted(observed_slots)
                item["observed_subject_slots"] = sorted(
                    observed_slots & self.subject_slots)
                if self.complete_audit and not subject_dependencies <= \
                        observed_slots & self.subject_slots:
                    raise AdapterNotApplicable(
                        "complete evidence program observation does not cover "
                        "the requirement's subject inputs")
            self.assertions.append(item)
            if not passed:
                if self.used_slots != set(range(len(self.inputs))):
                    raise AdapterNotApplicable(
                        "complete evidence program failed before reading every "
                        "declared input slot")
                raise _RequirementFalse(len(self.assertions) - 1)
            return None
        if name == "observe" and len(args) == 2 and type(args[0]) is str:
            label = args[0]
            if (not label or len(label.encode("utf-8", errors="strict")) > 256
                    or len(self.observations) >= MAX_OBSERVATIONS):
                raise AdapterNotApplicable("evidence program observation is invalid")
            self.observations.append({
                "label": label,
                "value": self._retained(args[1]),
            })
            return None
        raise AdapterNotApplicable(f"evidence program builtin {name!r} is unsupported")

    def _value(self, value: Any) -> Any:
        # Validate in place; do not construct a mirrored tree for temporaries.
        self._charge_scan(value)
        _validate_value(value)
        return value

    def _retained(self, value: Any) -> Any:
        """Create one small model-visible value under the global budget."""
        self._charge_scan(value)
        _validate_value(value)
        rendered_bound = _retained_json_upper_bound(value)
        if rendered_bound > MAX_RETAINED_OBSERVATION_CHARS:
            raise AdapterNotApplicable(
                "evidence program observation exceeds its retained bound")
        self._reserve(_storage_reservation(value) + rendered_bound * 4)
        return _json_safe(value)


class EvidenceProgramAdapter(MethodAdapter):
    """Execute a pure bounded inspection program over declared world files."""

    name = "evidence.program"
    version = 1
    basis = BASIS_WORLD
    accepted_method_reuse = False
    summary = (
        "Interpret a bounded pure evidence program over ordered declared world-file "
        "slots. This is not Python execution: imports, attributes, ambient I/O, "
        "commands, network, processes, time, randomness, and eval/exec do not exist. "
        "It can only report holds; any false check or error falls back to the agent executor."
    )
    arguments = {"source": f"{LANGUAGE} text, 1..{MAX_SOURCE_CHARS} characters"}

    def catalog_entry(self) -> Dict[str, Any]:
        entry = super().catalog_entry()
        entry.update({
            "direction": "holds-only; false/error/limit => ordinary executor fallback",
            "language": {
                "identity": LANGUAGE,
                "statements": [
                    "name = expression", "name += expression", "if/else",
                    "for name in bounded iterable", "def positional_function(...)",
                    "return/break/continue/pass", "require(bool,label,observed?)",
                    "observe(label,value)",
                ],
                "expressions": [
                    "bounded literals/list/tuple/string-key object", "name", "index/slice",
                    "if-expression", "strict boolean/comparison", "checked arithmetic/bitwise",
                ],
                "builtins": [
                    "input_bytes(slot)", "input_text(slot)", "input_json(slot)",
                    "len", "range", "keys", "is_type",
                    "be_uint", "be_int", "le_uint", "le_int",
                    "be_float64", "le_float64", "utf8", "sha256",
                    "require", "observe",
                ],
                "excluded": [
                    "import", "attribute", "filesystem path", "process", "network",
                    "clock", "randomness", "exception handling", "eval", "exec",
                ],
            },
            "input_contract": (
                "declare 1..8 unique regular world files in exact slot order; every slot "
                "must be read and at least one must have role=subject; semantic authorities "
                "that are not runtime-read files remain described in the MethodIR"
            ),
            "typed_transport": {
                "requirement": (
                    "for restricted no-tools Judger transport, every require must use "
                    "require(bool,label,observed); observed must be complete, untruncated, "
                    "and depend on every subject slot used by the predicate"
                ),
                "example": (
                    'value = input_text(0)\n'
                    'require(value == "expected", "exact value", value)'
                ),
                "fallback": (
                    "two-argument require remains valid adapter execution but uses the "
                    "ordinary fresh toolful Judger"
                ),
            },
            "resource_limits": {
                "source_chars": MAX_SOURCE_CHARS,
                "ast_nodes": MAX_AST_NODES,
                "ast_depth": MAX_AST_DEPTH,
                "input_files": MAX_INPUT_FILES,
                "input_bytes_each": MAX_INPUT_BYTES,
                "input_bytes_total": MAX_TOTAL_INPUT_BYTES,
                "fuel": MAX_FUEL,
                "loop_items": MAX_LOOP_ITEMS,
                "value_items": MAX_VALUE_ITEMS,
                "value_bytes": MAX_VALUE_BYTES,
                "cumulative_allocation_bytes": MAX_CUMULATIVE_ALLOCATION_BYTES,
                "integer_bits": MAX_INTEGER_BITS,
                "call_depth": MAX_CALL_DEPTH,
                "assertions": MAX_ASSERTIONS,
                "observations": MAX_OBSERVATIONS,
                "retained_observation_chars": MAX_RETAINED_OBSERVATION_CHARS,
                "receipt_chars": 12_000,
                "binary_integer_width_bytes": 512,
            },
        })
        return entry

    def validate(self, invocation: AdapterInvocation, method: Mapping[str, Any]) -> None:
        super().validate(invocation, method)
        self.exact_arguments(invocation.arguments, ("source",))
        source = invocation.arguments["source"]
        if (type(source) is not str or not source or len(source) > MAX_SOURCE_CHARS
                or "\x00" in source):
            raise AdapterNotApplicable("evidence program source is invalid or too large")
        try:
            source.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise AdapterNotApplicable("evidence program source is not valid Unicode") from error
        execution_inputs = [item for item in invocation.inputs if item["role"] != "authority"]
        if not 1 <= len(execution_inputs) <= MAX_INPUT_FILES:
            raise AdapterNotApplicable("evidence program requires 1..8 runtime file inputs")
        if len({item["ref"] for item in invocation.inputs}) != len(invocation.inputs):
            raise AdapterNotApplicable("evidence program inputs must have unique references")
        if not any(item["role"] == ROLE_SUBJECT for item in execution_inputs):
            raise AdapterNotApplicable("evidence program requires a subject input")
        if any(item["provenance"] not in WORLD_FILE_PROVENANCES for item in execution_inputs):
            raise AdapterNotApplicable("evidence program runtime inputs must be world files")

    def execute(self, invocation: AdapterInvocation, obligation, context) -> AdapterExecution:
        self.validate(invocation, {
            "basis": self.basis,
        })
        execution_inputs = [item for item in invocation.inputs if item["role"] != "authority"]
        data: list[bytes] = []
        consumed: list[dict] = []
        total = 0
        for item in execution_inputs:
            resolved = context.resolve(item["ref"])
            value, identity = _read_bounded_file(resolved, MAX_INPUT_BYTES)
            total += len(value)
            if total > MAX_TOTAL_INPUT_BYTES:
                raise AdapterNotApplicable("evidence program inputs exceed their total bound")
            data.append(value)
            consumed.append(identity)

        source = invocation.arguments["source"]
        subject_slots = {
            index for index, item in enumerate(execution_inputs)
            if item["role"] == ROLE_SUBJECT
        }
        try:
            result = _Interpreter(source, data, subject_slots).run()
        except (RecursionError, MemoryError) as error:
            raise AdapterNotApplicable(
                "evidence program exceeded a host safety boundary") from error
        observation = {
            "language": LANGUAGE,
            "source_sha256": _source_digest(source),
            "source_chars": len(source),
            "input_slots": [
                {
                    "slot": index,
                    "ref": item["ref"],
                    "provenance": item["provenance"],
                    "role": item["role"],
                    "bytes": len(data[index]),
                    "sha256": consumed[index]["sha256"],
                }
                for index, item in enumerate(execution_inputs)
            ],
            **result,
        }
        if len(canonical(observation)) > 12_000:
            raise AdapterNotApplicable("evidence program receipt exceeds its bound")
        return AdapterExecution(
            OUTCOME_HOLDS,
            None,
            WITNESS_NONE,
            observation,
            tuple(consumed),
        )


class EvidenceProgramCompleteAuditAdapter(EvidenceProgramAdapter):
    """Versioned complete method audit over stable declared-file snapshots."""

    version = 2
    summary = (
        "Interpret the same bounded pure evidence language over a stable, "
        "complete vector of declared regular-file bytes. All requirements "
        "held reports holds. One fully validated false requirement reports "
        "only method failed, never refuted; every invalid, racing, incomplete, "
        "or resource-limited execution falls back."
    )

    def catalog_entry(self) -> Dict[str, Any]:
        entry = super().catalog_entry()
        entry["language"] = {
            **entry["language"],
            "statements": [
                statement.replace(
                    "require(bool,label,observed?)",
                    "require(bool,label,observed)",
                )
                for statement in entry["language"]["statements"]
            ],
        }
        entry.update({
            "direction": (
                "holds-only; complete valid require-false => method failed; "
                "invalid/incomplete/race/error/limit => ordinary executor fallback"
            ),
            "input_contract": (
                "declare 1..8 unique private regular world files in exact slot "
                "order with no non-runtime authority slots; every declared slot "
                "must be read before require can fail; all files remain open and "
                "are rebound across one common observation window"
            ),
            "complete_audit": {
                "false_requirement": (
                    "failed execution of this exact MethodIR only; never an "
                    "obligation verdict, refutation, or absence witness"
                ),
                "require_contract": (
                    "every require has exactly three arguments; its predicate "
                    "depends on a subject and its bounded observed value covers "
                    "those subject slots"
                ),
                "method_contract": (
                    "the pure program must exhaust the declared method; require "
                    "is terminal at the first false predicate and no unrepresented "
                    "alternate procedure is implied"
                ),
                "snapshot": WORLD_INPUT_SNAPSHOT_SCHEMA,
                "directory_completeness": False,
                "future_closure": False,
            },
            "typed_transport": {
                "eligible": False,
                "reason": (
                    "phase 1 keeps evidence.program@2 on the ordinary fresh "
                    "toolful Judger path; failed methods are not judged"
                ),
            },
            "resource_limits": {
                **entry["resource_limits"],
                "receipt_bytes": MAX_RECEIPT_BYTES,
            },
        })
        return entry

    def validate(self, invocation: AdapterInvocation,
                 method: Mapping[str, Any]) -> None:
        super().validate(invocation, method)
        if any(item["role"] == ROLE_AUTHORITY for item in invocation.inputs):
            raise AdapterNotApplicable(
                "complete evidence program requires every declared input to be "
                "a runtime-read file slot")

    def execute(self, invocation: AdapterInvocation, obligation,
                context) -> AdapterExecution:
        self.validate(invocation, {"basis": self.basis})
        execution_inputs = [item for item in invocation.inputs
                            if item["role"] != ROLE_AUTHORITY]
        source = invocation.arguments["source"]
        with _stable_world_input_vector(
                context, execution_inputs) as (data, consumed, snapshot):
            subject_slots = {
                index for index, item in enumerate(execution_inputs)
                if item["role"] == ROLE_SUBJECT
            }
            try:
                result = _Interpreter(
                    source, data, subject_slots, complete_audit=True).run()
            except (RecursionError, MemoryError) as error:
                raise AdapterNotApplicable(
                    "complete evidence program exceeded a host safety boundary"
                ) from error

        input_slots = []
        for index, (declared, identity) in enumerate(
                zip(execution_inputs, consumed)):
            input_slots.append({
                "slot": index,
                "ref": declared["ref"],
                "provenance": declared["provenance"],
                "role": declared["role"],
                "bytes": identity["bytes"],
                "sha256": identity["sha256"],
                "file_identity": identity["file_identity"],
            })
        observation = {
            "schema": COMPLETE_AUDIT_OBSERVATION_SCHEMA,
            "language": LANGUAGE,
            "source_sha256": _source_digest(source),
            "source_chars": len(source),
            "input_slots": input_slots,
            "snapshot": snapshot,
            **result,
        }
        if len(canonical(observation).encode("utf-8", errors="strict")) \
                > MAX_RECEIPT_BYTES:
            raise AdapterNotApplicable(
                "complete evidence program receipt exceeds its byte bound")

        audit = result["complete_audit"]
        failed_index = audit["failed_assertion_index"]
        if failed_index is None:
            return AdapterExecution(
                OUTCOME_HOLDS, None, WITNESS_NONE, observation,
                tuple(consumed))
        failed = result["assertions"][failed_index]
        return AdapterExecution(
            OUTCOME_FAILED,
            {
                "reason": (
                    "the exact complete evidence program terminated at a "
                    "false requirement; this is not an obligation verdict"
                ),
                "failure_kind": "require-false",
                "failed_requirement": {
                    "label": failed["label"],
                    "expression_sha256": failed["expression_sha256"],
                },
            },
            WITNESS_NONE,
            observation,
            tuple(consumed),
        )
