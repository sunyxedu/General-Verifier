"""Host-pinned, data-only sanitizer runtime inventory, never a safety verdict.

Only an operator configuration supplies the manifest path.  Method variables,
claim text and solver-provided JSON are not configuration sources.  No command,
compiler or target is executed or rewritten by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from typing import Any

from ...core.model import content_hash

RUNTIME_SCHEMA = "gv-sanitizer-runtime-manifest-v1"
INVENTORY_SCHEMA = "gv-cyber-sanitizer-runtime-inventory-v1"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_RUNTIME_FILE_BYTES = 512 * 1024 * 1024
_ROLES = ("runner", "launcher", "policy_source", "shim", "gcc_runtime",
          "gcc_compiler", "clang_compiler")
_MANIFEST_PATH = "/usr/local/lib/gv-sanitizer/manifest.json"
_FIXED_PATHS = {
    "runner": "/usr/local/lib/gv-sanitizer/sanitizer_runner.py",
    "launcher": "/usr/local/bin/gv-sanitizer-run",
    "policy_source": "/usr/local/lib/gv-sanitizer/lsan_policy.c",
    "shim": "/usr/local/lib/gv-sanitizer/lsan_policy.so",
}
_COMPILER_ROOT = "/usr/"
_POLICY_SOURCE = b"int __lsan_is_turned_off(void) { return 1; }\n"
_CAPABILITIES = {
    "scope": "explicit_runner_only",
    "requested_leak_check_policy": "disabled_unchecked",
    "asan_memory_checks": "not_disabled",
    "ubsan_checks": "not_disabled",
    "proc_read_grant": False,
    "compiler_or_target_rewrite": False,
    "link_order_check_disabled": False,
    "future_target_compatibility_verified": False,
}
_PROFILES = {
    "clang-static": {"preload": ["shim"], "scope": "declared_compilation_runtime"},
    "gcc-shared": {"preload": ["gcc_runtime", "shim"],
                   "scope": "declared_compilation_runtime"},
}


class RuntimeInventoryError(ValueError):
    """A fixed host runtime configuration failed closed (static error codes)."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise RuntimeInventoryError("runtime_manifest_invalid") from exc


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeInventoryError("runtime_manifest_duplicate_key")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise RuntimeInventoryError("runtime_manifest_nonfinite")


def _strict_json(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_MANIFEST_BYTES:
        raise RuntimeInventoryError("runtime_manifest_oversize")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_pairs,
                           parse_constant=_invalid_constant)
    except RuntimeInventoryError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise RuntimeInventoryError("runtime_manifest_invalid") from exc
    budget = 2048

    def bounded(node, depth=0):
        nonlocal budget
        budget -= 1
        if budget < 0 or depth > 12:
            raise RuntimeInventoryError("runtime_manifest_bounds")
        if type(node) is dict:
            for key, child in node.items():
                if not isinstance(key, str) or len(key) > 4096:
                    raise RuntimeInventoryError("runtime_manifest_bounds")
                try:
                    key.encode("utf-8", errors="strict")
                except UnicodeError as exc:
                    raise RuntimeInventoryError("runtime_manifest_invalid") from exc
                bounded(child, depth + 1)
        elif type(node) is list:
            for child in node:
                bounded(child, depth + 1)
        elif type(node) is str:
            if len(node) > 4096:
                raise RuntimeInventoryError("runtime_manifest_bounds")
            try:
                node.encode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise RuntimeInventoryError("runtime_manifest_invalid") from exc
        elif type(node) is int:
            if not -(2**63) <= node < 2**63:
                raise RuntimeInventoryError("runtime_manifest_bounds")
        elif node is not None and type(node) is not bool:
            # The schema contains no floating-point values, even finite ones.
            raise RuntimeInventoryError("runtime_manifest_type")

    bounded(value)
    if type(value) is not dict:
        raise RuntimeInventoryError("runtime_manifest_type")
    return value


def _path(value: Any) -> str:
    if (type(value) is not str or not value or len(value) > 4096
            or not value.startswith("/") or value.startswith("//")
            or "\x00" in value or os.path.normpath(value) != value
            or value == "/"):
        raise RuntimeInventoryError("runtime_path_invalid")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise RuntimeInventoryError("runtime_path_invalid") from exc
    return value


def _digest(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _descriptor(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _directory_descriptor(info: os.stat_result) -> tuple[int, ...]:
    # Unrelated image-directory child timestamps are not file identity.
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid)


def _trusted(info: os.stat_result, *, directory: bool) -> None:
    if ((not stat.S_ISDIR(info.st_mode) if directory else not stat.S_ISREG(info.st_mode))
            or info.st_uid != 0 or info.st_mode & 0o022
            or (not directory and info.st_nlink != 1)):
        raise RuntimeInventoryError("runtime_file_untrusted")


def _secure_fd(path: str) -> tuple[int, tuple[tuple[int, ...], ...]]:
    """Walk every component with nofollow descriptors; never follow aliases."""
    _path(path)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeInventoryError("runtime_nofollow_unsupported")
    parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    directories = []
    try:
        for part in path.split("/")[1:-1]:
            info = os.fstat(parent)
            _trusted(info, directory=True)
            directories.append(_directory_descriptor(info))
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=parent)
            os.close(parent)
            parent = child
        info = os.fstat(parent)
        _trusted(info, directory=True)
        directories.append(_directory_descriptor(info))
        fd = os.open(path.rsplit("/", 1)[1],
                     os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
        try:
            _trusted(os.fstat(fd), directory=False)
        except BaseException:
            os.close(fd)
            raise
        return fd, tuple(directories)
    finally:
        os.close(parent)


@dataclass(frozen=True)
class _PinnedFile:
    path: str
    sha256: str
    descriptor: tuple[int, ...]
    directories: tuple[tuple[int, ...], ...]

    @property
    def size(self) -> int:
        return self.descriptor[5]


def _read_pin(path: str, *, limit: int, retain: bool = False, prefix_bytes: int = 0,
              expected: _PinnedFile | None = None) -> tuple[_PinnedFile, bytes]:
    fd = check_fd = None
    try:
        fd, directories = _secure_fd(path)
        before = _descriptor(os.fstat(fd))
        if before[5] < 0 or before[5] > limit:
            raise RuntimeInventoryError("runtime_file_oversize")
        if expected is not None and (expected.descriptor != before
                                     or expected.directories != directories):
            raise RuntimeInventoryError("runtime_file_changed")
        digest = hashlib.sha256()
        retained = []
        retained_size = 0
        size = 0
        while chunk := os.read(fd, min(65536, limit + 1 - size)):
            size += len(chunk)
            if size > limit:
                raise RuntimeInventoryError("runtime_file_oversize")
            digest.update(chunk)
            if retain:
                retained.append(chunk)
            elif retained_size < prefix_bytes:
                prefix = chunk[:prefix_bytes - retained_size]
                retained.append(prefix)
                retained_size += len(prefix)
        if before != _descriptor(os.fstat(fd)) or size != before[5]:
            raise RuntimeInventoryError("runtime_file_changed")
        # Re-walk the named path while the original FD is still held: inode
        # replacement and moved/replaced parents cannot hide behind that FD.
        check_fd, current_dirs = _secure_fd(path)
        if (before != _descriptor(os.fstat(check_fd)) or directories != current_dirs
                or before != _descriptor(os.fstat(fd))):
            raise RuntimeInventoryError("runtime_file_changed")
        pin = _PinnedFile(path, digest.hexdigest(), before, directories)
        if expected is not None and pin != expected:
            raise RuntimeInventoryError("runtime_file_changed")
        return pin, b"".join(retained)
    except RuntimeInventoryError:
        raise
    except (OSError, ValueError, UnicodeError) as exc:
        raise RuntimeInventoryError("runtime_file_unavailable") from exc
    finally:
        if check_fd is not None:
            os.close(check_fd)
        if fd is not None:
            os.close(fd)


def _validate_manifest(value: dict[str, Any]) -> None:
    if (set(value) != {"schema", "platform", "source_revision", "files",
                      "compiler_versions", "profiles", "capabilities"}
            or value["schema"] != RUNTIME_SCHEMA or not _digest(value["source_revision"])):
        raise RuntimeInventoryError("runtime_manifest_schema")
    platform = value["platform"]
    if (type(platform) is not dict or set(platform) != {"system", "machine"}
            or platform["system"] != "Linux" or platform["machine"] not in ("aarch64", "x86_64")):
        raise RuntimeInventoryError("runtime_platform_invalid")
    actual = os.uname()
    if actual.sysname != platform["system"] or actual.machine != platform["machine"]:
        raise RuntimeInventoryError("runtime_platform_mismatch")
    files = value["files"]
    if type(files) is not dict or set(files) != set(_ROLES):
        raise RuntimeInventoryError("runtime_files_invalid")
    paths = set()
    for role, entry in files.items():
        if (type(entry) is not dict or set(entry) != {"path", "sha256"}
                or not _digest(entry["sha256"])):
            raise RuntimeInventoryError("runtime_files_invalid")
        path = _path(entry["path"])
        if ((role in _FIXED_PATHS and path != _FIXED_PATHS[role])
                or (role not in _FIXED_PATHS and not path.startswith(_COMPILER_ROOT))):
            raise RuntimeInventoryError("runtime_file_role_path_invalid")
        if path in paths:
            raise RuntimeInventoryError("runtime_files_aliased")
        paths.add(path)
    source_revision = content_hash({
        "schema": "gv-sanitizer-source-revision-v1",
        "files": {role: files[role]["sha256"]
                  for role in ("runner", "launcher", "policy_source")},
    })
    if source_revision != value["source_revision"]:
        raise RuntimeInventoryError("runtime_source_revision_mismatch")
    versions = value["compiler_versions"]
    if (type(versions) is not dict or set(versions) != {"gcc", "clang"}
            or any(type(item) is not str or not item or len(item) > 512
                   or any(ord(c) < 32 or ord(c) == 127 for c in item)
                   for item in versions.values())):
        raise RuntimeInventoryError("runtime_compiler_versions_invalid")
    # Canonical JSON equality preserves booleans vs integers (True != 1).
    if (_canonical(value["profiles"]) != _canonical(_PROFILES)
            or _canonical(value["capabilities"]) != _canonical(_CAPABILITIES)):
        raise RuntimeInventoryError("runtime_capabilities_invalid")


@dataclass(frozen=True)
class CyberRuntimeInventory:
    """Immutable host snapshot; emitted copies cannot mutate future inventory."""

    _inventory_json: str
    _manifest: _PinnedFile
    _files: tuple[_PinnedFile, ...]

    @property
    def tool_root(self) -> str:
        """Fixed public runtime directory; never a solver-selected grant."""
        return os.path.dirname(self._manifest.path)

    @property
    def revision(self) -> str:
        self.validate_current()
        return content_hash(json.loads(self._inventory_json))

    def validate_current(self) -> None:
        _read_pin(self._manifest.path, limit=MAX_MANIFEST_BYTES, expected=self._manifest)
        for pin in self._files:
            _read_pin(pin.path, limit=MAX_RUNTIME_FILE_BYTES, expected=pin)

    def as_inventory(self) -> dict[str, Any]:
        self.validate_current()
        return json.loads(self._inventory_json)


def load_runtime_inventory(manifest_path: str | None) -> CyberRuntimeInventory | None:
    """Load an operator-fixed manifest; None adds no key or legacy revision."""
    if manifest_path is None:
        return None
    path = _path(manifest_path)
    if path != _MANIFEST_PATH:
        raise RuntimeInventoryError("runtime_manifest_path_invalid")
    manifest_pin, raw = _read_pin(path, limit=MAX_MANIFEST_BYTES, retain=True)
    value = _strict_json(raw)
    _validate_manifest(value)
    files = []
    measured = {}
    for role in _ROLES:
        entry = value["files"][role]
        if entry["path"] == path:
            raise RuntimeInventoryError("runtime_files_aliased")
        policy = role == "policy_source"
        binary = role in ("shim", "gcc_runtime", "gcc_compiler", "clang_compiler")
        pin, prefix = _read_pin(entry["path"],
                                limit=MAX_MANIFEST_BYTES if policy else MAX_RUNTIME_FILE_BYTES,
                                retain=policy, prefix_bytes=20 if binary else 0)
        if pin.sha256 != entry["sha256"]:
            raise RuntimeInventoryError("runtime_file_digest_mismatch")
        if policy and prefix != _POLICY_SOURCE:
            raise RuntimeInventoryError("runtime_policy_source_invalid")
        if binary:
            machine = {"aarch64": 183, "x86_64": 62}[value["platform"]["machine"]]
            if (len(prefix) != 20 or prefix[:7] != b"\x7fELF\x02\x01\x01"
                    or int.from_bytes(prefix[18:20], "little") != machine):
                raise RuntimeInventoryError("runtime_binary_abi_invalid")
        files.append(pin)
        measured[role] = {"path": pin.path, "sha256": pin.sha256, "size": pin.size}
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "manifest": {"path": path, "sha256": manifest_pin.sha256, "size": manifest_pin.size},
        "source_revision": value["source_revision"],
        "platform": value["platform"],
        "files": measured,
        "compiler_versions": value["compiler_versions"],
        "profiles": value["profiles"],
        "capabilities": value["capabilities"],
        "command_interface": {
            "entrypoint": measured["launcher"],
            "mode": "explicit_runtime_profile",
            "required_option": "--runtime-profile",
            "profiles": ["clang-static", "gcc-shared"],
            "command_separator": "--",
            "implicit_command_interception": False,
        },
        "mutation_policy": {"source_rewrite": False, "compiler_rewrite": False,
                            "target_rewrite": False},
        "child_leak_policy_validation": {
            "state": "unchecked", "hook_override_possible": True,
        },
        "asan_control_validation": {
            "state": "not_established_by_runtime_manifest",
            "scope": "declared_compilation_runtime",
            "automatic_safety_verdict": False,
        },
    }
    result = CyberRuntimeInventory(_canonical(inventory), manifest_pin, tuple(files))
    # Detect changes during multi-file configuration loading as well.
    result.validate_current()
    return result
