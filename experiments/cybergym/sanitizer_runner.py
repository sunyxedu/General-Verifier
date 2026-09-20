"""Explicit, task-independent leak-policy compatibility runner.

The selected profile is a declared compilation/runtime convention, not a
verification of arbitrary future executables. This tool emits no verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import struct
import subprocess
import sys
import time
import uuid


MANIFEST_PATH = "/usr/local/lib/gv-sanitizer/manifest.json"
SCHEMA = "gv-sanitizer-runtime-manifest-v1"
FIXED_FILES = {
    "runner": "/usr/local/lib/gv-sanitizer/sanitizer_runner.py",
    "launcher": "/usr/local/bin/gv-sanitizer-run",
    "policy_source": "/usr/local/lib/gv-sanitizer/lsan_policy.c",
    "shim": "/usr/local/lib/gv-sanitizer/lsan_policy.so",
}
FILE_ROLES = frozenset((*FIXED_FILES, "gcc_runtime", "gcc_compiler", "clang_compiler"))
PROFILES = {
    "clang-static": {"preload": ["shim"], "scope": "declared_compilation_runtime"},
    "gcc-shared": {"preload": ["gcc_runtime", "shim"], "scope": "declared_compilation_runtime"},
}
CAPABILITIES = {
    "scope": "explicit_runner_only", "requested_leak_check_policy": "disabled_unchecked",
    "asan_memory_checks": "not_disabled", "ubsan_checks": "not_disabled",
    "proc_read_grant": False, "compiler_or_target_rewrite": False,
    "link_order_check_disabled": False, "future_target_compatibility_verified": False,
}
POLICY_SOURCE = b"int __lsan_is_turned_off(void) { return 1; }\n"
LAUNCHER_SOURCE = b'#!/bin/sh\nexec /usr/bin/python3.11 -I /usr/local/lib/gv-sanitizer/sanitizer_runner.py "$@"\n'
ROOT_UID = 0
MAX_FILE_BYTES = 512 * 1024 * 1024
METADATA_PREFIX = "GV_SANITIZER_EXECUTION_METADATA "
HOST_TRUST = "host"
READONLY_TRUST = "readonly_usr_anchor"


class SanitizerError(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        # Do not echo arbitrary CLI values as diagnostics.
        raise SanitizerError("runtime_cli_invalid")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def source_revision(files):
    payload = {"schema": "gv-sanitizer-source-revision-v1", "files": {
        role: files[role]["sha256"] for role in ("runner", "launcher", "policy_source")}}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _same_data(left, right):
    try:
        return canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError, RecursionError):
        return False


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _trusted(info, directory=False):
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if (not expected_type(info.st_mode) or info.st_uid != ROOT_UID
            or info.st_mode & 0o022 or (not directory and info.st_nlink != 1)):
        raise SanitizerError("runtime_file_untrusted")


def read_trusted_file(path, *, limit=MAX_FILE_BYTES):
    """Walk descriptor-relative, without following any directory/file symlink."""
    if (type(path) is not str or not path.startswith("/") or path.startswith("//") or "\0" in path
            or str(Path(path)) != path or ".." in Path(path).parts):
        raise SanitizerError("runtime_path_invalid")
    parts = Path(path).parts[1:]
    if not parts:
        raise SanitizerError("runtime_path_invalid")
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        directory = os.open("/", flags | os.O_DIRECTORY)
    except OSError as exc:
        raise SanitizerError("runtime_file_unreadable") from exc
    try:
        _trusted(os.fstat(directory), directory=True)
        for part in parts[:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=directory)
            os.close(directory)
            directory = child
            _trusted(os.fstat(directory), directory=True)
        fd = os.open(parts[-1], flags, dir_fd=directory)
        try:
            before = os.fstat(fd)
            _trusted(before)
            if not 0 <= before.st_size <= limit:
                raise SanitizerError("runtime_file_too_large")
            data = bytearray()
            while True:
                chunk = os.read(fd, min(1024 * 1024, limit + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > limit:
                    raise SanitizerError("runtime_file_too_large")
            after = os.fstat(fd)
            if _identity(before) != _identity(after) or len(data) != before.st_size:
                raise SanitizerError("runtime_file_changed")
            return bytes(data), _identity(after)
        finally:
            os.close(fd)
    except OSError as exc:
        raise SanitizerError("runtime_file_unreadable") from exc
    finally:
        os.close(directory)


def _readonly(fd):
    return bool(os.fstatvfs(fd).f_flag & os.ST_RDONLY)


def _open_readonly_anchor():
    """A synthetic root is traversal only, never a reinterpretation of UID 0."""
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_DIRECTORY
    root = usr = None
    try:
        root = os.open("/", flags)
        usr = os.open("usr", flags, dir_fd=root)
        root_info, usr_info = os.fstat(root), os.fstat(usr)
        if (not stat.S_ISDIR(root_info.st_mode) or not stat.S_ISDIR(usr_info.st_mode)
                or usr_info.st_mode & 0o022 or not _readonly(usr)
                or (not _readonly(root) and root_info.st_dev == usr_info.st_dev)):
            raise SanitizerError("runtime_readonly_anchor_untrusted")
        return root, usr
    except (OSError, SanitizerError) as exc:
        for fd in (usr, root):
            if fd is not None:
                os.close(fd)
        if isinstance(exc, SanitizerError):
            raise
        raise SanitizerError("runtime_readonly_anchor_unreadable") from exc


def _open_readonly_file(path):
    if (type(path) is not str or not path.startswith("/usr/") or "\0" in path
            or str(Path(path)) != path or ".." in Path(path).parts):
        raise SanitizerError("runtime_readonly_path_invalid")
    root, usr = _open_readonly_anchor()
    fds = [root, usr]
    # Only the root's mount identity matters; its unrelated tmpfs entries may
    # change. Every /usr descendant is itself checked as a read-only object.
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        root_info = os.fstat(root)
        identity = [(root_info.st_dev, root_info.st_ino)]
        parts = Path(path).parts[2:]
        directory = usr
        for part in parts[:-1]:
            identity.append(_identity(os.fstat(directory)))
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=directory)
            fds.append(child)
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o022 or not _readonly(child):
                raise SanitizerError("runtime_readonly_file_untrusted")
            directory = child
        identity.append(_identity(os.fstat(directory)))
        fd = os.open(parts[-1], flags, dir_fd=directory)
        fds.append(fd)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o022
                or info.st_nlink != 1 or not _readonly(fd)):
            raise SanitizerError("runtime_readonly_file_untrusted")
        return fd, identity, fds
    except (OSError, SanitizerError) as exc:
        for fd in reversed(fds):
            os.close(fd)
        if isinstance(exc, SanitizerError):
            raise
        raise SanitizerError("runtime_readonly_file_unreadable") from exc


def _read_readonly_file(path, *, limit=MAX_FILE_BYTES):
    fd, directories, fds = _open_readonly_file(path)
    check_fds = []
    try:
        before = os.fstat(fd)
        if not 0 <= before.st_size <= limit:
            raise SanitizerError("runtime_file_too_large")
        data = bytearray()
        while chunk := os.read(fd, min(1024 * 1024, limit + 1 - len(data))):
            data.extend(chunk)
            if len(data) > limit:
                raise SanitizerError("runtime_file_too_large")
        # Rewalk the actual names, while retaining all original descriptor
        # leases; a renamed leaf/ancestor cannot hide behind the first FD.
        check_fd, current_dirs, check_fds = _open_readonly_file(path)
        if (len(data) != before.st_size or directories != current_dirs
                or _identity(before) != _identity(os.fstat(fd))
                or _identity(before) != _identity(os.fstat(check_fd))
                or not all(_readonly(leased_fd) for leased_fd in fds[1:] + check_fds[1:])):
            raise SanitizerError("runtime_file_changed")
        return bytes(data), _identity(before)
    except OSError as exc:
        raise SanitizerError("runtime_readonly_file_unreadable") from exc
    finally:
        for leased_fd in reversed(check_fds + fds):
            os.close(leased_fd)


def _detect_trust_mode():
    try:
        root, usr = _open_readonly_anchor()
    except SanitizerError:
        return HOST_TRUST
    os.close(usr)
    os.close(root)
    return READONLY_TRUST


def _read_runtime_file(path, *, limit=MAX_FILE_BYTES, trust_mode=HOST_TRUST):
    if trust_mode == HOST_TRUST:
        return read_trusted_file(path, limit=limit)
    if trust_mode == READONLY_TRUST:
        return _read_readonly_file(path, limit=limit)
    raise SanitizerError("runtime_trust_mode_invalid")


def _strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise SanitizerError("runtime_manifest_duplicate_key")
            result[key] = value
        return result
    try:
        def reject_number(_):
            raise SanitizerError("runtime_manifest_invalid_json")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                           parse_int=reject_number, parse_float=reject_number,
                           parse_constant=reject_number)
        def budget(item, depth=0):
            if depth > 12:
                raise SanitizerError("runtime_manifest_too_deep")
            if isinstance(item, dict):
                return 1 + sum(budget(v, depth + 1) for v in item.values())
            if isinstance(item, list):
                return 1 + sum(budget(v, depth + 1) for v in item)
            return 1
        if budget(value) > 2048:
            raise SanitizerError("runtime_manifest_too_many_nodes")
        return value
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, SanitizerError):
            raise
        raise SanitizerError("runtime_manifest_invalid_json") from exc


def validate_manifest(data):
    if (type(data) is not dict or set(data) != {"schema", "platform", "source_revision",
            "files", "compiler_versions", "profiles", "capabilities"}
            or data["schema"] != SCHEMA):
        raise SanitizerError("runtime_manifest_shape")
    if (platform.system() != "Linux" or not _same_data(data["platform"],
            {"system": "Linux", "machine": platform.machine()})):
        raise SanitizerError("runtime_platform_mismatch")
    if not _same_data(data["profiles"], PROFILES) or not _same_data(data["capabilities"], CAPABILITIES):
        raise SanitizerError("runtime_manifest_scope")
    files = data["files"]
    if type(files) is not dict or set(files) != FILE_ROLES:
        raise SanitizerError("runtime_manifest_files")
    for role, entry in files.items():
        if (type(entry) is not dict or set(entry) != {"path", "sha256"}
                or not _sha(entry["sha256"]) or type(entry["path"]) is not str):
            raise SanitizerError("runtime_manifest_file_entry")
        if role in FIXED_FILES and entry["path"] != FIXED_FILES[role]:
            raise SanitizerError("runtime_manifest_file_role")
        if role not in FIXED_FILES and not entry["path"].startswith("/usr/"):
            raise SanitizerError("runtime_manifest_file_role")
    versions = data["compiler_versions"]
    if (type(versions) is not dict or set(versions) != {"gcc", "clang"}
            or any(type(v) is not str or not v.strip() or len(v) > 512
                   or "\n" in v or "\r" in v or "\0" in v for v in versions.values())):
        raise SanitizerError("runtime_compiler_version_invalid")
    if not _sha(data["source_revision"]) or data["source_revision"] != source_revision(files):
        raise SanitizerError("runtime_source_revision_mismatch")


class RuntimeManifest:
    def __init__(self, data, manifest_sha256, identities, *, trust_mode=HOST_TRUST):
        self.data = json.loads(canonical_json(data))
        self.manifest_sha256 = manifest_sha256
        self.identities = dict(identities)
        self._trust_mode = trust_mode

    @property
    def trust_mode(self):
        return self._trust_mode

    def recheck(self):
        fresh = load_manifest(_trust_mode=self.trust_mode)
        if (fresh.manifest_sha256 != self.manifest_sha256
                or fresh.identities != self.identities
                or fresh.trust_mode != self.trust_mode
                or not _same_data(fresh.data, self.data)):
            raise SanitizerError("runtime_pins_changed")


def load_manifest(*, _trust_mode=HOST_TRUST):
    raw, identity = _read_runtime_file(MANIFEST_PATH, limit=64 * 1024, trust_mode=_trust_mode)
    data = _strict_json(raw)
    validate_manifest(data)
    identities = {MANIFEST_PATH: identity}
    paths = set()
    for role, entry in data["files"].items():
        if entry["path"] in paths:
            raise SanitizerError("runtime_file_roles_aliased")
        paths.add(entry["path"])
        contents, identity = _read_runtime_file(entry["path"], trust_mode=_trust_mode)
        if hashlib.sha256(contents).hexdigest() != entry["sha256"]:
            raise SanitizerError("runtime_pin_mismatch")
        if role == "policy_source" and contents != POLICY_SOURCE:
            raise SanitizerError("runtime_policy_source_mismatch")
        if role == "launcher" and contents != LAUNCHER_SOURCE:
            raise SanitizerError("runtime_launcher_source_mismatch")
        if role in {"shim", "gcc_runtime", "gcc_compiler", "clang_compiler"}:
            machine = {"aarch64": 183, "x86_64": 62}.get(platform.machine())
            if (machine is None or len(contents) < 20 or contents[:6] != b"\x7fELF\x02\x01"
                    or struct.unpack_from("<H", contents, 18)[0] != machine):
                raise SanitizerError("runtime_native_elf_invalid")
        identities[entry["path"]] = identity
    return RuntimeManifest(data, hashlib.sha256(raw).hexdigest(), identities, trust_mode=_trust_mode)


def scoped_environment(manifest, profile, environment):
    if type(profile) is not str or profile not in PROFILES:
        raise SanitizerError("runtime_profile_unknown")
    forbidden = ("LD_PRELOAD", "LD_AUDIT", "LD_DYNAMIC_WEAK", "ASAN_OPTIONS",
                 "LSAN_OPTIONS", "UBSAN_OPTIONS")
    if any(environment.get(name) for name in forbidden):
        raise SanitizerError("runtime_environment_conflict")
    result = dict(environment)
    result["LD_PRELOAD"] = ":".join(manifest.data["files"][role]["path"]
                                   for role in PROFILES[profile]["preload"])
    return result


def _metadata(record):
    sys.stderr.write(METADATA_PREFIX + canonical_json(record).decode("ascii") + "\n")
    sys.stderr.flush()


def run_command(manifest, profile, command, *, environment=None, emit=_metadata):
    if (not command or any(type(arg) is not str or "\0" in arg for arg in command)
            or not command[0]):
        raise SanitizerError("runtime_command_invalid")
    environment = scoped_environment(manifest, profile,
                                     os.environ if environment is None else environment)
    manifest.recheck()
    record = {
        "schema": "gv-sanitizer-execution-v1", "execution_id": uuid.uuid4().hex,
        "runtime_profile": profile, "runtime_scope": "declared_compilation_runtime",
        "future_target_compatibility_verified": False, "leak_checking": "unchecked",
        "requested_leak_check_policy": "disabled_unchecked", "absence_of_leaks_proven": False,
        "lsan_policy": "trusted_constant_turnoff_hook", "policy_effective_for_all_targets": False,
        "asan_memory_checks": "not_disabled", "ubsan_checks": "not_disabled",
        "proc_read_grant": False, "compiler_or_target_rewrite": False,
        "link_order_check_disabled": False, "argv": list(command), "cwd": os.getcwd(),
        "manifest_sha256": manifest.manifest_sha256,
        "filesystem_trust_mode": manifest.trust_mode,
        "source_revision": manifest.data["source_revision"],
        "preload": [{"role": role, **manifest.data["files"][role]}
                    for role in PROFILES[profile]["preload"]],
        "child_started": False,
    }
    started = time.monotonic()
    try:
        child = subprocess.Popen(command, env=environment)
    except OSError as exc:
        emit({**record, "event": "infra_error", "reason": "runtime_child_spawn_failed",
              "spawn_error_type": type(exc).__name__})
        return 125
    record.update(child_started=True, child_pid=child.pid)
    emit({**record, "event": "started"})
    returncode = child.wait()  # Intentionally no independent total deadline.
    exitcode = returncode if returncode >= 0 else 128 - returncode
    try:
        manifest.recheck()
    except SanitizerError:
        emit({**record, "event": "infra_error", "reason": "runtime_pins_changed",
              "child_returncode": returncode})
        return 125
    emit({**record, "event": "completed", "child_returncode": returncode,
          "child_exit_code": returncode if returncode >= 0 else None,
          "child_signal": -returncode if returncode < 0 else None,
          "wrapper_exit_code": exitcode,
          "elapsed_seconds": round(time.monotonic() - started, 6)})
    return exitcode


def main(argv=None):
    parser = _Parser(description="Explicit sanitizer runtime profile runner; leaks remain unchecked.")
    parser.add_argument("--runtime-profile", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    try:
        args = parser.parse_args(argv)
        if args.runtime_profile not in PROFILES:
            raise SanitizerError("runtime_profile_unknown")
        if args.command[:1] != ["--"]:
            raise SanitizerError("runtime_command_separator_required")
        command = args.command[1:]
        manifest = load_manifest(_trust_mode=_detect_trust_mode())
        if os.path.abspath(__file__) != FIXED_FILES["runner"]:
            raise SanitizerError("runtime_runner_location_mismatch")
        return run_command(manifest, args.runtime_profile, command)
    except SanitizerError as exc:
        _metadata({"schema": "gv-sanitizer-execution-v1", "event": "infra_error",
                   "reason": exc.reason, "child_started": False,
                   "runtime_scope": "declared_compilation_runtime", "leak_checking": "unchecked",
                   "requested_leak_check_policy": "disabled_unchecked", "absence_of_leaks_proven": False})
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
