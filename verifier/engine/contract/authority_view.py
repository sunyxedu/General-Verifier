"""Candidate-independent authority snapshot for final-contract compilation.

This module does not decide cache eligibility or run a model.  It reduces the
only reusable semantic inputs to immutable prompt bytes and an exact identity:
the harness task statement and, when present, the project's pre-task baseline
metadata.  Candidate outputs, the solver ledger, delivered-world data, and
physical host paths are never read or rendered.

Callers must still give the Compiler and Critic a genuinely no-tools backend
boundary.  Inlining this view is not a substitute for capability isolation.
"""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import stat
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


AUTHORITY_VIEW_SCHEMA = "gv-authority-compilation-view-v1"
MAX_TASK_BYTES = 256 * 1024
MAX_BASELINE_JSON_BYTES = 2 * 1024 * 1024
MAX_AUTHORITY_CONTEXT_BYTES = 4 * 1024 * 1024
MAX_BASELINE_FILES = 20_000
MAX_LOGICAL_PATH_BYTES = 4_096

_GIT_FIELDS = frozenset({"head", "refs", "status"})
# the pre-task baseline: git state and tree manifest (the authority),
# plus the environment fingerprint the task supplied (recorded alongside,
# not part of the minimal-change authority)
_BASELINE_REQUIRED = frozenset({"git.json", "manifest.json"})
_BASELINE_FILES = _BASELINE_REQUIRED | {"environment.json",
                                        "repositories.json"}
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_SIZE_IDENTITY = re.compile(r"size:(0|[1-9][0-9]*)")


class AuthorityViewUnavailable(ValueError):
    """The authority input cannot safely support reusable compilation."""


@dataclass(frozen=True)
class AuthorityCompilationView:
    """Frozen prompt context and its candidate-independent content identity."""

    schema: str
    context_json: str
    identity: str

    def context(self) -> Dict[str, Any]:
        """Return a detached decoded copy for diagnostics and tests."""
        return json.loads(self.context_json)

    def revalidate(self, state_dir: str) -> "AuthorityCompilationView":
        """Re-snapshot authority and fail if its exact semantics changed."""
        current = build_authority_compilation_view(state_dir)
        if current.identity != self.identity \
                or current.context_json != self.context_json:
            raise AuthorityViewUnavailable(
                "authority changed after the compilation view was built")
        return current


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _directory_flags() -> int:
    return (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0))


def _file_flags() -> int:
    return (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0))


def _same_file(before: os.stat_result, after: os.stat_result) -> bool:
    fields = (
        "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
        "st_mtime_ns", "st_ctime_ns",
    )
    return all(getattr(before, field) == getattr(after, field)
               for field in fields)


def _open_state_dir(state_dir: str) -> int:
    path = os.path.abspath(os.fspath(state_dir))
    if os.path.realpath(path) != path:
        raise AuthorityViewUnavailable(
            "authority state path must not traverse symlinks")
    try:
        expected = os.lstat(path)
    except OSError as error:
        raise AuthorityViewUnavailable(
            f"cannot open authority state directory: {error}") from error
    if not stat.S_ISDIR(expected.st_mode) or stat.S_ISLNK(expected.st_mode):
        raise AuthorityViewUnavailable(
            "authority state must be a stable real directory")
    fd = -1
    try:
        fd = os.open(path, _directory_flags())
        opened = os.fstat(fd)
    except OSError as error:
        if fd >= 0:
            os.close(fd)
        raise AuthorityViewUnavailable(
            f"cannot open authority state directory: {error}") from error
    if (not stat.S_ISDIR(opened.st_mode)
            or (expected.st_dev, expected.st_ino)
            != (opened.st_dev, opened.st_ino)):
        os.close(fd)
        raise AuthorityViewUnavailable(
            "authority state must be a stable real directory")
    return fd


def _open_directory_at(parent_fd: int, name: str) -> int:
    try:
        expected = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise AuthorityViewUnavailable(
            f"cannot open authority directory {name!r}: {error}") from error
    if not stat.S_ISDIR(expected.st_mode) or stat.S_ISLNK(expected.st_mode):
        raise AuthorityViewUnavailable(
            f"authority directory {name!r} is not stable and real")
    fd = -1
    try:
        fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
        opened = os.fstat(fd)
    except OSError as error:
        if fd >= 0:
            os.close(fd)
        raise AuthorityViewUnavailable(
            f"cannot open authority directory {name!r}: {error}") from error
    if (not stat.S_ISDIR(opened.st_mode)
            or (expected.st_dev, expected.st_ino)
            != (opened.st_dev, opened.st_ino)):
        os.close(fd)
        raise AuthorityViewUnavailable(
            f"authority directory {name!r} is not stable and real")
    return fd


def _read_regular_at(parent_fd: int, name: str, limit: int) -> bytes:
    try:
        expected = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise AuthorityViewUnavailable(
            f"cannot open authority file {name!r}: {error}") from error
    if (not stat.S_ISREG(expected.st_mode)
            or stat.S_ISLNK(expected.st_mode)
            or expected.st_nlink != 1):
        raise AuthorityViewUnavailable(
            f"authority file {name!r} must be an unlinked regular file")
    if expected.st_size > limit:
        raise AuthorityViewUnavailable(
            f"authority file {name!r} exceeds {limit} bytes")
    try:
        fd = os.open(name, _file_flags(), dir_fd=parent_fd)
    except OSError as error:
        raise AuthorityViewUnavailable(
            f"cannot open authority file {name!r}: {error}") from error
    try:
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode)
                or (expected.st_dev, expected.st_ino)
                != (opened.st_dev, opened.st_ino)
                or opened.st_nlink != 1):
            raise AuthorityViewUnavailable(
                f"authority file {name!r} must be an unlinked regular file")
        if opened.st_size > limit:
            raise AuthorityViewUnavailable(
                f"authority file {name!r} exceeds {limit} bytes")
        chunks = []
        total = 0
        while True:
            block = os.read(fd, min(1024 * 1024, limit + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > limit:
                raise AuthorityViewUnavailable(
                    f"authority file {name!r} exceeds {limit} bytes")
        closed = os.fstat(fd)
        if not _same_file(opened, closed):
            raise AuthorityViewUnavailable(
                f"authority file {name!r} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _object_without_duplicates(
    pairs: Iterable[Tuple[str, Any]],
) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _no_constants(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _strict_json(data: bytes, name: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_no_constants,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError,
            RecursionError) as error:
        raise AuthorityViewUnavailable(
            f"authority file {name!r} is not strict UTF-8 JSON: {error}"
        ) from error


def _task(data: bytes) -> Dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuthorityViewUnavailable(
            "task.md is not valid UTF-8") from error
    if not text.strip():
        raise AuthorityViewUnavailable("task.md does not contain a task")
    if "\x00" in text:
        raise AuthorityViewUnavailable("task.md contains a NUL byte")
    return {"logical_source": "task.md", "text": text}


def _git(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict) or not set(value) <= _GIT_FIELDS:
        raise AuthorityViewUnavailable(
            "baseline/git.json has an unsupported shape")
    out: Dict[str, str] = {}
    for key in sorted(value):
        item = value[key]
        if not isinstance(item, str) or "\x00" in item:
            raise AuthorityViewUnavailable(
                f"baseline/git.json field {key!r} must be a string")
        out[key] = item
    return out


def _logical_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AuthorityViewUnavailable(
            "baseline manifest paths must be non-empty strings")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise AuthorityViewUnavailable(
            "baseline manifest path is not valid UTF-8") from error
    if len(encoded) > MAX_LOGICAL_PATH_BYTES \
            or value.startswith("/") or value.startswith("//") \
            or posixpath.normpath(value) != value \
            or value in {".", ".."} or value.startswith("../"):
        raise AuthorityViewUnavailable(
            f"baseline manifest path is not canonical: {value!r}")
    return value


def _file_identity(value: Any) -> Dict[str, Any]:
    if not isinstance(value, str) or "\x00" in value:
        raise AuthorityViewUnavailable(
            "baseline manifest identities must be strings")
    if _HEX_SHA256.fullmatch(value):
        return {"kind": "sha256", "sha256": value}
    size = _SIZE_IDENTITY.fullmatch(value)
    if size:
        return {"kind": "size", "bytes": int(size.group(1))}
    if value.startswith("link:"):
        target = value[len("link:"):]
        if not target:
            raise AuthorityViewUnavailable(
                "baseline symlink identity has an empty target")
        try:
            target_bytes = target.encode("utf-8")
        except UnicodeEncodeError as error:
            raise AuthorityViewUnavailable(
                "baseline symlink target is not valid UTF-8") from error
        return {
            "kind": "symlink",
            # Absolute targets can contain machine-local roots.  Their exact
            # bytes remain part of authority identity without entering the
            # shared prompt.
            "target_sha256": hashlib.sha256(
                target_bytes).hexdigest(),
        }
    raise AuthorityViewUnavailable(
        "baseline manifest contains an unsupported file identity")


def _manifest(value: Any) -> Tuple[Dict[str, Any], ...]:
    if not isinstance(value, dict) or set(value) != {"root", "files"} \
            or not isinstance(value["root"], str) \
            or not isinstance(value["files"], dict):
        raise AuthorityViewUnavailable(
            "baseline/manifest.json has an unsupported shape")
    files = value["files"]
    if len(files) > MAX_BASELINE_FILES:
        raise AuthorityViewUnavailable(
            f"baseline manifest exceeds {MAX_BASELINE_FILES} files")
    normalized = []
    for path, identity in files.items():
        normalized.append({
            "path": _logical_path(path),
            "identity": _file_identity(identity),
        })
    normalized.sort(key=lambda item: item["path"])
    return tuple(normalized)


def _baseline(state_fd: int) -> Dict[str, Any]:
    try:
        present = os.stat(
            "baseline", dir_fd=state_fd, follow_symlinks=False)
    except FileNotFoundError:
        return {
            "status": "absent",
            "minimal_change_authority": (
                "unavailable; do not generate baseline-preservation or "
                "minimal-change obligations"
            ),
        }
    except OSError as error:
        raise AuthorityViewUnavailable(
            f"cannot inspect baseline authority: {error}") from error
    if not stat.S_ISDIR(present.st_mode) or stat.S_ISLNK(present.st_mode):
        raise AuthorityViewUnavailable(
            "baseline authority must be absent or a real directory")

    baseline_fd = _open_directory_at(state_fd, "baseline")
    try:
        opened = os.fstat(baseline_fd)
        try:
            with os.scandir(baseline_fd) as entries:
                names = []
                for entry in entries:
                    names.append(entry.name)
                    if len(names) > len(_BASELINE_FILES):
                        break
        except OSError as error:
            raise AuthorityViewUnavailable(
                f"cannot enumerate baseline authority: {error}") from error
        if (len(names) != len(set(names))
                or not _BASELINE_REQUIRED <= set(names) <= _BASELINE_FILES):
            raise AuthorityViewUnavailable(
                "baseline authority must contain git.json and manifest.json "
                "and at most environment.json and repositories.json besides")
        git = _git(_strict_json(
            _read_regular_at(
                baseline_fd, "git.json", MAX_BASELINE_JSON_BYTES),
            "baseline/git.json",
        ))
        files = _manifest(_strict_json(
            _read_regular_at(
                baseline_fd, "manifest.json", MAX_BASELINE_JSON_BYTES),
            "baseline/manifest.json",
        ))
        result = {
            "status": "present",
            "minimal_change_authority": "available",
            "git": git,
            "files": list(files),
        }
        closed = os.fstat(baseline_fd)
        if not _same_file(opened, closed):
            raise AuthorityViewUnavailable(
                "baseline authority changed while being read")
        return result
    finally:
        os.close(baseline_fd)


def _capture(state_dir: str) -> Dict[str, Any]:
    state_fd = _open_state_dir(state_dir)
    try:
        task = _task(_read_regular_at(state_fd, "task.md", MAX_TASK_BYTES))
        baseline = _baseline(state_fd)
    finally:
        os.close(state_fd)
    return {
        "schema": AUTHORITY_VIEW_SCHEMA,
        "task": task,
        "baseline": baseline,
    }


def build_authority_compilation_view(
    state_dir: str,
) -> AuthorityCompilationView:
    """Build a bounded authority-only view, or fail closed.

    Two complete reads make a mutation during construction observable.  A
    caller publishing a cold cache fill must additionally call ``revalidate``
    after the Compiler/Critic finish, because no portable userspace read can
    prevent an authority owner from changing a file after it was read.
    """
    first = _canonical(_capture(state_dir))
    second = _canonical(_capture(state_dir))
    if first != second:
        raise AuthorityViewUnavailable(
            "authority changed while the compilation view was built")
    encoded = second.encode("utf-8")
    if len(encoded) > MAX_AUTHORITY_CONTEXT_BYTES:
        raise AuthorityViewUnavailable(
            f"authority context exceeds {MAX_AUTHORITY_CONTEXT_BYTES} bytes")
    return AuthorityCompilationView(
        schema=AUTHORITY_VIEW_SCHEMA,
        context_json=second,
        identity=hashlib.sha256(encoded).hexdigest(),
    )
