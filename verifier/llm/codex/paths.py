"""Operator-bound directories of the offline profile: verifier-owned
writable roots created without following links, canonical policy roots,
and the empty 0500 directory a restricted session is bound to by an open
descriptor."""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from typing import Any

from ..schema import RoleError


@dataclass(frozen=True)
class _PrivateEmptyDirectory:
    path: str
    descriptor: int
    device: int
    inode: int
    mode: int
    uid: int
    gid: int
    parent_device: int
    parent_inode: int
    parent_mode: int
    parent_uid: int
    parent_gid: int


class _BoundExecutionDirectory(str):
    """Display as the audited path while launching through its bound fd."""

    def __new__(cls, path: str, descriptor: int):
        value = str.__new__(cls, path)
        value.descriptor = descriptor
        return value


def _ensure_verifier_owned_dir(path: str) -> None:
    """Create/check one writable verifier directory without following links."""
    path = os.path.abspath(path)
    if not os.path.lexists(path):
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
    try:
        st = os.lstat(path)
    except OSError as err:
        raise RoleError(f"cannot prepare offline verifier directory {path}: {err}")
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise RoleError(
            f"offline verifier directory must be a real directory: {path}")
    expected_uid = getattr(os, "geteuid", lambda: st.st_uid)()
    if st.st_uid != expected_uid:
        raise RoleError(
            f"offline verifier directory is not owned by this process: {path}")
    flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        fd = os.open(path, flags)
    except OSError as err:
        raise RoleError(
            f"cannot safely open offline verifier directory {path}: {err}")
    try:
        opened = os.fstat(fd)
        if (not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (st.st_dev, st.st_ino)):
            raise RoleError(
                f"offline verifier directory changed while checking: {path}")
        os.fchmod(fd, 0o700)
    finally:
        os.close(fd)


def _ensure_verifier_owned_chain(root: str, target: str) -> None:
    """Check every directory component from ``root`` through ``target``."""
    root = os.path.abspath(root)
    target = os.path.abspath(target)
    try:
        within = os.path.commonpath((root, target)) == root
    except ValueError:
        within = False
    if not within:
        raise RoleError(
            f"offline writable directory escapes verifier work tree: {target}")
    _ensure_verifier_owned_dir(root)
    relative = os.path.relpath(target, root)
    if relative == ".":
        return
    current = root
    for component in relative.split(os.sep):
        if component in ("", ".", ".."):
            raise RoleError(
                f"invalid offline verifier directory component: {target}")
        current = os.path.join(current, component)
        _ensure_verifier_owned_dir(current)


def _path_is_within(root: str, target: str) -> bool:
    """Whether an absolute target is root itself or a descendant."""
    try:
        return os.path.commonpath((root, target)) == root
    except ValueError:
        return False


def _canonical_policy_path(value: Any, label: str) -> str:
    """Normalize a policy root and reject symlinks in any component."""
    path = os.path.abspath(os.fspath(value))
    if os.path.realpath(path) != path:
        raise RoleError(
            f"offline {label} must not contain symlink path components: "
            f"{path}")
    return path


def _bind_private_empty_directory(value: Any,
                                  label: str) -> _PrivateEmptyDirectory:
    path = _canonical_policy_path(value, label)
    parent_path = os.path.dirname(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = None
    try:
        descriptor = os.open(path, flags)
        observed = os.fstat(descriptor)
        entries = tuple(os.listdir(descriptor))
        named = os.lstat(path)
        parent = os.lstat(parent_path)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise RoleError(f"{label} is unavailable: {path}") from error
    named_identity = (named.st_dev, named.st_ino, named.st_mode,
                      named.st_uid, named.st_gid)
    observed_identity = (observed.st_dev, observed.st_ino, observed.st_mode,
                         observed.st_uid, observed.st_gid)
    if (not stat.S_ISDIR(observed.st_mode)
            or observed_identity != named_identity
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) != 0o500
            or entries
            or not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.geteuid()
            or stat.S_IMODE(parent.st_mode) != 0o700):
        os.close(descriptor)
        raise RoleError(
            f"{label} must be an empty operator-bound 0500 directory "
            f"under an operator-owned 0700 parent: {path}")
    return _PrivateEmptyDirectory(
        path, descriptor, observed.st_dev, observed.st_ino, observed.st_mode,
        observed.st_uid, observed.st_gid, parent.st_dev, parent.st_ino,
        parent.st_mode, parent.st_uid, parent.st_gid)


def _revalidate_private_empty_directory(
        binding: _PrivateEmptyDirectory,
        label: str) -> str:
    try:
        observed = os.fstat(binding.descriptor)
        named = os.lstat(binding.path)
        entries = tuple(os.listdir(binding.descriptor))
        parent = os.lstat(os.path.dirname(binding.path))
    except OSError as error:
        raise RoleError(f"{label} is unavailable: {binding.path}") from error
    identity = (
        observed.st_dev, observed.st_ino, observed.st_mode,
        observed.st_uid, observed.st_gid,
        parent.st_dev, parent.st_ino, parent.st_mode,
        parent.st_uid, parent.st_gid,
    )
    expected = (
        binding.device, binding.inode, binding.mode, binding.uid, binding.gid,
        binding.parent_device, binding.parent_inode, binding.parent_mode,
        binding.parent_uid, binding.parent_gid,
    )
    named_identity = (named.st_dev, named.st_ino, named.st_mode,
                      named.st_uid, named.st_gid)
    if (identity != expected
            or named_identity != identity[:5] or entries
            or stat.S_IMODE(observed.st_mode) != 0o500
            or stat.S_IMODE(parent.st_mode) != 0o700):
        raise RoleError(f"{label} identity changed: {binding.path}")
    return binding.path
