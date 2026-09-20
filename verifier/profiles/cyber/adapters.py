"""Host-side cyber adapter boundary, independent of agent shell permissions.

These operations observe only declared public subjects and a claim's frozen
source record. They cannot invoke an evaluator or read controller secrets.
"""
from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass

from ...core.events import EventLog
from ...execution.adapters import AdapterContext, AdapterNotApplicable, ResolvedAddress


def _inside(path, root):
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


@dataclass(frozen=True)
class CyberResolvedAddress(ResolvedAddress):
    @contextmanager
    def _parent(self):
        # Bind every ancestor without following symlinks, not only the leaf.
        # A post-admission parent swap cannot redirect this read into secrets.
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
        descriptor = os.open(os.path.sep, flags)
        try:
            parts = self.host_path.split(os.sep)
            for part in parts[1:-1]:
                if part:
                    following = os.open(part, flags, dir_fd=descriptor)
                    os.close(descriptor)
                    descriptor = following
            yield descriptor, parts[-1] or "."
        except FileNotFoundError:
            raise
        except OSError:
            raise AdapterNotApplicable("cyber adapter cannot bind a public artifact parent") from None
        finally:
            os.close(descriptor)

    def lstat(self):
        with self._parent() as (parent, name):
            return os.stat(name, dir_fd=parent, follow_symlinks=False)

    def open_regular(self):
        descriptor = None
        try:
            with self._parent() as (parent, name):
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                     dir_fd=parent)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise AdapterNotApplicable("cyber adapter requires a private regular public file")
            return descriptor, info
        except (OSError, AdapterNotApplicable):
            if descriptor is not None:
                os.close(descriptor)
            raise AdapterNotApplicable("cyber adapter cannot open a public artifact") from None


class CyberAdapterContext(AdapterContext):
    def __init__(self, state_dir, live_world, log, claim_id, project_root):
        record = log.record_path_for(claim_id)
        boundary = EventLog.record_boundary_of(record)
        if (not log.is_record_path(record) or boundary is None
                or record != log.freeze_record(boundary)):
            raise AdapterNotApplicable("cyber adapter requires an authenticated frozen record")
        super().__init__(state_dir, live_world, EventLog(record))
        self.project_root = os.path.realpath(project_root) if project_root else None
        self.allowed_state = tuple(os.path.join(self.state_dir, name) for name in (
            "artifacts", "cyber/blobs", "baseline", "world"))
        self.task_path = os.path.join(self.state_dir, "task.md")
        self.private_roots = tuple(os.path.abspath(path) for path in (
            "/evaluator", "/controller", "/codex-home", "/gv",
            "/root/.codex", "/home/node/.codex",
            os.path.join(os.path.expanduser("~"), ".codex"),
            os.path.join(os.path.expanduser("~"), ".ssh"),
            os.path.join(os.path.expanduser("~"), ".aws")))

    def resolve(self, address):
        resolved = super().resolve(address)
        path = os.path.abspath(resolved.host_path)
        real = os.path.realpath(path)
        # Admission restricts names; CyberResolvedAddress binds all ancestors
        # at open time so a concurrent parent substitution cannot escape it.
        if path != real or any(_inside(path, root) for root in self.private_roots):
            raise AdapterNotApplicable("cyber adapter address is outside the public evidence boundary")
        if _inside(path, self.state_dir):
            allowed = path == self.task_path or any(_inside(path, root) for root in self.allowed_state)
        else:
            allowed = self.project_root is not None and _inside(path, self.project_root)
        if not allowed:
            raise AdapterNotApplicable("cyber adapter address is outside the public evidence boundary")
        return CyberResolvedAddress(resolved.address, path, resolved.mode, resolved.scheme)
