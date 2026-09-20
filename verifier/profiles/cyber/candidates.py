"""Immutable, single-file candidates backed by the verifier's artifact/event stores.

Candidates can contain arbitrary bytes (inputs, reports, patches, or programs).
Directories and multi-file bundles must first be packaged into one file. This
module records identity and performs exact copying; it makes no success verdict.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from ...core.events import EventLog
from ...core.model import content_hash
from ...execution.instruments import ArtifactStore

MAX_CANDIDATE_BYTES = 16 * 1024 * 1024
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


class CandidateError(ValueError):
    """A candidate or delivery path failed an identity/integrity check."""


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _read_fd(fd: int) -> bytes:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise CandidateError("candidate must be a regular file")
    if before.st_size > MAX_CANDIDATE_BYTES:
        raise CandidateError("candidate exceeds the 16 MiB limit")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    size = 0
    while chunk := os.read(fd, min(65536, MAX_CANDIDATE_BYTES + 1 - size)):
        chunks.append(chunk)
        size += len(chunk)
        if size > MAX_CANDIDATE_BYTES:
            raise CandidateError("candidate exceeds the 16 MiB limit")
    if _identity(before) != _identity(os.fstat(fd)) or size != before.st_size:
        raise CandidateError("candidate changed while being read")
    return b"".join(chunks)


def _parts(root: str, path: str) -> tuple[str, list[str]]:
    absolute = os.path.abspath(os.path.join(root, path))
    try:
        if os.path.commonpath((root, absolute)) != root:
            raise CandidateError("path is outside the project root")
    except ValueError as exc:
        raise CandidateError("path is outside the project root") from exc
    relative = os.path.relpath(absolute, root)
    parts = relative.split(os.sep)
    if relative == "." or any(part in ("", ".", "..") for part in parts):
        raise CandidateError("path must name a file inside the project root")
    return absolute, parts


def _parent_fd(root: str, parts: list[str]) -> int:
    fd = os.open(root, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
                              dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _regular_fd(root: str, path: str) -> int:
    _, parts = _parts(root, path)
    parent = _parent_fd(root, parts)
    try:
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NONBLOCK | _NOFOLLOW,
                     dir_fd=parent)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise CandidateError("candidate must be a regular file")
        return fd
    finally:
        os.close(parent)


def _remove_exact_entry(parent: int, name: str, expected_inode: tuple[int, int],
                        digest: str, size: int, check_location=None) -> bool:
    """Capture an entry atomically before removing only the owned exact bytes.

    A concurrently substituted leaf is restored without overwriting a newer
    entry. The retained parent descriptor also allows an aborted delivery to
    clean up its own output after that directory has been moved elsewhere.
    """
    quarantine = f".rollback-{secrets.token_hex(16)}"
    captured = removed = False
    try:
        os.rename(name, quarantine, src_dir_fd=parent, dst_dir_fd=parent)
        captured = True
        fd = os.open(quarantine, os.O_RDONLY | os.O_NONBLOCK | _NOFOLLOW, dir_fd=parent)
        try:
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != expected_inode:
                return False
            data = _read_fd(fd)
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                return False
            linked = os.stat(quarantine, dir_fd=parent, follow_symlinks=False)
            if (linked.st_dev, linked.st_ino) != expected_inode:
                return False
            if check_location is not None:
                check_location()
            os.unlink(quarantine, dir_fd=parent)
            removed = True
            os.fsync(parent)
            return True
        finally:
            os.close(fd)
    except (CandidateError, OSError):
        return False
    finally:
        if captured and not removed:
            try:
                os.link(quarantine, name, src_dir_fd=parent, dst_dir_fd=parent,
                        follow_symlinks=False)
                os.unlink(quarantine, dir_fd=parent)
            except OSError:
                # Retain the captured entry if another file now occupies the
                # original name; neither entry may be overwritten or deleted.
                pass


@dataclass
class CandidateLease:
    candidate: dict[str, Any]
    fd: int
    identity: tuple[int, ...]
    active: bool = True


class CandidateStore:
    def __init__(self, state_dir: str, project_root: str,
                 artifacts: ArtifactStore, log: EventLog, *, prepare_delivery=None):
        if prepare_delivery is not None and not callable(prepare_delivery):
            raise CandidateError("delivery preparation hook must be callable")
        self._prepare_delivery = prepare_delivery
        self.state_dir = os.path.realpath(state_dir)
        self.project_root = os.path.realpath(project_root)
        self.artifacts, self.log = artifacts, log
        os.makedirs(self.state_dir, exist_ok=True)
        self.blob_dir = os.path.join(self.state_dir, "cyber", "blobs")
        os.makedirs(self.blob_dir, mode=0o700, exist_ok=True)
        # Opening every component without following links is also a startup
        # integrity check; the configured roots are host-owned directories.
        check = _parent_fd(self.state_dir, ["cyber", "blobs", "probe"])
        os.close(check)

    def _checked_metadata(self, candidate: dict[str, Any]) -> dict[str, Any]:
        candidate = dict(candidate)
        digest = candidate.get("sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or candidate.get("candidate_id") != digest
                or candidate.get("blob_path") != os.path.join(self.blob_dir, digest)
                or type(candidate.get("size")) is not int
                or not 0 <= candidate["size"] <= MAX_CANDIDATE_BYTES):
            raise CandidateError("invalid candidate metadata")
        artifact_id = candidate.get("artifact_id")
        if (not isinstance(artifact_id, str) or len(artifact_id) != 64
                or any(c not in "0123456789abcdef" for c in artifact_id)):
            raise CandidateError("candidate artifact identity is missing")
        try:
            artifact = self.artifacts.get(artifact_id)
        except (OSError, ValueError) as exc:
            raise CandidateError("candidate artifact is unavailable") from exc
        expected = {k: v for k, v in candidate.items() if k != "artifact_id"}
        if content_hash(artifact) != artifact_id or artifact != expected:
            raise CandidateError("candidate artifact identity mismatch")
        return candidate

    def candidates(self) -> list[dict[str, Any]]:
        registered: dict[str, dict[str, Any]] = {}
        for event in self.log.replay():
            if event["type"] == "cyber_candidate_registered":
                candidate = self._checked_metadata(event["payload"]["candidate"])
                registered.setdefault(candidate["candidate_id"], candidate)
        return list(registered.values())

    def get(self, candidate_id: str) -> dict[str, Any]:
        for candidate in self.candidates():
            if candidate["candidate_id"] == candidate_id:
                return candidate
        raise CandidateError("candidate was not registered in the canonical event log")

    def _source_bytes(self, path: str) -> tuple[str, bytes]:
        if not isinstance(path, str):
            raise CandidateError("invalid candidate path")
        absolute, _ = _parts(self.project_root, path)
        try:
            fd = _regular_fd(self.project_root, absolute)
            try:
                data = _read_fd(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            raise CandidateError("candidate path is unavailable or contains a symlink") from exc
        return absolute, data

    def source_identity(self, path: str) -> dict[str, Any]:
        """Bind a staging source safely before retaining a registration job."""
        absolute, data = self._source_bytes(path)
        return {"source_path": absolute, "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data)}

    def register(self, path: str, label: str = "", *,
                 expected_identity: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(label, str) or len(label) > 4096:
            raise CandidateError("invalid candidate path or label")
        absolute, data = self._source_bytes(path)
        digest = hashlib.sha256(data).hexdigest()
        actual_identity = {"source_path": absolute, "sha256": digest, "size": len(data)}
        # Compare the bytes actually being registered, not a separate earlier
        # hash/read. No blob, artifact or event may be written after a mismatch.
        if expected_identity is not None and expected_identity != actual_identity:
            raise CandidateError("candidate source changed while queued; resubmit")
        blob_path = os.path.join(self.blob_dir, digest)
        blob_fd = _parent_fd(self.state_dir, ["cyber", "blobs", digest])
        temporary = f".candidate-{secrets.token_hex(16)}"
        written = None
        try:
            written = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                              0o400, dir_fd=blob_fd)
            view = memoryview(data)
            while view:
                view = view[os.write(written, view):]
            os.fsync(written)
            os.close(written)
            written = None
            try:
                os.link(temporary, digest, src_dir_fd=blob_fd, dst_dir_fd=blob_fd,
                        follow_symlinks=False)
            except FileExistsError:
                pass
            os.fsync(blob_fd)
        finally:
            if written is not None:
                os.close(written)
            try:
                os.unlink(temporary, dir_fd=blob_fd)
            except FileNotFoundError:
                pass
            finally:
                os.close(blob_fd)
        # Existing content-addressed blobs are never silently overwritten.
        fd = _regular_fd(self.state_dir, blob_path)
        try:
            if _read_fd(fd) != data:
                raise CandidateError("existing candidate blob is corrupt")
        finally:
            os.close(fd)
        metadata = {"schema": "cyber-candidate-v1", "candidate_id": digest,
                    "blob_path": blob_path, "sha256": digest, "size": len(data),
                    "source_path": absolute, "label": label}
        metadata["artifact_id"] = self.artifacts.put(dict(metadata))
        self.log.append("cyber_candidate_registered", {"candidate": metadata})
        return dict(metadata)

    @contextmanager
    def lease(self, candidate_id: str) -> Iterator[CandidateLease]:
        candidate = self.get(candidate_id)
        try:
            fd = _regular_fd(self.state_dir, candidate["blob_path"])
        except OSError as exc:
            raise CandidateError("candidate blob cannot be safely opened") from exc
        lease = CandidateLease(candidate, fd, _identity(os.fstat(fd)))
        try:
            self.validate(lease)
            yield lease
            self.validate(lease)
        finally:
            lease.active = False
            os.close(fd)

    def validate(self, lease: CandidateLease) -> dict[str, Any]:
        if not isinstance(lease, CandidateLease) or not lease.active:
            raise CandidateError("candidate lease is inactive")
        candidate = self.get(lease.candidate["candidate_id"])
        if candidate != lease.candidate or _identity(os.fstat(lease.fd)) != lease.identity:
            raise CandidateError("candidate lease identity changed")
        try:
            current = _regular_fd(self.state_dir, candidate["blob_path"])
            try:
                if _identity(os.fstat(current)) != lease.identity:
                    raise CandidateError("candidate blob path identity changed")
            finally:
                os.close(current)
            data = _read_fd(lease.fd)
        except OSError as exc:
            raise CandidateError("candidate blob became unavailable") from exc
        if (len(data) != candidate["size"]
                or hashlib.sha256(data).hexdigest() != candidate["sha256"]):
            raise CandidateError("candidate blob digest mismatch")
        return candidate

    def preflight_destination(self, destination: str) -> str:
        """Check delivery addressing before invoking any model roles."""
        absolute, parts = _parts(self.project_root, destination)
        try:
            parent = _parent_fd(self.project_root, parts)
            try:
                info = os.fstat(parent)
                try:
                    os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise CandidateError("destination already exists; refusing to overwrite")
                check = _parent_fd(self.project_root, parts)
                try:
                    current = os.fstat(check)
                    if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                        raise CandidateError("destination parent changed during preflight")
                finally:
                    os.close(check)
            finally:
                os.close(parent)
            return absolute
        except OSError as exc:
            raise CandidateError("destination parent is unavailable or contains a symlink") from exc

    def _delivery_metadata(self, receipt: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(receipt, dict) or receipt.get("schema") != "cyber-materialization-v1":
            raise CandidateError("invalid materialization receipt")
        candidate = self.get(receipt.get("candidate_id"))
        if (receipt.get("sha256") != candidate["sha256"] or receipt.get("size") != candidate["size"]
                or any(type(receipt.get(k)) is not int or receipt[k] < 0
                       for k in ("device", "inode", "parent_device", "parent_inode"))
                or not isinstance(receipt.get("destination"), str)):
            raise CandidateError("materialization receipt identity mismatch")
        absolute, parts = _parts(self.project_root, receipt["destination"])
        if absolute != receipt["destination"]:
            raise CandidateError("materialization receipt path is not canonical")
        return candidate, parts

    def verify_delivery(self, receipt: dict[str, Any]) -> dict[str, Any]:
        """Recheck the actual canonical leaf immediately before recording delivery."""
        candidate, parts = self._delivery_metadata(receipt)
        expected_parent = (receipt["parent_device"], receipt["parent_inode"])
        try:
            parent = _parent_fd(self.project_root, parts)
            try:
                parent_info = os.fstat(parent)
                if (parent_info.st_dev, parent_info.st_ino) != expected_parent:
                    raise CandidateError("materialization parent identity changed")
                fd = os.open(parts[-1], os.O_RDONLY | os.O_NONBLOCK | _NOFOLLOW, dir_fd=parent)
                try:
                    info = os.fstat(fd)
                    if (info.st_dev, info.st_ino) != (receipt["device"], receipt["inode"]):
                        raise CandidateError("materialized leaf identity changed")
                    data = _read_fd(fd)
                    if len(data) != candidate["size"] or hashlib.sha256(data).hexdigest() != candidate["sha256"]:
                        raise CandidateError("materialized leaf digest mismatch")
                    linked = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
                    if (linked.st_dev, linked.st_ino) != (info.st_dev, info.st_ino):
                        raise CandidateError("materialized leaf changed during verification")
                finally:
                    os.close(fd)
                check = _parent_fd(self.project_root, parts)
                try:
                    current = os.fstat(check)
                    if (current.st_dev, current.st_ino) != expected_parent:
                        raise CandidateError("materialization parent changed during verification")
                finally:
                    os.close(check)
            finally:
                os.close(parent)
            return dict(receipt)
        except OSError as exc:
            raise CandidateError("materialized output is unavailable or contains a symlink") from exc

    def rollback(self, receipt: dict[str, Any]) -> bool:
        """Remove only this receipt's exact output after a failed host event write.

        An atomic rename captures the directory entry before inspecting it. A
        replaced file or symlink is restored, never deleted. If another entry
        concurrently occupies the original name, the captured replacement is
        retained under its quarantine name instead of overwriting either file.
        """
        try:
            candidate, parts = self._delivery_metadata(receipt)
            parent = _parent_fd(self.project_root, parts)
        except (CandidateError, OSError):
            return False
        expected_parent = (receipt["parent_device"], receipt["parent_inode"])

        def check_parent() -> None:
            check = _parent_fd(self.project_root, parts)
            try:
                info = os.fstat(check)
                if (info.st_dev, info.st_ino) != expected_parent:
                    raise CandidateError("rollback parent identity changed")
            finally:
                os.close(check)

        try:
            info = os.fstat(parent)
            if (info.st_dev, info.st_ino) != expected_parent:
                return False
            check_parent()
            return _remove_exact_entry(parent, parts[-1], (receipt["device"], receipt["inode"]),
                                       candidate["sha256"], candidate["size"], check_parent)
        except (CandidateError, OSError):
            return False
        finally:
            os.close(parent)

    def materialize(self, lease: CandidateLease, destination: str) -> dict[str, Any]:
        """Create a new output atomically; existing files are always refused.

        The destination must have an existing, symlink-free parent inside the
        project. Superseding earlier outputs is deliberately not supported.
        """
        candidate = self.validate(lease)
        absolute = self.preflight_destination(destination)
        _, parts = _parts(self.project_root, absolute)
        try:
            parent = _parent_fd(self.project_root, parts)
        except OSError as exc:
            raise CandidateError("destination parent is unavailable or contains a symlink") from exc
        parent_identity = (os.fstat(parent).st_dev, os.fstat(parent).st_ino)
        temporary = f".delivery-{secrets.token_hex(16)}"
        fd = None
        linked = False
        delivered_identity = None
        preparing = False

        def check_parent() -> None:
            check = _parent_fd(self.project_root, parts)
            try:
                if (os.fstat(check).st_dev, os.fstat(check).st_ino) != parent_identity:
                    raise CandidateError("destination parent changed during delivery")
            finally:
                os.close(check)

        try:
            try:
                os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise CandidateError("destination already exists; refusing to overwrite")
            fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
                         0o400, dir_fd=parent)
            data = _read_fd(lease.fd)
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
            self.validate(lease)
            check_parent()
            info = os.fstat(fd)
            delivered_identity = (info.st_dev, info.st_ino)
            receipt = {"schema": "cyber-materialization-v1", "candidate_id": candidate["candidate_id"],
                       "destination": absolute, "sha256": candidate["sha256"],
                       "size": candidate["size"], "device": delivered_identity[0],
                       "inode": delivered_identity[1], "parent_device": parent_identity[0],
                       "parent_inode": parent_identity[1]}
            if self._prepare_delivery is not None:
                # Capture an immutable expectation before any canonical name
                # exists. The hook persists intent; it cannot select bytes,
                # inodes, destination, or a replacement receipt.
                if (_read_fd(fd) != data
                        or hashlib.sha256(data).hexdigest() != candidate["sha256"]):
                    raise CandidateError("prepared output digest mismatch")
                prepared = dict(receipt)
                preparing = True
                result = self._prepare_delivery(prepared)
                preparing = False
                if result is not None:
                    raise CandidateError("delivery preparation hook must return None")
                if (set(prepared) != set(receipt)
                        or any(type(prepared[key]) is not type(value)
                               or prepared[key] != value for key, value in receipt.items())):
                    raise CandidateError("delivery preparation hook changed the receipt")
                self.validate(lease)
                check_parent()
                current = os.fstat(fd)
                temporary_info = os.stat(temporary, dir_fd=parent, follow_symlinks=False)
                if (_identity(current) != _identity(info)
                        or (temporary_info.st_dev, temporary_info.st_ino) != delivered_identity
                        or not stat.S_ISREG(temporary_info.st_mode)
                        or temporary_info.st_nlink != 1
                        or _read_fd(fd) != data):
                    raise CandidateError("prepared output changed before publication")
            os.link(temporary, parts[-1], src_dir_fd=parent, dst_dir_fd=parent,
                    follow_symlinks=False)
            linked = True
            info = os.fstat(fd)
            delivered_identity = (info.st_dev, info.st_ino)
            check_parent()
            if (_read_fd(fd) != data or hashlib.sha256(data).hexdigest() != candidate["sha256"]):
                raise CandidateError("materialized output digest mismatch")
            self.validate(lease)
            delivered = os.open(parts[-1], os.O_RDONLY | os.O_NONBLOCK | _NOFOLLOW,
                                dir_fd=parent)
            try:
                info = os.fstat(delivered)
                if ((info.st_dev, info.st_ino) != delivered_identity
                        or _read_fd(delivered) != data):
                    raise CandidateError("canonical output changed during delivery")
            finally:
                os.close(delivered)
            check_parent()
            os.fsync(parent)
            return receipt
        except (OSError, CandidateError) as exc:
            if linked and delivered_identity is not None:
                _remove_exact_entry(parent, parts[-1], delivered_identity,
                                    candidate["sha256"], candidate["size"])
            if preparing:
                raise
            if isinstance(exc, CandidateError):
                raise
            raise CandidateError("exact materialization failed") from exc
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)
