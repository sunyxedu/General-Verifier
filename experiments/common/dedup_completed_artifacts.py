"""Content-deduplicate immutable Harbor artifacts from completed trials.

Terminal-Bench jobs record several attempts from the same task.  Large task
inputs (VM images, checkpoints, videos) are consequently copied once per
attempt even when their bytes and filesystem metadata are identical.  This
utility replaces only verified-identical regular files with hard links.

Safety properties:

* the target must look like one concrete Harbor job directory;
* only trial directories containing ``result.json`` are considered;
* symlinks and active/incomplete trials are skipped; files are grouped only
  when mode/owner/timestamps/flags and the complete extended-attribute set
  are identical;
* SHA-256 and a stable stat fingerprint are checked before replacement;
* dry-run is the default; mutation requires ``--apply``.

Candidate world identity hashes file contents, modes, directories, and
symlink targets rather than inode identity.  Hard-linking under these guards
therefore preserves the verifier-visible world while reclaiming duplicate
physical blocks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_MIN_BYTES = 1024 * 1024


class DedupError(RuntimeError):
    """The requested target or an allegedly immutable file was unsafe."""


@dataclass(frozen=True)
class FileFingerprint:
    device: int
    inode: int
    size: int
    mode: int
    uid: int
    gid: int
    mtime_ns: int
    flags: int
    xattrs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CandidateFile:
    path: Path
    fingerprint: FileFingerprint
    digest: str


@dataclass(frozen=True)
class DedupSummary:
    job: str
    completed_trials: int
    eligible_files: int
    duplicate_files: int
    links_created: int
    logical_bytes_deduplicated: int
    dry_run: bool


def _xattr_fingerprint(path: Path) -> tuple[tuple[str, str], ...]:
    """Return every xattr as ``(name, sha256(value))``, or fail closed.

    Some macOS Python builds omit the ``os.*xattr`` APIs even though APFS
    files carry ``com.apple.provenance``.  The system ``xattr`` utility is an
    exact fallback.  Keeping the full xattr set in the metadata key makes
    hard-linking safe without rendering this utility a no-op on those builds.
    """
    if hasattr(os, "listxattr") and hasattr(os, "getxattr"):
        try:
            names = sorted(os.listxattr(path, follow_symlinks=False))
            return tuple(
                (
                    name,
                    hashlib.sha256(os.getxattr(
                        path, name, follow_symlinks=False,
                    )).hexdigest(),
                )
                for name in names
            )
        except OSError as exc:
            raise DedupError(f"cannot inspect xattrs: {path}: {exc}") from exc

    executable = shutil.which("xattr")
    if executable is None:
        raise DedupError(f"cannot inspect xattrs: {path}")
    listed = subprocess.run(
        [executable, os.fspath(path)], capture_output=True, check=False,
    )
    if listed.returncode:
        detail = listed.stderr.decode("utf-8", errors="replace").strip()
        raise DedupError(f"cannot list xattrs: {path}: {detail}")
    result: list[tuple[str, str]] = []
    for raw_name in sorted(listed.stdout.splitlines()):
        if not raw_name:
            continue
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DedupError(f"non-UTF-8 xattr name: {path}") from exc
        value = subprocess.run(
            [executable, "-px", name, os.fspath(path)],
            capture_output=True, check=False,
        )
        if value.returncode:
            detail = value.stderr.decode("utf-8", errors="replace").strip()
            raise DedupError(f"cannot read xattr {name!r}: {path}: {detail}")
        try:
            raw_value = bytes.fromhex(value.stdout.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise DedupError(f"invalid xattr hex for {name!r}: {path}") from exc
        result.append((name, hashlib.sha256(raw_value).hexdigest()))
    return tuple(result)


def _fingerprint(path: Path) -> FileFingerprint:
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise DedupError(f"not a regular file: {path}")
    return FileFingerprint(
        device=info.st_dev,
        inode=info.st_ino,
        size=info.st_size,
        mode=stat.S_IMODE(info.st_mode),
        uid=info.st_uid,
        gid=info.st_gid,
        mtime_ns=info.st_mtime_ns,
        flags=getattr(info, "st_flags", 0),
        xattrs=_xattr_fingerprint(path),
    )


def _sha256(path: Path, expected: FileFingerprint) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = _fingerprint(path)
        if before != expected:
            raise DedupError(f"file changed before hashing: {path}")
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        after = _fingerprint(path)
    if after != expected:
        raise DedupError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _validate_job(job: Path) -> Path:
    resolved = job.resolve(strict=True)
    if not resolved.is_dir():
        raise DedupError(f"job is not a directory: {resolved}")
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        raise DedupError(f"refusing broad target: {resolved}")
    for marker in ("config.json", "lock.json"):
        if not (resolved / marker).is_file():
            raise DedupError(f"not a Harbor job (missing {marker}): {resolved}")
    return resolved


def _completed_artifact_roots(job: Path) -> list[Path]:
    roots: list[Path] = []
    for trial in sorted(job.iterdir()):
        if not trial.is_dir() or not (trial / "result.json").is_file():
            continue
        artifacts = trial / "artifacts"
        if artifacts.is_dir() and not artifacts.is_symlink():
            roots.append(artifacts)
    return roots


def _walk_regular_files(roots: Iterable[Path], min_bytes: int) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            dirnames[:] = [
                name for name in dirnames
                if not (directory_path / name).is_symlink()
            ]
            for name in filenames:
                path = directory_path / name
                try:
                    info = os.lstat(path)
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_size < min_bytes:
                    continue
                paths.append(path)
    return sorted(paths)


def _metadata_key(value: FileFingerprint) -> tuple[object, ...]:
    return (
        value.device, value.size, value.mode, value.uid, value.gid,
        value.mtime_ns, value.flags, value.xattrs,
    )


def deduplicate(
    job: str | os.PathLike[str], *, min_bytes: int = DEFAULT_MIN_BYTES,
    apply: bool = False,
) -> DedupSummary:
    """Deduplicate completed-trial artifacts and return an auditable summary."""
    if min_bytes < 1:
        raise DedupError("min_bytes must be positive")
    job_path = _validate_job(Path(job))
    roots = _completed_artifact_roots(job_path)
    paths = _walk_regular_files(roots, min_bytes)

    metadata_groups: dict[
        tuple[object, ...], list[tuple[Path, FileFingerprint]]
    ] = (
        defaultdict(list)
    )
    for path in paths:
        fingerprint = _fingerprint(path)
        metadata_groups[_metadata_key(fingerprint)].append((path, fingerprint))

    content_groups: dict[
        tuple[tuple[object, ...], str], list[CandidateFile]
    ] = (
        defaultdict(list)
    )
    for metadata, members in metadata_groups.items():
        if len(members) < 2:
            continue
        for path, fingerprint in members:
            digest = _sha256(path, fingerprint)
            content_groups[(metadata, digest)].append(
                CandidateFile(path, fingerprint, digest)
            )

    duplicate_files = 0
    links_created = 0
    logical_bytes = 0
    for members in content_groups.values():
        if len(members) < 2:
            continue
        canonical = members[0]
        for duplicate in members[1:]:
            if (
                canonical.fingerprint.device == duplicate.fingerprint.device
                and canonical.fingerprint.inode == duplicate.fingerprint.inode
            ):
                continue
            duplicate_files += 1
            logical_bytes += duplicate.fingerprint.size
            if not apply:
                continue
            if _fingerprint(canonical.path) != canonical.fingerprint:
                raise DedupError(f"canonical file changed: {canonical.path}")
            if _fingerprint(duplicate.path) != duplicate.fingerprint:
                raise DedupError(f"duplicate file changed: {duplicate.path}")
            if _sha256(canonical.path, canonical.fingerprint) != canonical.digest:
                raise DedupError(f"canonical digest changed: {canonical.path}")
            if _sha256(duplicate.path, duplicate.fingerprint) != duplicate.digest:
                raise DedupError(f"duplicate digest changed: {duplicate.path}")
            temporary = duplicate.path.with_name(
                f".{duplicate.path.name}.gv-dedup-{uuid.uuid4().hex}"
            )
            try:
                os.link(canonical.path, temporary, follow_symlinks=False)
                os.replace(temporary, duplicate.path)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            links_created += 1

    return DedupSummary(
        job=str(job_path),
        completed_trials=len(roots),
        eligible_files=len(paths),
        duplicate_files=duplicate_files,
        links_created=links_created,
        logical_bytes_deduplicated=logical_bytes,
        dry_run=not apply,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    summary = deduplicate(args.job, min_bytes=args.min_bytes, apply=args.apply)
    print(json.dumps(asdict(summary), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
