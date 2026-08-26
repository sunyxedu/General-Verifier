"""Safe interpretation of Harbor's collected-artifact manifest.

Harbor can leave a partially copied destination behind when collection fails.
Consumers must therefore trust the manifest status, not mere host-path
existence.  This module is intentionally independent of any benchmark task so
offline scoring and candidate sealing share exactly the same capture rules.
"""
from __future__ import annotations

import json
import os
import posixpath
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CapturedArtifact:
    """One manifest entry whose collection completed successfully.

    ``host_path`` names the captured filesystem node itself.  In particular,
    it is not resolved through a final symlink: Harbor legitimately preserves
    symlinks whose targets use the container's absolute namespace and are
    therefore dangling from the host's point of view.
    """

    source: str
    destination: str
    host_path: Path


def load_manifest(trial_dir: str | os.PathLike[str]) -> list[Any] | None:
    """Return a Harbor artifact manifest, or ``None`` for legacy jobs.

    Scoring historically tolerated missing and malformed manifests and fell
    back to ``artifacts/app``.  Callers that require a sealed manifest can
    distinguish ``None`` and fail closed.
    """
    path = Path(trial_dir) / "artifacts" / "manifest.json"
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, list) else None


def _safe_source(value: Any) -> str | None:
    if not isinstance(value, str) or not value.startswith("/"):
        return None
    if value in ("/", "/logs/artifacts") or "\x00" in value:
        return None
    if posixpath.normpath(value) != value:
        return None
    return value


def _safe_host_path(
    trial: Path, artifacts: Path, destination: Any,
) -> Path | None:
    """Resolve a manifest destination without following its final node.

    Resolving the leaf itself rejects valid captured symlinks: an absolute
    target such as ``/etc/foo`` belongs to the original container namespace,
    not the host namespace.  Resolve and constrain the parent instead, then
    use ``lstat`` semantics for the leaf.  This still rejects traversal and a
    symlinked parent that escapes ``artifacts/``.
    """
    if (not isinstance(destination, str) or not destination
            or "\x00" in destination
            or posixpath.normpath(destination) != destination):
        return None
    parts = Path(destination).parts
    if len(parts) < 2 or parts[0] != "artifacts":
        return None

    artifacts_real = os.path.realpath(artifacts)
    lexical = trial / destination
    parent_real = os.path.realpath(lexical.parent)
    try:
        if os.path.commonpath((artifacts_real, parent_real)) != artifacts_real:
            return None
    except ValueError:
        return None

    host = Path(parent_real) / lexical.name
    try:
        mode = os.lstat(host).st_mode
    except OSError:
        return None
    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
        return None
    return host


def successful_captures(
    trial_dir: str | os.PathLike[str], *,
    manifest: Iterable[Any] | None = None,
) -> list[CapturedArtifact]:
    """Resolve only ``status=ok`` entries that remain inside artifacts/.

    Failed, empty, malformed, missing, and escaping entries are excluded even
    when a partial destination exists.  If no readable manifest exists, the
    legacy ``artifacts/app`` layout is recognized.
    """
    trial = Path(trial_dir)
    artifacts = trial / "artifacts"
    entries = load_manifest(trial) if manifest is None else list(manifest)
    if entries is None:
        legacy = artifacts / "app"
        if legacy.is_dir() and not legacy.is_symlink():
            return [CapturedArtifact("/app", "artifacts/app", legacy)]
        return []

    captures: dict[str, CapturedArtifact] = {}
    conflicts: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("status") != "ok":
            continue
        source = _safe_source(entry.get("source"))
        destination = entry.get("destination")
        if source is None or not isinstance(destination, str) or not destination:
            continue
        host = _safe_host_path(trial, artifacts, destination)
        if host is None:
            continue
        capture = CapturedArtifact(source, destination, host)
        prior = captures.get(source)
        if prior is not None and prior != capture:
            conflicts.add(source)
            continue
        captures[source] = capture

    for source in conflicts:
        captures.pop(source, None)
    return [captures[source] for source in sorted(captures)]
