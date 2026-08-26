"""Capture task-agnostic post-agent runtime evidence inside a Harbor sandbox.

The ordinary Harbor artifact copier preserves files but not volatile process or
network-namespace state.  Some Terminal-Bench contracts intentionally leave a
service running, so an offline verifier otherwise cannot distinguish a daemon
that survived the agent shell from one that died with it.

This collector runs *after* the solver returns and *before* Harbor invokes the
official verifier.  It records raw procfs observations and enough relational
structure to join listening socket inodes to process file descriptors.  It
does not inspect process environments, task-specific ports, protocols, or
expected answers.  The output is evidence, not a verdict.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "gv-post-run-runtime-snapshot/v1"
PROC_NET_FILES = ("tcp", "tcp6", "udp", "udp6", "unix")
NAMESPACE_NAMES = ("cgroup", "ipc", "mnt", "net", "pid", "pid_for_children",
                   "time", "time_for_children", "user", "uts")
MAX_PROC_BYTES = 2 * 1024 * 1024


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path, limit: int = MAX_PROC_BYTES) -> dict[str, Any]:
    """Read one procfs pseudo-file once, retaining truncation explicitly."""
    try:
        with path.open("rb", buffering=0) as handle:
            data = handle.read(limit + 1)
    except (OSError, PermissionError) as exc:
        return {"status": "unreadable", "error": f"{type(exc).__name__}: {exc}"}
    truncated = len(data) > limit
    retained = data[:limit]
    return {
        "status": "ok",
        "bytes_retained": len(retained),
        "truncated": truncated,
        "sha256_retained": _sha256(retained),
        "base64": base64.b64encode(retained).decode("ascii"),
        "utf8": retained.decode("utf-8", errors="replace"),
    }


def _readlink(path: Path) -> dict[str, Any]:
    try:
        return {"status": "ok", "target": os.readlink(path)}
    except (OSError, PermissionError) as exc:
        return {"status": "unreadable", "error": f"{type(exc).__name__}: {exc}"}


def _namespaces(proc_dir: Path) -> dict[str, Any]:
    return {name: _readlink(proc_dir / "ns" / name) for name in NAMESPACE_NAMES}


def _fd_inventory(proc_dir: Path) -> list[dict[str, Any]]:
    fd_dir = proc_dir / "fd"
    try:
        names = sorted(fd_dir.iterdir(), key=lambda p: int(p.name))
    except (OSError, PermissionError, ValueError) as exc:
        return [{"status": "unreadable", "error": f"{type(exc).__name__}: {exc}"}]
    entries: list[dict[str, Any]] = []
    for path in names:
        link = _readlink(path)
        entries.append({"fd": path.name, **link})
    return entries


def _process(pid: int, proc_root: Path) -> dict[str, Any]:
    proc_dir = proc_root / str(pid)
    return {
        "pid": pid,
        "stat": _read_bytes(proc_dir / "stat"),
        "status": _read_bytes(proc_dir / "status"),
        "cmdline": _read_bytes(proc_dir / "cmdline"),
        "comm": _read_bytes(proc_dir / "comm"),
        "exe": _readlink(proc_dir / "exe"),
        "cwd": _readlink(proc_dir / "cwd"),
        "root": _readlink(proc_dir / "root"),
        "namespaces": _namespaces(proc_dir),
        "fds": _fd_inventory(proc_dir),
    }


def _pids(proc_root: Path) -> list[int]:
    try:
        return sorted(int(path.name) for path in proc_root.iterdir()
                      if path.name.isdigit())
    except OSError:
        return []


def _proc_net(output: Path, phase: str,
              proc_root: Path) -> list[dict[str, Any]]:
    phase_dir = output / f"proc-net-{phase}"
    phase_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for name in PROC_NET_FILES:
        source = proc_root / "net" / name
        record = _read_bytes(source, limit=16 * 1024 * 1024)
        target = phase_dir / f"{name}.txt"
        if record.get("status") == "ok":
            raw = base64.b64decode(record.pop("base64"))
            record.pop("utf8", None)
            target.write_bytes(raw)
            record["path"] = target.relative_to(output).as_posix()
            record["sha256"] = _sha256(raw)
        records.append({"name": name, **record})
    return records


def _installed_distributions() -> list[dict[str, str]]:
    packages: list[dict[str, str]] = []
    try:
        for dist in importlib.metadata.distributions():
            name = dist.metadata.get("Name") or ""
            packages.append({"name": name, "version": dist.version})
    except Exception as exc:  # package metadata is optional evidence
        return [{"name": "<inventory-error>",
                 "version": f"{type(exc).__name__}: {exc}"}]
    return sorted(packages, key=lambda item: (item["name"].lower(), item["version"]))


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(data),
        "sha256": _sha256(data),
    }


def capture(output: Path, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    """Create one fresh, self-describing runtime snapshot at ``output``."""
    if output.is_symlink() or output.exists():
        if output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.mkdir(parents=True, mode=0o755)
    process_dir = output / "processes"
    process_dir.mkdir()

    start_wall = datetime.now(timezone.utc).isoformat()
    start_mono = time.monotonic_ns()
    net_before = _proc_net(output, "before", proc_root)

    process_index: list[dict[str, Any]] = []
    for pid in _pids(proc_root):
        record = _process(pid, proc_root)
        path = process_dir / f"{pid}.json"
        path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")),
                        encoding="utf-8")
        process_index.append({"pid": pid, **_file_record(output, path)})

    net_after = _proc_net(output, "after", proc_root)
    collector = Path(__file__).resolve()
    collector_copy = output / "collector.py"
    shutil.copy2(collector, collector_copy)
    end_mono = time.monotonic_ns()
    end_wall = datetime.now(timezone.utc).isoformat()

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "capture_boundary": "after solver return, before official verifier",
        "purpose": "task-agnostic runtime evidence; no verdict is computed",
        "privacy": {"process_environments_captured": False},
        "time": {
            "start_utc": start_wall,
            "end_utc": end_wall,
            "start_monotonic_ns": start_mono,
            "end_monotonic_ns": end_mono,
        },
        "collector": _file_record(output, collector_copy),
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "uname": list(platform.uname()),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "boot_id": _read_bytes(proc_root / "sys/kernel/random/boot_id"),
            "uptime": _read_bytes(proc_root / "uptime"),
            "self_namespaces": _namespaces(proc_root / "self"),
            "pid1_namespaces": _namespaces(proc_root / "1"),
            "installed_python_distributions": _installed_distributions(),
        },
        "network": {"before": net_before, "after": net_after},
        "process_count": len(process_index),
        "processes": process_index,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    capture(args.output)


if __name__ == "__main__":
    main()
