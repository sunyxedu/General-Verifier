"""Merge physical shared-setup ledgers into one auditable accounting view.

The caller predeclares semantic coverage. This module only accounts for each
physical component once: it never discovers successful claims, retries work,
or copies common usage into either replay arm.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

from experiments.terminal_bench import analyze_model_spans, shared_setup
from verifier.contract_replay import MAX_SNAPSHOT_CLAIMS


MAX_COMPONENTS = MAX_SNAPSHOT_CLAIMS + 1
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_COMPONENT_FIELDS = frozenset({"component_id", "usage_path", "spans_path"})
_RUNTIME = {
    "model": "gpt-5.6-luna",
    "effort": "medium",
    "service_tier": "standard",
}


def _unknown_metrics() -> dict[str, Any]:
    return {
        "complete": False,
        "cost_complete": False,
        "wall_complete": False,
        **{field: None for field in shared_setup.COST_FIELDS},
        "wall_seconds": None,
    }


def _has_symlink(path: Path) -> bool:
    return any(item.is_symlink() for item in (path, *path.parents))


def _private_read(path: Path, cap: int) -> bytes:
    if not path.is_absolute() or ".." in path.parts or _has_symlink(path):
        raise ValueError("accounting source path is not direct and absolute")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_size > cap):
            raise ValueError("accounting source is not private and bounded")
        chunks = []
        remaining = cap + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > cap:
            raise ValueError("accounting source exceeds aggregate cap")
        return raw
    finally:
        os.close(descriptor)


def _open_atomic(destination: Path, name: str):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=destination)
    return os.fdopen(descriptor, "wb"), Path(temporary)


def _publish(handle, temporary: Path, target: Path) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    os.fchmod(handle.fileno(), 0o400)
    handle.close()
    os.replace(temporary, target)


def _recheck_sources(stability, source_records, findings) -> None:
    for index, label, path, size, digest in stability:
        metadata = source_records[index][label]
        try:
            raw = _private_read(path, size)
        except (OSError, ValueError):
            metadata["stable"] = False
            findings.append(f"changed_{label}:{index}")
            continue
        stable = len(raw) == size and hashlib.sha256(raw).hexdigest() == digest
        metadata["stable"] = stable
        if not stable:
            findings.append(f"changed_{label}:{index}")


def _recheck_aggregates(paths, expected_hashes, findings) -> None:
    for label, path in paths.items():
        try:
            raw = _private_read(path, analyze_model_spans.MAX_LEDGER_BYTES)
        except (OSError, ValueError):
            findings.append(f"invalid_aggregate_{label}")
            continue
        if hashlib.sha256(raw).hexdigest() != expected_hashes[label]:
            findings.append(f"changed_aggregate_{label}")


def merge_setup_accounting(
    destination: str | Path,
    *,
    setup_id: str,
    components: list[dict],
    wall_seconds: float,
) -> dict[str, Any]:
    """Concatenate exact source bytes and audit the global attempt namespace.

    Missing or unstable sources do not erase intact rows from other physical
    components. They do, however, make the returned charge vector unknown;
    the partial analyzer report is diagnostic evidence, never a lower bill.
    """
    destination = Path(destination).absolute()
    if os.path.lexists(destination):
        raise ValueError("accounting destination must be new")
    if _has_symlink(destination.parent):
        raise ValueError("accounting destination may not traverse symlinks")
    destination.mkdir(parents=True, exist_ok=False)

    findings: list[str] = []
    if not isinstance(setup_id, str) or _IDENTIFIER.fullmatch(setup_id) is None:
        findings.append("invalid_setup_id")
    if type(components) is not list or len(components) > MAX_COMPONENTS:
        findings.append("invalid_components")
        components = []

    usage_path = destination / "setup-usage.jsonl"
    spans_path = destination / "setup-usage.jsonl.spans.jsonl"
    handles = {}
    temporaries = {}
    for label, target in (("usage", usage_path), ("spans", spans_path)):
        handles[label], temporaries[label] = _open_atomic(destination, target.name)

    remaining = {
        "usage": analyze_model_spans.MAX_LEDGER_BYTES,
        "spans": analyze_model_spans.MAX_LEDGER_BYTES,
    }
    aggregate_hashes = {label: hashlib.sha256() for label in handles}
    source_records: list[dict[str, Any]] = []
    seen_component_ids: set[str] = set()
    seen_sources: set[str] = set()
    stability: list[tuple[int, str, Path, int, str]] = []
    try:
        for index, component in enumerate(components):
            record: dict[str, Any] = {
                "component_index": index,
                "component_id": None,
                "usage": None,
                "spans": None,
            }
            source_records.append(record)
            if type(component) is not dict or set(component) != _COMPONENT_FIELDS:
                findings.append(f"invalid_component:{index}")
                continue
            component_id = component.get("component_id")
            record["component_id"] = component_id
            if (not isinstance(component_id, str)
                    or _IDENTIFIER.fullmatch(component_id) is None):
                findings.append(f"invalid_component_id:{index}")
            elif component_id in seen_component_ids:
                findings.append(f"duplicate_component_id:{index}")
            else:
                seen_component_ids.add(component_id)

            for label, field in (("usage", "usage_path"),
                                 ("spans", "spans_path")):
                value = component.get(field)
                metadata = {
                    "path": value if isinstance(value, str) else None,
                    "sha256": None,
                    "bytes": None,
                    "included": False,
                    "stable": False,
                }
                record[label] = metadata
                try:
                    path = Path(value)
                    canonical = str(path.resolve(strict=True))
                except (OSError, RuntimeError, TypeError, ValueError):
                    findings.append(f"unavailable_{label}:{index}")
                    continue
                if canonical in seen_sources:
                    findings.append(f"duplicate_source_path:{index}:{label}")
                    continue
                seen_sources.add(canonical)
                try:
                    raw = _private_read(path, remaining[label])
                except (OSError, ValueError):
                    findings.append(f"unavailable_{label}:{index}")
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                metadata.update({
                    "path": canonical,
                    "sha256": digest,
                    "bytes": len(raw),
                })
                stability.append((index, label, path, len(raw), digest))
                if not raw or not raw.endswith(b"\n"):
                    findings.append(f"unterminated_{label}:{index}")
                    continue
                handles[label].write(raw)
                aggregate_hashes[label].update(raw)
                remaining[label] -= len(raw)
                metadata["included"] = True

        for label, target in (("usage", usage_path), ("spans", spans_path)):
            _publish(handles.pop(label), temporaries.pop(label), target)
        try:
            directory = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            findings.append("cannot_fsync_destination")
        else:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        for handle in handles.values():
            handle.close()
        for temporary in temporaries.values():
            try:
                temporary.unlink()
            except OSError:
                pass

    aggregate_paths = {"usage": usage_path, "spans": spans_path}
    expected_aggregate_hashes = {
        label: digest.hexdigest() for label, digest in aggregate_hashes.items()}
    _recheck_sources(stability, source_records, findings)
    _recheck_aggregates(
        aggregate_paths, expected_aggregate_hashes, findings)

    model_work = analyze_model_spans.analyze(usage_path, spans_path)
    usage_rows, usage_errors = analyze_model_spans._read_jsonl(usage_path)
    findings.extend(f"usage:{value}" for value in usage_errors)
    for row_no, row in enumerate(usage_rows, 1):
        if row.get("setup_id") != setup_id:
            findings.append(f"usage_setup_mismatch:{row_no}")
        if row.get("backend") != "codex-cli":
            findings.append(f"usage_backend_mismatch:{row_no}")
    audit = shared_setup.audit_model_work(
        usage_path,
        spans_path,
        scope="setup",
        wall_seconds=wall_seconds,
        expected_runtime=_RUNTIME,
    )
    findings.extend(f"model_work:{value}" for value in audit["findings"])
    # Analyzer/audit reads are outside our control. Recheck every byte after
    # both consumers so a concurrent rewrite can never inherit their result.
    _recheck_sources(stability, source_records, findings)
    _recheck_aggregates(
        aggregate_paths, expected_aggregate_hashes, findings)
    complete = not findings and audit["complete"] is True
    return {
        "complete": complete,
        "findings": sorted(set(findings)),
        "usage_path": str(usage_path),
        "spans_path": str(spans_path),
        "source_records": source_records,
        "model_work": model_work,
        "metrics": audit["metrics"] if complete else _unknown_metrics(),
    }
