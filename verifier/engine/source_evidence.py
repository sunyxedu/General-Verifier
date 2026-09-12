"""Lossless, content-addressed source-event views for verifier seats.

The canonical ``events.jsonl`` remains the internal append-only event bus.
This module projects only the fixed candidate/source event classes into
small content-addressed, read-only pages. Large UTF-8 string leaves are moved
to exact blobs;
their JSON pointers, byte counts, hashes, and short previews remain in the
page.  Nothing is summarized and every original payload can be reconstructed.

The view is evidence transport only.  It caches no model output or verdict,
adds no verifier stage, and does not interpret event contents.
"""
from __future__ import annotations

import hashlib
import json
import math
import errno
import os
import secrets
import stat
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ..core.events import (
    MAX_SOURCE_LEDGER_EVENTS,
    SOURCE_EVENT_TYPES,
    EventLog,
    SourceProjectionUnavailable,
)
from ..core.model import canonical, content_hash


SOURCE_EVIDENCE_PROTOCOL = "gv-source-evidence-view-v1"
SOURCE_EVIDENCE_PAGE_PROTOCOL = "gv-source-evidence-page-v1"
SOURCE_EVIDENCE_INDEX_PROTOCOL = "gv-source-evidence-index-v1"
INLINE_STRING_BYTES = 1024
PREVIEW_CHARACTERS = 192
TARGET_PAGE_BYTES = 16 * 1024
MAX_PAGE_BYTES = 1024 * 1024
MAX_BLOB_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BLOB_BYTES = 32 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 200_000
MAX_JSON_POINTER_BYTES = 4096
MAX_EXTERNALIZED_FIELDS = 20_000
MAX_DESCRIPTOR_BYTES = 8 * 1024 * 1024
MAX_INDEX_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_VIEW_FILES = 12_000
MAX_TOTAL_VIEW_BYTES = 64 * 1024 * 1024
VIEW_DIRECTORY = "source_evidence"


class SourceEvidenceInvalid(RuntimeError):
    """A promised lossless source view cannot be built or revalidated."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    try:
        return (canonical(value) + "\n").encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise SourceEvidenceInvalid(
            "source evidence cannot be encoded as strict UTF-8 JSON") from error


def _pointer_segment(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _bounded_preview(value: str) -> tuple[str, str]:
    if len(value) <= PREVIEW_CHARACTERS * 2:
        return value, ""
    return value[:PREVIEW_CHARACTERS], value[-PREVIEW_CHARACTERS:]


def _externalize_payload(
    value: Any,
    *,
    pointer: str,
    blobs: Dict[str, bytes],
    detached: list[dict[str, Any]],
    budget: list[int],
    descriptor_budget: list[int],
    depth: int = 0,
) -> Any:
    if depth > MAX_JSON_DEPTH:
        raise SourceEvidenceInvalid("source payload exceeds the depth limit")
    budget[0] += 1
    if budget[0] > MAX_JSON_NODES:
        raise SourceEvidenceInvalid("source payload exceeds the node limit")
    if isinstance(value, str):
        try:
            raw = value.encode("utf-8", errors="strict")
        except UnicodeError as error:
            raise SourceEvidenceInvalid(
                "source payload contains invalid Unicode") from error
        if len(raw) <= INLINE_STRING_BYTES:
            return value
        if len(raw) > MAX_BLOB_BYTES:
            raise SourceEvidenceInvalid(
                "one source string exceeds the exact blob limit")
        digest = _sha256(raw)
        existing = blobs.get(digest)
        if existing is not None and existing != raw:
            raise SourceEvidenceInvalid("source blob digest collision")
        if existing is None:
            blobs[digest] = raw
        head, tail = _bounded_preview(value)
        descriptor = {
            "json_pointer": pointer,
            "encoding": "utf-8",
            "characters": len(value),
            "utf8_bytes": len(raw),
            "sha256": digest,
            "blob": f"blobs/{digest}.blob",
            "head": head,
            "tail": tail,
        }
        pointer_bytes = len(pointer.encode("utf-8", errors="strict"))
        if pointer_bytes > MAX_JSON_POINTER_BYTES:
            raise SourceEvidenceInvalid(
                "source blob JSON pointer exceeds the byte limit")
        descriptor_budget[0] += 1
        descriptor_budget[1] += len(_json_bytes(descriptor))
        if descriptor_budget[0] > MAX_EXTERNALIZED_FIELDS:
            raise SourceEvidenceInvalid(
                "source view exceeds the detached-field limit")
        if descriptor_budget[1] > MAX_DESCRIPTOR_BYTES:
            raise SourceEvidenceInvalid(
                "source view exceeds the descriptor byte limit")
        detached.append(descriptor)
        # The exact pointer list disambiguates this transport marker from an
        # identical object that happened to occur in the original payload.
        return {"$gv_source_utf8_blob": descriptor}
    if isinstance(value, list):
        return [
            _externalize_payload(
                item,
                pointer=f"{pointer}/{index}",
                blobs=blobs,
                detached=detached,
                budget=budget,
                descriptor_budget=descriptor_budget,
                depth=depth + 1,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        result = {}
        for key in sorted(value):
            item = value[key]
            if not isinstance(key, str):
                raise SourceEvidenceInvalid("source object key is not text")
            try:
                key.encode("utf-8", errors="strict")
            except UnicodeError as error:
                raise SourceEvidenceInvalid(
                    "source object key contains invalid Unicode") from error
            result[key] = _externalize_payload(
                item,
                pointer=f"{pointer}/{_pointer_segment(key)}",
                blobs=blobs,
                detached=detached,
                budget=budget,
                descriptor_budget=descriptor_budget,
                depth=depth + 1,
            )
        return result
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise SourceEvidenceInvalid(
        f"unsupported source payload value {type(value).__name__}")


def _project_events(
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Dict[str, bytes]]:
    projected = []
    blobs: Dict[str, bytes] = {}
    budget = [0]
    descriptor_budget = [0, 0]
    previous_seq = 0
    for ordinal, event in enumerate(events, start=1):
        if (not isinstance(event, Mapping)
                or set(event) != {
                    "seq", "ts", "type", "payload", "event_digest",
                    "source_record_digest",
                }
                or event.get("type") not in SOURCE_EVENT_TYPES):
            raise SourceEvidenceInvalid("invalid source projection event")
        seq = event["seq"]
        timestamp = event["ts"]
        if (isinstance(seq, bool) or not isinstance(seq, int)
                or seq <= previous_seq
                or isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp)
                or not isinstance(event["payload"], dict)):
            raise SourceEvidenceInvalid("invalid source projection identity")
        expected_event_digest = EventLog.event_digest(dict(event))
        # The record digest intentionally binds the writer's raw JSON bytes,
        # so only EventLog's strict raw reader can establish its value here.
        if (event["event_digest"] != expected_event_digest
                or not isinstance(event["source_record_digest"], str)
                or len(event["source_record_digest"]) != 64
                or any(character not in "0123456789abcdef"
                       for character in event["source_record_digest"])):
            raise SourceEvidenceInvalid("invalid source projection digest")
        previous_seq = seq
        detached: list[dict[str, Any]] = []
        payload = _externalize_payload(
            event["payload"], pointer="/payload", blobs=blobs,
            detached=detached, budget=budget,
            descriptor_budget=descriptor_budget)
        projected.append({
            "source_ordinal": ordinal,
            "original_seq": event["seq"],
            "original_ts": event["ts"],
            "type": event["type"],
            "event_digest": event["event_digest"],
            "source_record_digest": event["source_record_digest"],
            "payload": payload,
            "externalized_utf8_fields": detached,
        })
    total_blob_bytes = sum(len(raw) for raw in blobs.values())
    if total_blob_bytes > MAX_TOTAL_BLOB_BYTES:
        raise SourceEvidenceInvalid("source blobs exceed the total byte limit")
    return projected, blobs


def _page_events(
    projected: Sequence[Mapping[str, Any]],
    blobs: Dict[str, bytes],
) -> tuple[list[list[dict[str, Any]]], Dict[str, bytes]]:
    pages: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for item in projected:
        event = dict(item)
        encoded = _json_bytes(event)
        if len(encoded) > MAX_PAGE_BYTES:
            raw_event = _json_bytes({
                "source_ordinal": event["source_ordinal"],
                "original_seq": event["original_seq"],
                "original_ts": event["original_ts"],
                "type": event["type"],
                "event_digest": event["event_digest"],
                "source_record_digest": event["source_record_digest"],
                "payload": event["payload"],
                "externalized_utf8_fields": event[
                    "externalized_utf8_fields"],
            })
            if len(raw_event) > MAX_BLOB_BYTES:
                raise SourceEvidenceInvalid(
                    "one projected source event exceeds the exact blob limit")
            digest = _sha256(raw_event)
            existing = blobs.get(digest)
            if existing is not None and existing != raw_event:
                raise SourceEvidenceInvalid("source blob digest collision")
            blobs[digest] = raw_event
            event = {
                "source_ordinal": event["source_ordinal"],
                "original_seq": event["original_seq"],
                "original_ts": event["original_ts"],
                "type": event["type"],
                "event_digest": event["event_digest"],
                "source_record_digest": event["source_record_digest"],
                "exact_event_blob": {
                    "encoding": "canonical-json-utf8",
                    "bytes": len(raw_event),
                    "sha256": digest,
                    "blob": f"blobs/{digest}.blob",
                },
            }
            encoded = _json_bytes(event)
        if current and current_bytes + len(encoded) > TARGET_PAGE_BYTES:
            pages.append(current)
            current = []
            current_bytes = 0
        current.append(event)
        current_bytes += len(encoded)
    if current or not pages:
        pages.append(current)
    if sum(len(raw) for raw in blobs.values()) > MAX_TOTAL_BLOB_BYTES:
        raise SourceEvidenceInvalid("source blobs exceed the total byte limit")
    return pages, blobs


def _render_files(
    events: Sequence[Mapping[str, Any]],
    view_id: str,
) -> Dict[str, bytes]:
    projected, blobs = _project_events(events)
    pages, blobs = _page_events(projected, blobs)
    files: Dict[str, bytes] = {}
    page_inventory = []
    event_index = []
    for page_index, page_events in enumerate(pages):
        page_value = {
            "schema": SOURCE_EVIDENCE_PAGE_PROTOCOL,
            "page": page_index,
            "event_count": len(page_events),
            "events": page_events,
        }
        raw = _json_bytes(page_value)
        if len(raw) > MAX_PAGE_BYTES:
            raise SourceEvidenceInvalid("source page exceeds its byte limit")
        digest = _sha256(raw)
        name = f"pages/page-{page_index:05d}-{digest}.json"
        files[name] = raw
        page_inventory.append({
            "page": page_index,
            "path": name,
            "event_count": len(page_events),
            "bytes": len(raw),
            "sha256": digest,
            "first_source_ordinal": (
                page_events[0]["source_ordinal"] if page_events else None),
            "last_source_ordinal": (
                page_events[-1]["source_ordinal"] if page_events else None),
        })
        for item in page_events:
            detached = item.get("externalized_utf8_fields", [])
            event_index.append({
                "source_ordinal": item["source_ordinal"],
                "original_seq": item["original_seq"],
                "original_ts": item["original_ts"],
                "type": item["type"],
                "event_digest": item["event_digest"],
                "source_record_digest": item["source_record_digest"],
                "page": page_index,
                "page_path": name,
                "externalized_fields": [
                    {key: field[key] for key in (
                        "json_pointer", "utf8_bytes", "sha256", "blob")}
                    for field in detached
                ],
                **({"exact_event_blob": item["exact_event_blob"]}
                   if "exact_event_blob" in item else {}),
            })

    index_lines = [canonical({
        "schema": SOURCE_EVIDENCE_INDEX_PROTOCOL,
        "view_id": view_id,
        "event_count": len(event_index),
    })]
    index_lines.extend(canonical(item) for item in event_index)
    index_raw = ("\n".join(index_lines) + "\n").encode(
        "utf-8", errors="strict")
    if len(index_raw) > MAX_INDEX_BYTES:
        raise SourceEvidenceInvalid("source evidence index exceeds its byte limit")
    files["index.jsonl"] = index_raw

    blob_inventory = []
    for digest, raw in sorted(blobs.items()):
        name = f"blobs/{digest}.blob"
        files[name] = raw
        blob_inventory.append({
            "path": name,
            "bytes": len(raw),
            "sha256": digest,
        })

    source_record_digests = [
        event["source_record_digest"] for event in events]
    content_file_count = len(files)
    content_bytes = sum(len(raw) for raw in files.values())
    manifest = {
        "schema": SOURCE_EVIDENCE_PROTOCOL,
        "view_id": view_id,
        "semantics": (
            "Lossless inert projection of candidate/source events only. "
            "Pages preserve original seq, timestamp, type, payload and both "
            "record and semantic event digests. "
            "A detached UTF-8 field is complete only at its exact blob path. "
            "The view makes no semantic claim, verdict, or future-closure claim."
        ),
        "source_event_types": list(SOURCE_EVENT_TYPES),
        "event_count": len(events),
        "first_original_seq": events[0]["seq"] if events else None,
        "last_original_seq": events[-1]["seq"] if events else None,
        "source_events_digest": content_hash(source_record_digests),
        "last_source_original_seq": events[-1]["seq"] if events else None,
        "projection_complete_at_observation": True,
        "proves_future_closure": False,
        "index": {
            "path": "index.jsonl",
            "bytes": len(index_raw),
            "sha256": _sha256(index_raw),
        },
        "pages": page_inventory,
        "blobs": blob_inventory,
        "content_file_count": content_file_count,
        "content_bytes": content_bytes,
        "inspection": (
            "Resolve paths relative to this manifest. Search index.jsonl by "
            "event type or original seq, open the named page, and open every "
            "referenced blob whose exact content matters."
        ),
    }
    manifest_raw = _json_bytes(manifest)
    if len(manifest_raw) > MAX_MANIFEST_BYTES:
        raise SourceEvidenceInvalid(
            "source evidence manifest exceeds its byte limit")
    files["manifest.json"] = manifest_raw
    if len(files) > MAX_VIEW_FILES:
        raise SourceEvidenceInvalid("source evidence exceeds its file limit")
    if sum(len(raw) for raw in files.values()) > MAX_TOTAL_VIEW_BYTES:
        raise SourceEvidenceInvalid(
            "source evidence exceeds its total byte limit")
    return files


def _open_directory(path: str, *, mode: int | None) -> int:
    if os.path.realpath(path) != path:
        raise SourceEvidenceInvalid(
            "source evidence path must be canonical")
    flags = (os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
             | getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        named = os.lstat(path)
        if (not stat.S_ISDIR(opened.st_mode)
                or stat.S_ISLNK(named.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (named.st_dev, named.st_ino)
                or opened.st_uid != os.geteuid()
                or (mode is not None
                    and stat.S_IMODE(opened.st_mode) != mode)):
            raise SourceEvidenceInvalid(
                "source evidence directory identity is unsafe")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_at(parent_fd: int, name: str, *, mode: int) -> int:
    if not name or name in {".", ".."} or os.sep in name:
        raise SourceEvidenceInvalid("invalid source evidence directory name")
    flags = (os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
             | getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (named.st_dev, named.st_ino)
                or opened.st_uid != os.geteuid()
                or stat.S_IMODE(opened.st_mode) != mode):
            raise SourceEvidenceInvalid(
                "source evidence directory identity is unsafe")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_file_at(parent_fd: int, name: str, raw: bytes) -> None:
    if not name or name in {".", ".."} or os.sep in name:
        raise SourceEvidenceInvalid("invalid source evidence member name")
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
             | getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(name, flags, 0o400, dir_fd=parent_fd)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("source evidence write made no progress")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_tree_at(parent_fd: int, name: str) -> None:
    """Best-effort cleanup for a private unpublished two-level tree."""
    try:
        root_fd = _open_directory_at(parent_fd, name, mode=0o700)
    except (OSError, SourceEvidenceInvalid):
        try:
            os.rmdir(name, dir_fd=parent_fd)
        except OSError:
            pass
        return
    try:
        for child in os.listdir(root_fd):
            try:
                info = os.stat(child, dir_fd=root_fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    child_fd = os.open(
                        child, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=root_fd)
                    try:
                        os.fchmod(child_fd, 0o700)
                        for member in os.listdir(child_fd):
                            os.unlink(member, dir_fd=child_fd)
                    finally:
                        os.close(child_fd)
                    os.rmdir(child, dir_fd=root_fd)
                else:
                    os.unlink(child, dir_fd=root_fd)
            except OSError:
                pass
    finally:
        os.close(root_fd)
    try:
        os.rmdir(name, dir_fd=parent_fd)
    except OSError:
        pass


def _publish(view_root: str, files: Mapping[str, bytes]) -> None:
    evidence_root = os.path.dirname(view_root)
    state_root = os.path.dirname(evidence_root)
    evidence_name = os.path.basename(evidence_root)
    view_name = os.path.basename(view_root)
    state_fd = evidence_fd = temporary_fd = -1
    child_fds: Dict[str, int] = {}
    temporary_name = ""
    try:
        state_fd = _open_directory(state_root, mode=None)
        try:
            os.mkdir(evidence_name, 0o700, dir_fd=state_fd)
        except FileExistsError:
            pass
        evidence_fd = _open_directory_at(
            state_fd, evidence_name, mode=0o700)
        for _attempt in range(32):
            temporary_name = ".building-" + secrets.token_hex(12)
            try:
                os.mkdir(temporary_name, 0o700, dir_fd=evidence_fd)
                break
            except FileExistsError:
                continue
        else:
            raise SourceEvidenceInvalid(
                "cannot allocate a source evidence publication")
        temporary_fd = _open_directory_at(
            evidence_fd, temporary_name, mode=0o700)
        for relative, raw in sorted(files.items()):
            parts = relative.split("/")
            if (not relative or relative.startswith("/")
                    or ".." in parts or len(parts) > 2):
                raise SourceEvidenceInvalid("invalid source evidence file name")
            parent_fd = temporary_fd
            if len(parts) == 2:
                directory = parts[0]
                if directory not in child_fds:
                    try:
                        os.mkdir(directory, 0o700, dir_fd=temporary_fd)
                    except FileExistsError:
                        pass
                    child_fds[directory] = _open_directory_at(
                        temporary_fd, directory, mode=0o700)
                parent_fd = child_fds[directory]
            _write_file_at(parent_fd, parts[-1], raw)
        for descriptor in child_fds.values():
            os.fchmod(descriptor, 0o500)
            os.fsync(descriptor)
        # The directory must be in its final durable state before it becomes
        # reachable by its content-addressed name.  Otherwise a crash between
        # rename and chmod can leave a permanently invalid 0700 publication.
        os.fchmod(temporary_fd, 0o500)
        os.fsync(temporary_fd)
        try:
            os.rename(
                temporary_name, view_name,
                src_dir_fd=evidence_fd, dst_dir_fd=evidence_fd)
            temporary_name = ""
        except OSError as error:
            conflict = error.errno in {errno.EEXIST, errno.ENOTEMPTY}
            if error.errno == errno.EACCES:
                try:
                    os.stat(
                        view_name, dir_fd=evidence_fd,
                        follow_symlinks=False)
                    conflict = True
                except OSError:
                    pass
            if not conflict:
                raise
        os.fsync(evidence_fd)
    except (OSError, ValueError) as error:
        if isinstance(error, SourceEvidenceInvalid):
            raise
        raise SourceEvidenceInvalid(
            f"cannot publish source evidence view: {error}") from error
    finally:
        for descriptor in child_fds.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_name and temporary_fd >= 0:
            try:
                os.fchmod(temporary_fd, 0o700)
            except OSError:
                pass
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_name and evidence_fd >= 0:
            _remove_tree_at(evidence_fd, temporary_name)
        if evidence_fd >= 0:
            os.close(evidence_fd)
        if state_fd >= 0:
            os.close(state_fd)


def _read_exact_regular_at(parent_fd: int, name: str, limit: int) -> bytes:
    flags = (os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
             | getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        before = os.fstat(descriptor)
        named_before = os.stat(
            name, dir_fd=parent_fd, follow_symlinks=False)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o400
                or before.st_size > limit
                or _member_identity(before) != _member_identity(named_before)):
            raise SourceEvidenceInvalid("source evidence member is unsafe")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise SourceEvidenceInvalid("source evidence member is too large")
        after = os.fstat(descriptor)
        named_after = os.stat(
            name, dir_fd=parent_fd, follow_symlinks=False)
        if (_member_identity(before) != _member_identity(after)
                or _member_identity(after) != _member_identity(named_after)):
            raise SourceEvidenceInvalid(
                "source evidence member changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _member_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_uid, value.st_gid, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _bounded_names(descriptor: int, expected_count: int) -> set[str]:
    if expected_count > MAX_VIEW_FILES:
        raise SourceEvidenceInvalid("source evidence exceeds its file limit")
    names: set[str] = set()
    with os.scandir(descriptor) as entries:
        for entry in entries:
            names.add(entry.name)
            if len(names) > expected_count or len(names) > MAX_VIEW_FILES:
                raise SourceEvidenceInvalid(
                    "source evidence inventory exceeds its bound")
    return names


def _validate_publication(view_root: str, files: Mapping[str, bytes]) -> None:
    evidence_root = os.path.dirname(view_root)
    state_root = os.path.dirname(evidence_root)
    evidence_name = os.path.basename(evidence_root)
    view_name = os.path.basename(view_root)
    state_fd = evidence_fd = view_fd = -1
    child_fds: Dict[str, int] = {}
    try:
        state_fd = _open_directory(state_root, mode=None)
        evidence_fd = _open_directory_at(
            state_fd, evidence_name, mode=0o700)
        view_fd = _open_directory_at(evidence_fd, view_name, mode=0o500)
        view_identity = _directory_identity(os.fstat(view_fd))
        root_files = {path for path in files if "/" not in path}
        child_files: Dict[str, set[str]] = {}
        for relative in files:
            parts = relative.split("/")
            if len(parts) == 2:
                child_files.setdefault(parts[0], set()).add(parts[1])
            elif len(parts) != 1:
                raise SourceEvidenceInvalid(
                    "invalid source evidence inventory path")
        expected_root = root_files | set(child_files)
        if _bounded_names(view_fd, len(expected_root)) != expected_root:
            raise SourceEvidenceInvalid("source evidence inventory changed")
        child_identities = {}
        for directory, names in child_files.items():
            descriptor = _open_directory_at(
                view_fd, directory, mode=0o500)
            child_fds[directory] = descriptor
            child_identities[directory] = _directory_identity(
                os.fstat(descriptor))
            if _bounded_names(descriptor, len(names)) != names:
                raise SourceEvidenceInvalid(
                    "source evidence inventory changed")
        for relative, raw in files.items():
            parts = relative.split("/")
            parent_fd = (view_fd if len(parts) == 1
                         else child_fds[parts[0]])
            actual = _read_exact_regular_at(
                parent_fd, parts[-1], max(len(raw), 1))
            if actual != raw:
                raise SourceEvidenceInvalid(
                    "source evidence member digest mismatch")
        # Close the inspection interval with anchored inventories and the
        # published directory's still-current name identity.
        for directory, names in child_files.items():
            descriptor = child_fds[directory]
            named_child = os.stat(
                directory, dir_fd=view_fd, follow_symlinks=False)
            if (_bounded_names(descriptor, len(names)) != names
                    or _directory_identity(os.fstat(descriptor))
                    != child_identities[directory]
                    or (named_child.st_dev, named_child.st_ino)
                    != (os.fstat(descriptor).st_dev,
                        os.fstat(descriptor).st_ino)):
                raise SourceEvidenceInvalid(
                    "source evidence child identity changed")
        if (_bounded_names(view_fd, len(expected_root)) != expected_root
                or _directory_identity(os.fstat(view_fd)) != view_identity):
            raise SourceEvidenceInvalid("source evidence inventory changed")
        named = os.stat(
            view_name, dir_fd=evidence_fd, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (
                os.fstat(view_fd).st_dev, os.fstat(view_fd).st_ino):
            raise SourceEvidenceInvalid(
                "source evidence publication identity changed")
        named_evidence = os.stat(
            evidence_name, dir_fd=state_fd, follow_symlinks=False)
        if (named_evidence.st_dev, named_evidence.st_ino) != (
                os.fstat(evidence_fd).st_dev,
                os.fstat(evidence_fd).st_ino):
            raise SourceEvidenceInvalid(
                "source evidence root identity changed")
    finally:
        for descriptor in child_fds.values():
            os.close(descriptor)
        if view_fd >= 0:
            os.close(view_fd)
        if evidence_fd >= 0:
            os.close(evidence_fd)
        if state_fd >= 0:
            os.close(state_fd)


def _view_id(events: Sequence[Mapping[str, Any]]) -> str:
    if len(events) > MAX_SOURCE_LEDGER_EVENTS:
        raise SourceEvidenceInvalid("source view exceeds the event limit")
    return content_hash({
        "schemas": {
            "manifest": SOURCE_EVIDENCE_PROTOCOL,
            "page": SOURCE_EVIDENCE_PAGE_PROTOCOL,
            "index": SOURCE_EVIDENCE_INDEX_PROTOCOL,
            "render_profile": "gv-source-evidence-render-v1",
        },
        "source_event_types": list(SOURCE_EVENT_TYPES),
        "source_record_digests": [
            event["source_record_digest"] for event in events],
        "paging": {
            "inline_string_bytes": INLINE_STRING_BYTES,
            "preview_characters": PREVIEW_CHARACTERS,
            "target_page_bytes": TARGET_PAGE_BYTES,
            "max_page_bytes": MAX_PAGE_BYTES,
            "max_blob_bytes": MAX_BLOB_BYTES,
            "max_total_blob_bytes": MAX_TOTAL_BLOB_BYTES,
            "max_json_depth": MAX_JSON_DEPTH,
            "max_json_nodes": MAX_JSON_NODES,
            "max_json_pointer_bytes": MAX_JSON_POINTER_BYTES,
            "max_externalized_fields": MAX_EXTERNALIZED_FIELDS,
            "max_descriptor_bytes": MAX_DESCRIPTOR_BYTES,
            "max_index_bytes": MAX_INDEX_BYTES,
            "max_manifest_bytes": MAX_MANIFEST_BYTES,
            "max_view_files": MAX_VIEW_FILES,
            "max_total_view_bytes": MAX_TOTAL_VIEW_BYTES,
        },
    })


@dataclass(frozen=True)
class SourceEvidenceView:
    """One content-checked view plus the exact projection it represents."""

    view_id: str
    manifest_path: str
    _root: str
    _files: Tuple[Tuple[str, bytes], ...]
    _log: EventLog

    def revalidate(self) -> bool:
        try:
            events = self._log.source_projection()
            if _view_id(events) != self.view_id:
                return False
            _validate_publication(self._root, dict(self._files))
            return True
        except (OSError, ValueError, TypeError, UnicodeError,
                RecursionError, SourceProjectionUnavailable,
                SourceEvidenceInvalid):
            return False


def bind_source_evidence_variables(
    log: EventLog,
    variables: Mapping[str, Any],
    *,
    enabled: bool,
) -> Dict[str, Any]:
    """Bind the optional view without changing a role's evidence semantics.

    Failure to construct the transport returns the original full ledger.
    The private fields are harness capabilities and are never rendered by a
    role template.  Callers still run the same seat with the same schema.
    """
    bound = dict(variables)
    if not enabled:
        return bound
    if type(log) is not EventLog:
        return bound
    mixed_ledger = os.path.abspath(log.path)
    locator = bound.get("ledger_path")
    frozen = log.is_record_path(locator)
    if locator != mixed_ledger and not frozen:
        # In particular, never replace the authority-only withheld boundary.
        return bound
    try:
        # Transport a frozen prefix, never the later live projection. The
        # original locator must also survive runtime opt-out, stale-view and
        # refresh fallback paths. Mixed/live callers keep their old semantics.
        source_log = EventLog(locator) if frozen else log
        view = build_source_evidence_view(source_log)
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError,
            SourceEvidenceInvalid):
        return bound
    bound["_mixed_ledger_path"] = os.path.abspath(source_log.path)
    bound["_source_evidence_view"] = view
    bound["ledger_path"] = view.manifest_path
    return bound


def refresh_source_evidence_variables(
    variables: Mapping[str, Any],
) -> Dict[str, Any]:
    """Refresh a stale bound view before one existing role attempt.

    A verifier append leaves the content-addressed source view current.  If
    a new source event arrived, rebuild from that complete projection.  Any
    optional-transport failure falls back to the original ledger rather than
    withholding evidence or inventing another retry path.
    """
    current = dict(variables)
    view = current.get("_source_evidence_view")
    if type(view) is not SourceEvidenceView:
        return current
    if view.revalidate():
        return current
    mixed_ledger = current.get("_mixed_ledger_path")
    current.pop("_source_evidence_view", None)
    if not isinstance(mixed_ledger, str):
        return current
    current["ledger_path"] = mixed_ledger
    return bind_source_evidence_variables(
        view._log, current, enabled=True)


def build_source_evidence_view(log: EventLog) -> SourceEvidenceView:
    """Build or rebind one complete lossless source view.

    The caller may fall back to the original ledger when this optional
    transport is unavailable.  A returned object, however, is never partial.
    """
    if type(log) is not EventLog:
        raise SourceEvidenceInvalid("source evidence requires an exact EventLog")
    try:
        events = log.source_projection()
    except SourceProjectionUnavailable as error:
        raise SourceEvidenceInvalid(str(error)) from error
    view_id = _view_id(events)
    state_root = os.path.realpath(os.path.dirname(os.path.abspath(log.path)))
    evidence_root = os.path.join(state_root, VIEW_DIRECTORY)
    view_root = os.path.join(evidence_root, view_id)
    files = _render_files(events, view_id)
    if not os.path.lexists(view_root):
        _publish(view_root, files)
    _validate_publication(view_root, files)
    view = SourceEvidenceView(
        view_id=view_id,
        manifest_path=os.path.join(view_root, "manifest.json"),
        _root=view_root,
        _files=tuple(sorted(files.items())),
        _log=log,
    )
    if not view.revalidate():
        raise SourceEvidenceInvalid(
            "source ledger changed while the evidence view was published")
    return view
