"""
Unified event stream: append-only JSONL, the system's only input and its only log. 
Replay = reading the same file.
"""
from __future__ import annotations

import json
import hashlib
import re
import math
import os
import stat
import tempfile
import threading
import time
from typing import Any, Dict, Iterator, Optional, Tuple

from .model import content_hash

SOURCE_EVENT_TYPES = (
    # The candidate/source record.  This tuple is an integrity boundary: a
    # source-evidence view must never infer source membership from names or
    # from whichever event types happen to exist at runtime.
    "session_start",
    "tool_call",
    "tool_result",
    "message",
    "commit",
    "stop",
)

MAX_SOURCE_LEDGER_BYTES = 32 * 1024 * 1024
MAX_SOURCE_LEDGER_EVENTS = 20_000

# Frozen records: an exact, immutable prefix of the ledger published once per
# verified claim.  Seats and the independent Judger read the same bounded
# file, so a verifier append during verification can never make a complete
# record scan look stale.
RECORD_DIRECTORY = "record"
_RECORD_NAME = re.compile(r"events-through-(\d+)-([0-9a-f]{16})\.jsonl\Z")

EVENT_TYPES = (
    *SOURCE_EVENT_TYPES,

    # Declarative claims reconstructed by the offline Extractor.  These feed
    # the same graph builder as live commits, but are verifier output rather
    # than candidate/source actions and therefore must not enter a source
    # evidence projection.
    "extracted_commit",

    # scheduling
    "verify_request",
    "record_frozen",
    "status_change",
    "certificate",

    # admission screening (online economy: a claim the record settles is
    # assumed, not checked)
    "claim_screened",

    # obligation & critic agent
    "obligations_compiled",
    "obligations_residualized",
    "decomposition_attacked",
    "contract_repair_declined",
    "contract_cache_miss",
    "contract_cache_hit",
    "contract_cache_published",
    "contract_cache_bypass",
    "contract_precompiled",
    
    # binding game
    "methods_enumerated",
    "sufficiency_audited",
    "executor_briefed",
    "judger_call_failed",
    "judger_verdict",
    "method_rejected",
    "method_deferred",
    "method_redo",
    "method_bound",
    "method_cache_hit",
    "method_cache_direction_audit",
    "method_cache_bypass",
    "method_cache_published",
    "adapter_discharge",
    "adapter_fallback",
    "discharge",

    # search support
    "dedup_hit",
    
    # memory (advisory)
    "memory_recorded",
)


class SourceProjectionUnavailable(ValueError):
    """The event ledger cannot be projected as a stable strict source."""


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _strict_json(raw: bytes) -> Any:
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )


def _validate_event_payload(value: Any, *, depth: int = 0) -> None:
    """Require the in-memory payload to have one lossless JSON identity."""
    if depth > 128:
        raise ValueError("event payload exceeds the JSON depth limit")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("event payload contains a non-finite number")
        return
    if isinstance(value, str):
        value.encode("utf-8", errors="strict")
        return
    if isinstance(value, list):
        for item in value:
            _validate_event_payload(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("event payload object keys must be text")
            key.encode("utf-8", errors="strict")
            _validate_event_payload(item, depth=depth + 1)
        return
    raise ValueError(
        f"event payload contains non-JSON value {type(value).__name__}")


def _directory_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    if not all(hasattr(os, name) for name in required):
        raise SourceProjectionUnavailable(
            "safe source projection is unsupported on this platform")
    return (os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0))


def _file_flags() -> int:
    # O_NONBLOCK makes an accidental special-file substitution fail without
    # waiting, while the fstat checks below still require a regular file.
    return (os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            | getattr(os, "O_CLOEXEC", 0))


def _file_identity(value: os.stat_result) -> Tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_event_log(path: str) -> bytes:
    """Read one real, private ledger path without following aliases.

    EventLog's writer lock is held by the caller.  Descriptor and path
    identities are checked on both sides of the read as defence against
    mutation by actors which do not participate in that lock.
    """
    absolute = os.path.abspath(os.fspath(path))
    parent = os.path.realpath(os.path.dirname(absolute))
    name = os.path.basename(absolute)
    if not name or name in {".", ".."} or os.sep in name:
        raise SourceProjectionUnavailable(
            "event ledger path must have a final file name")

    parent_fd = file_fd = -1
    try:
        expected_parent = os.lstat(parent)
        if (not stat.S_ISDIR(expected_parent.st_mode)
                or stat.S_ISLNK(expected_parent.st_mode)):
            raise SourceProjectionUnavailable(
                "event ledger parent must be a real directory")
        parent_fd = os.open(parent, _directory_flags())
        opened_parent = os.fstat(parent_fd)
        if (not stat.S_ISDIR(opened_parent.st_mode)
                or (opened_parent.st_dev, opened_parent.st_ino)
                != (expected_parent.st_dev, expected_parent.st_ino)):
            raise SourceProjectionUnavailable(
                "event ledger parent changed while opening")

        try:
            expected = os.stat(
                name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            # An uncreated EventLog has a complete empty source projection.
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return b""
            raise SourceProjectionUnavailable(
                "event ledger appeared while being projected")
        if (not stat.S_ISREG(expected.st_mode)
                or stat.S_ISLNK(expected.st_mode)
                or expected.st_nlink != 1):
            raise SourceProjectionUnavailable(
                "event ledger must be a private regular file")
        if expected.st_size > MAX_SOURCE_LEDGER_BYTES:
            raise SourceProjectionUnavailable(
                "event ledger exceeds the source projection byte limit")

        file_fd = os.open(name, _file_flags(), dir_fd=parent_fd)
        before = os.fstat(file_fd)
        if (not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or (before.st_dev, before.st_ino)
                != (expected.st_dev, expected.st_ino)):
            raise SourceProjectionUnavailable(
                "event ledger changed before reading")
        before_identity = _file_identity(before)
        chunks = []
        total = 0
        while True:
            block = os.read(
                file_fd,
                min(1024 * 1024, MAX_SOURCE_LEDGER_BYTES + 1 - total),
            )
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > MAX_SOURCE_LEDGER_BYTES:
                raise SourceProjectionUnavailable(
                    "event ledger exceeds the source projection byte limit")
        if _file_identity(os.fstat(file_fd)) != before_identity:
            raise SourceProjectionUnavailable(
                "event ledger changed while reading")
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _file_identity(current) != before_identity:
            raise SourceProjectionUnavailable(
                "event ledger path changed while reading")
        return b"".join(chunks)
    except SourceProjectionUnavailable:
        raise
    except (OSError, ValueError) as error:
        raise SourceProjectionUnavailable(
            f"cannot safely read event ledger: {error}") from error
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _strict_events(raw: bytes) -> Tuple[Dict[str, Any], ...]:
    if raw and not raw.endswith(b"\n"):
        raise SourceProjectionUnavailable(
            "event ledger has an incomplete final record")
    events = []
    expected_seq = 0
    try:
        # JSONL is delimited only by LF.  splitlines() would also treat the
        # legal Unicode character U+0085 as a line boundary after decoding.
        for line in raw.split(b"\n")[:-1]:
            if not line:
                raise ValueError("event ledger contains an empty record")
            event = _strict_json(line)
            expected_seq += 1
            if expected_seq > MAX_SOURCE_LEDGER_EVENTS:
                raise ValueError("event ledger exceeds the source event limit")
            if (not isinstance(event, dict)
                    or set(event) != {"seq", "ts", "type", "payload"}):
                raise ValueError("event has an unsupported shape")
            seq = event["seq"]
            timestamp = event["ts"]
            if (isinstance(seq, bool) or not isinstance(seq, int)
                    or seq != expected_seq):
                raise ValueError(
                    f"expected event seq {expected_seq}, got {seq!r}")
            if (isinstance(timestamp, bool)
                    or not isinstance(timestamp, (int, float))
                    or not math.isfinite(timestamp)):
                raise ValueError("event timestamp must be finite")
            if event["type"] not in EVENT_TYPES:
                raise ValueError(f"unknown event type {event['type']!r}")
            if not isinstance(event["payload"], dict):
                raise ValueError("event payload must be an object")
            # Reject escaped lone surrogates and any value which cannot be
            # represented by the canonical UTF-8 event identity.
            content_hash({
                "seq": seq,
                "type": event["type"],
                "payload": event["payload"],
            })
            expected_line = json.dumps(
                event, ensure_ascii=False, sort_keys=True, allow_nan=False,
            ).encode("utf-8", errors="strict")
            if line != expected_line:
                raise ValueError(
                    "event record is not in the EventLog writer encoding")
            event["_source_record_digest"] = hashlib.sha256(line).hexdigest()
            events.append(event)
    except SourceProjectionUnavailable:
        raise
    except (UnicodeDecodeError, UnicodeEncodeError, ValueError,
            json.JSONDecodeError, RecursionError, TypeError) as error:
        raise SourceProjectionUnavailable(
            f"event ledger is not strict UTF-8 JSONL: {error}") from error
    return tuple(events)


class EventLog:
    """
    Append-only JSONL. Every event gets a monotonically increasing seq.

    The file is only ever opened in append mode;
    replay validates monotonicity so silent truncation/reordering is detected rather than absorbed.
    """

    def __init__(self, path: str):
        self.path = path
        # one writer at a time: seq allocation and the write are atomic
        # together, so concurrent seats (parallel obligations/claims) can
        # never interleave into a gap or a duplicate that replay rejects
        self._lock = threading.Lock()
        self._seq = self._last_seq()
        # claim id -> absolute path of the frozen record bound to that
        # claim's verification; every seat resolves its ledger through here
        self._records: Dict[str, str] = {}
        self._records_lock = threading.Lock()

    def _last_seq(self) -> int:
        if not os.path.exists(self.path):
            return 0
        last = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line)["seq"]
        return last

    def append(self, type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if type not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {type}")
        if not isinstance(payload, dict):
            raise ValueError("event payload must be an object")
        _validate_event_payload(payload)
        with self._lock:
            next_seq = self._seq + 1
            event = {"seq": next_seq, "ts": time.time(), "type": type,
                     "payload": payload}
            # Encode before changing the in-memory sequence or touching the
            # file. Invalid Unicode/NaN can never create a permanent gap.
            raw = (json.dumps(
                event, ensure_ascii=False, sort_keys=True, allow_nan=False,
            ) + "\n").encode("utf-8", errors="strict")
            with open(self.path, "ab") as f:
                f.write(raw)
                f.flush()
            self._seq = next_seq
        return event


    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    def freeze_record(self, through_seq: int) -> str:
        """Publish the exact ledger prefix through ``through_seq``.

        The prefix is written verbatim (the writer's own encoding, one record
        per line) to ``<state>/record/events-through-<seq>-<digest16>.jsonl``
        as a read-only file and its absolute path is returned.  The name is
        content-addressed, so freezing the same boundary twice is idempotent
        and a tampered publication is detected rather than reused.
        """
        if (isinstance(through_seq, bool) or not isinstance(through_seq, int)
                or through_seq < 0):
            raise ValueError("record boundary must be a nonnegative integer")
        with self._lock:
            if through_seq > self._seq:
                raise ValueError("record boundary lies beyond the ledger")
            raw = _read_stable_event_log(self.path) if through_seq else b""
            kept = []
            count = 0
            for line in raw.split(b"\n"):
                if count == through_seq:
                    break
                if not line.strip():
                    continue
                try:
                    seq = json.loads(line)["seq"]
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(
                        "ledger prefix is not valid JSONL") from error
                count += 1
                if seq != count:
                    raise ValueError(
                        f"ledger prefix is not contiguous at seq {count}")
                kept.append(line)
            if count != through_seq:
                raise ValueError("ledger ends before the record boundary")
            prefix = b"\n".join(kept) + (b"\n" if kept else b"")
            digest = hashlib.sha256(prefix).hexdigest()
            directory = os.path.join(
                os.path.dirname(os.path.abspath(self.path)), RECORD_DIRECTORY)
            os.makedirs(directory, mode=0o700, exist_ok=True)
            target = os.path.join(
                directory, f"events-through-{through_seq}-{digest[:16]}.jsonl")
            if os.path.lexists(target):
                if os.path.islink(target) or not os.path.isfile(target):
                    raise ValueError("frozen record path is not a regular file")
                with open(target, "rb") as handle:
                    existing = handle.read(len(prefix) + 1)
                if existing != prefix:
                    raise ValueError("frozen record does not match its name")
                return target
            fd, temporary = tempfile.mkstemp(prefix=".freezing-", dir=directory)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(prefix)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o444)
                os.replace(temporary, target)
                temporary = ""
            finally:
                if temporary:
                    try:
                        os.remove(temporary)
                    except OSError:
                        pass
            return target

    @staticmethod
    def record_boundary_of(path: str) -> Optional[int]:
        """The ``through_seq`` named by a frozen record path, else ``None``."""
        match = _RECORD_NAME.match(os.path.basename(str(path)))
        return int(match.group(1)) if match else None

    def bind_record(self, claim_id: str, path: str) -> None:
        with self._records_lock:
            self._records[claim_id] = os.path.abspath(path)

    def record_path_for(self, claim_id: Optional[str]) -> str:
        """The ledger a seat verifying ``claim_id`` must read.

        A claim whose verification froze a record gets that immutable prefix;
        anything else keeps the live ledger, exactly as before.
        """
        with self._records_lock:
            bound = self._records.get(claim_id) if claim_id else None
        return bound or os.path.abspath(self.path)

    def is_record_path(self, path: Any) -> bool:
        if not isinstance(path, str):
            return False
        with self._records_lock:
            return path in set(self._records.values())

    def replay(self) -> Iterator[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return
        expected = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                expected += 1
                if event["seq"] != expected:
                    raise ValueError(
                        f"event log corrupted: expected seq {expected}, got {event['seq']}"
                    )
                yield event

    def get(self, seq: int) -> Optional[Dict[str, Any]]:
        for event in self.replay():
            if event["seq"] == seq:
                return event
        return None

    def source_projection(self) -> Tuple[Dict[str, Any], ...]:
        """Return one stable, detached projection of candidate-source events.

        The same lock which allocates sequence numbers and appends events is
        held for the complete safe read and projection.  Consequently a
        source view is always built from either side of an EventLog append,
        never from a torn in-process prefix.  Verifier events are parsed and
        sequence-checked but are intentionally absent from the result.
        """
        with self._lock:
            events = _strict_events(_read_stable_event_log(self.path))
            return tuple({
                "seq": event["seq"],
                "ts": event["ts"],
                "type": event["type"],
                "payload": event["payload"],
                "event_digest": self.event_digest(event),
                "source_record_digest": event["_source_record_digest"],
            } for event in events if event["type"] in SOURCE_EVENT_TYPES)

    @staticmethod
    def event_digest(event: Dict[str, Any]) -> str:
        """Digest over the immutable identity of an event (ts excluded so the
        ledger check does not depend on wall-clock noise)."""
        return content_hash({
            "seq": event["seq"], 
            "type": event["type"], 
            "payload": event["payload"]
        })
