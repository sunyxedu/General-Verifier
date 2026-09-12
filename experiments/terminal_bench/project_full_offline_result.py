"""Project one cross-version offline result onto a common strict contract view.

The original verifier predates the convenience fields now emitted by
``offline.score``.  This helper therefore reads the durable
``obligations_compiled`` event and aligns the certificate's final-obligation
outcomes to compiler order.  It is an evaluation reader only: it does not
import a verifier version, mutate its state, or infer a missing outcome.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


SCHEMA = "gv-full-offline-result-projection-v1"
MAX_EVENTS_BYTES = 64 * 1024 * 1024
MAX_EVENT_ROWS = 1_000_000
MAX_EVENT_LINE_BYTES = 2 * 1024 * 1024

_CONTRACT_FIELDS = (
    "id", "text", "premise_scope", "required_basis", "severity",
    "clause", "refutable_by_absence",
)
_OUTCOME_FIELDS = ("status", "reason", "basis", "strength")
_STAT_FIELDS = (
    "st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_size",
    "st_mtime_ns", "st_ctime_ns",
)
_STATUSES = frozenset({"verified", "refuted", "partial", "unverified", "stale"})
_BASES = frozenset({"record", "world"})
_STRENGTHS = frozenset({"decisive", "bounded", "analytic"})


class _ProjectionError(ValueError):
    def __init__(self, finding: str):
        super().__init__(finding)
        self.finding = finding


def _strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise _ProjectionError("events_duplicate_json_key")
        value[key] = item
    return value


def _nonfinite(_value):
    raise _ProjectionError("events_nonfinite_number")


def _identity(info):
    return tuple(getattr(info, field) for field in _STAT_FIELDS)


def _read_events(path: str | os.PathLike[str]) -> tuple[list[dict[str, Any]], str]:
    try:
        raw_path = os.fspath(path)
    except TypeError as error:
        raise _ProjectionError("events_invalid_path") from error
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise _ProjectionError("events_invalid_path")
    absolute = os.path.abspath(raw_path)
    if os.path.realpath(absolute) != absolute:
        raise _ProjectionError("events_noncanonical_path")
    try:
        before_path = os.lstat(absolute)
        flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0))
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise _ProjectionError("events_unavailable") from error
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                or opened.st_uid != os.geteuid()
                or opened.st_size > MAX_EVENTS_BYTES
                or _identity(opened) != _identity(before_path)):
            raise _ProjectionError("events_unsafe_file")
        chunks = []
        remaining = MAX_EVENTS_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as error:
        raise _ProjectionError("events_read_failed") from error
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(absolute)
    except OSError as error:
        raise _ProjectionError("events_changed_during_read") from error
    if (len(raw) > MAX_EVENTS_BYTES or len(raw) != opened.st_size
            or _identity(opened) != _identity(after)
            or _identity(after) != _identity(after_path)):
        raise _ProjectionError("events_changed_during_read")
    if not raw or not raw.endswith(b"\n"):
        raise _ProjectionError("events_empty_or_partial")
    lines = raw.split(b"\n")[:-1]
    if len(lines) > MAX_EVENT_ROWS:
        raise _ProjectionError("events_too_many_rows")
    rows = []
    for number, line in enumerate(lines, 1):
        if not line or len(line) > MAX_EVENT_LINE_BYTES:
            raise _ProjectionError(f"events_invalid_line:{number}")
        try:
            row = json.loads(
                line.decode("utf-8", errors="strict"),
                object_pairs_hook=_strict_object,
                parse_constant=_nonfinite,
            )
        except _ProjectionError:
            raise
        except (UnicodeError, ValueError, TypeError, RecursionError,
                OverflowError) as error:
            raise _ProjectionError(f"events_invalid_json:{number}") from error
        if type(row) is not dict:
            raise _ProjectionError(f"events_non_object:{number}")
        rows.append(row)
    return rows, hashlib.sha256(raw).hexdigest()


def _certificate_view(certificate: Any) -> tuple[str, dict[str, Any]]:
    if type(certificate) is not dict:
        raise _ProjectionError("certificate_non_object")
    final_id = certificate.get("final")
    entries = certificate.get("slice")
    if not isinstance(final_id, str) or not final_id:
        raise _ProjectionError("certificate_missing_final")
    if type(entries) is not list:
        raise _ProjectionError("certificate_missing_slice")
    if any(type(entry) is not dict for entry in entries):
        raise _ProjectionError("certificate_slice_malformed")
    finals = [entry for entry in entries
              if entry.get("id") == final_id]
    if len(finals) != 1:
        raise _ProjectionError("certificate_final_not_unique")
    final = finals[0]
    if type(final.get("obligations")) is not list:
        raise _ProjectionError("certificate_missing_final_outcomes")
    return final_id, final


def _ordered_contract(
    events: list[dict[str, Any]], final_id: str,
) -> list[dict[str, Any]]:
    compiled: Any = None
    found = False
    for event in events:
        if event.get("type") != "obligations_compiled":
            continue
        payload = event.get("payload")
        if type(payload) is dict and payload.get("claim") == final_id:
            compiled = payload.get("obligations")
            found = True
    if not found:
        raise _ProjectionError("final_contract_missing")
    if type(compiled) is not list:
        raise _ProjectionError("final_contract_malformed")
    if not compiled:
        raise _ProjectionError("final_contract_empty")
    result = []
    ids = set()
    for ordinal, obligation in enumerate(compiled):
        if (type(obligation) is not dict
                or any(field not in obligation for field in _CONTRACT_FIELDS)):
            raise _ProjectionError(f"contract_item_malformed:{ordinal}")
        oid = obligation["id"]
        if (not isinstance(oid, str) or not oid or oid in ids
                or not isinstance(obligation["text"], str)
                or not obligation["text"]
                or type(obligation["premise_scope"]) is not list
                or any(not isinstance(item, str)
                       for item in obligation["premise_scope"])
                or obligation["required_basis"] not in _BASES
                or obligation["severity"] not in {"must", "should"}
                or not isinstance(obligation["clause"], str)
                or type(obligation["refutable_by_absence"]) is not bool):
            raise _ProjectionError(f"contract_item_malformed:{ordinal}")
        ids.add(oid)
        result.append({
            "ordinal": ordinal,
            **{field: (list(obligation[field])
                       if field == "premise_scope" else obligation[field])
               for field in _CONTRACT_FIELDS},
        })
    return result


def _ordered_outcomes(
    final: Mapping[str, Any], contract: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    outcomes = {}
    for index, item in enumerate(final["obligations"]):
        if type(item) is not dict:
            raise _ProjectionError(f"outcome_item_malformed:{index}")
        oid = item.get("id")
        if not isinstance(oid, str) or not oid or oid in outcomes:
            raise _ProjectionError(f"outcome_id_invalid_or_duplicate:{index}")
        if (item.get("status") not in _STATUSES
                or (item.get("reason") is not None
                    and not isinstance(item.get("reason"), str))
                or item.get("basis") not in (*_BASES, None)
                or item.get("strength") not in (*_STRENGTHS, None)):
            raise _ProjectionError(f"outcome_item_malformed:{index}")
        outcomes[oid] = item
    contract_ids = {item["id"] for item in contract}
    if set(outcomes) - contract_ids:
        raise _ProjectionError("certificate_has_unexpected_obligation")
    result = []
    for item in contract:
        outcome = outcomes.get(item["id"])
        result.append({
            "ordinal": item["ordinal"],
            "obligation_id": item["id"],
            "evaluated": outcome is not None,
            **{field: outcome.get(field) if outcome is not None else None
               for field in _OUTCOME_FIELDS},
        })
    return result


def project_full_offline_result(
    events_path: str | os.PathLike[str], certificate: Any,
) -> dict[str, Any]:
    """Return an exact cross-version projection, or a fail-closed report."""
    report = {
        "schema": SCHEMA,
        "complete": False,
        "findings": [],
        "events_sha256": None,
        "final_claim_id": None,
        "ordered_contract": [],
        "ordered_outcomes": [],
    }
    try:
        events, digest = _read_events(events_path)
        report["events_sha256"] = digest
        final_id, final = _certificate_view(certificate)
        contract = _ordered_contract(events, final_id)
        outcomes = _ordered_outcomes(final, contract)
    except _ProjectionError as error:
        report["findings"] = [error.finding]
        return report
    except (OSError, ValueError, TypeError, KeyError, RecursionError,
            OverflowError):
        report["findings"] = ["projection_unavailable"]
        return report
    report.update(
        complete=True,
        final_claim_id=final_id,
        ordered_contract=contract,
        ordered_outcomes=outcomes,
    )
    return report


__all__ = ["SCHEMA", "project_full_offline_result"]
