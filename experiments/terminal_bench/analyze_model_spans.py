"""Read-only attribution of Codex usage attempts to verifier model spans.

Cost completeness and attribution completeness are intentionally independent:
an old, complete usage ledger remains complete cost data when no span sidecar
exists, but all of its work is reported as unattributed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


ATTEMPT_RECORD = "codex_cli_attempt"
SPAN_VERSION = 1
SPAN_TYPES = frozenset({
    "role_span_started", "role_span_finished",
    "role_attempt_started", "role_attempt_finished",
    "seat_acquired", "seat_released",
    "model_call_started", "model_call_finished",
})
TOKEN_FIELDS = (
    "input_tokens", "cached_input_tokens", "uncached_input_tokens",
    "output_tokens", "reasoning_tokens",
)
COST_TOKEN_FIELDS = (
    "input_tokens", "cached_input_tokens", "uncached_input_tokens",
    "output_tokens",
)
ATTEMPT_STATUSES = frozenset({
    "succeeded", "provider_retry", "provider_exhausted", "nonzero_exit",
    "timeout", "telemetry_error", "output_error", "schema_error",
    "interrupted", "launch_error",
})
MAX_LEDGER_BYTES = 256 * 1024 * 1024
MAX_LEDGER_ROWS = 1_000_000
MAX_LINE_CHARS = 2 * 1024 * 1024
ROLES = frozenset({
    "extract", "obligations", "critic", "oracles", "sufficiency",
    "discharge", "judger",
})
STAGES = frozenset({
    "extract", "contract_compile", "contract_critic", "oracle_enumerate",
    "oracle_repair", "sufficiency_preaudit", "discharge", "judger",
})
CLAIM_KINDS = frozenset({"assertion", "effect", "final"})
LOCI = frozenset({"candidate_extract", "support", "final_task"})
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_MODEL = re.compile(r"[A-Za-z0-9._:/+@-]{1,128}\Z")
ATTR_FIELDS = (
    "claim_id", "claim_kind", "locus", "obligation_id", "stage",
    "round", "method_attempt", "redo", "judger_retry_index",
    "independent_judger",
)
GROUP_FIELDS = ("attribution", "role", *ATTR_FIELDS)
JOIN_FIELDS = (
    "call_id", "role", "model", "attribution",
    "role_span_id", "role_attempt_id", "role_attempt_index", *ATTR_FIELDS,
)


def _read_jsonl(path: Path, *, optional: bool = False) \
        -> tuple[list[dict[str, Any]], list[str]]:
    if optional and not path.exists():
        return [], []
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_LEDGER_BYTES + 1)
        if len(data) > MAX_LEDGER_BYTES:
            return [], [f"ledger_too_large:{path.name}"]
        lines = data.decode("utf-8").split("\n")
    except (OSError, UnicodeError):
        return [], [f"cannot_read:{path.name}"]
    if len(lines) > MAX_LEDGER_ROWS:
        return [], [f"too_many_rows:{path.name}"]
    rows, errors = [], []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if len(line) > MAX_LINE_CHARS:
            errors.append(f"line_too_large:{path.name}:{lineno}")
            continue
        try:
            row = json.loads(line)
        except (ValueError, RecursionError):
            errors.append(f"invalid_json:{path.name}:{lineno}")
            continue
        if not isinstance(row, dict):
            errors.append(f"non_object:{path.name}:{lineno}")
            continue
        rows.append(row)
    return rows, errors


def _exact_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _seconds(row: Mapping[str, Any]) -> Optional[float]:
    value = row.get("duration_seconds")
    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and value >= 0 and math.isfinite(value):
        return float(value)
    value = row.get("duration_ns")
    if _exact_int(value):
        return value / 1_000_000_000
    return None


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in GROUP_FIELDS if field in row}


def _valid_identity(identity: Mapping[str, Any]) -> bool:
    for field, value in identity.items():
        if field == "attribution" and value != "attributed":
            return False
        if field == "role" and (
                not isinstance(value, str) or value not in ROLES):
            return False
        if field in ("claim_id", "obligation_id") \
                and (not isinstance(value, str) or _HEX64.fullmatch(value) is None):
            return False
        if field == "claim_kind" and (
                not isinstance(value, str) or value not in CLAIM_KINDS):
            return False
        if field == "locus" and (
                not isinstance(value, str) or value not in LOCI):
            return False
        if field == "stage" and (
                not isinstance(value, str) or value not in STAGES):
            return False
        elif field in ("round", "method_attempt", "redo",
                       "judger_retry_index"):
            if not _exact_int(value) or value > 1_000_000:
                return False
        elif field == "independent_judger" and not isinstance(value, bool):
            return False
    return True


def _valid_role_scope(identity: Mapping[str, Any]) -> bool:
    role = identity.get("role")
    stage = identity.get("stage")
    expected_stage = {
        "extract": "extract",
        "obligations": "contract_compile",
        "critic": "contract_critic",
        "oracles": "oracle_enumerate",
        "discharge": "discharge",
        "judger": "judger",
    }
    if role not in expected_stage:
        return False
    if role == "oracles":
        if stage not in {"oracle_enumerate", "oracle_repair"}:
            return False
    elif stage != expected_stage[role]:
        return False
    if role == "extract":
        return (identity.get("locus") == "candidate_extract"
                and _exact_int(identity.get("round"))
                and identity["round"] >= 1)
    required = ("claim_id", "claim_kind", "locus")
    if any(field not in identity for field in required):
        return False
    if ((identity["claim_kind"] == "final")
            != (identity["locus"] == "final_task")):
        return False
    if role in {"obligations", "critic"}:
        return _exact_int(identity.get("round"))
    if "obligation_id" not in identity:
        return False
    if role == "oracles":
        return _exact_int(identity.get("round"))
    if (not _exact_int(identity.get("method_attempt"))
            or identity["method_attempt"] < 1
            or not _exact_int(identity.get("redo"))):
        return False
    if role == "judger":
        return (_exact_int(identity.get("judger_retry_index"))
                and identity["judger_retry_index"] >= 1
                and isinstance(identity.get("independent_judger"), bool))
    return True


def _key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(identity.get(field) for field in GROUP_FIELDS)


def _bounded_id(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), default=lambda _value: "<invalid>",
        ).encode("utf-8")
    except Exception:
        encoded = type(value).__name__.encode("ascii", errors="replace")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _index_spans(rows: Iterable[dict[str, Any]], findings: list[str]):
    by_type: dict[str, dict[str, list[dict[str, Any]]]] = {
        record_type: defaultdict(list) for record_type in SPAN_TYPES
    }
    key_field = {
        "role_span_started": "role_span_id",
        "role_span_finished": "role_span_id",
        "role_attempt_started": "role_attempt_id",
        "role_attempt_finished": "role_attempt_id",
        "seat_acquired": "role_attempt_id",
        "seat_released": "role_attempt_id",
        "model_call_started": "call_id",
        "model_call_finished": "call_id",
    }
    for index, row in enumerate(rows, 1):
        record_type = row.get("record_type")
        if not isinstance(record_type, str) or record_type not in SPAN_TYPES:
            findings.append(f"unknown_span_record:{index}")
            continue
        if row.get("span_schema_version") != SPAN_VERSION:
            findings.append(f"invalid_span_version:{index}")
            continue
        identity = row.get(key_field[record_type])
        if not isinstance(identity, str) or _HEX32.fullmatch(identity) is None:
            findings.append(f"missing_span_id:{record_type}:{index}")
            continue
        by_type[record_type][identity].append(row)
    for record_type, identities in by_type.items():
        for identity, matches in identities.items():
            if len(matches) != 1:
                findings.append(
                    f"duplicate_span:{record_type}:"
                    f"{_bounded_id(identity)}:{len(matches)}")
    return by_type


def _one(index, record_type: str, identity: str,
         findings: list[str]) -> Optional[dict[str, Any]]:
    rows = index[record_type].get(identity, [])
    if len(rows) != 1:
        findings.append(
            f"span_count:{record_type}:{_bounded_id(identity)}:{len(rows)}")
        return None
    return rows[0]


def _same(left: Mapping[str, Any], right: Mapping[str, Any],
          fields: Iterable[str]) -> bool:
    return all(left.get(field) == right.get(field) for field in fields)


def _call_attribution(call_id: str, usage_rows, spans, findings):
    started = _one(spans, "model_call_started", call_id, findings)
    finished = _one(spans, "model_call_finished", call_id, findings)
    if started is None or finished is None:
        return None
    if (_HEX32.fullmatch(call_id) is None
            or started.get("role") not in ROLES
            or not isinstance(started.get("model"), str)
            or _MODEL.fullmatch(started["model"]) is None):
        findings.append(f"invalid_model_identity:{_bounded_id(call_id)}")
        return None
    if not _same(started, finished, JOIN_FIELDS):
        findings.append(f"model_call_mismatch:{_bounded_id(call_id)}")
        return None
    status = finished.get("status")
    if not isinstance(status, str) \
            or status not in {"succeeded", "error", "interrupted"} \
            or not _exact_int(finished.get("duration_ns")):
        findings.append(f"invalid_model_terminal:{_bounded_id(call_id)}")
        return None
    if not all(isinstance(row.get("role"), str) and row.get("role")
               for row in usage_rows):
        findings.append(f"invalid_usage_role:{_bounded_id(call_id)}")
        return None
    roles = {row.get("role") for row in usage_rows}
    if roles != {started.get("role")}:
        findings.append(f"usage_role_mismatch:{_bounded_id(call_id)}")
        return None
    if not all(isinstance(row.get("model"), str)
               and row.get("model") == started.get("model")
               for row in usage_rows):
        findings.append(f"usage_model_mismatch:{_bounded_id(call_id)}")
        return None
    if started.get("attribution") != "attributed":
        findings.append(f"explicit_unattributed:{_bounded_id(call_id)}")
        return None

    role_span_id = started.get("role_span_id")
    role_attempt_id = started.get("role_attempt_id")
    if (not isinstance(role_span_id, str)
            or _HEX32.fullmatch(role_span_id) is None
            or not isinstance(role_attempt_id, str)
            or _HEX32.fullmatch(role_attempt_id) is None):
        findings.append(f"missing_role_link:{_bounded_id(call_id)}")
        return None
    role_start = _one(spans, "role_span_started", role_span_id, findings)
    role_finish = _one(spans, "role_span_finished", role_span_id, findings)
    attempt_start = _one(
        spans, "role_attempt_started", role_attempt_id, findings)
    attempt_finish = _one(
        spans, "role_attempt_finished", role_attempt_id, findings)
    acquired = _one(spans, "seat_acquired", role_attempt_id, findings)
    released = _one(spans, "seat_released", role_attempt_id, findings)
    linked = (role_start, role_finish, attempt_start, attempt_finish,
              acquired, released)
    if any(row is None for row in linked):
        return None
    common = ("role_span_id", "role")
    if (not _same(role_start, role_finish, (*common, *ATTR_FIELDS))
            or not _same(attempt_start, attempt_finish,
                         ("role_attempt_id", "role_span_id", *ATTR_FIELDS))
            or not _same(started, attempt_start,
                         ("role_span_id", "role_attempt_id",
                          "role_attempt_index", *ATTR_FIELDS))
            or not _same(acquired, released,
                         ("role_span_id", "role_attempt_id",
                          "role_attempt_index", *ATTR_FIELDS))):
        findings.append(f"role_span_mismatch:{_bounded_id(call_id)}")
        return None
    if (role_start.get("role") != started.get("role")
            or acquired.get("role") != started.get("role")
            or released.get("role") != started.get("role")):
        findings.append(f"role_identity_mismatch:{_bounded_id(call_id)}")
        return None
    role_attempt_index = started.get("role_attempt_index")
    if (not _exact_int(role_attempt_index)
            or not 1 <= role_attempt_index <= 2):
        findings.append(f"invalid_role_attempt_index:{_bounded_id(call_id)}")
        return None
    role_status = role_finish.get("status")
    attempt_status = attempt_finish.get("status")
    if (not isinstance(role_status, str)
            or role_status not in {"succeeded", "error", "interrupted"}
            or not isinstance(attempt_status, str)
            or attempt_status not in {"succeeded", "error", "interrupted"}
            or not _exact_int(role_finish.get("duration_ns"))
            or not _exact_int(attempt_finish.get("duration_ns"))
            or not _exact_int(acquired.get("queue_duration_ns"))
            or not _exact_int(released.get("held_duration_ns"))):
        findings.append(f"invalid_role_terminal:{_bounded_id(call_id)}")
        return None
    identity = _identity(started)
    if not _valid_identity(identity):
        findings.append(f"invalid_attribution_metadata:{_bounded_id(call_id)}")
        return None
    if not _valid_role_scope(identity):
        findings.append(f"incomplete_role_scope:{_bounded_id(call_id)}")
        return None
    return identity


def _attempt_cost(row: Mapping[str, Any], *, legacy: bool,
                  findings: list[str], row_no: int):
    valid = True
    if legacy:
        started = True
        status = "legacy_success"
    else:
        started = row.get("started")
        raw_status = row.get("attempt_status")
        if not isinstance(raw_status, str) or raw_status not in ATTEMPT_STATUSES:
            findings.append(f"invalid_attempt_status:{row_no}")
            valid = False
            status = "invalid"
        else:
            status = raw_status
        if not isinstance(started, bool):
            findings.append(f"invalid_started:{row_no}")
            started = True  # preserve observed values, but fail completeness
            valid = False
    tokens = {field: row.get(field) for field in TOKEN_FIELDS}
    complete = valid
    if started:
        if not legacy and (row.get("usage_complete") is not True
                           or row.get("usage_status") != "complete"):
            complete = False
        if not all(_exact_int(tokens[field]) for field in COST_TOKEN_FIELDS):
            complete = False
        if all(_exact_int(tokens[field]) for field in (
                "input_tokens", "cached_input_tokens",
                "uncached_input_tokens")):
            if (tokens["cached_input_tokens"]
                    + tokens["uncached_input_tokens"]
                    != tokens["input_tokens"]):
                complete = False
        if not complete:
            findings.append(f"incomplete_usage:{row_no}")
    else:
        # Only a consistent pre-launch terminal record establishes no billing.
        # A contradictory row is unknown cost, never silently zero.
        clean_launch_error = (
            not legacy and status == "launch_error"
            and row.get("usage_complete") is False
            and row.get("usage_status") == "unknown"
            and all(row.get(field) is None for field in TOKEN_FIELDS)
        )
        if not clean_launch_error:
            findings.append(f"invalid_unstarted_attempt:{row_no}")
            complete = False
            started = True
        tokens = {field: None for field in TOKEN_FIELDS}
    duration = _seconds(row) if started else None
    return bool(started), str(status), tokens, duration, complete


def _new_group(identity: Mapping[str, Any]):
    return {
        **{field: identity.get(field) for field in GROUP_FIELDS},
        "calls": set(),
        "attempt_count": 0,
        "started_attempt_count": 0,
        "attempt_statuses": Counter(),
        "cost_complete": True,
        "tokens": {
            field: {"known_sum": 0, "unknown_started_attempts": 0}
            for field in TOKEN_FIELDS
        },
        "attempt_duration_seconds": {
            "known_sum": 0.0, "known_max": 0.0,
            "unknown_started_attempts": 0,
        },
    }


def _finish_group(group: dict[str, Any]) -> dict[str, Any]:
    group = dict(group)
    group["call_count"] = len(group.pop("calls"))
    group["attempt_statuses"] = dict(sorted(group["attempt_statuses"].items()))
    timing = group["attempt_duration_seconds"]
    timing["known_sum"] = round(timing["known_sum"], 9)
    timing["known_max"] = round(timing["known_max"], 9)
    group["timing_complete"] = timing["unknown_started_attempts"] == 0
    return group


def analyze(usage_path: Path, spans_path: Optional[Path] = None) \
        -> dict[str, Any]:
    usage_path = Path(usage_path)
    spans_path = (Path(spans_path) if spans_path is not None
                  else Path(str(usage_path.resolve()) + ".spans.jsonl"))
    usage_rows, usage_errors = _read_jsonl(usage_path)
    span_exists = spans_path.exists()
    span_rows, span_errors = _read_jsonl(spans_path, optional=True)
    attribution_findings = [f"usage_{error}" for error in usage_errors]
    attribution_findings.extend(span_errors)
    cost_findings = list(usage_errors)
    if not usage_rows:
        cost_findings.append("no_usage_attempts")
        attribution_findings.append("no_usage_attempts")
    spans = _index_spans(span_rows, attribution_findings)
    for started_type, finished_type in (
        ("role_span_started", "role_span_finished"),
        ("role_attempt_started", "role_attempt_finished"),
        ("seat_acquired", "seat_released"),
        ("model_call_started", "model_call_finished"),
    ):
        identities = set(spans[started_type]) | set(spans[finished_type])
        for identity in identities:
            if (len(spans[started_type].get(identity, [])) != 1
                    or len(spans[finished_type].get(identity, [])) != 1):
                attribution_findings.append(
                    f"incomplete_lifecycle:{started_type}:"
                    f"{_bounded_id(identity)}")

    usage_by_call: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_call_rows = []
    for row_no, row in enumerate(usage_rows, 1):
        role = row.get("role")
        if not isinstance(role, str) or not role or len(role) > 128:
            attribution_findings.append(f"invalid_usage_role:{row_no}")
        call_id = row.get("call_id")
        if isinstance(call_id, str) and _HEX32.fullmatch(call_id) is not None:
            usage_by_call[call_id].append(row)
        else:
            missing_call_rows.append((row_no, row))
            attribution_findings.append(f"missing_usage_call_id:{row_no}")

    attribution: dict[str, Optional[dict[str, Any]]] = {}
    if not span_exists and usage_rows:
        attribution_findings.append("span_sidecar_missing")
        attribution.update({call_id: None for call_id in usage_by_call})
    else:
        for call_id, rows in usage_by_call.items():
            attribution[call_id] = _call_attribution(
                call_id, rows, spans, attribution_findings)
    for call_id in spans["model_call_started"]:
        if call_id not in usage_by_call:
            attribution_findings.append(
                f"span_call_without_usage:{_bounded_id(call_id)}")
    for call_id in spans["model_call_finished"]:
        if call_id not in spans["model_call_started"]:
            attribution_findings.append(
                f"terminal_without_start:{_bounded_id(call_id)}")
    referenced_role_spans = {
        row.get("role_span_id")
        for rows in spans["model_call_started"].values() for row in rows
        if isinstance(row.get("role_span_id"), str)
    }
    referenced_role_attempts = {
        row.get("role_attempt_id")
        for rows in spans["model_call_started"].values() for row in rows
        if isinstance(row.get("role_attempt_id"), str)
    }
    for role_span_id in spans["role_span_started"]:
        if role_span_id not in referenced_role_spans:
            attribution_findings.append(
                f"role_span_without_call:{_bounded_id(role_span_id)}")
    for role_attempt_id in spans["role_attempt_started"]:
        if role_attempt_id not in referenced_role_attempts:
            attribution_findings.append(
                f"role_attempt_without_call:{_bounded_id(role_attempt_id)}")
    model_calls_by_attempt = Counter(
        row.get("role_attempt_id")
        for rows in spans["model_call_started"].values() for row in rows
        if row.get("attribution") == "attributed"
        and isinstance(row.get("role_attempt_id"), str)
    )
    role_attempt_ids = set(spans["role_attempt_started"])
    seat_ids = set(spans["seat_acquired"]) | set(spans["seat_released"])
    for role_attempt_id in role_attempt_ids:
        if model_calls_by_attempt[role_attempt_id] != 1:
            attribution_findings.append(
                f"role_attempt_call_count:{_bounded_id(role_attempt_id)}:"
                f"{model_calls_by_attempt[role_attempt_id]}")
        if role_attempt_id not in seat_ids:
            attribution_findings.append(
                f"role_attempt_without_seat:{_bounded_id(role_attempt_id)}")
    for role_attempt_id in seat_ids:
        if role_attempt_id not in role_attempt_ids:
            attribution_findings.append(
                f"seat_without_role_attempt:{_bounded_id(role_attempt_id)}")

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    seen_attempt_ids: set[str] = set()
    indices_by_call: dict[str, list[int]] = defaultdict(list)
    all_rows = [
        (row_no, row, row.get("call_id")
         if (isinstance(row.get("call_id"), str)
             and _HEX32.fullmatch(row.get("call_id")) is not None) else None)
        for row_no, row in enumerate(usage_rows, 1)
    ]
    for row_no, row, call_id in all_rows:
        if call_id is None:
            cost_findings.append(f"invalid_call_id:{row_no}")
        legacy = "record_type" not in row
        if not legacy:
            if (row.get("record_type") != ATTEMPT_RECORD
                    or row.get("schema_version") != 1
                    or row.get("attempt_schema_version") != 1):
                cost_findings.append(f"invalid_attempt_schema:{row_no}")
            attempt_id = row.get("attempt_id")
            if not isinstance(attempt_id, str) \
                    or _HEX32.fullmatch(attempt_id) is None \
                    or attempt_id in seen_attempt_ids:
                cost_findings.append(f"invalid_attempt_id:{row_no}")
            else:
                seen_attempt_ids.add(attempt_id)
            attempt_index = row.get("attempt_index")
            if call_id is not None and _exact_int(attempt_index) \
                    and attempt_index >= 1:
                indices_by_call[call_id].append(attempt_index)
            else:
                cost_findings.append(f"invalid_attempt_index:{row_no}")
        identity = attribution.get(call_id) if call_id is not None else None
        role = row.get("role")
        safe_role = role if isinstance(role, str) and role in ROLES else None
        identity = identity or {
            "attribution": "unattributed", "role": safe_role}
        key = _key(identity)
        group = groups.setdefault(key, _new_group(identity))
        group["attempt_count"] += 1
        if call_id is not None:
            group["calls"].add(call_id)
        started, status, tokens, duration, complete = _attempt_cost(
            row, legacy=legacy, findings=cost_findings, row_no=row_no)
        group["attempt_statuses"][status] += 1
        group["cost_complete"] &= complete
        if not started:
            continue
        group["started_attempt_count"] += 1
        for field, value in tokens.items():
            if _exact_int(value):
                group["tokens"][field]["known_sum"] += value
            else:
                group["tokens"][field]["unknown_started_attempts"] += 1
        timing = group["attempt_duration_seconds"]
        if duration is None:
            timing["unknown_started_attempts"] += 1
        else:
            timing["known_sum"] += duration
            timing["known_max"] = max(timing["known_max"], duration)

    for call_id, rows in usage_by_call.items():
        legacy_count = sum("record_type" not in row for row in rows)
        if legacy_count > 1 or (legacy_count and legacy_count != len(rows)):
            cost_findings.append(
                f"ambiguous_legacy_call:{_bounded_id(call_id)}")

    for call_id, indices in indices_by_call.items():
        if sorted(indices) != list(range(1, len(indices) + 1)):
            cost_findings.append(
                f"noncontiguous_attempt_indices:{_bounded_id(call_id)}")

    grouped = [_finish_group(group) for group in groups.values()]
    grouped.sort(key=lambda row: tuple(str(row.get(field) or "")
                                       for field in GROUP_FIELDS))
    attributed_calls = sum(
        1 for identity in attribution.values() if identity is not None)
    total_calls = len(usage_by_call)
    cost_complete = not cost_findings and all(
        group["cost_complete"] for group in grouped)
    attribution_complete = (
        span_exists and not attribution_findings and not missing_call_rows
        and attributed_calls == total_calls
    ) if usage_rows else not attribution_findings
    totals = {
        "attempt_count": sum(group["attempt_count"] for group in grouped),
        "started_attempt_count": sum(
            group["started_attempt_count"] for group in grouped),
        "attempt_statuses": dict(sorted(sum(
            (Counter(group["attempt_statuses"]) for group in grouped),
            Counter(),
        ).items())),
        "tokens": {
            field: {
                "known_sum": sum(
                    group["tokens"][field]["known_sum"] for group in grouped),
                "unknown_started_attempts": sum(
                    group["tokens"][field]["unknown_started_attempts"]
                    for group in grouped),
            }
            for field in TOKEN_FIELDS
        },
        "attempt_duration_seconds": {
            "known_sum": round(sum(
                group["attempt_duration_seconds"]["known_sum"]
                for group in grouped), 9),
            "known_max": max((
                group["attempt_duration_seconds"]["known_max"]
                for group in grouped), default=0.0),
            "unknown_started_attempts": sum(
                group["attempt_duration_seconds"]["unknown_started_attempts"]
                for group in grouped),
        },
    }
    return {
        "schema": "gv-model-span-analysis/v1",
        "usage_path": str(usage_path.resolve()),
        "spans_path": str(spans_path.resolve()),
        "cost_complete": cost_complete,
        "cost_findings": sorted(set(cost_findings)),
        "attribution_complete": attribution_complete,
        "attribution_findings": sorted(set(attribution_findings)),
        "call_count": total_calls,
        "attributed_call_count": attributed_calls,
        "unattributed_call_count": total_calls - attributed_calls,
        "attempt_count": len(usage_rows),
        "timing_complete": (
            not usage_errors
            and totals["attempt_duration_seconds"]
            ["unknown_started_attempts"] == 0),
        "totals": totals,
        "groups": grouped,
        "notes": [
            "known_sum is not a complete cost total when cost_complete=false",
            "cost_complete covers observed usage rows only; run lifecycle audits must establish that no attempts are missing",
            "role/attempt durations are not a dependency critical path",
            "missing spans never change usage-ledger cost completeness",
            "recognized role/stage enums are versioned with this v1 analyzer",
        ],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usage", type=Path)
    parser.add_argument("--spans", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = analyze(args.usage, args.spans)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
