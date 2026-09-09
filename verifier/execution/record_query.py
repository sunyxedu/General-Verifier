"""A closed declarative query algebra over immutable source ledger records.

``record.query`` covers common record obligations without executing an agent
or asking a Judger to reopen the complete ledger.  The Oracle still compiles
the MethodIR and the ordinary independent Judger still decides semantic
sufficiency for every successful result.  This adapter only replaces the
executor seat.

The algebra is deliberately small: one exact event, one existential event,
one structurally paired exchange, a last or terminal command predicate, or an
aggregate event count.  Expected literals are MethodIR inputs; the adapter
never derives them from the observed record.  A complete negative evaluation
fails only the method.  It is never an obligation refutation or an absence
witness.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from ..core.events import EVENT_TYPES, EventLog
from ..core.model import (
    BASIS_RECORD,
    OUTCOME_FAILED,
    OUTCOME_HOLDS,
    PROVENANCE_LEDGER,
    PROVENANCE_TASK_GIVEN,
    ROLE_AUTHORITY,
    ROLE_SUBJECT,
    WITNESS_NONE,
    canonical,
    content_hash,
)
from .adapters import (
    FIXED_INPUT_BINDING,
    MAX_LEDGER_EVENTS,
    AdapterExecution,
    AdapterInvocation,
    AdapterNotApplicable,
    FixedAdapterInput,
    MethodAdapter,
    _bounded_ledger_prefix,
    _json_pointer,
    _read_stable_ledger_eof,
    _validate_json_tree,
    _validate_ledger_pointer,
)
from .record_program import (
    SOURCE_EVENT_TYPES,
    _correlations,
    _normalized_invocation,
)


RECEIPT_SCHEMA = "gv-record-query-receipt-v1"
SOURCE_PROJECTION_SCHEMA = "gv-record-query-source-projection-v1"
DECODER_POLICY = "any-supported-v1"
MAX_TESTS_PER_EVENT = 8
MAX_TOTAL_TESTS = 24
MAX_POINTER_BYTES = 1_024
MAX_EXPECTED_BYTES = 4_096
MAX_RETAINED_EVENT_BYTES = 8 * 1024
MAX_RECEIPT_BYTES = 12 * 1024
MAX_TOOL_BYTES = 1_024
MAX_COMMAND_BYTES = 4 * 1024
_TEST_PREDICATES = frozenset({
    "equals", "one-of", "contains", "nonempty", "nonblank",
})
_COUNT_OPERATORS = frozenset({"equals", "at-least", "at-most"})
_NORMALIZED_OPERATIONS = frozenset({"exec_command", "bash", "terminal_input"})
OBSERVATION_FIELDS = frozenset({
    "schema", "query", "query_digest", "matched", "snapshot", "evaluation",
})
EVALUATION_FIELDS = {
    "event-at": frozenset({"kind", "candidate_count", "selected"}),
    "exists": frozenset({
        "kind", "matching_event_count", "matching_events_digest", "selected",
    }),
    "paired-exchange": frozenset({
        "kind", "matching_call_count", "matching_pair_count",
        "matching_pairs_digest", "selected",
    }),
    "last-decoded-command": frozenset({
        "kind", "decoded_invocation_count", "selected",
        "selected_operation", "selected_command", "scanned_source_suffix_count",
        "later_decoded_invocation_count",
    }),
    "terminal-exact-command": frozenset({
        "kind", "tool_call_count", "selected", "selected_decode_status",
        "selected_operation", "selected_command", "scanned_source_suffix_count",
        "later_tool_call_count",
    }),
    "count": frozenset({
        "kind", "count", "comparison", "matching_events_digest",
        "representative",
    }),
}


def _utf8_size(value: str, field: str, limit: int) -> int:
    if not isinstance(value, str):
        raise AdapterNotApplicable(f"{field} must be a string")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeError as error:
        raise AdapterNotApplicable(f"{field} is not valid Unicode") from error
    if size > limit:
        raise AdapterNotApplicable(f"{field} exceeds {limit} UTF-8 bytes")
    return size


def _strict_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and canonical(left) == canonical(right)


def _test_specs(value: Any, field: str) -> tuple[dict, ...]:
    if (not isinstance(value, list)
            or len(value) > MAX_TESTS_PER_EVENT):
        raise AdapterNotApplicable(
            f"{field} must be a list of at most {MAX_TESTS_PER_EVENT} tests")
    result = []
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(item, dict):
            raise AdapterNotApplicable(f"{item_field} must be an object")
        predicate = item.get("predicate")
        keys = ({"source", "pointer", "predicate", "expected"}
                if predicate in {"equals", "one-of", "contains"}
                else {"source", "pointer", "predicate"})
        if set(item) != keys or predicate not in _TEST_PREDICATES:
            raise AdapterNotApplicable(f"{item_field} has an invalid shape")
        source = item.get("source")
        if source not in {"payload", "invocation"}:
            raise AdapterNotApplicable(
                f"{item_field}.source must be payload or invocation")
        pointer = _validate_ledger_pointer(
            item.get("pointer"), f"{item_field}.pointer")
        if _utf8_size(pointer, f"{item_field}.pointer", MAX_POINTER_BYTES) \
                > MAX_POINTER_BYTES:
            raise AdapterNotApplicable(f"{item_field}.pointer is too large")
        normalized = {
            "source": source,
            "pointer": pointer,
            "predicate": predicate,
        }
        if predicate in {"equals", "one-of", "contains"}:
            expected = item.get("expected")
            try:
                _validate_json_tree(expected)
                encoded = canonical(expected).encode("utf-8", errors="strict")
            except (TypeError, ValueError, UnicodeError, RecursionError,
                    OverflowError) as error:
                raise AdapterNotApplicable(
                    f"{item_field}.expected is not bounded valid JSON") from error
            if len(encoded) > MAX_EXPECTED_BYTES:
                raise AdapterNotApplicable(
                    f"{item_field}.expected exceeds {MAX_EXPECTED_BYTES} bytes")
            if predicate == "contains" and (
                    not isinstance(expected, str) or not expected):
                raise AdapterNotApplicable(
                    f"{item_field}.contains requires a non-empty string")
            if predicate == "one-of":
                if (not isinstance(expected, list)
                        or not 1 <= len(expected) <= MAX_TESTS_PER_EVENT):
                    raise AdapterNotApplicable(
                        f"{item_field}.one-of requires 1.."
                        f"{MAX_TESTS_PER_EVENT} values")
                identities = {(type(item).__name__, canonical(item))
                              for item in expected}
                if len(identities) != len(expected):
                    raise AdapterNotApplicable(
                        f"{item_field}.one-of values must be unique")
            normalized["expected"] = expected
        result.append(normalized)
    return tuple(result)


def _event_predicate(value: Any, field: str) -> dict:
    if not isinstance(value, dict):
        raise AdapterNotApplicable(f"{field} must be an object")
    allowed = {
        "event_type", "tool", "session", "agent", "decode", "tests",
    }
    if set(value) - allowed or "event_type" not in value:
        raise AdapterNotApplicable(f"{field} has an invalid shape")
    event_type = value["event_type"]
    if event_type not in SOURCE_EVENT_TYPES:
        raise AdapterNotApplicable(
            f"{field}.event_type is not a source ledger event")
    normalized: dict[str, Any] = {"event_type": event_type}
    if "tool" in value:
        tool = value["tool"]
        if not isinstance(tool, str) or not tool:
            raise AdapterNotApplicable(f"{field}.tool must be non-empty")
        _utf8_size(tool, f"{field}.tool", MAX_TOOL_BYTES)
        normalized["tool"] = tool
    for name in ("session", "agent"):
        if name not in value:
            continue
        selected = value[name]
        if selected is not None:
            _utf8_size(selected, f"{field}.{name}", MAX_EXPECTED_BYTES)
        normalized[name] = selected
    if "decode" in value:
        if value["decode"] != DECODER_POLICY or event_type != "tool_call":
            raise AdapterNotApplicable(
                f"{field}.decode requires tool_call and {DECODER_POLICY}")
        normalized["decode"] = DECODER_POLICY
    tests = _test_specs(value.get("tests", []), f"{field}.tests")
    if any(item["source"] == "invocation" for item in tests) \
            and normalized.get("decode") != DECODER_POLICY:
        raise AdapterNotApplicable(
            f"{field} invocation tests require decode={DECODER_POLICY}")
    normalized["tests"] = list(tests)
    return normalized


def _query(value: Any) -> dict:
    if not isinstance(value, dict):
        raise AdapterNotApplicable("query must be an object")
    kind = value.get("kind")
    if kind == "event-at" and set(value) == {"kind", "seq", "event"}:
        seq = value["seq"]
        if type(seq) is not int or not 1 <= seq <= MAX_LEDGER_EVENTS:
            raise AdapterNotApplicable("event-at seq is out of range")
        result = {
            "kind": kind,
            "seq": seq,
            "event": _event_predicate(value["event"], "query.event"),
        }
    elif kind == "exists" and set(value) == {"kind", "event"}:
        result = {
            "kind": kind,
            "event": _event_predicate(value["event"], "query.event"),
        }
    elif kind == "paired-exchange" and set(value) == {
            "kind", "call", "result"}:
        call = _event_predicate(value["call"], "query.call")
        result_predicate = _event_predicate(
            value["result"], "query.result")
        if call["event_type"] != "tool_call" \
                or result_predicate["event_type"] != "tool_result":
            raise AdapterNotApplicable(
                "paired-exchange requires tool_call and tool_result")
        result = {
            "kind": kind, "call": call, "result": result_predicate,
        }
    elif kind in {"last-decoded-command", "terminal-exact-command"} \
            and set(value) == {
            "kind", "operation", "predicate", "expected"}:
        operation = value["operation"]
        expected = value["expected"]
        if operation not in _NORMALIZED_OPERATIONS \
                or value["predicate"] != "equals" \
                or not isinstance(expected, str):
            raise AdapterNotApplicable(
                f"{kind} requires a supported operation and an exact string")
        _utf8_size(expected, "query.expected", MAX_COMMAND_BYTES)
        result = dict(value)
    elif kind == "count" and set(value) == {
            "kind", "event", "comparison"}:
        comparison = value["comparison"]
        if (not isinstance(comparison, dict)
                or set(comparison) != {"operator", "expected"}
                or comparison.get("operator") not in _COUNT_OPERATORS
                or type(comparison.get("expected")) is not int
                or not 0 <= comparison["expected"] <= MAX_LEDGER_EVENTS):
            raise AdapterNotApplicable("count comparison is invalid")
        result = {
            "kind": kind,
            "event": _event_predicate(value["event"], "query.event"),
            "comparison": dict(comparison),
        }
    else:
        raise AdapterNotApplicable("query kind or shape is invalid")
    test_count = sum(
        len(item.get("tests", [])) for item in result.values()
        if isinstance(item, dict)
    )
    if test_count > MAX_TOTAL_TESTS:
        raise AdapterNotApplicable(
            f"query exceeds {MAX_TOTAL_TESTS} total event tests")
    return result


def _test_holds(actual: Any, spec: Mapping[str, Any]) -> bool:
    predicate = spec["predicate"]
    if predicate == "equals":
        return _strict_equal(actual, spec["expected"])
    if predicate == "one-of":
        return any(_strict_equal(actual, expected)
                   for expected in spec["expected"])
    if not isinstance(actual, str):
        return False
    if predicate == "contains":
        return spec["expected"] in actual
    if predicate == "nonempty":
        return bool(actual)
    return bool(actual.strip())


def _matches(event: Mapping[str, Any], predicate: Mapping[str, Any]) -> bool:
    if event["type"] != predicate["event_type"]:
        return False
    payload = event["payload"]
    if not isinstance(payload, dict):
        raise AdapterNotApplicable("source event payload must be an object")
    for name in ("tool", "session", "agent"):
        if name in predicate and (
                name not in payload
                or not _strict_equal(payload[name], predicate[name])):
            return False
    invocation = None
    if predicate.get("decode") == DECODER_POLICY:
        invocation = _normalized_invocation(event)
        if invocation.get("status") != "decoded":
            return False
    for spec in predicate["tests"]:
        source = payload if spec["source"] == "payload" else invocation
        if source is None:
            return False
        try:
            actual = _json_pointer(source, spec["pointer"])
        except LookupError:
            return False
        if not _test_holds(actual, spec):
            return False
    return True


def _capsule(event: Mapping[str, Any], *, correlation=None) -> dict:
    capsule = {
        "seq": event["seq"],
        "type": event["type"],
        "payload": event["payload"],
        "event_sha256": EventLog.event_digest(dict(event)),
        "invocation": _normalized_invocation(event),
    }
    if correlation is not None:
        capsule["correlation"] = correlation
    try:
        size = len(canonical(capsule).encode("utf-8", errors="strict"))
    except (TypeError, ValueError, UnicodeError, RecursionError,
            OverflowError) as error:
        raise AdapterNotApplicable(
            "selected record event is not canonical JSON") from error
    if size > MAX_RETAINED_EVENT_BYTES:
        raise AdapterNotApplicable(
            "selected record event exceeds the complete receipt boundary")
    return capsule


def _snapshot(
    all_events: Sequence[Mapping[str, Any]],
    in_range: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    after_seq: int,
    through_seq: int | None,
) -> dict:
    all_digests = [EventLog.event_digest(dict(item)) for item in all_events]
    range_digests = [EventLog.event_digest(dict(item)) for item in in_range]
    snapshot = {
        "ref": "ledger:events",
        "source_projection_schema": SOURCE_PROJECTION_SCHEMA,
        "source_event_types": sorted(SOURCE_EVENT_TYPES),
        "scope": scope,
        "after_seq": after_seq,
        "resolved_source_through_seq": (
            all_events[-1]["seq"] if all_events else 0),
        "source_event_count": len(all_events),
        "source_events_digest": content_hash(all_digests),
        "range_source_event_count": len(in_range),
        "range_source_events_digest": content_hash(range_digests),
        "complete_file_at_observation": scope == "observed-eof",
        "complete_through_boundary": scope == "prefix",
        "proves_future_closure": False,
    }
    if through_seq is not None:
        snapshot["through_seq"] = through_seq
    snapshot["projection_digest"] = content_hash(snapshot)
    return snapshot


def _compare_count(actual: int, comparison: Mapping[str, Any]) -> bool:
    operator, expected = comparison["operator"], comparison["expected"]
    if operator == "equals":
        return actual == expected
    if operator == "at-least":
        return actual >= expected
    return actual <= expected


def _evaluate(
    query: Mapping[str, Any],
    all_events: Sequence[Mapping[str, Any]],
    in_range: Sequence[Mapping[str, Any]],
) -> tuple[bool, dict]:
    kind = query["kind"]
    if kind == "event-at":
        selected = next(
            (event for event in in_range if event["seq"] == query["seq"]),
            None,
        )
        matched = selected is not None and _matches(selected, query["event"])
        return matched, {
            "kind": kind,
            "candidate_count": 1 if selected is not None else 0,
            "selected": [_capsule(selected)] if selected is not None else [],
        }

    if kind == "exists":
        matching = [event for event in in_range
                    if _matches(event, query["event"])]
        return bool(matching), {
            "kind": kind,
            "matching_event_count": len(matching),
            "matching_events_digest": content_hash([
                EventLog.event_digest(dict(item)) for item in matching]),
            "selected": [_capsule(matching[0])] if matching else [],
        }

    if kind == "paired-exchange":
        relationships = _correlations(list(all_events))
        matching_calls = [event for event in in_range
                          if _matches(event, query["call"])]
        selected_pairs = []
        ambiguous = []
        by_seq = {event["seq"]: event for event in in_range}
        for call in matching_calls:
            relationship = relationships.get(call["seq"], {"status": "unpaired"})
            if relationship.get("status") == "ambiguous":
                ambiguous.append(call["seq"])
                continue
            if relationship.get("status") != "paired":
                continue
            result = by_seq.get(relationship.get("peer_seq"))
            if (result is None or result["type"] != "tool_result"
                    or not _matches(result, query["result"])):
                continue
            selected_pairs.append((call, result, relationship,
                                   relationships[result["seq"]]))
        if ambiguous:
            raise AdapterNotApplicable(
                "a matching call has ambiguous correlation metadata")
        selected = []
        if selected_pairs:
            call, result, call_relation, result_relation = selected_pairs[0]
            selected = [
                _capsule(call, correlation=call_relation),
                _capsule(result, correlation=result_relation),
            ]
        return bool(selected_pairs), {
            "kind": kind,
            "matching_call_count": len(matching_calls),
            "matching_pair_count": len(selected_pairs),
            "matching_pairs_digest": content_hash([[
                EventLog.event_digest(dict(call)),
                EventLog.event_digest(dict(result)),
            ] for call, result, _left, _right in selected_pairs]),
            "selected": selected,
        }

    if kind == "last-decoded-command":
        decoded = []
        for event in in_range:
            if event["type"] != "tool_call":
                continue
            invocation = _normalized_invocation(event)
            if invocation.get("status") == "decoded":
                decoded.append((event, invocation))
        selected_event, selected_invocation = decoded[-1] if decoded else (None, None)
        matched = bool(
            selected_invocation is not None
            and selected_invocation.get("operation") == query["operation"]
            and selected_invocation.get("command") == query["expected"]
        )
        suffix_count = (sum(1 for event in all_events
                            if selected_event is not None
                            and event["seq"] > selected_event["seq"])
                        if selected_event is not None else 0)
        return matched, {
            "kind": kind,
            "decoded_invocation_count": len(decoded),
            "selected": ([_capsule(selected_event)]
                         if selected_event is not None else []),
            "selected_operation": (
                selected_invocation.get("operation")
                if selected_invocation is not None else None),
            "selected_command": (
                selected_invocation.get("command")
                if selected_invocation is not None else None),
            "scanned_source_suffix_count": suffix_count,
            "later_decoded_invocation_count": 0,
        }

    if kind == "terminal-exact-command":
        # Select by source event type before decoding.  This ordering is the
        # defining safety property: an opaque later call is terminal and makes
        # the exact-command predicate false instead of being skipped.
        tool_calls = [event for event in in_range
                      if event["type"] == "tool_call"]
        selected_event = tool_calls[-1] if tool_calls else None
        selected_invocation = (
            _normalized_invocation(selected_event)
            if selected_event is not None else None)
        matched = bool(
            selected_invocation is not None
            and selected_invocation.get("status") == "decoded"
            and selected_invocation.get("operation") == query["operation"]
            and selected_invocation.get("command") == query["expected"]
        )
        suffix_count = (sum(1 for event in all_events
                            if selected_event is not None
                            and event["seq"] > selected_event["seq"])
                        if selected_event is not None else 0)
        return matched, {
            "kind": kind,
            "tool_call_count": len(tool_calls),
            "selected": ([_capsule(selected_event)]
                         if selected_event is not None else []),
            "selected_decode_status": (
                selected_invocation.get("status")
                if selected_invocation is not None else None),
            "selected_operation": (
                selected_invocation.get("operation")
                if selected_invocation is not None else None),
            "selected_command": (
                selected_invocation.get("command")
                if selected_invocation is not None else None),
            "scanned_source_suffix_count": suffix_count,
            "later_tool_call_count": 0,
        }

    matching = [event for event in in_range
                if _matches(event, query["event"])]
    count = len(matching)
    return _compare_count(count, query["comparison"]), {
        "kind": "count",
        "count": count,
        "comparison": query["comparison"],
        "matching_events_digest": content_hash([
            EventLog.event_digest(dict(item)) for item in matching]),
        "representative": [_capsule(matching[0])] if matching else [],
    }


class RecordQueryAdapter(MethodAdapter):
    """Execute one closed query over a stable source-ledger projection."""

    name = "record.query"
    version = 1
    basis = BASIS_RECORD
    accepted_method_reuse = False
    fixed_inputs = (FixedAdapterInput("ledger:events", PROVENANCE_LEDGER),)
    summary = (
        "Evaluate one bounded declarative query over immutable source ledger "
        "events: event-at, exists, paired-exchange, last-decoded-command, "
        "terminal-exact-command, or count. Recorded commands are decoded only "
        "by frozen exact transport "
        "profiles and are never parsed or executed. A complete no-match fails "
        "only this method and never refutes the obligation."
    )
    arguments = {
        "scope": "prefix|observed-eof",
        "through_seq": "required raw-ledger watermark for prefix only",
        "after_seq": "optional exclusive raw-ledger lower bound, default 0",
        "query": (
            "one exact declarative event-at|exists|paired-exchange|"
            "last-decoded-command|terminal-exact-command|count query"
        ),
    }

    def evidence_contract(self) -> Dict[str, Any]:
        contract = super().evidence_contract()
        contract.update({
            "observation": {
                "schema": RECEIPT_SCHEMA,
                "required_fields": sorted(OBSERVATION_FIELDS),
                "evaluation_fields_by_query": {
                    kind: sorted(fields)
                    for kind, fields in EVALUATION_FIELDS.items()
                },
                "selected_event_fields": [
                    "seq", "type", "payload", "event_sha256", "invocation",
                ],
                "paired_event_additional_fields": ["correlation"],
                "snapshot": (
                    "complete source-event projection counts and digests for "
                    "the declared prefix or stable observed EOF"
                ),
                "terminal_exact_command": (
                    "the exact last tool_call capsule, its frozen decode, the "
                    "complete in-range tool_call count, and a zero later-call "
                    "aggregate at the declared boundary; opaque calls are not "
                    "skipped and the receipt does not prove future closure"
                ),
            },
            "retention_limits": {
                "selected_event_bytes_max": MAX_RETAINED_EVENT_BYTES,
                "observation_bytes_max": MAX_RECEIPT_BYTES,
                "complete_ledger_bytes_retained": False,
            },
        })
        return contract

    def catalog_entry(self) -> Dict[str, Any]:
        entry = super().catalog_entry()
        entry.update({
            "direction": (
                "holds-only; complete valid false query => failed; malformed, "
                "ambiguous, racing, oversized, or incomplete => fallback"
            ),
            "selection": (
                "When one query exactly covers the whole obligation predicate, "
                "use this closed declarative contract before an equivalent "
                "programmable or multi-clause contract. Never weaken temporal "
                "scope, terminality, correlation, or any other predicate merely "
                "to fit one query."
            ),
            "query_kinds": {
                "event-at": "test the exact source event at one raw seq",
                "exists": "retain the first exact matching source event",
                "paired-exchange": (
                    "retain one globally unique tool_call/tool_result pair"
                ),
                "last-decoded-command": (
                    "select the last decoded invocation before checking its "
                    "operation and exact command; proves no later decoded "
                    "invocation in the declared range, never that no later "
                    "opaque call or other action exists"
                ),
                "terminal-exact-command": (
                    "at stable observed EOF, select the last tool_call before "
                    "decoding, then require a frozen decoded invocation with "
                    "the exact operation and command; an opaque or different "
                    "later tool_call makes the query false, so a hold proves "
                    "there is no later tool_call within the declared observed "
                    "boundary, but never proves future closure"
                ),
                "count": (
                    "compare the complete matching-event count with one exact "
                    "equals, at-least, or at-most bound"
                ),
            },
            "query_shapes": {
                "event-at": {
                    "required_fields": ["kind", "seq", "event"],
                    "kind": "event-at",
                    "seq": f"integer 1..{MAX_LEDGER_EVENTS}",
                    "event": "event_predicate",
                },
                "exists": {
                    "required_fields": ["kind", "event"],
                    "kind": "exists", "event": "event_predicate",
                },
                "paired-exchange": {
                    "required_fields": ["kind", "call", "result"],
                    "kind": "paired-exchange",
                    "call": "tool_call event_predicate",
                    "result": "tool_result event_predicate",
                },
                "last-decoded-command": {
                    "required_fields": [
                        "kind", "operation", "predicate", "expected",
                    ],
                    "kind": "last-decoded-command",
                    "operation": sorted(_NORMALIZED_OPERATIONS),
                    "predicate": "equals", "expected": "exact string",
                },
                "terminal-exact-command": {
                    "required_fields": [
                        "kind", "operation", "predicate", "expected",
                    ],
                    "kind": "terminal-exact-command",
                    "operation": sorted(_NORMALIZED_OPERATIONS),
                    "predicate": "equals", "expected": "exact string",
                },
                "count": {
                    "required_fields": ["kind", "event", "comparison"],
                    "kind": "count", "event": "event_predicate",
                    "comparison": {
                        "required_fields": ["operator", "expected"],
                        "operator": sorted(_COUNT_OPERATORS),
                        "expected": f"integer 0..{MAX_LEDGER_EVENTS}",
                    },
                },
                "additional_fields": False,
            },
            "event_predicate": {
                "required": ["event_type"],
                "optional": [
                    "tool", "session", "agent", "decode", "tests",
                ],
                "decode": DECODER_POLICY,
                "tests": {
                    "source": "payload|invocation",
                    "pointer": "RFC 6901",
                    "predicate": sorted(_TEST_PREDICATES),
                    "one-of": (
                        f"expected is 1..{MAX_TESTS_PER_EVENT} unique exact "
                        "JSON values; matching uses strict JSON type and value"
                    ),
                },
                "invocation_test_requirement": (
                    f"every test with source=invocation requires "
                    f"decode={DECODER_POLICY}; undecodable tool calls do not "
                    "satisfy invocation tests"
                ),
            },
            "normalization": {
                "decoder_policy": DECODER_POLICY,
                "profiles": (
                    "the exact frozen profiles published by record.program@1"
                ),
                "operation_values": sorted(_NORMALIZED_OPERATIONS),
                "inert": True,
                "expected_values": (
                    "supplied only by MethodIR; never read from the ledger"
                ),
            },
            "scope_contract": {
                "prefix": "complete through the explicit raw-ledger boundary",
                "observed-eof": "complete stable file at one observation time",
                "future_closure": False,
                "source_event_types": sorted(SOURCE_EVENT_TYPES),
            },
            "complete_audit": {
                "false_query": (
                    "failed MethodIR only; never refuted, absence evidence, "
                    "or an obligation verdict"
                ),
                "invalid_query": "ordinary executor fallback",
            },
            "typed_transport": {
                "eligible_outcome": OUTCOME_HOLDS,
                "evidence": (
                    "complete bounded selected event(s), scope and aggregate "
                    "receipt plus exact replay before and after the fresh Judger"
                ),
            },
            "resource_limits": {
                "ledger_events": MAX_LEDGER_EVENTS,
                "tests_per_event": MAX_TESTS_PER_EVENT,
                "tests_total": MAX_TOTAL_TESTS,
                "pointer_bytes": MAX_POINTER_BYTES,
                "expected_bytes": MAX_EXPECTED_BYTES,
                "command_bytes": MAX_COMMAND_BYTES,
                "retained_event_bytes": MAX_RETAINED_EVENT_BYTES,
                "receipt_bytes": MAX_RECEIPT_BYTES,
            },
            "examples": [{
                "purpose": "one decoded nonblank direct shell command exists",
                "adapter_arguments": {
                    "scope": "observed-eof",
                    "query": {
                        "kind": "exists",
                        "event": {
                            "event_type": "tool_call",
                            "decode": DECODER_POLICY,
                            "tests": [{
                                "source": "invocation",
                                "pointer": "/operation",
                                "predicate": "one-of",
                                "expected": ["bash", "exec_command"],
                            }, {
                                "source": "invocation",
                                "pointer": "/command",
                                "predicate": "nonblank",
                            }],
                        },
                    },
                },
            }, {
                "purpose": "read one exact retained result event",
                "adapter_arguments": {
                    "scope": "observed-eof",
                    "query": {
                        "kind": "event-at", "seq": 128,
                        "event": {
                            "event_type": "tool_result", "tests": [],
                        },
                    },
                },
            }, {
                "purpose": "one decoded, structurally paired exchange",
                "adapter_arguments": {
                    "scope": "observed-eof",
                    "query": {
                        "kind": "paired-exchange",
                        "call": {
                            "event_type": "tool_call",
                            "decode": DECODER_POLICY,
                            "tests": [{
                                "source": "invocation",
                                "pointer": "/operation",
                                "predicate": "equals",
                                "expected": "exec_command",
                            }],
                        },
                        "result": {
                            "event_type": "tool_result", "tests": [],
                        },
                    },
                },
            }, {
                "purpose": "exact final decoded command",
                "replace_example_literals": True,
                "adapter_arguments": {
                    "scope": "observed-eof",
                    "query": {
                        "kind": "last-decoded-command",
                        "operation": "exec_command", "predicate": "equals",
                        "expected": "<exact command from the obligation>",
                    },
                },
            }, {
                "purpose": "exact terminal tool call at the observed boundary",
                "replace_example_literals": True,
                "adapter_arguments": {
                    "scope": "observed-eof",
                    "query": {
                        "kind": "terminal-exact-command",
                        "operation": "exec_command", "predicate": "equals",
                        "expected": "<exact command from the obligation>",
                    },
                },
            }],
        })
        return entry

    def validate(self, invocation: AdapterInvocation,
                 method: Mapping[str, Any]) -> None:
        super().validate(invocation, method)
        if invocation.input_binding != FIXED_INPUT_BINDING:
            raise AdapterNotApplicable(
                "record query requires verifier-owned fixed-v1 input binding")
        self.exact_arguments(
            invocation.arguments, ("scope", "query"),
            ("through_seq", "after_seq"))
        scope = invocation.arguments["scope"]
        after_seq = invocation.arguments.get("after_seq", 0)
        if type(after_seq) is not int or not 0 <= after_seq <= MAX_LEDGER_EVENTS:
            raise AdapterNotApplicable("after_seq is out of range")
        if scope == "prefix":
            through_seq = invocation.arguments.get("through_seq")
            if (type(through_seq) is not int
                    or not 1 <= through_seq <= MAX_LEDGER_EVENTS
                    or after_seq > through_seq):
                raise AdapterNotApplicable(
                    "prefix requires a bounded through_seq after after_seq")
        elif scope == "observed-eof":
            if "through_seq" in invocation.arguments:
                raise AdapterNotApplicable(
                    "observed-eof forbids through_seq")
        else:
            raise AdapterNotApplicable("record query scope is invalid")
        query = _query(invocation.arguments["query"])
        if query["kind"] == "terminal-exact-command" \
                and scope != "observed-eof":
            raise AdapterNotApplicable(
                "terminal-exact-command requires stable observed-eof scope")
        if query["kind"] == "event-at":
            if query["seq"] <= after_seq:
                raise AdapterNotApplicable("event-at seq is below the query range")
            if scope == "prefix" and query["seq"] > invocation.arguments[
                    "through_seq"]:
                raise AdapterNotApplicable("event-at seq exceeds the prefix")

        runtime = [item for item in invocation.inputs
                   if item["role"] != ROLE_AUTHORITY]
        if runtime != [{
                "ref": "ledger:events",
                "provenance": PROVENANCE_LEDGER,
                "role": ROLE_SUBJECT,
        }]:
            raise AdapterNotApplicable(
                "record query requires exactly the fixed ledger subject")
        authorities = [item for item in invocation.inputs
                       if item["role"] == ROLE_AUTHORITY]
        if (len(authorities) > 1 or any(
                item["provenance"] != PROVENANCE_TASK_GIVEN
                for item in authorities)):
            raise AdapterNotApplicable(
                "record query permits only one task-given authority")

    def execute(self, invocation: AdapterInvocation, obligation,
                context) -> AdapterExecution:
        try:
            self.validate(invocation, {"basis": self.basis})
            if obligation.required_basis != BASIS_RECORD:
                raise AdapterNotApplicable(
                    "record query cannot settle a non-record obligation")
            arguments = invocation.arguments
            scope = arguments["scope"]
            after_seq = arguments.get("after_seq", 0)
            if scope == "prefix":
                through_seq = arguments["through_seq"]
                raw_events = tuple(
                    _bounded_ledger_prefix(context.log, through_seq))
            else:
                eof = _read_stable_ledger_eof(context.log)
                raw_events = eof["events"]
                through_seq = None
                if after_seq > eof["resolved_through_seq"]:
                    raise AdapterNotApplicable(
                        "after_seq exceeds the observed ledger EOF")

            source_events = []
            for event, _bytes_read in raw_events:
                if (not isinstance(event, dict)
                        or event.get("type") not in EVENT_TYPES
                        or not isinstance(event.get("payload"), dict)):
                    raise AdapterNotApplicable(
                        "record query encountered an invalid ledger event")
                _validate_json_tree(event)
                if event["type"] in SOURCE_EVENT_TYPES:
                    source_events.append(event)
            in_range = [event for event in source_events
                        if event["seq"] > after_seq]
            query = _query(arguments["query"])
            matched, evaluation = _evaluate(
                query, source_events, in_range)
            snapshot = _snapshot(
                source_events, in_range, scope=scope, after_seq=after_seq,
                through_seq=through_seq,
            )
            observation = {
                "schema": RECEIPT_SCHEMA,
                "query": query,
                "query_digest": content_hash(query),
                "matched": matched,
                "snapshot": snapshot,
                "evaluation": evaluation,
            }
            if len(canonical(observation).encode("utf-8", errors="strict")) \
                    > MAX_RECEIPT_BYTES:
                raise AdapterNotApplicable(
                    "record query receipt exceeds its complete byte boundary")
            outcome = OUTCOME_HOLDS if matched else OUTCOME_FAILED
            witness = None if matched else {
                "reason": "the complete declarative record query did not hold",
                "failure_kind": "query-false",
            }
            return AdapterExecution(
                outcome, witness, WITNESS_NONE, observation, (snapshot,))
        except AdapterNotApplicable:
            raise
        except (KeyError, LookupError, TypeError, ValueError, UnicodeError,
                RecursionError, MemoryError, OverflowError, OSError) as error:
            raise AdapterNotApplicable(
                f"record.query failed closed: {error}") from error


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _typed_capsule_is_complete(
    capsule: Any,
    snapshot: Mapping[str, Any],
    *,
    paired: bool,
) -> bool:
    keys = {"seq", "type", "payload", "event_sha256", "invocation"}
    if paired:
        keys.add("correlation")
    if (not isinstance(capsule, dict) or set(capsule) != keys
            or type(capsule.get("seq")) is not int
            or capsule["type"] not in SOURCE_EVENT_TYPES
            or not isinstance(capsule.get("payload"), dict)
            or not _is_sha256(capsule.get("event_sha256"))):
        return False
    after_seq = snapshot["after_seq"]
    upper = (snapshot["through_seq"] if snapshot["scope"] == "prefix"
             else snapshot["resolved_source_through_seq"])
    if not after_seq < capsule["seq"] <= upper:
        return False
    event = {
        "seq": capsule["seq"], "type": capsule["type"],
        "payload": capsule["payload"],
    }
    _validate_json_tree(event)
    if (capsule["event_sha256"] != EventLog.event_digest(event)
            or capsule["invocation"] != _normalized_invocation(event)
            or len(canonical(capsule).encode(
                "utf-8", errors="strict")) > MAX_RETAINED_EVENT_BYTES):
        return False
    if paired:
        correlation = capsule["correlation"]
        if (not isinstance(correlation, dict)
                or set(correlation) != {"status", "peer_seq", "peer_sha256"}
                or correlation.get("status") != "paired"
                or type(correlation.get("peer_seq")) is not int
                or not _is_sha256(correlation.get("peer_sha256"))):
            return False
    return True


def typed_record_query_observation_is_complete(
    invocation: AdapterInvocation,
    evidence: Mapping[str, Any],
    result_identity: Mapping[str, Any],
) -> bool:
    """Admit a closed successful receipt; fresh replay checks aggregates."""
    try:
        if ((invocation.name, invocation.version) != ("record.query", 1)
                or invocation.input_binding != FIXED_INPUT_BINDING
                or result_identity.get("outcome") != OUTCOME_HOLDS
                or result_identity.get("witness") is not None
                or result_identity.get("witness_kind") != WITNESS_NONE):
            return False
        RecordQueryAdapter().validate(invocation, {"basis": BASIS_RECORD})
        execution = evidence.get("execution")
        if not isinstance(execution, dict):
            return False
        observation = execution.get("observation")
        consumed = execution.get("consumed")
        if (not isinstance(observation, dict)
                or set(observation) != OBSERVATION_FIELDS
                or observation.get("schema") != RECEIPT_SCHEMA
                or observation.get("matched") is not True
                or len(canonical(observation).encode("utf-8", errors="strict"))
                > MAX_RECEIPT_BYTES
                or not isinstance(consumed, list) or len(consumed) != 1):
            return False
        query = _query(invocation.arguments["query"])
        if (observation.get("query") != query
                or observation.get("query_digest") != content_hash(query)):
            return False
        snapshot = observation.get("snapshot")
        expected_keys = {
            "ref", "source_projection_schema", "source_event_types", "scope",
            "after_seq", "resolved_source_through_seq", "source_event_count",
            "source_events_digest", "range_source_event_count",
            "range_source_events_digest", "complete_file_at_observation",
            "complete_through_boundary", "proves_future_closure",
            "projection_digest",
        } | ({"through_seq"}
             if invocation.arguments["scope"] == "prefix" else set())
        if (not isinstance(snapshot, dict) or set(snapshot) != expected_keys
                or snapshot.get("ref") != "ledger:events"
                or snapshot.get("source_projection_schema")
                != SOURCE_PROJECTION_SCHEMA
                or snapshot.get("source_event_types")
                != sorted(SOURCE_EVENT_TYPES)
                or snapshot.get("scope") != invocation.arguments["scope"]
                or snapshot.get("after_seq")
                != invocation.arguments.get("after_seq", 0)
                or snapshot.get("complete_file_at_observation")
                is not (invocation.arguments["scope"] == "observed-eof")
                or snapshot.get("complete_through_boundary")
                is not (invocation.arguments["scope"] == "prefix")
                or snapshot.get("proves_future_closure") is not False):
            return False
        if invocation.arguments["scope"] == "prefix" and snapshot.get(
                "through_seq") != invocation.arguments["through_seq"]:
            return False
        without_projection = dict(snapshot)
        projection_digest = without_projection.pop("projection_digest", None)
        if projection_digest != content_hash(without_projection):
            return False
        for name in (
                "resolved_source_through_seq", "source_event_count",
                "range_source_event_count"):
            if type(snapshot.get(name)) is not int \
                    or not 0 <= snapshot[name] <= MAX_LEDGER_EVENTS:
                return False
        if (snapshot["range_source_event_count"] > snapshot["source_event_count"]
                or (snapshot["scope"] == "prefix"
                    and snapshot["resolved_source_through_seq"]
                    > snapshot["through_seq"])):
            return False
        for name in (
                "source_events_digest", "range_source_events_digest",
                "projection_digest"):
            if not _is_sha256(snapshot.get(name)):
                return False
        if consumed != [snapshot]:
            return False

        evaluation = observation.get("evaluation")
        if not isinstance(evaluation, dict) or evaluation.get("kind") \
                != query["kind"]:
            return False
        kind = query["kind"]
        expected_evaluation_keys = EVALUATION_FIELDS[kind]
        if set(evaluation) != expected_evaluation_keys:
            return False
        selected = evaluation.get(
            "representative" if kind == "count" else "selected")
        if not isinstance(selected, list):
            return False
        if kind == "paired-exchange":
            if len(selected) != 2 or not all(
                    _typed_capsule_is_complete(item, snapshot, paired=True)
                    for item in selected):
                return False
        elif kind == "count":
            if len(selected) > 1 or not all(
                    _typed_capsule_is_complete(item, snapshot, paired=False)
                    for item in selected):
                return False
        elif len(selected) != 1 or not _typed_capsule_is_complete(
                selected[0], snapshot, paired=False):
            return False
        if kind == "event-at":
            return (evaluation.get("candidate_count") == 1
                    and selected[0]["seq"] == query["seq"]
                    and _matches(selected[0], query["event"]))
        if kind == "exists":
            return (_matches(selected[0], query["event"])
                    and type(evaluation.get("matching_event_count")) is int
                    and 1 <= evaluation["matching_event_count"]
                    <= snapshot["range_source_event_count"]
                    and _is_sha256(evaluation.get(
                        "matching_events_digest")))
        if kind == "paired-exchange":
            call, result = selected
            call_payload, result_payload = call["payload"], result["payload"]
            metadata_matches = all(
                field in call_payload and field in result_payload
                and _strict_equal(call_payload[field], result_payload[field])
                for field in ("tool", "session", "agent", "tool_use_id")
            )
            return (
                _matches(call, query["call"])
                and _matches(result, query["result"])
                and call["type"] == "tool_call"
                and result["type"] == "tool_result"
                and call["seq"] < result["seq"]
                and call.get("correlation", {}).get("status") == "paired"
                and result.get("correlation", {}).get("status") == "paired"
                and call["correlation"].get("peer_seq") == result["seq"]
                and result["correlation"].get("peer_seq") == call["seq"]
                and call["correlation"].get("peer_sha256")
                == result["event_sha256"]
                and result["correlation"].get("peer_sha256")
                == call["event_sha256"]
                and metadata_matches
                and isinstance(call_payload.get("tool_use_id"), str)
                and bool(call_payload["tool_use_id"])
                and type(evaluation.get("matching_call_count")) is int
                and evaluation["matching_pair_count"]
                <= evaluation["matching_call_count"]
                <= snapshot["range_source_event_count"]
                and type(evaluation.get("matching_pair_count")) is int
                and 1 <= evaluation["matching_pair_count"]
                <= snapshot["range_source_event_count"]
                and _is_sha256(evaluation.get("matching_pairs_digest"))
            )
        if kind == "last-decoded-command":
            invocation_value = selected[0]["invocation"]
            return (
                invocation_value.get("status") == "decoded"
                and invocation_value.get("operation") == query["operation"]
                and invocation_value.get("command") == query["expected"]
                and evaluation.get("selected_operation") == query["operation"]
                and evaluation.get("selected_command") == query["expected"]
                and type(evaluation.get("decoded_invocation_count")) is int
                and 1 <= evaluation["decoded_invocation_count"]
                <= snapshot["range_source_event_count"]
                and type(evaluation.get("scanned_source_suffix_count")) is int
                and 0 <= evaluation["scanned_source_suffix_count"]
                < snapshot["source_event_count"]
                and evaluation.get("later_decoded_invocation_count") == 0
            )
        if kind == "terminal-exact-command":
            invocation_value = selected[0]["invocation"]
            return (
                selected[0]["type"] == "tool_call"
                and invocation_value.get("status") == "decoded"
                and invocation_value.get("operation") == query["operation"]
                and invocation_value.get("command") == query["expected"]
                and evaluation.get("selected_decode_status") == "decoded"
                and evaluation.get("selected_operation") == query["operation"]
                and evaluation.get("selected_command") == query["expected"]
                and type(evaluation.get("tool_call_count")) is int
                and 1 <= evaluation["tool_call_count"]
                <= snapshot["range_source_event_count"]
                and type(evaluation.get("scanned_source_suffix_count")) is int
                and 0 <= evaluation["scanned_source_suffix_count"]
                < snapshot["source_event_count"]
                and evaluation.get("later_tool_call_count") == 0
            )
        count = evaluation.get("count")
        return (
            type(count) is int and 0 <= count <= MAX_LEDGER_EVENTS
            and count <= snapshot["range_source_event_count"]
            and evaluation.get("comparison") == query["comparison"]
            and _is_sha256(evaluation.get("matching_events_digest"))
            and _compare_count(count, query["comparison"])
            and len(selected) == (1 if count else 0)
            and (not selected or _matches(selected[0], query["event"]))
        )
    except (AdapterNotApplicable, KeyError, LookupError, TypeError,
            ValueError, UnicodeError, RecursionError, MemoryError,
            OverflowError):
        return False
