"""A bounded declarative sequence query over the immutable event ledger.

``ledger.sequence`` is deliberately less expressive than
``record.program``.  It can establish only the existence of a short ordered
sequence of exact record shapes and string predicates.  Recorded invocations
may be normalized by the frozen ``record.program@1`` decoder, but neither the
selected strings nor decoded command arguments are parsed or executed.

Version 1 is holds-only and falls back on every no-match. Version 2 keeps the
same positive evidence contract, but a complete, stable, bounded observed-EOF
no-match is a method failure rather than an excuse to repeat the query with an
agent. It is never an obligation verdict. Prefix no-match, malformed input, a
resource boundary, or a racing ledger remains inapplicable in both versions.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    MAX_LEDGER_EVENTS,
    MAX_LEDGER_VALUE_BYTES,
    MAX_LEDGER_VALUE_BYTES_TOTAL,
    FIXED_INPUT_BINDING,
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
from .record_program import SOURCE_EVENT_TYPES, _normalized_invocation


MAX_CLAUSES = 8
MAX_RELATIONS = 16
MAX_STRING_PREDICATES = 32
MAX_PREDICATES_PER_CLAUSE = 8
MAX_CANDIDATES_PER_CLAUSE = 256
MAX_SEARCH_STATES = 100_000
MAX_IDENTIFIER_BYTES = 4_096
MAX_RECEIPT_BYTES = 8_192
MAX_POINTER_BYTES = 1_024
MAX_POINTER_SEGMENTS = 64
SOURCE_PROJECTION_SCHEMA = "gv-ledger-sequence-source-projection-v1"

_DECODER = "record.program@1"
_DECODER_PROFILES = frozenset({
    "codex-exec-wrapper-v1",
    "structured-exec-command-v1",
    "claude-bash-command-v1",
    "lav-bash-keystrokes-v1",
})
_PREDICATES = frozenset({"equals", "contains", "nonempty"})
_TERMINAL_POLICY = "no-later-solver-action-v1"
# A tool result is an observation, and stop is a lifecycle marker.  Neither is
# a new solver action.  This versioned protocol set is intentionally fixed;
# MethodIR cannot narrow it to manufacture terminality.
_SOLVER_ACTION_TYPES = frozenset({
    "session_start", "tool_call", "message", "commit",
})
assert SOURCE_EVENT_TYPES == _SOLVER_ACTION_TYPES | {"tool_result", "stop"}


@dataclass(frozen=True)
class _StringPredicate:
    pointer: str
    predicate: str
    expected: str | None


@dataclass(frozen=True)
class _DecoderSpec:
    profile: str
    strings: tuple[_StringPredicate, ...]


@dataclass(frozen=True)
class _Clause:
    identifier: str
    event_type: str
    selectors: tuple[tuple[str, Any], ...]
    payload_strings: tuple[_StringPredicate, ...]
    decoder: _DecoderSpec | None


@dataclass(frozen=True)
class _Relation:
    kind: str
    left: str
    right: str
    id_pointer: str | None = None


@dataclass(frozen=True)
class _Terminal:
    source: str


def _strict_utf8_size(value: str, field: str) -> int:
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeError as error:
        raise AdapterNotApplicable(f"{field} is not valid Unicode") from error
    if size > MAX_LEDGER_VALUE_BYTES:
        raise AdapterNotApplicable(
            f"{field} exceeds {MAX_LEDGER_VALUE_BYTES} UTF-8 bytes")
    return size


def _nonempty_string(value: Any, field: str, *, max_bytes: int = 256) -> str:
    if not isinstance(value, str) or not value:
        raise AdapterNotApplicable(f"{field} must be a non-empty string")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeError as error:
        raise AdapterNotApplicable(f"{field} is not valid Unicode") from error
    if size > max_bytes:
        raise AdapterNotApplicable(f"{field} exceeds {max_bytes} UTF-8 bytes")
    return value


def _sequence_pointer(value: Any, field: str) -> str:
    pointer = _validate_ledger_pointer(value, field)
    try:
        size = len(pointer.encode("utf-8", errors="strict"))
    except UnicodeError as error:
        raise AdapterNotApplicable(f"{field} is not valid Unicode") from error
    segments = 0 if pointer == "" else pointer.count("/")
    if size > MAX_POINTER_BYTES or segments > MAX_POINTER_SEGMENTS:
        raise AdapterNotApplicable(
            f"{field} exceeds the ledger sequence pointer boundary")
    return pointer


def _string_predicates(value: Any, field: str) -> tuple[_StringPredicate, ...]:
    if (not isinstance(value, list) or not value
            or len(value) > MAX_PREDICATES_PER_CLAUSE):
        raise AdapterNotApplicable(
            f"{field} must contain 1..{MAX_PREDICATES_PER_CLAUSE} predicates")
    result = []
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(item, dict):
            raise AdapterNotApplicable(f"{item_field} must be an object")
        predicate = item.get("predicate")
        expected_present = "expected" in item
        expected_keys = ({"pointer", "predicate", "expected"}
                         if predicate in {"equals", "contains"}
                         else {"pointer", "predicate"})
        if set(item) != expected_keys or predicate not in _PREDICATES:
            raise AdapterNotApplicable(
                f"{item_field} has an invalid predicate shape")
        pointer = _sequence_pointer(
            item.get("pointer"), f"{item_field}.pointer")
        expected = item.get("expected")
        if predicate in {"equals", "contains"}:
            if not expected_present or not isinstance(expected, str):
                raise AdapterNotApplicable(
                    f"{item_field}.expected must be a string")
            if predicate == "contains" and not expected:
                raise AdapterNotApplicable(
                    f"{item_field}.expected must be non-empty for contains")
            _strict_utf8_size(expected, f"{item_field}.expected")
        elif expected_present:
            raise AdapterNotApplicable(
                f"{item_field}.expected is forbidden for nonempty")
        result.append(_StringPredicate(pointer, predicate, expected))
    return tuple(result)


def _clauses(value: Any) -> tuple[_Clause, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_CLAUSES:
        raise AdapterNotApplicable(f"clauses must contain 1..{MAX_CLAUSES} entries")
    result = []
    identifiers = set()
    predicate_count = 0
    for index, item in enumerate(value):
        field = f"clauses[{index}]"
        if not isinstance(item, dict):
            raise AdapterNotApplicable(f"{field} must be an object")
        allowed = {
            "id", "event_type", "tool", "session", "agent",
            "payload_strings", "record_program",
        }
        if not {"id", "event_type"} <= set(item) or not set(item) <= allowed:
            raise AdapterNotApplicable(f"{field} has an invalid shape")
        identifier = _nonempty_string(item["id"], f"{field}.id", max_bytes=64)
        if identifier in identifiers:
            raise AdapterNotApplicable("clause ids must be unique")
        identifiers.add(identifier)
        event_type = _nonempty_string(
            item["event_type"], f"{field}.event_type", max_bytes=256)
        if event_type not in SOURCE_EVENT_TYPES:
            raise AdapterNotApplicable(
                f"{field}.event_type is not a source-side ledger event")

        selectors = []
        if "tool" in item:
            selectors.append(("tool", _nonempty_string(
                item["tool"], f"{field}.tool", max_bytes=256)))
        for name in ("session", "agent"):
            if name in item:
                selected = item[name]
                if selected is not None and not isinstance(selected, str):
                    raise AdapterNotApplicable(
                        f"{field}.{name} must be a string or null")
                if isinstance(selected, str):
                    _strict_utf8_size(selected, f"{field}.{name}")
                selectors.append((name, selected))

        payload_strings = ()
        if "payload_strings" in item:
            payload_strings = _string_predicates(
                item["payload_strings"], f"{field}.payload_strings")
            predicate_count += len(payload_strings)

        decoder = None
        if "record_program" in item:
            raw_decoder = item["record_program"]
            if not isinstance(raw_decoder, dict) or set(raw_decoder) != {
                    "decoder", "profile", "strings"}:
                raise AdapterNotApplicable(
                    f"{field}.record_program has an invalid shape")
            if raw_decoder["decoder"] != _DECODER:
                raise AdapterNotApplicable(
                    f"{field}.record_program.decoder must be {_DECODER!r}")
            profile = raw_decoder["profile"]
            if profile not in _DECODER_PROFILES:
                raise AdapterNotApplicable(
                    f"{field}.record_program.profile is unsupported")
            if event_type != "tool_call":
                raise AdapterNotApplicable(
                    f"{field}.record_program requires event_type=tool_call")
            strings = _string_predicates(
                raw_decoder["strings"], f"{field}.record_program.strings")
            predicate_count += len(strings)
            decoder = _DecoderSpec(profile, strings)

        if predicate_count > MAX_STRING_PREDICATES:
            raise AdapterNotApplicable(
                f"string predicates exceed {MAX_STRING_PREDICATES}")
        result.append(_Clause(
            identifier, event_type, tuple(selectors), payload_strings, decoder))
    return tuple(result)


def _relations(value: Any, clauses: Sequence[_Clause]) -> tuple[_Relation, ...]:
    if not isinstance(value, list) or len(value) > MAX_RELATIONS:
        raise AdapterNotApplicable(
            f"relations must be a list of at most {MAX_RELATIONS} entries")
    by_id = {clause.identifier: clause for clause in clauses}
    result = []
    seen = set()
    for index, item in enumerate(value):
        field = f"relations[{index}]"
        if not isinstance(item, dict):
            raise AdapterNotApplicable(f"{field} must be an object")
        kind = item.get("kind")
        if kind == "ordered" and set(item) == {"kind", "before", "after"}:
            left, right = item["before"], item["after"]
            pointer = None
        elif kind == "paired" and set(item) == {
                "kind", "call", "result"}:
            left, right = item["call"], item["result"]
            # Pair identity is part of the fixed EventLog protocol.  Letting
            # MethodIR choose another pointer (for example /session) would let
            # unrelated events manufacture a pair.
            pointer = "/tool_use_id"
        else:
            raise AdapterNotApplicable(f"{field} has an invalid relation shape")
        if (not isinstance(left, str) or not isinstance(right, str)
                or left not in by_id or right not in by_id or left == right):
            raise AdapterNotApplicable(
                f"{field} must reference two distinct declared clauses")
        if kind == "paired" and (
                by_id[left].event_type != "tool_call"
                or by_id[right].event_type != "tool_result"):
            raise AdapterNotApplicable(
                f"{field} requires strict tool_call/tool_result clauses")
        identity = (kind, left, right, pointer)
        if identity in seen:
            raise AdapterNotApplicable("relations must be unique")
        seen.add(identity)
        result.append(_Relation(kind, left, right, pointer))
    return tuple(result)


def _terminal(value: Any, clause_ids: set[str]) -> _Terminal | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"source", "policy"}:
        raise AdapterNotApplicable("terminal has an invalid shape")
    source = value["source"]
    if not isinstance(source, str) or source not in clause_ids:
        raise AdapterNotApplicable("terminal.source must reference a clause")
    if value["policy"] != _TERMINAL_POLICY:
        raise AdapterNotApplicable(
            f"terminal.policy must be {_TERMINAL_POLICY!r}")
    return _Terminal(source)


def _strict_equal(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def _evaluate_strings(
    value: Any,
    predicates: Sequence[_StringPredicate],
    examined_bytes: list[int],
    *,
    retain: bool = False,
) -> tuple[bool, list[Dict[str, Any]]]:
    evaluations = []
    for spec in predicates:
        try:
            actual = _json_pointer(value, spec.pointer)
        except LookupError:
            return False, evaluations
        if not isinstance(actual, str):
            return False, evaluations
        size = _strict_utf8_size(actual, "examined ledger string")
        examined_bytes[0] += size
        if examined_bytes[0] > MAX_LEDGER_VALUE_BYTES_TOTAL:
            raise AdapterNotApplicable(
                "examined strings exceed the cumulative byte limit")
        if spec.predicate == "equals":
            matched = actual == spec.expected
        elif spec.predicate == "contains":
            matched = spec.expected in actual
        else:
            matched = actual != ""
        if retain:
            evaluations.append({
                "pointer": spec.pointer,
                "predicate": spec.predicate,
                **({"expected": spec.expected}
                   if spec.predicate != "nonempty" else {}),
                "actual": actual,
                "matched": matched,
            })
        if not matched:
            return False, evaluations
    return True, evaluations


def _evaluate_clause(
    event: Mapping[str, Any],
    clause: _Clause,
    examined_bytes: list[int],
    *,
    retain: bool = False,
) -> tuple[bool, Dict[str, Any] | None]:
    if type(event.get("type")) is not str or event["type"] != clause.event_type:
        return False, None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return False, None
    selector_evaluations = []
    for name, expected in clause.selectors:
        if name not in payload or not _strict_equal(payload[name], expected):
            return False, None
        if retain:
            selector_evaluations.append({
                "field": name,
                "expected": expected,
                "actual": payload[name],
                "matched": True,
            })
    matched, payload_evaluations = _evaluate_strings(
        payload, clause.payload_strings, examined_bytes, retain=retain)
    if not matched:
        return False, None
    decoder_evaluation = None
    if clause.decoder is not None:
        decoded = _normalized_invocation(event)
        if (decoded.get("status") != "decoded"
                or decoded.get("profile") != clause.decoder.profile):
            return False, None
        matched, decoder_strings = _evaluate_strings(
            decoded, clause.decoder.strings, examined_bytes, retain=retain)
        if not matched:
            return False, None
        if retain:
            decoder_evaluation = {
                "decoder": _DECODER,
                "profile": clause.decoder.profile,
                "status": "decoded",
                "string_predicates": decoder_strings,
            }
    if not retain:
        return True, None
    return True, {
        "event_type": clause.event_type,
        "selectors": selector_evaluations,
        "payload_strings": payload_evaluations,
        "record_program": decoder_evaluation,
        "matched": True,
    }


def _matches_clause(
    event: Mapping[str, Any],
    clause: _Clause,
    examined_bytes: list[int],
) -> bool:
    matched, _evaluation = _evaluate_clause(
        event, clause, examined_bytes, retain=False)
    return matched


def _pair_groups(
    events: Sequence[Mapping[str, Any]], pointer: str,
) -> Dict[str, tuple[Mapping[str, Any], ...]]:
    groups: Dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        if event["type"] not in {"tool_call", "tool_result"}:
            continue
        payload = event["payload"]
        if not isinstance(payload, dict):
            continue
        try:
            value = _json_pointer(payload, pointer)
        except LookupError:
            continue
        if not isinstance(value, str) or not value:
            continue
        if _strict_utf8_size(value, "pair identifier") > MAX_IDENTIFIER_BYTES:
            raise AdapterNotApplicable(
                f"pair identifier exceeds {MAX_IDENTIFIER_BYTES} UTF-8 bytes")
        groups.setdefault(value, []).append(event)
    return {key: tuple(value) for key, value in groups.items()}


def _valid_pair(
    call: Mapping[str, Any],
    result: Mapping[str, Any],
    pointer: str,
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[bool, str | None]:
    if (call["type"] != "tool_call" or result["type"] != "tool_result"
            or call["seq"] >= result["seq"]):
        return False, None
    call_payload, result_payload = call["payload"], result["payload"]
    if not isinstance(call_payload, dict) or not isinstance(result_payload, dict):
        return False, None
    try:
        call_id = _json_pointer(call_payload, pointer)
        result_id = _json_pointer(result_payload, pointer)
    except LookupError:
        return False, None
    if (not isinstance(call_id, str) or not call_id
            or not _strict_equal(call_id, result_id)):
        return False, None
    grouped = groups.get(call_id, ())
    if (len(grouped) != 2 or grouped[0] is not call or grouped[1] is not result):
        return False, call_id
    for name in ("tool", "session", "agent"):
        if (name not in call_payload or name not in result_payload
                or not _strict_equal(call_payload[name], result_payload[name])):
            return False, call_id
    if (not isinstance(call_payload["tool"], str) or not call_payload["tool"]
            or (call_payload["session"] is not None
                and not isinstance(call_payload["session"], str))
            or (call_payload["agent"] is not None
                and not isinstance(call_payload["agent"], str))):
        return False, call_id
    return True, call_id


def _terminal_holds(
    selected: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    terminal: _Terminal | None,
) -> bool:
    if terminal is None:
        return True
    source = selected[terminal.source]
    return not any(
        event["seq"] > source["seq"]
        and event["type"] in _SOLVER_ACTION_TYPES
        for event in events
    )


def _complete_relations(
    clauses: Sequence[_Clause],
    chosen: Mapping[str, Mapping[str, Any]],
    relations: Sequence[_Relation],
    pair_groups: Mapping[_Relation, Mapping[str, Sequence[Mapping[str, Any]]]],
    source_event_count: int,
    source_events_digest: str,
) -> tuple[bool, list[Dict[str, Any]]]:
    # Clause order is itself semantic. Retain each adjacent edge even if the
    # method did not redundantly declare an ordered relation.
    ordered = {}
    for before, after in zip(clauses, clauses[1:]):
        left, right = chosen[before.identifier], chosen[after.identifier]
        ordered[(before.identifier, after.identifier)] = {
            "kind": "ordered", "before": before.identifier,
            "after": after.identifier, "before_seq": left["seq"],
            "after_seq": right["seq"], "source": "clause-order",
        }
    paired = []
    for relation in relations:
        left, right = chosen[relation.left], chosen[relation.right]
        if relation.kind == "ordered":
            if left["seq"] >= right["seq"]:
                return False, []
            item = ordered.setdefault((relation.left, relation.right), {
                "kind": "ordered", "before": relation.left,
                "after": relation.right, "before_seq": left["seq"],
                "after_seq": right["seq"],
            })
            item["declared"] = True
        else:
            valid, pair_id = _valid_pair(
                left, right, relation.id_pointer, pair_groups[relation])
            if not valid:
                return False, []
            paired.append({
                "kind": "paired", "call": relation.left,
                "result": relation.right,
                "call_seq": left["seq"], "result_seq": right["seq"],
                "id_pointer": relation.id_pointer,
                "id": pair_id,
                "matching_call_result_events": len(
                    pair_groups[relation].get(pair_id, ())),
                "unique_in_complete_source_scan": True,
                "scanned_source_event_count": source_event_count,
                "scanned_source_events_digest": source_events_digest,
            })
    return True, [*ordered.values(), *paired]


def _choose(
    clauses: Sequence[_Clause],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    relations: Sequence[_Relation],
    events: Sequence[Mapping[str, Any]],
    terminal: _Terminal | None,
    *,
    complete_no_match: bool = False,
) -> tuple[Dict[str, Mapping[str, Any]], list[Dict[str, Any]], int]:
    paired_relations = [item for item in relations if item.kind == "paired"]
    shared_pair_groups = (
        _pair_groups(events, "/tool_use_id") if paired_relations else {})
    pair_groups = {relation: shared_pair_groups for relation in paired_relations}
    source_events_digest = content_hash([
        EventLog.event_digest(event) for event in events])
    states = 0
    chosen: Dict[str, Mapping[str, Any]] = {}

    def visit(index: int, previous_seq: int):
        nonlocal states
        if index == len(clauses):
            valid, observed = _complete_relations(
                clauses, chosen, relations, pair_groups,
                len(events), source_events_digest)
            if valid and _terminal_holds(chosen, events, terminal):
                return dict(chosen), observed
            return None
        clause = clauses[index]
        # Clause declaration order is semantic sequence order.  Iteration by
        # increasing seq also fixes witness selection across replay.
        for event in candidates[clause.identifier]:
            states += 1
            if states > MAX_SEARCH_STATES:
                raise AdapterNotApplicable(
                    f"sequence search exceeds {MAX_SEARCH_STATES} states")
            if event["seq"] <= previous_seq:
                continue
            chosen[clause.identifier] = event
            result = visit(index + 1, event["seq"])
            if result is not None:
                return result
        chosen.pop(clause.identifier, None)
        return None

    found = visit(0, -1)
    if found is None:
        if complete_no_match:
            return {}, [], states
        raise AdapterNotApplicable(
            "no unambiguous ordered ledger sequence satisfies the declaration")
    return found[0], found[1], states


def _source_snapshot(
    all_events: Sequence[Mapping[str, Any]],
    in_range: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    after_seq: int,
    resolved_through: int,
) -> Dict[str, Any]:
    prefix_digests = [EventLog.event_digest(event) for event in all_events]
    range_digests = [EventLog.event_digest(event) for event in in_range]
    projection = [{
        "seq": event["seq"],
        "type": event["type"],
        "event_digest": digest,
    } for event, digest in zip(in_range, range_digests)]
    return {
        "ref": "ledger:events",
        "source_projection_schema": SOURCE_PROJECTION_SCHEMA,
        "source_event_types": sorted(SOURCE_EVENT_TYPES),
        "scope": scope,
        "after_seq": after_seq,
        **({"through_seq": resolved_through} if scope == "prefix" else {}),
        "source_event_count": len(prefix_digests),
        "source_events_digest": content_hash(prefix_digests),
        "resolved_source_through_seq": (
            all_events[-1]["seq"] if all_events else 0),
        "range_source_event_count": len(range_digests),
        "range_source_events_digest": content_hash(range_digests),
        "projection_digest": content_hash(projection),
        **({"complete_through_boundary": True}
           if scope == "prefix"
           else {"complete_file_at_observation": True}),
        "proves_future_closure": False,
    }


def _consumed_snapshot(snapshot: Mapping[str, Any]) -> tuple[Dict[str, Any], ...]:
    return ({
        key: snapshot[key]
        for key in (
            "ref", "source_projection_schema", "source_event_types",
            "scope", "after_seq",
            *(("through_seq",) if snapshot["scope"] == "prefix" else ()),
            "source_event_count", "source_events_digest",
            "resolved_source_through_seq", "range_source_event_count",
            "range_source_events_digest", "projection_digest",
            *(("complete_through_boundary",)
              if snapshot["scope"] == "prefix"
              else ("complete_file_at_observation",)),
            "proves_future_closure",
        )
    },)


class LedgerSequenceAdapter(MethodAdapter):
    """Establish one short ordered existential sequence, or fall back."""

    name = "ledger.sequence"
    version = 1
    basis = BASIS_RECORD
    accepted_method_reuse = False
    fixed_inputs = (FixedAdapterInput("ledger:events", PROVENANCE_LEDGER),)
    complete_no_match = False
    summary = (
        "Find 1..8 ordered existential ledger clauses using exact event and "
        "metadata selectors, RFC 6901 string predicates, structured ordering "
        "and globally unique call/result pairing. Optional terminal checks "
        "exclude declared later solver event types at observed EOF. "
        "The record.program@1 invocation decoder is frozen and inert."
    )
    arguments = {
        "scope": "prefix|observed-eof",
        "through_seq": "required inclusive watermark for prefix only",
        "after_seq": "optional exclusive lower watermark, default 0",
        "clauses": (
            "ordered list of 1..8 existential clauses; exact event_type and "
            "optional tool/session/agent, payload_strings, record_program"
        ),
        "relations": (
            "optional ordered relations; paired call/result is fixed to "
            "/tool_use_id and valid only at observed-eof"
        ),
        "terminal": (
            f"optional source clause plus policy={_TERMINAL_POLICY}; only valid "
            "at observed-eof and uses a fixed versioned solver-action set"
        ),
    }

    def catalog_entry(self) -> Dict[str, Any]:
        entry = super().catalog_entry()
        entry.update({
            "direction": "holds-only; absence/ambiguity/error/limit => fallback",
            "input_contract": (
                "adapter_input_binding=fixed-v1 materializes exactly one "
                "ref=ledger:events, provenance=ledger, role=subject; prefix "
                "consumes exactly through_seq; observed-eof fully scans the "
                "physical file but binds only the stable source-event projection; "
                "neither scope proves future closure"
            ),
            "invocation_decoder": {
                "identity": _DECODER,
                "profiles": sorted(_DECODER_PROFILES),
                "semantics": "decode frozen envelopes only; never parse or execute commands",
            },
            "source_projection": {
                "schema": SOURCE_PROJECTION_SCHEMA,
                "event_types": sorted(SOURCE_EVENT_TYPES),
                "verifier_events": "fully scanned but excluded from replay identity",
            },
            "limits": {
                "clauses": MAX_CLAUSES,
                "relations": MAX_RELATIONS,
                "candidates_per_clause": MAX_CANDIDATES_PER_CLAUSE,
                "search_states": MAX_SEARCH_STATES,
                "pointer_bytes": MAX_POINTER_BYTES,
                "pointer_segments": MAX_POINTER_SEGMENTS,
                "receipt_bytes": MAX_RECEIPT_BYTES,
            },
            "examples": [{
                "purpose": "prove one structurally unique completed tool exchange",
                "adapter_arguments": {
                    "scope": "observed-eof",
                    "clauses": [
                        {"id": "call", "event_type": "tool_call"},
                        {"id": "result", "event_type": "tool_result"},
                    ],
                    "relations": [{
                        "kind": "paired", "call": "call", "result": "result",
                    }],
                },
            }, {
                "purpose": "prove an exact final recorded command argument",
                "adapter_arguments": {
                    "scope": "observed-eof",
                    "clauses": [{
                        "id": "final-call",
                        "event_type": "tool_call",
                        "record_program": {
                            "decoder": _DECODER,
                            "profile": "codex-exec-wrapper-v1",
                            "strings": [{
                                "pointer": "/arguments/cmd",
                                "predicate": "equals",
                                "expected": "<exact command from the obligation>",
                            }],
                        },
                    }],
                    "terminal": {
                        "source": "final-call", "policy": _TERMINAL_POLICY,
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
                "ledger sequence requires verifier-owned fixed-v1 input binding")
        self.exact_arguments(
            invocation.arguments, ("scope", "clauses"),
            ("through_seq", "after_seq", "relations", "terminal"))
        clauses = _clauses(invocation.arguments["clauses"])
        relations = _relations(
            invocation.arguments.get("relations", []), clauses)
        scope = invocation.arguments["scope"]
        if scope not in {"prefix", "observed-eof"}:
            raise AdapterNotApplicable("scope must be prefix or observed-eof")
        if any(item.kind == "paired" for item in relations) \
                and scope != "observed-eof":
            raise AdapterNotApplicable(
                "paired relations require a complete observed-eof source scan")
        terminal = _terminal(
            invocation.arguments.get("terminal"),
            {item.identifier for item in clauses})
        if terminal is not None and scope != "observed-eof":
            raise AdapterNotApplicable(
                "terminal policy is valid only for observed-eof scope")

        runtime = [item for item in invocation.inputs
                   if item["role"] != ROLE_AUTHORITY]
        if runtime != [{
                "ref": "ledger:events",
                "provenance": PROVENANCE_LEDGER,
                "role": ROLE_SUBJECT,
        }]:
            raise AdapterNotApplicable(
                "ledger sequence requires exactly the fixed ledger subject")
        authorities = [item for item in invocation.inputs
                       if item["role"] == ROLE_AUTHORITY]
        if (len(authorities) > 1 or any(
                item["provenance"] != PROVENANCE_TASK_GIVEN
                for item in authorities)):
            raise AdapterNotApplicable(
                "ledger sequence permits only one task-given authority")

    def execute(self, invocation: AdapterInvocation, obligation, context):
        try:
            self.validate(invocation, {"basis": self.basis})
            if obligation.required_basis != BASIS_RECORD:
                raise AdapterNotApplicable(
                    "ledger sequence cannot settle a non-record obligation")
            return self._execute(invocation, context)
        except AdapterNotApplicable:
            raise
        except (KeyError, LookupError, TypeError, ValueError, UnicodeError,
                RecursionError, MemoryError, OverflowError, OSError) as error:
            raise AdapterNotApplicable(
                f"ledger.sequence failed closed: {error}") from error

    def _execute(self, invocation: AdapterInvocation, context):
        self.exact_arguments(
            invocation.arguments, ("scope", "clauses"),
            ("through_seq", "after_seq", "relations", "terminal"))
        arguments = invocation.arguments
        scope = arguments["scope"]
        after_seq = arguments.get("after_seq", 0)
        if type(after_seq) is not int or after_seq < 0:
            raise AdapterNotApplicable("after_seq must be a non-negative integer")
        if scope == "prefix":
            if "through_seq" not in arguments:
                raise AdapterNotApplicable(
                    "prefix scope requires through_seq")
            through_seq = arguments["through_seq"]
            if (type(through_seq) is not int or through_seq < 0
                    or through_seq > MAX_LEDGER_EVENTS):
                raise AdapterNotApplicable(
                    f"through_seq must be in 0..{MAX_LEDGER_EVENTS}")
            if after_seq > through_seq:
                raise AdapterNotApplicable("after_seq must not exceed through_seq")
        elif scope == "observed-eof":
            if "through_seq" in arguments:
                raise AdapterNotApplicable(
                    "observed-eof forbids through_seq")
            through_seq = None
        else:
            raise AdapterNotApplicable("scope must be prefix or observed-eof")

        clauses = _clauses(arguments["clauses"])
        relations = _relations(arguments.get("relations", []), clauses)
        if any(item.kind == "paired" for item in relations) \
                and scope != "observed-eof":
            raise AdapterNotApplicable(
                "paired relations require a complete observed-eof source scan")
        terminal = _terminal(
            arguments.get("terminal"), {item.identifier for item in clauses})
        if terminal is not None and scope != "observed-eof":
            raise AdapterNotApplicable(
                "terminal policy is valid only for observed-eof scope")
        self.require_subject(invocation, "ledger:events", (PROVENANCE_LEDGER,))

        eof = None
        if scope == "prefix":
            all_events_with_bytes = tuple(
                _bounded_ledger_prefix(context.log, through_seq))
            resolved_through = through_seq
        else:
            eof = _read_stable_ledger_eof(context.log)
            all_events_with_bytes = eof["events"]
            resolved_through = eof["resolved_through_seq"]
            if after_seq > resolved_through:
                raise AdapterNotApplicable(
                    "after_seq exceeds the observed ledger EOF")
        all_events = []
        for event, _bytes_read in all_events_with_bytes:
            if (not isinstance(event, dict)
                    or event.get("type") not in EVENT_TYPES
                    or not isinstance(event.get("payload"), dict)):
                raise AdapterNotApplicable(
                    "ledger sequence encountered an invalid ledger event")
            try:
                _validate_json_tree(event)
            except (UnicodeError, ValueError, TypeError, RecursionError,
                    MemoryError, OverflowError) as error:
                raise AdapterNotApplicable(
                    "ledger sequence event is not bounded valid JSON") from error
            if event["type"] in SOURCE_EVENT_TYPES:
                all_events.append(event)
        in_range = [event for event in all_events if event["seq"] > after_seq]

        examined_bytes = [0]
        snapshot = _source_snapshot(
            all_events, in_range, scope=scope, after_seq=after_seq,
            resolved_through=resolved_through)
        candidates: Dict[str, list[Mapping[str, Any]]] = {}
        missing = []
        for clause in clauses:
            matched = []
            for event in in_range:
                if _matches_clause(event, clause, examined_bytes):
                    matched.append(event)
                    if len(matched) > MAX_CANDIDATES_PER_CLAUSE:
                        raise AdapterNotApplicable(
                            f"clause {clause.identifier!r} exceeds the candidate cap")
            if not matched and not self.complete_no_match:
                raise AdapterNotApplicable(
                    f"clause {clause.identifier!r} has no matching ledger event")
            if not matched:
                missing.append(clause.identifier)
            candidates[clause.identifier] = matched

        selected_result = _choose(
            clauses, candidates, relations, all_events, terminal,
            complete_no_match=self.complete_no_match)
        if selected_result is None or not selected_result[0]:
            if scope != "observed-eof":
                raise AdapterNotApplicable(
                    "complete no-match requires a stable observed-eof scan")
            failure_kind = (
                "clause-no-match" if missing else "sequence-no-match")
            observation = {
                "schema": "gv-ledger-sequence-complete-audit-v1",
                "matched": False,
                "audit_complete": True,
                "failure_kind": failure_kind,
                "query_digest": content_hash(arguments),
                "clause_order": [clause.identifier for clause in clauses],
                "missing_clauses": missing,
                "candidate_counts": [{
                    "clause": clause.identifier,
                    "count": len(candidates[clause.identifier]),
                } for clause in clauses],
                "search_states": (
                    selected_result[2] if selected_result is not None else 0),
                "examined_string_bytes": examined_bytes[0],
                "snapshot": snapshot,
            }
            if len(canonical(observation).encode("utf-8", errors="strict")) \
                    > MAX_RECEIPT_BYTES:
                raise AdapterNotApplicable(
                    f"ledger sequence receipt exceeds {MAX_RECEIPT_BYTES} bytes")
            return AdapterExecution(
                OUTCOME_FAILED,
                {
                    "reason": (
                        "complete sequence audit produced no substantive "
                        "witness"),
                    "failure_kind": failure_kind,
                },
                WITNESS_NONE,
                observation,
                _consumed_snapshot(snapshot),
            )
        selected, observed_relations, search_states = selected_result
        witness_events = []
        for clause in clauses:
            event = selected[clause.identifier]
            matched, evaluation = _evaluate_clause(
                event, clause, [0], retain=True)
            if not matched or evaluation is None:
                raise AdapterNotApplicable(
                    "selected witness no longer satisfies its clause")
            retained = {
                "clause": clause.identifier,
                "seq": event["seq"],
                "type": event["type"],
                "payload": event["payload"],
                "event_digest": EventLog.event_digest(event),
                "evaluation": evaluation,
            }
            witness_events.append(retained)

        terminal_observation = None
        if terminal is not None:
            source = selected[terminal.source]
            suffix = [event for event in all_events
                      if event["seq"] > source["seq"]]
            later_actions = [event for event in suffix
                             if event["type"] in _SOLVER_ACTION_TYPES]
            if later_actions:
                raise AdapterNotApplicable(
                    "selected terminal witness has a later solver action")
            terminal_observation = {
                "source": terminal.source,
                "source_seq": source["seq"],
                "policy": _TERMINAL_POLICY,
                "solver_action_types": sorted(_SOLVER_ACTION_TYPES),
                "no_later_solver_action": True,
                "scanned_source_suffix_count": len(suffix),
                "later_solver_action_count": 0,
                "scanned_source_event_count": len(all_events),
                "scanned_source_events_digest": snapshot[
                    "source_events_digest"],
                "proves_future_closure": False,
            }
        observation = {
            "schema": "gv-ledger-sequence-observation-v1",
            "selection": "lexicographically-first-seq-tuple",
            "clause_order": [clause.identifier for clause in clauses],
            "witness_events": witness_events,
            "relations": observed_relations,
            "terminal": terminal_observation,
            "candidate_counts": [{
                "clause": clause.identifier,
                "count": len(candidates[clause.identifier]),
            } for clause in clauses],
            "search_states": search_states,
            "examined_string_bytes": examined_bytes[0],
            "snapshot": snapshot,
        }
        if len(canonical(observation).encode("utf-8", errors="strict")) \
                > MAX_RECEIPT_BYTES:
            raise AdapterNotApplicable(
                f"ledger sequence receipt exceeds {MAX_RECEIPT_BYTES} bytes")
        return AdapterExecution(
            OUTCOME_HOLDS, None, WITNESS_NONE, observation,
            _consumed_snapshot(snapshot))


class LedgerSequenceCompleteAuditAdapter(LedgerSequenceAdapter):
    """Versioned sequence query whose complete no-match is method failure."""

    version = 2
    complete_no_match = True
    summary = (
        "Run the same bounded ordered existential ledger query as version 1. "
        "A positive match produces the same holds evidence. A fully scanned, "
        "stable and untruncated observed-EOF no-match produces only failed "
        "(never holds or refuted), so the existing Oracle repair may choose "
        "another method. Prefix no-match, invalid input, races and resource "
        "boundaries still fall back."
    )

    def catalog_entry(self) -> Dict[str, Any]:
        entry = super().catalog_entry()
        entry["direction"] = (
            "holds-only; complete valid observed-eof no-match => failed; "
            "prefix no-match/invalid/incomplete/race/error/limit => fallback"
        )
        entry["complete_audit"] = {
            "positive": "same complete holds receipt as ledger.sequence@1",
            "no_match": (
                "method execution failed without an obligation verdict; "
                "never absence evidence and never a refutation"
            ),
            "future_closure": False,
            "method_contract": (
                "adapter arguments must exhaust the declared method; do not "
                "bind a procedure with an unrepresented alternate branch"
            ),
        }
        entry["examples"].append({
            "purpose": (
                "one-direction record certificate whose exact positive "
                "match entails the obligation; complete no-match only fails "
                "this method"
            ),
            "adapter_arguments": {
                "scope": "observed-eof",
                "clauses": [{
                    "id": "positive-witness",
                    "event_type": "tool_call",
                    "record_program": {
                        "decoder": _DECODER,
                        "profile": "codex-exec-wrapper-v1",
                        "strings": [{
                            "pointer": "/arguments/cmd",
                            "predicate": "contains",
                            "expected": (
                                "<exact record fragment whose presence "
                                "entails the obligation>"
                            ),
                        }],
                    },
                }],
            },
        })
        return entry


def typed_sequence_observation_is_complete(
    invocation: AdapterInvocation,
    evidence: Mapping[str, Any],
    result_identity: Mapping[str, Any],
) -> bool:
    """Validate the closed receipt admitted to the no-tools Judger transport.

    This does not decide whether the method is sufficient for an obligation.
    The packet builder separately performs exact fresh adapter replay before
    and after the independent Judger call.  Here we only ensure that every
    declared sequence predicate and relation has a complete bounded
    observation rather than a truncated witness. Source-wide counts are
    authenticated by the packet's exact pre/post replay, not by witness-only
    recomputation.
    """
    try:
        if ((invocation.name, invocation.version) not in {
                    ("ledger.sequence", 1), ("ledger.sequence", 2)}
                or result_identity.get("outcome") != OUTCOME_HOLDS
                or result_identity.get("witness") is not None
                or result_identity.get("witness_kind") != WITNESS_NONE
                or invocation.input_binding != FIXED_INPUT_BINDING):
            return False
        adapter = (LedgerSequenceAdapter()
                   if invocation.version == 1
                   else LedgerSequenceCompleteAuditAdapter())
        adapter.validate(invocation, {"basis": BASIS_RECORD})
        execution = evidence.get("execution")
        if not isinstance(execution, dict):
            return False
        observation = execution.get("observation")
        consumed = execution.get("consumed")
        if (not isinstance(observation, dict)
                or set(observation) != {
                    "schema", "selection", "clause_order", "witness_events",
                    "relations", "terminal", "candidate_counts",
                    "search_states", "examined_string_bytes", "snapshot",
                }
                or observation.get("schema")
                != "gv-ledger-sequence-observation-v1"
                or observation.get("selection")
                != "lexicographically-first-seq-tuple"
                or len(canonical(observation).encode(
                    "utf-8", errors="strict")) > MAX_RECEIPT_BYTES
                or not isinstance(consumed, list) or len(consumed) != 1
                or not isinstance(consumed[0], dict)):
            return False

        arguments = invocation.arguments
        clauses = _clauses(arguments["clauses"])
        relations = _relations(arguments.get("relations", []), clauses)
        clause_ids = [clause.identifier for clause in clauses]
        if observation.get("clause_order") != clause_ids:
            return False
        witnesses = observation.get("witness_events")
        if not isinstance(witnesses, list) or len(witnesses) != len(clauses):
            return False
        after_seq = arguments.get("after_seq", 0)
        selected: Dict[str, Mapping[str, Any]] = {}
        previous_seq = after_seq
        for clause, witness in zip(clauses, witnesses):
            if (not isinstance(witness, dict) or set(witness) != {
                    "clause", "seq", "type", "payload", "event_digest",
                    "evaluation"}
                    or witness.get("clause") != clause.identifier
                    or type(witness.get("seq")) is not int
                    or witness["seq"] <= previous_seq
                    or witness.get("type") != clause.event_type
                    or not isinstance(witness.get("payload"), dict)):
                return False
            if (arguments["scope"] == "prefix"
                    and witness["seq"] > arguments["through_seq"]):
                return False
            event = {
                "seq": witness["seq"], "type": witness["type"],
                "payload": witness["payload"],
            }
            _validate_json_tree(event)
            if witness.get("event_digest") != EventLog.event_digest(event):
                return False
            matched, expected_evaluation = _evaluate_clause(
                event, clause, [0], retain=True)
            if (not matched or expected_evaluation is None
                    or witness.get("evaluation") != expected_evaluation):
                return False
            selected[clause.identifier] = event
            previous_seq = witness["seq"]

        candidate_counts = observation.get("candidate_counts")
        if (not isinstance(candidate_counts, list)
                or len(candidate_counts) != len(clauses)):
            return False
        for clause, count in zip(clauses, candidate_counts):
            if (not isinstance(count, dict)
                    or set(count) != {"clause", "count"}
                    or count.get("clause") != clause.identifier
                    or type(count.get("count")) is not int
                    or not 1 <= count["count"] <= MAX_CANDIDATES_PER_CLAUSE):
                return False
        if (type(observation.get("search_states")) is not int
                or not 1 <= observation["search_states"] <= MAX_SEARCH_STATES
                or type(observation.get("examined_string_bytes")) is not int
                or not 0 <= observation["examined_string_bytes"]
                <= MAX_LEDGER_VALUE_BYTES_TOTAL):
            return False

        snapshot = observation.get("snapshot")
        scope = arguments["scope"]
        snapshot_keys = {
            "ref", "source_projection_schema", "source_event_types",
            "scope", "after_seq", "source_event_count",
            "source_events_digest", "resolved_source_through_seq",
            "range_source_event_count", "range_source_events_digest",
            "projection_digest",
            "proves_future_closure",
        } | ({"through_seq", "complete_through_boundary"}
             if scope == "prefix" else {"complete_file_at_observation"})
        if (not isinstance(snapshot, dict) or set(snapshot) != snapshot_keys
                or snapshot.get("ref") != "ledger:events"
                or snapshot.get("source_projection_schema")
                != SOURCE_PROJECTION_SCHEMA
                or snapshot.get("source_event_types")
                != sorted(SOURCE_EVENT_TYPES)
                or snapshot.get("scope") != scope
                or snapshot.get("after_seq") != after_seq
                or (scope == "prefix" and snapshot.get(
                    "complete_through_boundary") is not True)
                or (scope == "observed-eof" and snapshot.get(
                    "complete_file_at_observation") is not True)
                or snapshot.get("proves_future_closure") is not False
                or any(not isinstance(snapshot.get(name), str)
                       or len(snapshot[name]) != 64
                       or any(character not in "0123456789abcdef"
                              for character in snapshot[name])
                       for name in (
                           "source_events_digest", "range_source_events_digest",
                           "projection_digest"))):
            return False
        if scope == "prefix" and snapshot.get("through_seq") \
                != arguments["through_seq"]:
            return False
        source_count = snapshot.get("source_event_count")
        range_count = snapshot.get("range_source_event_count")
        source_through = snapshot.get("resolved_source_through_seq")
        if (type(source_count) is not int
                or type(range_count) is not int
                or type(source_through) is not int
                or not len(clauses) <= range_count <= source_count
                <= MAX_LEDGER_EVENTS
                or source_through < previous_seq
                or (scope == "prefix"
                    and source_through > arguments["through_seq"])):
            return False

        paired_relations = [item for item in relations if item.kind == "paired"]
        selected_pair_groups = (
            _pair_groups(tuple(selected.values()), "/tool_use_id")
            if paired_relations else {})
        pair_groups = {
            relation: selected_pair_groups for relation in paired_relations}
        valid, expected_relations = _complete_relations(
            clauses, selected, relations, pair_groups,
            source_count, snapshot["source_events_digest"])
        if not valid or observation.get("relations") != expected_relations:
            return False

        terminal_spec = _terminal(
            arguments.get("terminal"), set(clause_ids))
        terminal = observation.get("terminal")
        if terminal_spec is None:
            if terminal is not None:
                return False
        else:
            if (not isinstance(terminal, dict) or set(terminal) != {
                    "source", "source_seq", "policy", "solver_action_types",
                    "no_later_solver_action", "scanned_source_suffix_count",
                    "later_solver_action_count", "scanned_source_event_count",
                    "scanned_source_events_digest", "proves_future_closure",
                }
                    or terminal.get("source") != terminal_spec.source
                    or terminal.get("source_seq")
                    != selected[terminal_spec.source]["seq"]
                    or terminal.get("policy") != _TERMINAL_POLICY
                    or terminal.get("solver_action_types")
                    != sorted(_SOLVER_ACTION_TYPES)
                    or terminal.get("no_later_solver_action") is not True
                    or terminal.get("later_solver_action_count") != 0
                    or terminal.get("scanned_source_event_count") != source_count
                    or terminal.get("scanned_source_events_digest")
                    != snapshot["source_events_digest"]
                    or terminal.get("proves_future_closure") is not False
                    or type(terminal.get("scanned_source_suffix_count")) is not int
                    or not 0 <= terminal["scanned_source_suffix_count"]
                    < source_count):
                return False

        expected_consumed = {
            "ref": "ledger:events",
            "source_projection_schema": SOURCE_PROJECTION_SCHEMA,
            "source_event_types": sorted(SOURCE_EVENT_TYPES),
            "scope": scope,
            "after_seq": after_seq,
            **({"through_seq": arguments["through_seq"]}
               if scope == "prefix" else {}),
            "source_event_count": source_count,
            "source_events_digest": snapshot["source_events_digest"],
            "resolved_source_through_seq": source_through,
            "range_source_event_count": range_count,
            "range_source_events_digest": snapshot[
                "range_source_events_digest"],
            "projection_digest": snapshot["projection_digest"],
            **({"complete_through_boundary": True}
               if scope == "prefix"
               else {"complete_file_at_observation": True}),
            "proves_future_closure": False,
        }
        return consumed == [expected_consumed]
    except (AdapterNotApplicable, KeyError, LookupError, TypeError,
            ValueError, UnicodeError, RecursionError, MemoryError,
            OverflowError):
        return False
