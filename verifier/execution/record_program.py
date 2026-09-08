"""A bounded pure program over a normalized immutable ledger projection.

``record.program`` is the record-basis sibling of ``evidence.program``.  It
uses the same closed interpreter, but its sole runtime input is verifier-owned:
a canonical JSON projection of source-side ledger events.  It never executes
or interprets a recorded command.  A small, versioned transport decoder may
recover an exact command argument from a known envelope; every other envelope
is retained verbatim and marked opaque.

The adapter is holds-only.  Malformed records, ambiguity, false requirements,
resource exhaustion, or a racing ledger make it inapplicable and preserve the
ordinary executor/Judger path.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Mapping

from ..core.events import EVENT_TYPES, EventLog
from ..core.model import (
    BASIS_RECORD,
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
    _read_stable_ledger_eof,
    _validate_json_tree,
)
from .evidence_program import (
    LANGUAGE,
    MAX_ASSERTIONS,
    MAX_AST_DEPTH,
    MAX_AST_NODES,
    MAX_CALL_DEPTH,
    MAX_CUMULATIVE_ALLOCATION_BYTES,
    MAX_FUEL,
    MAX_INPUT_BYTES,
    MAX_INTEGER_BITS,
    MAX_LOOP_ITEMS,
    MAX_OBSERVATIONS,
    MAX_RETAINED_OBSERVATION_CHARS,
    MAX_SOURCE_CHARS,
    MAX_VALUE_BYTES,
    MAX_VALUE_ITEMS,
    _Interpreter,
    _source_digest,
)


NORMALIZATION_SCHEMA = "gv-normalized-record-events-v1"
MAX_RECEIPT_CHARS = 12_000
MAX_CORRELATION_ID_BYTES = 4_096
MAX_ASSERTION_EVENT_REFS = 16

# This is the source-side part of the versioned EventLog protocol, not a
# content heuristic.  Verification events are deliberately absent from the
# program input, so verifier-only concurrent appends do not change an
# observed-EOF source projection.  A protocol change requires a new adapter
# version.
SOURCE_EVENT_TYPES = frozenset({
    "session_start", "tool_call", "tool_result", "message", "commit", "stop",
})
assert SOURCE_EVENT_TYPES <= frozenset(EVENT_TYPES)


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


_WRAPPER_PREFIX = re.compile(
    r"\s*const\s+r\s*=\s*await\s+tools\.exec_command\s*\(\s*", re.ASCII)
_WRAPPER_SUFFIX = re.compile(
    r"\s*\)\s*;\s*text\s*\(\s*r\.output\s*\)\s*;\s*\Z", re.ASCII)


def _decode_codex_exec_wrapper(value: Any) -> Dict[str, Any] | None:
    """Decode one exact inert transport envelope, never its shell string."""
    if not isinstance(value, str) or len(value) > MAX_INPUT_BYTES:
        return None
    prefix = _WRAPPER_PREFIX.match(value)
    if prefix is None:
        return None
    decoder = json.JSONDecoder(
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )
    try:
        arguments, end = decoder.raw_decode(value, prefix.end())
    except (ValueError, json.JSONDecodeError, RecursionError, OverflowError):
        return None
    if (_WRAPPER_SUFFIX.fullmatch(value, end) is None
            or not isinstance(arguments, dict)
            or not isinstance(arguments.get("cmd"), str)):
        return None
    try:
        _validate_json_tree(arguments)
        arguments["cmd"].encode("utf-8", errors="strict")
    except (UnicodeError, ValueError, AdapterNotApplicable):
        return None
    return {
        "status": "decoded",
        "profile": "codex-exec-wrapper-v1",
        "operation": "exec_command",
        "command": arguments["cmd"],
        "arguments": arguments,
    }


def _normalized_invocation(event: Mapping[str, Any]) -> Dict[str, Any]:
    if event.get("type") != "tool_call":
        return {"status": "not-applicable"}
    payload = event["payload"]
    tool = payload.get("tool")
    value = payload.get("input")

    # These are frozen transport schemas.  A key that merely happens to be
    # named cmd/command on an unrelated tool is never promoted to a command.
    if tool == "exec" and isinstance(value, str):
        decoded = _decode_codex_exec_wrapper(value)
        if decoded is not None:
            return decoded
    if tool == "exec_command" and isinstance(value, dict) \
            and isinstance(value.get("cmd"), str):
        return {
            "status": "decoded",
            "profile": "structured-exec-command-v1",
            "operation": "exec_command",
            "command": value["cmd"],
            "arguments": value,
        }
    if tool == "Bash" and isinstance(value, dict) \
            and isinstance(value.get("command"), str):
        return {
            "status": "decoded",
            "profile": "claude-bash-command-v1",
            "operation": "bash",
            "command": value["command"],
            "arguments": value,
        }
    if tool == "bash" and isinstance(value, dict) \
            and isinstance(value.get("keystrokes"), str):
        return {
            "status": "decoded",
            "profile": "lav-bash-keystrokes-v1",
            "operation": "terminal_input",
            "command": value["keystrokes"],
            "arguments": value,
        }
    return {"status": "opaque"}


def _correlations(events: list[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Pair only a globally unique call/result identifier and exact metadata."""
    by_id: Dict[str, list[Mapping[str, Any]]] = {}
    relationships: Dict[int, Dict[str, Any]] = {}
    for event in events:
        if event["type"] not in {"tool_call", "tool_result"}:
            relationships[event["seq"]] = {"status": "not-applicable"}
            continue
        event_id = event["payload"].get("tool_use_id")
        if (not isinstance(event_id, str) or not event_id
                or len(event_id.encode("utf-8", errors="strict"))
                > MAX_CORRELATION_ID_BYTES):
            relationships[event["seq"]] = {"status": "unpaired"}
            continue
        by_id.setdefault(event_id, []).append(event)

    for grouped in by_id.values():
        if len(grouped) != 2:
            for event in grouped:
                relationships[event["seq"]] = {"status": "ambiguous"}
            continue
        call, result = grouped
        if call["type"] == "tool_result" and result["type"] == "tool_call":
            call, result = result, call
        metadata_matches = all(
            field in call["payload"] and field in result["payload"]
            and canonical(call["payload"][field])
            == canonical(result["payload"][field])
            for field in ("tool", "session", "agent", "tool_use_id")
        )
        if (call["type"] != "tool_call" or result["type"] != "tool_result"
                or call["seq"] >= result["seq"] or not metadata_matches):
            for event in grouped:
                relationships[event["seq"]] = {"status": "ambiguous"}
            continue
        call_digest = EventLog.event_digest(dict(call))
        result_digest = EventLog.event_digest(dict(result))
        relationships[call["seq"]] = {
            "status": "paired",
            "peer_seq": result["seq"],
            "peer_sha256": result_digest,
        }
        relationships[result["seq"]] = {
            "status": "paired",
            "peer_seq": call["seq"],
            "peer_sha256": call_digest,
        }
    return relationships


def _projection(all_events, scope: Mapping[str, Any]) -> tuple[dict, bytes]:
    source_events = []
    for event, _bytes_read in all_events:
        if (not isinstance(event, dict)
                or event.get("type") not in EVENT_TYPES
                or not isinstance(event.get("payload"), dict)):
            raise AdapterNotApplicable("record program ledger event is invalid")
        try:
            # Correlation inspects identifier bytes before the final document
            # is canonicalized.  Validate each event first so hostile Unicode
            # or an oversized tree is always an ordinary inapplicable adapter,
            # never an exception leaking out of normalization.
            _validate_json_tree(event)
        except (UnicodeError, ValueError, TypeError, RecursionError,
                OverflowError) as error:
            raise AdapterNotApplicable(
                "record program ledger event is not bounded valid JSON") from error
        if event["type"] in SOURCE_EVENT_TYPES:
            source_events.append(event)
    relationships = _correlations(source_events)
    normalized = []
    event_digests = []
    for event in source_events:
        digest = EventLog.event_digest(event)
        event_digests.append(digest)
        normalized.append({
            "seq": event["seq"],
            "type": event["type"],
            "payload": event["payload"],
            "event_sha256": digest,
            "correlation": relationships[event["seq"]],
            "invocation": _normalized_invocation(event),
        })
    events_digest = content_hash(event_digests)
    document = {
        "schema": NORMALIZATION_SCHEMA,
        "scope": dict(scope),
        "source_event_count": len(normalized),
        "source_events_digest": events_digest,
        "events": normalized,
    }
    try:
        _validate_json_tree(document)
        raw = canonical(document).encode("utf-8", errors="strict")
    except (UnicodeError, ValueError, TypeError, RecursionError,
            OverflowError) as error:
        raise AdapterNotApplicable(
            "record program projection is not bounded canonical JSON") from error
    if len(raw) > MAX_INPUT_BYTES:
        raise AdapterNotApplicable(
            f"record program projection exceeds {MAX_INPUT_BYTES} bytes")
    return document, raw


def _typed_observation_validation(assertions, document: Mapping[str, Any]) -> dict:
    by_seq = {event["seq"]: event["event_sha256"]
              for event in document["events"]}
    scope = {
        **document["scope"],
        "source_events_digest": document["source_events_digest"],
    }
    referenced: Dict[int, str] = {}
    valid = bool(assertions)
    for assertion in assertions:
        observed = assertion.get("observed") if isinstance(assertion, dict) else None
        if not isinstance(observed, dict) or set(observed) != {
                "event_refs", "facts", "scope"} or observed.get("scope") != scope:
            valid = False
            continue
        refs = observed.get("event_refs")
        if (not isinstance(refs, list) or not refs
                or len(refs) > MAX_ASSERTION_EVENT_REFS):
            valid = False
            continue
        seen = set()
        for ref in refs:
            if (not isinstance(ref, dict) or set(ref) != {"seq", "sha256"}
                    or type(ref.get("seq")) is not int
                    or not isinstance(ref.get("sha256"), str)
                    or ref["seq"] in seen
                    or by_seq.get(ref["seq"]) != ref["sha256"]):
                valid = False
                continue
            seen.add(ref["seq"])
            referenced[ref["seq"]] = ref["sha256"]
    return {
        "valid": valid,
        "referenced_events": [
            {"seq": seq, "sha256": digest}
            for seq, digest in sorted(referenced.items())
        ],
    }


class RecordProgramAdapter(MethodAdapter):
    """Execute the pure evidence VM over one fixed record projection."""

    name = "record.program"
    version = 1
    basis = BASIS_RECORD
    accepted_method_reuse = False
    fixed_inputs = (FixedAdapterInput("ledger:events", PROVENANCE_LEDGER),)
    summary = (
        "Interpret the bounded pure evidence language over one verifier-owned "
        "canonical projection of immutable source ledger events. Calls and "
        "results are paired only when structurally unambiguous; recorded "
        "commands are retained as inert strings and are never parsed or run. "
        "The adapter reports holds only; every failure uses the ordinary executor."
    )
    arguments = {
        "source": f"{LANGUAGE} text, 1..{MAX_SOURCE_CHARS} characters",
        "scope": "prefix|observed-eof",
        "through_seq": (
            "required positive raw-ledger watermark for prefix; forbidden for "
            "observed-eof"
        ),
    }

    def catalog_entry(self) -> Dict[str, Any]:
        entry = super().catalog_entry()
        entry.update({
            "direction": "holds-only; false/error/limit => ordinary executor fallback",
            "language": {
                "identity": LANGUAGE,
                "reuse": "the exact evidence.program@1 interpreter",
            },
            "normalization": {
                "schema": NORMALIZATION_SCHEMA,
                "events": "canonical seq/type/payload plus immutable event digest",
                "correlation": (
                    "unique exact tool_use_id and matching tool/session/agent only; "
                    "missing, reused, reordered, or mismatched records never pair"
                ),
                "invocations": (
                    "versioned exact transport-envelope decoding only; no shell "
                    "parsing, command execution, path heuristic, or task builtin"
                ),
                "observed_eof": (
                    "complete stable ledger file at one observation time; verifier "
                    "events are excluded from the source projection; never proves "
                    "future closure"
                ),
            },
            "input_contract": (
                "adapter_input_binding=fixed-v1 materializes exactly one "
                "ledger:events provenance=ledger subject slot; an optional sole "
                "task_given authority is not a runtime slot"
            ),
            "judgment_transport": (
                "always the ordinary fresh toolful Judger; an arbitrary record "
                "program and sparse event references never qualify for the "
                "restricted no-tools packet"
            ),
            "example": (
                "record = input_json(0)\n"
                "found = False\n"
                "for event in record[\"events\"]:\n"
                "    if event[\"type\"] == \"tool_call\" and "
                "event[\"invocation\"][\"status\"] == \"decoded\" and "
                "event[\"correlation\"][\"status\"] == \"paired\":\n"
                "        found = True\n"
                "require(found, \"one structurally paired invocation exists\")"
            ),
            "resource_limits": {
                "source_chars": MAX_SOURCE_CHARS,
                "ast_nodes": MAX_AST_NODES,
                "ast_depth": MAX_AST_DEPTH,
                "ledger_events": MAX_LEDGER_EVENTS,
                "projection_bytes": MAX_INPUT_BYTES,
                "fuel": MAX_FUEL,
                "loop_items": MAX_LOOP_ITEMS,
                "value_items": MAX_VALUE_ITEMS,
                "value_bytes": MAX_VALUE_BYTES,
                "cumulative_allocation_bytes": MAX_CUMULATIVE_ALLOCATION_BYTES,
                "integer_bits": MAX_INTEGER_BITS,
                "call_depth": MAX_CALL_DEPTH,
                "assertions": MAX_ASSERTIONS,
                "observations": MAX_OBSERVATIONS,
                "event_refs_per_assertion": MAX_ASSERTION_EVENT_REFS,
                "retained_observation_chars": MAX_RETAINED_OBSERVATION_CHARS,
                "receipt_chars": MAX_RECEIPT_CHARS,
            },
        })
        return entry

    def validate(self, invocation: AdapterInvocation,
                 method: Mapping[str, Any]) -> None:
        super().validate(invocation, method)
        if invocation.input_binding != FIXED_INPUT_BINDING:
            raise AdapterNotApplicable(
                "record program requires verifier-owned fixed-v1 input binding")
        self.exact_arguments(
            invocation.arguments, ("source", "scope"), ("through_seq",))
        source = invocation.arguments["source"]
        if (type(source) is not str or not source or len(source) > MAX_SOURCE_CHARS
                or "\x00" in source):
            raise AdapterNotApplicable("record program source is invalid or too large")
        try:
            source.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise AdapterNotApplicable(
                "record program source is not valid Unicode") from error
        scope = invocation.arguments["scope"]
        through_present = "through_seq" in invocation.arguments
        if scope == "prefix":
            through = invocation.arguments.get("through_seq")
            if (not through_present or type(through) is not int
                    or not 1 <= through <= MAX_LEDGER_EVENTS):
                raise AdapterNotApplicable(
                    "record program prefix requires a bounded positive through_seq")
        elif scope == "observed-eof":
            if through_present:
                raise AdapterNotApplicable(
                    "record program observed-eof forbids through_seq")
        else:
            raise AdapterNotApplicable("record program scope is invalid")

        runtime = [item for item in invocation.inputs
                   if item["role"] != ROLE_AUTHORITY]
        if runtime != [{
                "ref": "ledger:events",
                "provenance": PROVENANCE_LEDGER,
                "role": ROLE_SUBJECT,
        }]:
            raise AdapterNotApplicable(
                "record program requires exactly the fixed ledger subject")
        authorities = [item for item in invocation.inputs
                       if item["role"] == ROLE_AUTHORITY]
        if (len(authorities) > 1 or any(
                item["provenance"] != PROVENANCE_TASK_GIVEN
                for item in authorities)):
            raise AdapterNotApplicable(
                "record program permits only one task-given authority")

    def execute(self, invocation: AdapterInvocation, obligation,
                context) -> AdapterExecution:
        self.validate(invocation, {"basis": self.basis})
        if obligation.required_basis != BASIS_RECORD:
            raise AdapterNotApplicable(
                "record program cannot settle a non-record obligation")
        if invocation.arguments["scope"] == "prefix":
            through = invocation.arguments["through_seq"]
            events = tuple(_bounded_ledger_prefix(context.log, through))
            scope = {
                "kind": "prefix",
                "through_seq": through,
                "complete_through_boundary": True,
                "proves_future_closure": False,
            }
        else:
            eof = _read_stable_ledger_eof(context.log)
            events = eof["events"]
            scope = {
                "kind": "observed-eof",
                "complete_file_at_observation": True,
                "proves_future_closure": False,
            }
        document, raw = _projection(events, scope)
        source = invocation.arguments["source"]
        try:
            result = _Interpreter(source, [raw], {0}).run()
        except (RecursionError, MemoryError) as error:
            raise AdapterNotApplicable(
                "record program exceeded a host safety boundary") from error
        typed_validation = _typed_observation_validation(
            result["assertions"], document)
        projection_digest = hashlib.sha256(raw).hexdigest()
        observation = {
            "language": LANGUAGE,
            "normalization_schema": NORMALIZATION_SCHEMA,
            "source_sha256": _source_digest(source),
            "source_chars": len(source),
            "record_scope": scope,
            "projection": {
                "bytes": len(raw),
                "sha256": projection_digest,
                "source_event_count": document["source_event_count"],
                "source_events_digest": document["source_events_digest"],
            },
            "input_slots": [{
                "slot": 0,
                "ref": "ledger:events",
                "provenance": PROVENANCE_LEDGER,
                "role": ROLE_SUBJECT,
                "bytes": len(raw),
                "sha256": projection_digest,
            }],
            "typed_observation_validation": typed_validation,
            **result,
        }
        if len(canonical(observation)) > MAX_RECEIPT_CHARS:
            raise AdapterNotApplicable("record program receipt exceeds its bound")
        consumed = ({
            "ref": "ledger:events",
            "scope": scope,
            "normalization_schema": NORMALIZATION_SCHEMA,
            "projection_bytes": len(raw),
            "projection_sha256": projection_digest,
            "source_event_count": document["source_event_count"],
            "source_events_digest": document["source_events_digest"],
        },)
        return AdapterExecution(
            OUTCOME_HOLDS, None, WITNESS_NONE, observation, consumed)
