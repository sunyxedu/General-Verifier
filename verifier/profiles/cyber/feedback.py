"""Outcome-blind transport for the cybersecurity Oracle's repair pass.

This filter consumes host-assembled BindingGame diagnostics, not an authority
or a model verdict. Only the method's *pre-execution declaration*, fixed
structural classifications and retry/availability metadata reach the compiler.
Execution outcomes, witnesses, free-text audit reasons, output and artifact
contents remain in the original event/evidence record for the executor and
independent Judger. Nothing here settles an obligation or changes acceptance.

An unavailable instrument is a reported structural blocker, not proof that
the security proposition holds or fails. A fixed evidence boundary describes
the host's offline mode; it does not promote a retryable tool error to an
eternally missing capability. Unknown reasons are deliberately unclassified.
"""
from __future__ import annotations

import json
from typing import Any

from ...core.model import content_hash


SCHEMA = "gv-cyber-oracle-feedback-v1"
MAX_FEEDBACK_BYTES = 4 * 1024 * 1024
MAX_ITEMS = 256
MAX_DECLARATION_CHARS = 1024 * 1024
_STRENGTHS = frozenset({"decisive", "bounded", "analytic"})
_PROVENANCE = frozenset({"task_given", "delivered", "ledger", "verifier", "public"})
_INPUT_ROLES = frozenset({"subject", "input", "authority"})


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate feedback field")
        result[key] = value
    return result


def _no_constant(_value):
    raise ValueError("non-finite feedback value")


def _mapping(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError("invalid feedback object")
    return value


def _items(value: Any) -> list:
    if not isinstance(value, list) or len(value) > MAX_ITEMS:
        raise ValueError("invalid feedback list")
    return value


def _text(value: Any) -> str:
    if not isinstance(value, str) or len(value) > MAX_DECLARATION_CHARS:
        raise ValueError("invalid method declaration")
    return value


def _boolean(value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError("invalid feedback flag")
    return value


def _enum(value: Any, choices: frozenset) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError("invalid declaration enum")
    return value


def _method(source: Any) -> dict:
    """Whitelist the bound MethodIR, never implementation/fallback notes."""
    source = _mapping(source)
    declaration = {
        "procedure": _text(source.get("procedure", "")),
        "artifact": _text(source.get("artifact", source.get("promised_artifact", ""))),
        "level": _text(source.get("level", "")),
        "basis": _enum(source.get("basis", "record"), frozenset({"record", "world"})),
        "strength": _enum(source.get("strength", "analytic"), _STRENGTHS),
        "bridges": [_text(value) for value in _items(source.get("bridges", []))],
    }
    protocol = {"kind": "custom"}
    if "adapter" in source:
        version = source.get("adapter_version")
        if type(version) is not int or version <= 0:
            raise ValueError("invalid adapter declaration")
        inputs = []
        for value in _items(source.get("adapter_inputs", [])):
            value = _mapping(value)
            inputs.append({
                "ref": _text(value.get("ref")),
                "provenance": _enum(value.get("provenance"), _PROVENANCE),
                "role": _enum(value.get("role"), _INPUT_ROLES),
            })
        arguments = _text(source.get("adapter_arguments"))
        # This is declared code/data, not a runtime observation. Preserve its
        # exact text, but reject malformed envelopes rather than repair them.
        _mapping(json.loads(arguments, object_pairs_hook=_object,
                            parse_constant=_no_constant))
        protocol = {
            "kind": "adapter",
            "adapter": _text(source["adapter"]),
            "adapter_version": version,
            "adapter_input_binding": _enum(
                source.get("adapter_input_binding", "explicit-v1"),
                frozenset({"explicit-v1", "fixed-v1"})),
            "adapter_arguments": arguments,
            "adapter_inputs": inputs,
        }
    identity = content_hash({**declaration, "execution": protocol})
    return {"method_id": identity, "declaration": declaration,
            "execution": protocol}


def _classification(attempt: dict) -> str:
    # These are BindingGame-owned prefixes. Never copy their suffixes or
    # classify by searching a witness, output, or arbitrary audit sentence.
    if _boolean(attempt.get("verifier_unavailable", False)):
        return "verifier_unavailable"
    reason = attempt.get("terminal_reason")
    if not isinstance(reason, str):
        return "method_rejected"
    for prefix, category in (
        ("sufficiency pre-audit:", "method_insufficient"),
        ("judger method insufficient", "method_insufficient"),
        ("judger degradation unjustified", "degradation_not_authorized"),
        ("unobtainable", "capability_unavailable"),
        ("execution failure at retry limit", "execution_failed"),
        ("refutation below decisive strength", "strength_scope_invalid"),
        ("refutation witness of kind", "witness_protocol_invalid"),
        ("circular evidence:", "authority_independence_invalid"),
    ):
        if reason.startswith(prefix):
            return category
    redos = _items(attempt.get("redos", []))
    audit = redos[-1].get("audit") if redos else attempt.get("initial_audit")
    if isinstance(audit, dict):
        if audit.get("sufficient") == "no":
            return "method_insufficient"
        if audit.get("faithful") == "no":
            return "execution_not_faithful"
    return "method_rejected"


def filter_oracle_feedback(raw: str) -> str:
    """BindingGame hook ``Callable[[str], str]``; malformed input fails closed.

    Deliberately no raw-string fallback and no observation-derived hashes.
    Unknown fields are dropped, never passed through to the compiler.
    """
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_FEEDBACK_BYTES:
        raise ValueError("invalid feedback transport")
    source = {} if raw == "(none)" else _mapping(json.loads(
        raw, object_pairs_hook=_object, parse_constant=_no_constant))
    attempts = []
    for index, value in enumerate(_items(source.get("exhausted_attempts", []))):
        value = _mapping(value)
        declaration = value.get("declared_method")
        if declaration is None:
            declaration = value.get("method", {})
        redos = _items(value.get("redos", []))
        for redo in redos:
            _mapping(redo)
        attempts.append({
            "attempt": index + 1,
            **_method(declaration),
            "classification": _classification(value),
            "redo_count": len(redos),
            "already_retried": bool(redos),
        })
    structural = []
    for value in _items(source.get("structural_rejections", [])):
        value = _mapping(value)
        procedure = _text(value.get("procedure", ""))
        matching = [attempt for attempt in attempts
                    if attempt["declaration"]["procedure"] == procedure]
        # No free-text reason, including when no attempt can be correlated.
        structural.append({
            "method_id": matching[0]["method_id"] if matching else None,
            "classification": (matching[0]["classification"] if matching
                               else "structural_rejection"),
        })
    deferred = []
    for value in _items(source.get("deferred_methods", [])):
        value = _mapping(value)
        deferred.append({
            "procedure": _text(value.get("procedure", "")),
            "strength": _enum(value.get("strength"), _STRENGTHS),
            "waiting_on": [_enum(tier, _STRENGTHS)
                           for tier in _items(value.get("waiting_on", []))],
        })
    unavailable = _items(source.get("unavailable_boundaries", []))
    context = _mapping(source.get("repair_context", {}))
    result = {
        "schema": SCHEMA,
        "repair": raw != "(none)",
        "initial_empty": _boolean(source.get("initial_empty", False)),
        "fixed_evidence_boundary": (
            _boolean(context["fixed_evidence_boundary"])
            if "fixed_evidence_boundary" in context else None),
        "unavailable_boundaries": {
            "count": len(unavailable),
            "state": "structurally_unavailable" if unavailable else "none_reported",
        },
        "structural_rejections": structural,
        "exhausted_attempts": attempts,
        "deferred_methods": deferred,
    }
    return json.dumps(result, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
