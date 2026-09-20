"""Atomic wire contract for Oracle-owned executor selection.

The Oracle chooses a complete method and declares whether the existing
adapter router or custom executor should execute it. Lowering is structural:
it never chooses an adapter, fills arguments, or changes method semantics.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Dict

from .schema import RoleError, validate


_INPUT_SCHEMA = {
    "ref": str,
    "provenance": ("task_given", "delivered", "ledger", "verifier", "public"),
    "role": ("subject", "input", "authority"),
}

_METHOD_FIELDS = {
    "procedure": str,
    "artifact": str,
    "level": str,
    "basis": ("world", "record"),
    "strength": ("decisive", "bounded", "analytic"),
    "bridges": [str],
}

# EXECUTABLE FORM: a custom method may carry the complete script the
# executor runs verbatim (no agent seat; the script judger reads the
# retained output). Optional: the agent executor carries out any method
# without one. An adapter-bound method already names its executor, so it
# never carries a script.
_METHOD_OPTIONAL_FIELDS = {
    "script": str,
}

_EXECUTION_SCHEMA = {
    "kind": ("adapter", "custom"),
    "custom_gap_reason": str,
    "adapter": str,
    "adapter_version": int,
    "adapter_input_binding": ("none", "explicit-v1", "fixed-v1"),
    "adapter_arguments": str,
    "adapter_inputs": [_INPUT_SCHEMA],
}


# Model-facing contract. Every execution field is required so a partial
# adapter binding cannot silently become a custom execution request.
ORACLE_EXECUTION_WIRE_SCHEMA: Dict[str, Any] = {
    "methods": [{
        **_METHOD_FIELDS,
        **{f"{key}?": kind for key, kind in _METHOD_OPTIONAL_FIELDS.items()},
        "execution": _EXECUTION_SCHEMA,
    }],
}


_TOP_KEYS = frozenset({"methods"})
_METHOD_KEYS = frozenset({*_METHOD_FIELDS, "execution"})
_METHOD_OPTIONAL_KEYS = frozenset(_METHOD_OPTIONAL_FIELDS)
_EXECUTION_KEYS = frozenset(_EXECUTION_SCHEMA)
_INPUT_KEYS = frozenset(_INPUT_SCHEMA)


def _require_exact_keys(
    value: Dict[str, Any], expected: frozenset[str], path: str,
    optional: frozenset[str] = frozenset(),
) -> None:
    actual = frozenset(value)
    if expected <= actual <= expected | optional:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected - optional)
    details = []
    if missing:
        details.append(f"missing={missing!r}")
    if extra:
        details.append(f"extra={extra!r}")
    raise RoleError(f"{path}: keys must be exact ({', '.join(details)})")


def _strict_argument_object(raw: str, path: str) -> None:
    def no_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=no_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RoleError(f"{path}: must encode strict JSON: {error}") from error
    if type(value) is not dict:
        raise RoleError(f"{path}: must encode an object")


def lower_oracle_execution_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and lower one complete Oracle wire payload.

    Adapter envelopes map one-to-one to the established flat MethodIR.
    Custom envelopes lose only their diagnostic gap reason and execution
    sentinels. The input object is never mutated.
    """
    validate(ORACLE_EXECUTION_WIRE_SCHEMA, payload)
    _require_exact_keys(payload, _TOP_KEYS, "$")

    methods = []
    for index, wire_method in enumerate(payload["methods"]):
        method_path = f"$.methods[{index}]"
        _require_exact_keys(wire_method, _METHOD_KEYS, method_path,
                            optional=_METHOD_OPTIONAL_KEYS)
        execution = wire_method["execution"]
        _require_exact_keys(
            execution, _EXECUTION_KEYS, f"{method_path}.execution",
        )
        for input_index, item in enumerate(execution["adapter_inputs"]):
            _require_exact_keys(
                item,
                _INPUT_KEYS,
                f"{method_path}.execution.adapter_inputs[{input_index}]",
            )

        kind = execution["kind"]
        version = execution["adapter_version"]
        if type(version) is not int:
            raise RoleError(
                f"{method_path}.execution.adapter_version: expected integer",
            )

        lowered = {
            key: deepcopy(wire_method[key]) for key in _METHOD_FIELDS
        }
        script = str(wire_method.get("script") or "")
        if kind == "adapter":
            if script.strip():
                raise RoleError(
                    f"{method_path}.script: an adapter-bound method names "
                    "its executor; the executable form is for custom "
                    "execution only",
                )
            if execution["custom_gap_reason"] != "":
                raise RoleError(
                    f"{method_path}.execution.custom_gap_reason: adapter "
                    "execution requires the empty sentinel",
                )
            if not execution["adapter"].strip():
                raise RoleError(
                    f"{method_path}.execution.adapter: adapter execution "
                    "requires a non-empty name",
                )
            if version <= 0:
                raise RoleError(
                    f"{method_path}.execution.adapter_version: adapter "
                    "execution requires a positive version",
                )
            binding = execution["adapter_input_binding"]
            if binding not in ("explicit-v1", "fixed-v1"):
                raise RoleError(
                    f"{method_path}.execution.adapter_input_binding: adapter "
                    "execution requires explicit-v1 or fixed-v1",
                )
            _strict_argument_object(
                execution["adapter_arguments"],
                f"{method_path}.execution.adapter_arguments",
            )
            lowered.update({
                "adapter": execution["adapter"],
                "adapter_version": version,
                "adapter_arguments": execution["adapter_arguments"],
                "adapter_inputs": deepcopy(execution["adapter_inputs"]),
            })
            if binding == "fixed-v1":
                lowered["adapter_input_binding"] = "fixed-v1"
        else:
            if not execution["custom_gap_reason"].strip():
                raise RoleError(
                    f"{method_path}.execution.custom_gap_reason: custom "
                    "execution requires a specific non-empty gap reason",
                )
            if (
                execution["adapter"] != ""
                or version != 0
                or execution["adapter_input_binding"] != "none"
                or execution["adapter_arguments"] != ""
                or execution["adapter_inputs"] != []
            ):
                raise RoleError(
                    f"{method_path}.execution: custom execution requires "
                    "empty adapter sentinels",
                )
            if script.strip():
                lowered["script"] = script
        methods.append(lowered)

    return {"methods": methods}


def raise_oracle_execution_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Lift one established flat MethodIR payload to the wire envelope.

    This is the exact inverse of :func:`lower_oracle_execution_payload` for
    complete adapter bindings and for custom methods.  It exists for fixtures
    and process stubs that emulate a model: production never lifts model
    output, and a partial adapter binding is rejected rather than repaired.
    """
    methods = []
    for method in payload["methods"]:
        lifted = {key: deepcopy(method[key]) for key in _METHOD_FIELDS}
        if str(method.get("script") or "").strip():
            lifted["script"] = method["script"]
        adapter_keys = {
            "adapter", "adapter_version", "adapter_arguments", "adapter_inputs",
        }
        present = adapter_keys & set(method)
        if present and present != adapter_keys:
            raise ValueError(
                "cannot lift a partial adapter binding: missing "
                f"{sorted(adapter_keys - present)!r}")
        if present:
            lifted["execution"] = {
                "kind": "adapter",
                "custom_gap_reason": "",
                "adapter": method["adapter"],
                "adapter_version": method["adapter_version"],
                "adapter_input_binding": (
                    "fixed-v1"
                    if method.get("adapter_input_binding") == "fixed-v1"
                    else "explicit-v1"),
                "adapter_arguments": method["adapter_arguments"],
                "adapter_inputs": deepcopy(method["adapter_inputs"]),
            }
        else:
            lifted["execution"] = {
                "kind": "custom",
                "custom_gap_reason": (
                    "No registered adapter executes this procedure."),
                "adapter": "",
                "adapter_version": 0,
                "adapter_input_binding": "none",
                "adapter_arguments": "",
                "adapter_inputs": [],
            }
        methods.append(lifted)
    return {"methods": methods}
