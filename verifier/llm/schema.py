"""
Schema mini-DSL, validation gate, and fixture backend for verifier seats.

Two pieces, both mechanical:
  - The schema mini-DSL + validate(): the output gate every seat's answer must pass. 
    Validation failures retry once with the error attached, then raise. 
    A role that cannot produce well-formed output is a FAILED generation, and per the master principle that must surface, not be papered over.
  - FixtureBackend: deterministic role-name -> payload backend for tests and replay. Used as a seam.
"""
from __future__ import annotations

from typing import Any, Dict, List


class RoleError(RuntimeError):
    pass


class SeatTimeout(RoleError):
    """A seat exhausted its time budget. Not a malformed answer that
    feedback can fix and not a transient fault: the same prompt would spend
    the same budget again, so the role's retry does not repeat the call."""


# Minimal schema validation (no external deps).
# schema: {"field": type} where type in {str, bool, int, float, list, dict},
# {"field": [element_spec]} for a list (elements may be any spec, e.g. [str]
# or [{...}]), or {"field": ("a", "b")} for a closed string enum. Mirrors
# agents.to_json_schema. The two must accept the same specs.

def optional_key(key: str) -> bool:
    """A schema key ending in "?" is OPTIONAL: validated when present,
    allowed absent (the consumer applies its own fail-closed default)."""
    return key.endswith("?")


def field_name(key: str) -> str:
    return key[:-1] if optional_key(key) else key


def validate(schema: Dict[str, Any], payload: Any, path: str = "$") -> None:
    if not isinstance(payload, dict):
        raise RoleError(f"{path}: expected object, got {type(payload).__name__}")
    for key, spec in schema.items():
        name = field_name(key)
        if name not in payload:
            if optional_key(key):
                continue
            raise RoleError(f"{path}.{name}: missing")
        _check(spec, payload[name], f"{path}.{name}")


def _check(spec: Any, value: Any, path: str) -> None:
    if isinstance(spec, tuple):
        if value not in spec:
            raise RoleError(f"{path}: {value!r} not in {spec}")
    elif isinstance(spec, list):
        if not isinstance(value, list):
            raise RoleError(f"{path}: expected list")
        for i, item in enumerate(value):
            _check(spec[0], item, f"{path}[{i}]")
    elif isinstance(spec, dict):
        validate(spec, value, path)
    else:
        if not isinstance(value, spec):
            raise RoleError(
                f"{path}: expected {spec.__name__}, "
                f"got {type(value).__name__}"
            )


# fixture backend
class FixtureBackend:
    """Deterministic backend: role name -> payload or callable(vars)->payload. Used as a seam."""

    def __init__(self, fixtures: Dict[str, Any], *, wire_output: bool = False):
        if type(wire_output) is not bool:
            raise ValueError("wire_output must be a bool")
        self.fixtures = fixtures
        # Existing fixtures replay downstream MethodIR. New wire-contract
        # tests opt in explicitly instead of rewriting persisted fixtures.
        self.wire_output = wire_output
        self.uses_role_wire_schema = wire_output
        self.calls: List[Dict[str, Any]] = []

    def complete(self, role_name: str, prompt: str, variables: Dict[str, Any]) -> Any:
        self.calls.append({"role": role_name, "variables": variables})
        if role_name not in self.fixtures:
            raise RoleError(f"no fixture for role {role_name}")
        fixture = self.fixtures[role_name]
        return fixture(variables) if callable(fixture) else fixture
