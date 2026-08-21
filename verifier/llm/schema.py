"""
Schema mini-DSL, validation gate, and fixture backend for verifier seats.

Two pieces, both mechanical:
  - the schema mini-DSL + validate(): the output gate every seat's answer
    must pass. Validation failures retry once with the error attached, then
    raise — a role that cannot produce well-formed output is a FAILED
    generation, and per the master principle that must surface, not be
    papered over.
  - FixtureBackend: deterministic role-name -> payload backend for tests
    and replay.
"""
from __future__ import annotations

from typing import Any, Dict, List


class RoleError(RuntimeError):
    pass


# ---- minimal schema validation (no external deps) ----------------------------
# schema: {"field": type} where type in {str, bool, int, float, list, dict},
# {"field": [element_spec]} for a list (elements may be any spec, e.g. [str]
# or [{...}]), or {"field": ("a", "b")} for a closed string enum. Mirrors
# agents.to_json_schema — the two must accept the same specs.

def validate(schema: Dict[str, Any], payload: Any, path: str = "$") -> None:
    if not isinstance(payload, dict):
        raise RoleError(f"{path}: expected object, got {type(payload).__name__}")
    for key, spec in schema.items():
        if key not in payload:
            raise RoleError(f"{path}.{key}: missing")
        _check(spec, payload[key], f"{path}.{key}")


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

    def __init__(self, fixtures: Dict[str, Any]):
        self.fixtures = fixtures
        self.calls: List[Dict[str, Any]] = []

    def complete(self, role_name: str, prompt: str, variables: Dict[str, Any]) -> Any:
        self.calls.append({"role": role_name, "variables": variables})
        if role_name not in self.fixtures:
            raise RoleError(f"no fixture for role {role_name}")
        fixture = self.fixtures[role_name]
        return fixture(variables) if callable(fixture) else fixture
