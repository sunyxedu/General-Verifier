"""The identities reuse keys are made of: a seat role's (everything that
shapes its call) and a runtime's (its declared reuse-identity fields). Two
calls may share an accepted method or a compiled contract only when these
coincide."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .runtime import (
    REUSE_IDENTITY_FIELDS, capabilities_of, runtime_identity_type,
)


def role_identity(role) -> Dict[str, Any]:
    def normalize(value):
        if isinstance(value, type):
            return f"{value.__module__}.{value.__qualname__}"
        if isinstance(value, dict):
            return {str(key): normalize(item)
                    for key, item in sorted(
                        value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    return normalize({
        "name": getattr(role, "name", ""),
        "template": getattr(role, "template", ""),
        "schema": getattr(role, "schema", {}),
        "tools": getattr(role, "tools", []),
        "model": getattr(role, "model", None),
        "max_turns": getattr(role, "max_turns", None),
        "system": getattr(role, "system", ""),
    })


def backend_identity(backend) -> Optional[Dict[str, Any]]:
    identity: Dict[str, Any] = {
        "type": runtime_identity_type(backend),
    }
    found = False
    declared = capabilities_of(backend).reuse_identity()
    for name in REUSE_IDENTITY_FIELDS:
        if name in declared:
            value = declared[name]
            if isinstance(value, dict):
                value = dict(sorted(value.items()))
            if value is None or isinstance(value, (str, int, float, bool, dict)):
                identity[name] = value
                found = True
    # FixtureBackend is a deterministic test/replay seam.  Unknown custom
    # backends without a stable semantic identity simply do not get reuse.
    if not found and type(backend).__name__ != "FixtureBackend":
        return None
    return identity
