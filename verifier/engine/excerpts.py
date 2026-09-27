"""Bounded renderings of recorded values for the seats' context."""
from __future__ import annotations

import hashlib
import json
from typing import Dict


def bounded_context_value(value, limit: int) -> Dict:
    """Preserve a small value exactly; otherwise expose an auditable
    excerpt without pretending it is complete."""
    if isinstance(value, str):
        rendered = value
    else:
        rendered = json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), default=str)
    if len(rendered) <= limit:
        return {"value": value, "truncated": False}
    return {
        "excerpt": rendered[:limit],
        "truncated": True,
        "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "chars": len(rendered),
    }
