"""Configuration gates for optional verifier execution machinery."""
from __future__ import annotations

import os
from typing import Mapping, Optional


TYPED_ADAPTERS_ENV = "GV_TYPED_ADAPTERS"


def typed_adapters_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return the explicit typed-adapter policy, enabled by default.

    Only ``0`` and ``1`` are accepted.  A typo must not silently select a
    different verifier for an experiment or deployment.
    """
    values = os.environ if environ is None else environ
    if TYPED_ADAPTERS_ENV not in values:
        return True
    value = values[TYPED_ADAPTERS_ENV]
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"{TYPED_ADAPTERS_ENV} must be exactly 0 or 1")
