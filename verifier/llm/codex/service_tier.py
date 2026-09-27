"""Explicit, reproducible Codex service-tier configuration."""
from __future__ import annotations

from typing import Optional


DEFAULT_SERVICE_TIER = "standard"
SERVICE_TIERS = ("standard", "fast")


def normalize_service_tier(service_tier: Optional[str]) -> str:
    """Resolve the user-facing Codex service tier, failing closed."""
    resolved = service_tier or DEFAULT_SERVICE_TIER
    if resolved not in SERVICE_TIERS:
        raise ValueError(
            f"invalid Codex service tier {resolved!r}; "
            f"expected one of {SERVICE_TIERS}")
    return resolved


def service_tier_cli_args(service_tier: Optional[str]) -> list[str]:
    """Return explicit Codex CLI overrides for standard or Fast mode.

    Both branches override user config.  A nominally standard experiment must
    not inherit ``fast_mode`` from ``~/.codex/config.toml`` and silently join a
    different latency/credit tier.
    """
    resolved = normalize_service_tier(service_tier)
    if resolved == "fast":
        return ["-c", 'service_tier="fast"', "--enable", "fast_mode"]
    return ["-c", 'service_tier="default"', "--disable", "fast_mode"]
