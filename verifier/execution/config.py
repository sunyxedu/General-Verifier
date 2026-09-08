"""Configuration gates for optional verifier execution machinery."""
from __future__ import annotations

import json
import os
import stat
from typing import Mapping, Optional


TYPED_ADAPTERS_ENV = "GV_TYPED_ADAPTERS"
TYPED_JUDGER_PACKET_ENV = "GV_TYPED_JUDGER_PACKET"
CONTRACT_CACHE_ENV = "GV_CONTRACT_CACHE"
APPEND_ONLY_DELTAS_ENV = "GV_APPEND_ONLY_DELTAS"
APPEND_ONLY_DELTA_PROTOCOL = "obligations-array-delta-v1"
OFFLINE_RUNTIME_READ_ROOTS_ENV = "GV_OFFLINE_RUNTIME_READ_ROOTS"

_MAX_OFFLINE_RUNTIME_READ_ROOTS = 64
_MAX_OFFLINE_RUNTIME_ROOT_LENGTH = 4096


def _paths_overlap(left: str, right: str) -> bool:
    try:
        common = os.path.commonpath((left, right))
    except ValueError:
        return False
    return common in (left, right)


def offline_runtime_read_roots(
    environ: Optional[Mapping[str, str]] = None,
    *,
    home: Optional[str] = None,
) -> tuple[str, ...]:
    """Return operator-authorized public runtime trees for offline seats.

    The value is a JSON array of existing canonical absolute directories.
    It is deliberately not inferred from candidate variables or ``PATH``.
    User-home trees are excluded because they can contain credentials,
    historical results, and other candidate data rather than public runtime
    material.
    """
    values = os.environ if environ is None else environ
    raw = values.get(OFFLINE_RUNTIME_READ_ROOTS_ENV, "[]")
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            f"{OFFLINE_RUNTIME_READ_ROOTS_ENV} must be a JSON array") from error
    if not isinstance(parsed, list):
        raise ValueError(
            f"{OFFLINE_RUNTIME_READ_ROOTS_ENV} must be a JSON array")
    if len(parsed) > _MAX_OFFLINE_RUNTIME_READ_ROOTS:
        raise ValueError(
            f"{OFFLINE_RUNTIME_READ_ROOTS_ENV} has too many roots")

    home_path = os.path.realpath(
        os.path.abspath(os.path.expanduser("~") if home is None else home))
    roots = []
    seen = set()
    for index, value in enumerate(parsed):
        label = f"{OFFLINE_RUNTIME_READ_ROOTS_ENV}[{index}]"
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty string")
        if len(value) > _MAX_OFFLINE_RUNTIME_ROOT_LENGTH or "\x00" in value:
            raise ValueError(f"{label} is not a valid bounded path")
        if not os.path.isabs(value):
            raise ValueError(f"{label} must be absolute")
        canonical = os.path.realpath(value)
        if value != canonical:
            raise ValueError(f"{label} must be canonical and contain no symlinks")
        if canonical == os.path.sep:
            raise ValueError(f"{label} must not grant the filesystem root")
        if _paths_overlap(canonical, home_path):
            raise ValueError(f"{label} must not overlap the user home")
        try:
            info = os.lstat(canonical)
        except OSError as error:
            raise ValueError(f"{label} must be an existing directory") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{label} must be an existing real directory")
        if canonical in seen:
            raise ValueError(f"{label} duplicates an earlier root")
        seen.add(canonical)
        roots.append(canonical)
    return tuple(roots)


def contract_cache_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Experimental authority-only compilation; opt in until evaluated."""
    values = os.environ if environ is None else environ
    value = values.get(CONTRACT_CACHE_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{CONTRACT_CACHE_ENV} must be exactly 0 or 1")
    return value == "1"


def append_only_deltas_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether exact Critic append repairs use the additions-only protocol.

    This remains an explicit experiment until its result and cost gates have
    been exercised.  A misspelling must not silently change Compiler/Critic
    semantics or share an identity with the default full-replacement path.
    """
    values = os.environ if environ is None else environ
    value = values.get(APPEND_ONLY_DELTAS_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{APPEND_ONLY_DELTAS_ENV} must be exactly 0 or 1")
    return value == "1"


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


def typed_judger_packet_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Experimental typed-adapter evidence transport; strictly opt in.

    This flag changes only how a fresh independent Judger receives an
    already-produced typed-adapter receipt.  It never authorizes method
    reuse, an adapter, or a verdict.  Keep the default off until the
    restricted Judger transport and no-loss evaluation are complete.
    """
    values = os.environ if environ is None else environ
    value = values.get(TYPED_JUDGER_PACKET_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(
            f"{TYPED_JUDGER_PACKET_ENV} must be exactly 0 or 1")
    return value == "1"
