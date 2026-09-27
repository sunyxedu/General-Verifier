"""The verifier's process settings: every ``GV_*`` variable the verifier
itself reads is parsed here, and nowhere else.

Two kinds of setting, told apart by WHEN they are read. Construction-time
settings are grouped into frozen objects (``ExtensionConfig``,
``OnlineConfig``, ``ParallelismConfig``, together ``VerifierConfig``) that
the composition root loads at the moment it assembles a kernel or a
server, so a test or a harness that sets the environment first gets what
it set. Call-time settings (resource limits the execution layer consults
on each launch, the server's lock wait) are small readers the low-level
module calls when it needs the value, exactly as often as before.

Experiment switches are strict: only ``0`` and ``1`` are accepted, so a
typo can never silently select a different verifier for a run. Online
economies are off-switches (anything but ``0`` keeps them on) and numeric
limits fall back to their default on an unparseable value.

Provider and direct-seat settings (``GV_PROVIDER``, ``GV_MODEL``,
``GV_DIRECT_*`` and the provider credentials) may also come from the
repository's shipped ``.env``: the real environment wins, then ``.env``,
then the default. They are parsed here into ``ProviderSettings`` and
``DirectSettings``; which endpoint those select is decided by the one
provider preset table (``llm/endpoints/presets.py``).

This module imports nothing from the verifier (only the standard library),
so any layer may ask it for a setting without an import cycle, and a
harness running under its own Python can load it by path.
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Mapping, Optional, Set, Tuple


TYPED_ADAPTERS_ENV = "GV_TYPED_ADAPTERS"
TYPED_JUDGER_PACKET_ENV = "GV_TYPED_JUDGER_PACKET"
CONTRACT_CACHE_ENV = "GV_CONTRACT_CACHE"
METHOD_CACHE_ENV = "GV_METHOD_CACHE"
APPEND_ONLY_DELTAS_ENV = "GV_APPEND_ONLY_DELTAS"
SUFFICIENCY_PREAUDIT_ENV = "GV_SUFFICIENCY_PREAUDIT"
FINAL_CONTRACT_PRECOMPILE_ENV = "GV_FINAL_CONTRACT_PRECOMPILE"
APPEND_ONLY_DELTA_PROTOCOL = "obligations-array-delta-v1"
OFFLINE_RUNTIME_READ_ROOTS_ENV = "GV_OFFLINE_RUNTIME_READ_ROOTS"

BACKGROUND_CHECKS_ENV = "GV_BACKGROUND_CHECKS"
FINAL_PRECOMPILE_ENV = "GV_FINAL_PRECOMPILE"
SCREENER_ENV = "GV_SCREENER"
CRITIC_ROUNDS_NONFINAL_ENV = "GV_CRITIC_ROUNDS_NONFINAL"
UNIFY_FINAL_ENV = "GV_UNIFY_FINAL"
SPECULATIVE_DISCHARGE_ENV = "GV_SPECULATIVE_DISCHARGE"

PARALLEL_OBLIGATIONS_ENV = "GV_PARALLEL_OBLIGATIONS"
PARALLEL_CLAIMS_ENV = "GV_PARALLEL_CLAIMS"
SHARED_DIR_ENV = "GV_SHARED_DIR"
CERTIFICATE_SCOPE_ENV = "GV_CERTIFICATE_SCOPE"

MAX_CONCURRENT_SEATS_ENV = "GV_MAX_CONCURRENT_SEATS"
SEAT_ROOM_MB_ENV = "GV_SEAT_ROOM_MB"
SCRATCH_CAP_GB_ENV = "GV_SCRATCH_CAP_GB"
SCRIPT_TIMEOUT_ENV = "GV_SCRIPT_TIMEOUT"
SERVER_LOCK_WAIT_ENV = "GV_SERVER_LOCK_WAIT"

BROWSER_OBSERVER_ENV = "GV_BROWSER_OBSERVER"
BROWSER_OBSERVER_SHA256_ENV = "GV_BROWSER_OBSERVER_SHA256"
BROWSER_EXECUTABLE_ENV = "GV_BROWSER_EXECUTABLE"
BROWSER_EXECUTABLE_SHA256_ENV = "GV_BROWSER_EXECUTABLE_SHA256"

PARALLEL_OBLIGATIONS_DEFAULT = 8
PARALLEL_CLAIMS_DEFAULT = 4
MAX_CONCURRENT_SEATS_DEFAULT = 12
# one seat process with what it runs, plus slack for the solver's next step
SEAT_ROOM_MB_DEFAULT = 512
SCRATCH_CAP_GB_DEFAULT = 6.0
SCRIPT_TIMEOUT_DEFAULT = 300   # the oracle is told "a few minutes"
# seconds a starting server waits for the state lock a previous server of
# the same client may still hold while it exits
SERVER_LOCK_WAIT_DEFAULT = 30.0

# a setting whose name carries one of these is a credential: its value is
# never recorded in a launch identity
SECRET_MARKS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")

_MAX_OFFLINE_RUNTIME_READ_ROOTS = 64
_MAX_OFFLINE_RUNTIME_ROOT_LENGTH = 4096


def _values(environ: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _binary(values: Mapping[str, str], name: str, default: bool,
            requirement: str) -> bool:
    """A strict switch: unset is the default, otherwise exactly 0 or 1."""
    if name not in values:
        return default
    value = values[name]
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"{name} {requirement}")


def _on_unless_zero(values: Mapping[str, str], name: str) -> bool:
    return values.get(name, "1") != "0"


def _positive_int(values: Mapping[str, str], name: str, fallback: int) -> int:
    try:
        return max(1, int(values.get(name, fallback)))
    except ValueError:
        return fallback


# strict experiment switches

def contract_cache_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Parse the process-local contract-cache flag.

    Off by default.  When on, the default kernel shares one Critic-approved,
    candidate-blind final contract per task authority across the candidates
    scored in this process; the Compiler and Critic still run once per task
    and every candidate still executes and audits its own evidence.
    """
    return _binary(_values(environ), CONTRACT_CACHE_ENV, False,
                   "must be exactly 0 or 1")


def method_cache_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Parse the process-local accepted-method cache flag.

    Off by default.  When on, the default kernel reuses Judger-accepted
    typed methods for identical obligations of the same task authority in
    the direction the Judger accepted; the adapter still executes afresh for
    every candidate and unaccepted directions still get a fresh audit.
    """
    return _binary(_values(environ), METHOD_CACHE_ENV, False,
                   "must be exactly 0 or 1")


def append_only_deltas_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether exact Critic append repairs use the additions-only protocol.

    This remains an explicit experiment until its result and cost gates have
    been exercised.  A misspelling must not silently change Compiler/Critic
    semantics or share an identity with the default full-replacement path.
    """
    return _binary(_values(environ), APPEND_ONLY_DELTAS_ENV, False,
                   "must be exactly 0 or 1")


def typed_adapters_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return the explicit typed-adapter policy, enabled by default.

    Only ``0`` and ``1`` are accepted.  A typo must not silently select a
    different verifier for an experiment or deployment.
    """
    return _binary(_values(environ), TYPED_ADAPTERS_ENV, True,
                   "must be exactly 0 or 1")


def typed_judger_packet_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Experimental typed-adapter evidence transport; strictly opt in.

    This flag changes only how a fresh independent Judger receives an
    already-produced typed-adapter receipt.  It never authorizes method
    reuse, an adapter, or a verdict.  Keep the default off until the
    restricted Judger transport and no-loss evaluation are complete.
    """
    return _binary(_values(environ), TYPED_JUDGER_PACKET_ENV, False,
                   "must be exactly 0 or 1")


def sufficiency_preaudit_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether a proposed agent method is audited for sufficiency before it
    runs.  Explicit ``0``/``1`` only; disabled unless requested."""
    return _binary(_values(environ), SUFFICIENCY_PREAUDIT_ENV, False,
                   "must be 0 or 1")


def final_contract_precompile_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether offline scoring compiles the candidate-blind final contract
    concurrently with extraction.  A schedule, not a judgment: the kernel
    adopts the result only under an identical contract key.  Explicit
    ``0``/``1`` only; disabled unless requested.  Read only where that
    schedule can apply (Codex seats, the built-in kernel), so it is not part
    of ``ExtensionConfig``."""
    return _binary(_values(environ), FINAL_CONTRACT_PRECOMPILE_ENV, False,
                   "must be 0 or 1")


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
    values = _values(environ)
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


def browser_observer_settings(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[Tuple[str, str, str, str]]:
    """The explicitly configured browser observer as (observer path, its
    digest, browser path, its digest), or None when none is configured.
    Never auto-discovered; a partial configuration is an error."""
    values = _values(environ)
    configured = (values.get(BROWSER_OBSERVER_ENV),
                  values.get(BROWSER_OBSERVER_SHA256_ENV),
                  values.get(BROWSER_EXECUTABLE_ENV),
                  values.get(BROWSER_EXECUTABLE_SHA256_ENV))
    if all(item is None for item in configured):
        return None
    if not all(configured):
        raise ValueError(
            "GV_BROWSER_OBSERVER, GV_BROWSER_OBSERVER_SHA256, "
            "GV_BROWSER_EXECUTABLE, and GV_BROWSER_EXECUTABLE_SHA256 must "
            "be set together")
    return configured


# construction-time groups

@dataclass(frozen=True)
class ExtensionConfig:
    """The strict opt-in (or, for typed adapters, opt-out) experiments that
    change what the verifier runs; each one is part of the offline config
    identity."""
    typed_adapters: bool
    sufficiency_preaudit: bool
    typed_judger_packet: bool
    contract_cache: bool
    method_cache: bool
    append_only_deltas: bool

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None,
                 ) -> "ExtensionConfig":
        values = _values(environ)
        return cls(
            typed_adapters=typed_adapters_enabled(values),
            sufficiency_preaudit=sufficiency_preaudit_enabled(values),
            typed_judger_packet=typed_judger_packet_enabled(values),
            contract_cache=contract_cache_enabled(values),
            method_cache=method_cache_enabled(values),
            append_only_deltas=append_only_deltas_enabled(values),
        )

    def require_consistent(self) -> None:
        """The typed Judger packet carries a typed adapter's receipt, so it
        cannot run without typed adapters."""
        if self.typed_judger_packet and not self.typed_adapters:
            raise ValueError(
                "typed-adapter Judger packet requires typed adapters")


@dataclass(frozen=True)
class OnlineConfig:
    """The online economies, each on unless set to ``0``; they apply only
    with a live world (offline scoring never reads them as on)."""
    background_checks: bool       # intermediate commits return before checks
    final_precompile: bool        # the final's contract compiled meanwhile
    screener: bool                # admission screener on commits
    critic_rounds_nonfinal: int   # advisory critic budget for non-finals
    unify_final: bool             # the final takes the path of any claim
    speculative_discharge: bool   # checks overlap the critic's review

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None,
                 ) -> "OnlineConfig":
        values = _values(environ)
        try:
            nonfinal = max(0, int(values.get(CRITIC_ROUNDS_NONFINAL_ENV, "0")))
        except ValueError:
            nonfinal = 0
        return cls(
            background_checks=_on_unless_zero(values, BACKGROUND_CHECKS_ENV),
            final_precompile=_on_unless_zero(values, FINAL_PRECOMPILE_ENV),
            screener=_on_unless_zero(values, SCREENER_ENV),
            critic_rounds_nonfinal=nonfinal,
            unify_final=_on_unless_zero(values, UNIFY_FINAL_ENV),
            speculative_discharge=_on_unless_zero(
                values, SPECULATIVE_DISCHARGE_ENV),
        )


@dataclass(frozen=True)
class ParallelismConfig:
    """Execution strategy: it changes the schedule, never the judgment, so
    none of it enters the offline config identity. ``shared_dir`` and
    ``certificate_scope`` are None when unset; the caller owns their
    defaults."""
    parallel_obligations: int
    parallel_claims: int
    shared_dir: Optional[str]
    certificate_scope: Optional[str]

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None,
                 ) -> "ParallelismConfig":
        values = _values(environ)
        return cls(
            parallel_obligations=_positive_int(
                values, PARALLEL_OBLIGATIONS_ENV, PARALLEL_OBLIGATIONS_DEFAULT),
            parallel_claims=_positive_int(
                values, PARALLEL_CLAIMS_ENV, PARALLEL_CLAIMS_DEFAULT),
            shared_dir=values.get(SHARED_DIR_ENV) or None,
            certificate_scope=values.get(CERTIFICATE_SCOPE_ENV) or None,
        )


@dataclass(frozen=True)
class VerifierConfig:
    """Everything the composition root reads when it assembles a kernel."""
    extensions: ExtensionConfig
    online: OnlineConfig
    parallelism: ParallelismConfig

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None,
                 ) -> "VerifierConfig":
        values = _values(environ)
        return cls(extensions=ExtensionConfig.from_env(values),
                   online=OnlineConfig.from_env(values),
                   parallelism=ParallelismConfig.from_env(values))


def launch_settings(environ: Optional[Mapping[str, str]] = None,
                    ) -> Dict[str, str]:
    """The verifier's own settings in the environment, sorted, for a
    server's launch identity: every ``GV_*`` variable and ``CODEX_HOME``,
    never a credential's value."""
    env = {k: v for k, v in _values(environ).items()
           if (k.startswith("GV_") or k == "CODEX_HOME")
           and not any(m in k.upper() for m in SECRET_MARKS)}
    return dict(sorted(env.items()))


# call-time readers: consulted by the execution layer on each use

def max_concurrent_seats() -> int:
    """Process-wide cap on in-flight seat calls (<= 0 disables the cap)."""
    try:
        return int(os.environ.get(MAX_CONCURRENT_SEATS_ENV,
                                  str(MAX_CONCURRENT_SEATS_DEFAULT)))
    except ValueError:
        return MAX_CONCURRENT_SEATS_DEFAULT


def seat_room_mb() -> int:
    """Memory a new seat process needs before it launches (<= 0 disables
    the wait)."""
    try:
        return int(os.environ.get(SEAT_ROOM_MB_ENV, SEAT_ROOM_MB_DEFAULT))
    except ValueError:
        return SEAT_ROOM_MB_DEFAULT


def scratch_cap_gb() -> float:
    """A running check's scratch-directory budget (0 disables it)."""
    try:
        return float(os.environ.get(SCRATCH_CAP_GB_ENV,
                                    SCRATCH_CAP_GB_DEFAULT))
    except ValueError:
        return SCRATCH_CAP_GB_DEFAULT


def script_timeout_seconds() -> float:
    """Wall-clock bound on one verifier-run check script."""
    try:
        return float(os.environ.get(SCRIPT_TIMEOUT_ENV,
                                    SCRIPT_TIMEOUT_DEFAULT))
    except ValueError:
        return float(SCRIPT_TIMEOUT_DEFAULT)


def server_lock_wait_seconds() -> float:
    """How long a starting server waits for its state lock. Unlike the
    limits above, an unparseable value is an error at startup."""
    return float(os.environ.get(SERVER_LOCK_WAIT_ENV,
                                SERVER_LOCK_WAIT_DEFAULT))


# provider and direct-seat settings: environment, then the shipped .env

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOTENV_PATH = os.path.join(ROOT, ".env")

PROVIDER_ENV = "GV_PROVIDER"
MODEL_ENV = "GV_MODEL"
TEXT_ONLY_ENV = "GV_TEXT_ONLY"
CLAUDE_CONFIG_DIR_ENV = "GV_CLAUDE_CONFIG_DIR"
PROVIDER_DEFAULT = "deepseek"
# the credentials and endpoint overrides the provider presets may name
PROVIDER_VALUE_ENVS = (
    "CLAUDE_CODE_OAUTH_TOKEN", "GV_API_KEY", "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "GV_ANTHROPIC_BASE_URL",
    "GV_LITELLM_MODEL",
)

DIRECT_SEATS_ENV = "GV_DIRECT_SEATS"
DIRECT_PROVIDER_ENV = "GV_DIRECT_PROVIDER"     # anthropic | openai | openrouter
DIRECT_MODEL_ENV = "GV_DIRECT_MODEL"
DIRECT_EFFORT_ENV = "GV_DIRECT_EFFORT"
DIRECT_VALUE_ENVS = (
    "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL",
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
)


def _load_dotenv(path: str) -> Dict[str, str]:
    """Minimal .env parser: KEY=VALUE lines, #-comments, optional quotes."""
    conf: Dict[str, str] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip("'\"")
                conf[k.strip()] = v
    except OSError:
        pass
    return conf


_DOTENV = _load_dotenv(DOTENV_PATH)


def cfg(name: str, default: Optional[str] = None) -> Optional[str]:
    """Real environment > .env file > baked-in default (an empty value
    counts as unset)."""
    v = os.environ.get(name)
    if v is None:
        v = _DOTENV.get(name)
    return v if v not in (None, "") else default


Lookup = Callable[[str], Optional[str]]


def _lookup(environ: Optional[Mapping[str, str]]) -> Lookup:
    """``cfg`` for the process; an injected mapping stays self-contained
    (tests and callers that supply a complete environment), with an empty
    value read as unset like ``cfg`` does."""
    if environ is None:
        return cfg
    return lambda name: environ.get(name) or None


@dataclass(frozen=True)
class ProviderSettings:
    """The agent provider every Claude Code seat runs on. ``values`` holds
    the credentials and endpoint overrides by variable name; the provider
    preset says which of them the chosen provider reads."""
    provider: str
    model: Optional[str]
    text_only: bool
    claude_config_dir: Optional[str]
    # the inherited CLAUDE_CONFIG_DIR: the real environment only, never .env
    inherited_config_dir: Optional[str]
    values: Mapping[str, Optional[str]]

    def value(self, name: Optional[str]) -> Optional[str]:
        return self.values.get(name) if name else None

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None,
                 ) -> "ProviderSettings":
        setting = _lookup(environ)
        inherited = (os.environ if environ is None else environ).get(
            "CLAUDE_CONFIG_DIR")
        return cls(
            provider=(setting(PROVIDER_ENV) or PROVIDER_DEFAULT).lower(),
            model=setting(MODEL_ENV),
            # a text-only model cannot accept image content blocks
            text_only=(setting(TEXT_ONLY_ENV) or "") == "1",
            claude_config_dir=setting(CLAUDE_CONFIG_DIR_ENV),
            inherited_config_dir=inherited,
            values={name: setting(name) for name in PROVIDER_VALUE_ENVS},
        )


@dataclass(frozen=True)
class DirectSettings:
    """Which roles run as direct model calls and on which provider.
    ``seats`` is the lowercased GV_DIRECT_SEATS (``default`` when unset),
    ``provider`` the lowercased GV_DIRECT_PROVIDER ("" when unset)."""
    seats: str
    provider: str
    model: Optional[str]
    effort: Optional[str]
    values: Mapping[str, Optional[str]]

    def value(self, name: Optional[str]) -> Optional[str]:
        return self.values.get(name) if name else None

    def seat_roles(self, default_roles: FrozenSet[str]) -> Set[str]:
        """GV_DIRECT_SEATS unset or ``default`` → the tool-free set;
        ``0``/``off`` → none; otherwise a comma-separated list of roles."""
        raw = self.seats
        if raw in ("0", "off", "none", ""):
            return set()
        if raw in ("1", "default", "on"):
            return set(default_roles)
        return {part.strip() for part in raw.split(",") if part.strip()}

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None,
                 ) -> "DirectSettings":
        setting = _lookup(environ)
        return cls(
            seats=(setting(DIRECT_SEATS_ENV) or "default").strip().lower(),
            provider=(setting(DIRECT_PROVIDER_ENV) or "").strip().lower(),
            model=setting(DIRECT_MODEL_ENV),
            effort=setting(DIRECT_EFFORT_ENV),
            values={name: setting(name) for name in DIRECT_VALUE_ENVS},
        )


# run options: fixed for a run by its configuration, read by the engine

@dataclass(frozen=True)
class RunOptions:
    """Evidence-presentation options of one run (both explicit opt-in
    experiments of the Codex seats, both part of the offline config
    identity). The seat runtime renders their instructions; the kernel, the
    binding game and the executor bind the evidence views they need."""
    compact_inspection: bool = False
    judger_prefetch: bool = False

    def __post_init__(self):
        if type(self.compact_inspection) is not bool:
            raise ValueError("compact_inspection must be a bool")
        if type(self.judger_prefetch) is not bool:
            raise ValueError("judger_prefetch must be a bool")
