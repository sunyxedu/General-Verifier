"""Candidate-independent, tool-free Compiler/Critic sessions.

Nothing here replaces the Oracle or the independent Judger. Unknown backends
do not qualify. Codex additionally uses an empty execution directory, explicit
capability restrictions, and an allowlisted JSON event stream: unexpected tool
activity invalidates the generation instead of entering a shared contract.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

from .agents import AgentBackend, AgentRole
from .codex_backend import (
    CodexBackend,
    _completed_response_without_usage,
    _native_jsonl_lines,
)
from .schema import FixtureBackend, RoleError


AUTHORITY_BOUNDARY_VERSION = "gv-authority-no-tools-v1"
_FIXTURE_SCOPES = weakref.WeakKeyDictionary()
_FIXTURE_SCOPES_LOCK = threading.Lock()
AUTHORITY_ONLY_NOTE = (
    "\nAUTHORITY-ONLY COMPILATION BOUNDARY: Only the inlined task and pre-task "
    "baseline metadata are available. You have no tools, candidate answer, "
    "candidate files, solver ledger, or graph. Compile the complete task "
    "contract for an arbitrary current candidate. Preserve every quantifier, "
    "modality, source-relative term, process constraint, and output binding. "
    "Use task-named logical paths and 'the current candidate final response' "
    "where the task calls for an answer rather than a file; do not invent its "
    "value. Candidate-dependent references are parameters for the later "
    "Oracle, not facts available here. An unavailable referenced authority "
    "must remain source-relative, never be guessed or silently weakened. "
    "Resolve relative dates only from authoritative timestamps; the runtime "
    "clock is not task authority. Preserve unanchored time references for "
    "the later Oracle instead of substituting the compilation time. "
    "Do not attempt to read local files, use tools, or search the web."
)

_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "shell_snapshot", "apps", "plugins",
    "remote_plugin", "multi_agent", "memories", "browser_use",
    "browser_use_external", "browser_use_full_cdp_access", "computer_use",
    "in_app_browser", "workspace_dependencies", "code_mode_host", "hooks",
    "skill_search", "skill_mcp_dependency_install", "image_generation",
    "view_image", "sleep_tool", "goals",
)
# Codex 0.153.4 emits this capability-disabled notice before turn.started,
# even with code_mode=false. It contains no tool result or candidate data.
# Unknown errors and any tool activity still reject the entire generation.
_DISABLED_CODE_MODE_NOTICE = (
    "Code Mode is unavailable because code-mode host is disabled. Code mode "
    "will fail closed; enable `features.code_mode_host` and install "
    "`codex-code-mode-host`."
)


class AuthorityBackendUnavailable(RoleError):
    """The backend cannot provide the required authority-only boundary."""


def authority_policy_args(scratch_dir: str) -> list[str]:
    """Permissions apply to command tools; feature gates cover other tools.

    References: learn.chatgpt.com/docs/permissions and
    learn.chatgpt.com/docs/config-file/config-reference. Permission profiles
    must not be combined with legacy --sandbox/YOLO flags.
    """
    scratch = os.path.realpath(scratch_dir)
    filesystem = json.dumps({":root": "deny", ":minimal": "read",
                             scratch: "read"}, separators=(",", ":"))
    # TOML inline tables use '=' rather than JSON's ':'. Quote keys/values
    # independently, never interpolate task-controlled strings as TOML.
    filesystem = "{" + ",".join(
        json.dumps(key) + "=" + json.dumps(value)
        for key, value in json.loads(filesystem).items()) + "}"
    values = [
        'approval_policy="never"',
        'default_permissions="gv-authority"',
        'permissions.gv-authority.extends=":read-only"',
        f"permissions.gv-authority.filesystem={filesystem}",
        'permissions.gv-authority.network.enabled=false',
        'web_search="disabled"', 'tools.web_search=false',
        'features.code_mode=false',
        'features.skip_host_skill_discovery=true',
        'suppress_unstable_features_warning=true',
        'project_doc_max_bytes=0', 'mcp_servers={}',
        'shell_environment_policy.inherit="none"',
        'shell_environment_policy.experimental_use_profile=false',
        *[f"features.{name}=false" for name in _DISABLED_FEATURES],
    ]
    return ["--ignore-user-config", "--ignore-rules", "--strict-config",
            "--ephemeral", *[
        item for value in values for item in ("-c", value)]]


def validate_authority_stream(stdout: str) -> None:
    """Only a complete, tool-free single turn can certify a cached contract."""
    phase = "new"
    for line in _native_jsonl_lines(stdout):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (ValueError, TypeError) as error:
            raise AuthorityBackendUnavailable(
                "authority generation emitted a non-JSON event") from error
        if not isinstance(event, dict):
            raise AuthorityBackendUnavailable("invalid authority event")
        kind = event.get("type")
        if phase == "thread" and kind == "item.completed":
            item = event.get("item")
            if (isinstance(item, dict) and item.get("type") == "error"
                    and item.get("message") == _DISABLED_CODE_MODE_NOTICE
                    and set(item) <= {"id", "type", "message"}):
                continue
        if kind == "thread.started" and phase == "new":
            phase = "thread"
        elif kind == "turn.started" and phase == "thread":
            phase = "turn"
        elif kind == "turn.completed" and phase == "turn":
            phase = "complete"
        elif phase == "turn" and kind in (
                "item.started", "item.updated", "item.completed"):
            item = event.get("item")
            if not isinstance(item, dict) or item.get("type") not in (
                    "reasoning", "agent_message"):
                raise AuthorityBackendUnavailable(
                    "authority generation attempted a tool or unknown item")
        else:
            raise AuthorityBackendUnavailable(
                "authority generation emitted an unsupported event")
    if phase != "complete":
        raise AuthorityBackendUnavailable(
            "authority generation requires exactly one complete turn")


class _AuthorityCodexBackend(CodexBackend):
    def _execution_policy_args(self, role_name, variables):
        return authority_policy_args(self._authority_scratch)

    def _execution_cwd(self, role_name, variables):
        return self._authority_scratch

    def _search_enabled(self, role_name):
        return False

    def _requires_json(self):
        return True

    def _validate_attempt_stream(self, stdout):
        validate_authority_stream(stdout)

    def _recover_completed_response(self, stdout):
        # The existing capability validator runs first. Permit only its exact
        # known startup notice, not arbitrary errors, tool activity or a new
        # event dialect. The recovery parser still checks unique item IDs.
        return _completed_response_without_usage(
            stdout, permitted_startup_notice=_DISABLED_CODE_MODE_NOTICE)


@dataclass(frozen=True)
class AuthorityBackendSession:
    backend: Any
    obligations_role: Any
    critic_role: Any
    identity: dict

    def revalidate(self):
        if self.identity["calendar_epoch"] != _calendar_epoch():
            raise AuthorityBackendUnavailable(
                "authority runtime calendar changed during compilation")


def _calendar_epoch():
    return {
        "utc_date": time.strftime("%Y-%m-%d", time.gmtime()),
        "local_date": time.strftime("%Y-%m-%d", time.localtime()),
        "timezone": list(time.tzname),
    }


@contextmanager
def authority_compilation_backend(backend, obligations_role, critic_role):
    """Create a private role registry; never mutate a shared candidate backend."""
    from ..engine.method_cache import _backend_identity

    if any(type(role) is not AgentRole
           for role in (obligations_role, critic_role)):
        raise AuthorityBackendUnavailable("unsupported authority-only role")
    roles = [replace(role, tools=[], system=role.system + AUTHORITY_ONLY_NOTE)
             for role in (obligations_role, critic_role)]
    registry = {role.name: role for role in roles}
    identity = {"boundary": AUTHORITY_BOUNDARY_VERSION,
                "backend": _backend_identity(backend),
                # Codex's environment message carries a date/timezone. A
                # process surviving midnight must not reuse yesterday's
                # interpretation of a relative-date authority.
                "calendar_epoch": _calendar_epoch()}
    # This exact built-in type is an explicit deterministic test seam, not a
    # promise that arbitrary duck-typed backends cannot inspect candidates.
    if type(backend) is FixtureBackend:
        # Object ids can be reused after GC while a cache entry survives.
        with _FIXTURE_SCOPES_LOCK:
            if backend not in _FIXTURE_SCOPES:
                _FIXTURE_SCOPES[backend] = uuid.uuid4().hex
            identity["fixture_scope"] = _FIXTURE_SCOPES[backend]
        yield AuthorityBackendSession(backend, *roles, identity)
        return
    # a routing backend seats the compiler and critic on its direct
    # transport; the authority-only session runs on that transport alone
    # (a single model call sees only its prompt: nothing to inspect)
    from .direct_backend import DirectBackend, RoutingBackend
    if type(backend) is RoutingBackend:
        if not (backend.routes_direct(obligations_role.name)
                and backend.routes_direct(critic_role.name)):
            backend = backend.agent
        else:
            backend = backend.direct
    if type(backend) not in (CodexBackend, AgentBackend, DirectBackend):
        raise AuthorityBackendUnavailable("unsupported authority-only backend")
    if type(backend) is CodexBackend and getattr(
            CodexBackend, "AUTHORITY_HOOKS_VERSION", None) != 1:
        raise AuthorityBackendUnavailable("Codex authority hooks unavailable")
    if type(backend) is CodexBackend:
        try:
            version = subprocess.run(
                ["codex", "--version"], check=True, capture_output=True,
                text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            raise AuthorityBackendUnavailable("Codex runtime identity unavailable") from error
        if not version.startswith("codex-cli ") or len(version) > 100:
            raise AuthorityBackendUnavailable("invalid Codex runtime identity")
        identity["runtime"] = version
    with tempfile.TemporaryDirectory(prefix="gv-authority-") as scratch:
        scratch = os.path.realpath(scratch)
        if type(backend) is CodexBackend:
            model_spec = backend.model + "".join(
                f";{name}={model}" for name, model in
                sorted(backend.model_overrides.items()))
            isolated = _AuthorityCodexBackend(
                roles=registry, cwd=backend.cwd, model=model_spec,
                effort=backend.effort, timeout=backend.timeout,
                service_tier=backend.service_tier,
                compact_inspection=backend.compact_inspection,
                judger_prefetch=backend.judger_prefetch)
            # Clone the captured instructions, not potentially changed module
            # defaults: the session identity above describes this exact backend.
            isolated._artifact_address_note = backend._artifact_address_note
            isolated._compact_inspection_note = backend._compact_inspection_note
            isolated._compact_report_note = backend._compact_report_note
            isolated._judger_prefetch_note = backend._judger_prefetch_note
            isolated._judger_prefetch_protocol = backend._judger_prefetch_protocol
            isolated._authority_scratch = scratch
            identity["policy"] = authority_policy_args("/AUTHORITY_SCRATCH")
        elif type(backend) is DirectBackend:
            isolated = DirectBackend(backend.transport, roles=registry,
                                     cwd=scratch, model=backend.model)
        else:
            isolated = AgentBackend(
                roles=registry, cwd=scratch, model=backend.model,
                permission_mode=backend.permission_mode)
        yield AuthorityBackendSession(isolated, *roles, identity)
