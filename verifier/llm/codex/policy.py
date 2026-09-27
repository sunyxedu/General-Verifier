"""The Codex CLI boundary of a seat: its permission profile, sandbox and
feature gates.

Three profiles. A live-world seat keeps the user-requested YOLO mode, so an
exec-mode run never stalls on a permission stop. An offline seat runs under
the root-deny ``gv-offline`` profile: the current state is readable, the
verifier-owned work tree and workspace are writable, and the operator-bound
shared toolchain/cache roots are granted as each seat declares
(``SEAT_GRANTS``); host caches are denied, commands get a fresh state-local
HOME/cache/tmp, and ambient integrations are disabled. A no-tools session
(the Codex translation of a no-tools ``ToolPolicy``) runs in an empty
scratch directory with every tool channel closed, and its JSON event stream
is checked (``validate_no_tools_stream``) to prove that no tool ran.
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ...config import offline_runtime_read_roots
from ..policy import ToolPolicy
from ..schema import AuthorityBackendUnavailable, RoleError
from .paths import (
    _canonical_policy_path,
    _ensure_verifier_owned_chain,
    _path_is_within,
)
from .stream import _native_jsonl_lines


_OFFLINE_WORLD_MODES = frozenset({"captured", "unavailable"})
_OFFLINE_PROFILE = "gv-offline"

# Offline seats keep the shell/code-mode path needed to build arbitrary
# verification instruments. Ambient host integrations are different: they
# are neither represented in the captured world nor confined by the command
# sandbox, so remove them from the seat entirely. ``view_image`` remains
# enabled because evidence seats inspect renderings they create below their
# writable roots. Codex does not document that this native handler is governed
# by filesystem permission profiles, so this policy alone must not be treated
# as a confinement guarantee for that channel.
_OFFLINE_DISABLED_FEATURES = (
    "apps", "enable_mcp_apps", "plugins", "remote_plugin",
    "plugin_sharing", "tool_call_mcp_elicitation", "multi_agent",
    "memories", "browser_use", "browser_use_external",
    "browser_use_full_cdp_access", "computer_use", "in_app_browser",
    "in_app_local_automation", "workspace_dependencies", "hooks",
    "skill_search", "skill_mcp_dependency_install", "image_generation",
    "recommended_plugins", "goals", "shell_snapshot",
)

# The profile a live-world seat runs under.
LIVE_POLICY_ARGS = ("--dangerously-bypass-approvals-and-sandbox",)

TOOLCHAIN_READ = "read"
TOOLCHAIN_WRITE = "write"


@dataclass(frozen=True)
class SeatGrants:
    """What one seat is granted beyond its profile's shared boundary.

    ``web_search``: the knowledge channel, when the run allows it.
    ``images``: captured-world image inputs attached to the call.
    ``toolchain``: the operator toolchain, read/execute or writable.
    ``shared_cache``: the operator's shared package-manager cache.
    """
    web_search: bool = False
    images: bool = False
    toolchain: Optional[str] = None
    shared_cache: bool = False


# Web search is a KNOWLEDGE channel for the contract/method seats only.
# The evidence seats (discharge, judger) stay offline-searchless: witnesses
# must come from sealed local execution, and audits must rest on the sealed
# record alone — a mutable, unauthenticated web page can never be either.
# The discharge installs into the toolchain and the shared cache; the
# independent Judger may rerun that toolchain but not alter it.
SEAT_GRANTS: Mapping[str, SeatGrants] = {
    "extract": SeatGrants(web_search=True),
    "obligations": SeatGrants(web_search=True),
    "critic": SeatGrants(web_search=True),
    "oracles": SeatGrants(web_search=True),
    "discharge": SeatGrants(images=True, toolchain=TOOLCHAIN_WRITE,
                            shared_cache=True),
    "judger": SeatGrants(images=True, toolchain=TOOLCHAIN_READ),
}
_NO_GRANTS = SeatGrants()


def seat_grants(role_name: str) -> SeatGrants:
    """The grants a seat declares (none for a seat not in the table)."""
    return SEAT_GRANTS.get(role_name, _NO_GRANTS)


def is_offline_execution(role_name: str, variables: Mapping[str, Any],
                         cwd: Optional[str]) -> bool:
    """Whether a seat runs under the offline profile: a captured or
    unavailable world, or extraction over a runtime's state directory."""
    return bool(
        variables.get("world_mode") in _OFFLINE_WORLD_MODES
        or (cwd is not None and role_name == "extract")
    )


def _toml_string(value: str) -> str:
    """A JSON string is also a valid TOML basic string."""
    return json.dumps(value, ensure_ascii=False)


class OfflineProfile:
    """The operator-bound roots of a runtime's offline seats and the
    root-deny permission profile each seat runs under. The roots come only
    from verifier/operator construction: role variables may name the same
    roots so prompts can use them, but may never add or replace a
    filesystem grant."""

    def __init__(self, toolchain_dir: Optional[str],
                 shared_cache_dir: Optional[str]):
        self.toolchain_dir = (
            _canonical_policy_path(
                toolchain_dir, "operator toolchain directory")
            if toolchain_dir else None
        )
        self.shared_cache_dir = (
            _canonical_policy_path(
                shared_cache_dir, "operator shared cache directory")
            if shared_cache_dir else None
        )

    def tool_roots(self, state_dir: str) -> tuple[str, Optional[str]]:
        toolchain = self.toolchain_dir or os.path.join(
            state_dir, "work", "_toolchains")
        cache = self.shared_cache_dir
        if cache is not None and (
                _path_is_within(toolchain, cache)
                or _path_is_within(cache, toolchain)):
            raise RoleError(
                "offline operator toolchain and shared cache must be "
                "disjoint")
        return toolchain, cache

    def state_dir(self, cwd: Optional[str],
                  variables: Mapping[str, Any]) -> str:
        """The state directory an offline seat is isolated in: the
        runtime's cwd, else the directory of the ledger it reads."""
        # Offline score() constructs the backend with cwd=state_dir.  Keep a
        # fallback for direct callers that provide only the ledger path, but
        # fail closed rather than place an allegedly isolated HOME in an
        # unrelated process directory.
        state_dir = cwd
        ledger_locator = (variables.get("_mixed_ledger_path")
                          or variables.get("ledger_path"))
        if not state_dir and ledger_locator:
            state_dir = os.path.dirname(
                os.path.abspath(str(ledger_locator)))
        if not state_dir:
            raise RoleError(
                "offline execution needs a state cwd or ledger path")
        state_dir = _canonical_policy_path(state_dir, "state directory")
        try:
            st = os.lstat(state_dir)
        except OSError as err:
            raise RoleError(f"offline state directory is unavailable: {err}")
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise RoleError(
                f"offline state must be a real directory: {state_dir}")
        expected_uid = getattr(os, "geteuid", lambda: st.st_uid)()
        if st.st_uid != expected_uid:
            raise RoleError(
                f"offline state is not owned by this process: {state_dir}")
        return state_dir

    def validate_scope(self, cwd: Optional[str], state_dir: str,
                       toolchain_dir: Optional[str],
                       shared_cache_dir: Optional[str]) -> None:
        """Require Service and backend to share one operator-owned scope."""
        state = _canonical_policy_path(state_dir, "state directory")
        if cwd is not None:
            backend_state = _canonical_policy_path(
                cwd, "backend state directory")
            if backend_state != state:
                raise RoleError(
                    "offline Service state does not match the backend cwd")
        expected_toolchain, expected_cache = self.tool_roots(state)
        supplied_toolchain = (
            _canonical_policy_path(toolchain_dir, "toolchain_dir")
            if toolchain_dir else os.path.join(state, "work", "_toolchains")
        )
        supplied_cache = (
            _canonical_policy_path(shared_cache_dir, "shared_cache_dir")
            if shared_cache_dir else None
        )
        if supplied_toolchain != expected_toolchain:
            raise RoleError(
                "offline Service toolchain does not match the backend's "
                "operator-owned scope")
        if supplied_cache != expected_cache:
            raise RoleError(
                "offline Service shared cache does not match the backend's "
                "operator-owned scope")

    def args(self, role_name: str, variables: Mapping[str, Any],
             state_dir: str, *, search: bool) -> list[str]:
        """The CLI boundary of offline seat ``role_name`` isolated in
        ``state_dir``; ``search`` opens its knowledge channel."""
        grants = seat_grants(role_name)
        try:
            runtime_read_roots = offline_runtime_read_roots()
        except ValueError as error:
            raise RoleError(str(error)) from error
        for root in runtime_read_roots:
            if (_path_is_within(root, state_dir)
                    or _path_is_within(state_dir, root)):
                raise RoleError(
                    "offline runtime read root overlaps the isolated state: "
                    f"{root}")

        work = os.path.join(state_dir, "work")
        home = os.path.join(work, "_offline_home")
        cache = os.path.join(home, ".cache")
        tmp = os.path.join(home, "tmp")
        hf = os.path.join(cache, "huggingface")
        env = {
            "HOME": home,
            "XDG_CACHE_HOME": cache,
            "TMPDIR": tmp,
            "HF_HOME": hf,
            "HF_DATASETS_CACHE": os.path.join(hf, "datasets"),
            "HF_HUB_CACHE": os.path.join(hf, "hub"),
            # Older huggingface_hub releases used the longer spelling.
            "HUGGINGFACE_HUB_CACHE": os.path.join(hf, "hub"),
            "TRANSFORMERS_CACHE": os.path.join(hf, "transformers"),
            "HF_XET_CACHE": os.path.join(hf, "xet"),
            "HF_ASSETS_CACHE": os.path.join(hf, "assets"),
        }
        # Check every security-sensitive path component before granting the
        # work tree write access.  os.makedirs(exist_ok=True) would silently
        # follow a candidate-planted ``work``/HOME/cache symlink.
        for path in dict.fromkeys((work, *env.values())):
            _ensure_verifier_owned_chain(work, path)
        # Shared tool roots are bound when the backend is constructed.  The
        # prompt variables may repeat those paths for the discharge role, but
        # are not an authority to add or replace a filesystem grant.
        operator_toolchain, operator_cache = self.tool_roots(state_dir)
        for key, expected in (
            ("toolchain_dir", operator_toolchain),
            ("shared_cache_dir", operator_cache),
        ):
            declared = variables.get(key)
            if not declared:
                continue
            declared = _canonical_policy_path(declared, key)
            if expected is None or declared != expected:
                raise RoleError(
                    f"offline {key} does not match the backend's "
                    "operator-owned scope")

        # Additional writable roots inside the state have distinct meanings.
        # A scratch workdir cannot escape state/work and the durable workspace
        # cannot escape state/workspace. External toolchain/cache grants below
        # come exclusively from the backend's operator-owned scope.
        read_roots: list[str] = []
        writable_roots: list[str] = []
        toolchain_root: Optional[str] = None
        workspace_base = os.path.join(state_dir, "workspace")
        for key in ("workdir", "workspace_dir"):
            candidate = variables.get(key)
            if not candidate:
                continue
            candidate = _canonical_policy_path(candidate, key)
            if _path_is_within(work, candidate):
                _ensure_verifier_owned_chain(work, candidate)
                continue
            if key == "workdir":
                raise RoleError(
                    f"offline workdir escapes verifier work tree: {candidate}")
            if key == "workspace_dir":
                if not _path_is_within(workspace_base, candidate):
                    raise RoleError(
                        "offline workspace_dir escapes the state workspace: "
                        f"{candidate}")
                _ensure_verifier_owned_chain(workspace_base, candidate)
                writable_roots.append(candidate)
                continue
        if grants.toolchain is not None:
            toolchain_root = operator_toolchain
            if _path_is_within(work, toolchain_root):
                _ensure_verifier_owned_chain(work, toolchain_root)
            else:
                if (_path_is_within(state_dir, toolchain_root)
                        or _path_is_within(toolchain_root, state_dir)):
                    raise RoleError(
                        "offline operator toolchain overlaps the isolated "
                        f"state: {toolchain_root}")
                _ensure_verifier_owned_chain(toolchain_root, toolchain_root)
            if grants.toolchain == TOOLCHAIN_READ:
                # More specific than state/work=write for a cold, state-local
                # toolchain: the independent Judger may rerun it but not alter
                # what the discharge installed.
                read_roots.append(toolchain_root)
            elif not _path_is_within(work, toolchain_root):
                writable_roots.append(toolchain_root)

        if grants.shared_cache and operator_cache is not None:
            if _path_is_within(work, operator_cache):
                _ensure_verifier_owned_chain(work, operator_cache)
            else:
                if (_path_is_within(state_dir, operator_cache)
                        or _path_is_within(operator_cache, state_dir)):
                    raise RoleError(
                        "offline operator shared cache overlaps the isolated "
                        f"state: {operator_cache}")
                _ensure_verifier_owned_chain(operator_cache, operator_cache)
                writable_roots.append(operator_cache)
        for runtime_root in runtime_read_roots:
            for writable_root in (work, *writable_roots):
                if (_path_is_within(runtime_root, writable_root)
                        or _path_is_within(writable_root, runtime_root)):
                    raise RoleError(
                        "offline runtime read root overlaps a writable root: "
                        f"{runtime_root}")
        if grants.shared_cache and operator_cache is not None:
            # package-manager caches shared across trials: downloads and
            # unpacked wheels are reused, installs into a fresh toolchain
            # become hardlinks instead of fetches
            env.update({
                "UV_CACHE_DIR": os.path.join(operator_cache, "uv"),
                "UV_LINK_MODE": "hardlink",
                "PIP_CACHE_DIR": os.path.join(operator_cache, "pip"),
                "npm_config_cache": os.path.join(operator_cache, "npm"),
                "CARGO_HOME": os.path.join(operator_cache, "cargo"),
                "GOMODCACHE": os.path.join(operator_cache, "gomod"),
            })
        path_parts = []
        if toolchain_root is not None:
            path_parts.append(os.path.join(toolchain_root, "bin"))
        path_parts.extend(
            root for root in runtime_read_roots
            if os.path.basename(root) == "bin"
        )
        path_parts.extend(("/usr/bin", "/bin", "/usr/sbin", "/sbin"))
        env["PATH"] = os.pathsep.join(dict.fromkeys(path_parts))
        env_table = "{" + ",".join(
            f"{key}={_toml_string(value)}" for key, value in env.items()
        ) + "}"
        filesystem = "{" + ",".join((
            '":root"="deny"',
            '":minimal"="read"',
            f"{_toml_string(state_dir)}=\"read\"",
            f"{_toml_string(work)}=\"write\"",
            *(f"{_toml_string(root)}=\"read\""
              for root in dict.fromkeys((*runtime_read_roots, *read_roots))),
            *(f"{_toml_string(root)}=\"write\""
              for root in dict.fromkeys(writable_roots)),
            # Keep explicit host-cache denies even under :root=deny. They
            # override any matching :minimal exception in this or a future
            # runtime and protect auth/session-bearing Codex state.
            '"~/.cache"="deny"',
            '"~/Library/Caches"="deny"',
            '"~/.huggingface"="deny"',
            '"~/.codex"="deny"',
        )) + "}"

        # Permission profiles and the legacy --sandbox flag do not compose:
        # selecting the profile through default_permissions is intentional.
        # The Codex process itself keeps its real HOME (and therefore auth);
        # shell_environment_policy applies to commands launched by the seat.
        return [
            "--ignore-user-config",
            "--ignore-rules",
            "--strict-config",
            "--ephemeral",
            "-c", f'default_permissions="{_OFFLINE_PROFILE}"',
            "-c", 'approval_policy="never"',
            "-c", f'permissions.{_OFFLINE_PROFILE}.extends=":read-only"',
            "-c", f"permissions.{_OFFLINE_PROFILE}.filesystem={filesystem}",
            # Network stays ON for the offline profile: discharge instruments
            # routinely need pinned toolchains (grpcio-tools==X, protoc,
            # build deps) that no pre-provisioned venv can anticipate, and a
            # DNS-dead sandbox turns those checks into structural
            # unobtainables. The trade: delivered world code exercised by an
            # instrument may reach the network during verification. State
            # integrity is still guarded by the immutable-input seal; results
            # that depend on a fetched toolchain must pin exact versions.
            "-c", f"permissions.{_OFFLINE_PROFILE}.network.enabled=true",
            "-c", ("tools.web_search=true" if search
                   else "tools.web_search=false"),
            "-c", "features.skip_host_skill_discovery=true",
            "-c", "suppress_unstable_features_warning=true",
            "-c", "project_doc_max_bytes=0",
            "-c", "mcp_servers={}",
            *[
                item
                for feature in _OFFLINE_DISABLED_FEATURES
                for item in ("-c", f"features.{feature}=false")
            ],
            "-c", 'shell_environment_policy.inherit="core"',
            "-c", f"shell_environment_policy.set={env_table}",
        ]


# The no-tools policy's feature gates: every tool channel other than the
# command tools the permission profile governs.
_NO_TOOLS_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "shell_snapshot", "apps", "plugins",
    "remote_plugin", "multi_agent", "memories", "browser_use",
    "browser_use_external", "browser_use_full_cdp_access", "computer_use",
    "in_app_browser", "workspace_dependencies", "code_mode_host", "hooks",
    "skill_search", "skill_mcp_dependency_install", "image_generation",
    "view_image", "sleep_tool", "goals",
)


def no_tools_policy_args(scratch_dir: str, policy: ToolPolicy) -> list[str]:
    """The Codex CLI boundary of a no-tools ``policy`` run in an empty
    ``scratch_dir``. Permissions apply to command tools; feature gates
    cover other tools; the policy's denied integrations are closed after
    the shared profile.

    References: learn.chatgpt.com/docs/permissions and
    learn.chatgpt.com/docs/config-file/config-reference. Permission profiles
    must not be combined with legacy --sandbox/YOLO flags.
    """
    if not policy.no_tools:
        raise ValueError("the Codex no-tools profile needs a no-tools policy")
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
        *[f"features.{name}=false" for name in _NO_TOOLS_DISABLED_FEATURES],
    ]
    return ["--ignore-user-config", "--ignore-rules", "--strict-config",
            "--ephemeral", *[
        item for value in values for item in ("-c", value)], *[
        item
        for feature in policy.denied_integrations
        for item in ("-c", f"features.{feature}=false")
    ]]


# Codex 0.153.4 emits this capability-disabled notice before turn.started,
# even with code_mode=false. It contains no tool result or candidate data.
# Unknown errors and any tool activity still reject the entire generation.
DISABLED_CODE_MODE_NOTICE = (
    "Code Mode is unavailable because code-mode host is disabled. Code mode "
    "will fail closed; enable `features.code_mode_host` and install "
    "`codex-code-mode-host`."
)


def validate_no_tools_stream(stdout: str) -> None:
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
                    and item.get("message") == DISABLED_CODE_MODE_NOTICE
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
