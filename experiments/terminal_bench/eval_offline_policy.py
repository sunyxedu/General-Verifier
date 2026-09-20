"""Common offline Codex permission policy for source-comparison evals.

This module is an evaluation overlay, not part of the verifier runtime and
not a claim that a baseline is pristine.  It temporarily replaces only a
selected source tree's ``CodexBackend._execution_policy_args`` method.  Role
prompts, tools, models, retry budgets, and response handling remain owned by
that source tree.

The overlay deliberately has no verifier imports.  A worker can import this
file, then load one selected frozen verifier tree and explicitly bind each
offline state to operator-owned filesystem roots.  Candidate variables may
repeat a binding but cannot add authority.

The separately named online entry point applies the same filesystem and tool
isolation to calls explicitly marked ``world_mode="live"``.  It never changes
the original offline entry point's accepted modes or identity protocol.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import marshal
import os
import stat
import threading
import types
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Optional


POLICY_PROTOCOL = "gv-common-offline-codex-policy-v1"
POLICY_IDENTITY_SCHEMA = "gv-common-offline-codex-policy-identity-v1"
ONLINE_POLICY_PROTOCOL = "gv-common-online-codex-policy-v1"
ONLINE_POLICY_IDENTITY_SCHEMA = "gv-common-online-codex-policy-identity-v1"

_OFFLINE_PROFILE = "gv-offline"
_OFFLINE_WORLD_MODES = frozenset({"captured", "unavailable"})
_LEGACY_SEARCH_ROLES = frozenset(
    {"extract", "obligations", "critic", "oracles"})
_DISABLED_FEATURES = (
    "apps", "enable_mcp_apps", "plugins", "remote_plugin",
    "plugin_sharing", "tool_call_mcp_elicitation", "multi_agent",
    "memories", "browser_use", "browser_use_external",
    "browser_use_full_cdp_access", "computer_use", "in_app_browser",
    "in_app_local_automation", "workspace_dependencies", "hooks",
    "skill_search", "skill_mcp_dependency_install", "image_generation",
    "recommended_plugins", "goals", "shell_snapshot",
)
_MAX_BINDINGS = 256
_MAX_RUNTIME_ROOTS = 64
_MAX_PATH_BYTES = 4096
_MAX_SOURCE_BYTES = 8 * 1024 * 1024


class EvalOfflinePolicyError(RuntimeError):
    """The eval overlay cannot establish its promised offline boundary."""


def _paths_overlap(left: str, right: str) -> bool:
    try:
        common = os.path.commonpath((left, right))
    except ValueError:
        return False
    return common in (left, right)


def _path_is_within(root: str, target: str) -> bool:
    try:
        return os.path.commonpath((root, target)) == root
    except ValueError:
        return False


def _canonical_path(value: Any, label: str) -> str:
    try:
        raw = os.fspath(value)
    except TypeError as error:
        raise EvalOfflinePolicyError(f"{label} must be a path string") from error
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise EvalOfflinePolicyError(f"{label} must be a non-empty path string")
    if len(os.fsencode(raw)) > _MAX_PATH_BYTES:
        raise EvalOfflinePolicyError(f"{label} is too long")
    absolute = os.path.abspath(raw)
    if raw != absolute:
        raise EvalOfflinePolicyError(f"{label} must be an absolute canonical path")
    if os.path.realpath(absolute) != absolute:
        raise EvalOfflinePolicyError(
            f"{label} must contain no symlink path components")
    return absolute


def _existing_real_dir(path: str, label: str) -> os.stat_result:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise EvalOfflinePolicyError(
            f"{label} must be an existing directory") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise EvalOfflinePolicyError(f"{label} must be an existing real directory")
    return info


def _owned_real_dir(path: str, label: str) -> os.stat_result:
    info = _existing_real_dir(path, label)
    expected_uid = getattr(os, "geteuid", lambda: info.st_uid)()
    if info.st_uid != expected_uid:
        raise EvalOfflinePolicyError(f"{label} is not owned by this process")
    return info


@dataclass(frozen=True)
class OfflinePolicyBinding:
    """Operator authority for exactly one isolated offline state.

    Construction is read-only.  Writable verifier directories are created
    only when a role actually asks for its policy arguments.
    """

    state_dir: str
    toolchain_dir: Optional[str] = None
    shared_cache_dir: Optional[str] = None
    runtime_read_roots: tuple[str, ...] = ()
    _state_identity: tuple[int, int, int] = field(
        init=False, repr=False, compare=False)
    _runtime_identities: tuple[tuple[str, int, int], ...] = field(
        init=False, repr=False, compare=False)
    _operator_root_identities: tuple[tuple[str, int, int], ...] = field(
        init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        state = _canonical_path(self.state_dir, "state_dir")
        if state == os.path.sep:
            raise EvalOfflinePolicyError("state_dir must not be the filesystem root")
        state_info = _owned_real_dir(state, "state_dir")

        default_toolchain = os.path.join(state, "work", "_toolchains")
        toolchain = _canonical_path(
            self.toolchain_dir or default_toolchain, "toolchain_dir")
        cache = (
            _canonical_path(self.shared_cache_dir, "shared_cache_dir")
            if self.shared_cache_dir else None
        )
        if cache is not None and _paths_overlap(toolchain, cache):
            raise EvalOfflinePolicyError(
                "toolchain_dir and shared_cache_dir must be disjoint")
        work = os.path.join(state, "work")
        operator_root_identities = []
        for label, root in (
            ("toolchain_dir", toolchain),
            ("shared_cache_dir", cache),
        ):
            if root is None or _path_is_within(work, root):
                continue
            if _paths_overlap(state, root):
                raise EvalOfflinePolicyError(
                    f"{label} must not overlap state_dir outside state/work")
            info = _owned_real_dir(root, label)
            operator_root_identities.append((root, info.st_dev, info.st_ino))

        roots = tuple(self.runtime_read_roots)
        if len(roots) > _MAX_RUNTIME_ROOTS:
            raise EvalOfflinePolicyError("too many runtime_read_roots")
        home = os.path.realpath(os.path.abspath(os.path.expanduser("~")))
        normalized_roots = []
        root_identities = []
        seen = set()
        for index, value in enumerate(roots):
            label = f"runtime_read_roots[{index}]"
            root = _canonical_path(value, label)
            if root == os.path.sep:
                raise EvalOfflinePolicyError(
                    f"{label} must not grant the filesystem root")
            if _paths_overlap(root, home):
                raise EvalOfflinePolicyError(f"{label} must not overlap HOME")
            if _paths_overlap(root, state):
                raise EvalOfflinePolicyError(f"{label} must not overlap state_dir")
            if (_paths_overlap(root, toolchain)
                    or (cache is not None and _paths_overlap(root, cache))):
                raise EvalOfflinePolicyError(
                    f"{label} must not overlap an operator tool root")
            if root in seen:
                raise EvalOfflinePolicyError(f"{label} duplicates an earlier root")
            info = _existing_real_dir(root, label)
            seen.add(root)
            normalized_roots.append(root)
            root_identities.append((root, info.st_dev, info.st_ino))

        object.__setattr__(self, "state_dir", state)
        object.__setattr__(self, "toolchain_dir", toolchain)
        object.__setattr__(self, "shared_cache_dir", cache)
        object.__setattr__(self, "runtime_read_roots", tuple(normalized_roots))
        object.__setattr__(
            self, "_state_identity",
            (state_info.st_dev, state_info.st_ino, state_info.st_uid))
        object.__setattr__(
            self, "_runtime_identities", tuple(root_identities))
        object.__setattr__(
            self, "_operator_root_identities",
            tuple(operator_root_identities))

    def _revalidate(self) -> None:
        info = _owned_real_dir(self.state_dir, "state_dir")
        if (info.st_dev, info.st_ino, info.st_uid) != self._state_identity:
            raise EvalOfflinePolicyError("state_dir identity changed after planning")
        for root, expected_dev, expected_ino in self._runtime_identities:
            current = _existing_real_dir(root, "runtime_read_root")
            if (current.st_dev, current.st_ino) != (expected_dev, expected_ino):
                raise EvalOfflinePolicyError(
                    "runtime_read_root identity changed after planning")
        for root, expected_dev, expected_ino in self._operator_root_identities:
            current = _owned_real_dir(root, "operator tool root")
            if (current.st_dev, current.st_ino) != (expected_dev, expected_ino):
                raise EvalOfflinePolicyError(
                    "operator tool root identity changed after planning")

    def _identity_record(self) -> dict[str, Any]:
        return {
            "state_dir": self.state_dir,
            "toolchain_dir": self.toolchain_dir,
            "shared_cache_dir": self.shared_cache_dir,
            "runtime_read_roots": list(self.runtime_read_roots),
        }


def _stable_file_bytes(path: str, label: str) -> bytes:
    path = _canonical_path(path, label)
    before_path = _existing_real_dir(os.path.dirname(path), f"{label} parent")
    del before_path
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise EvalOfflinePolicyError(f"cannot open {label}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_SOURCE_BYTES:
            raise EvalOfflinePolicyError(f"{label} is not a bounded regular file")
        chunks = []
        remaining = _MAX_SOURCE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(data) > _MAX_SOURCE_BYTES:
        raise EvalOfflinePolicyError(f"{label} exceeds the source cap")
    try:
        current = os.lstat(path)
    except OSError as error:
        raise EvalOfflinePolicyError(f"{label} changed during identity read") from error
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_size,
        item.st_mtime_ns, item.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise EvalOfflinePolicyError(f"{label} changed during identity read")
    return data


def _function_identities(function: Any, label: str) -> tuple[str, str]:
    if not inspect.isfunction(function):
        raise EvalOfflinePolicyError(f"{label} must be a Python function")
    try:
        source = inspect.getsource(function).encode("utf-8")
    except (OSError, TypeError) as error:
        raise EvalOfflinePolicyError(f"cannot inspect {label} source") from error
    source_hash = hashlib.sha256(source).hexdigest()
    code_hash = hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
    return source_hash, code_hash


def _source_file(module: types.ModuleType, backend_class: type) -> str:
    path = inspect.getsourcefile(backend_class) or getattr(module, "__file__", None)
    if not path:
        raise EvalOfflinePolicyError("backend module has no inspectable source file")
    return _canonical_path(path, "backend source")


@dataclass(frozen=True)
class EvalOfflinePolicyPlan:
    """Read-only, source-bound plan used for a later temporary install."""

    module: types.ModuleType = field(repr=False, compare=False)
    backend_class: type = field(repr=False, compare=False)
    original_method: Any = field(repr=False, compare=False)
    bindings: tuple[OfflinePolicyBinding, ...] = field(repr=False)
    _identity: Mapping[str, Any] = field(repr=False, compare=False)

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._identity)


@dataclass(frozen=True)
class EvalOnlinePolicyPlan:
    """Read-only, source-bound plan for an isolated live-world eval."""

    module: types.ModuleType = field(repr=False, compare=False)
    backend_class: type = field(repr=False, compare=False)
    original_method: Any = field(repr=False, compare=False)
    bindings: tuple[OfflinePolicyBinding, ...] = field(repr=False)
    _identity: Mapping[str, Any] = field(repr=False, compare=False)

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._identity)


def plan_eval_offline_policy(
    module: types.ModuleType,
    *,
    bindings: Iterable[OfflinePolicyBinding],
) -> EvalOfflinePolicyPlan:
    """Build an immutable plan without creating or changing directories."""
    if not isinstance(module, types.ModuleType):
        raise EvalOfflinePolicyError("module must be an imported Python module")
    backend_class = getattr(module, "CodexBackend", None)
    if not isinstance(backend_class, type):
        raise EvalOfflinePolicyError("module must expose CodexBackend")
    if backend_class.__module__ != module.__name__:
        raise EvalOfflinePolicyError("CodexBackend must be defined by the module")
    original = inspect.getattr_static(backend_class, "_execution_policy_args", None)
    original_source_hash, original_code_hash = _function_identities(
        original, "CodexBackend._execution_policy_args")

    normalized = tuple(bindings)
    if not normalized or len(normalized) > _MAX_BINDINGS:
        raise EvalOfflinePolicyError("bindings must contain 1..256 states")
    if not all(isinstance(item, OfflinePolicyBinding) for item in normalized):
        raise EvalOfflinePolicyError("bindings must contain OfflinePolicyBinding values")
    states = [item.state_dir for item in normalized]
    if len(states) != len(set(states)):
        raise EvalOfflinePolicyError("each state_dir may be bound only once")

    backend_source = _stable_file_bytes(
        _source_file(module, backend_class), "backend source")
    overlay_source = _stable_file_bytes(__file__, "overlay source")
    policy_source_hash, policy_code_hash = _function_identities(
        _build_policy_args, "overlay policy implementation")
    bindings_bytes = json.dumps(
        [item._identity_record() for item in normalized],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    identity = MappingProxyType({
        "schema": POLICY_IDENTITY_SCHEMA,
        "protocol": POLICY_PROTOCOL,
        "backend_module": module.__name__,
        "backend_source_sha256": hashlib.sha256(backend_source).hexdigest(),
        "original_method_source_sha256": original_source_hash,
        "original_method_code_sha256": original_code_hash,
        "overlay_source_sha256": hashlib.sha256(overlay_source).hexdigest(),
        "policy_method_source_sha256": policy_source_hash,
        "policy_method_code_sha256": policy_code_hash,
        "bindings_sha256": hashlib.sha256(bindings_bytes).hexdigest(),
        "binding_count": len(normalized),
        "scope": "exact-class-offline-only",
    })
    return EvalOfflinePolicyPlan(
        module=module, backend_class=backend_class, original_method=original,
        bindings=normalized, _identity=identity)


def plan_eval_online_policy(
    module: types.ModuleType,
    *,
    bindings: Iterable[OfflinePolicyBinding],
) -> EvalOnlinePolicyPlan:
    """Build an immutable live-world plan without changing directories.

    This is an explicit evaluation overlay, not the backend's production
    online policy.  It reuses the offline renderer only after binding a live
    call to an exact operator-owned state directory.
    """
    if not isinstance(module, types.ModuleType):
        raise EvalOfflinePolicyError("module must be an imported Python module")
    backend_class = getattr(module, "CodexBackend", None)
    if not isinstance(backend_class, type):
        raise EvalOfflinePolicyError("module must expose CodexBackend")
    if backend_class.__module__ != module.__name__:
        raise EvalOfflinePolicyError("CodexBackend must be defined by the module")
    original = inspect.getattr_static(backend_class, "_execution_policy_args", None)
    original_source_hash, original_code_hash = _function_identities(
        original, "CodexBackend._execution_policy_args")

    normalized = tuple(bindings)
    if not normalized or len(normalized) > _MAX_BINDINGS:
        raise EvalOfflinePolicyError("bindings must contain 1..256 states")
    if not all(isinstance(item, OfflinePolicyBinding) for item in normalized):
        raise EvalOfflinePolicyError("bindings must contain OfflinePolicyBinding values")
    states = [item.state_dir for item in normalized]
    if len(states) != len(set(states)):
        raise EvalOfflinePolicyError("each state_dir may be bound only once")

    backend_source = _stable_file_bytes(
        _source_file(module, backend_class), "backend source")
    overlay_source = _stable_file_bytes(__file__, "overlay source")
    policy_source_hash, policy_code_hash = _function_identities(
        _build_policy_args, "overlay policy implementation")
    bindings_bytes = json.dumps(
        [item._identity_record() for item in normalized],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    identity = MappingProxyType({
        "schema": ONLINE_POLICY_IDENTITY_SCHEMA,
        "protocol": ONLINE_POLICY_PROTOCOL,
        "backend_module": module.__name__,
        "backend_source_sha256": hashlib.sha256(backend_source).hexdigest(),
        "original_method_source_sha256": original_source_hash,
        "original_method_code_sha256": original_code_hash,
        "overlay_source_sha256": hashlib.sha256(overlay_source).hexdigest(),
        "policy_method_source_sha256": policy_source_hash,
        "policy_method_code_sha256": policy_code_hash,
        "bindings_sha256": hashlib.sha256(bindings_bytes).hexdigest(),
        "binding_count": len(normalized),
        "scope": "exact-class-live-only",
    })
    return EvalOnlinePolicyPlan(
        module=module, backend_class=backend_class, original_method=original,
        bindings=normalized, _identity=identity)


def _ensure_owned_dir(path: str) -> None:
    if not os.path.lexists(path):
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
    info = _owned_real_dir(path, "offline writable directory")
    flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise EvalOfflinePolicyError(
            "cannot safely open offline writable directory") from error
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)):
            raise EvalOfflinePolicyError(
                "offline writable directory changed during validation")
        os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)


def _ensure_owned_chain(root: str, target: str) -> None:
    if not _path_is_within(root, target):
        raise EvalOfflinePolicyError(
            "offline writable directory escapes its operator root")
    _ensure_owned_dir(root)
    relative = os.path.relpath(target, root)
    if relative == ".":
        return
    current = root
    for component in relative.split(os.sep):
        if component in ("", ".", ".."):
            raise EvalOfflinePolicyError("invalid writable path component")
        current = os.path.join(current, component)
        _ensure_owned_dir(current)


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _role_error(plan: EvalOfflinePolicyPlan, message: str) -> BaseException:
    error_class = getattr(plan.module, "RoleError", None)
    if not isinstance(error_class, type) or not issubclass(error_class, BaseException):
        return EvalOfflinePolicyError(message)
    return error_class(message)


def _checked_declared_path(
    value: Any, label: str, plan: EvalOfflinePolicyPlan,
) -> str:
    try:
        return _canonical_path(value, label)
    except EvalOfflinePolicyError as error:
        raise _role_error(plan, str(error)) from error


def _search_enabled(plan: EvalOfflinePolicyPlan, backend: Any, role_name: str) -> bool:
    hook = getattr(backend, "_search_enabled", None)
    if hook is not None:
        if not callable(hook):
            raise _role_error(plan, "backend _search_enabled is not callable")
        return bool(hook(role_name))
    roles = getattr(plan.module, "_SEARCH_ROLES", _LEGACY_SEARCH_ROLES)
    try:
        return role_name in roles
    except TypeError as error:
        raise _role_error(plan, "backend search-role policy is invalid") from error


def _binding_for_backend(
    plan: EvalOfflinePolicyPlan | EvalOnlinePolicyPlan,
    backend: Any,
    role_name: str,
    variables: Mapping[str, Any],
) -> OfflinePolicyBinding:
    if type(backend) is not plan.backend_class:
        message = (
            "eval online policy supports only the exact CodexBackend class"
            if isinstance(plan, EvalOnlinePolicyPlan)
            else "eval offline policy supports only the exact CodexBackend class"
        )
        raise _role_error(plan, message)
    if isinstance(plan, EvalOnlinePolicyPlan):
        world_mode = variables.get("world_mode")
        if type(world_mode) is not str or world_mode != "live":
            raise _role_error(
                plan, "eval online policy requires world_mode='live'")
    else:
        offline_check = getattr(backend, "_is_offline_execution", None)
        if not callable(offline_check) or not offline_check(role_name, variables):
            raise _role_error(
                plan, "eval offline policy refuses live or unclassified execution")
    cwd = getattr(backend, "cwd", None)
    if cwd is None:
        raise _role_error(
            plan, "eval offline policy requires an operator-bound backend cwd")
    try:
        state = _canonical_path(cwd, "backend cwd")
    except EvalOfflinePolicyError as error:
        raise _role_error(plan, str(error)) from error
    by_state = {item.state_dir: item for item in plan.bindings}
    binding = by_state.get(state)
    if binding is None:
        raise _role_error(plan, "backend cwd has no eval offline policy binding")
    try:
        binding._revalidate()
    except EvalOfflinePolicyError as error:
        raise _role_error(plan, str(error)) from error

    # Current backends can also carry roots in their constructor.  When those
    # fields exist, require them to agree with the overlay rather than letting
    # two operator configurations silently diverge.  Legacy backends simply
    # lack these attributes and are governed by the overlay binding alone.
    if hasattr(backend, "offline_toolchain_dir"):
        declared = getattr(backend, "offline_toolchain_dir")
        expected_default = os.path.join(state, "work", "_toolchains")
        effective = (
            _checked_declared_path(declared, "backend toolchain", plan)
            if declared else expected_default
        )
        if effective != binding.toolchain_dir:
            raise _role_error(plan, "backend toolchain disagrees with eval binding")
    if hasattr(backend, "offline_shared_cache_dir"):
        declared = getattr(backend, "offline_shared_cache_dir")
        effective = (
            _checked_declared_path(declared, "backend shared cache", plan)
            if declared else None
        )
        if effective != binding.shared_cache_dir:
            raise _role_error(plan, "backend shared cache disagrees with eval binding")
    return binding


def _build_policy_args(
    plan: EvalOfflinePolicyPlan,
    backend: Any,
    role_name: str,
    variables: Mapping[str, Any],
) -> list[str]:
    """Build the current common policy from one immutable operator binding."""
    if not isinstance(variables, Mapping):
        raise _role_error(plan, "role variables must be a mapping")
    binding = _binding_for_backend(plan, backend, role_name, variables)
    state = binding.state_dir
    toolchain = binding.toolchain_dir
    shared_cache = binding.shared_cache_dir
    runtime_roots = binding.runtime_read_roots
    assert toolchain is not None

    work = os.path.join(state, "work")
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
        "HUGGINGFACE_HUB_CACHE": os.path.join(hf, "hub"),
        "TRANSFORMERS_CACHE": os.path.join(hf, "transformers"),
        "HF_XET_CACHE": os.path.join(hf, "xet"),
        "HF_ASSETS_CACHE": os.path.join(hf, "assets"),
    }
    try:
        for path in dict.fromkeys((work, *env.values())):
            _ensure_owned_chain(work, path)
    except EvalOfflinePolicyError as error:
        raise _role_error(plan, str(error)) from error

    for key, expected in (
        ("toolchain_dir", toolchain),
        ("shared_cache_dir", shared_cache),
    ):
        declared = variables.get(key)
        if not declared:
            continue
        supplied = _checked_declared_path(declared, key, plan)
        if expected is None or supplied != expected:
            raise _role_error(
                plan, f"offline {key} does not match the operator binding")

    read_roots: list[str] = []
    writable_roots: list[str] = []
    workspace = os.path.join(state, "workspace")
    workdir = variables.get("workdir")
    if workdir:
        supplied = _checked_declared_path(workdir, "workdir", plan)
        if not _path_is_within(work, supplied):
            raise _role_error(plan, "offline workdir escapes state/work")
        try:
            _ensure_owned_chain(work, supplied)
        except EvalOfflinePolicyError as error:
            raise _role_error(plan, str(error)) from error
    workspace_dir = variables.get("workspace_dir")
    if workspace_dir:
        supplied = _checked_declared_path(workspace_dir, "workspace_dir", plan)
        if supplied != workspace:
            raise _role_error(
                plan, "offline workspace_dir does not match state/workspace")
        try:
            _ensure_owned_chain(workspace, workspace)
        except EvalOfflinePolicyError as error:
            raise _role_error(plan, str(error)) from error
        writable_roots.append(workspace)

    if role_name in {"discharge", "judger"}:
        if _path_is_within(work, toolchain):
            try:
                _ensure_owned_chain(work, toolchain)
            except EvalOfflinePolicyError as error:
                raise _role_error(plan, str(error)) from error
        else:
            if _paths_overlap(state, toolchain):
                raise _role_error(
                    plan, "operator toolchain overlaps the isolated state")
            try:
                _ensure_owned_chain(toolchain, toolchain)
            except EvalOfflinePolicyError as error:
                raise _role_error(plan, str(error)) from error
        if role_name == "judger":
            read_roots.append(toolchain)
        elif not _path_is_within(work, toolchain):
            writable_roots.append(toolchain)

    if role_name == "discharge" and shared_cache is not None:
        if _path_is_within(work, shared_cache):
            try:
                _ensure_owned_chain(work, shared_cache)
            except EvalOfflinePolicyError as error:
                raise _role_error(plan, str(error)) from error
        else:
            if _paths_overlap(state, shared_cache):
                raise _role_error(
                    plan, "operator shared cache overlaps the isolated state")
            try:
                _ensure_owned_chain(shared_cache, shared_cache)
            except EvalOfflinePolicyError as error:
                raise _role_error(plan, str(error)) from error
            writable_roots.append(shared_cache)

    for runtime_root in runtime_roots:
        for writable_root in (work, *writable_roots):
            if _paths_overlap(runtime_root, writable_root):
                raise _role_error(
                    plan, "runtime read root overlaps an offline writable root")

    if role_name == "discharge" and shared_cache is not None:
        env.update({
            "UV_CACHE_DIR": os.path.join(shared_cache, "uv"),
            "UV_LINK_MODE": "hardlink",
            "PIP_CACHE_DIR": os.path.join(shared_cache, "pip"),
            "npm_config_cache": os.path.join(shared_cache, "npm"),
            "CARGO_HOME": os.path.join(shared_cache, "cargo"),
            "GOMODCACHE": os.path.join(shared_cache, "gomod"),
        })

    path_parts = []
    if role_name in {"discharge", "judger"}:
        path_parts.append(os.path.join(toolchain, "bin"))
    path_parts.extend(
        root for root in runtime_roots if os.path.basename(root) == "bin")
    path_parts.extend(("/usr/bin", "/bin", "/usr/sbin", "/sbin"))
    env["PATH"] = os.pathsep.join(dict.fromkeys(path_parts))
    env_table = "{" + ",".join(
        f"{key}={_toml_string(value)}" for key, value in env.items()) + "}"
    filesystem = "{" + ",".join((
        '":root"="deny"',
        '":minimal"="read"',
        f"{_toml_string(state)}=\"read\"",
        f"{_toml_string(work)}=\"write\"",
        *(f"{_toml_string(root)}=\"read\""
          for root in dict.fromkeys((*runtime_roots, *read_roots))),
        *(f"{_toml_string(root)}=\"write\""
          for root in dict.fromkeys(writable_roots)),
        '"~/.cache"="deny"',
        '"~/Library/Caches"="deny"',
        '"~/.huggingface"="deny"',
        '"~/.codex"="deny"',
    )) + "}"
    return [
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--ephemeral",
        "-c", f'default_permissions="{_OFFLINE_PROFILE}"',
        "-c", 'approval_policy="never"',
        "-c", f'permissions.{_OFFLINE_PROFILE}.extends=":read-only"',
        "-c", f"permissions.{_OFFLINE_PROFILE}.filesystem={filesystem}",
        "-c", f"permissions.{_OFFLINE_PROFILE}.network.enabled=true",
        "-c", ("tools.web_search=true" if _search_enabled(
            plan, backend, role_name) else "tools.web_search=false"),
        # Match the native offline policy: ignored user config cannot
        # provide the top-level setting that actually removes web search.
        *(() if _search_enabled(plan, backend, role_name)
          else ("-c", 'web_search="disabled"')),
        "-c", "features.skip_host_skill_discovery=true",
        "-c", "suppress_unstable_features_warning=true",
        "-c", "project_doc_max_bytes=0",
        "-c", "mcp_servers={}",
        *[
            item
            for feature in _DISABLED_FEATURES
            for item in ("-c", f"features.{feature}=false")
        ],
        "-c", 'shell_environment_policy.inherit="core"',
        "-c", f"shell_environment_policy.set={env_table}",
    ]


_INSTALL_LOCK = threading.Lock()
_ACTIVE_INSTALLS: "weakref.WeakKeyDictionary[types.ModuleType, object]" = (
    weakref.WeakKeyDictionary())


def _replacement_for(plan: EvalOfflinePolicyPlan):
    def _execution_policy_args(self, role_name: str,
                               variables: Mapping[str, Any]) -> list[str]:
        return _build_policy_args(plan, self, role_name, variables)

    _execution_policy_args.__name__ = "_execution_policy_args"
    _execution_policy_args.__qualname__ = "CodexBackend._execution_policy_args"
    return _execution_policy_args


@contextmanager
def apply_eval_offline_policy(
    plan: EvalOfflinePolicyPlan,
) -> Iterator[EvalOfflinePolicyPlan]:
    """Install one preflighted plan and restore the exact original method."""
    if not isinstance(plan, EvalOfflinePolicyPlan):
        raise EvalOfflinePolicyError("plan must be an EvalOfflinePolicyPlan")
    module = plan.module
    backend_class = plan.backend_class
    replacement = _replacement_for(plan)
    with _INSTALL_LOCK:
        if module in _ACTIVE_INSTALLS:
            raise EvalOfflinePolicyError(
                "an eval offline policy is already installed for this module")
        if inspect.getattr_static(
                backend_class, "_execution_policy_args", None) is not plan.original_method:
            raise EvalOfflinePolicyError(
                "backend policy method changed after the plan was built")
        _ACTIVE_INSTALLS[module] = replacement
        try:
            setattr(backend_class, "_execution_policy_args", replacement)
        except BaseException:
            del _ACTIVE_INSTALLS[module]
            raise

    body_error: Optional[BaseException] = None
    try:
        yield plan
    except BaseException as error:
        body_error = error
    restore_error: Optional[BaseException] = None
    try:
        with _INSTALL_LOCK:
            current = inspect.getattr_static(
                backend_class, "_execution_policy_args", None)
            try:
                setattr(
                    backend_class, "_execution_policy_args", plan.original_method)
            except BaseException as error:
                restore_error = error
            else:
                if current is not replacement:
                    restore_error = EvalOfflinePolicyError(
                        "backend policy method changed during overlay use")
            _ACTIVE_INSTALLS.pop(module, None)
    except BaseException as error:
        restore_error = restore_error or error

    if body_error is not None and restore_error is not None:
        raise BaseExceptionGroup(
            "eval offline policy body and restoration both failed",
            [body_error, restore_error])
    if body_error is not None:
        raise body_error.with_traceback(body_error.__traceback__)
    if restore_error is not None:
        raise restore_error


@contextmanager
def eval_offline_policy(
    module: types.ModuleType,
    *,
    bindings: Iterable[OfflinePolicyBinding],
) -> Iterator[EvalOfflinePolicyPlan]:
    """Convenience context manager combining read-only plan and install."""
    plan = plan_eval_offline_policy(module, bindings=bindings)
    with apply_eval_offline_policy(plan):
        yield plan


@contextmanager
def apply_eval_online_policy(
    plan: EvalOnlinePolicyPlan,
) -> Iterator[EvalOnlinePolicyPlan]:
    """Install one live-world eval plan and restore the original method."""
    if not isinstance(plan, EvalOnlinePolicyPlan):
        raise EvalOfflinePolicyError("plan must be an EvalOnlinePolicyPlan")
    module = plan.module
    backend_class = plan.backend_class
    replacement = _replacement_for(plan)
    with _INSTALL_LOCK:
        if module in _ACTIVE_INSTALLS:
            raise EvalOfflinePolicyError(
                "an eval policy is already installed for this module")
        if inspect.getattr_static(
                backend_class, "_execution_policy_args", None) is not plan.original_method:
            raise EvalOfflinePolicyError(
                "backend policy method changed after the plan was built")
        _ACTIVE_INSTALLS[module] = replacement
        try:
            setattr(backend_class, "_execution_policy_args", replacement)
        except BaseException:
            del _ACTIVE_INSTALLS[module]
            raise

    body_error: Optional[BaseException] = None
    try:
        yield plan
    except BaseException as error:
        body_error = error
    restore_error: Optional[BaseException] = None
    try:
        with _INSTALL_LOCK:
            current = inspect.getattr_static(
                backend_class, "_execution_policy_args", None)
            try:
                setattr(
                    backend_class, "_execution_policy_args", plan.original_method)
            except BaseException as error:
                restore_error = error
            else:
                if current is not replacement:
                    restore_error = EvalOfflinePolicyError(
                        "backend policy method changed during overlay use")
            _ACTIVE_INSTALLS.pop(module, None)
    except BaseException as error:
        restore_error = restore_error or error

    if body_error is not None and restore_error is not None:
        raise BaseExceptionGroup(
            "eval online policy body and restoration both failed",
            [body_error, restore_error])
    if body_error is not None:
        raise body_error.with_traceback(body_error.__traceback__)
    if restore_error is not None:
        raise restore_error


@contextmanager
def eval_online_policy(
    module: types.ModuleType,
    *,
    bindings: Iterable[OfflinePolicyBinding],
) -> Iterator[EvalOnlinePolicyPlan]:
    """Plan and temporarily install the isolated live-world eval policy."""
    plan = plan_eval_online_policy(module, bindings=bindings)
    with apply_eval_online_policy(plan):
        yield plan
