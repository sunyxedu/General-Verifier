"""Public-evidence execution policy for the cybersecurity role family.

This is a seat boundary, not a second verifier or a benchmark adapter. The
normal Codex backend still runs each agent with its original wire contract.
"""
from __future__ import annotations

import os
from pathlib import Path
import tomllib

from verifier.llm.codex_backend import (
    CodexBackend, _ensure_verifier_owned_chain, _toml_string,
)
from verifier.llm.schema import RoleError


PLANNER_ROLES = frozenset({"obligations", "critic", "oracles", "sufficiency"})


class CyberCodexBackend(CodexBackend):
    """Read declared public inputs; write only verifier-owned scratch.

    No shell network grant is inferred from a task or model response. Live
    public source is read-only for execution/audit seats, never a compiler
    answer source. A custom host backend can supply other authorized tools.
    """

    DENIED_ROOTS = ("/evaluator", "/codex-home", "/gv", "/controller",
                    "/root/.codex", "/home/node/.codex")

    def __init__(self, *, project_root: str | None = None,
                 runtime_inventory=None, **kwargs):
        super().__init__(**kwargs)
        self.project_root = (os.path.abspath(os.fspath(project_root))
                             if project_root is not None else None)
        self.search_roles = frozenset()
        self.runtime_inventory = runtime_inventory
        if runtime_inventory is not None:
            from .runtime import CyberRuntimeInventory
            if not isinstance(runtime_inventory, CyberRuntimeInventory):
                raise RoleError("runtime inventory must be host-selected")
            runtime_inventory.as_inventory()

    def _is_offline_execution(self, role_name, variables):
        # Select a bounded shell, without misrepresenting live-world prompts.
        return True

    def _execution_policy_args(self, role_name, variables):
        state = Path(self._offline_state_dir(variables))
        allowed = (state / "work", state / "workspace")
        for key in ("workdir", "workspace_dir"):
            value = variables.get(key)
            if value:
                path = Path(os.path.abspath(os.fspath(value)))
                if not any(path == root or path.is_relative_to(root) for root in allowed):
                    raise RoleError("verifier writable paths must remain in owned scratch")
        # Preserve the PR's operator-bound shared toolchain/cache support.
        # Method variables cannot select these grants; the base backend also
        # validates their exact match to its host-supplied configuration.
        toolchain, cache = self._offline_tool_roots(str(state))
        protected = [Path(root) for root in self.DENIED_ROOTS]
        protected.extend(Path(os.path.expanduser("~")) / name
                         for name in (".codex", ".ssh", ".aws"))
        if any(Path(root).is_relative_to(private)
               for root in (toolchain, cache) if root is not None
               for private in protected):
            raise RoleError("operator tools overlap a private evidence boundary")
        for key, expected in (("toolchain_dir", toolchain), ("shared_cache_dir", cache)):
            value = variables.get(key)
            if value and (expected is None or os.path.abspath(os.fspath(value)) != expected):
                raise RoleError("verifier tools must match operator-owned scratch")
        args = super()._execution_policy_args(role_name, variables)
        filesystem = [i for i, value in enumerate(args)
                      if value.startswith("permissions.") and ".filesystem=" in value]
        network = [i for i, value in enumerate(args)
                   if value.startswith("permissions.") and ".network.enabled=" in value]
        if (len(filesystem) != 1 or len(network) != 1
                or "--dangerously-bypass-approvals-and-sandbox" in args
                or 'approval_policy="never"' not in args
                or not args[filesystem[0]].endswith("}")):
            raise RoleError("inherited verifier permissions did not fail closed")
        prefix, table = args[filesystem[0]].split("=", 1)
        grants = tomllib.loads("grants=" + table)["grants"]
        grants.update({path: "deny" for path in self.DENIED_ROOTS})
        grants[str(state / "verifier-home")] = "deny"
        args[network[0]] = args[network[0]].rsplit("=", 1)[0] + "=false"
        if self.runtime_inventory is not None:
            runtime = self.runtime_inventory.as_inventory()  # Revalidate dependency pins.
            root = Path(self.runtime_inventory.tool_root)
            launcher = Path(runtime["files"]["launcher"]["path"])
            boundaries = (*protected, state)
            if any(path == private or path.is_relative_to(private)
                   or private.is_relative_to(path)
                   for path in (root, launcher) for private in boundaries):
                raise RoleError("runtime tools overlap a private evidence boundary")
            if self.project_root is not None:
                project = Path(self.project_root)
                if any(path == project or path.is_relative_to(project)
                       or project.is_relative_to(path) for path in (root, launcher)):
                    raise RoleError("runtime tools overlap mutable task material")
            if role_name in {"discharge", "judger"}:
                # A fixed public execution tool is not a shell environment
                # override, a proc grant, or permission to alter tool files.
                grants[str(root)] = "read"
                grants[str(launcher)] = "read"
            else:
                # The fixed tools live under the minimal /usr read mount.
                # Omission would not hide them from answer-blind planners.
                grants[str(root)] = "deny"
                grants[str(launcher)] = "deny"
        if role_name in PLANNER_ROLES:
            # An allowlist survives new proof/telemetry transports. Retain
            # explicit denies as a machine-checkable statement of the boundary.
            hidden = ("events.jsonl", "claims.sqlite", "claims.sqlite-wal",
                      "claims.sqlite-shm", "record", "source_evidence",
                      "artifacts", "cyber", "work", "workspace",
                      "cybergym-observations", "hook_feed.jsonl", "_blind",
                      "verification_jobs.jsonl", "transport_calls.jsonl",
                      "role_calls.jsonl", "role_phases.jsonl")
            grants.update({str(state / path): "deny" for path in hidden})
            grants[str(state)] = "deny"
            for path in ("task.md", "baseline", "oracle_catalog"):
                grants[str(state / path)] = "read"
            if self.project_root is not None:
                grants[self.project_root] = "deny"
            write_scopes = list(allowed)
            if self.project_root is not None:
                write_scopes.append(Path(self.project_root))
            for path, grant in tuple(grants.items()):
                if grant == "write" and any(Path(path).is_relative_to(root)
                        for root in write_scopes):
                    grants[path] = "deny"
            blind = state / "_blind" / role_name
            _ensure_verifier_owned_chain(str(state), str(blind))
            grants[str(blind)] = "write"
            environment = [i for i, value in enumerate(args)
                           if value.startswith("shell_environment_policy.set=")]
            if len(environment) != 1:
                raise RoleError("inherited shell environment did not fail closed")
            env_index = environment[0]
            env = tomllib.loads(args[env_index])["shell_environment_policy"]["set"]
            old_home = str(state / "work" / "_offline_home")
            for key, value in env.items():
                if isinstance(value, str) and (value == old_home or value.startswith(old_home + os.sep)):
                    env[key] = str(blind / "home") + value[len(old_home):]
                    _ensure_verifier_owned_chain(str(state), env[key])
            args[env_index] = "shell_environment_policy.set={" + ",".join(
                f"{key}={_toml_string(value)}" for key, value in env.items()) + "}"
        elif self.project_root is not None and role_name in {"discharge", "judger"}:
            # The project is public evidence, not a write grant. More-specific
            # private roots retain precedence even if the project contains them.
            if not any(self.project_root == root or Path(self.project_root).is_relative_to(Path(root))
                       for root in self.DENIED_ROOTS):
                grants.setdefault(self.project_root, "read")
        args[filesystem[0]] = prefix + "={" + ",".join(
            f"{_toml_string(path)}={_toml_string(grant)}"
            for path, grant in grants.items()) + "}"
        return args

    def _execution_cwd(self, role_name, variables):
        if role_name in PLANNER_ROLES:
            state = self._offline_state_dir(variables)
            path = os.path.join(state, "_blind", role_name)
            _ensure_verifier_owned_chain(state, path)
            return path
        return super()._execution_cwd(role_name, variables)
