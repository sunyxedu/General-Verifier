"""Verifier seats on the Codex CLI (ChatGPT-subscription models).

Same duck type as AgentBackend / FixtureBackend — `complete(role_name,
prompt, variables) -> payload` — so every seam that accepts a backend
(kernel seats, the extractor, tests) takes this one unchanged. Each call
is one `codex exec`: the role's system text and rendered template become
the instruction, and the final assistant message must be a JSON object
matching the role's schema; it is parsed and re-validated by the same
mini-DSL gate as always, so a malformed generation surfaces as RoleError
and retries through AgentRole.call exactly like the SDK path.

Capability boundaries: semantic seats and live-world execution retain the
user-requested YOLO mode (--dangerously-bypass-approvals-and-sandbox), so
exec-mode runs never stall on a permission stop.  Offline evidence seats
(discharge and judger in captured/unavailable world modes) instead run in a
workspace profile with command network disabled, host caches denied, and a
fresh state-local HOME/cache/tmp.  That prevents a historical check from
silently becoming a replay against a mutable remote or a cache left by a
different run while preserving writes to the state's work/toolchain tree.

The SDK path enforces the non-executing seats' no-execution palette as a hard
boundary; for the Codex backend it holds only at the prompt level ("you cannot
execute anything"), which is a fidelity gap to keep in mind when comparing
backends.

Fidelity note: codex drives its own agent loop; per-turn budgets and the
exact tool set differ from the SDK seats. Scores from the two backends
are comparable to each other only within a backend.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional

from .agents import ALL_AGENT_ROLES, to_json_schema
from .schema import RoleError, validate

DEFAULT_MODEL = "gpt-5.6-luna"

_OFFLINE_EVIDENCE_ROLES = frozenset({"discharge", "judger"})
_OFFLINE_WORLD_MODES = frozenset({"captured", "unavailable"})
_OFFLINE_PROFILE = "gv-offline"


def _toml_string(value: str) -> str:
    """A JSON string is also a valid TOML basic string."""
    return json.dumps(value, ensure_ascii=False)


def extract_json(text: str) -> Any:
    """First JSON object in `text` (fences and prose tolerated)."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _end = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise RoleError(f"no JSON object in output: {text[:200]!r}")


class CodexBackend:
    """Runs verifier roles through `codex exec` on a subscription model."""

    def __init__(self, roles: Optional[Dict[str, Any]] = None,
                 cwd: Optional[str] = None, model: Optional[str] = None,
                 effort: str = "medium", timeout: int = 1800):
        self.roles = dict(roles) if roles is not None else dict(ALL_AGENT_ROLES)
        self.cwd = cwd
        self.model = model or DEFAULT_MODEL
        self.effort = effort
        self.timeout = timeout

    def _execution_policy_args(self, role_name: str,
                               variables: Dict[str, Any]) -> list[str]:
        """Return the CLI boundary for this seat and evidence context."""
        offline_evidence = (
            role_name in _OFFLINE_EVIDENCE_ROLES
            and variables.get("world_mode") in _OFFLINE_WORLD_MODES
        )
        if not offline_evidence:
            return ["--dangerously-bypass-approvals-and-sandbox"]

        # Offline score() constructs the backend with cwd=state_dir.  Keep a
        # fallback for direct callers that provide only the ledger path, but
        # fail closed rather than place an allegedly isolated HOME in an
        # unrelated process directory.
        state_dir = self.cwd
        if not state_dir and variables.get("ledger_path"):
            state_dir = os.path.dirname(
                os.path.abspath(str(variables["ledger_path"])))
        if not state_dir:
            raise RoleError(
                f"{role_name}: offline evidence execution needs a state cwd")
        state_dir = os.path.abspath(state_dir)

        home = os.path.join(state_dir, "work", "_offline_home")
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
        for path in dict.fromkeys(env.values()):
            os.makedirs(path, exist_ok=True)
        env_table = "{" + ",".join(
            f"{key}={_toml_string(value)}" for key, value in env.items()
        ) + "}"

        # Permission profiles and the legacy --sandbox flag do not compose:
        # selecting the profile through default_permissions is intentional.
        # The Codex process itself keeps its real HOME (and therefore auth);
        # shell_environment_policy applies to commands launched by the seat.
        return [
            "--ignore-user-config",
            "-c", f'default_permissions="{_OFFLINE_PROFILE}"',
            "-c", 'approval_policy="never"',
            "-c", f'permissions.{_OFFLINE_PROFILE}.extends=":workspace"',
            "-c", (
                f"permissions.{_OFFLINE_PROFILE}.filesystem="
                '{"~/.cache"="deny","~/Library/Caches"="deny",'
                '"~/.huggingface"="deny","~/.codex"="deny"}'
            ),
            "-c", f"permissions.{_OFFLINE_PROFILE}.network.enabled=false",
            "-c", 'shell_environment_policy.inherit="core"',
            "-c", f"shell_environment_policy.set={env_table}",
        ]

    def complete(self, role_name: str, prompt: str,
                 variables: Dict[str, Any]) -> Any:
        role = self.roles.get(role_name)
        if role is None:
            raise RoleError(f"unknown agent role: {role_name}")
        schema = json.dumps(to_json_schema(role.schema), ensure_ascii=False)
        instruction = (
            (role.system + "\n\n" if role.system else "")
            + prompt
            + "\n\nYour FINAL message must be exactly one JSON object "
              "matching this JSON Schema, with no other prose around it:\n"
            + schema)
        policy_args = self._execution_policy_args(role_name, variables)
        out = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        out.close()
        cmd = ["codex", "exec", "--model", self.model,
               "--skip-git-repo-check"]
        cmd.extend(policy_args)
        cmd.extend([
            "-c", f"model_reasoning_effort={self.effort}",
            "-o", out.name, "--", instruction,
        ])
        try:
            with open(os.devnull) as devnull:
                # stdin MUST be closed: codex otherwise waits on it forever
                proc = subprocess.run(
                    cmd, cwd=self.cwd, stdin=devnull, capture_output=True,
                    text=True, timeout=self.timeout)
            text = ""
            if os.path.exists(out.name):
                with open(out.name, encoding="utf-8", errors="replace") as f:
                    text = f.read().strip()
            if not text:
                text = (proc.stdout or "").strip()
            if not text:
                raise RoleError(
                    f"{role_name}: codex exec produced no output "
                    f"(rc={proc.returncode}, stderr={proc.stderr[-300:]!r})")
            payload = extract_json(text)
            validate(role.schema, payload)
            return payload
        except subprocess.TimeoutExpired:
            raise RoleError(
                f"{role_name}: codex exec timed out after {self.timeout}s")
        finally:
            try:
                os.remove(out.name)
            except OSError:
                pass
