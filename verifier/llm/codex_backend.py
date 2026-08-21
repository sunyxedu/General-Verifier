"""Verifier seats on the Codex CLI (ChatGPT-subscription models).

Same duck type as AgentBackend / FixtureBackend — `complete(role_name,
prompt, variables) -> payload` — so every seam that accepts a backend
(kernel seats, the extractor, tests) takes this one unchanged. Each call
is one `codex exec`: the role's system text and rendered template become
the instruction, and the final assistant message must be a JSON object
matching the role's schema; it is parsed and re-validated by the same
mini-DSL gate as always, so a malformed generation surfaces as RoleError
and retries through AgentRole.call exactly like the SDK path.

Capability boundaries: every seat runs in codex's YOLO mode
(--dangerously-bypass-approvals-and-sandbox, user directive) — approvals
and sandboxing off, so exec-mode runs never stall on a permission stop.
The SDK path enforces the non-executing seats' no-execution palette as a
hard boundary; here it holds only at the prompt level ("you cannot
execute anything"), which is a fidelity gap to keep in mind when
comparing backends.

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
        out = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        out.close()
        cmd = ["codex", "exec", "--model", self.model,
               "--skip-git-repo-check",
               "--dangerously-bypass-approvals-and-sandbox",
               "-c", f"model_reasoning_effort={self.effort}",
               "-o", out.name, "--", instruction]
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
