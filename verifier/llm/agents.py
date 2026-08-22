"""
Verifier roles as Claude Agent SDK agents.

Why agents rather than single-shot completions: most verifier roles genuinely
need to act, not just emit text. The oracle enumerator must probe what is
installed; the discharge executor must run the check; the judger
must open the artifact it is judging. A role that cannot touch the world can
only give an opinion — which is the very thing this system refuses to trust.

Structured output uses the SDK's official mechanism (`output_format` with a
JSON schema): the harness validates the agent's final answer against the
schema and re-prompts on mismatch; the validated object arrives on
`ResultMessage.structured_output`. A run that ends without one is a FAILED
generation and surfaces as RoleError — never papered over. We still run our
own `validate()` on what comes back: the SDK's gate is convenience, ours is
the contract.

Capability boundaries are enforced with `tools` (the tool palette), not
`allowed_tools`: per the SDK docs, allowed_tools only pre-approves calls and
does not restrict the palette — under bypassPermissions an agent could still
reach Bash. The palette is part of a role's identity (the obligation compiler
being unable to execute anything is what makes it answer-blind), so it must
be a hard boundary.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import prompts, provider
from .schema import RoleError, validate
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

class _TransientSDKError(RuntimeError):
    """Infrastructure failure worth backing off and retrying (internal)."""


def to_json_schema(spec: Any) -> Dict[str, Any]:
    """
    Mini-DSL (schema.validate) -> JSON.

    One schema source of truth: the same spec drives the SDK's output gate
    and our own post-hoc validate().  str -> string; (a, b, ...) -> string
    enum; [x] -> array of x; {k: v} -> object with all keys required.
    """
    if spec is str:
        return {"type": "string"}
    if spec is bool:
        return {"type": "boolean"}
    if spec is int:
        return {"type": "integer"}
    if spec is float:
        return {"type": "number"}
    if isinstance(spec, tuple):
        return {"type": "string", "enum": list(spec)}
    if isinstance(spec, list):
        return {"type": "array", "items": to_json_schema(spec[0])}
    if isinstance(spec, dict):
        return {
            "type": "object",
            "properties": {k: to_json_schema(v) for k, v in spec.items()},
            "required": list(spec.keys()),
            "additionalProperties": False,
        }
    raise ValueError(f"unsupported schema spec: {spec!r}")


@dataclass
class AgentRole:
    """A verifier role: generic prompt + output contract + permitted palette."""

    name: str
    template: str
    schema: Dict[str, Any]
    tools: List[str] = field(default_factory=list)
    model: Optional[str] = None
    max_turns: int = 20
    system: str = ""

    def render(self, variables: Dict[str, Any]) -> str:
        return self.template.format(**variables)

    def call(self, backend, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Two attempts (schema error fed back once), then RoleError."""
        prompt = self.render(variables)
        last_error: Optional[RoleError] = None
        for _attempt in range(2):
            try:
                payload = backend.complete(self.name, prompt, variables)
                validate(self.schema, payload)
                return payload
            except RoleError as err:
                last_error = err
                prompt = prompt + f"\n\n[previous attempt failed; fix it and retry] {err}"
        raise RoleError(f"{self.name}: failed twice: {last_error}")


class AgentBackend:
    """
    Runs AgentRoles through the Claude Agent SDK.

    One SDK session per call: roles are stateless by construction, which is
    what keeps the auditors independent of the things they audit.

    permission_mode defaults to bypassPermissions because roles run headless
    (a permission prompt would silently deny). This is safe to the extent the
    palette is: restriction lives in `tools`, not in permissions.
    """

    def __init__(
        self,
        roles: Optional[Dict[str, AgentRole]] = None,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        permission_mode: str = "bypassPermissions",
    ):
        self.roles = dict(roles) if roles is not None else dict(ALL_AGENT_ROLES)
        self.cwd = cwd
        self.model = model
        self.permission_mode = permission_mode
        self.transcripts: List[Dict[str, Any]] = []

    def complete(self, role_name: str, prompt: str, variables: Dict[str, Any]) -> Any:
        """
        Backend entry point (same signature as FixtureBackend.complete).

        Loop-safe sync wrapper: callable from plain sync code AND from code executing inside a running event loop (e.g. an MCP tool handler in the MCP server).
        In the latter case the coroutine runs on a private loop in a worker thread, because asyncio.run would raise on a running loop.
        """
        role = self.roles.get(role_name)
        if role is None:
            raise RoleError(f"unknown agent role: {role_name}")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(role, prompt))
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.run(role, prompt)).result()

    # statuses worth waiting out (rate limit / overload / gateway); the CLI
    # sometimes reports these only as the opaque "error result: success"
    TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
    transient_delays = (0, 20, 60, 120)   # seconds before each attempt

    async def run(self, role: AgentRole, prompt: str) -> Dict[str, Any]:
        """
        One role run with backoff on transient infrastructure failures.
        Immediate re-prompts against a rate limit are useless (the Role-level
        retry exists for GENERATION failures); infrastructure failures back
        off and retry here, and only then surface as RoleError.
        """
        last: Optional[str] = None
        for delay in self.transient_delays:
            if delay:
                await asyncio.sleep(delay)
            try:
                return await self._attempt(role, prompt)
            except _TransientSDKError as err:
                last = str(err)
        raise RoleError(f"{role.name}: transient sdk failure persisted: {last}")

    async def _attempt(self, role: AgentRole, prompt: str) -> Dict[str, Any]:
        stderr_tail: List[str] = []

        def keep_stderr(line: str) -> None:
            stderr_tail.append(line)
            del stderr_tail[:-20]

        options = self._build_options(role, keep_stderr)

        result: Optional[ResultMessage] = None
        error: Optional[Exception] = None
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result = message
        except Exception as err:  # noqa: BLE001 - classified below
            error = err

        self.transcripts.append({
            "role": role.name,
            "session_id": getattr(result, "session_id", None),
            "subtype": getattr(result, "subtype", None),
            "num_turns": getattr(result, "num_turns", None),
            "total_cost_usd": getattr(result, "total_cost_usd", None),
            "usage": getattr(result, "usage", None),
            "api_error_status": getattr(result, "api_error_status", None),
            "stderr_tail": list(stderr_tail),
        })

        if result is not None and result.structured_output is not None and result.subtype == "success":
            payload = result.structured_output
            validate(role.schema, payload)  # our gate, independent of the SDK's
            return payload

        status = getattr(result, "api_error_status", None)
        detail = (
            f"subtype={getattr(result, 'subtype', None)}, "
            f"api_error_status={status}, "
            f"errors={getattr(result, 'errors', None)}, exception={error}"
        )
        if status in self.TRANSIENT_STATUSES:
            raise _TransientSDKError(f"{role.name}: {detail}")
        if status is None and error is not None and "error result: success" in str(error):
            # the CLI's opaque placeholder. Usually a rate limit; retrying
            # then failing also covers the non-transient case (e.g. auth).
            raise _TransientSDKError(f"{role.name}: {detail}")
        if error is not None:
            raise RoleError(f"{role.name}: sdk failure: {detail}") from error
        raise RoleError(f"{role.name}: no structured output ({detail})")

    def _build_options(self, role: AgentRole, stderr_cb) -> "ClaudeAgentOptions":
        system_prompt: Any = {"type": "preset", "preset": "claude_code"}
        if role.system:
            system_prompt["append"] = role.system
        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=role.model or self.model,
            tools=list(role.tools),            # hard palette (capability boundary)
            allowed_tools=list(role.tools),    # and pre-approved within it
            permission_mode=self.permission_mode,
            max_turns=role.max_turns,
            cwd=self.cwd,
            setting_sources=[],                # hermetic: no ambient project settings
            env=provider.claude_env(base={}),  # every seat runs on the
                                               # configured provider (SDK
                                               # merges over os.environ)
            output_format={
                "type": "json_schema",
                "schema": to_json_schema(role.schema),
            },
            max_buffer_size=64 * 1024 * 1024,  # default 1MB truncates long tool output
            stderr=stderr_cb,
        )


# role definitions
# Prompt text lives in prompts.py; this block is pure wiring: output contract
# (schema) + permitted palette (tools) + turn budget.

OUTCOMES = ("holds", "refuted", "failed", "unobtainable")

COMPILE_OBLIGATIONS_AGENT = AgentRole(
    name="obligations",
    system=prompts.OBLIGATIONS_SYSTEM,
    template=prompts.OBLIGATIONS_TEMPLATE,
    schema={"obligations": [{"text": str}]},
    tools=["Read", "Grep", "Glob"],
)

CRITIC_AGENT = AgentRole(
    name="critic",
    system=prompts.CRITIC_SYSTEM,
    template=prompts.CRITIC_TEMPLATE,
    schema={
        "verdict": ("pass", "insufficient"),
        "counterexample_world": str,
        "missing_condition": str,
    },
    tools=["Read", "Grep", "Glob"],   # answer-blind like the compiler; no execution
)

ENUMERATE_ORACLES_AGENT = AgentRole(
    name="oracles",
    system=prompts.ORACLES_SYSTEM,
    template=prompts.ORACLES_TEMPLATE,
    schema={
        "methods": [
            {
                "procedure": str,
                "artifact": str,
                "level": str,
                # evidence basis: what a successful run of this method
                # entitles downstream reliance to (see prompts.ORACLES_SYSTEM)
                "basis": ("world", "record"),
            }
        ]
    },
    tools=["Read", "Grep", "Glob", "WebSearch", "WebFetch"],
    max_turns=15,      # research takes room; still no Bash — no probing
)

DISCHARGE_AGENT = AgentRole(
    name="discharge",
    system=prompts.DISCHARGE_SYSTEM,
    template=prompts.DISCHARGE_TEMPLATE,
    schema={"outcome": OUTCOMES, "witness": str, "artifact": str},
    tools=["Bash", "Read", "Write", "Grep"],
)

JUDGER_AGENT = AgentRole(
    name="judger",
    system=prompts.JUDGER_SYSTEM,
    template=prompts.JUDGER_TEMPLATE,
    schema={"faithful": ("yes", "no"), "reason": str},
    tools=["Read", "Bash", "Grep"],
)

EXTRACT_CLAIMS_AGENT = AgentRole(
    name="extract",
    system=prompts.EXTRACT_SYSTEM,
    template=prompts.EXTRACT_TEMPLATE,
    # revises is required-but-empty like discharge's witness (the mini-DSL has
    # no optional keys); kinds exclude "effect". Effects come from the
    # ledger replay, never from the extractor.
    schema={
        "commits": [
            {
                "label": str,
                "kind": ("assertion", "final"),
                "text": str,
                "deps": [str],
                "revises": str,
            }
        ]
    },
    tools=["Read", "Grep", "Glob"],   # reconstruction, no execution
    max_turns=30,                     # long ledgers take reading room
)

ALL_AGENT_ROLES = {
    role.name: role
    for role in (
        COMPILE_OBLIGATIONS_AGENT,
        CRITIC_AGENT,
        ENUMERATE_ORACLES_AGENT,
        DISCHARGE_AGENT,
        JUDGER_AGENT,
        EXTRACT_CLAIMS_AGENT,
    )
}
