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
import contextlib
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import prompts
from .model_spans import (
    role_attempt, role_span, seat_acquired, seat_released,
    seat_wait_started, submit_with_context,
)
from .schema import RoleError, validate, field_name, optional_key
# The Claude Agent SDK backs the DEFAULT backend only; alternate backends
# (CodexBackend) never touch it. Imported lazily so that a seats=codex run
# does not depend on the SDK's import chain (mcp -> pydantic -> native
# extensions) even loading — the palette of the active backend, not the
# default one, should decide what must be importable.
def _sdk():
    import claude_agent_sdk
    return claude_agent_sdk

class _TransientSDKError(RuntimeError):
    """Infrastructure failure worth backing off and retrying (internal)."""


# Raster formats the CLI's Read tool renders as image content blocks. On a
# text-only provider such a block makes the whole request unroutable
# (e.g. OpenRouter: 404 "No endpoints found that support image input"), so
# under provider.TEXT_ONLY the Read is denied and the seat is told to reason
# from textual evidence instead.
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")


async def _deny_image_read(input_data, tool_use_id, context):
    tool_input = (input_data or {}).get("tool_input") or {}
    path = str(tool_input.get("file_path", ""))
    if path.lower().endswith(_IMAGE_EXTS):
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "text-only provider: image files cannot enter model "
                    "context here. Treat the image as an opaque artifact: "
                    "reason from its path, size, hash, and the surrounding "
                    "textual evidence; if the obligation hinges on unseen "
                    "image content, treat that method as unobtainable "
                    "rather than guessing."
                ),
            }
        }
    return {}


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
            "properties": {field_name(k): to_json_schema(v)
                           for k, v in spec.items()},
            "required": [field_name(k) for k in spec
                         if not optional_key(k)],
            "additionalProperties": False,
        }
    raise ValueError(f"unsupported schema spec: {spec!r}")


def _seat_limit() -> int:
    """Process-wide cap on in-flight seat calls, whatever the nesting of
    trial x claim x obligation parallelism above it. Env
    GV_MAX_CONCURRENT_SEATS (default 12); <= 0 disables the cap."""
    try:
        return int(os.environ.get("GV_MAX_CONCURRENT_SEATS", "12"))
    except ValueError:
        return 12


class _SeatGate:
    """A bounded semaphore sized lazily from the env, shared by every role
    in the process; ``slot()`` yields one in-flight permit."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sem: Optional[threading.BoundedSemaphore] = None
        self._size = 0

    def _current(self) -> Optional[threading.BoundedSemaphore]:
        limit = _seat_limit()
        if limit <= 0:
            return None
        with self._lock:
            if self._sem is None or self._size != limit:
                self._sem = threading.BoundedSemaphore(limit)
                self._size = limit
            return self._sem

    @contextlib.contextmanager
    def slot(self):
        sem = self._current()
        if sem is None:
            yield
            return
        sem.acquire()
        try:
            yield
        finally:
            sem.release()


SEAT_GATE = _SeatGate()


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
        # Start before acquiring SEAT_GATE: queueing is part of end-to-end
        # role latency, while each schema retry remains separately visible.
        with role_span(self.name):
            for attempt_index in range(1, 3):
                try:
                    with role_attempt(attempt_index):
                        wait_started = seat_wait_started()
                        with SEAT_GATE.slot():
                            acquired = seat_acquired(wait_started)
                            try:
                                payload = backend.complete(
                                    self.name, prompt, variables)
                            finally:
                                seat_released(acquired)
                        validate(self.schema, payload)
                    return payload
                except RoleError as err:
                    last_error = err
                    prompt = (
                        prompt
                        + f"\n\n[previous attempt failed; fix it and retry] {err}"
                    )
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
            return submit_with_context(
                pool, asyncio.run, self.run(role, prompt)).result()

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

        sdk = _sdk()
        result = None
        error: Optional[Exception] = None
        try:
            async for message in sdk.query(prompt=prompt, options=options):
                if isinstance(message, sdk.ResultMessage):
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

    def _build_options(self, role: AgentRole, stderr_cb):
        from . import provider

        system_prompt: Any = {"type": "preset", "preset": "claude_code"}
        if role.system:
            system_prompt["append"] = role.system
        sdk = _sdk()
        return sdk.ClaudeAgentOptions(
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
            hooks=(
                {"PreToolUse": [sdk.HookMatcher(matcher="Read",
                                            hooks=[_deny_image_read])]}
                if provider.TEXT_ONLY else None
            ),
        )


# role definitions
# Prompt text lives in prompts.py; this block is pure wiring: output contract
# (schema) + permitted palette (tools) + turn budget.

OUTCOMES = ("holds", "refuted", "failed", "unobtainable")

COMPILE_OBLIGATIONS_AGENT = AgentRole(
    name="obligations",
    system=prompts.OBLIGATIONS_SYSTEM,
    template=prompts.OBLIGATIONS_TEMPLATE,
    # required_basis is the obligation's evidence floor (world | record),
    # enforced mechanically: methods below it never bind, and evidence
    # below it never settles the node
    # severity is the modality of the source clause (must | should):
    # only a must-obligation's refutation refutes the claim. clause is
    # the verbatim span it was compiled from. refutable_by_absence marks
    # a presence requirement (the only shape an absence witness may
    # refute). Optional at the gate; the kernel defaults each to the
    # fail-closed reading (must, "", False).
    schema={"obligations": [{"text": str,
                             "required_basis": ("world", "record"),
                             "severity?": ("must", "should"),
                             "clause?": str,
                             "refutable_by_absence?": bool}]},
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
                # evidence strength: decisive = a decision procedure for
                # the obligation as written (the only tier that may
                # refute); bounded/analytic = weaker fallbacks that need
                # bridges and an audited degradation to bind
                "strength": ("decisive", "bounded", "analytic"),
                # the bridging assumptions that transport a non-decisive
                # method's positive verdict to the obligation; [] for
                # decisive (the mini-DSL has no optional keys)
                "bridges": [str],
                # Optional typed executor binding.  The oracle still owns
                # method choice; these fields merely let a registered,
                # versioned adapter execute that exact method without an
                # executor seat.  Partial/unknown bindings fall back to the
                # agent executor rather than affecting the verdict.
                "adapter?": str,
                "adapter_version?": int,
                "adapter_arguments?": str,
                "adapter_inputs?": [{
                    "ref": str,
                    "provenance": ("task_given", "delivered", "ledger",
                                   "verifier", "public"),
                    "role": ("subject", "input", "authority"),
                }],
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
    # assumed_premises: premise claim ids (from the workspace index)
    # whose CONTENT the check assumed instead of re-establishing; [] when
    # the check stands on its own. Gated mechanically at the kernel to
    # the claim's own dependency closure.
    # witness_kind: what a refutation is made of (execution |
    # world_state | record_contradiction | absence | none) — the kernel
    # admits only positive kinds, and absence only against a presence
    # requirement; missing reads as none (a refutation then settles
    # nothing). inputs: every substantive source the check consumed,
    # with its provenance class and the role it played — a delivered
    # artifact in the authority role makes a positive verdict circular
    # and the kernel refuses it.
    schema={"outcome": OUTCOMES, "witness": str, "artifact": str,
            "assumed_premises": [str],
            "witness_kind?": ("execution", "world_state",
                              "record_contradiction", "absence", "none"),
            "inputs?": [{"ref": str,
                         "provenance": ("task_given", "delivered",
                                        "ledger", "verifier", "public"),
                         "role": ("subject", "input", "authority")}]},
    tools=["Bash", "Read", "Write", "Grep"],
)

JUDGER_AGENT = AgentRole(
    name="judger",
    system=prompts.JUDGER_SYSTEM,
    template=prompts.JUDGER_TEMPLATE,
    schema={
        "sufficient": ("yes", "no"),
        "faithful": ("yes", "no"),
        # the third signature: for a non-decisive method, whether every
        # stronger tier was genuinely structurally unavailable (audited
        # against the degradation record); n/a for decisive methods
        "degradation": ("justified", "unjustified", "n/a"),
        "reason": str,
    },
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
