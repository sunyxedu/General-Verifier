"""Verifier roles as Claude Agent SDK agents.

Why agents rather than single-shot completions: most verifier roles genuinely
need to act, not just emit text. The oracle enumerator must probe what is
installed; the discharge executor must run the check; the entailment auditor
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

from .roles import RoleError, validate
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

class _TransientSDKError(RuntimeError):
    """Infrastructure failure worth backing off and retrying (internal)."""


def to_json_schema(spec: Any) -> Dict[str, Any]:
    """Mini-DSL (roles.validate) -> JSON Schema draft-07.

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
        """Same contract as roles.Role.call: two attempts, then RoleError."""
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
    """Runs AgentRoles through the Claude Agent SDK.

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
        """Backend entry point (same signature as FixtureBackend.complete).

        Loop-safe sync wrapper: callable from plain sync code AND from code
        executing inside a running event loop (e.g. an MCP tool handler in the
        sidecar) — in the latter case the coroutine runs on a private loop in
        a worker thread, because asyncio.run would raise on a running loop.
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
        """One role run with backoff on transient infrastructure failures.
        Immediate re-prompts against a rate limit are useless (the Role-level
        retry exists for GENERATION failures); infrastructure failures back
        off and retry here, and only then surface as RoleError."""
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

        self.transcripts.append(
            {
                "role": role.name,
                "session_id": getattr(result, "session_id", None),
                "subtype": getattr(result, "subtype", None),
                "num_turns": getattr(result, "num_turns", None),
                "total_cost_usd": getattr(result, "total_cost_usd", None),
                "usage": getattr(result, "usage", None),
                "api_error_status": getattr(result, "api_error_status", None),
                "stderr_tail": list(stderr_tail),
            }
        )

        if result is not None and result.structured_output is not None \
                and result.subtype == "success":
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
        if status is None and error is not None \
                and "error result: success" in str(error):
            # the CLI's opaque placeholder — usually a rate limit; retrying
            # then failing also covers the non-transient (e.g. auth) case
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
            output_format={
                "type": "json_schema",
                "schema": to_json_schema(role.schema),
            },
            max_buffer_size=64 * 1024 * 1024,  # default 1MB truncates long tool output
            stderr=stderr_cb,
        )


# ---- role definitions --------------------------------------------------------
# Templates stay generic. Domain content arrives at runtime: from the environment
# the agent can probe, from the model's latent knowledge, and from retrieved
# memory — never from text written here.

OUTCOMES = ("holds", "refuted", "failed")

COMPILE_OBLIGATIONS_AGENT = AgentRole(
    name="obligations",
    system=(
        "You are the obligation compiler. You see only the claim's statement, "
        "not its argument (answer-blind). You may read files in the environment "
        "to understand the task context, but you cannot execute anything.\n"
        "A corollary of answer-blindness: obligations can only be conditions "
        "that are \"necessary for the claim to be true and checkable against "
        "the world\" (independently verifiable by computation, execution, "
        "formalization, or similar means); requirements about how the argument "
        "is written (e.g. \"the proof must perform a certain rewrite or cover "
        "a certain case\") are outside your remit — every intermediate step of "
        "the argument enters the graph as its own claim and is verified "
        "separately. Your output will be checked exhaustively, item by item, "
        "so prefer few and on-point over many and uncheckable."
    ),
    template=(
        "Claim statement: {claim_text}\n"
        "Historical obligation patterns (retrieval hits, may be empty): {memory_hits}\n\n"
        "List the conditions (obligations) that are \"necessary for the claim to "
        "be true\" and independently checkable without seeing any argument. "
        "For each obligation give:\n"
        "  text: the obligation content (a world-level condition about the "
        "conclusion itself)\n"
        "  failure_looks_like: what the failure evidence concretely looks like "
        "if this obligation does not hold\n"
        "(it must point to a concrete, presentable artifact; if you cannot "
        "give one, leave an empty string)."
    ),
    schema={"obligations": [{"text": str, "failure_looks_like": str}]},
    tools=["Read", "Grep", "Glob"],
)

ENUMERATE_ORACLES_AGENT = AgentRole(
    name="oracles",
    system=(
        "You are the oracle enumerator. Imagining methods is your job; judging "
        "existence belongs to probing: whether an instrument is ready must be "
        "confirmed by actually probing the environment (e.g. using Bash to "
        "check that commands and libraries exist), never by assumption. "
        "When an instrument is not ready but can be installed, install it and "
        "confirm — installation is a one-time cost, not a boundary."
    ),
    template=(
        "Claim: {claim_text}\n"
        "Obligation: {obligation_text}\n"
        "Claim features: {features}\n"
        "Methods known to work (may be empty): {registry}\n\n"
        "Enumerate all methods that can check this obligation, ordered by "
        "evidence strength, strongest first. For each method give:\n"
        "  procedure: concretely how to do it (including what tool/library to use)\n"
        "  artifact: what auditable evidence the method produces\n"
        "  failure_looks_like: what the failure evidence looks like when this "
        "method finds a problem"
    ),
    schema={
        "methods": [
            {
                "procedure": str,
                "artifact": str,
                "failure_looks_like": str,
            }
        ]
    },
    tools=["Bash", "Read"],
)

GENERATE_VARIANTS_AGENT = AgentRole(
    name="variants",
    system=(
        "You are the variant generator. You generate \"plausible-but-wrong "
        "variants\" and must mechanically confirm that each variant is indeed "
        "false (by actually running a minimal check) — otherwise you would "
        "wrongly condemn the verification methods. If you cannot confirm one, "
        "do not submit it."
    ),
    template=(
        "Original claim: {claim_text}\nObligation: {obligation_text}\n"
        "Historical attack patterns (retrieval hits, may be empty): {memory_hits}\n\n"
        "Generate {n} wrong variants: structurally similar to the original "
        "claim, plausible at first glance, but actually false.\n"
        "For each variant give:\n"
        "  text: the variant statement\n"
        "  disproof: the evidence confirming it is false from the check you "
        "actually ran (command and output summary)"
    ),
    schema={"variants": [{"text": str, "disproof": str}]},
    tools=["Bash", "Write", "Read"],
)

COVERAGE_PATCH_AGENT = AgentRole(
    name="coverage_patch",
    system=(
        "You are the obligation patcher. A variant confirmed false was not "
        "caught by any existing obligation — this is a coverage hole in the "
        "obligation set. You must write a new obligation that catches this "
        "class of error. The obligation must be a condition that is \"necessary "
        "for the claim to be true and checkable against the world\"; it must "
        "not target how the argument is written."
    ),
    template=(
        "Original claim: {claim_text}\nVariant that slipped through: {variant_text}\n"
        "Evidence that this variant is false: {disproof}\n\n"
        "Write a new obligation that catches this class of error:\n"
        "  text: the obligation content (a world-level necessary condition)\n"
        "  failure_looks_like: what the failure evidence concretely looks like "
        "when this obligation does not hold"
    ),
    schema={"text": str, "failure_looks_like": str},
    tools=[],          # a pure text task — same name serves both backends
    max_turns=4,
)

VALIDATE_VARIANT_AGENT = AgentRole(
    name="validate_variant",
    system=(
        "You are the exam-question quality inspector. A \"wrong variant\" exam "
        "question let every candidate method pass — either all the methods "
        "share a blind spot, or the variant is not actually false (the "
        "generator was wrong). You must actually rerun the check to confirm "
        "whether it is true or false; do not judge from text alone. When in "
        "doubt, rule actually_false=yes (better to wrongly fault a method "
        "than to let a blind spot through)."
    ),
    template=(
        "Original claim: {claim_text}\nTargeted obligation: {obligation_text}\n"
        "Variant: {variant_text}\nGenerator's evidence of falsity (disproof): {disproof}\n\n"
        "Actually run the check to confirm whether this variant really is "
        "false, and false precisely by violating this obligation.\n"
        "  actually_false: yes | no\n"
        "  reason: your checking process and conclusion"
    ),
    schema={"actually_false": ("yes", "no"), "reason": str},
    tools=["Bash", "Read", "Write"],
)

DISCHARGE_AGENT = AgentRole(
    name="discharge",
    system=(
        "You are the discharge executor. You carry out the check by the given "
        "method and produce an auditable artifact. Crashes, timeouts, and "
        "unparseable output are all honestly recorded as failed — never "
        "converted into a pass."
    ),
    template=(
        "Claim: {claim_text}\nObligation: {obligation_text}\n"
        "Bound method: {procedure}\nWorking directory: {workdir}\n"
        "Toolchain directory: {toolchain_dir} (install heavyweight "
        "dependencies and project scaffolding here, shared and reused across "
        "checks; the working directory holds only this check's scripts and "
        "outputs)\n\n"
        "Run the check. Write the check script and its full output to disk in "
        "the working directory (they are the artifact).\n"
        "  outcome: holds (the obligation holds) | refuted (a counterexample "
        "was found) | failed (the check itself did not run)\n"
        "  witness: for refuted give the concrete counterexample, for failed "
        "give the failure reason, otherwise an empty string\n"
        "  artifact: paths of the files written to disk (a comma-separated "
        "string, not a JSON array)"
    ),
    schema={"outcome": OUTCOMES, "witness": str, "artifact": str},
    tools=["Bash", "Read", "Write", "Grep"],
)

ENTAILMENT_AUDIT_AGENT = AgentRole(
    name="entailment",
    system=(
        "You are the entailment auditor (gate 2). You do not judge whether "
        "the claim is true, only whether this evidence logically discharges "
        "the obligation. Sampled evidence cannot entail a universal "
        "conclusion — such gaps must be pointed out. You must open the "
        "artifact and check it yourself (rerunning the verification when "
        "necessary); do not rely on summaries alone."
    ),
    template=(
        "Obligation: {obligation_text}\nClaimed failure pattern: {failure_looks_like}\n"
        "Submitted evidence (including artifact paths): {evidence}\n\n"
        "Check the evidence itself and judge whether it logically entails "
        "that the obligation holds.\n"
        "  entails: yes | no\n"
        "  reason: your reasoning, pointing out any gap in the evidence chain"
    ),
    schema={"entails": ("yes", "no"), "reason": str},
    tools=["Read", "Bash", "Grep"],
)

ALL_AGENT_ROLES = {
    role.name: role
    for role in (
        COMPILE_OBLIGATIONS_AGENT,
        ENUMERATE_ORACLES_AGENT,
        GENERATE_VARIANTS_AGENT,
        VALIDATE_VARIANT_AGENT,
        COVERAGE_PATCH_AGENT,
        DISCHARGE_AGENT,
        ENTAILMENT_AUDIT_AGENT,
    )
}
