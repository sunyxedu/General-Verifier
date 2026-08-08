"""Verifier roles as Claude Agent SDK agents.

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

from . import provider
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
        MCP server) — in the latter case the coroutine runs on a private loop in
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


# ---- role definitions --------------------------------------------------------
# Templates stay generic. Domain content arrives at runtime: from the environment
# the agent can probe, from the model's latent knowledge, and from retrieved
# memory — never from text written here.

OUTCOMES = ("holds", "refuted", "failed", "unobtainable")

COMPILE_OBLIGATIONS_AGENT = AgentRole(
    name="obligations",
    system=(
        "You are the obligation compiler. You see the claim's statement and "
        "its position in the claim graph — but never the solver's argument "
        "(answer-blind). You may read files in the environment to understand "
        "the task context, but you cannot execute anything.\n"
        "Graph rule: premises marked VERIFIED were independently established "
        "by this system — treat them as facts and prefer RESIDUAL "
        "obligations: the world-checkable conditions that, together with "
        "those verified premises, establish this claim (do not re-demand "
        "what a verified premise already settles). Premises not marked "
        "verified are context only — never rely on them.\n"
        "You are also given the WHOLE claim graph. Search it for statements "
        "already proven anywhere (status verified — claims and obligation "
        "nodes alike): if a condition you would demand is already "
        "established there, rely on it instead of re-demanding it. And when "
        "a condition you emit matches an already-proven statement word for "
        "word, reuse that exact wording — identical text is what lets the "
        "graph share the node and transfer the settled verdict for free.\n"
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
        "Verified premises from the claim graph (established facts — residual "
        "obligations may rely on them): {verified_deps}\n"
        "Declared but not-yet-verified premises (context only — do NOT rely "
        "on them): {pending_deps}\n"
        "The whole claim graph (kind · status · text): {graph}\n"
        "Historical obligation patterns (retrieval hits, may be empty): {memory_hits}\n\n"
        "List the conditions (obligations) that are \"necessary for the claim to "
        "be true\" and independently checkable without seeing any argument. "
        "Every obligation must be a world-level condition about the conclusion "
        "itself, concrete enough that a check of it could actually fail — an "
        "obligation that nothing could ever falsify is worthless, do not emit "
        "it. For each obligation give:\n"
        "  text: the obligation content"
    ),
    schema={"obligations": [{"text": str}]},
    tools=["Read", "Grep", "Glob"],
)

ENUMERATE_ORACLES_AGENT = AgentRole(
    name="oracles",
    system=(
        "You are the oracle enumerator, and SUFFICIENCY IS YOUR "
        "RESPONSIBILITY: list only methods whose successful, faithful "
        "execution would by itself establish this obligation. No one "
        "downstream re-judges sufficiency — a method that falls short of "
        "the obligation's quantifier or scope (a finite sample against an "
        "unbounded universal) must simply not appear on your list.\n"
        "If this verification task touches a domain you do not thoroughly "
        "understand, RESEARCH IT FIRST: read the relevant documentation, "
        "instructions, references and proof techniques (locally and on the "
        "web) until you actually understand what a sufficient check looks "
        "like in this domain. Your guarantees and your level grading are "
        "what the whole game executes against — getting the levels right "
        "matters more than answering fast. Do NOT use research to probe "
        "which instruments are installed — instrument availability is the "
        "execution stage's problem: a missing instrument gets installed at "
        "discharge time, and a method whose instrument is truly "
        "unobtainable will fail there honestly. Never omit a method merely "
        "because its tool may be absent.\n"
        "Grade every method with an assurance level OF YOUR OWN DEVISING: "
        "you decide the scale and the labels — from the simplest check that "
        "already suffices up to the most assured (e.g. a machine-checked "
        "formal proof). The level is your signed declaration of how strong "
        "the method's guarantee is, and it is disclosed on the certificate.\n"
        "Order the list ascending by your own grading: the simplest "
        "sufficient method first, the most assured last — execution tries "
        "them in exactly the order you give."
    ),
    template=(
        "Claim: {claim_text}\n"
        "Obligation: {obligation_text}\n"
        "Claim features: {features}\n"
        "Methods known to work (may be empty): {registry}\n\n"
        "Enumerate the sufficient methods for this obligation, ordered by "
        "assurance level ascending (simplest sufficient first). Research "
        "the domain first if you need to. For each method give:\n"
        "  procedure: concretely how to do it (including what class of "
        "instrument/library to use)\n"
        "  artifact: what auditable evidence the method produces\n"
        "  level: its assurance level — your own scale, your own words"
    ),
    schema={
        "methods": [
            {
                "procedure": str,
                "artifact": str,
                "level": str,
            }
        ]
    },
    tools=["Read", "Grep", "Glob", "WebSearch", "WebFetch"],
    max_turns=15,      # research takes room; still no Bash — no probing
)

DISCHARGE_AGENT = AgentRole(
    name="discharge",
    system=(
        "You are the discharge executor. You carry out the check by the given "
        "method and produce an auditable artifact. If the method's instrument "
        "is not installed, installing it is part of the job (into the "
        "toolchain directory; installation is a one-time cost, not a "
        "boundary); only when it is genuinely unobtainable here — after "
        "actually attempting — do you report outcome=unobtainable, stating "
        "why as the witness. Crashes, timeouts, and unparseable output are "
        "all honestly recorded as failed — never converted into a pass."
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
        "was found) | failed (the check itself did not run) | unobtainable "
        "(the method's instrument cannot be installed or obtained in this "
        "environment, confirmed by actually trying)\n"
        "  witness: for refuted give the concrete counterexample, for failed "
        "give the failure reason, otherwise an empty string\n"
        "  artifact: paths of the files written to disk (a comma-separated "
        "string, not a JSON array)"
    ),
    schema={"outcome": OUTCOMES, "witness": str, "artifact": str},
    tools=["Bash", "Read", "Write", "Grep"],
)

JUDGER_AGENT = AgentRole(
    name="judger",
    system=(
        "You are the judger of EXECUTION FIDELITY. Sufficiency was settled "
        "upstream: the oracle guaranteed that this method's success "
        "establishes the obligation, and you do NOT re-judge that. You "
        "answer one question: was the bound method faithfully executed, "
        "and the promised artifact genuinely delivered? Faithful means the "
        "procedure was followed as specified — right instrument, right "
        "scope and range, complete (no sorry left in a proof, no silently "
        "narrowed domain, no substituted statement) — and the deliverable "
        "is the promised kind, produced by this run.\n"
        "You are the one who LOOKS AT THE RESULT: open the artifact and "
        "examine the actual outputs yourself, not the narration; there is "
        "no separate re-run stage, so rerun the check yourself whenever "
        "reproducibility is in doubt. When ruling no, phrase the reason as "
        "a concrete fix instruction — it goes straight back to the executor "
        "for a redo (a bounded loop of at most three rounds)."
    ),
    template=(
        "Obligation (context only): {obligation_text}\n"
        "Bound method's procedure: {procedure}\n"
        "Promised artifact: {promised_artifact}\n"
        "Submitted evidence (including artifact paths): {evidence}\n\n"
        "Examine the result itself and judge:\n"
        "  faithful: yes | no — the method was executed as specified and "
        "the promised artifact delivered\n"
        "  reason: your reasoning; for 'no', a concrete fix instruction"
    ),
    schema={"faithful": ("yes", "no"), "reason": str},
    tools=["Read", "Bash", "Grep"],
)

ALL_AGENT_ROLES = {
    role.name: role
    for role in (
        COMPILE_OBLIGATIONS_AGENT,
        ENUMERATE_ORACLES_AGENT,
        DISCHARGE_AGENT,
        JUDGER_AGENT,
    )
}
