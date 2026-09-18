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
import contextvars
import contextlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import prompts
from .model_spans import (
    role_attempt, role_span, seat_acquired, seat_released,
    seat_wait_started, submit_with_context,
)
from .schema import RoleError, validate, field_name, optional_key
from .oracle_execution import (
    ORACLE_EXECUTION_WIRE_SCHEMA,
    lower_oracle_execution_payload,
)
from ..engine.source_evidence import (
    SourceEvidenceView,
    refresh_source_evidence_variables,
)
# The Claude Agent SDK backs the DEFAULT backend only; alternate backends
# (CodexBackend) never touch it. Imported lazily so that a seats=codex run
# does not depend on the SDK's import chain (mcp -> pydantic -> native
# extensions) even loading — the palette of the active backend, not the
# default one, should decide what must be importable.
def _sdk():
    import claude_agent_sdk
    return claude_agent_sdk

# the SDK attempts of the seat call in progress (per call, propagated to
# the worker thread with the context): the receipt sums exactly these
_ATTEMPT_SINK: "contextvars.ContextVar[Optional[List[Dict[str, Any]]]]" = \
    contextvars.ContextVar("gv_attempt_sink", default=None)


class _TransientSDKError(RuntimeError):
    """Infrastructure failure worth backing off and retrying (internal)."""


class ExhaustedInfrastructureError(RoleError):
    """A backend exhausted its infrastructure retries, not a generation defect."""


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
    # schema remains the established downstream/replay contract. A role may
    # expose a stricter model-facing shape and lower it atomically before
    # anything downstream observes the payload.
    wire_schema: Optional[Dict[str, Any]] = None
    wire_lower: Optional[
        Callable[[Dict[str, Any]], Dict[str, Any]]
    ] = None

    def render(self, variables: Dict[str, Any]) -> str:
        return self.template.format(**variables)

    def call(self, backend, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Two attempts (schema error fed back once), then RoleError."""
        last_error: Optional[RoleError] = None
        retry_feedback = ""
        oracle_catalog_force_baseline = False
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
                                # Bind after queueing: a view captured before
                                # the seat gate could go stale while waiting
                                # and waste a complete paid backend call.
                                attempt_variables = dict(variables)
                                source_view = attempt_variables.get(
                                    "_source_evidence_view")
                                mixed_ledger_path = attempt_variables.get(
                                    "_mixed_ledger_path")
                                try:
                                    allows_compact_source = getattr(
                                        backend,
                                        "allows_compact_source_evidence",
                                        None,
                                    )
                                except Exception:
                                    allows_compact_source = None
                                compact_source_allowed = None
                                if (type(source_view) is SourceEvidenceView
                                        and callable(allows_compact_source)):
                                    try:
                                        compact_source_allowed = \
                                            allows_compact_source(
                                                self.name,
                                                attempt_variables,
                                            )
                                    except Exception:
                                        # An absent or faulty optional
                                        # capability cannot change the
                                        # established evidence transport.
                                        compact_source_allowed = None
                                valid_mixed_path = False
                                if (type(source_view) is SourceEvidenceView
                                        and isinstance(
                                            mixed_ledger_path, str)
                                        and os.path.isabs(mixed_ledger_path)):
                                    try:
                                        valid_mixed_path = (
                                            mixed_ledger_path
                                            == os.path.abspath(
                                                source_view._log.path))
                                    except (AttributeError, OSError,
                                            TypeError, ValueError):
                                        valid_mixed_path = False
                                if (compact_source_allowed is False
                                        and valid_mixed_path):
                                    # Some runtimes mutate filesystem metadata
                                    # merely by reading a directory. Respect
                                    # an explicit opt-out before paying for a
                                    # call whose immutable-view postcheck could
                                    # therefore never succeed.
                                    attempt_variables["ledger_path"] = \
                                        mixed_ledger_path
                                    attempt_variables.pop(
                                        "_source_evidence_view", None)
                                else:
                                    attempt_variables = \
                                        refresh_source_evidence_variables(
                                            variables)
                                catalog_snapshot = None
                                if self.name == "oracles":
                                    from ..execution.adapters import \
                                        OracleCatalogView
                                    catalog_view = attempt_variables.get(
                                        "adapter_catalog")
                                    if type(catalog_view) is OracleCatalogView:
                                        permits_lazy_catalog = False
                                        capability = getattr(
                                            backend,
                                            "allows_lazy_oracle_catalog",
                                            None,
                                        )
                                        if callable(capability):
                                            try:
                                                permits_lazy_catalog = \
                                                    capability(
                                                        self.name,
                                                        attempt_variables,
                                                    ) is True
                                            except Exception:
                                                permits_lazy_catalog = False
                                        if (oracle_catalog_force_baseline
                                                or not permits_lazy_catalog):
                                            attempt_variables = dict(
                                                attempt_variables)
                                            attempt_variables["adapter_catalog"] = \
                                                catalog_view.fallback_catalog()
                                        else:
                                            catalog_snapshot = \
                                                catalog_view.snapshot()
                                            if catalog_snapshot is None:
                                                oracle_catalog_force_baseline = \
                                                    True
                                                attempt_variables = dict(
                                                    attempt_variables)
                                                attempt_variables[
                                                    "adapter_catalog"] = \
                                                    catalog_view.fallback_catalog()
                                source_view = attempt_variables.get(
                                    "_source_evidence_view")
                                if (type(source_view) is SourceEvidenceView
                                        and not source_view.revalidate()):
                                    # The optimization is optional. A race in
                                    # publication must expose the original
                                    # full ledger, not consume a model retry.
                                    attempt_variables = dict(attempt_variables)
                                    attempt_variables["ledger_path"] = \
                                        attempt_variables["_mixed_ledger_path"]
                                    attempt_variables.pop(
                                        "_source_evidence_view", None)
                                    source_view = None
                                catalog_changed = False
                                try:
                                    prompt = (self.render(attempt_variables)
                                              + retry_feedback)
                                    payload = backend.complete(
                                        self.name, prompt, attempt_variables)
                                finally:
                                    if catalog_snapshot is not None:
                                        catalog_changed = not \
                                            catalog_snapshot.revalidate()
                                        catalog_snapshot.close()
                                        if catalog_changed:
                                            oracle_catalog_force_baseline = True
                                if catalog_changed:
                                    raise RoleError(
                                        "oracles: adapter catalog changed "
                                        "during audit")
                            finally:
                                seat_released(acquired)
                        if (type(source_view) is SourceEvidenceView
                                and not source_view.revalidate()):
                            raise RoleError(
                                f"{self.name}: source evidence changed during audit")
                        use_wire = (
                            self.wire_schema is not None
                            and getattr(
                                backend, "uses_role_wire_schema", False,
                            ) is True
                        )
                        if use_wire:
                            validate(self.wire_schema, payload)
                            if self.wire_lower is None:
                                raise RoleError(
                                    f"{self.name}: wire schema has no lowerer",
                                )
                            payload = self.wire_lower(payload)
                        validate(self.schema, payload)
                    return payload
                except RoleError as err:
                    if isinstance(err, ExhaustedInfrastructureError):
                        # The backend already exhausted provider retries.
                        # Do not double that budget or inject provider faults
                        # into a semantic/schema-repair prompt.
                        raise
                    last_error = err
                    retry_feedback = (
                        f"\n\n[previous attempt failed; fix it and retry] {err}"
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
        self.uses_role_wire_schema = True
        self.transcripts: List[Dict[str, Any]] = []

    def allows_lazy_oracle_catalog(
        self, role_name: str, variables: Dict[str, Any],
    ) -> bool:
        """Allow locators only for an actually read-only Oracle palette."""
        if role_name != "oracles":
            return False
        role = self.roles.get(role_name)
        read_only_tools = frozenset({
            "Read", "Grep", "Glob", "WebSearch", "WebFetch",
        })
        return (role is not None
                and all(isinstance(tool, str) for tool in role.tools)
                and set(role.tools) <= read_only_tools)

    def complete(self, role_name: str, prompt: str, variables: Dict[str, Any]) -> Any:
        """
        Backend entry point (same signature as FixtureBackend.complete).

        Loop-safe sync wrapper: callable from plain sync code AND from code executing inside a running event loop (e.g. an MCP tool handler in the MCP server).
        In the latter case the coroutine runs on a private loop in a worker thread, because asyncio.run would raise on a running loop.
        """
        role = self.roles.get(role_name)
        if role is None:
            raise RoleError(f"unknown agent role: {role_name}")
        started = time.time()
        attempts: List[Dict[str, Any]] = []   # this call's SDK attempts only
        error: Optional[str] = None
        token = _ATTEMPT_SINK.set(attempts)
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.run(role, prompt))
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as pool:
                return submit_with_context(
                    pool, asyncio.run, self.run(role, prompt)).result()
        except Exception as err:  # noqa: BLE001 - re-raised after the receipt
            error = str(err)[:300]
            raise
        finally:
            _ATTEMPT_SINK.reset(token)
            self._receipt(role, started, attempts, error, variables)

    def _receipt(self, role: AgentRole, started: float,
                 attempts: List[Dict[str, Any]],
                 error: Optional[str], variables: Dict[str, Any]) -> None:
        """One JSONL row per seat call in <cwd>/role_calls.jsonl: role,
        wall time, turns and usage of every SDK attempt. Pure telemetry
        for latency/cost breakdowns; never read by any seat."""
        if not self.cwd:
            return
        row = {
            "role": role.name,
            "started": started,
            "finished": time.time(),
            "seconds": round(time.time() - started, 3),
            "attempts": len(attempts),
            "num_turns": sum(a.get("num_turns") or 0 for a in attempts),
            "cost_usd": sum(a.get("total_cost_usd") or 0 for a in attempts),
            "usage": [a.get("usage") for a in attempts],
            "subtypes": [a.get("subtype") for a in attempts],
            "error": error,
            "claim_text": str(variables.get("claim_text", ""))[:120],
            "obligation_text": str(variables.get("obligation_text", ""))[:160],
        }
        try:
            with open(os.path.join(self.cwd, "role_calls.jsonl"), "a",
                      encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    # statuses worth waiting out (rate limit / overload / gateway); the CLI
    # sometimes reports these only as the opaque "error result: success"
    TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
    # seconds before each attempt: a subscription's rate limit recovers on
    # the order of minutes, and a seat lost to it costs a whole claim
    transient_delays = (0, 20, 60, 120, 240, 480)

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
        raise ExhaustedInfrastructureError(
            f"{role.name}: transient sdk failure persisted: {last}")

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

        transcript = {
            "role": role.name,
            "session_id": getattr(result, "session_id", None),
            "subtype": getattr(result, "subtype", None),
            "num_turns": getattr(result, "num_turns", None),
            "total_cost_usd": getattr(result, "total_cost_usd", None),
            "usage": getattr(result, "usage", None),
            "api_error_status": getattr(result, "api_error_status", None),
            "stderr_tail": list(stderr_tail),
            "error": str(error)[:300] if error is not None else None,
        }
        self.transcripts.append(transcript)
        sink = _ATTEMPT_SINK.get(None)
        if sink is not None:
            sink.append(transcript)

        if result is not None and result.structured_output is not None and result.subtype == "success":
            payload = result.structured_output
            validate(role.wire_schema or role.schema, payload)
            # Our gate is independent of the SDK's structured-output gate.
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
            # every seat runs on the configured provider; the SDK merges
            # this over os.environ, so PATH is passed explicitly without
            # the verifier's own virtualenv (a seat's `python3` must be
            # the world's, not ours)
            env=provider.claude_env(
                base={"PATH": os.environ.get("PATH", "")}),
            output_format={
                "type": "json_schema",
                "schema": to_json_schema(role.wire_schema or role.schema),
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
    # excluded is the compiler's exclusion ledger: source clauses that read
    # as instructions to the candidate but generate no obligation under the
    # runner-interface quotient or governing modality, each with its
    # reason.  The Critic reviews these rulings with the contract; the
    # kernel never verifies them.  Optional: absent reads as an empty ledger.
    # task_bearing: whether the authoritative task itself requires the
    # condition, or the claim asserted it beyond the task — returned with
    # a refutation so the solver knows what the deliverable must satisfy
    schema={"obligations": [{"text": str,
                             "required_basis": ("world", "record"),
                             "severity?": ("must", "should"),
                             "clause?": str,
                             "refutable_by_absence?": bool,
                             "task_bearing?": ("required", "beyond_task",
                                               "unclear")}],
            "excluded?": [{"clause": str, "reason": str}]},
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
                "adapter_input_binding?": ("fixed-v1",),
                "adapter_arguments?": str,
                "adapter_inputs?": [{
                    "ref": str,
                    "provenance": ("task_given", "delivered", "ledger",
                                   "verifier", "public"),
                    "role": ("subject", "input", "authority"),
                }],
                # EXECUTABLE FORM: a fixed bash script the executor runs
                # verbatim (no agent), read afterwards by the script
                # judger. Optional — the agent executor carries out any
                # method without one.
                "script?": str,
            }
        ]
    },
    tools=["Read", "Grep", "Glob", "WebSearch", "WebFetch"],
    # enumeration is design, not execution: the prompt states the tool
    # budget the seat is held to, and the turn cap backs it mechanically
    max_turns=12,
    wire_schema=ORACLE_EXECUTION_WIRE_SCHEMA,
    wire_lower=lower_oracle_execution_payload,
)

SCRIPT_JUDGER_AGENT = AgentRole(
    name="script_judger",
    system=prompts.SCRIPT_JUDGER_SYSTEM,
    template=prompts.SCRIPT_JUDGER_TEMPLATE,
    # the second signer for a mechanically executed method: sufficiency
    # from the script text, outcome from its recorded output. Fidelity is
    # by construction (the script ran verbatim), so no redo loop exists
    # for this path. The run is inlined: no tools, one turn.
    schema={"sufficient": ("yes", "no"), "outcome": OUTCOMES,
            "witness": str,
            "witness_kind?": ("execution", "world_state",
                              "record_contradiction", "absence", "none"),
            "inputs?": [{"ref": str,
                         "provenance": ("task_given", "delivered",
                                        "ledger", "verifier", "public"),
                         "role": ("subject", "input", "authority")}],
            "degradation": ("justified", "unjustified", "n/a"),
            "reason": str},
    tools=[],
    # no tools, but a structured answer still takes the harness a turn
    # or two beyond the reply itself; a cap of two cut honest answers off
    max_turns=4,
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

# Pre-execution sufficiency audit of one proposed method: the Judger's
# first question asked before the executor spends minutes on it.  Tool-less
# and single-turn by design; it never sees or settles evidence.
SUFFICIENCY_AGENT = AgentRole(
    name="sufficiency",
    system=prompts.SUFFICIENCY_SYSTEM,
    template=prompts.SUFFICIENCY_TEMPLATE,
    schema={
        "sufficient": ("yes", "no"),
        "reason": str,
    },
    tools=[],
    max_turns=1,
)

SCREENER_AGENT = AgentRole(
    name="screener",
    system=prompts.SCREENER_SYSTEM,
    template=prompts.SCREENER_TEMPLATE,
    # a LEVEL, not a status: high admits the claim as confident (assumed,
    # unverified, no instrument spent); medium/low route it through the
    # full pipeline. The whole record excerpt is inlined so the seat needs
    # no tools — one turn, the cheapest call in the system.
    schema={"confidence": ("high", "medium", "low"),
            "grounds": str, "reason": str},
    tools=[],
    # no tools, but a structured answer still takes the harness a turn
    # or two beyond the reply itself; a cap of two cut honest answers off
    max_turns=4,
)

ALL_AGENT_ROLES = {
    role.name: role
    for role in (
        COMPILE_OBLIGATIONS_AGENT,
        CRITIC_AGENT,
        ENUMERATE_ORACLES_AGENT,
        SUFFICIENCY_AGENT,
        DISCHARGE_AGENT,
        JUDGER_AGENT,
        EXTRACT_CLAIMS_AGENT,
        SCREENER_AGENT,
        SCRIPT_JUDGER_AGENT,
    )
}
