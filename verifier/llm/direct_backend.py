"""Direct model calls for the verifier seats that need no tools.

A seat is a role call: prompt in, schema-checked JSON out. The Agent SDK
and Codex backends run every seat as a full agent process — CLI start-up,
a coding-assistant system preset, a tool loop, a structured-output gate —
which costs 40–150 s per call even when the role never touches a tool.
Several roles are text-in, text-out by construction: the screener, the
obligation compiler, the critic, the script judger, the sufficiency
pre-audit. Everything they may consult is already in the rendered prompt.

DirectRuntime runs such a role as one structured request to a model
endpoint (``llm/endpoints``), so the answer is the JSON the role contract
demands and nothing else. RoutingRuntime (``seat_runner``) seats the
tool-free roles on it and leaves the tool-using roles (executor, judger over a live world,
extractor over a ledger, oracle reading a catalog) on the agent runtime,
behind the same ``complete(role_name, prompt, variables)`` interface the
kernel already speaks. Prompts, schemas, retries and validation are
untouched: only the transport changes.

The endpoint comes from the one provider preset table through
``endpoints.resolve_endpoint``; every endpoint needs an API key, since a
subscription login cannot call a raw model API.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import DirectSettings
from .agents import ALL_AGENT_ROLES, AgentRole, ClaudeCodeRuntime, to_json_schema
from .endpoints import (
    DEFAULT_MAX_TOKENS, MAX_OUTPUT_TOKENS, AwaitingCredit,
    OutputBudgetExhausted, TransientTransportError, resolve_endpoint,
)
from .model_spans import model_call_span
from .policy import NoToolsSession, ToolPolicy
from .receipts import seat_call_receipt
from .retry import Backoff, RetryRule, retry_transient
from .runtime import RuntimeCapabilities
from .schema import RoleError
from .seat_runner import RoutingRuntime

try:                                    # the transports' HTTP client
    import httpx as _httpx
    # RequestError: transport failures and a body that failed to decode;
    # OSError: a raw socket or TLS error (ssl.SSLError, "UNEXPECTED_EOF")
    # that escaped httpx's mapping — every one a request that never
    # completed, nothing the model said
    _CONNECTION_ERRORS: Tuple[type, ...] = (_httpx.RequestError, OSError)
except ImportError:                     # pragma: no cover - httpx is a dependency
    _CONNECTION_ERRORS = (OSError,)

# the roles that need no tool: their prompt carries every input they may
# consult (task context, record excerpts, the contract, the script and its
# output). Roles with a live palette stay on the agent backend.
DEFAULT_DIRECT_ROLES = frozenset(
    {"screener", "script_judger", "sufficiency", "obligations", "critic"})

# a seat running without tools must know it, or it will narrate a Read it
# cannot perform and then guess at the result
DIRECT_NOTE = (
    "\n\nEXECUTION FORM OF THIS CALL: you are answering in a single model "
    "call with no tools. Everything you may consult is in this message; "
    "there is nothing to open, read, grep or run. Decide from the material "
    "given. If a judgment genuinely requires material that is not here, "
    "say so explicitly in your reasoning field and choose the conservative "
    "option the role allows; never describe a lookup you did not perform "
    "or invent its result."
)

# The provider refusing for want of credit is an account state the operator
# fixes by topping up, not a verdict on the call: wait it out on the SAME
# provider (the direct seats are never silently moved to another backend).
BILLING_DELAYS = (60, 120, 300, 600, 600, 600, 600, 600)
TRANSIENT_DELAYS = (0, 20, 60, 120, 240)


class DirectRuntime(RuntimeCapabilities):
    """Runs tool-free roles as single structured model calls."""

    RUNTIME_IDENTITY = "verifier.llm.direct_backend.DirectBackend"

    def __init__(self, endpoint, roles: Optional[Dict[str, AgentRole]] = None,
                 cwd: Optional[str] = None, model: Optional[str] = None):
        self.endpoint = endpoint
        self.roles = dict(roles) if roles is not None else dict(ALL_AGENT_ROLES)
        self.cwd = cwd
        self.model = model or getattr(endpoint, "model", None)
        self.uses_role_wire_schema = True
        self.transcripts: List[Dict[str, Any]] = []
        # the output budget each role last needed: a role whose reasoning
        # outgrew the default once starts there next time instead of paying
        # a whole truncated attempt again on every call
        self._role_budget: Dict[str, int] = {}
        self._budget_lock = threading.Lock()

    def allows_lazy_oracle_catalog(self, role_name: str,
                                   variables: Dict[str, Any]) -> bool:
        return False   # no tools: the catalog must be inlined, never lazily read

    def routes_direct(self, role_name: str) -> bool:
        return True    # every role here is one HTTP request

    def reuse_identity(self) -> Dict[str, Any]:
        return {"model": self.model}

    def supports_policy(self, policy: ToolPolicy) -> bool:
        # a single model request has no tool to use, but no event stream
        # to audit either
        return policy.no_tools and not policy.stream_audit

    def no_tools_runtime(self, roles: Dict[str, AgentRole],
                         session: NoToolsSession) -> "DirectRuntime":
        return DirectRuntime(self.endpoint, roles=roles,
                             cwd=session.scratch(), model=self.model)

    def complete(self, role_name: str, prompt: str,
                 variables: Dict[str, Any]) -> Any:
        role = self.roles.get(role_name)
        if role is None:
            raise RoleError(f"unknown agent role: {role_name}")
        schema = to_json_schema(role.wire_schema or role.schema)
        system = (role.system or "") + DIRECT_NOTE
        model = role.model or self.model
        started = time.time()
        attempts: List[Dict[str, Any]] = []
        call_id = uuid.uuid4().hex
        with seat_call_receipt(
                self.cwd, lambda error: self._receipt(
                    role, started, attempts, error, variables)):
            with model_call_span(call_id, role=role_name, model=str(model)):
                return self._call(role, system, prompt, schema, model, attempts)

    def _call(self, role: AgentRole, system: str, prompt: str,
              schema: Dict[str, Any], model: Optional[str],
              attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        with self._budget_lock:
            budget = [self._role_budget.get(role.name, DEFAULT_MAX_TOKENS)]

        def attempt() -> Dict[str, Any]:
            # an exhausted output budget is retried at once with a larger
            # one; the transient failures wait (below)
            while True:
                max_tokens = budget[0]
                attempt_started = time.time()
                try:
                    try:
                        payload, usage = self.endpoint.complete(
                            system=system, prompt=prompt, schema=schema,
                            model=model, max_tokens=max_tokens)
                    except _CONNECTION_ERRORS as err:
                        raise TransientTransportError(
                            f"connection: {type(err).__name__}: {err}") from err
                except OutputBudgetExhausted as err:
                    last = str(err)
                    attempts.append({"role": role.name, "seconds": round(
                        time.time() - attempt_started, 3), "error": last,
                        "max_tokens": max_tokens, "usage": err.usage or None})
                    if max_tokens >= MAX_OUTPUT_TOKENS:
                        raise RoleError(f"{role.name}: output budget exhausted "
                                        f"at {max_tokens} tokens: {last}")
                    budget[0] = min(2 * max_tokens, MAX_OUTPUT_TOKENS)
                    with self._budget_lock:
                        self._role_budget[role.name] = max(
                            budget[0], self._role_budget.get(role.name, 0))
                    continue
                except AwaitingCredit as err:
                    attempts.append({"role": role.name, "seconds": round(
                        time.time() - attempt_started, 3), "error": str(err),
                        "awaiting_credit": True})
                    raise
                except TransientTransportError as err:
                    attempts.append({"role": role.name, "seconds": round(
                        time.time() - attempt_started, 3), "error": str(err)})
                    raise
                except Exception as err:
                    # any other failure still took an attempt: on record with
                    # its error, like the retried ones
                    attempts.append({"role": role.name, "seconds": round(
                        time.time() - attempt_started, 3),
                        "error": f"{type(err).__name__}: {err}"[:300]})
                    raise
                attempts.append({"role": role.name, "seconds": round(
                    time.time() - attempt_started, 3), "usage": usage})
                self.transcripts.append({"role": role.name, "usage": usage})
                return payload

        return retry_transient(attempt, (
            RetryRule((AwaitingCredit,), Backoff(BILLING_DELAYS),
                      exhausted=lambda err: RoleError(
                          f"{role.name}: provider credit still "
                          f"exhausted after waiting: {err}")),
            RetryRule((TransientTransportError,), Backoff(TRANSIENT_DELAYS[1:]),
                      exhausted=lambda err: RoleError(
                          f"{role.name}: transient transport "
                          f"failure persisted: {err}")),
        ), sleep=time.sleep)

    def _receipt(self, role: AgentRole, started: float,
                 attempts: List[Dict[str, Any]], error: Optional[str],
                 variables: Dict[str, Any]) -> Dict[str, Any]:
        """Same receipt row as the Claude Code runtime, with backend=direct,
        so latency and cost breakdowns see every seat call whatever ran it."""
        return {
            "role": role.name,
            "backend": "direct",
            "transport": getattr(self.endpoint, "name", "direct"),
            "model": role.model or self.model,
            "started": started,
            "finished": time.time(),
            "seconds": round(time.time() - started, 3),
            "attempts": len(attempts),
            "usage": [a.get("usage") for a in attempts],
            # why each attempt that did not answer failed (None if it did)
            "attempt_errors": [a.get("error") for a in attempts],
            "error": error,
            "claim_text": str(variables.get("claim_text", ""))[:120],
            "obligation_text": str(variables.get("obligation_text", ""))[:160],
        }


def direct_seat_roles(environ: Optional[Dict[str, str]] = None) -> Set[str]:
    """Which roles run as direct model calls (``DirectSettings.seat_roles``):
    GV_DIRECT_SEATS unset or ``default`` → the tool-free set; ``0``/``off``
    → none; otherwise a comma-separated list of role names."""
    return DirectSettings.from_env(environ).seat_roles(DEFAULT_DIRECT_ROLES)


def direct_endpoint(environ: Optional[Dict[str, str]] = None):
    """The model endpoint the configuration supports for direct seats, or
    None when the only credential is a subscription login (which no raw
    API accepts). The provider and its API key may be configured in the
    shipped .env; an injected mapping stays self-contained for tests and
    callers that intentionally supply a complete environment."""
    from . import provider
    return resolve_endpoint(DirectSettings.from_env(environ),
                            provider.agent_provider())


def build_backend(roles: Dict[str, AgentRole], cwd: Optional[str],
                  model: Optional[str], environ: Optional[Dict[str, str]] = None,
                  agent=None):
    """The kernel's runtime: ``agent`` (default: the Claude Code runtime;
    an injected Codex runtime works the same) with the tool-free roles
    seated on a direct model endpoint when one is configured."""
    if agent is None:
        agent = ClaudeCodeRuntime(roles=roles, cwd=cwd, model=model)
    settings = DirectSettings.from_env(environ)
    direct_roles = settings.seat_roles(DEFAULT_DIRECT_ROLES)
    if not direct_roles:
        return agent
    from . import provider
    endpoint = resolve_endpoint(settings, provider.agent_provider())
    if endpoint is None:
        return agent
    direct = DirectRuntime(endpoint, roles=roles, cwd=cwd, model=endpoint.model)
    return RoutingRuntime(direct, agent, direct_roles)
