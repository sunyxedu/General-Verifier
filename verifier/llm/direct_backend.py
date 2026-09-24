"""Direct model calls for the verifier seats that need no tools.

A seat is a role call: prompt in, schema-checked JSON out. The Agent SDK
and Codex backends run every seat as a full agent process — CLI start-up,
a coding-assistant system preset, a tool loop, a structured-output gate —
which costs 40–150 s per call even when the role never touches a tool.
Several roles are text-in, text-out by construction: the screener, the
obligation compiler, the critic, the script judger, the sufficiency
pre-audit. Everything they may consult is already in the rendered prompt.

DirectBackend runs such a role as one model request with a forced,
schema-typed tool call, so the answer is the JSON the role contract
demands and nothing else. RoutingBackend seats the tool-free roles on it
and leaves the tool-using roles (executor, judger over a live world,
extractor over a ledger, oracle reading a catalog) on the agent backend,
behind the same ``complete(role_name, prompt, variables)`` interface the
kernel already speaks. Prompts, schemas, retries and validation are
untouched: only the transport changes.

Transports: the Anthropic Messages API (first-party or any endpoint that
speaks the protocol) and the OpenAI Responses API. Both need an API key;
a subscription login cannot call a raw model API.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .agents import ALL_AGENT_ROLES, AgentBackend, AgentRole, to_json_schema
from .model_spans import model_call_span
from .schema import RoleError

try:                                    # the transports' HTTP client
    import httpx as _httpx
    # RequestError: transport failures and a body that failed to decode;
    # OSError: a raw socket or TLS error (ssl.SSLError, "UNEXPECTED_EOF")
    # that escaped httpx's mapping — every one a request that never
    # completed, nothing the model said
    _CONNECTION_ERRORS: Tuple[type, ...] = (_httpx.RequestError, OSError)
except ImportError:                     # pragma: no cover - httpx is a dependency
    _CONNECTION_ERRORS = (OSError,)

DIRECT_SEATS_ENV = "GV_DIRECT_SEATS"
DIRECT_PROVIDER_ENV = "GV_DIRECT_PROVIDER"     # anthropic | openai | openrouter
DIRECT_MODEL_ENV = "GV_DIRECT_MODEL"

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

TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
# The provider refusing for want of credit is an account state the operator
# fixes by topping up, not a verdict on the call: wait it out on the SAME
# provider (the direct seats are never silently moved to another backend).
BILLING_STATUSES = {402}
BILLING_DELAYS = (60, 120, 300, 600, 600, 600, 600, 600)
TRANSIENT_DELAYS = (0, 20, 60, 120, 240)
# the output budget a direct call starts with. Reasoning counts against it,
# and at max effort obligation compilation alone spends 35-50k tokens: a
# smaller start pays a whole truncated attempt before the retry
DEFAULT_MAX_TOKENS = 64000
MAX_OUTPUT_TOKENS = 128000   # the most a retry asks for
EMIT_TOOL = "emit"


class TransientTransportError(RuntimeError):
    """A status worth waiting out (rate limit, overload, gateway)."""


class AwaitingCredit(TransientTransportError):
    """The provider refused for want of credit; waited out on a longer
    schedule than an overload, on the same provider."""


class OutputBudgetExhausted(RuntimeError):
    """The model spent the whole output budget before answering (long
    reasoning at high effort); one retry with a larger budget is due. The
    provider's usage for the spent attempt rides along: it was billed."""

    def __init__(self, detail: str, usage: Optional[Dict[str, Any]] = None):
        super().__init__(detail)
        self.usage = dict(usage or {})


def strict_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """The OpenAI-style strict JSON schema for a role schema: every
    property required, an optional one made nullable. The role's optional
    keys come back as null and are stripped again by
    :func:`strip_optional_nulls`, so the payload downstream is exactly what
    the role contract describes."""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        required = set(out.get("required") or [])
        props = {}
        for key, sub in out["properties"].items():
            sub = strict_schema(sub)
            if key not in required:
                sub = {"anyOf": [sub, {"type": "null"}]}
            props[key] = sub
        out["properties"] = props
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    if out.get("type") == "array" and isinstance(out.get("items"), dict):
        out["items"] = strict_schema(out["items"])
    return out


def strip_optional_nulls(payload: Any, schema: Dict[str, Any]) -> Any:
    """Remove the nulls a strict-mode answer carries for the role schema's
    optional keys (recursively), leaving present values untouched."""
    if not isinstance(schema, dict):
        return payload
    if (schema.get("type") == "object" and isinstance(payload, dict)
            and isinstance(schema.get("properties"), dict)):
        required = set(schema.get("required") or [])
        out = {}
        for key, value in payload.items():
            sub = schema["properties"].get(key)
            if value is None and sub is not None and key not in required:
                continue
            out[key] = strip_optional_nulls(value, sub) if sub else value
        return out
    if (schema.get("type") == "array" and isinstance(payload, list)
            and isinstance(schema.get("items"), dict)):
        return [strip_optional_nulls(item, schema["items"]) for item in payload]
    return payload


def _response_json(response) -> Any:
    """A 200's JSON body. A body cut off in transit (a connection dropped
    mid-read) does not parse; that is the transport's failure, retried like
    one, not an answer the model gave."""
    try:
        return response.json()
    except ValueError as err:
        raise TransientTransportError(
            f"HTTP {response.status_code} with an unparseable body "
            f"({len(response.content)} bytes): {err}") from err


class AnthropicMessagesTransport:
    """One forced tool call per request against /v1/messages."""

    def __init__(self, api_key: str, model: str,
                 base_url: Optional[str] = None, timeout: float = 900.0,
                 auth_header: str = "x-api-key"):
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self.timeout = timeout
        self.auth_header = auth_header
        self.name = f"anthropic-messages@{self.base_url}"

    def _headers(self) -> Dict[str, str]:
        headers = {"anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        if self.auth_header == "authorization":
            headers["authorization"] = f"Bearer {self.api_key}"
        else:
            headers["x-api-key"] = self.api_key
        return headers

    def complete(self, *, system: str, prompt: str, schema: Dict[str, Any],
                 model: Optional[str] = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import httpx
        body = {
            "model": model or self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{
                "name": EMIT_TOOL,
                "description": "Return the role's complete answer as the "
                               "structured object the contract requires.",
                "input_schema": schema,
            }],
            "tool_choice": {"type": "tool", "name": EMIT_TOOL},
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/v1/messages",
                                   headers=self._headers(), json=body)
        if response.status_code in BILLING_STATUSES:
            raise AwaitingCredit(
                f"HTTP {response.status_code}: {response.text[:300]}")
        if response.status_code in TRANSIENT_STATUSES:
            raise TransientTransportError(
                f"HTTP {response.status_code}: {response.text[:300]}")
        if response.status_code != 200:
            raise RoleError(
                f"direct call failed: HTTP {response.status_code}: "
                f"{response.text[:500]}")
        data = _response_json(response)
        for block in data.get("content") or []:
            if block.get("type") == "tool_use" and block.get("name") == EMIT_TOOL:
                payload = block.get("input")
                if not isinstance(payload, dict):
                    raise RoleError("direct call returned a non-object tool input")
                return payload, dict(data.get("usage") or {})
        text = " ".join(str(b.get("text", "")) for b in data.get("content") or [])
        raise RoleError(
            "direct call returned no structured answer "
            f"(stop_reason={data.get('stop_reason')}): {text[:300]}")


class OpenAIResponsesTransport:
    """One strict JSON-schema response per request against /v1/responses."""

    def __init__(self, api_key: str, model: str,
                 base_url: Optional[str] = None, timeout: float = 900.0,
                 reasoning_effort: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.openai.com").rstrip("/")
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.name = f"openai-responses@{self.base_url}"

    def complete(self, *, system: str, prompt: str, schema: Dict[str, Any],
                 model: Optional[str] = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import httpx
        body: Dict[str, Any] = {
            "model": model or self.model,
            "instructions": system,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "json_schema", "name": EMIT_TOOL,
                                "schema": strict_schema(schema), "strict": True}},
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/v1/responses",
                headers={"authorization": f"Bearer {self.api_key}",
                         "content-type": "application/json"},
                json=body)
        if response.status_code in BILLING_STATUSES:
            raise AwaitingCredit(
                f"HTTP {response.status_code}: {response.text[:300]}")
        if response.status_code in TRANSIENT_STATUSES:
            raise TransientTransportError(
                f"HTTP {response.status_code}: {response.text[:300]}")
        if response.status_code != 200:
            raise RoleError(
                f"direct call failed: HTTP {response.status_code}: "
                f"{response.text[:500]}")
        data = _response_json(response)
        text = data.get("output_text")
        if not text:
            parts: List[str] = []
            for item in data.get("output") or []:
                for content in item.get("content") or []:
                    if content.get("type") == "output_text":
                        parts.append(content.get("text", ""))
            text = "".join(parts)
        try:
            payload = json.loads(text or "")
        except ValueError as err:
            raise RoleError(
                f"direct call returned non-JSON text: {str(text)[:300]}") from err
        if not isinstance(payload, dict):
            raise RoleError("direct call returned a non-object answer")
        return strip_optional_nulls(payload, schema), dict(data.get("usage") or {})


class OpenAIChatTransport:
    """One strict JSON-schema completion per request against
    /v1/chat/completions — the OpenAI-compatible surface OpenRouter and
    most gateways expose."""

    def __init__(self, api_key: str, model: str,
                 base_url: Optional[str] = None, timeout: float = 900.0,
                 reasoning_effort: Optional[str] = None,
                 extra_headers: Optional[Dict[str, str]] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://openrouter.ai/api").rstrip("/")
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.extra_headers = dict(extra_headers or {})
        self.name = f"openai-chat@{self.base_url}"

    def complete(self, *, system: str, prompt: str, schema: Dict[str, Any],
                 model: Optional[str] = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import httpx
        body: Dict[str, Any] = {
            "model": model or self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": EMIT_TOOL, "schema": strict_schema(schema),
                "strict": True}},
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        headers = {"authorization": f"Bearer {self.api_key}",
                   "content-type": "application/json", **self.extra_headers}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/v1/chat/completions",
                                   headers=headers, json=body)
        if response.status_code in BILLING_STATUSES:
            raise AwaitingCredit(
                f"HTTP {response.status_code}: {response.text[:300]}")
        if response.status_code in TRANSIENT_STATUSES:
            raise TransientTransportError(
                f"HTTP {response.status_code}: {response.text[:300]}")
        if response.status_code != 200:
            raise RoleError(
                f"direct call failed: HTTP {response.status_code}: "
                f"{response.text[:500]}")
        data = _response_json(response)
        if data.get("error"):
            raise RoleError(f"direct call failed: {json.dumps(data['error'])[:500]}")
        choices = data.get("choices") or []
        if not choices:
            raise RoleError("direct call returned no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content")
        if isinstance(text, list):   # some gateways return content parts
            text = "".join(part.get("text", "") for part in text
                           if isinstance(part, dict))
        finish = choice.get("finish_reason") or choice.get("native_finish_reason")
        if not text:
            # an exhausted output budget (reasoning ate it) is worth one more
            # attempt with a larger budget; anything else is the model's
            # failure to answer, reported with what it did return
            detail = (f"finish_reason={finish}; refusal={str(message.get('refusal'))[:200]}; "
                      f"reasoning_chars={len(message.get('reasoning') or '')}")
            if finish in ("length", "max_tokens", "max_output_tokens"):
                raise OutputBudgetExhausted(detail, data.get("usage"))
            raise RoleError(f"direct call returned no answer text ({detail})")
        try:
            payload = json.loads(text or "")
        except ValueError as err:
            if finish in ("length", "max_tokens", "max_output_tokens"):
                # the answer was cut at the budget: retry larger, not fail
                raise OutputBudgetExhausted(
                    f"finish_reason={finish}; truncated answer of {len(text)} chars",
                    data.get("usage")) from err
            raise RoleError(
                f"direct call returned non-JSON text ({finish}): {str(text)[:300]}") from err
        if not isinstance(payload, dict):
            raise RoleError("direct call returned a non-object answer")
        return strip_optional_nulls(payload, schema), dict(data.get("usage") or {})


class DirectBackend:
    """Runs tool-free roles as single structured model calls."""

    def __init__(self, transport, roles: Optional[Dict[str, AgentRole]] = None,
                 cwd: Optional[str] = None, model: Optional[str] = None):
        self.transport = transport
        self.roles = dict(roles) if roles is not None else dict(ALL_AGENT_ROLES)
        self.cwd = cwd
        self.model = model or getattr(transport, "model", None)
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
        error: Optional[str] = None
        call_id = uuid.uuid4().hex
        try:
            with model_call_span(call_id, role=role_name, model=str(model)):
                return self._call(role, system, prompt, schema, model, attempts)
        except Exception as err:  # noqa: BLE001 - re-raised after the receipt
            error = str(err)[:300]
            raise
        finally:
            self._receipt(role, started, attempts, error, variables)

    def _call(self, role: AgentRole, system: str, prompt: str,
              schema: Dict[str, Any], model: Optional[str],
              attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        last: Optional[str] = None
        with self._budget_lock:
            max_tokens = self._role_budget.get(role.name, DEFAULT_MAX_TOKENS)
        transient_waits = iter(TRANSIENT_DELAYS[1:])
        billing_waits = iter(BILLING_DELAYS)
        while True:
            attempt_started = time.time()
            try:
                try:
                    payload, usage = self.transport.complete(
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
                max_tokens = min(2 * max_tokens, MAX_OUTPUT_TOKENS)
                with self._budget_lock:
                    self._role_budget[role.name] = max(
                        max_tokens, self._role_budget.get(role.name, 0))
                continue
            except AwaitingCredit as err:
                last = str(err)
                attempts.append({"role": role.name, "seconds": round(
                    time.time() - attempt_started, 3), "error": last,
                    "awaiting_credit": True})
                delay = next(billing_waits, None)
                if delay is None:
                    raise RoleError(f"{role.name}: provider credit still "
                                    f"exhausted after waiting: {last}")
                time.sleep(delay)
                continue
            except TransientTransportError as err:
                last = str(err)
                attempts.append({"role": role.name, "seconds": round(
                    time.time() - attempt_started, 3), "error": last})
                delay = next(transient_waits, None)
                if delay is None:
                    raise RoleError(f"{role.name}: transient transport "
                                    f"failure persisted: {last}")
                time.sleep(delay)
                continue
            except Exception as err:
                # any other failure still took an attempt: on record with its
                # error, like the retried ones
                attempts.append({"role": role.name, "seconds": round(
                    time.time() - attempt_started, 3),
                    "error": f"{type(err).__name__}: {err}"[:300]})
                raise
            attempts.append({"role": role.name, "seconds": round(
                time.time() - attempt_started, 3), "usage": usage})
            self.transcripts.append({"role": role.name, "usage": usage})
            return payload

    def _receipt(self, role: AgentRole, started: float,
                 attempts: List[Dict[str, Any]], error: Optional[str],
                 variables: Dict[str, Any]) -> None:
        """Same receipt row as AgentBackend, with backend=direct, so latency
        and cost breakdowns see every seat call whatever ran it."""
        if not self.cwd:
            return
        row = {
            "role": role.name,
            "backend": "direct",
            "transport": getattr(self.transport, "name", "direct"),
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
        try:
            with open(os.path.join(self.cwd, "role_calls.jsonl"), "a",
                      encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass


class RoutingBackend:
    """Seats each role on the backend it needs: tool-free roles on the
    direct model call, tool-using roles on the agent backend. Everything
    the kernel reads or sets on a backend (``roles``, capability probes,
    offline scope validation) is forwarded to the agent backend, which
    remains the seat of record for anything with a palette."""

    def __init__(self, direct: DirectBackend, agent, direct_roles: Iterable[str]):
        self.direct = direct
        self.agent = agent
        self.direct_roles: Set[str] = set(direct_roles)
        self.uses_role_wire_schema = True

    @property
    def roles(self) -> Dict[str, AgentRole]:
        return self.agent.roles

    @roles.setter
    def roles(self, value: Dict[str, AgentRole]) -> None:
        self.agent.roles = value
        self.direct.roles = dict(value)

    def routes_direct(self, role_name: str) -> bool:
        return role_name in self.direct_roles

    def complete(self, role_name: str, prompt: str,
                 variables: Dict[str, Any]) -> Any:
        if self.routes_direct(role_name):
            # A direct seat runs on the direct provider only — never
            # silently on the agent backend, which is a different model
            # call with a different cost and latency profile. Outages and
            # exhausted credit are waited out inside the direct call.
            return self.direct.complete(role_name, prompt, variables)
        return self.agent.complete(role_name, prompt, variables)

    def allows_lazy_oracle_catalog(self, role_name: str,
                                   variables: Dict[str, Any]) -> bool:
        if self.routes_direct(role_name):
            return False
        probe = getattr(self.agent, "allows_lazy_oracle_catalog", None)
        return bool(probe(role_name, variables)) if callable(probe) else False

    def __getattr__(self, name: str) -> Any:
        # capability probes and seat-of-record state live on the agent side
        return getattr(self.agent, name)


def direct_seat_roles(environ: Optional[Dict[str, str]] = None) -> Set[str]:
    """Which roles run as direct model calls: GV_DIRECT_SEATS unset or
    ``default`` → the tool-free set; ``0``/``off`` → none; otherwise a
    comma-separated list of role names."""
    if environ is None:
        from . import provider
        configured = provider.cfg(DIRECT_SEATS_ENV)
    else:
        configured = environ.get(DIRECT_SEATS_ENV)
    raw = (configured or "default").strip().lower()
    if raw in ("0", "off", "none", ""):
        return set()
    if raw in ("1", "default", "on"):
        return set(DEFAULT_DIRECT_ROLES)
    return {part.strip() for part in raw.split(",") if part.strip()}


def direct_transport(environ: Optional[Dict[str, str]] = None):
    """The direct transport the configuration supports, or None when the
    only credential is a subscription login (which no raw API accepts)."""
    from . import provider
    # The provider and its API key may be configured in the shipped .env.
    # An injected mapping stays self-contained for tests and callers that
    # intentionally supply a complete environment.
    setting = (provider.cfg if environ is None else environ.get)
    choice = (setting(DIRECT_PROVIDER_ENV) or "").strip().lower()
    if choice == "openrouter" or (not choice and provider.PROVIDER == "openrouter"):
        key = setting("OPENROUTER_API_KEY") or (
            provider.API_KEY if provider.PROVIDER == "openrouter" else None)
        if not key:
            return None
        return OpenAIChatTransport(
            key, setting(DIRECT_MODEL_ENV) or "openai/gpt-5.6-luna",
            base_url=setting("OPENROUTER_BASE_URL"),
            reasoning_effort=setting("GV_DIRECT_EFFORT") or None,
            extra_headers={"HTTP-Referer": "https://github.com/hemlix/general-verifier",
                           "X-Title": "General Verifier"})
    if choice == "openai" or (not choice and provider.PROVIDER not in
                              ("anthropic", "deepseek", "custom")
                              and setting("OPENAI_API_KEY")):
        key = setting("OPENAI_API_KEY")
        if not key:
            return None
        return OpenAIResponsesTransport(
            key, setting(DIRECT_MODEL_ENV) or "gpt-5.6-luna",
            base_url=setting("OPENAI_BASE_URL"),
            reasoning_effort=setting("GV_DIRECT_EFFORT") or None)
    if provider.PROVIDER in ("anthropic", "deepseek", "custom", "openrouter"):
        if not provider.API_KEY:
            return None   # login-only: no raw API behind an OAuth token
        auth = "authorization" if provider.PROVIDER == "openrouter" else "x-api-key"
        return AnthropicMessagesTransport(
            provider.API_KEY, setting(DIRECT_MODEL_ENV) or str(provider.MODEL),
            base_url=provider.ANTHROPIC_BASE_URL, auth_header=auth)
    return None


def build_backend(roles: Dict[str, AgentRole], cwd: Optional[str],
                  model: Optional[str], environ: Optional[Dict[str, str]] = None,
                  agent=None):
    """The kernel's backend: ``agent`` (default: the Agent SDK backend;
    an injected Codex backend works the same) with the tool-free roles
    seated on a direct transport when one is configured."""
    if agent is None:
        agent = AgentBackend(roles=roles, cwd=cwd, model=model)
    direct_roles = direct_seat_roles(environ)
    if not direct_roles:
        return agent
    transport = direct_transport(environ)
    if transport is None:
        return agent
    direct = DirectBackend(transport, roles=roles, cwd=cwd,
                           model=(os.environ if environ is None else environ)
                           .get(DIRECT_MODEL_ENV) or getattr(transport, "model", None))
    return RoutingBackend(direct, agent, direct_roles)
