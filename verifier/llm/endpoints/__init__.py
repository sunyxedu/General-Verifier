"""Model endpoints: one structured model request per call, over the wire.

A ``ModelEndpoint`` answers ``complete(system=, prompt=, schema=, model=,
max_tokens=)`` with the schema-typed JSON object and the provider's usage
record. Three wire protocols implement it: the Anthropic Messages API (a
forced tool call; first-party or any endpoint that speaks the protocol),
the OpenAI Responses API and OpenAI-compatible chat completions (strict
JSON schema). All need an API key; a subscription login cannot call a raw
model API. Which endpoint a configuration selects comes from the one
provider preset table (``presets.PROVIDERS``) through ``resolve_endpoint``.

A status worth waiting out raises ``TransientTransportError`` (credit:
``AwaitingCredit``), an exhausted output budget ``OutputBudgetExhausted``;
the retry schedule belongs to the runtime that calls the endpoint.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Protocol, Tuple

from ..schema import RoleError
from . import presets
from .presets import (
    ANTHROPIC_MESSAGES, OPENAI_CHAT, OPENAI_RESPONSES, EndpointSpec,
)


class ModelEndpoint(Protocol):
    """One structured model request per call."""
    name: str    # "<protocol>@<base url>", recorded in seat receipts
    model: str   # the model a call uses unless it names another

    def complete(self, *, system: str, prompt: str, schema: Dict[str, Any],
                 model: Optional[str] = None,
                 max_tokens: int = ...,
                 ) -> Tuple[Dict[str, Any], Dict[str, Any]]: ...


TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
# The provider refusing for want of credit is an account state the operator
# fixes by topping up, not a verdict on the call (the caller waits it out on
# the same provider).
BILLING_STATUSES = {402}
# the output budget a direct call starts with: the model's own ceiling
# (GPT-5.6 Luna on OpenRouter, max_completion_tokens 128000). Reasoning
# counts against it, and a smaller start paid a whole truncated attempt
# before its retry; the budget is only a cap, billing is by tokens used
MAX_OUTPUT_TOKENS = 128000   # the most any call asks for
DEFAULT_MAX_TOKENS = MAX_OUTPUT_TOKENS
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


class AnthropicMessagesEndpoint:
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


class OpenAIResponsesEndpoint:
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


class OpenAIChatEndpoint:
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


_WIRES = {
    ANTHROPIC_MESSAGES: AnthropicMessagesEndpoint,
    OPENAI_RESPONSES: OpenAIResponsesEndpoint,
    OPENAI_CHAT: OpenAIChatEndpoint,
}


def open_endpoint(spec: EndpointSpec) -> ModelEndpoint:
    """The wire implementation of a resolved endpoint."""
    wire = _WIRES[spec.protocol]
    if wire is AnthropicMessagesEndpoint:
        return wire(spec.api_key, spec.model, base_url=spec.base_url,
                    auth_header=spec.auth_header)
    if wire is OpenAIResponsesEndpoint:
        return wire(spec.api_key, spec.model, base_url=spec.base_url,
                    reasoning_effort=spec.reasoning_effort)
    return wire(spec.api_key, spec.model, base_url=spec.base_url,
                reasoning_effort=spec.reasoning_effort,
                extra_headers=dict(spec.extra_headers))


def resolve_endpoint(direct: Any, agent: presets.AgentProvider,
                     ) -> Optional[ModelEndpoint]:
    """The endpoint the direct seats call under these settings
    (``verifier.config.DirectSettings``) and the Claude Code seats'
    provider, or None when no raw API is reachable."""
    spec = presets.resolve_direct_endpoint(direct, agent)
    return open_endpoint(spec) if spec is not None else None
