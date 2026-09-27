"""Direct model calls for tool-free seats: routing, contract, receipts."""
import json
import os
from types import SimpleNamespace

import httpx
import pytest

from verifier.llm import direct_backend as db
from verifier.llm import endpoints
from verifier.llm.agents import ALL_AGENT_ROLES, AgentRole
from verifier.llm.schema import FixtureBackend, RoleError
from verifier.llm.seat_runner import RoutingRuntime


class FakeTransport:
    name = "fake"
    model = "fake-model"

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def complete(self, *, system, prompt, schema, model=None, max_tokens=0):
        self.calls.append({"system": system, "prompt": prompt,
                           "schema": schema, "model": model})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer, {"input_tokens": 10, "output_tokens": 5}


def test_tool_free_roles_route_direct_and_others_stay_on_the_agent(tmp_path):
    transport = FakeTransport([{"verdict": "open", "confidence": "high", "basis": "world", "grounds": "g", "reason": "r"}])
    direct = db.DirectRuntime(transport, cwd=str(tmp_path))
    agent = FixtureBackend({"discharge": {"outcome": "holds"}})
    agent.roles = dict(ALL_AGENT_ROLES)
    routing = RoutingRuntime(direct, agent, db.DEFAULT_DIRECT_ROLES)

    assert routing.routes_direct("screener") and not routing.routes_direct("discharge")
    assert routing.complete("screener", "p", {}) == {
        "verdict": "open", "confidence": "high", "basis": "world",
        "grounds": "g", "reason": "r"}
    assert routing.complete("discharge", "p", {}) == {"outcome": "holds"}
    assert transport.calls[0]["schema"]["type"] == "object"
    assert "no tools" in transport.calls[0]["system"]
    # the kernel re-registers harness roles on the backend; both sides see it
    routing.roles = dict(ALL_AGENT_ROLES, extra=ALL_AGENT_ROLES["screener"])
    assert "extra" in direct.roles and "extra" in agent.roles
    # oracle catalog laziness is impossible without tools
    assert routing.allows_lazy_oracle_catalog("screener", {}) is False


def test_direct_backend_writes_a_receipt_and_retries_transient_failures(
        tmp_path, monkeypatch):
    monkeypatch.setattr(db, "TRANSIENT_DELAYS", (0, 0, 0))
    transport = FakeTransport([endpoints.TransientTransportError("HTTP 529"),
                               {"verdict": "open", "confidence": "low", "basis": "world", "grounds": "g", "reason": "r"}])
    direct = db.DirectRuntime(transport, cwd=str(tmp_path))
    role = ALL_AGENT_ROLES["screener"]
    payload = role.call(direct, {name: "" for name in _variables(role)})
    assert payload["confidence"] == "low"
    rows = [json.loads(l) for l in open(tmp_path / "role_calls.jsonl")]
    assert rows[-1]["backend"] == "direct" and rows[-1]["attempts"] == 2


def test_schema_violations_are_fed_back_through_the_role_retry(tmp_path):
    transport = FakeTransport([{"confidence": "very", "grounds": "g", "reason": "r"},
                               {"verdict": "open", "confidence": "medium", "basis": "world", "grounds": "g", "reason": "r"}])
    direct = db.DirectRuntime(transport, cwd=str(tmp_path))
    role = ALL_AGENT_ROLES["screener"]
    payload = role.call(direct, {name: "" for name in _variables(role)})
    assert payload["confidence"] == "medium"
    assert "[previous attempt failed" in transport.calls[1]["prompt"]


def test_anthropic_transport_forces_the_emit_tool_and_reads_its_input(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "thinking"},
                        {"type": "tool_use", "name": "emit",
                         "input": {"ok": True}}],
            "usage": {"input_tokens": 3, "output_tokens": 2}})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        transport=httpx.MockTransport(handler), **kw))
    transport = endpoints.AnthropicMessagesEndpoint("key", "m", base_url="https://x.test")
    payload, usage = transport.complete(system="s", prompt="p",
                                        schema={"type": "object"})
    assert payload == {"ok": True} and usage["input_tokens"] == 3
    assert seen["body"]["tool_choice"] == {"type": "tool", "name": "emit"}
    assert seen["body"]["tools"][0]["input_schema"] == {"type": "object"}
    assert seen["headers"]["x-api-key"] == "key"


def test_anthropic_transport_classifies_statuses(monkeypatch):
    def handler(request):
        return httpx.Response(429, text="slow down")
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        transport=httpx.MockTransport(handler), **kw))
    transport = endpoints.AnthropicMessagesEndpoint("key", "m", base_url="https://x.test")
    with pytest.raises(endpoints.TransientTransportError):
        transport.complete(system="s", prompt="p", schema={"type": "object"})


def test_direct_seat_roles_from_env():
    assert db.direct_seat_roles({}) == set(db.DEFAULT_DIRECT_ROLES)
    assert db.direct_seat_roles({"GV_DIRECT_SEATS": "off"}) == set()
    assert db.direct_seat_roles({"GV_DIRECT_SEATS": "critic, oracles"}) == {
        "critic", "oracles"}


def test_login_only_credentials_offer_no_direct_transport(monkeypatch):
    from verifier.llm import provider
    monkeypatch.setattr(provider, "PROVIDER", "anthropic")
    monkeypatch.setattr(provider, "API_KEY", None)
    assert db.direct_endpoint({}) is None
    monkeypatch.setattr(provider, "API_KEY", "sk-test")
    monkeypatch.setattr(provider, "ANTHROPIC_BASE_URL", None)
    monkeypatch.setattr(provider, "MODEL", "claude-sonnet-5")
    transport = db.direct_endpoint({})
    assert isinstance(transport, endpoints.AnthropicMessagesEndpoint)
    assert transport.model == "claude-sonnet-5"


def test_openai_chat_transport_asks_for_strict_json_and_parses_it(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps({"ok": True})}}], "usage": {"prompt_tokens": 4}})
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        transport=httpx.MockTransport(handler), **kw))
    transport = endpoints.OpenAIChatEndpoint("or-key", "openai/gpt-5.6-luna",
                                       base_url="https://router.test/api",
                                       reasoning_effort="medium")
    payload, usage = transport.complete(system="s", prompt="p",
                                        schema={"type": "object"})
    assert payload == {"ok": True} and usage["prompt_tokens"] == 4
    assert seen["auth"] == "Bearer or-key"
    assert seen["body"]["response_format"]["json_schema"]["strict"] is True
    assert seen["body"]["reasoning"] == {"effort": "medium"}
    assert seen["body"]["messages"][0] == {"role": "system", "content": "s"}


def test_openrouter_direct_transport_from_env():
    transport = db.direct_endpoint({"GV_DIRECT_PROVIDER": "openrouter",
                                     "OPENROUTER_API_KEY": "k",
                                     "GV_DIRECT_MODEL": "openai/gpt-5.6-luna"})
    assert isinstance(transport, endpoints.OpenAIChatEndpoint)
    assert transport.model == "openai/gpt-5.6-luna"
    assert db.direct_endpoint({"GV_DIRECT_PROVIDER": "openrouter"}) is None


def test_strict_schema_round_trips_optional_keys():
    from verifier.llm.agents import to_json_schema
    role = ALL_AGENT_ROLES["oracles"]
    schema = to_json_schema(role.wire_schema or role.schema)
    strict = endpoints.strict_schema(schema)
    method = strict["properties"]["methods"]["items"]
    assert "script" in method["required"]
    assert {"type": "null"} in method["properties"]["script"]["anyOf"]
    payload = {"methods": [{"procedure": "p", "script": None}]}
    assert endpoints.strip_optional_nulls(payload, schema) == {"methods": [{"procedure": "p"}]}
    kept = {"methods": [{"procedure": "p", "script": "echo"}]}
    assert endpoints.strip_optional_nulls(kept, schema) == kept


def test_build_backend_wraps_an_injected_agent_backend(monkeypatch):
    env = {"GV_DIRECT_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "k"}
    codex_like = FixtureBackend({"discharge": {"outcome": "holds"}})
    codex_like.roles = dict(ALL_AGENT_ROLES)
    backend = db.build_backend(dict(ALL_AGENT_ROLES), cwd=None, model="m",
                               environ=env, agent=codex_like)
    assert isinstance(backend, RoutingRuntime) and backend.agent is codex_like
    assert backend.routes_direct("critic") and not backend.routes_direct("discharge")
    plain = db.build_backend(dict(ALL_AGENT_ROLES), cwd=None, model="m",
                             environ={"GV_DIRECT_SEATS": "off"}, agent=codex_like)
    assert plain is codex_like


def test_empty_answer_with_exhausted_budget_retries_with_a_larger_budget(monkeypatch):
    # a start below the ceiling (a model with a higher one) still retries
    monkeypatch.setattr(db, "DEFAULT_MAX_TOKENS", db.MAX_OUTPUT_TOKENS // 4)
    calls = []

    def handler(request):
        body = json.loads(request.content); calls.append(body["max_tokens"])
        if len(calls) == 1:
            return httpx.Response(200, json={"choices": [{"finish_reason": "length",
                "message": {"content": None, "reasoning": "x" * 50}}]})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
            "message": {"content": json.dumps({"verdict": "open", "confidence": "low", "basis": "world", "grounds": "g", "reason": "r"})}}],
            "usage": {}})
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        transport=httpx.MockTransport(handler), **kw))
    transport = endpoints.OpenAIChatEndpoint("k", "m", base_url="https://x.test")
    direct = db.DirectRuntime(transport)
    role = ALL_AGENT_ROLES["screener"]
    out = role.call(direct, {name: "" for name in _variables(role)})
    assert out["confidence"] == "low"
    assert calls == [db.DEFAULT_MAX_TOKENS, 2 * db.DEFAULT_MAX_TOKENS]


def test_a_role_that_outgrew_the_budget_starts_at_the_budget_it_needed(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DEFAULT_MAX_TOKENS", db.MAX_OUTPUT_TOKENS // 4)
    calls = []
    answer = {"choices": [{"finish_reason": "stop",
        "message": {"content": json.dumps({"verdict": "open", "confidence": "low", "basis": "world", "grounds": "g", "reason": "r"})}}],
        "usage": {"completion_tokens": 40000}}

    def handler(request):
        body = json.loads(request.content); calls.append(body["max_tokens"])
        if len(calls) == 1:
            return httpx.Response(200, json={"choices": [{"finish_reason": "length",
                "message": {"content": None}}], "usage": {"completion_tokens": 32000}})
        return httpx.Response(200, json=answer)
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        transport=httpx.MockTransport(handler), **kw))
    transport = endpoints.OpenAIChatEndpoint("k", "m", base_url="https://x.test")
    direct = db.DirectRuntime(transport, cwd=str(tmp_path))
    role = ALL_AGENT_ROLES["screener"]
    variables = {name: "" for name in _variables(role)}
    role.call(direct, variables)
    role.call(direct, variables)
    # the second call pays no truncated attempt
    assert calls == [db.DEFAULT_MAX_TOKENS, 2 * db.DEFAULT_MAX_TOKENS,
                     2 * db.DEFAULT_MAX_TOKENS]
    receipts = [json.loads(l) for l in open(tmp_path / "role_calls.jsonl")]
    # the truncated attempt was billed: its usage is on the receipt
    assert receipts[0]["usage"][0] == {"completion_tokens": 32000}
    assert receipts[1]["attempts"] == 1


def test_a_call_starts_at_the_model_ceiling_and_a_truncation_there_is_final(monkeypatch):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content)["max_tokens"])
        return httpx.Response(200, json={"choices": [{"finish_reason": "length",
            "message": {"content": None}}]})
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        transport=httpx.MockTransport(handler), **kw))
    direct = db.DirectRuntime(endpoints.OpenAIChatEndpoint("k", "m", base_url="https://x.test"))
    role = ALL_AGENT_ROLES["screener"]
    with pytest.raises(RoleError, match="output budget exhausted"):
        role.call(direct, {name: "" for name in _variables(role)})
    # never a smaller first attempt; the role's own retry asks at the ceiling too
    assert calls and set(calls) == {db.MAX_OUTPUT_TOKENS}


def test_authority_compilation_runs_on_the_direct_transport_of_a_routing_backend(tmp_path):
    from verifier.llm.authority_backend import authority_compilation_backend
    transport = FakeTransport([])
    direct = db.DirectRuntime(transport, cwd=str(tmp_path))
    agent = FixtureBackend({}); agent.roles = dict(ALL_AGENT_ROLES)
    routing = RoutingRuntime(direct, agent, db.DEFAULT_DIRECT_ROLES)
    with authority_compilation_backend(routing, ALL_AGENT_ROLES["obligations"],
                                       ALL_AGENT_ROLES["critic"]) as session:
        assert isinstance(session.backend, db.DirectRuntime)
        assert session.backend.endpoint is transport
        assert session.obligations_role.tools == [] and "obligations" in session.backend.roles


def _variables(role: AgentRole):
    import re
    return set(re.findall(r"{(\w+)}", role.template))
