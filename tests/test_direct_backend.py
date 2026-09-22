"""Direct model calls for tool-free seats: routing, contract, receipts."""
import json
import os
from types import SimpleNamespace

import httpx
import pytest

from verifier.llm import direct_backend as db
from verifier.llm.agents import ALL_AGENT_ROLES, AgentRole
from verifier.llm.schema import FixtureBackend, RoleError


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
    transport = FakeTransport([{"confidence": "high", "grounds": "g", "reason": "r"}])
    direct = db.DirectBackend(transport, cwd=str(tmp_path))
    agent = FixtureBackend({"discharge": {"outcome": "holds"}})
    agent.roles = dict(ALL_AGENT_ROLES)
    routing = db.RoutingBackend(direct, agent, db.DEFAULT_DIRECT_ROLES)

    assert routing.routes_direct("screener") and not routing.routes_direct("discharge")
    assert routing.complete("screener", "p", {}) == {
        "confidence": "high", "grounds": "g", "reason": "r"}
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
    transport = FakeTransport([db.TransientTransportError("HTTP 529"),
                               {"confidence": "low", "grounds": "g", "reason": "r"}])
    direct = db.DirectBackend(transport, cwd=str(tmp_path))
    role = ALL_AGENT_ROLES["screener"]
    payload = role.call(direct, {name: "" for name in _variables(role)})
    assert payload["confidence"] == "low"
    rows = [json.loads(l) for l in open(tmp_path / "role_calls.jsonl")]
    assert rows[-1]["backend"] == "direct" and rows[-1]["attempts"] == 2


def test_schema_violations_are_fed_back_through_the_role_retry(tmp_path):
    transport = FakeTransport([{"confidence": "very", "grounds": "g", "reason": "r"},
                               {"confidence": "medium", "grounds": "g", "reason": "r"}])
    direct = db.DirectBackend(transport, cwd=str(tmp_path))
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
    transport = db.AnthropicMessagesTransport("key", "m", base_url="https://x.test")
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
    transport = db.AnthropicMessagesTransport("key", "m", base_url="https://x.test")
    with pytest.raises(db.TransientTransportError):
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
    assert db.direct_transport({}) is None
    monkeypatch.setattr(provider, "API_KEY", "sk-test")
    monkeypatch.setattr(provider, "ANTHROPIC_BASE_URL", None)
    monkeypatch.setattr(provider, "MODEL", "claude-sonnet-5")
    transport = db.direct_transport({})
    assert isinstance(transport, db.AnthropicMessagesTransport)
    assert transport.model == "claude-sonnet-5"


def _variables(role: AgentRole):
    import re
    return set(re.findall(r"{(\w+)}", role.template))
