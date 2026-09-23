"""Direct seats stay on the direct provider: outages and exhausted credit
are waited out there, never answered by the agent backend."""
import pytest

from verifier.llm import direct_backend as db
from verifier.llm.direct_backend import (AwaitingCredit, DirectBackend,
                                         RoutingBackend,
                                         TransientTransportError)
from verifier.llm.schema import RoleError


class _Agent:
    roles = {}

    def __init__(self):
        self.calls = []

    def complete(self, role_name, prompt, variables):
        self.calls.append(role_name)
        return {"answer": "from-agent"}


class _FailingDirect:
    roles = {}

    def complete(self, role_name, prompt, variables):
        raise RoleError("direct call failed: HTTP 402: credits")


def test_a_direct_failure_is_not_answered_by_the_agent():
    agent = _Agent()
    routing = RoutingBackend(_FailingDirect(), agent, {"screener"})
    with pytest.raises(RoleError):
        routing.complete("screener", "p", {})
    assert agent.calls == []


def test_non_direct_roles_run_on_the_agent():
    agent = _Agent()
    routing = RoutingBackend(_FailingDirect(), agent, {"screener"})
    assert routing.complete("oracles", "p", {}) == {"answer": "from-agent"}
    assert agent.calls == ["oracles"]


class _ScriptedTransport:
    """Raises the scripted errors in order, then answers."""
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def complete(self, *, system, prompt, schema, model, max_tokens):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return {"verdict": "ok"}, {"total_tokens": 1}


def _call(transport, monkeypatch, slept):
    monkeypatch.setattr(db.time, "sleep", lambda s: slept.append(s))
    backend = DirectBackend(transport)
    role = next(iter(backend.roles.values()))
    attempts = []
    return backend._call(role, "sys", "prompt", {"type": "object"}, None,
                         attempts), attempts


def test_exhausted_credit_is_waited_out_on_the_same_provider(monkeypatch):
    slept = []
    transport = _ScriptedTransport([AwaitingCredit("HTTP 402"),
                                    AwaitingCredit("HTTP 402")])
    payload, attempts = _call(transport, monkeypatch, slept)
    assert payload == {"verdict": "ok"} and transport.calls == 3
    assert slept == list(db.BILLING_DELAYS[:2])
    assert [a.get("awaiting_credit") for a in attempts[:2]] == [True, True]


def test_credit_that_never_returns_fails_after_the_billing_schedule(monkeypatch):
    slept = []
    transport = _ScriptedTransport(
        [AwaitingCredit("HTTP 402")] * (len(db.BILLING_DELAYS) + 1))
    with pytest.raises(RoleError, match="credit still exhausted"):
        _call(transport, monkeypatch, slept)
    assert slept == list(db.BILLING_DELAYS)


def test_connection_errors_are_transient(monkeypatch):
    import httpx
    slept = []
    transport = _ScriptedTransport([httpx.ConnectError("refused")])
    payload, _ = _call(transport, monkeypatch, slept)
    assert payload == {"verdict": "ok"} and slept == [db.TRANSIENT_DELAYS[1]]


def test_overload_keeps_its_short_schedule(monkeypatch):
    slept = []
    transport = _ScriptedTransport([TransientTransportError("HTTP 429")] * 2)
    payload, _ = _call(transport, monkeypatch, slept)
    assert payload == {"verdict": "ok"}
    assert slept == list(db.TRANSIENT_DELAYS[1:3])
