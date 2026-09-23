"""A direct seat that cannot answer falls back to the agent seat of record."""
from verifier.llm.direct_backend import RoutingBackend
from verifier.llm.schema import RoleError


class _Direct:
    roles = {}

    def __init__(self, err):
        self.err = err
        self.calls = 0

    def complete(self, role_name, prompt, variables):
        self.calls += 1
        raise self.err


class _Agent:
    roles = {}

    def __init__(self):
        self.calls = []

    def complete(self, role_name, prompt, variables):
        self.calls.append(role_name)
        return {"answer": "from-agent"}


def test_direct_failure_routes_the_role_to_the_agent():
    direct = _Direct(RoleError("direct call failed: HTTP 402: credits"))
    agent = _Agent()
    routing = RoutingBackend(direct, agent, {"screener"})
    out = routing.complete("screener", "p", {})
    assert out == {"answer": "from-agent"}
    assert direct.calls == 1 and agent.calls == ["screener"]
    assert routing.direct_fallbacks[0]["role"] == "screener"
    assert "402" in routing.direct_fallbacks[0]["error"]


def test_non_direct_roles_never_touch_the_direct_seat():
    direct = _Direct(RoleError("unused"))
    agent = _Agent()
    routing = RoutingBackend(direct, agent, {"screener"})
    routing.complete("oracles", "p", {})
    assert direct.calls == 0 and agent.calls == ["oracles"]
    assert routing.direct_fallbacks == []
