"""The seat chain: each concern of a seat call is one stage, run in order."""
import pytest

from verifier.llm import seat_runner
from verifier.llm.agents import AgentRole
from verifier.llm.runtime import RuntimeDecorator
from verifier.llm.schema import (
    FixtureBackend, RoleError, SeatResourceKilled, SeatTimeout,
)
from verifier.llm.seat_runner import (
    FeedbackRetry, RoutingRuntime, SeatRunner, dispatch,
)


class _Recording:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def run(self, attempt, proceed):
        self.events.append(("enter", self.name, attempt.index))
        try:
            return proceed(attempt)
        finally:
            self.events.append(("exit", self.name, attempt.index))


def test_stages_wrap_the_dispatch_outermost_first():
    events = []
    backend = FixtureBackend({"critic": {"ok": True}})
    runner = SeatRunner([_Recording("a", events), _Recording("b", events)])
    role = AgentRole("critic", "t={x}", {"ok": bool})
    assert runner.call(role, backend, {"x": 1}) == {"ok": True}
    assert events == [("enter", "a", 1), ("enter", "b", 1),
                      ("exit", "b", 1), ("exit", "a", 1)]
    # no binding stage: the terminal renders and passes the caller's variables
    assert backend.calls == [{"role": "critic", "variables": {"x": 1}}]


def test_feedback_retry_feeds_the_error_back_once_then_fails_twice():
    prompts = []

    class Backend:
        def complete(self, role_name, prompt, variables):
            prompts.append(prompt)
            return {"ok": "no"}

    role = AgentRole("critic", "t={x}", {"ok": bool})
    with pytest.raises(RoleError, match="critic: failed twice"):
        role.call(Backend(), {"x": 1})
    assert prompts[0] == "t=1"
    assert prompts[1].startswith(
        "t=1\n\n[previous attempt failed; fix it and retry] $.ok")


@pytest.mark.parametrize("error, attempts, fed_back", [
    (SeatTimeout("spent"), 1, False),
    (SeatResourceKilled("killed"), 2, False),
])
def test_budget_and_killed_process_are_not_fed_back(error, attempts, fed_back):
    prompts = []

    class Backend:
        def complete(self, role_name, prompt, variables):
            prompts.append(prompt)
            raise error

    role = AgentRole("critic", "t", {"ok": bool})
    with pytest.raises(type(error) if attempts == 1 else RoleError):
        role.call(Backend(), {})
    assert len(prompts) == attempts
    assert all("previous attempt failed" not in p for p in prompts)


def test_output_validation_lowers_the_wire_schema_only_for_a_wire_runtime():
    role = AgentRole("oracles", "t", {"methods": [str]},
                     wire_schema={"wire": [str]},
                     wire_lower=lambda payload: {"methods": payload["wire"]})
    wire = FixtureBackend({"oracles": {"wire": ["m"]}}, wire_output=True)
    plain = FixtureBackend({"oracles": {"methods": ["m"]}})
    assert role.call(wire, {}) == {"methods": ["m"]}
    assert role.call(plain, {}) == {"methods": ["m"]}


def test_the_default_runner_holds_a_gate_permit_around_the_runtime_call(
        monkeypatch):
    held = []

    class Gate:
        def __init__(self):
            self.inside = False

        def slot(self):
            gate = self

            class Slot:
                def __enter__(self):
                    gate.inside = True

                def __exit__(self, *exc):
                    gate.inside = False
            return Slot()

    gate = Gate()
    monkeypatch.setattr(seat_runner.SEAT_GATE, "slot", gate.slot)

    class Backend:
        def complete(self, role_name, prompt, variables):
            held.append(gate.inside)
            return {"ok": True}

    AgentRole("critic", "t", {"ok": bool}).call(Backend(), {})
    assert held == [True] and not gate.inside


def test_the_retry_policy_and_terminal_are_replaceable():
    class Once(FeedbackRetry):
        def run(self, call, attempt):
            return attempt(seat_runner.SeatAttempt(call, 1))

    seen = []

    def terminal(attempt):
        seen.append(attempt.call.role.name)
        return dispatch(attempt)

    runner = SeatRunner([], retry=Once(), terminal=terminal)
    backend = FixtureBackend({"critic": {"ok": True}})
    assert runner.call(AgentRole("critic", "t", {"ok": bool}), backend, {}) \
        == {"ok": True}
    assert seen == ["critic"]


class _Agent:
    model, effort, cwd = "m", "high", "/state"
    roles = {}

    def complete(self, role_name, prompt, variables):
        return {"by": "agent"}


class _Direct:
    roles = {}

    def complete(self, role_name, prompt, variables):
        return {"by": "direct"}


def test_routing_dispatches_by_seat_and_reads_only_declared_state():
    routing = RoutingRuntime(_Direct(), _Agent(), {"critic"})
    assert isinstance(routing, RuntimeDecorator)
    assert routing.complete("critic", "p", {}) == {"by": "direct"}
    assert routing.complete("judger", "p", {}) == {"by": "agent"}
    assert (routing.model, routing.effort, routing.cwd) == ("m", "high", "/state")
    # seat-of-record state the agent lacks is absent, not invented
    assert not hasattr(routing, "service_tier")
    # anything else is not read through to the agent any more
    assert not hasattr(routing, "roles_extra")
    routing.model = "shadow"
    assert routing.model == "shadow" and routing.agent.model == "m"
    assert routing.reuse_identity() == {"model": "m", "effort": "high"}
