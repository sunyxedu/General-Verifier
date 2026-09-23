"""A seat that exhausted its time budget is not called again with the same
prompt; a malformed answer still gets its one retry with feedback."""
from __future__ import annotations

import pytest

from verifier.llm.agents import AgentRole
from verifier.llm.schema import RoleError, SeatTimeout


class Backend:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def complete(self, role_name, prompt, variables):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return {"ok": True}


def role():
    return AgentRole("discharge", "check it", {"ok": bool})


def test_a_timed_out_seat_is_not_rerun():
    backend = Backend([SeatTimeout("discharge: codex exec timed out after 1800s")])
    with pytest.raises(SeatTimeout):
        role().call(backend, {})
    assert backend.calls == 1


def test_a_malformed_answer_still_gets_its_retry():
    backend = Backend([RoleError("schema: missing field ok")])
    assert role().call(backend, {}) == {"ok": True}
    assert backend.calls == 2
