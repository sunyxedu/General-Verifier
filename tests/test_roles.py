import pytest

from verifier.roles import (
    Role, RoleError, FixtureBackend, validate, COMPILE_OBLIGATIONS,
)


def test_validate_accepts_and_rejects():
    schema = {"obligations": [{"text": str, "failure_looks_like": str}]}
    validate(schema, {"obligations": [{"text": "t", "failure_looks_like": "f"}]})
    with pytest.raises(RoleError, match="missing"):
        validate(schema, {"obligations": [{"text": "t"}]})
    with pytest.raises(RoleError, match="expected list"):
        validate(schema, {"obligations": "nope"})
    with pytest.raises(RoleError, match="expected str"):
        validate(schema, {"obligations": [{"text": 1, "failure_looks_like": "f"}]})


def test_role_call_with_fixture():
    backend = FixtureBackend(
        {
            "compile_obligations": {
                "obligations": [
                    {"text": "O1", "failure_looks_like": "一个反例"},
                    {"text": "优雅", "failure_looks_like": ""},
                ]
            }
        }
    )
    out = COMPILE_OBLIGATIONS.call(
        backend, {"claim_text": "6 | n^3-n for n<=100", "memory_hits": "[]"}
    )
    assert len(out["obligations"]) == 2
    assert backend.calls[0]["role"] == "compile_obligations"


def test_role_retries_once_then_raises():
    attempts = []

    def flaky(variables):
        attempts.append(1)
        return {"wrong_key": True}

    backend = FixtureBackend({"compile_obligations": flaky})
    with pytest.raises(RoleError, match="after retry"):
        COMPILE_OBLIGATIONS.call(backend, {"claim_text": "x", "memory_hits": "[]"})
    assert len(attempts) == 2  # 一次重试，带着错误信息


def test_retry_recovers_if_second_attempt_valid():
    state = {"n": 0}

    def eventually_valid(variables):
        state["n"] += 1
        if state["n"] == 1:
            return {"oops": 1}
        return {"obligations": []}

    backend = FixtureBackend({"compile_obligations": eventually_valid})
    out = COMPILE_OBLIGATIONS.call(backend, {"claim_text": "x", "memory_hits": "[]"})
    assert out == {"obligations": []}


def test_template_renders_variables():
    role = Role("r", "claim: {claim_text}", {"x": int})
    assert role.render({"claim_text": "abc"}) == "claim: abc"
