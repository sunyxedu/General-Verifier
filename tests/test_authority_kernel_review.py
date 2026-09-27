"""Independent adversarial review of authority-only contract-cache wiring.

These tests never invoke a paid backend.  They exercise only deterministic
fixtures and construction-time properties of the built-in backend wrappers.
"""
from __future__ import annotations

import copy
import json
import os

import pytest

from verifier.core.events import EventLog
from verifier.core.model import CLAIM_FINAL, Claim
from verifier.core.store import ClaimStore, TRIGGER_LAZY
from verifier.engine.contract.cache import CompiledFinalContractCache
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.agents import (
    ClaudeCodeRuntime,
    AgentRole,
    COMPILE_OBLIGATIONS_AGENT,
    CRITIC_AGENT,
)
from verifier.llm.authority_backend import (
    AuthorityBackendUnavailable,
    authority_compilation_backend,
)
from verifier.llm.codex.policy import validate_no_tools_stream
from verifier.llm.codex.runtime import CodexRuntime
from verifier.llm.schema import FixtureBackend, RoleError


INITIAL = {"obligations": [{
    "text": "The current candidate supplies the requested output.",
    "required_basis": "record",
    "severity": "must",
    "clause": "Supply the requested output.",
    "refutable_by_absence": True,
}]}
APPENDED = {"obligations": [
    *INITIAL["obligations"],
    {
        "text": "The supplied output is valid Unicode.",
        "required_basis": "record",
        "severity": "must",
        "clause": "Unicode output is accepted.",
        "refutable_by_absence": False,
    },
]}
PASS = {
    "verdict": "pass",
    "counterexample_world": "none",
    "missing_condition": "none",
}
ATTACK = {
    "verdict": "insufficient",
    "counterexample_world": "The output is present but not valid Unicode.",
    "missing_condition": "APPEND OBLIGATION: require valid Unicode.",
}


def _state(root, name, *, task="Supply the requested output.", baseline=None):
    state = root / name
    state.mkdir()
    (state / "task.md").write_text(task, encoding="utf-8")
    if baseline is not None:
        base = state / "baseline"
        base.mkdir()
        (base / "git.json").write_text(json.dumps({
            "head": "0123456789abcdef", "status": "",
        }), encoding="utf-8")
        (base / "manifest.json").write_text(json.dumps({
            "root": baseline["root"],
            "files": {"src/input.txt": baseline["identity"]},
        }), encoding="utf-8")
    return state


def _kernel(state, backend, cache, final):
    log = EventLog(str(state / "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_LAZY)
    machine = Kernel(
        store,
        log,
        backend,
        ArtifactStore(str(state / "artifacts")),
        contract_cache=cache,
        critic_rounds=2,
    )
    claim_id = store.add_claim(Claim(CLAIM_FINAL, final))
    return machine, store.get(claim_id)


def _events(machine, *kinds):
    return [event for event in machine.log.replay()
            if event["type"] in kinds]


def test_cross_candidate_repair_is_sanitized_and_hit_binds_current_claim(tmp_path):
    """The cold compiler, repair, critic and warm hit stay answer-blind."""
    critic_verdicts = iter((copy.deepcopy(ATTACK), copy.deepcopy(PASS)))

    def compile_(variables):
        return copy.deepcopy(
            APPENDED if variables["critic_feedback"] else INITIAL)

    backend = FixtureBackend({
        "obligations": compile_,
        "critic": lambda _variables: next(critic_verdicts),
    })
    cache = CompiledFinalContractCache()
    a = _state(tmp_path, "candidate-a", baseline={
        "root": "/host/candidate-a/source",
        "identity": "a" * 64,
    })
    b = _state(tmp_path, "candidate-b", baseline={
        "root": "/different/host/candidate-b/source",
        "identity": "a" * 64,
    })
    (a / "final.txt").write_text("FINAL_A_SECRET", encoding="utf-8")
    (b / "final.txt").write_text("FINAL_B_SECRET", encoding="utf-8")
    (a / "world").mkdir()
    (a / "world" / "result.txt").write_text(
        "WORLD_A_SECRET", encoding="utf-8")
    first, first_row = _kernel(a, backend, cache, "ANSWER_A_SECRET")
    second, second_row = _kernel(b, backend, cache, "ANSWER_B_SECRET")
    first.log.append("message", {"text": "LEDGER_A_SECRET"})

    first_obligations = first.contracts.compile_obligations(first_row)
    calls_after_cold_fill = len(backend.calls)
    second_obligations = second.contracts.compile_obligations(second_row)

    assert [call["role"] for call in backend.calls] == [
        "obligations", "critic", "obligations", "critic",
    ]
    assert calls_after_cold_fill == len(backend.calls)
    assert first_obligations[0].claim_id == first_row["id"]
    assert second_obligations[0].claim_id == second_row["id"]
    assert [item.text for item in first_obligations] == [
        item.text for item in second_obligations]

    forbidden = (
        "ANSWER_A_SECRET", "ANSWER_B_SECRET", "FINAL_A_SECRET",
        "FINAL_B_SECRET", "WORLD_A_SECRET", "LEDGER_A_SECRET",
        str(a), str(b), "/host/candidate-a", "/different/host/candidate-b",
    )
    for call in backend.calls:
        rendered = json.dumps(call["variables"], ensure_ascii=False)
        assert all(value not in rendered for value in forbidden)
        assert call["variables"]["world_mode"] == "authority-only"
        if call["role"] == "obligations":
            assert call["variables"]["graph"] == "[]"
            assert call["variables"]["memory_hits"] == "[]"
        else:
            assert "graph" not in call["variables"]
            assert "memory_hits" not in call["variables"]
    repair = backend.calls[2]["variables"]
    assert ATTACK["counterexample_world"] in repair["critic_feedback"]
    assert INITIAL["obligations"][0]["text"] in repair["critic_feedback"]

    assert len(_events(second, "contract_cache_hit")) == 1
    audit = json.dumps(
        _events(first, "contract_cache_miss", "contract_cache_published")
        + _events(second, "contract_cache_hit"),
        ensure_ascii=False,
    )
    assert all(value not in audit for value in forbidden)


def test_baseline_semantic_change_misses_but_host_root_change_does_not(tmp_path):
    backend = FixtureBackend({
        "obligations": lambda _variables: copy.deepcopy(INITIAL),
        "critic": copy.deepcopy(PASS),
    })
    cache = CompiledFinalContractCache()
    roots_and_identities = (
        ("a", "/host/one", "1" * 64),
        ("b", "/host/two", "1" * 64),
        ("c", "/host/three", "2" * 64),
    )
    machines = []
    for name, host_root, identity in roots_and_identities:
        state = _state(tmp_path, name, baseline={
            "root": host_root, "identity": identity,
        })
        machine, row = _kernel(state, backend, cache, f"candidate {name}")
        machine.contracts.compile_obligations(row)
        machines.append(machine)

    # a compiles, b reuses despite its irrelevant host root, c recompiles
    # because its authoritative pre-task file identity changed.
    assert [call["role"] for call in backend.calls] == [
        "obligations", "critic", "obligations", "critic",
    ]
    assert _events(machines[1], "contract_cache_hit")
    assert _events(machines[2], "contract_cache_miss")
    assert len(cache) == 2


def test_compiler_failure_releases_flight_without_legacy_or_publish(tmp_path):
    backend = FixtureBackend({
        "obligations": lambda _variables: (_ for _ in ()).throw(
            RoleError("synthetic compiler failure")),
        "critic": copy.deepcopy(PASS),
    })
    cache = CompiledFinalContractCache()
    state = _state(tmp_path, "candidate")
    machine, row = _kernel(state, backend, cache, "PRIVATE CANDIDATE ANSWER")

    with pytest.raises(RoleError, match="failed twice"):
        machine.contracts.compile_obligations(row)

    assert not cache._flights
    assert len(cache) == 0
    assert _events(machine, "contract_cache_miss")
    assert not _events(
        machine, "contract_cache_hit", "contract_cache_published",
        "contract_cache_bypass")
    # A generation failure is not retried through the candidate-aware legacy
    # path; both schema-level retries remained inside the authority boundary.
    assert len(backend.calls) == 2
    assert all(call["variables"]["world_mode"] == "authority-only"
               for call in backend.calls)
    assert all("PRIVATE CANDIDATE ANSWER" not in json.dumps(call["variables"])
               for call in backend.calls)


def test_online_revision_hits_contract_but_reexecutes_candidate_evidence(tmp_path):
    """Contract reuse must not become verdict/evidence reuse across epochs."""
    payload = {"obligations": [{
        "text": "The current candidate's delivered result is correct.",
        "required_basis": "world",
        "severity": "must",
        "clause": "Deliver a correct result.",
        "refutable_by_absence": False,
    }]}
    backend = FixtureBackend({
        "obligations": lambda _variables: copy.deepcopy(payload),
        "critic": copy.deepcopy(PASS),
    })

    class ChangingWorld:
        def __init__(self):
            self.outcomes = iter(("holds", "refuted"))
            self.calls = []

        def discharge(self, row, obligation, method=None, tag=""):
            outcome = next(self.outcomes)
            self.calls.append((row["id"], obligation.id, outcome))
            return DischargeResult(
                outcome=outcome,
                witness={"candidate": row["id"], "outcome": outcome},
                witness_kind="world_state",
                artifact="",
                instrument="synthetic-world",
                basis="world",
                strength="decisive",
            )

    state = _state(tmp_path, "online")
    store = ClaimStore(trigger=TRIGGER_LAZY)
    log = EventLog(str(state / "events.jsonl"))
    world = ChangingWorld()
    machine = Kernel(
        store,
        log,
        backend,
        ArtifactStore(str(state / "artifacts")),
        contract_cache=CompiledFinalContractCache(),
        critic_rounds=1,
        discharger=world,
    )
    first = store.add_claim(Claim(CLAIM_FINAL, "candidate version one"))
    assert machine.verify(first) == "verified"
    replacement = store.add_claim(Claim(
        CLAIM_FINAL,
        "candidate version two",
        revises=first,
    ))
    assert store.status(first) == "stale"
    assert machine.verify(replacement) == "refuted"

    assert len(world.calls) == 2
    assert world.calls[0][1] == world.calls[1][1]
    assert len([call for call in backend.calls
                if call["role"] == "obligations"]) == 1
    assert _events(machine, "contract_cache_hit")


def test_builtin_sessions_are_toolless_scratch_scoped_and_preserve_models(tmp_path):
    candidate_cwd = tmp_path / "candidate-secret-directory"
    candidate_cwd.mkdir()

    codex = CodexRuntime(
        cwd=str(candidate_cwd),
        model="base-model;obligations=compile-model;critic=review-model",
    )
    original_tools = list(COMPILE_OBLIGATIONS_AGENT.tools)
    with authority_compilation_backend(
            codex, COMPILE_OBLIGATIONS_AGENT, CRITIC_AGENT) as session:
        isolated = session.backend
        scratch = isolated._execution_cwd("obligations", {})
        assert scratch != str(candidate_cwd)
        assert os.path.isdir(scratch)
        assert os.listdir(scratch) == []
        assert isolated._model_for("obligations") == "compile-model"
        assert isolated._model_for("critic") == "review-model"
        assert isolated._model_for("other") == "base-model"
        assert session.obligations_role.tools == []
        assert session.critic_role.tools == []
        assert str(candidate_cwd) not in json.dumps(session.identity)
    assert not os.path.exists(scratch)
    assert COMPILE_OBLIGATIONS_AGENT.tools == original_tools

    agent = ClaudeCodeRuntime(
        cwd=str(candidate_cwd), model="fixed-agent-model",
        permission_mode="default",
    )
    with authority_compilation_backend(
            agent, COMPILE_OBLIGATIONS_AGENT, CRITIC_AGENT) as session:
        assert type(session.backend) is ClaudeCodeRuntime
        scratch = session.backend.cwd
        assert scratch != str(candidate_cwd)
        assert os.path.isdir(scratch)
        assert os.listdir(scratch) == []
        assert session.backend.model == "fixed-agent-model"
        assert session.backend.permission_mode == "default"
        assert all(role.tools == [] for role in (
            session.obligations_role, session.critic_role))
        assert str(candidate_cwd) not in json.dumps(session.identity)
    assert not os.path.exists(scratch)
    assert agent.cwd == str(candidate_cwd)


def test_custom_backend_and_role_subclasses_cannot_claim_the_boundary():
    class CustomBackend:
        model = "same-looking-model"

    class CustomRole(AgentRole):
        def call(self, backend, variables):  # pragma: no cover - must not run
            raise AssertionError("custom code crossed the authority boundary")

    with pytest.raises(AuthorityBackendUnavailable):
        with authority_compilation_backend(
                CustomBackend(), COMPILE_OBLIGATIONS_AGENT, CRITIC_AGENT):
            pytest.fail("custom backend entered an authority-only session")

    custom_role = CustomRole(
        name=COMPILE_OBLIGATIONS_AGENT.name,
        template=COMPILE_OBLIGATIONS_AGENT.template,
        schema=COMPILE_OBLIGATIONS_AGENT.schema,
        tools=list(COMPILE_OBLIGATIONS_AGENT.tools),
        model=COMPILE_OBLIGATIONS_AGENT.model,
        max_turns=COMPILE_OBLIGATIONS_AGENT.max_turns,
        system=COMPILE_OBLIGATIONS_AGENT.system,
    )
    with pytest.raises(AuthorityBackendUnavailable):
        with authority_compilation_backend(
                FixtureBackend({}), custom_role, CRITIC_AGENT):
            pytest.fail("custom role code entered an authority-only session")


@pytest.mark.parametrize("events", [
    # No turn start, duplicate turn start, output before the turn, and events
    # after completion must not be accepted as one complete isolated turn.
    ["thread", "complete"],
    ["thread", "turn", "turn", "complete"],
    ["thread", "item", "turn", "complete"],
    ["thread", "turn", "complete", "item"],
])
def test_authority_stream_requires_one_ordered_turn(events):
    mapping = {
        "thread": {"type": "thread.started", "thread_id": "synthetic"},
        "turn": {"type": "turn.started"},
        "item": {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "{}"},
        },
        "complete": {"type": "turn.completed", "usage": {}},
    }
    raw = "\n".join(json.dumps(mapping[name]) for name in events)
    with pytest.raises(AuthorityBackendUnavailable):
        validate_no_tools_stream(raw)
