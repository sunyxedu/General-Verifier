"""Append-only Compiler repair protocol; local fixture tests only."""
from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from experiments.terminal_bench.validate_prepared_contract_entry import (
    validate_prepared_contract_entry,
)
from test_contract_replay import service, snapshot
from test_prepare_contract_entry import harness
from verifier import offline
from verifier.contract_replay import compile_input_digests, replay_identity
from verifier.core.events import EventLog
from verifier.core.model import CLAIM_ASSERTION, Claim, content_hash
from verifier.core.store import ClaimStore, TRIGGER_LAZY
from verifier.engine.contract_cache import contract_cache_key
from verifier.engine.kernel import Kernel
from verifier.config import (
    APPEND_ONLY_DELTA_PROTOCOL,
    append_only_deltas_enabled,
)
from verifier.execution.instruments import ArtifactStore
from verifier.llm.agents import COMPILE_OBLIGATIONS_AGENT, CRITIC_AGENT
from verifier.llm.codex_backend import CodexRuntime
from verifier.llm.schema import FixtureBackend


OLD = {
    "text": "The requested report exists.",
    "required_basis": "world",
    "severity": "must",
    "clause": "Produce the requested report.",
    "refutable_by_absence": True,
}
TAIL = {
    "text": "The complete report is valid UTF-8.",
    "required_basis": "world",
    "severity": "must",
    "clause": "The report must be valid UTF-8.",
    "refutable_by_absence": False,
}
CHANGED_LIKE = {
    **OLD,
    "text": "The requested report exists and is nonempty.",
    "clause": "Produce a nonempty requested report.",
}
REPLACEMENT = {
    **OLD,
    "text": "The report is present at the required logical address.",
}
INITIAL = {"obligations": [OLD]}
DELTA = {"obligations": [TAIL]}
FULL = {"obligations": [OLD, TAIL]}
PASS = {
    "verdict": "pass",
    "counterexample_world": "none",
    "missing_condition": "none",
}
APPEND_ATTACK = {
    "verdict": "insufficient",
    "counterexample_world": "The report exists but has invalid encoding.",
    "missing_condition": "APPEND OBLIGATION: require valid UTF-8.",
}


class RecordingBackend(FixtureBackend):
    def __init__(self, obligations, critics):
        obligations = iter(copy.deepcopy(obligations))
        critics = iter(copy.deepcopy(critics))
        super().__init__({
            "obligations": lambda _variables: next(obligations),
            "critic": lambda _variables: next(critics),
        })
        self.prompts = []

    def complete(self, role_name, prompt, variables):
        self.prompts.append({"role": role_name, "prompt": prompt})
        return super().complete(role_name, prompt, variables)


def _machine(tmp_path, monkeypatch, obligations, critics, *, enabled=True,
             rounds=3, name="state", kind=CLAIM_ASSERTION, cache=None):
    if enabled is None:
        monkeypatch.delenv("GV_APPEND_ONLY_DELTAS", raising=False)
    else:
        monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "1" if enabled else "0")
    state = tmp_path / name
    state.mkdir()
    (state / "task.md").write_text(
        "Produce the requested report as valid UTF-8.", encoding="utf-8")
    backend = RecordingBackend(obligations, critics)
    store = ClaimStore(trigger=TRIGGER_LAZY)
    machine = Kernel(
        store,
        EventLog(str(state / "events.jsonl")),
        backend,
        ArtifactStore(str(state / "artifacts")),
        critic_rounds=rounds,
        contract_cache=cache,
    )
    cid = store.add_claim(Claim(kind, "The requested report is complete."))
    return machine, store.get(cid), backend


def _roles(backend):
    return [call["role"] for call in backend.calls]


def _events(machine, kind):
    return [event for event in machine.log.replay()
            if event["type"] == kind]


def _public(obligations):
    return [{
        "text": item.text,
        "required_basis": item.required_basis,
        "severity": item.severity,
        "clause": item.clause,
        "refutable_by_absence": item.refutable_by_absence,
    } for item in obligations]


def test_append_delta_keeps_ococ_and_second_critic_gets_complete_contract(
        tmp_path, monkeypatch):
    original, delta = copy.deepcopy(INITIAL), copy.deepcopy(DELTA)
    machine, row, backend = _machine(
        tmp_path, monkeypatch, [original, delta], [APPEND_ATTACK, PASS])

    obligations = machine._compile_obligations(row)

    assert _roles(backend) == ["obligations", "critic", "obligations", "critic"]
    assert json.loads(backend.calls[3]["variables"]["obligations_json"]) == \
        FULL["obligations"]
    assert _public(obligations) == FULL["obligations"]
    assert backend.calls[2]["variables"]["critic_feedback"].startswith(
        "A critic found a sufficiency gap which it classified as an append-only")
    assert APPEND_ONLY_DELTA_PROTOCOL in backend.prompts[2]["prompt"]
    assert original == INITIAL
    assert delta == DELTA


def test_delta_merge_deep_copies_old_and_new_payloads_across_threads():
    previous, delta = copy.deepcopy(INITIAL), copy.deepcopy(DELTA)
    with ThreadPoolExecutor(max_workers=8) as pool:
        merged = list(pool.map(
            lambda _index: Kernel._merge_append_delta(previous, delta),
            range(24),
        ))

    merged[0]["obligations"][0]["text"] = "mutated result"
    merged[0]["obligations"][1]["clause"] = "mutated result"
    previous["obligations"][0]["severity"] = "should"
    delta["obligations"][0]["severity"] = "should"

    assert all(item == FULL for item in merged[1:])
    assert len({id(item["obligations"][0]) for item in merged}) == len(merged)
    assert len({id(item["obligations"][1]) for item in merged}) == len(merged)


@pytest.mark.parametrize("bad_delta", [
    {"obligations": []},
    copy.deepcopy(FULL),
])
def test_unambiguous_bad_delta_is_rejected_then_retried_within_budget(
        tmp_path, monkeypatch, bad_delta):
    machine, row, backend = _machine(
        tmp_path, monkeypatch,
        [INITIAL, bad_delta, DELTA], [APPEND_ATTACK, PASS], rounds=3)

    obligations = machine._compile_obligations(row)

    assert _roles(backend) == [
        "obligations", "critic", "obligations", "obligations", "critic"]
    assert _public(obligations) == FULL["obligations"]
    guard = [event for event in _events(machine, "decomposition_attacked")
             if event["payload"].get("source") == "mechanical_append_guard"]
    assert len(guard) == 1
    assert guard[0]["payload"]["repair_attempt"] == 1
    assert "MECHANICAL APPEND-DELTA VIOLATION" in \
        backend.calls[3]["variables"]["critic_feedback"]
    assert json.loads(backend.calls[-1]["variables"]["obligations_json"]) == \
        FULL["obligations"]


def test_invalid_delta_retries_are_bounded_and_exhaustion_fails_closed(
        tmp_path, monkeypatch):
    machine, row, backend = _machine(
        tmp_path, monkeypatch,
        [INITIAL, {"obligations": []}, FULL], [APPEND_ATTACK], rounds=3)

    assert machine._compile_obligations(row) is None
    assert _roles(backend) == [
        "obligations", "critic", "obligations", "obligations"]
    guard = [event for event in _events(machine, "decomposition_attacked")
             if event["payload"].get("source") == "mechanical_append_guard"]
    assert [event["payload"]["repair_attempt"] for event in guard] == [1, 2]
    evidence = machine.store.evidence_for(row["id"])
    assert evidence[-1]["outcome"] == "failed"
    assert machine.store.status(row["id"]) == "unverified"


def test_changed_old_like_tail_cannot_replace_old_and_fresh_critic_can_reject(
        tmp_path, monkeypatch):
    second_attack = {
        "verdict": "insufficient",
        "counterexample_world": "The added condition is not source-required.",
        "missing_condition": "REMOVE: the overstrong added condition.",
    }
    machine, row, backend = _machine(
        tmp_path, monkeypatch,
        [INITIAL, {"obligations": [CHANGED_LIKE]}],
        [APPEND_ATTACK, second_attack], rounds=2)

    assert machine._compile_obligations(row) is None
    assert _roles(backend) == ["obligations", "critic", "obligations", "critic"]
    reviewed = json.loads(backend.calls[-1]["variables"]["obligations_json"])
    assert reviewed == [OLD, CHANGED_LIKE]
    assert reviewed[0] == OLD


@pytest.mark.parametrize("instruction", [
    "ADD: require a path binding.",
    "REMOVE: the prior condition is overstrong.",
])
def test_non_append_repairs_keep_complete_replacement_semantics(
        tmp_path, monkeypatch, instruction):
    attack = {
        "verdict": "insufficient",
        "counterexample_world": "The original contract needs a rewrite.",
        "missing_condition": instruction,
    }
    replacement = {"obligations": [REPLACEMENT]}
    machine, row, backend = _machine(
        tmp_path, monkeypatch, [INITIAL, replacement], [attack, PASS])

    obligations = machine._compile_obligations(row)

    assert _roles(backend) == ["obligations", "critic", "obligations", "critic"]
    assert _public(obligations) == replacement["obligations"]
    assert json.loads(backend.calls[-1]["variables"]["obligations_json"]) == \
        replacement["obligations"]
    assert APPEND_ONLY_DELTA_PROTOCOL not in backend.prompts[2]["prompt"]


def test_default_off_keeps_legacy_full_append_repair(tmp_path, monkeypatch):
    machine, row, backend = _machine(
        tmp_path, monkeypatch, [INITIAL, FULL], [APPEND_ATTACK, PASS],
        enabled=None)

    obligations = machine._compile_obligations(row)

    assert machine.append_only_deltas is False
    assert machine.append_only_delta_protocol is None
    assert _roles(backend) == ["obligations", "critic", "obligations", "critic"]
    assert _public(obligations) == FULL["obligations"]
    assert APPEND_ONLY_DELTA_PROTOCOL not in backend.prompts[2]["prompt"]


@pytest.mark.parametrize(("value", "expected"), [
    (None, False), ("0", False), ("1", True),
])
def test_delta_flag_has_strict_explicit_semantics(value, expected):
    environ = {} if value is None else {"GV_APPEND_ONLY_DELTAS": value}
    assert append_only_deltas_enabled(environ) is expected


@pytest.mark.parametrize("value", ["", "true", "false", "2", " 1", "1 "])
def test_delta_flag_rejects_ambiguous_values(value):
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        append_only_deltas_enabled({"GV_APPEND_ONLY_DELTAS": value})


def test_offline_and_contract_cache_identities_separate_delta_protocol(
        monkeypatch):
    monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "0")
    off_config = offline.config_fields()
    monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "1")
    on_config = offline.config_fields()
    assert off_config["append_only_deltas"] is False
    assert off_config["append_only_delta_protocol"] is None
    assert on_config["append_only_deltas"] is True
    assert on_config["append_only_delta_protocol"] == APPEND_ONLY_DELTA_PROTOCOL
    assert off_config["config"] != on_config["config"]

    common = {
        "view": SimpleNamespace(identity={"source": "fixed"}),
        "variables": {"claim_text": "fixed", "task_context": "fixed"},
        "backend_identity": {"backend": "fixed"},
        "obligations_role": COMPILE_OBLIGATIONS_AGENT,
        "critic_role": CRITIC_AGENT,
        "critic_rounds": 4,
    }
    off_key = contract_cache_key(
        **common, append_only_deltas=False,
        append_only_delta_protocol=None)
    on_key = contract_cache_key(
        **common, append_only_deltas=True,
        append_only_delta_protocol=APPEND_ONLY_DELTA_PROTOCOL)
    assert off_key != on_key


def test_local_codex_delta_prepare_and_independent_consumer_round_trip(
        tmp_path, monkeypatch):
    """The exported full payload stays bound although the last Compiler sent a delta."""
    # Reuse the existing local Codex JSONL protocol stub.  Its initial helper
    # identity is deliberately unused because this test enables a distinct
    # compiler protocol after the harness has installed the stub.
    local = harness(
        tmp_path, monkeypatch,
        responses=[INITIAL, APPEND_ATTACK, DELTA, PASS],
    )
    monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "1")

    services = []

    def factory(state):
        sc = service(
            state,
            CodexRuntime(cwd=str(state), model="gpt-5.6-luna"),
        )
        services.append(sc)
        return sc

    authority = factory(local.source)
    graph = snapshot(authority)
    config = offline.config_fields(
        model="gpt-5.6-luna", seats="codex", effort="medium",
        service_tier="standard",
    )
    assert config["append_only_deltas"] is True
    assert config["append_only_delta_protocol"] == APPEND_ONLY_DELTA_PROTOCOL
    identity = replay_identity(authority.kernel, graph, config)
    inputs = compile_input_digests(authority.kernel, graph)
    claim_id = authority.builder.labels["A"]

    from experiments.terminal_bench.prepare_contract_entry import (
        prepare_contract_entry,
    )

    destination = (tmp_path / "delta-prepare").absolute()
    produced = prepare_contract_entry(
        local.source, destination,
        setup_id="delta-local-test", claim_id=claim_id,
        service_factory=factory, config=config,
        expected_identity=identity,
        expected_compile_input_sha256=inputs[claim_id],
    )

    receipt = produced["receipt"]
    assert receipt["status"] == "passed"
    assert receipt["findings"] == []
    assert [call["role"] for call in receipt["calls"]] == [
        "obligations", "critic", "obligations", "critic"]
    assert sum(len(call["attempts"]) for call in receipt["calls"]) == 4
    assert len(local.calls) == 4
    assert produced["entry"]["payload"] == FULL
    assert produced["entry"]["payload"] != DELTA
    assert produced["entry"]["payload_sha256"] == content_hash(FULL)

    consumed = validate_prepared_contract_entry(
        destination,
        expected_setup_id="delta-local-test",
        expected_claim_id=claim_id,
        expected_identity=identity,
        expected_compile_input_sha256=inputs[claim_id],
        expected_receipt_sha256=produced["receipt_sha256"],
        candidate_entry=produced["entry"],
    )
    assert consumed["complete"] is True
    assert consumed["findings"] == []
    assert consumed["entry"] == produced["entry"]
    assert consumed["model_work"]["attempt_count"] == 4
    assert consumed["model_work"]["cost_complete"] is True
    assert consumed["model_work"]["attribution_complete"] is True
