"""Exact contract reuse: isolation, concurrency, failure and identity tests."""
from __future__ import annotations

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from verifier.core.events import EventLog
from verifier.core.model import CLAIM_FINAL, Claim
from verifier.core.store import ClaimStore, TRIGGER_LAZY
from verifier.engine.contract_cache import (
    CompiledFinalContract, CompiledFinalContractCache,
)
from verifier.engine.kernel import Kernel
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend, RoleError


PAYLOAD = {"obligations": [{
    "text": "The current candidate final response satisfies the task.",
    "required_basis": "world", "severity": "must",
    "clause": "Return the requested result.", "refutable_by_absence": False,
}]}
PASS = {"verdict": "pass", "counterexample_world": "n/a",
        "missing_condition": "n/a"}
ATTACK = {"verdict": "insufficient", "counterexample_world": "a missing case",
          "missing_condition": "ADD: every case is required"}


def fixture(**overrides):
    return FixtureBackend({"obligations": lambda v: copy.deepcopy(PAYLOAD),
                           "critic": copy.deepcopy(PASS), **overrides})


def kernel(tmp_path, backend, cache, name="candidate", task="Return the requested result.",
           final="CANDIDATE SECRET", **kwargs):
    state = tmp_path / name
    state.mkdir()
    if task is not None:
        (state / "task.md").write_text(task)
    log = EventLog(str(state / "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_LAZY)
    machine = Kernel(store, log, backend, ArtifactStore(str(state / "artifacts")),
                     contract_cache=cache, **kwargs)
    cid = store.add_claim(Claim(CLAIM_FINAL, final))
    return machine, store.get(cid), state


def events(machine, kind):
    return [e for e in machine.log.replay() if e["type"] == kind]


@pytest.mark.parametrize("live", [True, False])
def test_same_task_candidates_compile_once_and_materialize_fresh_claims(tmp_path, live):
    backend, cache = fixture(), CompiledFinalContractCache()
    a, ar, _ = kernel(tmp_path, backend, cache, name="a", live_world=live)
    b, br, _ = kernel(tmp_path, backend, cache, name="b", final="different answer",
                      live_world=live)
    first = a._compile_obligations(ar)
    second = b._compile_obligations(br)
    assert [call["role"] for call in backend.calls] == ["obligations", "critic"]
    assert first[0].claim_id == ar["id"] != br["id"] == second[0].claim_id
    assert first[0].id == second[0].id
    assert len(events(b, "contract_cache_hit")) == 1
    assert first[0].premise_scope == second[0].premise_scope == ()


def test_compiler_critic_and_repairs_receive_no_candidate_context(tmp_path):
    verdicts = iter([ATTACK, PASS])
    backend = fixture(critic=lambda v: next(verdicts))
    machine, row, state = kernel(tmp_path, backend, CompiledFinalContractCache())
    (state / "world").mkdir()
    (state / "world" / "answer.txt").write_text("WORLD SECRET")
    machine.log.append("message", {"text": "LEDGER SECRET"})
    machine.memory_hits = "MEMORY SECRET"
    assert machine._compile_obligations(row)
    assert [call["role"] for call in backend.calls] == [
        "obligations", "critic", "obligations", "critic"]
    for call in backend.calls:
        encoded = json.dumps(call["variables"])
        for forbidden in (row["text"], str(state), "WORLD SECRET", "LEDGER SECRET",
                          "MEMORY SECRET"):
            assert forbidden not in encoded
        assert call["variables"]["claim_text"] == backend.calls[0]["variables"]["claim_text"]


def test_task_change_and_role_change_never_hit(tmp_path):
    backend, cache = fixture(), CompiledFinalContractCache()
    a, ar, _ = kernel(tmp_path, backend, cache, name="a")
    a._compile_obligations(ar)
    b, br, _ = kernel(tmp_path, backend, cache, name="b", task="A different task")
    b._compile_obligations(br)
    c, cr, _ = kernel(tmp_path, backend, cache, name="c")
    c.obligations_role = replace(c.obligations_role, system="changed contract semantics")
    c._compile_obligations(cr)
    d, dr, _ = kernel(tmp_path, backend, cache, name="d", critic_rounds=2)
    d._compile_obligations(dr)
    assert len(backend.calls) == 8
    assert len(cache) == 4


@pytest.mark.parametrize("change", ["task", "baseline"])
def test_authority_mutation_during_compilation_is_never_published(tmp_path, change):
    backend, cache = fixture(), CompiledFinalContractCache()
    machine, row, state = kernel(tmp_path, backend, cache)
    def compile_(variables):
        if variables["world_mode"] == "authority-only":
            if change == "task":
                (state / "task.md").write_text("Revised task")
            else:
                (state / "baseline").mkdir(exist_ok=True)
        return copy.deepcopy(PAYLOAD)
    backend.fixtures["obligations"] = compile_
    assert machine._compile_obligations(row)
    assert len(cache) == 0
    assert events(machine, "contract_cache_bypass")


@pytest.mark.parametrize("mode", ["missing_task", "no_critic", "zero_rounds", "nonfinal"])
def test_ineligible_contract_uses_legacy_candidate_specific_compiler(tmp_path, mode):
    backend, cache = fixture(), CompiledFinalContractCache()
    kwargs = {"task": None} if mode == "missing_task" else {}
    if mode == "no_critic":
        kwargs["critic_role"] = None
    if mode == "zero_rounds":
        kwargs["critic_rounds"] = 0
    machine, row, _ = kernel(tmp_path, backend, cache, **kwargs)
    if mode == "nonfinal":
        row["kind"] = "assertion"
    machine._compile_obligations(row)
    assert backend.calls[0]["variables"]["claim_text"] == row["text"]
    assert len(cache) == 0


def test_rejected_contract_not_cached_and_next_candidate_can_recover(tmp_path):
    backend, cache = fixture(critic=ATTACK), CompiledFinalContractCache()
    a, ar, _ = kernel(tmp_path, backend, cache, name="a", critic_rounds=1)
    assert a._compile_obligations(ar) is None
    assert len(cache) == 0
    backend.fixtures["critic"] = PASS
    b, br, _ = kernel(tmp_path, backend, cache, name="b", critic_rounds=1)
    assert b._compile_obligations(br)
    assert len(cache) == 1


def test_empty_and_malformed_contracts_are_not_cached(tmp_path):
    for index, payload in enumerate(({"obligations": []}, {"obligations": "bad"})):
        backend, cache = fixture(obligations=payload), CompiledFinalContractCache()
        machine, row, _ = kernel(tmp_path, backend, cache, name=str(index))
        if index:
            with pytest.raises(RoleError):
                machine._compile_obligations(row)
        else:
            assert machine._compile_obligations(row) == []
        assert len(cache) == 0
        assert cache.acquire("unrelated").leader


def test_interrupted_fill_releases_same_key(tmp_path):
    backend, cache = fixture(), CompiledFinalContractCache()
    machine, row, _ = kernel(tmp_path, backend, cache)
    def interrupt(_):
        raise KeyboardInterrupt
    backend.fixtures["obligations"] = interrupt
    with pytest.raises(KeyboardInterrupt):
        machine._compile_obligations(row)
    assert not cache._flights
    backend.fixtures["obligations"] = PAYLOAD
    assert machine._compile_obligations(row)


def test_24_candidates_single_flight_and_exact_contract(tmp_path):
    started, release = threading.Event(), threading.Event()
    backend, cache = fixture(), CompiledFinalContractCache()
    def compile_(_):
        started.set()
        assert release.wait(5)
        return copy.deepcopy(PAYLOAD)
    backend.fixtures["obligations"] = compile_
    candidates = [kernel(tmp_path, backend, cache, name=str(i), final=f"answer {i}")
                  for i in range(24)]
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(k._compile_obligations, row) for k, row, _ in candidates]
        assert started.wait(5)
        release.set()
        results = [f.result(timeout=10) for f in futures]
    assert len(backend.calls) == 2
    assert len({result[0].id for result in results}) == 1
    assert len({result[0].claim_id for result in results}) == 24


def test_cache_copy_isolation_and_lru():
    cache = CompiledFinalContractCache(max_entries=2)
    source = copy.deepcopy(PAYLOAD)
    entry = CompiledFinalContract.from_payload(source)
    lease = cache.acquire("a")
    lease.finish(entry)
    source["obligations"][0]["text"] = "modified source"
    hit = cache.acquire("a")
    hit.entry.payload["obligations"][0]["text"] = "modified hit"
    assert cache.acquire("a").entry.payload == PAYLOAD
    cache.acquire("b").finish(entry)
    cache.acquire("a")
    cache.acquire("c").finish(entry)
    assert cache.acquire("b").leader
    assert len(cache) == 2


def test_failed_fill_wakes_waiters_without_caching_failure():
    cache = CompiledFinalContractCache()
    leader = cache.acquire("key")
    started = threading.Event()
    def follower():
        started.set()
        return cache.acquire("key")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(follower)
        assert started.wait(2)
        leader.finish()
        result = future.result(timeout=2)
    assert result.entry is None
    assert len(cache) == 0


def test_contract_cache_flag_is_strict_opt_in_and_part_of_score_identity(monkeypatch):
    from verifier.execution.config import contract_cache_enabled
    from verifier.offline import config_fields
    monkeypatch.delenv("GV_CONTRACT_CACHE", raising=False)
    assert contract_cache_enabled() is False
    before = config_fields(model="gpt-5.6-luna", seats="codex")
    monkeypatch.setenv("GV_CONTRACT_CACHE", "1")
    assert contract_cache_enabled() is True
    after = config_fields(model="gpt-5.6-luna", seats="codex")
    assert before["contract_cache"] is False
    assert after["contract_cache"] is True
    assert before["config"] != after["config"]
    monkeypatch.setenv("GV_CONTRACT_CACHE", "true")
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        contract_cache_enabled()


def test_runtime_calendar_rollover_during_fill_is_not_published(tmp_path, monkeypatch):
    import verifier.llm.authority_backend as authority_backend
    epoch = {"value": "day-one"}
    monkeypatch.setattr(authority_backend, "_calendar_epoch", lambda: dict(epoch))
    def compile_(variables):
        epoch["value"] = "day-two"
        return copy.deepcopy(PAYLOAD)
    backend, cache = fixture(obligations=compile_), CompiledFinalContractCache()
    machine, row, _ = kernel(tmp_path, backend, cache)
    assert machine._compile_obligations(row)
    assert len(cache) == 0
    assert events(machine, "contract_cache_bypass")
