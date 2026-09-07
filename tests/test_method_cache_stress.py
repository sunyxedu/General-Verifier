"""Integration, policy, and concurrency tests for accepted MethodIR reuse."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_WORLD,
    OUTCOME_FAILED,
    OUTCOME_HOLDS,
    OUTCOME_REFUTED,
    Obligation,
)
from verifier.engine.game import BindingGame
from verifier.engine.method_cache import (
    AcceptedMethod,
    AcceptedMethodCache,
    default_accepted_method_cache,
)
from verifier.execution.adapters import (
    AdapterRoutingDischarger,
    default_adapter_registry,
)
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.agents import JUDGER_AGENT
from verifier.llm.schema import FixtureBackend, RoleError
from verifier.mcp_server import Service, _default_kernel


JUDGE_YES = {
    "sufficient": "yes",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "the typed method decides the exact obligation",
}
CRITIC_PASS = {
    "verdict": "pass",
    "counterexample_world": "n/a",
    "missing_condition": "n/a",
}


def typed_method(address="/app/result.json", *, provenance="delivered",
                 strength="decisive", bridges=None):
    return {
        "procedure": "read the exact status with the registered adapter",
        "artifact": "content-addressed typed evidence",
        "level": "deterministic",
        "basis": BASIS_WORLD,
        "strength": strength,
        "bridges": [] if bridges is None else bridges,
        "adapter": "document.json-value",
        "adapter_version": 1,
        "adapter_arguments": json.dumps({
            "address": address,
            "pointer": "/status",
            "expected": "completed",
        }, sort_keys=True),
        "adapter_inputs": [{
            "ref": address,
            "provenance": provenance,
            "role": "subject",
        }],
    }


class Fallback:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append((row["id"], obligation.id, method, tag))
        digest = self.artifacts.put({
            "instrument": "stress-fallback", "claim": row["id"],
            "obligation": obligation.id, "tag": tag,
        })
        return DischargeResult(
            outcome=OUTCOME_HOLDS,
            witness=None,
            artifact=digest,
            instrument="stress-fallback",
            witness_kind="execution",
        )


def candidate(tmp_path, name, status="completed", *, task="Make status completed",
              roots=("/app",), baseline=None, live=False):
    state = tmp_path / name
    state.mkdir()
    (state / "task.md").write_text(task)
    if not live:
        world = state / "world" / "app"
        world.mkdir(parents=True)
        (world / "result.json").write_text(json.dumps({"status": status}))
        (state / "world_manifest.json").write_text(json.dumps({
            "scheme": "filesystem", "captured_roots": list(roots)}))
    if baseline is not None:
        base = state / "baseline"
        base.mkdir()
        (base / "git.json").write_text(json.dumps({"head": "a" * 40}))
        (base / "manifest.json").write_text(json.dumps({
            "root": f"/candidate/{name}", "files": baseline}))
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = Fallback(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=live)
    return state, artifacts, log, fallback, routing


def game_for(tmp_path, name, backend, cache, *, status="completed",
             task="Make status completed", roots=("/app",), baseline=None,
             live=False, judger_role=JUDGER_AGENT):
    state, artifacts, log, fallback, routing = candidate(
        tmp_path, name, status, task=task, roots=roots, baseline=baseline,
        live=live)
    game = BindingGame(
        backend, artifacts, log, routing, live_world=live,
        method_cache=cache, judger_role=judger_role)
    return state, game, artifacts, log, fallback


def test_default_offline_services_reaudit_when_fresh_outcome_changes(tmp_path):
    cache = default_accepted_method_cache()
    cache.clear()
    method = typed_method()
    compile_result = {"obligations": [{
        "text": "status equals completed",
        "required_basis": "world",
        "severity": "must",
        "clause": "Make status completed",
        "refutable_by_absence": False,
    }]}

    def build(name, status, fixtures):
        state, _, _, _, _ = candidate(tmp_path, name, status)
        backend = FixtureBackend({
            "obligations": compile_result, "critic": CRITIC_PASS, **fixtures})
        service = Service(
            str(state), live_world=False,
            kernel_factory=lambda sc: _default_kernel(
                sc, model=None, backend=backend, critic_rounds=1,
                parallel_obligations=1, parallel_claims=1))
        return service, backend

    first, backend1 = build("service-1", "completed", {
        "oracles": {"methods": [method]}, "judger": JUDGE_YES})
    second, backend2 = build("service-2", "wrong", {
        "oracles": {"methods": [method]}, "judger": JUDGE_YES})

    assert first.commit("F1", "final", "candidate one is done", [])["status"] \
        == "verified"
    assert second.commit("F2", "final", "candidate two finished", [])["status"] \
        == "refuted"
    assert [call["role"] for call in backend1.calls] == [
        "obligations", "critic", "oracles", "judger"]
    assert [call["role"] for call in backend2.calls] == [
        "obligations", "critic", "oracles", "judger"]
    events = list(second.log.replay())
    assert any(event["type"] == "method_cache_bypass" for event in events)
    assert not any(event["type"] == "method_cache_hit" for event in events)


def test_default_online_service_uses_adapter_without_executor_seat(tmp_path):
    cache = default_accepted_method_cache()
    cache.clear()
    delivery = tmp_path / "live-delivery" / "result.json"
    delivery.parent.mkdir()
    delivery.write_text('{"status":"completed"}')
    state = tmp_path / "online-state"
    state.mkdir()
    (state / "task.md").write_text("Keep the live result completed")
    backend = FixtureBackend({
        "obligations": {"obligations": [{
            "text": "live status equals completed",
            "required_basis": "world",
            "severity": "must",
            "clause": "Keep the live result completed",
            "refutable_by_absence": False,
        }]},
        "oracles": {"methods": [typed_method(str(delivery))]},
        "judger": JUDGE_YES,
        "critic": CRITIC_PASS,
    })
    service = Service(
        str(state), live_world=True,
        kernel_factory=lambda sc: _default_kernel(
            sc, model=None, backend=backend, critic_rounds=1,
            parallel_obligations=1, parallel_claims=1))

    result = service.commit("F", "final", "live delivery is complete", [])

    assert result["status"] == "verified"
    assert [call["role"] for call in backend.calls] == [
        "obligations", "critic", "oracles", "judger"]
    assert any(event["type"] == "adapter_discharge"
               for event in service.log.replay())


@pytest.mark.parametrize("policy", [
    "non-final", "premise-scope", "bounded", "bridged", "task-given",
])
def test_reuse_policy_excludes_methods_without_full_safe_signature(
        tmp_path, policy):
    cache = AcceptedMethodCache()
    row = {"id": "c1", "kind": "final", "text": "done"}
    obligation = Obligation(
        "c1", "status equals completed", BASIS_WORLD,
        premise_scope=("premise",) if policy == "premise-scope" else ())
    method = typed_method(
        strength="bounded" if policy == "bounded" else "decisive",
        bridges=["extra condition"] if policy == "bridged" else None,
        provenance="task_given" if policy == "task-given" else "delivered",
    )
    verdict = dict(JUDGE_YES)
    if policy == "bounded":
        verdict["degradation"] = "justified"
    if policy == "non-final":
        row["kind"] = "assertion"
    backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": verdict})
    _, game, _, _, _ = game_for(
        tmp_path, f"policy-{policy}", backend, cache)

    assert game.run(row, obligation).outcome == OUTCOME_HOLDS
    assert len(cache) == 0


@pytest.mark.parametrize("difference", [
    "task", "baseline", "world-shape", "backend", "judger-prompt",
])
def test_cache_identity_separates_semantically_different_runs(
        tmp_path, difference):
    cache = AcceptedMethodCache()
    method = typed_method()
    backend1 = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES})
    backend1.model = "model-a"
    _, game1, _, _, _ = game_for(
        tmp_path, "identity-1", backend1, cache,
        baseline={"given.txt": "hash-a"})
    obligation = Obligation(
        "c1", "status equals completed", BASIS_WORLD,
        clause="Make status completed")
    assert game1.run(
        {"id": "c1", "kind": "final", "text": "done"},
        obligation).outcome == OUTCOME_HOLDS
    assert len(cache) == 1

    task = "A different task" if difference == "task" else "Make status completed"
    baseline = ({"given.txt": "hash-b"} if difference == "baseline"
                else {"given.txt": "hash-a"})
    roots = ("/app", "/data") if difference == "world-shape" else ("/app",)
    backend2 = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES})
    backend2.model = "model-b" if difference == "backend" else "model-a"
    judger = (replace(JUDGER_AGENT, system=JUDGER_AGENT.system + "\nchanged")
              if difference == "judger-prompt" else JUDGER_AGENT)
    _, game2, _, log2, _ = game_for(
        tmp_path, "identity-2", backend2, cache, task=task, roots=roots,
        baseline=baseline, judger_role=judger)

    assert game2.run(
        {"id": "c2", "kind": "final", "text": "finished"},
        obligation).outcome == OUTCOME_HOLDS
    assert [call["role"] for call in backend2.calls] == [
        "oracles", "judger"]
    assert not any(event["type"] == "method_cache_hit"
                   for event in log2.replay())


def test_single_flight_stress_reexecutes_24_independent_candidates(tmp_path):
    cache = AcceptedMethodCache()
    method = typed_method()
    entered = threading.Event()
    release = threading.Event()

    def delayed_judger(_variables):
        entered.set()
        assert release.wait(10)
        return JUDGE_YES

    jobs = []
    for index in range(24):
        # Single-flight reuse remains valuable for candidates whose freshly
        # executed adapter result has the same direction as the accepted
        # source result.
        status = "completed"
        fixtures = ({"oracles": {"methods": [method]},
                     "judger": delayed_judger} if index == 0 else {})
        backend = FixtureBackend(fixtures)
        _, game, artifacts, log, fallback = game_for(
            tmp_path, f"stress-{index}", backend, cache, status=status)
        row = {"id": f"c{index}", "kind": "final",
               "text": f"candidate {index} completed"}
        obligation = Obligation(
            row["id"], "status equals completed", BASIS_WORLD)
        jobs.append((status, backend, game, row, obligation, artifacts,
                     log, fallback))

    with ThreadPoolExecutor(max_workers=24) as pool:
        first = pool.submit(jobs[0][2].run, jobs[0][3], jobs[0][4])
        assert entered.wait(10)
        futures = [first] + [
            pool.submit(job[2].run, job[3], job[4]) for job in jobs[1:]
        ]
        release.set()
        results = [future.result(timeout=10) for future in futures]

    for index, (job, result) in enumerate(zip(jobs, results)):
        assert result.outcome == OUTCOME_HOLDS
        assert job[7].calls == []
        bound = job[5].get(result.artifact)["bound"]
        assert job[5].get(bound["discharge_artifact"])["claim"] == job[3]["id"]
        if index:
            assert bound["audit"]["cache"]["source_outcome"] \
                == OUTCOME_HOLDS
    assert sum(len(job[1].calls) for job in jobs) == 2
    assert sum(any(event["type"] == "method_cache_hit"
                   for event in job[6].replay()) for job in jobs) == 23


@pytest.mark.parametrize(
    "source_status,source_outcome,fresh_status,fresh_outcome",
    [
        ("completed", OUTCOME_HOLDS, "wrong", OUTCOME_REFUTED),
        ("wrong", OUTCOME_REFUTED, "completed", OUTCOME_HOLDS),
    ],
)
def test_outcome_direction_change_bypasses_cached_judgment(
        tmp_path, source_status, source_outcome, fresh_status, fresh_outcome):
    cache = AcceptedMethodCache()
    method = typed_method()
    obligation = Obligation(
        "source", "status equals completed", BASIS_WORLD)
    source_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES})
    _, source_game, _, _, _ = game_for(
        tmp_path, "direction-source", source_backend, cache,
        status=source_status)

    assert source_game.run(
        {"id": "source", "kind": "final", "text": "source candidate"},
        obligation).outcome == source_outcome
    assert len(cache) == 1

    fresh_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES})
    _, fresh_game, _, fresh_log, fresh_fallback = game_for(
        tmp_path, "direction-fresh", fresh_backend, cache,
        status=fresh_status)
    fresh = fresh_game.run(
        {"id": "fresh", "kind": "final", "text": "fresh candidate"},
        Obligation("fresh", "status equals completed", BASIS_WORLD))

    assert fresh.outcome == fresh_outcome
    assert [call["role"] for call in fresh_backend.calls] == [
        "oracles", "judger"]
    assert fresh_fallback.calls == []
    events = list(fresh_log.replay())
    reuse_attempts = [
        event for event in events
        if event["type"] == "adapter_discharge"
        and event["payload"].get("accepted_method_reuse_attempt") is True
    ]
    assert len(reuse_attempts) == 1
    assert any(event["type"] == "method_cache_bypass" for event in events)
    assert not any(event["type"] == "method_cache_hit" for event in events)


def test_opposite_fresh_outcome_is_controlled_by_a_fresh_judger(tmp_path):
    cache = AcceptedMethodCache()
    method = typed_method()
    source_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES})
    _, source_game, _, _, _ = game_for(
        tmp_path, "conservative-source", source_backend, cache,
        status="wrong")
    obligation = Obligation(
        "source", "status equals completed", BASIS_WORLD)

    assert source_game.run(
        {"id": "source", "kind": "final", "text": "source candidate"},
        obligation).outcome == OUTCOME_REFUTED

    fresh_backend = FixtureBackend({
        # The duplicate repair response is rejected mechanically after the
        # fresh Judger rejects the method, leaving the game failed closed.
        "oracles": {"methods": [method]},
        "judger": {
            "sufficient": "no",
            "faithful": "yes",
            "degradation": "n/a",
            "reason": "this certificate does not cover the holds direction",
        },
    })
    _, fresh_game, _, fresh_log, _ = game_for(
        tmp_path, "conservative-fresh", fresh_backend, cache,
        status="completed")
    fresh = fresh_game.run(
        {"id": "fresh", "kind": "final", "text": "fresh candidate"},
        Obligation("fresh", "status equals completed", BASIS_WORLD))

    assert fresh.outcome == OUTCOME_FAILED
    assert [call["role"] for call in fresh_backend.calls] == [
        "oracles", "judger", "oracles"]
    events = list(fresh_log.replay())
    assert any(event["type"] == "method_cache_bypass" for event in events)
    assert not any(event["type"] == "method_cache_hit" for event in events)


def test_negative_direction_reuses_only_while_fresh_outcome_stays_refuted(
        tmp_path):
    cache = AcceptedMethodCache()
    address = "/app/report.txt"
    method = {
        "procedure": "check whether the exact report path is a file",
        "artifact": "content-addressed typed evidence",
        "level": "deterministic",
        "basis": BASIS_WORLD,
        "strength": "decisive",
        "bridges": [],
        "adapter": "filesystem.exists",
        "adapter_version": 1,
        "adapter_arguments": json.dumps({
            "address": address, "kind": "file",
        }, sort_keys=True),
        "adapter_inputs": [{
            "ref": address, "provenance": "delivered", "role": "subject",
        }],
    }
    task = "Create report.txt containing approved"

    def compound_obligation(claim):
        return Obligation(
            claim, "report.txt exists and its exact content is approved",
            BASIS_WORLD, clause=task, refutable_by_absence=True)

    source_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES})
    _, source_game, _, _, _ = game_for(
        tmp_path, "negative-source", source_backend, cache, task=task)
    source = source_game.run(
        {"id": "source", "kind": "final", "text": "source candidate"},
        compound_obligation("source"))

    assert source.outcome == OUTCOME_REFUTED
    assert len(cache) == 1

    same_backend = FixtureBackend({})
    _, same_game, same_artifacts, same_log, same_fallback = game_for(
        tmp_path, "negative-same", same_backend, cache, task=task)
    same = same_game.run(
        {"id": "same", "kind": "final", "text": "same candidate"},
        compound_obligation("same"))

    assert same.outcome == OUTCOME_REFUTED
    assert same_backend.calls == []
    assert same_fallback.calls == []
    same_bound = same_artifacts.get(same.artifact)["bound"]
    assert same_bound["audit"]["cache"]["source_outcome"] \
        == OUTCOME_REFUTED
    assert any(event["type"] == "method_cache_hit"
               for event in same_log.replay())

    fresh_backend = FixtureBackend({
        "oracles": {"methods": [method]},
        "judger": {
            "sufficient": "no",
            "faithful": "yes",
            "degradation": "n/a",
            "reason": "existence alone does not establish exact content",
        },
    })
    fresh_state, fresh_game, _, fresh_log, fresh_fallback = game_for(
        tmp_path, "negative-opposite", fresh_backend, cache, task=task)
    (fresh_state / "world" / "app" / "report.txt").write_text("wrong")
    fresh = fresh_game.run(
        {"id": "fresh", "kind": "final", "text": "fresh candidate"},
        compound_obligation("fresh"))

    assert fresh.outcome == OUTCOME_FAILED
    assert [call["role"] for call in fresh_backend.calls] == [
        "oracles", "judger", "oracles"]
    assert fresh_fallback.calls == []
    events = list(fresh_log.replay())
    assert sum(
        event["type"] == "adapter_discharge"
        and event["payload"].get("accepted_method_reuse_attempt") is True
        for event in events
    ) == 1
    assert any(event["type"] == "method_cache_bypass" for event in events)
    assert not any(event["type"] == "method_cache_hit" for event in events)


def test_authority_change_during_hit_forces_full_game(tmp_path):
    cache = AcceptedMethodCache()
    method = typed_method()
    backend1 = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES})
    _, game1, _, _, _ = game_for(tmp_path, "race-1", backend1, cache)
    obligation = Obligation("c1", "status equals completed", BASIS_WORLD)
    row = {"id": "c1", "kind": "final", "text": "done"}
    assert game1.run(row, obligation).outcome == OUTCOME_HOLDS

    legacy = {
        "procedure": "run the full existing executor",
        "artifact": "ordinary executor evidence",
        "level": "ordinary", "basis": BASIS_WORLD,
        "strength": "decisive", "bridges": [],
    }
    backend2 = FixtureBackend({
        "oracles": {"methods": [legacy]}, "judger": JUDGE_YES})
    state2, game2, _, log2, fallback2 = game_for(
        tmp_path, "race-2", backend2, cache)
    original = game2.discharger.discharge_adapter_only

    def execute_then_change_authority(*args, **kwargs):
        result = original(*args, **kwargs)
        (state2 / "task.md").write_text("Authority changed during execution")
        return result

    game2.discharger.discharge_adapter_only = execute_then_change_authority
    result = game2.run(
        {"id": "c2", "kind": "final", "text": "finished"}, obligation)

    assert result.outcome == OUTCOME_HOLDS
    assert [call["role"] for call in backend2.calls] == [
        "oracles", "judger"]
    assert len(fallback2.calls) == 1
    bypass = [event for event in log2.replay()
              if event["type"] == "method_cache_bypass"]
    assert "authority changed" in bypass[-1]["payload"]["reason"]


def test_cache_lru_and_detached_entries():
    cache = AcceptedMethodCache(max_entries=2)

    def entry(number):
        return AcceptedMethod(
            method={"number": number},
            source_binding_artifact=str(number),
            source_discharge_artifact=str(number),
            source_outcome=OUTCOME_HOLDS,
            audit={"sufficient": "yes"},
        )

    for key, number in (("a", 1), ("b", 2)):
        lease = cache.acquire(key)
        assert lease.leader
        lease.finish(entry(number))
    hit = cache.acquire("a")
    hit.entry.method["number"] = 99
    lease = cache.acquire("c")
    lease.finish(entry(3))

    assert cache.acquire("a").entry.method["number"] == 1
    assert cache.acquire("a").entry.source_outcome == OUTCOME_HOLDS
    evicted = cache.acquire("b")
    assert evicted.leader is True
    evicted.finish()
    assert len(cache) == 2


def test_failed_single_flight_unblocks_and_successful_follower_reseeds(
        tmp_path):
    cache = AcceptedMethodCache()
    method = typed_method()
    entered = threading.Event()
    release = threading.Event()

    def failing_oracle(_variables):
        entered.set()
        assert release.wait(10)
        raise RoleError("simulated provider failure")

    failed_backend = FixtureBackend({"oracles": failing_oracle})
    _, failed_game, _, _, _ = game_for(
        tmp_path, "failed-leader", failed_backend, cache)
    good_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES})
    _, good_game, _, _, _ = game_for(
        tmp_path, "good-follower", good_backend, cache)
    obligation = Obligation("c", "status equals completed", BASIS_WORLD)

    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(
            failed_game.run,
            {"id": "bad", "kind": "final", "text": "bad"}, obligation)
        assert entered.wait(10)
        good = pool.submit(
            good_game.run,
            {"id": "good", "kind": "final", "text": "good"}, obligation)
        release.set()
        assert failed.result(timeout=10).outcome == "failed"
        assert good.result(timeout=10).outcome == OUTCOME_HOLDS

    assert len(cache) == 1
    empty_backend = FixtureBackend({})
    _, third_game, _, _, _ = game_for(
        tmp_path, "later-candidate", empty_backend, cache)
    assert third_game.run(
        {"id": "later", "kind": "final", "text": "later"},
        obligation).outcome == OUTCOME_HOLDS
    assert empty_backend.calls == []
