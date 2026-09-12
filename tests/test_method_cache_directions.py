"""Both accepted-method outcome directions remain independently certified."""
from __future__ import annotations

import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

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
from verifier.engine.method_cache import AcceptedMethod, AcceptedMethodCache
from verifier.execution.adapters import (
    AdapterRoutingDischarger,
    default_adapter_registry,
)
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend, RoleError


JUDGE_YES = {
    "sufficient": "yes",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "the exact typed observation decides this direction",
}
JUDGE_NO = {
    "sufficient": "no",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "this direction has not been established by the method",
}


def _method(name, arguments, address="/app/result.json"):
    return {
        "procedure": f"execute registered adapter {name}",
        "artifact": "content-addressed typed evidence",
        "level": "deterministic",
        "basis": BASIS_WORLD,
        "strength": "decisive",
        "bridges": [],
        "adapter": name,
        "adapter_version": 1,
        "adapter_arguments": json.dumps(arguments, sort_keys=True),
        "adapter_inputs": [{
            "ref": address, "provenance": "delivered", "role": "subject",
        }],
    }


class _Fallback:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append((row["id"], obligation.id, method, tag))
        artifact = self.artifacts.put({
            "instrument": "direction-test-fallback", "claim": row["id"],
        })
        return DischargeResult(
            outcome=OUTCOME_FAILED,
            witness="typed path unexpectedly unavailable",
            artifact=artifact,
            instrument="direction-test-fallback",
        )


def _game(tmp_path, name, backend, cache, prepare):
    state = tmp_path / name
    world = state / "world" / "app"
    world.mkdir(parents=True)
    (state / "task.md").write_text(
        "Check the delivered result against the exact requirement.",
        encoding="utf-8",
    )
    (state / "world_manifest.json").write_text(json.dumps({
        "scheme": "filesystem", "captured_roots": ["/app"],
    }), encoding="utf-8")
    prepare(world)
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = _Fallback(artifacts)
    routing = AdapterRoutingDischarger(
        fallback,
        default_adapter_registry(),
        artifacts,
        log,
        str(state),
        live_world=False,
    )
    game = BindingGame(
        backend,
        artifacts,
        log,
        routing,
        live_world=False,
        method_cache=cache,
    )
    return game, artifacts, log, fallback


def _row(name):
    return {"id": name, "kind": "final", "text": f"candidate {name}"}


def _obligation(name, *, absence=False):
    return Obligation(
        name,
        "the delivered result meets the exact requirement",
        BASIS_WORLD,
        clause="Check the delivered result against the exact requirement.",
        refutable_by_absence=absence,
    )


def _events(log, kind):
    return [event for event in log.replay() if event["type"] == kind]


@pytest.mark.parametrize("adapter", [
    "filesystem.exists", "filesystem.sha256", "document.json-value",
])
def test_each_pure_adapter_learns_both_directions_once(tmp_path, adapter):
    good = b"accepted bytes"
    if adapter == "filesystem.exists":
        address = "/app/result.json"
        method = _method(adapter, {"address": address, "kind": "file"})

        def prepare_holds(world):
            (world / "result.json").write_bytes(good)

        def prepare_refuted(_world):
            pass

        absence = True
    elif adapter == "filesystem.sha256":
        address = "/app/result.json"
        method = _method(adapter, {
            "address": address,
            "expected_sha256": hashlib.sha256(good).hexdigest(),
        })

        def prepare_holds(world):
            (world / "result.json").write_bytes(good)

        def prepare_refuted(world):
            (world / "result.json").write_bytes(b"different bytes")

        absence = False
    else:
        address = "/app/result.json"
        method = _method(adapter, {
            "address": address, "pointer": "/status",
            "expected": "accepted",
        })

        def prepare_holds(world):
            (world / "result.json").write_text(
                '{"status":"accepted"}', encoding="utf-8")

        def prepare_refuted(world):
            (world / "result.json").write_text(
                '{"status":"rejected"}', encoding="utf-8")

        absence = False

    cache = AcceptedMethodCache()
    cases = (
        ("holds-source", prepare_holds, True),
        ("refuted-source", prepare_refuted, True),
        ("holds-hit", prepare_holds, False),
        ("refuted-hit", prepare_refuted, False),
    )
    calls = []
    results = []
    logs = []
    for name, prepare, cold_direction in cases:
        backend = FixtureBackend(
            {"oracles": {"methods": [copy.deepcopy(method)]},
             "judger": copy.deepcopy(JUDGE_YES)}
            if cold_direction else {})
        game, _artifacts, log, fallback = _game(
            tmp_path, f"{adapter}-{name}", backend, cache, prepare)
        result = game.run(_row(name), _obligation(name, absence=absence))
        assert fallback.calls == []
        calls.extend(backend.calls)
        results.append(result.outcome)
        logs.append(log)

    assert results == [
        OUTCOME_HOLDS, OUTCOME_REFUTED, OUTCOME_HOLDS, OUTCOME_REFUTED,
    ]
    assert [call["role"] for call in calls] == [
        "oracles", "judger", "judger",
    ]
    entries = cache.acquire(next(iter(cache._entries))).entries
    assert {entry.source_outcome for entry in entries} == {
        OUTCOME_HOLDS, OUTCOME_REFUTED,
    }
    assert _events(logs[2], "method_cache_hit")
    assert _events(logs[3], "method_cache_hit")
    direction_audits = _events(logs[1], "method_cache_direction_audit")
    assert len(direction_audits) == 1
    direction_audit = direction_audits[0]["payload"]
    assert direction_audit["schema"] == "gv-method-direction-audit-v1"
    assert direction_audit["judger_accepted"] is True
    assert direction_audit["authority_stable_after_judger"] is True
    assert direction_audit["direction_eligible"] is True
    assert direction_audit["source_outcome"] == OUTCOME_HOLDS
    assert direction_audit["fresh_outcome"] == OUTCOME_REFUTED
    assert len(direction_audit["method_digest"]) == 64
    assert len(direction_audit["source_binding_artifact"]) == 64
    assert len(direction_audit["source_discharge_artifact"]) == 64
    assert len(direction_audit["fresh_discharge_artifact"]) == 64
    for log in logs[2:]:
        attempts = [event for event in _events(log, "adapter_discharge")
                    if event["payload"].get(
                        "accepted_method_reuse_attempt") is True]
        assert len(attempts) == 1


def test_rejected_opposite_audit_does_not_create_that_direction(tmp_path):
    method = _method("document.json-value", {
        "address": "/app/result.json", "pointer": "/status",
        "expected": "accepted",
    })

    def prepare(value):
        def write(world):
            (world / "result.json").write_text(json.dumps({"status": value}))
        return write

    cache = AcceptedMethodCache()
    source_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES,
    })
    source, _, _, _ = _game(
        tmp_path, "source", source_backend, cache, prepare("accepted"))
    assert source.run(_row("source"), _obligation("source")).outcome \
        == OUTCOME_HOLDS

    rejected_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_NO,
    })
    rejected, _, rejected_log, _ = _game(
        tmp_path, "rejected", rejected_backend, cache, prepare("wrong"))
    assert rejected.run(_row("rejected"), _obligation("rejected")).outcome \
        == OUTCOME_FAILED
    assert [call["role"] for call in rejected_backend.calls] == [
        "judger", "oracles", "judger", "oracles",
    ]
    direction_audit = _events(
        rejected_log, "method_cache_direction_audit")[0]["payload"]
    assert direction_audit["judger_accepted"] is False
    assert direction_audit["authority_stable_after_judger"] is True
    assert direction_audit["direction_eligible"] is False
    assert direction_audit["fresh_outcome"] == OUTCOME_REFUTED
    assert direction_audit["fresh_discharge_artifact"]
    rejected_bypass = [
        event["payload"] for event in _events(
            rejected_log, "method_cache_bypass")
        if "direction_audit" in event["payload"]
    ]
    assert rejected_bypass[0]["direction_audit"]["reason"] == \
        JUDGE_NO["reason"]
    key = next(iter(cache._entries))
    assert [entry.source_outcome for entry in cache.acquire(key).entries] \
        == [OUTCOME_HOLDS]

    accepted_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES,
    })
    accepted, _, _, _ = _game(
        tmp_path, "accepted", accepted_backend, cache, prepare("wrong"))
    assert accepted.run(_row("accepted"), _obligation("accepted")).outcome \
        == OUTCOME_REFUTED
    assert [call["role"] for call in accepted_backend.calls] == ["judger"]

    hit_backend = FixtureBackend({})
    hit, _, hit_log, _ = _game(
        tmp_path, "hit", hit_backend, cache, prepare("wrong"))
    assert hit.run(_row("hit"), _obligation("hit")).outcome \
        == OUTCOME_REFUTED
    assert hit_backend.calls == []
    assert _events(hit_log, "method_cache_hit")


def test_direction_certificate_never_cross_binds_a_different_method(tmp_path):
    method_a = _method("document.json-value", {
        "address": "/app/result.json", "pointer": "/status",
        "expected": "accepted",
    })
    method_b = _method("document.json-value", {
        "address": "/app/result.json", "pointer": "/mirror",
        "expected": "accepted",
    })

    def prepare(status, mirror):
        def write(world):
            (world / "result.json").write_text(json.dumps({
                "status": status, "mirror": mirror,
            }))
        return write

    cache = AcceptedMethodCache()
    first_backend = FixtureBackend({
        "oracles": {"methods": [method_a]}, "judger": JUDGE_YES,
    })
    first, _, _, _ = _game(
        tmp_path, "method-a", first_backend, cache,
        prepare("accepted", "accepted"))
    assert first.run(_row("a"), _obligation("a")).outcome == OUTCOME_HOLDS

    verdicts = iter([copy.deepcopy(JUDGE_NO), copy.deepcopy(JUDGE_YES)])
    second_backend = FixtureBackend({
        "oracles": {"methods": [method_b]},
        "judger": lambda _variables: next(verdicts),
    })
    second, _, _, _ = _game(
        tmp_path, "method-b", second_backend, cache,
        prepare("wrong", "wrong"))
    assert second.run(_row("b"), _obligation("b")).outcome \
        == OUTCOME_REFUTED
    assert [call["role"] for call in second_backend.calls] == [
        "judger", "oracles", "judger",
    ]

    # Method A now observes refuted, but only method B owns the independently
    # approved refuted signature. Method B observes holds, whose signature is
    # owned by A. Neither certificate may authenticate the other's evidence.
    audit_backend = FixtureBackend({
        "oracles": {"methods": [method_a]}, "judger": JUDGE_YES,
    })
    audit, _, audit_log, _ = _game(
        tmp_path, "cross-binding", audit_backend, cache,
        prepare("wrong", "accepted"))
    assert audit.run(_row("cross"), _obligation("cross")).outcome \
        == OUTCOME_REFUTED
    assert [call["role"] for call in audit_backend.calls] == [
        "oracles", "judger",
    ]
    assert not _events(audit_log, "method_cache_hit")
    attempts = [event for event in _events(audit_log, "adapter_discharge")
                if event["payload"].get(
                    "accepted_method_reuse_attempt") is True]
    assert len(attempts) == 2
    entries = cache.acquire(next(iter(cache._entries))).entries
    by_direction = {entry.source_outcome: entry for entry in entries}
    assert by_direction[OUTCOME_HOLDS].method == method_a
    assert by_direction[OUTCOME_REFUTED].method == method_b


def test_missing_direction_single_flights_parallel_judger_audits(tmp_path):
    method = _method("document.json-value", {
        "address": "/app/result.json", "pointer": "/status",
        "expected": "accepted",
    })

    def prepare(value):
        def write(world):
            (world / "result.json").write_text(json.dumps({"status": value}))
        return write

    cache = AcceptedMethodCache()
    seed_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES,
    })
    seed, _, _, _ = _game(
        tmp_path, "seed", seed_backend, cache, prepare("accepted"))
    assert seed.run(_row("seed"), _obligation("seed")).outcome \
        == OUTCOME_HOLDS

    entered = threading.Event()
    release = threading.Event()

    def delayed_judger(_variables):
        entered.set()
        assert release.wait(10)
        return copy.deepcopy(JUDGE_YES)

    jobs = []
    for index in range(12):
        backend = FixtureBackend({
            "oracles": {"methods": [method]},
            "judger": delayed_judger,
        })
        game, _, log, fallback = _game(
            tmp_path, f"parallel-{index}", backend, cache,
            prepare("wrong"))
        jobs.append((backend, game, log, fallback))

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        leader = pool.submit(
            jobs[0][1].run, _row("p0"), _obligation("p0"))
        assert entered.wait(10)
        followers = [pool.submit(
            job[1].run, _row(f"p{index}"), _obligation(f"p{index}"))
            for index, job in enumerate(jobs[1:], 1)]
        release.set()
        results = [leader.result(timeout=10)] + [
            future.result(timeout=10) for future in followers]

    assert all(result.outcome == OUTCOME_REFUTED for result in results)
    assert sum(len(job[0].calls) for job in jobs) == 1
    assert all(job[3].calls == [] for job in jobs)
    assert sum(bool(_events(job[2], "method_cache_hit"))
               for job in jobs) == len(jobs) - 1
    for _, _, log, _ in jobs:
        attempts = [event for event in _events(log, "adapter_discharge")
                    if event["payload"].get(
                        "accepted_method_reuse_attempt") is True]
        assert len(attempts) == 1


def test_failed_direction_leader_wakes_follower_that_can_seed_it(tmp_path):
    method = _method("document.json-value", {
        "address": "/app/result.json", "pointer": "/status",
        "expected": "accepted",
    })

    def prepare(value):
        def write(world):
            (world / "result.json").write_text(json.dumps({"status": value}))
        return write

    cache = AcceptedMethodCache()
    seed_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES,
    })
    seed, _, _, _ = _game(
        tmp_path, "failure-seed", seed_backend, cache, prepare("accepted"))
    assert seed.run(_row("seed"), _obligation("seed")).outcome \
        == OUTCOME_HOLDS

    entered = threading.Event()
    release = threading.Event()

    def failing_oracle(_variables):
        entered.set()
        assert release.wait(10)
        raise RoleError("synthetic direction fill failure")

    failed_backend = FixtureBackend({"oracles": failing_oracle})
    failed, _, _, _ = _game(
        tmp_path, "failed-direction", failed_backend, cache,
        prepare("wrong"))
    good_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES,
    })
    good, _, _, _ = _game(
        tmp_path, "recovered-direction", good_backend, cache,
        prepare("wrong"))
    follower_observed_fresh_evidence = threading.Event()
    original_adapter_only = good.discharger.discharge_adapter_only

    def observe_before_direction_wait(*args, **kwargs):
        result = original_adapter_only(*args, **kwargs)
        follower_observed_fresh_evidence.set()
        return result

    good.discharger.discharge_adapter_only = observe_before_direction_wait

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(
            failed.run, _row("failed"), _obligation("failed"))
        assert entered.wait(10)
        follower = pool.submit(
            good.run, _row("good"), _obligation("good"))
        assert follower_observed_fresh_evidence.wait(10)
        release.set()
        assert leader.result(timeout=10).outcome == OUTCOME_FAILED
        assert follower.result(timeout=10).outcome == OUTCOME_REFUTED

    assert not cache._flights
    assert [call["role"] for call in good_backend.calls] == [
        "oracles", "judger",
    ]
    entries = cache.acquire(next(iter(cache._entries))).entries
    assert {entry.source_outcome for entry in entries} == {
        OUTCOME_HOLDS, OUTCOME_REFUTED,
    }


def test_authority_mutation_before_direction_acquire_never_seeds_old_key(
        tmp_path):
    method = _method("document.json-value", {
        "address": "/app/result.json", "pointer": "/status",
        "expected": "accepted",
    })

    def prepare(value):
        def write(world):
            (world / "result.json").write_text(json.dumps({"status": value}))
        return write

    cache = AcceptedMethodCache()
    seed_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES,
    })
    seed, _, _, _ = _game(
        tmp_path, "mutation-seed", seed_backend, cache, prepare("accepted"))
    assert seed.run(_row("seed"), _obligation("seed")).outcome \
        == OUTCOME_HOLDS

    backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES,
    })
    game, _, log, _ = _game(
        tmp_path, "mutation-opposite", backend, cache, prepare("wrong"))
    row, obligation = _row("opposite"), _obligation("opposite")
    old_key = game._method_cache_key(row, obligation)
    original = game.discharger.discharge_adapter_only

    def execute_then_mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (tmp_path / "mutation-opposite" / "task.md").write_text(
            "The authoritative requirement changed during observation.",
            encoding="utf-8",
        )
        return result

    game.discharger.discharge_adapter_only = execute_then_mutate
    assert game.run(row, obligation).outcome == OUTCOME_REFUTED

    assert game._method_cache_key(row, obligation) != old_key
    assert len(cache) == 1
    assert [entry.source_outcome
            for entry in cache.acquire(old_key).entries] == [OUTCOME_HOLDS]
    assert not cache._flights
    bypass = _events(log, "method_cache_bypass")
    assert "authority changed" in bypass[-1]["payload"]["reason"]


@pytest.mark.parametrize("mutation_stage", ["judger", "before-publish"])
def test_authority_mutation_during_direction_audit_never_seeds_old_key(
        tmp_path, mutation_stage):
    method = _method("document.json-value", {
        "address": "/app/result.json", "pointer": "/status",
        "expected": "accepted",
    })

    def prepare(value):
        def write(world):
            (world / "result.json").write_text(json.dumps({"status": value}))
        return write

    cache = AcceptedMethodCache()
    seed_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES,
    })
    seed, _, _, _ = _game(
        tmp_path, f"{mutation_stage}-seed", seed_backend, cache,
        prepare("accepted"))
    assert seed.run(_row("seed"), _obligation("seed")).outcome \
        == OUTCOME_HOLDS

    state = tmp_path / f"{mutation_stage}-opposite"
    judger_calls = 0

    def judge_and_maybe_mutate(_variables):
        nonlocal judger_calls
        judger_calls += 1
        if mutation_stage == "judger" and judger_calls == 1:
            (state / "task.md").write_text(
                "Authority changed while the fresh direction was audited.",
                encoding="utf-8",
            )
        return copy.deepcopy(JUDGE_YES)

    backend = FixtureBackend({
        "oracles": {"methods": [method]},
        "judger": judge_and_maybe_mutate,
    })
    game, _, log, _ = _game(
        tmp_path, f"{mutation_stage}-opposite", backend, cache,
        prepare("wrong"))
    row, obligation = _row("opposite"), _obligation("opposite")
    old_key = game._method_cache_key(row, obligation)
    if mutation_stage == "before-publish":
        original_accept = game._accepted_method
        accept_calls = 0

        def accept_then_mutate(*args, **kwargs):
            nonlocal accept_calls
            accepted = original_accept(*args, **kwargs)
            accept_calls += 1
            if accept_calls == 1:
                (state / "task.md").write_text(
                    "Authority changed immediately before publication.",
                    encoding="utf-8",
                )
            return accepted

        game._accepted_method = accept_then_mutate

    assert game.run(row, obligation).outcome == OUTCOME_REFUTED

    assert game._method_cache_key(row, obligation) != old_key
    assert [call["role"] for call in backend.calls] == [
        "judger", "oracles", "judger",
    ]
    assert [entry.source_outcome
            for entry in cache.acquire(old_key).entries] == [OUTCOME_HOLDS]
    assert not cache._flights
    direction_audit = _events(
        log, "method_cache_direction_audit")[0]["payload"]
    assert direction_audit["judger_accepted"] is True
    if mutation_stage == "judger":
        assert direction_audit["authority_stable_after_judger"] is False
        assert direction_audit["direction_eligible"] is False
    else:
        assert direction_audit["authority_stable_after_judger"] is True
        assert direction_audit["direction_eligible"] is True
    bypass = _events(log, "method_cache_bypass")
    assert "authority changed" in bypass[-1]["payload"]["reason"]


def test_direction_judger_infrastructure_failure_falls_back_and_seeds(
        tmp_path):
    method = _method("document.json-value", {
        "address": "/app/result.json", "pointer": "/status",
        "expected": "accepted",
    })

    def prepare(value):
        def write(world):
            (world / "result.json").write_text(json.dumps({"status": value}))
        return write

    cache = AcceptedMethodCache()
    seed_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES,
    })
    seed, _, _, _ = _game(
        tmp_path, "infra-seed", seed_backend, cache, prepare("accepted"))
    assert seed.run(_row("seed"), _obligation("seed")).outcome \
        == OUTCOME_HOLDS

    calls = 0

    def fail_fast_audit_then_recover(_variables):
        nonlocal calls
        calls += 1
        if calls <= 4:
            raise RoleError("synthetic direction Judger outage")
        return copy.deepcopy(JUDGE_YES)

    backend = FixtureBackend({
        "oracles": {"methods": [method]},
        "judger": fail_fast_audit_then_recover,
    })
    game, _, log, _ = _game(
        tmp_path, "infra-opposite", backend, cache, prepare("wrong"))

    assert game.run(_row("opposite"), _obligation("opposite")).outcome \
        == OUTCOME_REFUTED
    assert [call["role"] for call in backend.calls] == [
        "judger", "judger", "judger", "judger", "oracles", "judger",
    ]
    assert len(_events(log, "judger_call_failed")) == 2
    direction_audit = _events(
        log, "method_cache_direction_audit")[0]["payload"]
    assert direction_audit["judger_accepted"] is False
    assert direction_audit["faithful"] == "no"
    entries = cache.acquire(next(iter(cache._entries))).entries
    assert {entry.source_outcome for entry in entries} == {
        OUTCOME_HOLDS, OUTCOME_REFUTED,
    }


def test_keyboard_interrupt_releases_missing_direction_flight(tmp_path):
    method = _method("document.json-value", {
        "address": "/app/result.json", "pointer": "/status",
        "expected": "accepted",
    })

    def prepare(value):
        def write(world):
            (world / "result.json").write_text(json.dumps({"status": value}))
        return write

    cache = AcceptedMethodCache()
    seed_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES,
    })
    seed, _, _, _ = _game(
        tmp_path, "interrupt-seed", seed_backend, cache, prepare("accepted"))
    assert seed.run(_row("seed"), _obligation("seed")).outcome \
        == OUTCOME_HOLDS

    def interrupt(_variables):
        raise KeyboardInterrupt

    interrupted, _, _, _ = _game(
        tmp_path,
        "interrupt-direction",
        FixtureBackend({"judger": interrupt}),
        cache,
        prepare("wrong"),
    )
    with pytest.raises(KeyboardInterrupt):
        interrupted.run(_row("interrupt"), _obligation("interrupt"))
    assert not cache._flights
    key = next(iter(cache._entries))
    assert [entry.source_outcome for entry in cache.acquire(key).entries] \
        == [OUTCOME_HOLDS]

    recovery_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES,
    })
    recovery, _, _, _ = _game(
        tmp_path, "interrupt-recovery", recovery_backend, cache,
        prepare("wrong"))
    assert recovery.run(_row("recovery"), _obligation("recovery")).outcome \
        == OUTCOME_REFUTED
    assert not cache._flights


@pytest.mark.parametrize("interrupt_stage", ["audit-log", "accept-method"])
def test_keyboard_interrupt_after_direction_judger_releases_flight(
        tmp_path, interrupt_stage):
    method = _method("document.json-value", {
        "address": "/app/result.json", "pointer": "/status",
        "expected": "accepted",
    })

    def prepare(value):
        def write(world):
            (world / "result.json").write_text(json.dumps({"status": value}))
        return write

    cache = AcceptedMethodCache()
    seed_backend = FixtureBackend({
        "oracles": {"methods": [method]}, "judger": JUDGE_YES,
    })
    seed, _, _, _ = _game(
        tmp_path, f"lifecycle-{interrupt_stage}-seed", seed_backend, cache,
        prepare("accepted"))
    assert seed.run(_row("seed"), _obligation("seed")).outcome \
        == OUTCOME_HOLDS

    interrupted, _, log, _ = _game(
        tmp_path, f"lifecycle-{interrupt_stage}",
        FixtureBackend({"judger": JUDGE_YES}), cache, prepare("wrong"))
    if interrupt_stage == "audit-log":
        original_append = log.append

        def interrupt_audit_event(event_type, payload):
            if event_type == "method_cache_direction_audit":
                raise KeyboardInterrupt
            return original_append(event_type, payload)

        log.append = interrupt_audit_event
    else:
        def interrupt_accept(*_args, **_kwargs):
            raise KeyboardInterrupt

        interrupted._accepted_method = interrupt_accept

    with pytest.raises(KeyboardInterrupt):
        interrupted.run(_row("interrupt"), _obligation("interrupt"))
    assert not cache._flights

    recovery_backend = FixtureBackend({"judger": JUDGE_YES})
    recovery, _, _, _ = _game(
        tmp_path, f"lifecycle-{interrupt_stage}-recovery",
        recovery_backend, cache, prepare("wrong"))
    assert recovery.run(_row("recovery"), _obligation("recovery")).outcome \
        == OUTCOME_REFUTED
    assert [call["role"] for call in recovery_backend.calls] == ["judger"]
    assert not cache._flights


def test_direction_slots_are_detached_first_wins_and_count_as_one_lru_key():
    cache = AcceptedMethodCache(max_entries=2)

    def entry(outcome, number):
        return AcceptedMethod(
            method={"number": number},
            source_binding_artifact=f"binding-{number}",
            source_discharge_artifact=f"discharge-{number}",
            source_outcome=outcome,
            audit={"direction": outcome, "number": number},
        )

    seed = cache.acquire("same")
    seed.finish(entry(OUTCOME_HOLDS, 1))
    direction = cache.acquire("same", direction=OUTCOME_REFUTED)
    assert direction.leader
    direction.finish(entry(OUTCOME_REFUTED, 2))

    snapshot = cache.acquire("same")
    assert {item.source_outcome for item in snapshot.entries} == {
        OUTCOME_HOLDS, OUTCOME_REFUTED,
    }
    assert len(cache) == 1
    snapshot.entries[0].method["number"] = 99
    assert {item.method["number"] for item in cache.acquire("same").entries} \
        == {1, 2}

    winner, published = cache.publish_if_absent(
        "same", entry(OUTCOME_REFUTED, 3))
    assert published is False
    assert winner.method["number"] == 2
    assert {item.method["number"] for item in cache.acquire("same").entries} \
        == {1, 2}
