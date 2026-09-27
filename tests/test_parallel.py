"""Parallel execution is a schedule, not a judgment: the same obligations
get the same evidence whether they run one after another or at once.
These tests pin the invariants that make that true — serialized ledger and
store, one discharge per node, cancellation only of unstarted checks, the
lazy trigger, and the certificate's concurrent slice."""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_WORLD, Claim, CLAIM_ASSERTION, CLAIM_FINAL,
)
from verifier.core.store import ClaimStore, TRIGGER_LAZY, TRIGGER_ON_COMMIT
from verifier.engine.certificate import request_certificate
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend


class Discharger:
    """Scripted per-obligation results with optional delays; counts calls
    and the peak number of concurrent discharges."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def discharge(self, row, obligation, method=None, tag=""):
        outcome, delay = self.script.get(obligation.text, ("holds", 0.0))
        with self._lock:
            self.calls.append(obligation.text)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(delay)
        finally:
            with self._lock:
                self.active -= 1
        return DischargeResult(
            outcome=outcome, witness=("no" if outcome == "refuted" else None),
            artifact="", instrument="stub", basis=BASIS_WORLD,
            witness_kind="execution")


def build(tmp_path, compile_map, script, *, parallel_obligations=1,
          parallel_claims=1, trigger=TRIGGER_ON_COMMIT, live_world=True,
          memory=None, compact_inspection=False):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=trigger)
    backend = FixtureBackend({"obligations": lambda v: {
        "obligations": compile_map(v["claim_text"])}})
    backend.compact_inspection = compact_inspection
    discharger = Discharger(script)
    kernel = Kernel(store, log, backend,
                    ArtifactStore(os.path.join(tmp_path, "art")),
                    critic_role=None, discharger=discharger,
                    live_world=live_world, memory=memory,
                    parallel_obligations=parallel_obligations,
                    parallel_claims=parallel_claims)
    return store, kernel, log, discharger


# ---- infrastructure ------------------------------------------------------

def test_concurrent_ledger_appends_stay_monotonic(tmp_path):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: log.append("message", {"i": i}), range(400)))
    seqs = [e["seq"] for e in log.replay()]      # replay validates gaps
    assert seqs == list(range(1, 401))


def test_store_serves_many_threads(tmp_path):
    store = ClaimStore(os.path.join(tmp_path, "c.sqlite"),
                       trigger=TRIGGER_LAZY)
    ids = [store.add_claim(Claim(CLAIM_ASSERTION, f"s{i}")) for i in range(50)]
    def work(i):
        store.set_verified(ids[i], basis="world")
        return store.effective(ids[i])["effective_status"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert set(pool.map(work, range(50))) == {"verified"}


def test_lazy_trigger_never_schedules(tmp_path):
    fired = []
    store = ClaimStore(trigger=TRIGGER_LAZY)
    store.on_trigger(fired.append)
    a = store.add_claim(Claim(CLAIM_ASSERTION, "a"))
    store.add_claim(Claim(CLAIM_ASSERTION, "b", deps=(a,)))
    assert fired == []


# ---- obligations within a claim ---------------------------------------------

def obligations_of(n):
    return lambda text: [{"text": f"{text} #{i}", "required_basis": "world"}
                         for i in range(n)]


def test_parallel_obligations_run_concurrently_and_fold_identically(tmp_path):
    script = {f"c #{i}": ("holds", 0.15) for i in range(4)}
    s1, k1, _, d1 = build(tmp_path / "seq", obligations_of(4), script,
                          parallel_obligations=1)
    s2, k2, _, d2 = build(tmp_path / "par", obligations_of(4), script,
                          parallel_obligations=4)
    t0 = time.time(); c1 = s1.add_claim(Claim(CLAIM_FINAL, "c")); seq = time.time() - t0
    t0 = time.time(); c2 = s2.add_claim(Claim(CLAIM_FINAL, "c")); par = time.time() - t0
    assert s1.status(c1) == s2.status(c2) == "verified"
    assert sorted(d1.calls) == sorted(d2.calls)
    assert d2.peak >= 2 and d1.peak == 1
    assert par < seq


def test_must_refutation_cancels_unstarted_checks(tmp_path):
    # pool of 2: #0 refutes quickly, #1 is slow, #2/#3 have not started
    # when the refutation lands and are cancelled; #1 still finishes
    script = {"c #0": ("refuted", 0.0), "c #1": ("holds", 0.3),
              "c #2": ("holds", 0.0), "c #3": ("holds", 0.0)}
    store, kernel, log, d = build(tmp_path, obligations_of(4), script,
                                  parallel_obligations=2)
    cid = store.add_claim(Claim(CLAIM_FINAL, "c"))
    assert store.status(cid) == "refuted"
    assert "c #0" in d.calls and len(d.calls) <= 3
    skipped = [e for e in log.replay() if e["type"] == "status_change"
               and e["payload"].get("status") == "skipped"]
    assert len(skipped) == 4 - len(d.calls)


def test_should_refutation_does_not_cancel(tmp_path):
    def compile_map(text):
        return [{"text": "c #0", "required_basis": "world",
                 "severity": "should"},
                {"text": "c #1", "required_basis": "world"},
                {"text": "c #2", "required_basis": "world"}]
    script = {"c #0": ("refuted", 0.0), "c #1": ("holds", 0.05),
              "c #2": ("holds", 0.05)}
    store, kernel, _, d = build(tmp_path, compile_map, script,
                                parallel_obligations=2)
    cid = store.add_claim(Claim(CLAIM_FINAL, "c"))
    assert store.status(cid) == "partial"
    assert sorted(d.calls) == ["c #0", "c #1", "c #2"]


def test_shared_node_is_discharged_once_across_concurrent_claims(tmp_path):
    """Two claims compile the same obligation text; verified concurrently,
    the node lock makes the second wait and take the dedup path."""
    compile_map = lambda text: [{"text": "shared condition",   # noqa: E731
                                 "required_basis": "world"}]
    script = {"shared condition": ("holds", 0.2)}
    store, kernel, log, d = build(tmp_path, compile_map, script,
                                  parallel_obligations=2, parallel_claims=2,
                                  trigger=TRIGGER_LAZY)
    a = store.add_claim(Claim(CLAIM_ASSERTION, "a"))
    b = store.add_claim(Claim(CLAIM_ASSERTION, "b"))
    f = store.add_claim(Claim(CLAIM_FINAL, "f", deps=(a, b)))
    cert = request_certificate(store, kernel, log,
                               ArtifactStore(os.path.join(tmp_path, "art2")),
                               f)
    assert cert["verdict"] == "verified"
    assert {e["status"] for e in cert["slice"]} == {"verified"}
    assert d.calls == ["shared condition"]     # one discharge for three claims
    assert any(e["type"] == "dedup_hit" for e in log.replay())


def test_certificate_verifies_slice_concurrently(tmp_path):
    n = 6
    def compile_map(text):
        return [{"text": f"{text} holds", "required_basis": "world"}]
    script = {f"s{i} holds": ("holds", 0.2) for i in range(n)}
    script["f holds"] = ("holds", 0.0)
    store, kernel, log, d = build(tmp_path, compile_map, script,
                                  parallel_obligations=1, parallel_claims=n,
                                  trigger=TRIGGER_LAZY)
    ids = [store.add_claim(Claim(CLAIM_ASSERTION, f"s{i}")) for i in range(n)]
    f = store.add_claim(Claim(CLAIM_FINAL, "f", deps=tuple(ids)))
    t0 = time.time()
    cert = request_certificate(store, kernel, log,
                               ArtifactStore(os.path.join(tmp_path, "art2")),
                               f)
    elapsed = time.time() - t0
    assert cert["verdict"] == "verified"
    assert all(e["effective_status"] == "verified" for e in cert["slice"])
    assert d.peak >= 2
    assert elapsed < n * 0.2                      # not the sequential sum


def test_parallel_certificate_submits_independent_final_first(tmp_path):
    started = []
    lock = threading.Lock()
    final_started = threading.Event()

    def compile_map(text):
        if text == "final":
            with lock:
                started.append(text)
                final_started.set()
        else:
            assert final_started.wait(1)
            with lock:
                started.append(text)
        return [{"text": f"{text} holds", "required_basis": "world"}]

    store, kernel, log, _d = build(
        tmp_path, compile_map, {}, parallel_claims=2, trigger=TRIGGER_LAZY,
        live_world=False, compact_inspection=True)
    supports = [store.add_claim(Claim(CLAIM_ASSERTION, f"s{i}"))
                for i in range(4)]
    final = store.add_claim(Claim(CLAIM_FINAL, "final", deps=tuple(supports)))

    certificate = request_certificate(
        store, kernel, log,
        ArtifactStore(os.path.join(tmp_path, "certificate-artifacts")),
        final,
    )

    assert certificate["verdict"] == "verified"
    assert started[0] == "final"
    assert sorted(started) == ["final", "s0", "s1", "s2", "s3"]


def test_parallel_certificate_preserves_order_for_mutable_inputs(tmp_path):
    class Memory:
        def hits_for_obligations(self, _text):
            return "[]"

        def record_obligations(self, _text, _obligations, _status):
            pass

        def record_attack(self, _claim, _target, _witness, *, kind):
            pass

    def first_started(label, *, live_world, memory, compact_inspection):
        started = []
        lock = threading.Lock()
        support_started = threading.Event()
        case_path = tmp_path / label
        case_path.mkdir()

        def compile_map(text):
            if text == "support":
                support_started.set()
            else:
                assert support_started.wait(1)
            with lock:
                started.append(text)
            return [{"text": f"{text} holds", "required_basis": "world"}]

        store, kernel, log, _d = build(
            case_path, compile_map, {}, parallel_claims=1,
            trigger=TRIGGER_LAZY, live_world=live_world, memory=memory,
            compact_inspection=compact_inspection)
        support = store.add_claim(Claim(CLAIM_ASSERTION, "support"))
        final = store.add_claim(Claim(CLAIM_FINAL, "final", deps=(support,)))
        kernel.parallel_claims = 2
        request_certificate(
            store, kernel, log,
            ArtifactStore(os.path.join(case_path, "certificate")),
            final,
        )
        return started[0]

    assert first_started(
        "live", live_world=True, memory=None,
        compact_inspection=True) == "support"
    assert first_started(
        "memory", live_world=False, memory=Memory(),
        compact_inspection=True) == "support"
    assert first_started(
        "expanded", live_world=False, memory=None,
        compact_inspection=False) == "support"


def test_verify_is_serialized_per_claim(tmp_path):
    compile_map = lambda text: [{"text": "x", "required_basis": "world"}]  # noqa: E731
    store, kernel, _, d = build(tmp_path, compile_map, {"x": ("holds", 0.2)},
                                trigger=TRIGGER_LAZY)
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "a"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        statuses = list(pool.map(kernel.verify, [cid] * 4))
    assert statuses == ["verified"] * 4
    assert d.calls == ["x"]


# ---- shared toolchain and caches through the codex sandbox -------------------

def test_codex_policy_grants_shared_roots_and_points_caches_at_them(tmp_path):
    from verifier.llm.codex_backend import CodexRuntime
    state = tmp_path / "state"
    work = state / "work"
    work.mkdir(parents=True)
    shared = tmp_path / "_shared" / "task-x"
    toolchain = shared / "toolchain"
    cache = shared / "cache"
    toolchain.mkdir(parents=True)
    cache.mkdir(parents=True)
    backend = CodexRuntime(
        cwd=str(state), model="m",
        offline_toolchain_dir=str(toolchain),
        offline_shared_cache_dir=str(cache),
    )
    args = backend._execution_policy_args("discharge", {
        "world_mode": "captured",
        "workdir": str(work / "abc-bound"),
        "toolchain_dir": str(toolchain),
        "shared_cache_dir": str(cache),
    })
    joined = "\n".join(args)
    assert f'"{work}"="write"' in joined.replace("\\/", "/") \
        or f'{work}' in joined
    assert str(toolchain) in joined and str(cache) in joined
    assert "UV_CACHE_DIR" in joined and str(cache / "uv") in joined
    assert "PIP_CACHE_DIR" in joined
    # the state-local HOME is unchanged: only package caches are shared
    assert str(work / "_offline_home") in joined


def test_codex_policy_without_shared_dirs_is_unchanged(tmp_path):
    from verifier.llm.codex_backend import CodexRuntime
    state = tmp_path / "state"
    (state / "work").mkdir(parents=True)
    backend = CodexRuntime(cwd=str(state), model="m")
    args = backend._execution_policy_args("discharge", {
        "world_mode": "captured", "workdir": str(state / "work" / "w")})
    joined = "\n".join(args)
    assert "UV_CACHE_DIR" not in joined
    assert joined.count('="write"') == 1
