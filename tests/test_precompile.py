"""The final contract compiled beside extraction is adopted only exactly."""
from __future__ import annotations

import copy
import json
import os

import pytest

from verifier.core.store import TRIGGER_LAZY
from verifier.engine.certificate import CERTIFICATE_SCOPE_FINAL, request_certificate
from verifier.engine.kernel import compile_roles_for_run
from verifier.engine.contract.precompile import PrecompiledFinalContract
from verifier.config import final_contract_precompile_enabled
from verifier.llm.schema import FixtureBackend, RoleError
from verifier.app.assembly import build_kernel, harness_roles
from verifier.mcp_server import Service
from test_certificate_scope import _fixtures


def _state(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text("Issue one command.\n", encoding="utf-8")
    with open(state / "hook_feed.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "tool_call", "payload": {
            "tool": "bash", "input": {"command": "true"},
            "tool_use_id": "call-1"}}) + "\n")
        handle.write(json.dumps({"type": "tool_result", "payload": {
            "tool": "bash", "tool_use_id": "call-1", "excerpt": ""}}) + "\n")
    return state


def _start(state, backend, *, critic_rounds=3, contract_cache=None):
    roles = harness_roles(str(state))
    compile_role, critic_role = compile_roles_for_run(
        backend.run_options, roles["obligations"], roles["critic"])
    return PrecompiledFinalContract.start(
        str(state), backend, compile_role, critic_role, critic_rounds,
        False, None, contract_cache=contract_cache)


def _service(state, backend, monkeypatch, handle=None, *, critic_rounds=3):
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "0")
    service = Service(
        str(state), live_world=False, trigger=TRIGGER_LAZY,
        critic_rounds=critic_rounds,
        kernel_factory=lambda sc: build_kernel(
            sc, None, backend=backend, critic_rounds=critic_rounds),
    )
    if handle is not None:
        service.kernel.adopt_precompiled_final_contract(handle)
    service.drain_hooks()
    service.commit("f", "final", "the task is complete", ["effect:2"])
    return service


def _events(service, kind):
    return [e["payload"] for e in service.log.replay() if e["type"] == kind]


def test_flag_defaults_off_and_is_strict(monkeypatch):
    monkeypatch.delenv("GV_FINAL_CONTRACT_PRECOMPILE", raising=False)
    assert final_contract_precompile_enabled() is False
    assert final_contract_precompile_enabled({"GV_FINAL_CONTRACT_PRECOMPILE": "1"}) is True
    with pytest.raises(ValueError):
        final_contract_precompile_enabled({"GV_FINAL_CONTRACT_PRECOMPILE": "x"})


def test_precompiled_contract_is_adopted_under_the_same_key(tmp_path, monkeypatch):
    state = _state(tmp_path)
    backend = FixtureBackend(_fixtures())
    handle = _start(state, backend)
    outcome = handle.wait()
    assert outcome.error is None and outcome.bypass_reason is None
    assert outcome.payload == _fixtures()["obligations"]
    assert [kind for kind, _payload in outcome.events] == ["decomposition_attacked"]
    assert outcome.events[0][1]["claim"] == handle.pseudo_claim_id
    compile_calls = [c for c in backend.calls if c["role"] == "obligations"]
    assert len(compile_calls) == 1
    assert compile_calls[0]["variables"]["world_mode"] == "authority-only"

    service = _service(state, backend, monkeypatch, handle)
    certificate = request_certificate(
        service.store, service.kernel, service.log, service.artifacts,
        service.builder.finals[-1], scope=CERTIFICATE_SCOPE_FINAL)
    assert certificate["verdict"] == "verified"
    final_id = service.builder.finals[-1]
    adopted = _events(service, "contract_precompiled")
    assert adopted == [{**adopted[0], "adopted": True, "claim": final_id,
                        "precompiled_claim": handle.pseudo_claim_id,
                        "critic_rounds_used": 1}]
    # the buffered critic round was replayed under the real claim id and
    # the compiler was not called a second time
    attacks = _events(service, "decomposition_attacked")
    assert [a["claim"] for a in attacks] == [final_id]
    assert len([c for c in backend.calls if c["role"] == "obligations"]) == 1
    assert len([c for c in backend.calls if c["role"] == "critic"]) == 1


def test_different_critic_budget_is_not_adopted(tmp_path, monkeypatch):
    state = _state(tmp_path)
    backend = FixtureBackend(_fixtures())
    handle = _start(state, backend, critic_rounds=2)
    handle.wait()
    service = _service(state, backend, monkeypatch, handle, critic_rounds=3)
    request_certificate(service.store, service.kernel, service.log,
                        service.artifacts, service.builder.finals[-1],
                        scope=CERTIFICATE_SCOPE_FINAL)
    record = _events(service, "contract_precompiled")
    assert record[0]["adopted"] is False
    assert "differs" in record[0]["reason"]
    # inline compile ran as before
    assert len([c for c in backend.calls if c["role"] == "obligations"]) == 2
    assert _events(service, "decomposition_attacked")


def test_failed_precompile_falls_back_to_inline_compile(tmp_path, monkeypatch):
    state = _state(tmp_path)
    fixtures = _fixtures()
    contract = fixtures["obligations"]
    calls = {"n": 0}

    def flaky(_variables):
        calls["n"] += 1
        if calls["n"] <= 2:  # both schema-retry attempts of the first call
            raise RoleError("transport")
        return contract

    fixtures["obligations"] = flaky
    backend = FixtureBackend(fixtures)
    handle = _start(state, backend)
    outcome = handle.wait()
    assert isinstance(outcome.error, RoleError)
    service = _service(state, backend, monkeypatch, handle)
    certificate = request_certificate(
        service.store, service.kernel, service.log, service.artifacts,
        service.builder.finals[-1], scope=CERTIFICATE_SCOPE_FINAL)
    assert certificate["verdict"] == "verified"
    record = _events(service, "contract_precompiled")
    assert record[0]["adopted"] is False
    assert record[0]["reason"].startswith("precompile failed")


def test_critic_fail_closed_is_replayed_as_evidence(tmp_path, monkeypatch):
    state = _state(tmp_path)
    fixtures = _fixtures()
    fixtures["critic"] = {"verdict": "insufficient",
                          "counterexample_world": "a world where it fails",
                          "missing_condition": "the missing condition"}
    backend = FixtureBackend(fixtures)
    handle = _start(state, backend, critic_rounds=2)
    outcome = handle.wait()
    assert outcome.payload is None
    assert outcome.evidence  # the critique recorded its fail-closed evidence
    service = _service(state, backend, monkeypatch, handle, critic_rounds=2)
    certificate = request_certificate(
        service.store, service.kernel, service.log, service.artifacts,
        service.builder.finals[-1], scope=CERTIFICATE_SCOPE_FINAL)
    assert certificate["verdict"] == "unverified"
    final_id = service.builder.finals[-1]
    assert _events(service, "contract_precompiled")[0]["adopted"] is True
    assert service.store.evidence_for(final_id)
    assert all(a["claim"] == final_id
               for a in _events(service, "decomposition_attacked"))


def test_precompile_runs_the_append_repair_loop(tmp_path, monkeypatch):
    """The borrowed critique loop must reach its static append guard."""
    state = _state(tmp_path)
    fixtures = _fixtures()
    contract = fixtures["obligations"]["obligations"]
    appended = {"obligations": contract + [{
        "text": "the command's result was observed", "required_basis": "record",
        "severity": "must", "clause": "Observe the result.",
        "refutable_by_absence": True}]}
    compiles = iter([fixtures["obligations"], appended])
    verdicts = iter([
        {"verdict": "insufficient",
         "counterexample_world": "the command ran but its result was never seen",
         "missing_condition": "APPEND OBLIGATION: the result was observed"},
        {"verdict": "pass", "counterexample_world": "n/a",
         "missing_condition": "n/a"},
    ])
    fixtures["obligations"] = lambda _v: next(compiles)
    fixtures["critic"] = lambda _v: next(verdicts)
    backend = FixtureBackend(fixtures)
    handle = _start(state, backend)
    outcome = handle.wait()
    assert outcome.error is None, repr(outcome.error)
    assert outcome.payload == appended
    assert [kind for kind, _p in outcome.events] == [
        "decomposition_attacked", "decomposition_attacked"]
    service = _service(state, backend, monkeypatch, handle)
    certificate = request_certificate(
        service.store, service.kernel, service.log, service.artifacts,
        service.builder.finals[-1], scope=CERTIFICATE_SCOPE_FINAL)
    assert certificate["verdict"] == "verified"
    assert _events(service, "contract_precompiled")[0]["adopted"] is True
    assert len(_events(service, "decomposition_attacked")) == 2


def _cached_service(state, backend, monkeypatch, handle):
    monkeypatch.setenv("GV_CONTRACT_CACHE", "1")
    return _service(state, backend, monkeypatch, handle)


def test_precompile_publishes_into_the_contract_cache_on_a_miss(
        tmp_path, monkeypatch):
    from verifier.engine.contract.cache import (
        default_compiled_final_contract_cache,
    )
    cache = default_compiled_final_contract_cache()
    cache.clear()
    state = _state(tmp_path)
    backend = FixtureBackend(_fixtures())
    handle = _start(state, backend, contract_cache=cache)
    outcome = handle.wait()
    assert outcome.error is None and outcome.bypass_reason is None
    assert len(cache) == 0  # only the kernel's own miss publishes

    service = _cached_service(state, backend, monkeypatch, handle)
    certificate = request_certificate(
        service.store, service.kernel, service.log, service.artifacts,
        service.builder.finals[-1], scope=CERTIFICATE_SCOPE_FINAL)
    assert certificate["verdict"] == "verified"
    final_id = service.builder.finals[-1]
    assert [m["claim"] for m in _events(service, "contract_cache_miss")] == [final_id]
    adopted = _events(service, "contract_precompiled")
    assert adopted[0]["adopted"] is True and adopted[0]["claim"] == final_id
    published = _events(service, "contract_cache_published")
    assert [p["claim"] for p in published] == [final_id]
    assert published[0]["key"] == outcome.key
    assert len(cache) == 1
    # one Compiler and one Critic call for the whole candidate
    assert len([c for c in backend.calls if c["role"] == "obligations"]) == 1
    assert len([c for c in backend.calls if c["role"] == "critic"]) == 1

    # The task's next candidate: the thread finds the published key and
    # spends nothing; the kernel's lookup is a plain hit.
    (tmp_path / "second").mkdir()
    second_state = _state(tmp_path / "second")
    second = _start(second_state, backend, contract_cache=cache)
    second_outcome = second.wait()
    assert second_outcome.key == outcome.key
    assert second_outcome.bypass_reason == "contract cache already holds this key"
    assert second_outcome.payload is None and second_outcome.events == []
    second_service = _cached_service(second_state, backend, monkeypatch, second)
    request_certificate(
        second_service.store, second_service.kernel, second_service.log,
        second_service.artifacts, second_service.builder.finals[-1],
        scope=CERTIFICATE_SCOPE_FINAL)
    assert _events(second_service, "contract_cache_hit")
    assert not _events(second_service, "contract_precompiled")
    assert len([c for c in backend.calls if c["role"] == "obligations"]) == 1
    cache.clear()


def test_precompile_yields_to_an_in_flight_compilation(tmp_path, monkeypatch):
    from verifier.engine.contract.cache import CompiledFinalContractCache
    cache = CompiledFinalContractCache()
    state = _state(tmp_path)
    backend = FixtureBackend(_fixtures())
    probe = _start(state, backend)  # cache-free run only to learn the key
    key = probe.wait().key
    calls_before = len(backend.calls)
    lease = cache.acquire(key)
    assert lease.leader
    try:
        handle = _start(state, backend, contract_cache=cache)
        outcome = handle.wait()
    finally:
        lease.finish()
    assert outcome.bypass_reason == (
        "contract compilation for this key is already in flight")
    assert len(backend.calls) == calls_before
