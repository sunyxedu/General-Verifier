"""No-network tests for the explicit single-arm contract replay diagnostic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.terminal_bench import replay_prepared_contracts as replay
from test_contract_replay import fixture_bundle, snapshot, source_state
from verifier.contract_replay import ReplaySetupError, replay_identity
from verifier.core.store import TRIGGER_LAZY
from verifier.engine.game import BindingGame
from verifier.engine.kernel import Kernel
from verifier.engine.method_cache import AcceptedMethodCache
from verifier.execution.discharge import DischargeResult
from verifier.llm.codex_backend import CodexBackend
from verifier.llm.schema import FixtureBackend
from verifier.mcp_server import Service
from verifier.llm.oracle_execution import raise_oracle_execution_payload


CONFIG = {
    "model": "gpt-5.6-luna",
    "effort": "medium",
    "service_tier": "standard",
    "typed_adapters": False,
}
ORACLE = {"methods": [{
    "procedure": "independently inspect the captured task input",
    "artifact": "a retained check result",
    "level": "independent recomputation",
    "basis": "world",
    "strength": "decisive",
    "bridges": [],
}]}
JUDGER = {
    "sufficient": "yes",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "the retained independent check decides the obligation",
}
USAGE = {
    "input_tokens": 100,
    "cached_input_tokens": 40,
    "output_tokens": 20,
    "reasoning_output_tokens": 5,
}


class CountingDischarger:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append({"claim": row["id"], "obligation": obligation.id})
        digest = self.artifacts.put({
            "instrument": "local-fixture-executor",
            "claim": row["id"],
            "obligation": obligation.id,
            "ordinal": len(self.calls),
        })
        return DischargeResult(
            outcome="holds",
            witness="the local fixture check held",
            artifact=digest,
            instrument="local-fixture-executor",
            basis="world",
            strength="decisive",
            witness_kind="execution",
        )


def _service_factory(
    backend_factory, executors, *, method_cache=None, binders=None,
):
    def factory(state):
        state = Path(state)
        backend = backend_factory(state)

        def kernel_factory(sc):
            executor = CountingDischarger(sc.artifacts)
            executors.append(executor)
            binder = BindingGame(
                backend, sc.artifacts, sc.log, executor, live_world=False,
                method_cache=method_cache)
            if binders is not None:
                binders.append(binder)
            return Kernel(
                sc.store,
                sc.log,
                backend,
                sc.artifacts,
                discharger=executor,
                binder=binder,
                live_world=False,
                parallel_obligations=1,
                parallel_claims=1,
            )

        return Service(
            str(state),
            trigger=TRIGGER_LAZY,
            live_world=False,
            kernel_factory=kernel_factory,
        )

    return factory


def _prepared(tmp_path, factory):
    source = source_state(tmp_path)
    original = factory(source)
    graph = snapshot(original)
    identity = replay_identity(original.kernel, graph, CONFIG)
    raw = fixture_bundle(original, CONFIG)
    rounds = json.loads((source / "extract.json").read_text())["rounds"]
    return source, identity, raw, rounds


def _run(source, identity, raw, rounds, destination, factory, **changes):
    kwargs = dict(
        replay_id="replay-1",
        setup_id="setup-1",
        arm="treatment",
        service_factory=factory,
        config=CONFIG,
        extracted_rounds=rounds,
        bundle_bytes=raw,
        expected_bundle_sha256=hashlib.sha256(raw).hexdigest(),
        expected_identity=identity,
    )
    kwargs.update(changes)
    return replay.replay_prepared_contracts(
        source.resolve(), destination.resolve(), **kwargs)


def test_process_stub_replays_only_oracle_executor_judger_and_accounts_context(
        tmp_path, monkeypatch):
    process_calls = []
    executors = []
    binders = []
    shared_method_cache = AcceptedMethodCache(max_entries=7)

    def version(command, **_kwargs):
        assert command == ["codex", "--version"]
        return SimpleNamespace(returncode=0, stdout="codex-cli 0.153.4")

    def process(command, **kwargs):
        instruction = command[-1]
        role = "judger" if '"sufficient"' in instruction else "oracles"
        payload = (JUDGER if role == "judger"
                   else raise_oracle_execution_payload(ORACLE))
        output = json.dumps(payload)
        Path(command[command.index("-o") + 1]).write_text(output)
        process_calls.append({"role": role, "cwd": kwargs["cwd"]})
        events = [
            {"type": "thread.started", "thread_id": f"stub-{len(process_calls)}"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {
                "id": "answer", "type": "agent_message", "text": output}},
            {"type": "turn.completed", "usage": USAGE},
        ]
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(json.dumps(item) for item in events),
            stderr="",
        )

    monkeypatch.setattr("verifier.llm.authority_backend.subprocess.run", version)
    monkeypatch.setattr("verifier.llm.codex_backend._run_codex_process", process)
    factory = _service_factory(
        lambda state: CodexBackend(
            cwd=str(state), model="gpt-5.6-luna", effort="medium",
            service_tier="standard"),
        executors,
        method_cache=shared_method_cache,
        binders=binders,
    )
    source, identity, raw, rounds = _prepared(tmp_path, factory)
    executors.clear()  # the identity-only source Service never executed evidence

    result = _run(
        source, identity, raw, rounds, tmp_path / "replay", factory)

    assert result["diagnostic_status"] == "complete"
    assert result["complete"] is True
    assert result["verification_complete"] is True
    assert result["accounting_complete"] is True
    assert result["features"]["verdict"] == "verified"
    assert result["features"]["extract_rounds"] == rounds
    assert result["cost_report"]["complete"] is True
    assert result["cost_report"]["metrics"]["started_attempts"] == 4
    assert result["cost_report"]["metrics"]["wall_seconds"] \
        == result["arm_wall_seconds"]
    assert len(result["usage_sha256"]) == len(result["spans_sha256"]) == 64
    assert [item["role"] for item in process_calls] == [
        "oracles", "judger", "oracles", "judger"]
    assert len(executors) == 1 and len(executors[0].calls) == 2
    assert binders[-1].method_cache is not shared_method_cache
    assert binders[-1].method_cache.max_entries == shared_method_cache.max_entries
    usage = [json.loads(line) for line in
             (tmp_path / "replay" / "replay-usage.jsonl").read_text().splitlines()]
    assert all(row["replay_id"] == "replay-1"
               and row["setup_id"] == "setup-1"
               and row["arm"] == "treatment" for row in usage)
    # Neither prepared contract generation role can run in the replay arm.
    assert not any(item["role"] in {"obligations", "critic"}
                   for item in process_calls)
    assert not (tmp_path / "replay" / "state" / "score.json").exists()


def test_fixture_certificate_is_retained_but_unknown_cost_is_inconclusive(tmp_path):
    backend = FixtureBackend({"oracles": ORACLE, "judger": JUDGER})
    executors = []
    factory = _service_factory(lambda _state: backend, executors)
    source, identity, raw, rounds = _prepared(tmp_path, factory)
    backend.calls.clear()
    executors.clear()

    result = _run(
        source, identity, raw, rounds, tmp_path / "fixture-replay", factory)

    assert result["diagnostic_status"] == "inconclusive"
    assert result["complete"] is False
    assert result["verification_complete"] is True
    assert result["accounting_complete"] is False
    assert result["features"]["verdict"] == "verified"
    assert result["certificate"]["slice"]
    assert result["cost_report"]["metrics"]["started_attempts"] is None
    assert result["usage_sha256"] is None
    assert len(result["spans_sha256"]) == 64
    assert "replay_accounting_incomplete" in result["findings"]
    assert [call["role"] for call in backend.calls] == [
        "oracles", "judger", "oracles", "judger"]
    assert len(executors) == 1 and len(executors[0].calls) == 2


def test_round_mismatch_is_terminal_and_never_constructs_service(tmp_path):
    backend = FixtureBackend({"oracles": ORACLE, "judger": JUDGER})
    executors = []
    built = []
    base = _service_factory(lambda _state: backend, executors)

    def factory(state):
        built.append(state)
        return base(state)

    source, identity, raw, rounds = _prepared(tmp_path, base)
    destination = tmp_path / "wrong-rounds"

    result = _run(
        source, identity, raw, rounds, destination, factory,
        extracted_rounds=rounds + 1)

    assert result["status"] == "terminal"
    assert result["diagnostic_status"] == "inconclusive"
    assert result["error_class"] == "ReplaySetupError"
    assert built == []
    assert (destination / "replay-start.json").exists()
    assert (destination / "replay-result.json").exists()
    assert (destination / "state").exists()


def test_loader_failure_has_no_outer_retry_and_keeps_terminal_state(tmp_path):
    backend = FixtureBackend({"oracles": ORACLE, "judger": JUDGER})
    executors = []
    builds = []
    base = _service_factory(lambda _state: backend, executors)

    def factory(state):
        builds.append(state)
        return base(state)

    source, identity, _raw, rounds = _prepared(tmp_path, base)
    malformed = b"{}"
    destination = tmp_path / "bad-bundle"

    result = _run(
        source, identity, malformed, rounds, destination, factory,
        expected_bundle_sha256=hashlib.sha256(malformed).hexdigest())

    assert result["status"] == "terminal"
    assert result["complete"] is False
    assert result["error_class"] in {"ContractReplayError", "ReplaySetupError"}
    assert len(builds) == 1
    assert backend.calls == []
    assert (destination / "state" / "events.jsonl").exists()
    assert (destination / "replay-result.json").exists()


def test_nonlazy_factory_is_rejected_before_any_evidence_call(tmp_path):
    backend = FixtureBackend({"oracles": ORACLE, "judger": JUDGER})
    executors = []
    base = _service_factory(lambda _state: backend, executors)
    source, identity, raw, rounds = _prepared(tmp_path, base)

    def nonlazy(state):
        service = base(state)
        service.store.trigger = "on_commit"
        return service

    result = _run(
        source, identity, raw, rounds, tmp_path / "nonlazy", nonlazy)

    assert result["diagnostic_status"] == "inconclusive"
    assert result["error_class"] == "ReplaySetupError"
    assert backend.calls == []
    assert all(not executor.calls for executor in executors)


def test_custom_method_cache_cannot_import_cross_arm_judger_state(tmp_path):
    backend = FixtureBackend({"oracles": ORACLE, "judger": JUDGER})
    executors = []

    class UnknownCache:
        pass

    factory = _service_factory(
        lambda _state: backend, executors, method_cache=UnknownCache())
    source, identity, raw, rounds = _prepared(tmp_path, factory)

    result = _run(
        source, identity, raw, rounds, tmp_path / "custom-cache", factory)

    assert result["diagnostic_status"] == "inconclusive"
    assert result["error_class"] == "ReplaySetupError"
    assert backend.calls == []
    assert all(not executor.calls for executor in executors)


def test_source_change_after_certificate_withholds_diagnostic_features(
        tmp_path, monkeypatch):
    backend = FixtureBackend({"oracles": ORACLE, "judger": JUDGER})
    executors = []
    factory = _service_factory(lambda _state: backend, executors)
    source, identity, raw, rounds = _prepared(tmp_path, factory)
    backend.calls.clear()
    original = replay.request_certificate

    def mutate_after(*args, **kwargs):
        certificate = original(*args, **kwargs)
        (source / "task.md").write_text("mutated after replay evidence")
        return certificate

    monkeypatch.setattr(replay, "request_certificate", mutate_after)

    result = _run(
        source, identity, raw, rounds, tmp_path / "mutated", factory)

    assert result["verification_complete"] is False
    assert result["features"] is None
    assert result["certificate"] is None
    assert "replay_final_validation_failed" in result["findings"]
    assert (tmp_path / "mutated" / "replay-result.json").exists()


def test_usage_change_during_audit_cannot_publish_complete_cost(
        tmp_path, monkeypatch):
    process_calls = []
    executors = []

    monkeypatch.setattr(
        "verifier.llm.authority_backend.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="codex-cli 0.153.4"),
    )

    def process(command, **_kwargs):
        instruction = command[-1]
        payload = JUDGER if '"sufficient"' in instruction else ORACLE
        output = json.dumps(payload)
        Path(command[command.index("-o") + 1]).write_text(output)
        process_calls.append(payload)
        events = [
            {"type": "thread.started", "thread_id": f"stub-{len(process_calls)}"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {
                "id": "answer", "type": "agent_message", "text": output}},
            {"type": "turn.completed", "usage": USAGE},
        ]
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(json.dumps(item) for item in events), stderr="")

    monkeypatch.setattr("verifier.llm.codex_backend._run_codex_process", process)
    factory = _service_factory(
        lambda state: CodexBackend(cwd=str(state), model="gpt-5.6-luna"),
        executors,
    )
    source, identity, raw, rounds = _prepared(tmp_path, factory)
    original_audit = replay._audit_model_work

    def mutate(usage_path, *args, **kwargs):
        answer = original_audit(usage_path, *args, **kwargs)
        with usage_path.open("ab") as handle:
            handle.write(b"\n")
        return answer

    monkeypatch.setattr(replay, "_audit_model_work", mutate)

    result = _run(
        source, identity, raw, rounds, tmp_path / "usage-change", factory)

    assert result["verification_complete"] is True
    assert result["accounting_complete"] is False
    assert result["cost_report"]["metrics"]["started_attempts"] is None
    assert "replay_usage_ledgers_changed_during_audit" in result["findings"]


def test_interrupt_writes_terminal_record_then_propagates(tmp_path):
    backend = FixtureBackend({"oracles": ORACLE, "judger": JUDGER})
    executors = []
    base = _service_factory(lambda _state: backend, executors)
    source, identity, raw, rounds = _prepared(tmp_path, base)
    destination = tmp_path / "interrupted"

    def interrupted(_state):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _run(source, identity, raw, rounds, destination, interrupted)

    terminal = json.loads((destination / "replay-result.json").read_text())
    assert terminal["status"] == "terminal"
    assert terminal["diagnostic_status"] == "inconclusive"
    assert terminal["error_class"] == "KeyboardInterrupt"
    assert terminal["features"] is None


@pytest.mark.parametrize("field", ["source", "destination"])
def test_relative_or_parent_paths_are_rejected_before_output(tmp_path, field):
    backend = FixtureBackend({"oracles": ORACLE, "judger": JUDGER})
    executors = []
    factory = _service_factory(lambda _state: backend, executors)
    source, identity, raw, rounds = _prepared(tmp_path, factory)
    destination = tmp_path / "path-check"
    source_arg = source.resolve()
    destination_arg = destination.resolve()
    if field == "source":
        source_arg = Path("relative-source")
    else:
        destination_arg = tmp_path / "nested" / ".." / "path-check"

    with pytest.raises(ReplaySetupError, match="absolute normalized"):
        replay.replay_prepared_contracts(
            source_arg,
            destination_arg,
            replay_id="replay-1",
            setup_id="setup-1",
            arm="control",
            service_factory=factory,
            config=CONFIG,
            extracted_rounds=rounds,
            bundle_bytes=raw,
            expected_bundle_sha256=hashlib.sha256(raw).hexdigest(),
            expected_identity=identity,
        )
    assert not destination.exists()
