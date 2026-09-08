"""BindingGame wiring for the optional Judger evidence packet."""
from __future__ import annotations

import json
import os
import stat

import pytest

from verifier.core.events import EventLog
from verifier.core.model import BASIS_RECORD, Obligation
from verifier.engine import game as game_module
from verifier.engine.game import Attempt, BindingGame
from verifier.engine.judger_prefetch import JudgerEvidencePacket
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend


YES = {
    "sufficient": "yes",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "fixture independent audit",
}


def _method():
    return {
        "procedure": "inspect the complete retained exact record",
        "artifact": "one content-addressed evidence envelope",
        "level": "deterministic",
        "basis": BASIS_RECORD,
        "strength": "decisive",
        "bridges": [],
    }


def _fixture(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text("Check the retained exact record.")
    log = EventLog(str(state / "events.jsonl"))
    log.append("tool_call", {
        "tool": "instrument.unrelated-v3", "input": {"value": "alpha"},
    })
    artifacts = ArtifactStore(str(state / "artifacts"))
    digest = artifacts.put({
        "schema": "fixture-evidence-v1", "observed": "alpha"})
    result = DischargeResult(
        outcome="holds",
        witness=None,
        artifact=digest,
        instrument="fixture",
        basis=BASIS_RECORD,
        strength="decisive",
        inputs=({"ref": "ledger:events", "provenance": "ledger",
                 "role": "subject"},),
    )
    obligation = Obligation(
        "claim-1", "the retained exact record contains alpha", BASIS_RECORD)
    return state, log, artifacts, result, obligation


class _CaptureBackend:
    def __init__(self, *, enabled=False, mutate=None):
        self.judger_prefetch = enabled
        self.mutate = mutate
        self.calls = []

    def complete(self, role_name, prompt, variables):
        packet = variables.get("_judger_evidence_packet")
        self.calls.append({
            "role": role_name,
            "packet": packet,
            "ordinary_keys": tuple(sorted(
                key for key in variables if not key.startswith("_"))),
        })
        if self.mutate is not None and len(self.calls) == 1:
            self.mutate()
        return dict(YES)


def _judge(tmp_path, backend, *, retries=1, live_world=False):
    _state, log, artifacts, result, obligation = _fixture(tmp_path)
    game = BindingGame(
        backend, artifacts, log, None,
        live_world=live_world, judger_error_retries=retries)
    attempt = Attempt(
        method=_method(),
        discharge_artifact=result.artifact,
        discharge_outcome=result.outcome,
    )
    verdict = game._judge(
        obligation, _method(), result, attempt,
        method_attempt=1, independent_judger=True)
    return verdict, attempt, log, artifacts, result


@pytest.mark.parametrize("live_world", [False, True])
def test_metadata_only_change_does_not_spend_a_second_judger_call(
        tmp_path, live_world):
    def metadata_only():
        state = tmp_path / "state"
        for path in [state / "events.jsonl", *list((state / "artifacts").glob("*.json"))]:
            before = path.stat()
            os.chmod(path, stat.S_IMODE(before.st_mode))

    backend = _CaptureBackend(enabled=True, mutate=metadata_only)
    verdict, _attempt, log, _artifacts, _result = _judge(
        tmp_path, backend, retries=0, live_world=live_world)
    assert verdict == YES
    assert len(backend.calls) == 1
    assert type(backend.calls[0]["packet"]) is JudgerEvidencePacket
    assert not any(event["type"] == "judger_call_failed" for event in log.replay())


def test_default_off_preserves_the_original_variables_path(tmp_path, monkeypatch):
    backend = _CaptureBackend(enabled=False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled prefetch must not inspect source files")

    monkeypatch.setattr(
        game_module, "build_judger_evidence_packet", forbidden)
    verdict, attempt, _log, _artifacts, _result = _judge(tmp_path, backend)

    assert verdict == YES and attempt.audit == YES
    assert len(backend.calls) == 1
    assert backend.calls[0]["packet"] is None
    assert "_judger_evidence_packet" not in backend.calls[0]["ordinary_keys"]


@pytest.mark.parametrize("live_world", [False, True])
def test_enabled_packet_is_private_typed_and_freshly_revalidated(tmp_path, live_world):
    backend = _CaptureBackend(enabled=True)
    verdict, attempt, _log, _artifacts, result = _judge(
        tmp_path, backend, live_world=live_world)

    assert verdict == YES and attempt.audit == YES
    assert len(backend.calls) == 1
    packet = backend.calls[0]["packet"]
    assert type(packet) is JudgerEvidencePacket
    assert packet.artifact_digest == result.artifact
    assert json.loads(packet.render())["sources"]["ledger"]["attached"] is True
    # _judge appends its verdict only after packet revalidation.
    assert packet.revalidate() is False


@pytest.mark.parametrize("changed_source", ["ledger", "artifact"])
@pytest.mark.parametrize("live_world", [False, True])
def test_changed_packet_is_discarded_then_original_path_uses_existing_retry(
    tmp_path, changed_source, live_world,
):
    state, log, artifacts, result, obligation = _fixture(tmp_path)
    artifact_path = state / "artifacts" / f"{result.artifact}.json"

    def mutate():
        if changed_source == "ledger":
            log.append("message", {"text": "late concurrent append"})
        else:
            raw = artifact_path.read_bytes()
            replacement = artifact_path.with_suffix(".replacement")
            replacement.write_bytes(raw)
            replacement.replace(artifact_path)

    backend = _CaptureBackend(enabled=True, mutate=mutate)
    game = BindingGame(
        backend, artifacts, log, None,
        live_world=live_world, judger_error_retries=1)
    attempt = Attempt(_method(), discharge_artifact=result.artifact,
                      discharge_outcome=result.outcome)

    verdict = game._judge(
        obligation, _method(), result, attempt,
        method_attempt=1, independent_judger=True)

    assert verdict == YES and attempt.audit == YES
    assert len(backend.calls) == 2
    assert type(backend.calls[0]["packet"]) is JudgerEvidencePacket
    assert backend.calls[1]["packet"] is None
    failures = [event for event in log.replay()
                if event["type"] == "judger_call_failed"]
    assert len(failures) == 1
    assert failures[0]["payload"]["call"] == 1
    assert failures[0]["payload"]["will_retry"] is True
    assert "prefetch changed during audit" in failures[0]["payload"]["error"]


@pytest.mark.parametrize("live_world", [False, True])
def test_complete_packet_never_overrides_judger_rejection(tmp_path, live_world):
    backend = FixtureBackend({"judger": {
        **YES, "sufficient": "no", "reason": "fixture missing semantic evidence"}})
    backend.judger_prefetch = True
    verdict, attempt, _log, _artifacts, _result = _judge(
        tmp_path, backend, live_world=live_world)
    assert verdict["sufficient"] == "no"
    assert attempt.audit["sufficient"] == "no"
    assert len(backend.calls) == 1


def test_changed_packet_with_no_existing_retry_fails_closed(tmp_path):
    state, log, artifacts, result, obligation = _fixture(tmp_path)
    backend = _CaptureBackend(
        enabled=True,
        mutate=lambda: log.append("message", {"text": "late append"}),
    )
    game = BindingGame(
        backend, artifacts, log, None,
        live_world=False, judger_error_retries=0)
    attempt = Attempt(_method(), discharge_artifact=result.artifact,
                      discharge_outcome=result.outcome)

    verdict = game._judge(
        obligation, _method(), result, attempt,
        method_attempt=1, independent_judger=True)

    assert len(backend.calls) == 1
    assert verdict["sufficient"] == "unknown"
    assert verdict["faithful"] == "no" and verdict["_infra"] is True
    assert "prefetch changed during audit" in verdict["reason"]


def test_build_failure_uses_original_path_without_spending_retry(
    tmp_path, monkeypatch,
):
    backend = _CaptureBackend(enabled=True)
    calls = []

    def unavailable(*args):
        calls.append(args)
        return None

    monkeypatch.setattr(
        game_module, "build_judger_evidence_packet", unavailable)
    verdict, _attempt, _log, _artifacts, _result = _judge(tmp_path, backend)

    assert verdict == YES and len(calls) == 1
    assert len(backend.calls) == 1 and backend.calls[0]["packet"] is None


class _OneShotDischarger:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def discharge(self, *_args, **_kwargs):
        self.calls += 1
        return self.result


class _ExplodingCache:
    def acquire(self, *_args, **_kwargs):
        raise AssertionError("prefetch mode must not consult accepted-method cache")


def test_prefetch_mode_forces_fresh_oracle_executor_and_judger_even_with_cache(
    tmp_path,
):
    _state, log, artifacts, result, obligation = _fixture(tmp_path)
    backend = FixtureBackend({
        "oracles": {"methods": [_method()]},
        "judger": dict(YES),
    })
    backend.judger_prefetch = True
    discharger = _OneShotDischarger(result)
    game = BindingGame(
        backend, artifacts, log, discharger,
        live_world=False, method_cache=_ExplodingCache())
    row = {"id": "claim-1", "kind": "final",
           "text": "the retained exact record contains alpha"}

    settled = game.run(row, obligation)

    assert settled.outcome == "holds" and discharger.calls == 1
    assert [call["role"] for call in backend.calls] == ["oracles", "judger"]


@pytest.mark.parametrize("truthy", [1, "yes", object()])
def test_only_literal_true_enables_prefetch(tmp_path, truthy, monkeypatch):
    backend = _CaptureBackend(enabled=truthy)
    called = False

    def spy(*_args):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(game_module, "build_judger_evidence_packet", spy)
    verdict, _attempt, _log, _artifacts, _result = _judge(tmp_path, backend)
    assert verdict == YES and called is False
    assert backend.calls[0]["packet"] is None
