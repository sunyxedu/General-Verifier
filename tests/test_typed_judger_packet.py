"""Fail-closed boundaries for the typed-adapter Judger packet seam."""
from __future__ import annotations

import json

import pytest

from verifier.core.events import EventLog
from verifier.core.model import BASIS_WORLD, Obligation
from verifier.engine import game as game_module
from verifier.engine.game import Attempt, BindingGame
from verifier.engine import typed_judger_packet as packet_module
from verifier.execution.adapters import (
    AdapterRoutingDischarger,
    default_adapter_registry,
)
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore


YES = {
    "sufficient": "yes",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "fresh independent fixture review",
}


class _Fallback:
    def discharge(self, *_args, **_kwargs):
        raise AssertionError("fixture must stay on the typed adapter")


class _CaptureBackend:
    def __init__(self, *, images=False, mutate=None):
        self.calls = []
        self.images = images
        self.mutate = mutate

    def _captured_images(self, role_name, variables):
        assert role_name == "judger"
        return (object(),) if self.images else ()

    def complete(self, role_name, prompt, variables):
        self.calls.append({
            "role": role_name,
            "packet": variables.get("_typed_adapter_judger_packet"),
        })
        if self.mutate is not None and len(self.calls) == 1:
            self.mutate()
        return dict(YES)


def _fixture(tmp_path):
    state = tmp_path / "state"
    state.mkdir(parents=True)
    (state / "task.md").write_text(
        "The delivered marker must exist.", encoding="utf-8")
    log = EventLog(str(state / "events.jsonl"))
    artifacts = ArtifactStore(str(state / "artifacts"))
    target = tmp_path / "marker.txt"
    target.write_text("alpha", encoding="utf-8")
    obligation = Obligation(
        "claim-1",
        "the delivered marker exists as a regular file",
        BASIS_WORLD,
        clause="marker must exist",
    )
    method = {
        "procedure": "inspect the exact delivered address and file kind",
        "artifact": "a content-addressed adapter evidence envelope",
        "level": "deterministic",
        "basis": BASIS_WORLD,
        "strength": "decisive",
        "bridges": [],
        "adapter": "filesystem.exists",
        "adapter_version": 1,
        "adapter_arguments": json.dumps({
            "address": str(target), "kind": "file"}, sort_keys=True),
        "adapter_inputs": [{
            "ref": str(target), "provenance": "delivered",
            "role": "subject",
        }],
    }
    route = AdapterRoutingDischarger(
        _Fallback(), default_adapter_registry(), artifacts, log,
        str(state), live_world=True)
    row = {"id": obligation.claim_id, "kind": "final",
           "text": obligation.text}
    result = route.discharge(row, obligation, method)
    assert result.instrument == "adapter:filesystem.exists@1"
    return state, target, log, artifacts, route, obligation, method, result


def _build(tmp_path, **kwargs):
    fixture = _fixture(tmp_path)
    _state, _target, _log, artifacts, route, obligation, method, result = fixture
    packet = packet_module.build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result,
        "(none — decisive method)", **kwargs)
    return fixture, packet


def test_packet_binds_complete_inputs_and_replays_without_old_judgment(tmp_path):
    (state, _target, _log, _artifacts, _route, obligation, method, result), packet = \
        _build(tmp_path)

    assert type(packet) is packet_module.TypedAdapterJudgerPacket
    document = json.loads(packet.render())
    assert document["schema"] == packet_module.TYPED_JUDGER_PACKET_PROTOCOL
    assert document["task_context"] == json.dumps({
        "path": str(state / "task.md"),
        "text": "The delivered marker must exist.",
    }, ensure_ascii=False)
    assert document["obligation"]["id"] == obligation.id
    assert document["method_ir"] == method
    assert document["adapter_contract"]["identity"] == "filesystem.exists@1"
    assert document["adapter_contract"]["invocation"]["inputs"] \
        == method["adapter_inputs"]
    assert document["result"]["artifact_digest"] == result.artifact
    assert document["content_addressed_evidence"]["digest"] == result.artifact
    assert document["verifier_execution_receipt"][
        "fresh_exact_replay_digest"] == packet.replay_digest
    assert document["source_slot_coverage"] \
        ["all_declared_inputs_freshly_read"] is True
    assert document["source_slot_coverage"] \
        ["original_source_bytes_attached"] is False
    assert "verdict" not in document and "audit" not in document
    assert packet.revalidate() is True


def test_initial_eligibility_is_closed_and_ledger_prefix_receipts_replay(tmp_path):
    assert packet_module.ELIGIBLE_ADAPTERS == frozenset({
        ("filesystem.exists", 1),
        ("filesystem.sha256", 1),
        ("document.json-value", 1),
        ("ledger.event", 1),
        ("ledger.tool-call-string", 1),
        ("ledger.tool-call-string", 2),
    })
    for version in (1, 2):
        state = tmp_path / f"ledger-v{version}"
        state.mkdir()
        (state / "task.md").write_text(
            "The retained tool output must equal alpha.", encoding="utf-8")
        log = EventLog(str(state / "events.jsonl"))
        log.append("tool_call", {"tool": "fixture.tool", "output": "alpha"})
        artifacts = ArtifactStore(str(state / "artifacts"))
        obligation = Obligation(
            f"claim-v{version}", "the retained output equals alpha")
        arguments = {
            "selector": "last",
            "value_pointer": "/output",
            "through_seq": 1,
            "tool_names": ["fixture.tool"],
        }
        if version == 1:
            arguments["expected"] = "alpha"
        else:
            arguments.update({"predicate": "equals", "expected": "alpha"})
        method = {
            "procedure": "compare the exact retained output",
            "artifact": "complete bounded prefix receipt",
            "level": "deterministic",
            "basis": "record",
            "strength": "decisive",
            "bridges": [],
            "adapter": "ledger.tool-call-string",
            "adapter_version": version,
            "adapter_arguments": json.dumps(arguments, sort_keys=True),
            "adapter_inputs": [{
                "ref": "ledger:events", "provenance": "ledger",
                "role": "subject",
            }],
        }
        route = AdapterRoutingDischarger(
            _Fallback(), default_adapter_registry(), artifacts, log,
            str(state), live_world=False)
        result = route.discharge(
            {"id": obligation.claim_id}, obligation, method)
        packet = packet_module.build_typed_adapter_judger_packet(
            route, artifacts, obligation, method, result, "decisive")
        assert packet is not None and packet.revalidate() is True
        receipt = json.loads(packet.render())["verifier_execution_receipt"]
        assert receipt["observation"]["matches"] is True


def test_world_or_task_change_fails_post_call_replay(tmp_path):
    (state, target, *_rest), packet = _build(tmp_path)
    target.write_text("alpha changed", encoding="utf-8")
    assert packet.revalidate() is False

    (_state2, _target2, *_rest2), packet2 = _build(tmp_path / "second")
    task = tmp_path / "second" / "state" / "task.md"
    task.write_text("changed authority", encoding="utf-8")
    assert packet2.revalidate() is False


def test_evidence_inode_or_catalog_change_fails_revalidation(tmp_path, monkeypatch):
    (state, _target, _log, _artifacts, route, _obligation, _method, result), packet = \
        _build(tmp_path)
    path = state / "artifacts" / f"{result.artifact}.json"
    raw = path.read_bytes()
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(raw)
    replacement.replace(path)
    assert packet.revalidate() is False

    (_fixture2, packet2) = _build(tmp_path / "second")
    route2 = _fixture2[4]
    monkeypatch.setattr(route2, "adapter_catalog", lambda: "[]")
    assert packet2.revalidate() is False


@pytest.mark.parametrize("identity", [
    ("browser.html", 1),
    ("ledger.tool-call-string", 3),
    ("extension.custom", 1),
])
def test_browser_observed_eof_and_extension_adapters_are_not_eligible(
    tmp_path, identity,
):
    fixture = _fixture(tmp_path)
    _state, _target, _log, artifacts, route, obligation, method, result = fixture
    changed = dict(method, adapter=identity[0], adapter_version=identity[1])
    assert packet_module.build_typed_adapter_judger_packet(
        route, artifacts, obligation, changed, result, "none") is None


def test_agent_images_incomplete_and_oversized_evidence_fall_back(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path)
    state, _target, _log, artifacts, route, obligation, method, result = fixture
    assert packet_module.build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result, "none",
        captured_images_present=True) is None
    assert packet_module.build_typed_adapter_judger_packet(
        route, artifacts, obligation, method,
        DischargeResult(
            outcome=result.outcome, witness=result.witness,
            artifact=result.artifact, instrument="agent:discharge",
            inputs=result.inputs),
        "none") is None

    artifact_path = state / "artifacts" / f"{result.artifact}.json"
    artifact_path.write_text('{"incomplete":', encoding="utf-8")
    assert packet_module.build_typed_adapter_judger_packet(
        route, artifacts, obligation, method, result, "none") is None

    # Restore a new fixture and force the final all-or-nothing cap below one
    # complete document. No partial rendering is returned.
    fixture2 = _fixture(tmp_path / "second")
    artifacts2, route2, obligation2, method2, result2 = (
        fixture2[3], fixture2[4], fixture2[5], fixture2[6], fixture2[7])
    monkeypatch.setattr(packet_module, "MAX_TYPED_JUDGER_PACKET_BYTES", 1)
    assert packet_module.build_typed_adapter_judger_packet(
        route2, artifacts2, obligation2, method2, result2, "none") is None


@pytest.mark.parametrize("source_defect", ["unused", "duplicate"])
def test_unused_or_duplicate_declared_source_slot_falls_back(
    tmp_path, source_defect,
):
    fixture = _fixture(tmp_path)
    _state, target, _log, artifacts, route, obligation, method, _result = fixture
    extra = {
        "ref": str(target) + ".authority",
        "provenance": "task_given",
        "role": "authority",
    }
    changed = dict(method)
    changed["adapter_inputs"] = list(method["adapter_inputs"])
    changed["adapter_inputs"].append(
        extra if source_defect == "unused" else dict(method["adapter_inputs"][0]))
    # The ordinary adapter permits advisory/duplicate declarations, but the
    # packet must not call either one mechanically covered.
    result = route.discharge(
        {"id": obligation.claim_id}, obligation, changed)
    assert packet_module.build_typed_adapter_judger_packet(
        route, artifacts, obligation, changed, result, "none") is None


def test_binding_game_default_off_and_image_fallback_preserve_original_path(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path)
    _state, _target, log, artifacts, route, obligation, method, result = fixture
    backend = _CaptureBackend()
    game = BindingGame(backend, artifacts, log, route, live_world=True)
    attempt = Attempt(method, discharge_artifact=result.artifact,
                      discharge_outcome=result.outcome)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("default-off game must not build a typed packet")

    monkeypatch.setattr(
        game_module, "build_typed_adapter_judger_packet", forbidden)
    verdict = game._judge(obligation, method, result, attempt)
    assert verdict == YES
    assert backend.calls[0]["packet"] is None

    monkeypatch.undo()
    backend2 = _CaptureBackend(images=True)
    game2 = BindingGame(
        backend2, artifacts, log, route, live_world=True,
        typed_adapter_judger_packet=True)
    attempt2 = Attempt(method, discharge_artifact=result.artifact,
                       discharge_outcome=result.outcome)
    assert game2._judge(obligation, method, result, attempt2) == YES
    assert backend2.calls[0]["packet"] is None


def test_enabled_game_attaches_private_packet_and_disables_method_reuse(tmp_path):
    fixture = _fixture(tmp_path)
    _state, _target, log, artifacts, route, obligation, method, result = fixture
    backend = _CaptureBackend()

    class _ExplodingCache:
        def acquire(self, *_args, **_kwargs):
            raise AssertionError("packet experiment must disable method reuse")

    game = BindingGame(
        backend, artifacts, log, route, live_world=True,
        method_cache=_ExplodingCache(), typed_adapter_judger_packet=True)
    assert game._method_cache_key(
        {"id": obligation.claim_id}, obligation) is None
    attempt = Attempt(method, discharge_artifact=result.artifact,
                      discharge_outcome=result.outcome)
    assert game._judge(obligation, method, result, attempt) == YES
    assert type(backend.calls[0]["packet"]) \
        is packet_module.TypedAdapterJudgerPacket


def test_post_call_replay_change_uses_existing_judger_retry(tmp_path):
    fixture = _fixture(tmp_path)
    _state, target, log, artifacts, route, obligation, method, result = fixture
    backend = _CaptureBackend(
        mutate=lambda: target.write_text("changed", encoding="utf-8"))
    game = BindingGame(
        backend, artifacts, log, route, live_world=True,
        judger_error_retries=1, typed_adapter_judger_packet=True)
    attempt = Attempt(method, discharge_artifact=result.artifact,
                      discharge_outcome=result.outcome)

    assert game._judge(obligation, method, result, attempt) == YES
    assert len(backend.calls) == 2
    assert type(backend.calls[0]["packet"]) \
        is packet_module.TypedAdapterJudgerPacket
    assert backend.calls[1]["packet"] is None
    failures = [event for event in log.replay()
                if event["type"] == "judger_call_failed"]
    assert len(failures) == 1
    assert "typed-adapter Judger packet changed" in failures[0]["payload"]["error"]


@pytest.mark.parametrize("invalid", [1, "1", None, object()])
def test_binding_game_requires_literal_bool(tmp_path, invalid):
    fixture = _fixture(tmp_path)
    _state, _target, log, artifacts, route, *_rest = fixture
    with pytest.raises(ValueError, match="must be a bool"):
        BindingGame(
            _CaptureBackend(), artifacts, log, route,
            typed_adapter_judger_packet=invalid)
