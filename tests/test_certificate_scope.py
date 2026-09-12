"""A final-scoped certificate settles only the task contract's claim."""
from __future__ import annotations

import copy
import json

import pytest

from verifier import offline
from verifier.engine.certificate import (
    CERTIFICATE_SCOPE_FINAL, CERTIFICATE_SCOPE_FULL, request_certificate,
)
from verifier.core.store import TRIGGER_LAZY
from verifier.llm.schema import FixtureBackend
from verifier.mcp_server import Service, _default_kernel
from test_frozen_record import _fixtures as _record_fixtures


def _fixtures():
    # a world-basis contract: a final resting only on record evidence would
    # abstain by policy, which is not what these tests are about
    fixtures = copy.deepcopy(_record_fixtures())
    fixtures["obligations"]["obligations"][0].update(
        {"text": "the delivered file holds the answer",
         "required_basis": "world", "refutable_by_absence": False})
    fixtures["oracles"]["methods"][0].update(
        {"procedure": "read the delivered file and compare", "basis": "world"})
    fixtures["discharge"]["inputs"] = [{
        "ref": "world:/app/answer.txt", "provenance": "delivered",
        "role": "subject"}]
    return fixtures


def _service(tmp_path, backend, monkeypatch):
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "0")
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text("Issue one command.\n", encoding="utf-8")
    with open(state / "hook_feed.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "tool_call", "payload": {
            "tool": "bash", "input": {"command": "true"},
            "tool_use_id": "call-1"}}) + "\n")
        handle.write(json.dumps({"type": "tool_result", "payload": {
            "tool": "bash", "tool_use_id": "call-1", "excerpt": ""}}) + "\n")
    service = Service(
        str(state), live_world=False, trigger=TRIGGER_LAZY,
        kernel_factory=lambda sc: _default_kernel(sc, None, backend=backend),
    )
    service.drain_hooks()
    service.commit("s1", "assertion", "one command ran", ["effect:2"])
    service.commit("s2", "assertion", "its output was empty", ["s1"])
    service.commit("f", "final", "the task is complete", ["s2"])
    return service


@pytest.mark.parametrize("scope", [CERTIFICATE_SCOPE_FULL, CERTIFICATE_SCOPE_FINAL])
def test_scope_changes_which_claims_are_settled_not_the_verdict(
        tmp_path, monkeypatch, scope):
    backend = FixtureBackend(_fixtures())
    service = _service(tmp_path, backend, monkeypatch)
    final_id = service.builder.finals[-1]

    certificate = request_certificate(
        service.store, service.kernel, service.log, service.artifacts,
        final_id, scope=scope)

    assert certificate["verdict"] == "verified"
    statuses = {entry["text"]: entry["status"] for entry in certificate["slice"]
                if entry["kind"] != "effect"}
    if scope == CERTIFICATE_SCOPE_FULL:
        assert set(statuses.values()) == {"verified"}
    else:
        assert statuses["the task is complete"] == "verified"
        assert statuses["one command ran"] == "unverified"
        assert statuses["its output was empty"] == "unverified"
    # the effect the narration cites is never model work; it stays a ledger
    # check in either scope and is only run when a verified claim needs it
    request = next(e for e in service.log.replay()
                   if e["type"] == "verify_request")
    assert request["payload"]["scope"] == scope
    roles = [call["role"] for call in backend.calls]
    if scope == CERTIFICATE_SCOPE_FULL:
        assert roles.count("obligations") == 3
    else:
        assert roles.count("obligations") == 1
        assert roles.count("judger") == 1


def test_unknown_scope_is_rejected_before_any_verification(tmp_path, monkeypatch):
    backend = FixtureBackend(_fixtures())
    service = _service(tmp_path, backend, monkeypatch)
    with pytest.raises(ValueError, match="unknown certificate scope"):
        request_certificate(service.store, service.kernel, service.log,
                            service.artifacts, service.builder.finals[-1],
                            scope="narrow")
    assert backend.calls == []


def test_scope_is_part_of_the_scoring_identity():
    full = offline.config_fields(model="m", seats="codex")
    final = offline.config_fields(model="m", seats="codex",
                                  certificate_scope="final")
    assert full["certificate_scope"] == "full"
    assert final["certificate_scope"] == "final"
    assert full["config"] != final["config"]
    with pytest.raises(ValueError, match="certificate_scope"):
        offline.config_fields(model="m", seats="codex",
                              certificate_scope="narrow")
