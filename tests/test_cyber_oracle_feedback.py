"""Hermetic compiler-transport checks; no model, CLI or benchmark calls."""
from __future__ import annotations

import copy
import json
import subprocess

import pytest

from verifier.core.events import EventLog
from verifier.core.model import Obligation, content_hash
from verifier.engine import game as game_module
from verifier.engine.game import Attempt, BindingGame, _attempt_diagnostic
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend, RoleError
from verifier.profiles.cyber import build_roles
from verifier.profiles.cyber.feedback import SCHEMA, filter_oracle_feedback


SECRET = "PRIVATE_RUNTIME_OUTPUT_NEVER_COMPILER_INPUT_世界"
ROW = {"id": "c1", "kind": "assertion", "text": "the declared security property holds"}
METHOD = {
    "procedure": "inspect the specified subject using the declared method",
    "artifact": "retained verification evidence",
    "level": "complete check",
    "basis": "world",
    "strength": "decisive",
    "bridges": [],
}
YES = {"sufficient": "yes", "faithful": "yes", "degradation": "n/a",
       "reason": "independent fixture audit"}


@pytest.fixture(autouse=True)
def no_external_process(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("this fixture may not launch a process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def _raw_feedback(*, reason="execution failure at retry limit: " + SECRET,
                  method=None, outcome="failed", audit=None, redos=None):
    method = copy.deepcopy(method or METHOD)
    return {
        "initial_empty": False,
        "repair_context": {"fixed_evidence_boundary": True, "output": SECRET},
        "structural_rejections": [{"procedure": method["procedure"], "reason": SECRET}],
        "unavailable_boundaries": [SECRET],
        "exhausted_attempts": [{
            "method": {"procedure": SECRET, "artifact": SECRET},
            "declared_method": method,
            "initial_outcome": outcome,
            "initial_audit": audit or {**YES, "reason": SECRET, "witness": SECRET},
            "redos": redos or [],
            "terminal_reason": reason,
            "private_artifact_bytes": SECRET,
        }],
        "deferred_methods": [],
        "raw_output": SECRET,
    }


def _filtered(source):
    return json.loads(filter_oracle_feedback(json.dumps(source, ensure_ascii=False)))


def test_filter_drops_outcomes_reasons_witnesses_private_bytes_and_unknown_fields():
    result = _filtered(_raw_feedback(redos=[{
        "reason": SECRET, "outcome": "refuted", "artifact": SECRET,
        "audit": {**YES, "reason": SECRET}, "output": SECRET,
    }]))
    encoded = json.dumps(result, ensure_ascii=False)
    assert SECRET not in encoded
    assert "outcome" not in encoded
    assert "reason" not in encoded
    assert "witness" not in encoded
    assert "audit" not in encoded
    assert result["schema"] == SCHEMA
    assert result["fixed_evidence_boundary"] is True
    assert result["unavailable_boundaries"] == {
        "count": 1, "state": "structurally_unavailable"}
    attempt = result["exhausted_attempts"][0]
    assert attempt["classification"] == "execution_failed"
    assert attempt["redo_count"] == 1 and attempt["already_retried"] is True
    assert attempt["declaration"] == METHOD


@pytest.mark.parametrize("prefix,classification", [
    ("sufficiency pre-audit:", "method_insufficient"),
    ("judger method insufficient (redo 2):", "method_insufficient"),
    ("judger degradation unjustified:", "degradation_not_authorized"),
    ("unobtainable (redo 1):", "capability_unavailable"),
    ("execution failure at retry limit (redo 1):", "execution_failed"),
    ("refutation below decisive strength is inadmissible:", "strength_scope_invalid"),
    ("refutation witness of kind:", "witness_protocol_invalid"),
    ("circular evidence:", "authority_independence_invalid"),
    ("an arbitrary audit sentence:", "method_rejected"),
])
def test_classification_is_fixed_and_never_contains_free_text(prefix, classification):
    result = _filtered(_raw_feedback(reason=prefix + SECRET))
    assert result["exhausted_attempts"][0]["classification"] == classification
    assert SECRET not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("audit,classification", [
    ({**YES, "sufficient": "no"}, "method_insufficient"),
    ({**YES, "faithful": "no"}, "execution_not_faithful"),
    ({**YES, "reason": "unobtainable: " + SECRET}, "method_rejected"),
])
def test_audit_repair_metadata_does_not_search_untrusted_reason(audit, classification):
    result = _filtered(_raw_feedback(reason="judger: " + SECRET, audit=audit))
    assert result["exhausted_attempts"][0]["classification"] == classification


def test_verifier_infrastructure_is_not_an_execution_redo_authority():
    source = _raw_feedback()
    source["exhausted_attempts"][0]["verifier_unavailable"] = True
    result = _filtered(source)
    assert result["exhausted_attempts"][0]["classification"] == "verifier_unavailable"
    assert result["exhausted_attempts"][0]["already_retried"] is False


def test_adapter_protocol_and_identity_are_predeclared_not_observation_hashed():
    method = {**METHOD, "adapter": "fixture.synthetic", "adapter_version": 1,
              "adapter_input_binding": "fixed-v1",
              "adapter_arguments": '{"operation":"compare"}',
              "adapter_inputs": [{"ref": "host-retained-subject", "role": "subject",
                                  "provenance": "task_given", "private": SECRET}],
              "redo_feedback": SECRET, "adapter_fallback_reason": SECRET}
    source = _raw_feedback(method=method)
    first = _filtered(source)["exhausted_attempts"][0]
    source["exhausted_attempts"][0]["initial_outcome"] = "refuted"
    source["exhausted_attempts"][0]["private_artifact_bytes"] += "another result"
    second = _filtered(source)["exhausted_attempts"][0]
    assert first == second
    assert first["execution"] == {
        "kind": "adapter", "adapter": "fixture.synthetic", "adapter_version": 1,
        "adapter_input_binding": "fixed-v1",
        "adapter_arguments": method["adapter_arguments"],
        "adapter_inputs": [{"ref": "host-retained-subject", "role": "subject",
                            "provenance": "task_given"}],
    }
    assert first["method_id"] == content_hash({**METHOD, "execution": first["execution"]})
    assert SECRET not in json.dumps(first, ensure_ascii=False)


def test_initial_feedback_has_no_fabricated_boundary_knowledge():
    result = json.loads(filter_oracle_feedback("(none)"))
    assert result["repair"] is False
    assert result["fixed_evidence_boundary"] is None
    assert result["exhausted_attempts"] == []


def test_unavailable_witness_text_does_not_claim_permanent_capability_loss():
    result = _filtered(_raw_feedback(reason="unobtainable: " + SECRET))
    assert result["unavailable_boundaries"]["state"] == "structurally_unavailable"
    assert "permanent" not in json.dumps(result)


def test_deferred_tiers_preserve_declared_protocol_without_reason():
    source = _raw_feedback()
    source["deferred_methods"] = [{"procedure": "declared weaker method",
                                   "strength": "bounded", "waiting_on": ["decisive"],
                                   "reason": SECRET}]
    result = _filtered(source)
    assert result["deferred_methods"] == [{"procedure": "declared weaker method",
                                           "strength": "bounded", "waiting_on": ["decisive"]}]
    assert SECRET not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("raw", [
    "not json " + SECRET, "[]", '{"initial_empty":true,"initial_empty":false}',
    '{"raw_output":NaN}', '{"initial_empty":"false"}',
    '{"exhausted_attempts":[42]}',
    '{"deferred_methods":[{"strength":"invented"}]}',
])
def test_invalid_feedback_fails_closed(raw):
    with pytest.raises(ValueError):
        filter_oracle_feedback(raw)


class _SequenceDischarger:
    def __init__(self, artifacts, outcomes, *, mutate=False):
        self.artifacts = artifacts
        self.outcomes = iter(outcomes)
        self.methods = []
        self.mutate = mutate

    def discharge(self, row, obligation, method=None, tag=""):
        self.methods.append(copy.deepcopy(method))
        outcome = next(self.outcomes)
        if self.mutate:
            method["procedure"] = SECRET
            self.mutate = False
        artifact = self.artifacts.put({"instrument": "fixture", "raw_output": SECRET})
        return DischargeResult(outcome=outcome, witness=SECRET, witness_kind="execution",
                               artifact=artifact, instrument="fixture")


def _game(tmp_path, *, cyber=True, outcomes=("failed", "holds"), judger=None,
          mutate=False, freeze=True):
    artifacts = ArtifactStore(str(tmp_path / "artifacts"))
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("session_start", {"task": "synthetic fixture"})
    if freeze:
        log.bind_record(ROW["id"], log.freeze_record(log.last_seq()))
    batches = iter([{"methods": [copy.deepcopy(METHOD)]},
                    {"methods": [{**METHOD, "procedure": "a different declared method"}]}])
    backend = FixtureBackend({"oracles": lambda _variables: next(batches),
                              "judger": judger or YES})
    discharger = _SequenceDischarger(artifacts, outcomes, mutate=mutate)
    role_args = {}
    if cyber:
        roles = build_roles()
        role_args = {"oracles_role": roles["oracles"], "judger_role": roles["judger"],
                     "oracle_feedback_filter": filter_oracle_feedback}
    game = BindingGame(backend, artifacts, log, discharger, **role_args)
    return game, backend, discharger


def test_cyber_filters_repair_but_keeps_original_attempts_evidence_and_logs(tmp_path):
    game, backend, _ = _game(tmp_path)
    result = game.run(ROW, Obligation(ROW["id"], ROW["text"]))
    assert result.outcome == "holds"
    oracle_calls = [call for call in backend.calls if call["role"] == "oracles"]
    assert len(oracle_calls) == 2
    assert SECRET not in json.dumps([call["variables"] for call in oracle_calls], ensure_ascii=False)
    repair = json.loads(oracle_calls[1]["variables"]["method_feedback"])
    assert repair["exhausted_attempts"][0]["classification"] == "execution_failed"
    assert SECRET in (tmp_path / "events.jsonl").read_text()
    retained = game.artifacts.get(result.artifact)
    assert SECRET in retained["attempts"][0]["rejected"]
    assert retained["attempts"][0]["discharge_outcome"] == "failed"
    judger_calls = [call for call in backend.calls if call["role"] == "judger"]
    assert SECRET in json.dumps(judger_calls[0]["variables"]["evidence"], ensure_ascii=False)


def test_default_keeps_original_feedback_and_attempt_serialization(tmp_path):
    game, backend, _ = _game(tmp_path, cyber=False)
    result = game.run(ROW, Obligation(ROW["id"], ROW["text"]))
    oracle_calls = [call for call in backend.calls if call["role"] == "oracles"]
    assert oracle_calls[0]["variables"]["method_feedback"] == "(none)"
    feedback = json.loads(oracle_calls[1]["variables"]["method_feedback"])
    assert feedback["exhausted_attempts"][0]["initial_outcome"] == "failed"
    assert SECRET in feedback["exhausted_attempts"][0]["terminal_reason"]
    assert "repair_context" not in feedback
    assert "declared_method" not in feedback["exhausted_attempts"][0]
    retained = game.artifacts.get(result.artifact)
    assert set(retained["attempts"][0]) == {
        "method", "rejected", "discharge_artifact", "discharge_outcome", "audit", "redos", "briefing"}


def test_cyber_preexecution_contract_survives_executor_mutation(tmp_path):
    game, backend, _ = _game(tmp_path, mutate=True)
    game.run(ROW, Obligation(ROW["id"], ROW["text"]))
    oracle_calls = [call for call in backend.calls if call["role"] == "oracles"]
    feedback = json.loads(oracle_calls[1]["variables"]["method_feedback"])
    assert feedback["exhausted_attempts"][0]["declaration"] == METHOD
    assert SECRET not in oracle_calls[1]["variables"]["method_feedback"]


def test_cyber_does_not_remove_executor_redo_feedback_or_blind_judger_evidence(tmp_path):
    audits = iter([{**YES, "faithful": "no", "reason": SECRET}] * 4 + [YES])
    game, backend, discharger = _game(
        tmp_path, outcomes=("holds",) * 5, judger=lambda _variables: next(audits))
    result = game.run(ROW, Obligation(ROW["id"], ROW["text"]))
    assert result.outcome == "holds"
    assert SECRET in discharger.methods[1]["redo_feedback"]
    oracle_calls = [call for call in backend.calls if call["role"] == "oracles"]
    feedback = json.loads(oracle_calls[1]["variables"]["method_feedback"])
    attempt = feedback["exhausted_attempts"][0]
    assert attempt["redo_count"] == 3 and attempt["already_retried"] is True
    assert attempt["classification"] == "execution_not_faithful"
    assert SECRET not in oracle_calls[1]["variables"]["method_feedback"]
    assert len([call for call in backend.calls if call["role"] == "judger"]) == 5


def test_cyber_record_inventory_is_frozen_but_keeps_exact_dialect_metadata(tmp_path):
    game, backend, _ = _game(tmp_path, freeze=False)
    game.log.append("tool_call", {"tool": "known.tool", "input": {"known_key": "frozen value"}})
    game.log.bind_record(ROW["id"], game.log.freeze_record(game.log.last_seq()))
    game.log.append("tool_call", {"tool": SECRET, "input": {SECRET: SECRET}})
    game._enumerate(ROW, Obligation(ROW["id"], ROW["text"]))
    shape = json.loads(backend.calls[0]["variables"]["record_shape"])
    assert shape["available"] is True
    assert shape["tool_call"]["tools"] == {"known.tool": 1}
    assert shape["tool_call"]["input_keys"] == {"known_key": 1}
    assert shape["source_event_count"] == 2
    assert SECRET not in json.dumps(shape, ensure_ascii=False)
    assert "frozen value" not in json.dumps(shape)


def test_cyber_unbound_record_never_falls_back_to_live_inventory(tmp_path):
    game, backend, _ = _game(tmp_path, freeze=False)
    game.log.append("tool_call", {"tool": SECRET, "input": {SECRET: SECRET}})
    game._enumerate(ROW, Obligation(ROW["id"], ROW["text"]))
    shape = json.loads(backend.calls[0]["variables"]["record_shape"])
    assert shape == {"schema": "gv-record-shape-v1", "available": False,
                     "reason": "frozen_record_inventory_unavailable"}


def test_cyber_record_inventory_failure_has_no_raw_error_reason(tmp_path, monkeypatch):
    game, backend, _ = _game(tmp_path)
    monkeypatch.setattr(game_module, "record_shape_json", lambda _log: json.dumps({
        "schema": "gv-record-shape-v1", "available": False, "reason": SECRET}))
    game._enumerate(ROW, Obligation(ROW["id"], ROW["text"]))
    assert SECRET not in backend.calls[0]["variables"]["record_shape"]


def test_cyber_world_note_hidden_from_compilers_but_retained_for_judger(tmp_path):
    (tmp_path / "world").mkdir()
    (tmp_path / "world_manifest.json").write_text(json.dumps({
        "captured_roots": ["/subject"], "scheme": "filesystem", "note": SECRET}))
    game, backend, _ = _game(tmp_path, outcomes=("holds",))
    game.sufficiency_role = build_roles()["sufficiency"]
    backend.fixtures["sufficiency"] = {"sufficient": "yes", "reason": "complete method"}
    result = game.run(ROW, Obligation(ROW["id"], ROW["text"]))
    assert result.outcome == "holds"
    for call in backend.calls:
        mapping = json.loads(call["variables"]["world_map"])
        if call["role"] in {"oracles", "sufficiency"}:
            assert "note" not in mapping
            assert mapping["captured_roots"] == ["/subject"]
        elif call["role"] == "judger":
            assert mapping["note"] == SECRET


def test_default_world_note_and_live_inventory_unchanged(tmp_path):
    (tmp_path / "world").mkdir()
    (tmp_path / "world_manifest.json").write_text(json.dumps({"note": SECRET}))
    game, backend, _ = _game(tmp_path, cyber=False, freeze=False)
    game.log.append("tool_call", {"tool": SECRET, "input": {"key": "value"}})
    game._enumerate(ROW, Obligation(ROW["id"], ROW["text"]))
    variables = backend.calls[0]["variables"]
    assert json.loads(variables["world_map"])["note"] == SECRET
    assert SECRET in variables["record_shape"]


def test_cyber_never_builds_compact_source_view_for_oracle(tmp_path, monkeypatch):
    game, backend, _ = _game(tmp_path)
    backend.compact_inspection = True
    calls = []
    original = game_module.bind_source_evidence_variables
    def observe(log, variables, *, enabled):
        calls.append(enabled)
        return original(log, variables, enabled=enabled)
    monkeypatch.setattr(game_module, "bind_source_evidence_variables", observe)
    game._enumerate(ROW, Obligation(ROW["id"], ROW["text"]))
    assert calls == [False]
    assert "_source_evidence_view" not in backend.calls[0]["variables"]
    assert backend.calls[0]["variables"]["ledger_path"] == game.log.record_path_for(ROW["id"])


@pytest.mark.parametrize("failure", ["raises", "not_text"])
def test_filter_failure_never_falls_back_to_raw_or_calls_oracle(tmp_path, failure):
    game, backend, _ = _game(tmp_path)
    def broken(_raw):
        if failure == "raises":
            raise ValueError(SECRET)
        return {"output": SECRET}
    game.oracle_feedback_filter = broken
    with pytest.raises(RoleError, match="^oracle feedback filtering failed$") as error:
        game._enumerate(ROW, Obligation(ROW["id"], ROW["text"]), method_feedback=SECRET)
    assert SECRET not in str(error.value)
    assert backend.calls == []


def test_default_attempt_diagnostic_contract_is_byte_equivalent():
    attempt = Attempt(method=METHOD, rejected="rejected " + SECRET,
                      discharge_outcome="failed", audit={**YES, "_private": SECRET},
                      redos=[{"reason": SECRET, "outcome": "failed", "audit": YES}])
    expected = {
        "method": {"procedure": METHOD["procedure"], "promised_artifact": METHOD["artifact"],
                   "level": METHOD["level"], "basis": "world", "strength": "decisive", "bridges": []},
        "initial_outcome": "failed", "initial_audit": YES,
        "redos": [{"reason": SECRET, "outcome": "failed", "audit": YES}],
        "terminal_reason": "rejected " + SECRET,
    }
    assert json.dumps(_attempt_diagnostic(attempt), ensure_ascii=False) == json.dumps(expected, ensure_ascii=False)
