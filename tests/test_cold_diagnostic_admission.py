"""Independent fail-closed admission tests for the cold diagnostic."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.terminal_bench import replay_prepared_contracts
from experiments.terminal_bench import run_cold_adapter_diagnostic as runner
from test_replay_prepared_contracts import (
    CONFIG, JUDGER, ORACLE, USAGE, _prepared, _service_factory,
)
from verifier import offline
from verifier.core.model import content_hash
from verifier.llm.codex_backend import CodexBackend


def _feature(*, verdict="verified", status="verified", reason="first wording"):
    obligation = {
        "ordinal": 0,
        "id": "a" * 64,
        "text": "the delivered result satisfies the requested condition",
        "premise_scope": [],
        "required_basis": "world",
        "severity": "must",
        "clause": "satisfies the requested condition",
        "refutable_by_absence": False,
    }
    outcome = {
        "ordinal": 0,
        "obligation_id": obligation["id"],
        "evaluated": True,
        "status": status,
        "reason": reason,
        "basis": "world",
        "strength": "decisive",
    }
    return {
        "feature_schema_version": offline.FEATURE_SCHEMA_VERSION,
        "unscoreable": False,
        "verdict": verdict,
        "verdict_strength": "decisive",
        "certificate_id": "b" * 64,
        "slice_size": 1,
        "claims": {status: 1},
        "claims_effective": {status: 1},
        "conditional_verified": 0,
        "obligations": {status: 1},
        "obligations_total": 1,
        "final_obligations": {status: 1},
        "final_obligations_total": 1,
        "final_contrary_signals": 0,
        "extract_rounds": 1,
        "fingerprint": "fixture",
        "contract_audit_status": "available",
        "final_obligation_contract": [obligation],
        "final_obligation_outcomes_ordered": [outcome],
    }


def _comparison(tmp_path, control, treatment):
    records = {stage: {"status": "complete"} for stage in runner.STAGES}
    for arm, features in (("control", control), ("treatment", treatment)):
        root = tmp_path / arm
        root.mkdir(parents=True)
        result = {"features": features}
        (root / "replay-result.json").write_text(
            json.dumps(result), encoding="utf-8")
        records[arm]["worker_result_sha256"] = content_hash(result)
    return runner._compare_features(tmp_path, records), records


def test_feature_comparison_rejects_verdict_change(tmp_path):
    control = _feature(verdict="verified")
    treatment = _feature(verdict="refuted")

    comparison, _records = _comparison(tmp_path, control, treatment)

    assert comparison["available"] is True
    assert comparison["same_verdict"] is False
    assert comparison["same_semantic_features"] is False
    assert comparison["same_rank_key"] is False


def test_feature_comparison_rejects_obligation_status_change(tmp_path):
    control = _feature(status="verified")
    treatment = _feature(status="refuted")

    comparison, _records = _comparison(tmp_path, control, treatment)

    assert comparison["available"] is True
    assert comparison["same_verdict"] is True
    assert comparison["same_ordered_final_outcomes"] is False
    assert comparison["same_semantic_features"] is False


def test_feature_comparison_ignores_only_judger_reason_wording(tmp_path):
    control = _feature(reason="the retained check is faithful")
    treatment = _feature(reason="independent evidence faithfully establishes it")
    treatment["certificate_id"] = "c" * 64

    comparison, _records = _comparison(tmp_path, control, treatment)

    assert comparison["available"] is True
    assert comparison["same_verdict"] is True
    assert comparison["same_semantic_features"] is True
    assert comparison["same_rank_key"] is True
    assert comparison["same_final_contract"] is True
    assert comparison["same_ordered_final_outcomes"] is True


def test_damaged_feature_contract_is_unavailable(tmp_path):
    control = _feature()
    treatment = _feature()
    treatment["final_obligation_outcomes_ordered"][0]["status"] = "unknown-status"

    comparison, _records = _comparison(tmp_path, control, treatment)

    assert comparison == {"available": False}


def test_feature_comparison_rejects_result_changed_after_worker_admission(tmp_path):
    comparison, records = _comparison(tmp_path, _feature(), _feature())
    assert comparison["available"] is True
    path = tmp_path / "treatment" / "replay-result.json"
    changed = json.loads(path.read_text())
    changed["features"]["verdict"] = "refuted"
    path.write_text(json.dumps(changed), encoding="utf-8")

    assert runner._compare_features(tmp_path, records) == {"available": False}


def test_json_supports_bounded_records_larger_than_64k(tmp_path):
    path = tmp_path / "large.json"
    value = {"schema": "large-fixture", "payload": "x" * 70_000}
    path.write_text(json.dumps(value), encoding="utf-8")

    assert runner._json(path) == value


def test_json_rejects_duplicate_object_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"complete","status":"inconclusive"}',
                    encoding="utf-8")

    with pytest.raises(ValueError):
        runner._json(path)


def test_json_rejects_symlinked_record(tmp_path):
    target = tmp_path / "target.json"
    target.write_text('{"status":"complete"}', encoding="utf-8")
    link = tmp_path / "record.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        runner._json(link)


def _recorded_replay(tmp_path, monkeypatch):
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
            {"type": "thread.started",
             "thread_id": f"stub-{len(process_calls)}"},
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

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", process)
    factory = _service_factory(
        lambda state: CodexBackend(
            cwd=str(state), model="gpt-5.6-luna", effort="medium",
            service_tier="standard"),
        executors,
    )
    source, identity, bundle, rounds = _prepared(tmp_path, factory)
    output = tmp_path / "diagnostic"
    stage = "control"
    result = replay_prepared_contracts.replay_prepared_contracts(
        source.resolve(), (output / stage).resolve(),
        replay_id="cold-local-control",
        setup_id="cold-local-setup",
        arm=stage,
        service_factory=factory,
        config=CONFIG,
        extracted_rounds=rounds,
        bundle_bytes=bundle,
        expected_bundle_sha256=hashlib.sha256(bundle).hexdigest(),
        expected_identity=identity,
    )
    assert result["complete"] is True
    request = {
        "output": str(output.resolve()),
        "run_id": "cold-local",
        "setup_id": "cold-local-setup",
    }
    prepared = {
        "bundle_sha256": result["bundle_sha256"],
        "identity": result["contract_identity"],
    }
    monkeypatch.setattr(
        runner, "_recorded_config", lambda _request, _stage: copy.deepcopy(CONFIG))
    monkeypatch.setattr(
        runner, "_admit_setup",
        lambda _request, **_kwargs: (copy.deepcopy(prepared), bundle))
    return request, stage, result, output


def test_arm_bill_admission_accepts_real_recorded_usage_and_context(
        tmp_path, monkeypatch):
    request, stage, result, _output = _recorded_replay(tmp_path, monkeypatch)

    runner._check_arm_bills(request, stage, result)


@pytest.mark.parametrize("attack", ["ledger", "digest", "identity", "context"])
def test_arm_bill_admission_rejects_hash_identity_or_context_drift(
        tmp_path, monkeypatch, attack):
    request, stage, result, output = _recorded_replay(tmp_path, monkeypatch)
    changed = copy.deepcopy(result)
    if attack == "ledger":
        with (output / stage / "replay-usage.jsonl").open("ab") as handle:
            handle.write(b"\n")
    elif attack == "digest":
        changed["usage_sha256"] = "0" * 64
    elif attack == "identity":
        changed["contract_identity"]["graph_sha256"] = "0" * 64
    else:
        changed["arm"] = "treatment"

    with pytest.raises(ValueError):
        runner._check_arm_bills(request, stage, changed)
