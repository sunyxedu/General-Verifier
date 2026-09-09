"""Compose setup and both complete evidence arms; no paid model calls.

Synthetic protocol counts validate conservation of billing, not speed/cost
improvement. The local deterministic executor is explicitly a test fixture.
"""
import json
from pathlib import Path

from experiments.terminal_bench.prepare_shared_contracts import (
    SOURCE_ROOT, codex_backend, prepare_shared_contracts,
)
from experiments.terminal_bench.replay_prepared_contracts import replay_prepared_contracts
from experiments.terminal_bench.run_typed_adapter_eval import _source_fingerprint
from experiments.terminal_bench.shared_setup import charge_pair
from test_contract_replay import COMMITS
from test_prepare_contract_entry import harness, PAYLOAD, PASS
from test_replay_prepared_contracts import ORACLE, JUDGER, _service_factory
from verifier.llm.oracle_execution import raise_oracle_execution_payload
from verifier import offline


def test_complete_cold_setup_and_two_independent_evidence_arms_conserve_cost(tmp_path, monkeypatch):
    responses = [{"commits": COMMITS}, PAYLOAD, PASS, PAYLOAD, PASS] + [raise_oracle_execution_payload(ORACLE), JUDGER] * 4
    h = harness(tmp_path, monkeypatch, responses=responses)
    config = offline.config_fields(model="gpt-5.6-luna", seats="codex", effort="medium", service_tier="standard")
    executors = []
    factory = _service_factory(lambda state: codex_backend(state, config), executors)
    setup = prepare_shared_contracts(
        h.source, tmp_path / "setup", setup_id="cold-flow", config=config,
        expected_source_fingerprint=_source_fingerprint(SOURCE_ROOT),
        backend_factory=lambda state: codex_backend(state, config), service_factory=factory)
    assert setup["complete"], setup["findings"]
    assert setup["test_only"] is True
    assert len(h.calls) == 5
    pin = json.loads((tmp_path / "setup" / "bundle-pin.json").read_text())
    assert pin["bundle_sha256"] == setup["bundle_sha256"]
    raw = Path(setup["bundle_path"]).read_bytes()
    results, attempt_ids = [], []
    for arm in ("control", "treatment"):
        result = replay_prepared_contracts(
            setup["extracted_source"], tmp_path / arm, replay_id=arm,
            setup_id="cold-flow", arm=arm, service_factory=factory, config=config,
            extracted_rounds=setup["extracted_rounds"], bundle_bytes=raw,
            expected_bundle_sha256=pin["bundle_sha256"], expected_identity=setup["identity"])
        assert result["complete"], result["findings"]
        assert result["features"]["verdict"] == "verified"
        assert result["features"]["contract_audit_status"] == "available"
        results.append(result)
        rows = [json.loads(line) for line in (tmp_path / arm / "replay-usage.jsonl").read_text().splitlines()]
        assert {row["role"] for row in rows} == {"oracles", "judger"}
        assert len(rows) == 4
        attempt_ids.append({row["attempt_id"] for row in rows})
        assert not (tmp_path / arm / "state" / "score.json").exists()
    assert len(h.calls) == 13
    assert attempt_ids[0].isdisjoint(attempt_ids[1])
    assert [len(executor.calls) for executor in executors if executor.calls] == [2, 2]
    charges = charge_pair(setup["metrics"], results[0]["cost_report"]["metrics"], results[1]["cost_report"]["metrics"])
    assert charges["complete"]
    assert charges["physical"]["started_attempts"] == 13
    assert [charges["charged"][arm]["started_attempts"] for arm in ("control", "treatment")] == [9, 9]
    assert charges["physical"]["uncached_input_tokens"] == 13 * 70
    for index, arm in enumerate(("control", "treatment")):
        assert charges["charged"][arm]["wall_seconds"] == setup["metrics"]["wall_seconds"] + results[index]["arm_wall_seconds"]
    # Synthetic verified/verified establishes scheduling/charge plumbing,
    # not zero-loss performance on Terminal-Bench or an actual treatment gain.
