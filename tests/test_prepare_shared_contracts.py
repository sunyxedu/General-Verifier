"""Real extraction/Compiler/Critic composition, using local protocol stubs."""
import json

import pytest

from experiments.terminal_bench import prepare_shared_contracts as setup
from experiments.terminal_bench.run_typed_adapter_eval import _source_fingerprint
from test_contract_replay import COMMITS
from test_prepare_contract_entry import harness, PAYLOAD, PASS
from verifier import offline
from verifier.engine.contract.bundle import ContractBundleIdentity
from verifier.contract_replay import load_replay_contracts, clone_extracted_inputs


def prepared(tmp_path, monkeypatch, *, responses=None, usage=None):
    h = harness(tmp_path, monkeypatch,
                responses=responses if responses is not None else [{"commits": COMMITS}, PAYLOAD, PASS, PAYLOAD, PASS],
                usage=usage)
    config = offline.config_fields(model="gpt-5.6-luna", seats="codex", effort="medium", service_tier="standard")
    result = setup.prepare_shared_contracts(
        h.source, tmp_path / "shared", setup_id="cold-setup-test", config=config,
        expected_source_fingerprint=_source_fingerprint(setup.SOURCE_ROOT),
        backend_factory=lambda state: setup.codex_backend(state, config),
        service_factory=lambda state: setup.codex_service(state, config))
    return h, result, config


def test_complete_shared_setup_charges_real_extract_and_every_contract(tmp_path, monkeypatch):
    h, result, config = prepared(tmp_path, monkeypatch)
    assert result["complete"], result["findings"]
    assert len(h.calls) == 5
    work = result["accounting"]["model_work"]
    assert work["cost_complete"] and work["attribution_complete"]
    assert work["totals"]["started_attempt_count"] == 5
    assert work["totals"]["tokens"]["input_tokens"]["known_sum"] == 500
    assert result["metrics"]["started_attempts"] == 5
    assert result["metrics"]["wall_seconds"] > 0
    state = json.loads((tmp_path / "shared" / "setup-state.json").read_text())
    plan = json.loads((tmp_path / "shared" / "contract-plan.json").read_text())
    assert state["status"] == "complete"
    assert len(state["claims"]) == 2
    assert all(item["status"] == "finished" for item in state["claims"])
    assert all(item["status"] == "not_started" for item in plan["preparations"])
    assert state["wall_seconds"] == (state["finished_at_monotonic_ns"] - state["started_at_monotonic_ns"]) / 1e9
    assert (tmp_path / "shared" / "bundle-pin.json").exists()
    # The exported full bundle is admitted on a fresh source, not the
    # preparation's mutated compiler ledger or ClaimStore.
    arm = tmp_path / "fresh-arm"
    clone_extracted_inputs(result["extracted_source"], arm)
    sc = setup.codex_service(arm, config)
    provider = load_replay_contracts(sc.kernel, sc.builder.finals[-1], config,
        (tmp_path / "shared" / "contracts.json").read_bytes(), expected_sha256=result["bundle_sha256"])
    assert provider.identity == ContractBundleIdentity.from_value(result["identity"])
    assert len(h.calls) == 5


def test_compile_failure_stops_new_spend_without_erasing_failed_attempts(tmp_path, monkeypatch):
    h, result, _ = prepared(tmp_path, monkeypatch,
                           responses=[{"commits": COMMITS}, {"wrong": 1}, {"wrong": 2}])
    assert not result["complete"] and result["bundle_sha256"] is None
    assert len(h.calls) == 3
    work = result["accounting"]["model_work"]
    assert work["cost_complete"]
    assert work["totals"]["started_attempt_count"] == 3
    state = json.loads((tmp_path / "shared" / "setup-state.json").read_text())
    assert [item["status"] for item in state["claims"]] == ["failed", "not_started"]
    assert not (tmp_path / "shared" / "contracts.json").exists()


def test_failed_extraction_keeps_every_schema_and_structural_retry(tmp_path, monkeypatch):
    h, result, config = prepared(tmp_path, monkeypatch, responses=[{"wrong": 1}] * 6)
    assert not result["complete"]
    assert len(h.calls) == config["max_rounds"] * 2
    assert result["accounting"]["model_work"]["totals"]["started_attempt_count"] == len(h.calls)
    state = json.loads((tmp_path / "shared" / "setup-state.json").read_text())
    assert state["extraction"]["status"] == "failed" and state["claims"] == []


def test_unknown_provider_usage_is_not_charged_as_zero(tmp_path, monkeypatch):
    h, result, _ = prepared(tmp_path, monkeypatch, usage={})
    assert not result["complete"]
    assert len(h.calls) == 3  # extract + first claim; no resampling or final compile.
    work = result["accounting"]["model_work"]
    assert not work["cost_complete"]
    assert work["totals"]["tokens"]["input_tokens"]["unknown_started_attempts"] == 3
    assert result["metrics"]["uncached_input_tokens"] is None


def test_new_setup_root_is_never_reused_or_overwritten(tmp_path, monkeypatch):
    h, result, config = prepared(tmp_path, monkeypatch)
    before = (tmp_path / "shared" / "setup-result.json").read_bytes()
    with pytest.raises(ValueError, match="new"):
        setup.prepare_shared_contracts(h.source, tmp_path / "shared", setup_id="again", config=config,
            expected_source_fingerprint=_source_fingerprint(setup.SOURCE_ROOT),
            backend_factory=lambda state: setup.codex_backend(state, config),
            service_factory=lambda state: setup.codex_service(state, config))
    assert (tmp_path / "shared" / "setup-result.json").read_bytes() == before
    assert len(h.calls) == 5


@pytest.mark.parametrize("stage", ["accounting", "reconsume", "source_fingerprint", "bundle_write", "pin_write"])
def test_finalization_failures_never_leave_a_successful_admission(tmp_path, monkeypatch, stage):
    if stage == "accounting":
        def broken(*args, **kwargs):
            raise OSError("local accounting failure")
        monkeypatch.setattr("experiments.terminal_bench.merge_setup_accounting.merge_setup_accounting", broken)
    elif stage == "reconsume":
        original = setup.assemble_prepared_contracts
        count = 0
        def broken(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("local reconsumption failure")
            return original(*args, **kwargs)
        monkeypatch.setattr(setup, "assemble_prepared_contracts", broken)
    elif stage == "source_fingerprint":
        original = setup._source_fingerprint
        count = 0
        def broken(*args, **kwargs):
            nonlocal count
            count += 1
            if count > 1:
                raise OSError("local fingerprint failure")
            return original(*args, **kwargs)
        monkeypatch.setattr(setup, "_source_fingerprint", broken)
    elif stage == "pin_write":
        original = setup.offline._write_json_atomic
        def broken(path, value):
            if path.endswith("/bundle-pin.json"):
                raise OSError("local pin failure")
            return original(path, value)
        monkeypatch.setattr(setup.offline, "_write_json_atomic", broken)
    else:
        original = setup._new_bytes
        def broken(path, raw):
            if path.name == "contracts.json":
                raise OSError("local export failure")
            return original(path, raw)
        monkeypatch.setattr(setup, "_new_bytes", broken)
    h, result, _ = prepared(tmp_path, monkeypatch)
    assert len(h.calls) == 5
    assert result["complete"] is False
    assert result["bundle_path"] is result["bundle_sha256"] is None
    state = json.loads((tmp_path / "shared" / "setup-state.json").read_text())
    assert state["status"] == "inconclusive"
    assert state["findings"]
    assert not (tmp_path / "shared" / "bundle-pin.json").exists()
    # Failed finalization never deletes the actual provider bills.
    ledgers = [tmp_path / "shared" / "extract-usage.jsonl"] + list(
        (tmp_path / "shared" / "claims").glob("*/prepare-usage.jsonl"))
    assert sum(len(path.read_text().splitlines()) for path in ledgers) == 5


@pytest.mark.parametrize("terminal_name", ["setup-state.json", "setup-result.json"])
def test_terminal_write_failure_cannot_publish_admission_pin(tmp_path, monkeypatch, terminal_name):
    original = setup.offline._write_json_atomic
    def broken(path, value):
        terminal = value.get("status") == "complete" or value.get("complete") is True
        if path.endswith("/" + terminal_name) and terminal:
            raise OSError("terminal record unavailable")
        return original(path, value)
    monkeypatch.setattr(setup.offline, "_write_json_atomic", broken)
    with pytest.raises(OSError, match="terminal record"):
        prepared(tmp_path, monkeypatch)
    assert not (tmp_path / "shared" / "bundle-pin.json").exists()
    ledgers = [tmp_path / "shared" / "extract-usage.jsonl"] + list(
        (tmp_path / "shared" / "claims").glob("*/prepare-usage.jsonl"))
    assert sum(len(path.read_text().splitlines()) for path in ledgers) == 5
