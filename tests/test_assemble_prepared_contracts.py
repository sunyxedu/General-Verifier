"""All-claims admission is separate from individual receipt validation."""
import copy

import pytest

from experiments.terminal_bench.validate_prepared_contract_entry import assemble_prepared_contracts
from test_prepare_contract_entry import harness, PAYLOAD, PASS
from test_contract_replay import snapshot, service
from verifier.contract_replay import (
    attach_replay_contracts, clone_extracted_inputs, compile_input_digests, load_replay_contracts,
)
from verifier.engine.contract_bundle import SealedContractBundle
from verifier.llm.codex_backend import CodexRuntime


def group(tmp_path, monkeypatch):
    local = harness(tmp_path, monkeypatch, [PAYLOAD, PASS, PAYLOAD, PASS])
    prepared = [local.prepare("support", "support"), local.prepare("final", "final")]
    graph = snapshot(local.services[0])
    args = {
        "expected_setup_id": "local-test", "expected_identity": local.identity,
        "expected_claims": [{key: row[key] for key in ("id", "kind", "text", "deps", "revises")}
                            for row in graph["claims"]],
        "expected_compile_inputs": compile_input_digests(local.services[0].kernel, graph),
        "preparations": [{"root": tmp_path / name, "claim_id": item["receipt"]["claim_id"],
                          "receipt_sha256": item["receipt_sha256"], "candidate_entry": item["entry"]}
                         for name, item in zip(("support", "final"), prepared)],
    }
    return local, args


def test_all_claims_pass_a_full_graph_bundle_without_new_model_calls(tmp_path, monkeypatch):
    local, args = group(tmp_path, monkeypatch)
    result = assemble_prepared_contracts(**args)
    assert result["complete"] is True, result["findings"]
    assert len(local.calls) == 4
    assert len(result["preparations"]) == 2
    assert all(item["model_work"]["totals"]["started_attempt_count"] == 2
               for item in result["preparations"])
    bundle = SealedContractBundle.from_bytes(
        result["bundle_bytes"], expected_sha256=result["bundle_sha256"],
        expected_identity=local.identity, expected_compile_inputs=args["expected_compile_inputs"])
    bundle.require_replayable()
    assert len(bundle.claims) == len(args["expected_claims"])
    assert result["accounting_scope"] == "direct_claims_only_excludes_extraction_and_replay"


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "bad_candidate", "bad_receipt", "wrong_input_map", "shortened_graph", "relative_root", "duplicate_root"])
def test_no_successful_subset_can_replace_the_full_selected_graph(tmp_path, monkeypatch, change):
    local, args = group(tmp_path, monkeypatch)
    if change == "missing":
        args["preparations"].pop()
    elif change == "duplicate":
        args["preparations"][1] = copy.deepcopy(args["preparations"][0])
    elif change == "extra":
        args["preparations"].append(copy.deepcopy(args["preparations"][0]))
    elif change == "bad_candidate":
        args["preparations"][0]["candidate_entry"]["payload"]["obligations"][0]["text"] = "Changed"
    elif change == "bad_receipt":
        args["preparations"][0]["receipt_sha256"] = "f" * 64
    elif change == "wrong_input_map":
        args["expected_compile_inputs"].pop(args["preparations"][0]["claim_id"])
    elif change == "relative_root":
        args["preparations"][0]["root"] = "support"
    elif change == "duplicate_root":
        args["preparations"][1]["root"] = args["preparations"][0]["root"]
    else:
        args["expected_claims"].pop(0)  # Off-slice context also stays bound.
    result = assemble_prepared_contracts(**args)
    assert result["complete"] is False
    assert result["bundle_bytes"] is None and result["bundle_sha256"] is None
    assert result["findings"]
    assert len(local.calls) == 4
    if change in {"bad_candidate", "bad_receipt"}:
        assert len(result["preparations"]) == 2  # Do not stop reading bills on first failure.
        assert all(item["model_work"]["totals"]["started_attempt_count"] == 2
                   for item in result["preparations"])
    elif change == "missing":
        assert len(result["preparations"]) == 1


def test_declaration_order_does_not_change_bundle_order(tmp_path, monkeypatch):
    local, args = group(tmp_path, monkeypatch)
    first = assemble_prepared_contracts(**args)
    args["preparations"].reverse()
    second = assemble_prepared_contracts(**args)
    assert first["complete"] is second["complete"] is True
    assert first["bundle_bytes"] == second["bundle_bytes"]
    assert len(local.calls) == 4


def test_validated_full_bundle_loads_into_two_fresh_diagnostic_arms(tmp_path, monkeypatch):
    local, args = group(tmp_path, monkeypatch)
    result = assemble_prepared_contracts(**args)
    assert result["complete"]
    config = {"model": "gpt-5.6-luna", "effort": "medium", "service_tier": "standard"}
    arms = []
    for name in ("control", "treatment"):
        state = tmp_path / name
        clone_extracted_inputs(local.source, state)
        sc = service(state, CodexRuntime(cwd=str(state), model="gpt-5.6-luna"))
        provider = load_replay_contracts(
            sc.kernel, sc.builder.finals[-1], config, result["bundle_bytes"],
            expected_sha256=result["bundle_sha256"])
        attach_replay_contracts(sc.kernel, provider)
        materialized = {cid: sc.kernel._compile_obligations(sc.store.get(cid))
                        for cid in args["expected_compile_inputs"]}
        provider.validate_inputs()
        arms.append((sc, materialized))
    assert len(local.calls) == 4  # Shared contracts cause no extra compilation.
    assert arms[0][1] == arms[1][1]
    assert arms[0][0].kernel is not arms[1][0].kernel
    assert arms[0][0].store is not arms[1][0].store
    for sc, _ in arms:
        assert sc.store.db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0
    # This tests admission/materialization only, not completed evidence work
    # or measured A/B latency. Each arm must still run Oracle/executor/Judger.
