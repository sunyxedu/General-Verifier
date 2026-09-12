"""Adversarial tests for the read-only prepared-contract consumer.

The producer harness uses a local Codex protocol stub.  No model or network
service is invoked here.
"""
from __future__ import annotations

import copy
import json
import sqlite3

import pytest

import experiments.terminal_bench.validate_prepared_contract_entry as validator
from experiments.terminal_bench.validate_prepared_contract_entry import (
    validate_prepared_contract_entry,
)
from test_prepare_contract_entry import PASS, PAYLOAD, harness


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prepared(tmp_path, monkeypatch, *, kind="support", responses=None,
              usage=None):
    local = harness(tmp_path, monkeypatch, responses=responses, usage=usage)
    produced = local.prepare(kind)
    receipt = produced["receipt"]
    return local, produced, {
        "root": tmp_path / "prepare",
        "expected_setup_id": receipt["setup_id"],
        "expected_claim_id": receipt["claim_id"],
        "expected_identity": local.identity,
        "expected_compile_input_sha256": receipt["compile_input_sha256"],
        "expected_receipt_sha256": produced["receipt_sha256"],
        "candidate_entry": produced["entry"],
    }


def _assert_rejected_but_cost_retained(result, *, attempts=2, input_tokens=200):
    assert result["complete"] is False
    assert result["entry"] is None
    assert result["findings"]
    work = result["model_work"]
    assert work is not None
    assert work["cost_complete"] is True
    assert work["totals"]["started_attempt_count"] == attempts
    assert work["totals"]["tokens"]["input_tokens"]["known_sum"] == input_tokens


@pytest.mark.parametrize("kind", ["support", "final"])
def test_valid_support_and_final_are_accepted_read_only(
        tmp_path, monkeypatch, kind):
    local, produced, args = _prepared(tmp_path, monkeypatch, kind=kind)
    root = args["root"]
    before = {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }
    calls_before = copy.deepcopy(local.calls)

    result = validate_prepared_contract_entry(**args)

    after = {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }
    assert result["complete"] is True
    assert result["findings"] == []
    assert result["entry"] == produced["entry"]
    assert result["model_work"]["cost_complete"] is True
    assert result["model_work"]["attribution_complete"] is True
    assert before == after
    assert local.calls == calls_before


def test_schema_retry_then_success_is_accepted_and_all_attempts_are_charged(
        tmp_path, monkeypatch):
    local, produced, args = _prepared(
        tmp_path, monkeypatch,
        responses=[PAYLOAD, {"bad_schema": True}, PASS],
    )

    result = validate_prepared_contract_entry(**args)

    assert result["complete"] is True
    assert result["entry"] == produced["entry"]
    assert result["model_work"]["attempt_count"] == 3
    assert result["model_work"]["totals"]["attempt_statuses"] == {
        "schema_error": 1, "succeeded": 2}
    assert len(local.calls) == 3


def test_post_parse_caller_mutation_cannot_change_validated_snapshot(
        tmp_path, monkeypatch):
    _local, produced, args = _prepared(tmp_path, monkeypatch)
    caller_candidate = args["candidate_entry"]
    expected_snapshot = copy.deepcopy(produced["entry"])
    original_parse = validator._parse_entries

    def parse_then_mutate(*parse_args, **parse_kwargs):
        parsed = original_parse(*parse_args, **parse_kwargs)
        caller_candidate["obligations"][0]["text"] = "post-parse mutation"
        return parsed

    monkeypatch.setattr(validator, "_parse_entries", parse_then_mutate)

    result = validator.validate_prepared_contract_entry(**args)

    assert result["complete"] is True
    assert caller_candidate != expected_snapshot
    assert result["entry"] == expected_snapshot
    assert result["entry"] is not caller_candidate


def test_candidate_dict_subclass_is_rejected_but_cost_is_retained(
        tmp_path, monkeypatch):
    class CandidateDict(dict):
        pass

    _local, produced, args = _prepared(tmp_path, monkeypatch)
    args["candidate_entry"] = CandidateDict(produced["entry"])

    result = validate_prepared_contract_entry(**args)

    _assert_rejected_but_cost_retained(result)


def test_expected_identity_plain_mapping_is_accepted(tmp_path, monkeypatch):
    local, produced, args = _prepared(tmp_path, monkeypatch)
    args["expected_identity"] = local.identity.as_dict()

    result = validate_prepared_contract_entry(**args)

    assert result["complete"] is True
    assert result["findings"] == []
    assert result["entry"] == produced["entry"]


def test_unknown_usage_remains_unknown_after_independent_validation(
        tmp_path, monkeypatch):
    _local, _produced, args = _prepared(tmp_path, monkeypatch, usage={})

    result = validate_prepared_contract_entry(**args)

    assert result["complete"] is False
    assert result["entry"] is None
    work = result["model_work"]
    assert work is not None
    assert work["cost_complete"] is False
    assert work["totals"]["started_attempt_count"] == 2
    assert work["totals"]["tokens"]["input_tokens"] == {
        "known_sum": 0, "unknown_started_attempts": 2}


@pytest.mark.parametrize("target", ["source", "store"])
def test_late_source_or_store_tamper_is_rejected_but_cost_is_retained(
        tmp_path, monkeypatch, target):
    _local, _produced, args = _prepared(tmp_path, monkeypatch)
    state = args["root"] / "state"
    if target == "source":
        path = state / "task.md"
        path.chmod(0o600)
        path.write_text("late replacement task", encoding="utf-8")
    else:
        connection = sqlite3.connect(state / "claims.sqlite")
        try:
            connection.execute(
                "UPDATE claims SET text='late graph mutation' WHERE kind='assertion'")
            connection.commit()
        finally:
            connection.close()

    result = validate_prepared_contract_entry(**args)

    _assert_rejected_but_cost_retained(result)


def test_duplicate_json_key_is_rejected_but_cost_is_still_recomputed(
        tmp_path, monkeypatch):
    _local, _produced, args = _prepared(tmp_path, monkeypatch)
    path = args["root"] / "prepare-receipt.json"
    raw = path.read_text(encoding="utf-8").strip()
    assert raw.endswith("}")
    path.chmod(0o600)
    path.write_text(raw[:-1] + ',"status":"passed"}\n', encoding="utf-8")

    result = validate_prepared_contract_entry(**args)

    _assert_rejected_but_cost_retained(result)


def test_receipt_content_tamper_fails_but_retains_observed_cost(
        tmp_path, monkeypatch):
    _local, _produced, args = _prepared(tmp_path, monkeypatch)
    path = args["root"] / "prepare-receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["accounting_scope"] = "tampered"
    path.chmod(0o600)
    path.write_text(json.dumps(receipt), encoding="utf-8")

    result = validate_prepared_contract_entry(**args)

    _assert_rejected_but_cost_retained(result)


def test_candidate_content_tamper_fails_but_retains_observed_cost(
        tmp_path, monkeypatch):
    _local, produced, args = _prepared(tmp_path, monkeypatch)
    candidate = copy.deepcopy(produced["entry"])
    candidate["obligations"][0]["text"] = "tampered obligation"
    args["candidate_entry"] = candidate

    result = validate_prepared_contract_entry(**args)

    _assert_rejected_but_cost_retained(result)


def test_independent_receipt_hash_pin_cannot_be_substituted(
        tmp_path, monkeypatch):
    _local, produced, args = _prepared(tmp_path, monkeypatch)
    replacement = "f" * 64
    assert replacement != produced["receipt_sha256"]
    args["expected_receipt_sha256"] = replacement

    result = validate_prepared_contract_entry(**args)

    _assert_rejected_but_cost_retained(result)


def test_extra_usage_attempt_is_rejected_and_included_in_observed_cost(
        tmp_path, monkeypatch):
    _local, _produced, args = _prepared(tmp_path, monkeypatch)
    path = args["root"] / "prepare-usage.jsonl"
    rows = _jsonl(path)
    extra = copy.deepcopy(rows[0])
    extra.update({
        "call_id": "f" * 32,
        "attempt_id": "e" * 32,
        "attempt_index": 1,
    })
    _write_jsonl(path, [*rows, extra])

    result = validate_prepared_contract_entry(**args)

    _assert_rejected_but_cost_retained(
        result, attempts=3, input_tokens=300)
    assert result["model_work"]["attribution_complete"] is False


def test_missing_model_span_terminal_is_rejected_without_erasing_cost(
        tmp_path, monkeypatch):
    _local, _produced, args = _prepared(tmp_path, monkeypatch)
    path = args["root"] / "prepare-usage.jsonl.spans.jsonl"
    rows = _jsonl(path)
    victim = next(
        row for row in reversed(rows)
        if row.get("record_type") == "model_call_finished")
    rows.remove(victim)
    _write_jsonl(path, rows)

    result = validate_prepared_contract_entry(**args)

    _assert_rejected_but_cost_retained(result)
    assert result["model_work"]["attribution_complete"] is False


def test_critic_event_tamper_is_rejected_without_erasing_cost(
        tmp_path, monkeypatch):
    _local, _produced, args = _prepared(tmp_path, monkeypatch)
    path = args["root"] / "state" / "events.jsonl"
    rows = _jsonl(path)
    event = next(row for row in rows
                 if row.get("type") == "decomposition_attacked")
    event["payload"]["verdict"] = "insufficient"
    _write_jsonl(path, rows)

    result = validate_prepared_contract_entry(**args)

    _assert_rejected_but_cost_retained(result)


@pytest.mark.parametrize("field", ["setup_id", "state_dir"])
def test_usage_operator_context_tamper_is_rejected_without_erasing_cost(
        tmp_path, monkeypatch, field):
    _local, _produced, args = _prepared(tmp_path, monkeypatch)
    path = args["root"] / "prepare-usage.jsonl"
    rows = _jsonl(path)
    rows[0][field] = "different-operator-context"
    _write_jsonl(path, rows)

    result = validate_prepared_contract_entry(**args)

    _assert_rejected_but_cost_retained(result)
