"""Adversarial receipt tests for direct contract preparation.

All ledgers are synthetic and local.  These tests exercise only the
operator-side accounting join; they never invoke a model or alter the normal
online/offline verifier schedule.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.terminal_bench import prepare_contract_entry as producer
from verifier.core.model import CLAIM_FINAL, Claim
from verifier.core.store import ClaimStore, TRIGGER_LAZY


FINAL = Claim(CLAIM_FINAL, "the delivered result satisfies the task")
OTHER_CLAIM_ID = "f" * 64


def _usage(call_number: int, role: str) -> dict:
    return {
        "schema_version": 1,
        "record_type": "codex_cli_attempt",
        "attempt_schema_version": 1,
        "call_id": f"{call_number:032x}",
        "attempt_id": f"{call_number + 100:032x}",
        "attempt_index": 1,
        "started": True,
        "attempt_status": "succeeded",
        "usage_status": "complete",
        "usage_complete": True,
        "role": role,
        "model": "gpt-5.6-luna",
        "effort": "medium",
        "service_tier": "standard",
        "prompt_sha256": f"{call_number + 500:064x}",
        "duration_seconds": 1.0,
        "input_tokens": 100,
        "cached_input_tokens": 60,
        "uncached_input_tokens": 40,
        "output_tokens": 10,
        "reasoning_tokens": None,
    }


def _spans(
    call_number: int,
    role: str,
    *,
    claim_id: str = FINAL.id,
    round_no: int = 0,
) -> list[dict]:
    call_id = f"{call_number:032x}"
    role_span_id = f"{call_number + 200:032x}"
    role_attempt_id = f"{call_number + 300:032x}"
    stage = {"obligations": "contract_compile",
             "critic": "contract_critic"}[role]
    base = {
        "span_schema_version": 1,
        "claim_id": claim_id,
        "claim_kind": CLAIM_FINAL,
        "locus": "final_task",
        "stage": stage,
        "round": round_no,
    }
    return [
        {**base, "record_type": "role_span_started",
         "role_span_id": role_span_id, "role": role,
         "monotonic_ns": 1, "unix_ns": 1},
        {**base, "record_type": "role_attempt_started",
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "monotonic_ns": 2, "unix_ns": 2},
        {**base, "record_type": "seat_acquired",
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "role": role, "queue_duration_ns": 1,
         "monotonic_ns": 3, "unix_ns": 3},
        {**base, "record_type": "model_call_started", "call_id": call_id,
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "role": role,
         "model": "gpt-5.6-luna", "attribution": "attributed",
         "monotonic_ns": 4, "unix_ns": 4},
        {**base, "record_type": "model_call_finished", "call_id": call_id,
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "role": role,
         "model": "gpt-5.6-luna", "attribution": "attributed",
         "duration_ns": 5, "status": "succeeded",
         "monotonic_ns": 9, "unix_ns": 9},
        {**base, "record_type": "seat_released",
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "role": role, "held_duration_ns": 7,
         "monotonic_ns": 10, "unix_ns": 10},
        {**base, "record_type": "role_attempt_finished",
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "duration_ns": 9,
         "status": "succeeded", "monotonic_ns": 11, "unix_ns": 11},
        {**base, "record_type": "role_span_finished",
         "role_span_id": role_span_id, "role": role, "duration_ns": 11,
         "status": "succeeded", "monotonic_ns": 12, "unix_ns": 12},
    ]


def _event(*, claim_id=FINAL.id, round_no=0, verdict="pass", seq=1):
    return {
        "seq": seq,
        "type": "decomposition_attacked",
        "payload": {
            "claim": claim_id,
            "round": round_no,
            "verdict": verdict,
            "counterexample_world": "n/a",
            "missing_condition": "n/a",
        },
    }


def _write(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path):
    usage_path = tmp_path / "usage.jsonl"
    spans_path = tmp_path / "usage.jsonl.spans.jsonl"
    usage = [_usage(1, "obligations"), _usage(2, "critic")]
    spans = [*_spans(1, "obligations"), *_spans(2, "critic")]
    events = [_event()]
    _write(usage_path, usage)
    _write(spans_path, spans)
    return usage_path, spans_path, usage, spans, events


def _audit(usage_path, spans_path, events, *, expected_context=None):
    return producer._receipt_calls(
        usage_path, spans_path, FINAL.id, CLAIM_FINAL, events,
        expected_context=expected_context)


def _fail_role_attempt(spans, role: str, *, forge_role_success: bool) -> None:
    for row in spans:
        if row.get("role") != role:
            continue
        if row["record_type"] in {"model_call_finished", "role_attempt_finished"}:
            row["status"] = "error"
        elif row["record_type"] == "role_span_finished":
            row["status"] = "succeeded" if forge_role_success else "error"


def test_exact_compiler_then_critic_receipt_is_complete(tmp_path):
    usage_path, spans_path, _usage_rows, _spans_rows, events = _fixture(
        tmp_path)

    calls, terminal, report, findings = _audit(
        usage_path, spans_path, events)

    assert findings == []
    assert report["cost_complete"] is True
    assert report["attribution_complete"] is True
    assert [call["role"] for call in calls] == ["obligations", "critic"]
    assert terminal["verdict"] == "pass"
    assert terminal["event_seq"] == 1
    assert terminal["call_ids"] == [f"{2:032x}"]


@pytest.mark.parametrize(
    "event_update, expected",
    [
        ({"claim_id": OTHER_CLAIM_ID}, "critic_receipt_scope_mismatch"),
        ({"round_no": 1}, "critic_receipt_scope_mismatch"),
        # bool is an int subclass; a forged false round must not equal round 0.
        ({"round_no": False}, "critic_receipt_scope_mismatch"),
    ],
)
def test_forged_critic_event_scope_adds_publication_blocking_finding(
        tmp_path, event_update, expected):
    usage_path, spans_path, _usage_rows, _spans_rows, _events = _fixture(
        tmp_path)
    events = [_event(**event_update)]

    _calls, _terminal, _report, findings = _audit(
        usage_path, spans_path, events)

    assert expected in findings
    # ``terminal`` is retained only as diagnostic material. The producer's
    # publication gate is the empty findings set, so this receipt is rejected.
    assert findings


def test_missing_lifecycle_and_extra_usage_attempt_are_incomplete(tmp_path):
    usage_path, spans_path, usage, spans, events = _fixture(tmp_path)
    missing_terminal = [
        row for row in spans
        if not (row["record_type"] == "role_span_finished"
                and row.get("role") == "critic")
    ]
    _write(spans_path, missing_terminal)

    _calls, terminal, report, findings = _audit(
        usage_path, spans_path, events)

    assert report["attribution_complete"] is False
    assert "incomplete_attribution" in findings

    _write(spans_path, spans)
    _write(usage_path, [*usage, _usage(3, "critic")])
    _calls, terminal, report, findings = _audit(
        usage_path, spans_path, events)
    assert report["attribution_complete"] is False
    assert "incomplete_attribution" in findings


def test_extra_role_calls_cannot_hide_behind_one_pass_event(tmp_path):
    usage_path, spans_path, usage, spans, events = _fixture(tmp_path)
    _write(usage_path, [*usage, _usage(3, "critic")])
    _write(spans_path, [*spans, *_spans(3, "critic", round_no=1)])

    _calls, terminal, _report, findings = _audit(
        usage_path, spans_path, events)

    assert "critic_receipt_count_mismatch" in findings

    # A Compiler call after the passing Critic also invalidates terminality.
    _write(usage_path, [*usage, _usage(3, "obligations")])
    _write(spans_path, [*spans, *_spans(3, "obligations", round_no=1)])
    _calls, terminal, _report, findings = _audit(
        usage_path, spans_path, events)
    assert terminal is None
    assert "no_terminal_critic_pass" in findings


def test_extra_review_event_fails_receipt_count_binding(tmp_path):
    usage_path, spans_path, _usage_rows, _spans_rows, events = _fixture(
        tmp_path)
    duplicate = copy.deepcopy(events[0])
    duplicate["seq"] = 2
    _calls, _terminal, _report, findings = _audit(
        usage_path, spans_path, [*events, duplicate])
    assert "critic_receipt_count_mismatch" in findings


def test_runtime_and_claim_span_mutations_are_fail_closed(tmp_path):
    usage_path, spans_path, usage, spans, events = _fixture(tmp_path)
    usage[0]["effort"] = "high"
    _write(usage_path, usage)

    _calls, _terminal, _report, findings = _audit(
        usage_path, spans_path, events)
    assert "runtime_mismatch" in findings

    _write(usage_path, [_usage(1, "obligations"), _usage(2, "critic")])
    for row in spans:
        if row.get("stage") == "contract_critic":
            row["claim_id"] = OTHER_CLAIM_ID
    _write(spans_path, spans)
    _calls, _terminal, _report, findings = _audit(
        usage_path, spans_path, events)
    assert "unexpected_role_scope" in findings


def test_failed_model_and_role_attempts_cannot_be_hidden_by_role_success(
        tmp_path):
    usage_path, spans_path, usage, spans, events = _fixture(tmp_path)
    for row in usage:
        row["attempt_status"] = "schema_error"
    _fail_role_attempt(spans, "obligations", forge_role_success=True)
    _fail_role_attempt(spans, "critic", forge_role_success=True)
    _write(usage_path, usage)
    _write(spans_path, spans)

    calls, _terminal, report, findings = _audit(
        usage_path, spans_path, events)

    # The span graph is structurally complete and its outer role spans claim
    # success.  The receipt must still inspect the actual model/role attempts.
    assert report["attribution_complete"] is True
    assert [call["terminal_status"] for call in calls] == [
        "succeeded", "succeeded"]
    assert findings.count("schema_result_status_mismatch") == 2
    assert findings


def test_terminal_critic_schema_error_cannot_authorize_entry(tmp_path):
    usage_path, spans_path, usage, spans, events = _fixture(tmp_path)
    usage[-1]["attempt_status"] = "schema_error"
    _fail_role_attempt(spans, "critic", forge_role_success=False)
    _write(usage_path, usage)
    _write(spans_path, spans)

    calls, _terminal, report, findings = _audit(
        usage_path, spans_path, events)

    assert report["cost_complete"] is True
    assert calls[-1]["attempts"][-1]["attempt_status"] == "schema_error"
    assert calls[-1]["terminal_status"] == "error"
    assert "schema_result_status_mismatch" in findings
    assert "role_did_not_succeed" in findings
    assert "critic_receipt_scope_mismatch" in findings


def test_all_preceding_provider_retries_remain_in_receipt_cost(tmp_path):
    usage_path, spans_path, usage, spans, events = _fixture(tmp_path)
    compiler_success = usage[0]
    retries = []
    for index in (1, 2):
        retry = copy.deepcopy(compiler_success)
        retry["attempt_id"] = f"{800 + index:032x}"
        retry["attempt_index"] = index
        retry["attempt_status"] = "provider_retry"
        retries.append(retry)
    compiler_success["attempt_index"] = 3
    _write(usage_path, [*retries, *usage])

    calls, terminal, report, findings = _audit(
        usage_path, spans_path, events)

    assert findings == []
    assert terminal["verdict"] == "pass"
    assert report["cost_complete"] is True
    assert report["attempt_count"] == 4
    assert report["totals"]["attempt_statuses"] == {
        "provider_retry": 2, "succeeded": 2}
    assert report["totals"]["tokens"]["input_tokens"]["known_sum"] == 400
    assert [row["attempt_status"] for row in calls[0]["attempts"]] == [
        "provider_retry", "provider_retry", "succeeded"]


@pytest.mark.parametrize("changed_field", ["setup_id", "state_dir"])
def test_usage_context_must_match_operator_setup_and_state(
        tmp_path, changed_field):
    usage_path, spans_path, usage, _spans, events = _fixture(tmp_path)
    expected = {
        "setup_id": "setup-a",
        "state_dir": "/operator-owned/sealed-state",
        "backend": "codex-cli",
    }
    for row in usage:
        row.update(expected)
    _write(usage_path, usage)
    assert _audit(
        usage_path, spans_path, events, expected_context=expected)[-1] == []

    usage[-1][changed_field] = "different-operator-context"
    _write(usage_path, usage)
    _calls, _terminal, _report, findings = _audit(
        usage_path, spans_path, events, expected_context=expected)
    assert "usage_context_mismatch" in findings


def test_boolean_repair_attempt_is_not_accepted_as_integer_zero(tmp_path):
    usage_path, spans_path, _usage_rows, _spans_rows, events = _fixture(
        tmp_path)
    events[0]["payload"]["repair_attempt"] = False

    _calls, _terminal, _report, findings = _audit(
        usage_path, spans_path, events)

    assert "invalid_event_round" in findings
    assert findings


def test_graph_mutation_breaks_the_fresh_snapshot_check():
    store = ClaimStore(trigger=TRIGGER_LAZY)
    store.add_claim(FINAL)
    snapshot = {"claims": [{
        "id": FINAL.id,
        "kind": FINAL.kind,
        "text": FINAL.text,
        "deps": [],
        "revises": None,
        "initial_status": "unverified",
    }]}
    kernel = SimpleNamespace(store=store)

    assert producer._unchanged_graph(kernel, snapshot) is True
    store.set_verified(FINAL.id, basis="world")
    assert producer._unchanged_graph(kernel, snapshot) is False
