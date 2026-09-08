import json
from pathlib import Path

from experiments.terminal_bench import analyze_model_spans as analysis


def _write(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _usage(call_id="1" * 32, attempt_index=1, **updates):
    row = {
        "schema_version": 1,
        "record_type": "codex_cli_attempt",
        "attempt_schema_version": 1,
        "call_id": call_id,
        "attempt_id": f"{attempt_index:032x}",
        "attempt_index": attempt_index,
        "started": True,
        "attempt_status": "succeeded",
        "usage_status": "complete",
        "usage_complete": True,
        "role": "judger",
        "model": "gpt-5.6-luna",
        "duration_seconds": 2.5,
        "input_tokens": 100,
        "cached_input_tokens": 70,
        "uncached_input_tokens": 30,
        "output_tokens": 11,
        "reasoning_tokens": 4,
    }
    row.update(updates)
    return row


def _spans(call_id="1" * 32, **metadata):
    meta = {
        "claim_id": "a" * 64,
        "claim_kind": "final",
        "locus": "final_task",
        "obligation_id": "b" * 64,
        "stage": "judger",
        "round": 0,
        "method_attempt": 2,
        "redo": 1,
        "judger_retry_index": 1,
        "independent_judger": False,
        **metadata,
    }
    role_span_id = "2" * 32
    role_attempt_id = "3" * 32
    base = {"span_schema_version": 1, **meta}
    return [
        {**base, "record_type": "role_span_started",
         "role_span_id": role_span_id, "role": "judger",
         "monotonic_ns": 1, "unix_ns": 1},
        {**base, "record_type": "role_attempt_started",
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "monotonic_ns": 2, "unix_ns": 2},
        {**base, "record_type": "seat_acquired",
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "role": "judger", "queue_duration_ns": 3,
         "monotonic_ns": 3, "unix_ns": 3},
        {**base, "record_type": "model_call_started", "call_id": call_id,
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "role": "judger", "model": "gpt-5.6-luna",
         "attribution": "attributed", "monotonic_ns": 4, "unix_ns": 4},
        {**base, "record_type": "model_call_finished", "call_id": call_id,
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "role": "judger", "model": "gpt-5.6-luna",
         "attribution": "attributed", "duration_ns": 5, "status": "succeeded",
         "monotonic_ns": 9, "unix_ns": 9},
        {**base, "record_type": "seat_released",
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "role": "judger", "held_duration_ns": 7,
         "monotonic_ns": 10, "unix_ns": 10},
        {**base, "record_type": "role_attempt_finished",
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "duration_ns": 9, "status": "succeeded",
         "monotonic_ns": 11, "unix_ns": 11},
        {**base, "record_type": "role_span_finished",
         "role_span_id": role_span_id, "role": "judger",
         "duration_ns": 11, "status": "succeeded",
         "monotonic_ns": 12, "unix_ns": 12},
    ]


def test_complete_join_groups_exact_attempt_cost(tmp_path):
    usage = tmp_path / "usage.jsonl"
    spans = Path(str(usage) + ".spans.jsonl")
    _write(usage, [_usage()])
    _write(spans, _spans())

    report = analysis.analyze(usage)

    assert report["cost_complete"] is True
    assert report["attribution_complete"] is True
    assert report["call_count"] == report["attributed_call_count"] == 1
    group, = report["groups"]
    assert group["attribution"] == "attributed"
    assert group["claim_id"] == "a" * 64
    assert group["locus"] == "final_task"
    assert group["obligation_id"] == "b" * 64
    assert group["stage"] == "judger"
    assert group["redo"] == 1
    assert group["call_count"] == group["attempt_count"] == 1
    assert group["tokens"]["input_tokens"] == {
        "known_sum": 100, "unknown_started_attempts": 0}
    assert group["attempt_duration_seconds"]["known_sum"] == 2.5


def test_provider_attempts_are_not_folded_and_unknown_cost_stays_unknown(
        tmp_path):
    usage = tmp_path / "usage.jsonl"
    spans = Path(str(usage) + ".spans.jsonl")
    retry = _usage(
        attempt_index=1, attempt_status="provider_retry",
        usage_status="unknown", usage_complete=False,
        duration_seconds=1.0,
        input_tokens=None, cached_input_tokens=None,
        uncached_input_tokens=None, output_tokens=None, reasoning_tokens=None,
    )
    success = _usage(attempt_index=2)
    _write(usage, [retry, success])
    _write(spans, _spans())

    report = analysis.analyze(usage)

    assert report["cost_complete"] is False
    assert report["attribution_complete"] is True
    group, = report["groups"]
    assert group["call_count"] == 1
    assert group["attempt_count"] == group["started_attempt_count"] == 2
    assert group["attempt_statuses"] == {
        "provider_retry": 1, "succeeded": 1}
    assert group["tokens"]["input_tokens"] == {
        "known_sum": 100, "unknown_started_attempts": 1}
    assert "incomplete_usage:1" in report["cost_findings"]


def test_new_schema_reasoning_unknown_does_not_invent_billable_cost_gap(
        tmp_path):
    usage = tmp_path / "usage.jsonl"
    _write(usage, [_usage(reasoning_tokens=None)])
    _write(Path(str(usage) + ".spans.jsonl"), _spans())

    report = analysis.analyze(usage)

    assert report["cost_complete"] is True
    assert report["attribution_complete"] is True
    assert report["groups"][0]["tokens"]["reasoning_tokens"] == {
        "known_sum": 0, "unknown_started_attempts": 1}


def test_legacy_cost_remains_complete_without_span_sidecar(tmp_path):
    usage = tmp_path / "legacy.jsonl"
    legacy = {
        "call_id": "9" * 32, "role": "discharge",
        "model": "gpt-5.6-luna",
        "input_tokens": 40, "cached_input_tokens": 10,
        "uncached_input_tokens": 30, "output_tokens": 5,
        "duration_seconds": 1.25,
    }
    _write(usage, [legacy])

    report = analysis.analyze(usage)

    assert report["cost_complete"] is True
    assert report["attribution_complete"] is False
    assert report["attributed_call_count"] == 0
    assert report["unattributed_call_count"] == 1
    assert report["attribution_findings"] == ["span_sidecar_missing"]
    group, = report["groups"]
    assert group["attribution"] == "unattributed"
    assert group["role"] == "discharge"
    assert group["tokens"]["reasoning_tokens"] == {
        "known_sum": 0, "unknown_started_attempts": 1}


def test_missing_duplicate_or_mismatched_terminal_spans_only_break_attribution(
        tmp_path):
    usage = tmp_path / "usage.jsonl"
    spans_path = Path(str(usage) + ".spans.jsonl")
    _write(usage, [_usage()])
    rows = _spans()
    rows.append(dict(next(
        row for row in rows if row["record_type"] == "model_call_started")))
    rows[:] = [row for row in rows
               if row["record_type"] != "model_call_finished"]
    _write(spans_path, rows)

    report = analysis.analyze(usage)

    assert report["cost_complete"] is True
    assert report["attribution_complete"] is False
    assert report["unattributed_call_count"] == 1
    joined = " ".join(report["attribution_findings"])
    assert "duplicate_span:model_call_started" in joined
    assert "incomplete_lifecycle:model_call_started" in joined


def test_model_role_and_attempt_index_are_strictly_bound(tmp_path):
    cases = []
    cases.append(([_usage(model="different-model")], _spans(),
                  "usage_model_mismatch"))
    role_rows = _spans()
    for row in role_rows:
        if row["record_type"] in {"role_span_started", "role_span_finished"}:
            row["role"] = "critic"
    cases.append(([_usage()], role_rows, "role_identity_mismatch"))
    index_rows = _spans()
    for row in index_rows:
        if "role_attempt_index" in row:
            row["role_attempt_index"] = "bad-index"
    cases.append(([_usage()], index_rows, "invalid_role_attempt_index"))

    for index, (usage_rows, span_rows, expected) in enumerate(cases):
        case = tmp_path / str(index)
        case.mkdir()
        usage = case / "usage.jsonl"
        _write(usage, usage_rows)
        _write(Path(str(usage) + ".spans.jsonl"), span_rows)
        report = analysis.analyze(usage)
        assert report["cost_complete"] is True
        assert report["attribution_complete"] is False
        assert expected in " ".join(report["attribution_findings"])


def test_role_attempt_cannot_authorize_two_calls_or_orphan_seat(tmp_path):
    usage = tmp_path / "usage.jsonl"
    spans_path = Path(str(usage) + ".spans.jsonl")
    second_call = "4" * 32
    _write(usage, [
        _usage(),
        _usage(second_call, attempt_id="5" * 32),
    ])
    rows = _spans()
    rows.extend(row for row in _spans(second_call)
                if row["record_type"].startswith("model_call_"))
    orphan_id = "6" * 32
    rows.extend([
        {"span_schema_version": 1, "record_type": "seat_acquired",
         "role_attempt_id": orphan_id, "role_span_id": "7" * 32,
         "role_attempt_index": 1, "role": "judger",
         "queue_duration_ns": 1},
        {"span_schema_version": 1, "record_type": "seat_released",
         "role_attempt_id": orphan_id, "role_span_id": "7" * 32,
         "role_attempt_index": 1, "role": "judger",
         "held_duration_ns": 1},
    ])
    _write(spans_path, rows)

    report = analysis.analyze(usage)

    assert report["cost_complete"] is True
    assert report["attribution_complete"] is False
    findings = " ".join(report["attribution_findings"])
    assert "role_attempt_call_count" in findings
    assert "seat_without_role_attempt" in findings


def test_duplicate_legacy_call_is_ambiguous_cost_not_double_counted_as_complete(
        tmp_path):
    usage = tmp_path / "legacy.jsonl"
    row = {
        "call_id": "9" * 32, "role": "discharge",
        "model": "gpt-5.6-luna", "input_tokens": 40,
        "cached_input_tokens": 10, "uncached_input_tokens": 30,
        "output_tokens": 5,
    }
    _write(usage, [row, row])

    report = analysis.analyze(usage)

    assert report["cost_complete"] is False
    assert any(item.startswith("ambiguous_legacy_call:")
               for item in report["cost_findings"])


def test_empty_usage_is_unknown_not_vacuously_complete(tmp_path):
    usage = tmp_path / "usage.jsonl"
    usage.write_text("", encoding="utf-8")

    report = analysis.analyze(usage)

    assert report["cost_complete"] is False
    assert report["attribution_complete"] is False
    assert report["cost_findings"] == ["no_usage_attempts"]
    assert report["attribution_findings"] == ["no_usage_attempts"]


def test_claim_roles_require_business_scope_but_extract_does_not(tmp_path):
    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    usage = missing_dir / "usage.jsonl"
    rows = _spans()
    for row in rows:
        row.pop("obligation_id", None)
    _write(usage, [_usage()])
    _write(Path(str(usage) + ".spans.jsonl"), rows)
    missing = analysis.analyze(usage)
    assert missing["cost_complete"] is True
    assert missing["attribution_complete"] is False
    assert "incomplete_role_scope" in " ".join(
        missing["attribution_findings"])

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    usage = extract_dir / "usage.jsonl"
    rows = _spans()
    for row in rows:
        for field in (
                "claim_id", "claim_kind", "obligation_id", "method_attempt",
                "redo", "judger_retry_index", "independent_judger"):
            row.pop(field, None)
        row["locus"] = "candidate_extract"
        row["stage"] = "extract"
        row["round"] = 1
        if "role" in row:
            row["role"] = "extract"
    _write(usage, [_usage(role="extract")])
    _write(Path(str(usage) + ".spans.jsonl"), rows)
    extract = analysis.analyze(usage)
    assert extract["cost_complete"] is True
    assert extract["attribution_complete"] is True
    assert extract["groups"][0]["claim_id"] is None
    assert extract["groups"][0]["stage"] == "extract"


def test_corrupt_types_and_nonfinite_duration_do_not_crash(tmp_path):
    usage = tmp_path / "usage.jsonl"
    row = _usage(role={"candidate": "must-not-echo"}, duration_seconds=float("nan"))
    _write(usage, [row])

    report = analysis.analyze(usage)

    assert report["cost_complete"] is True
    assert report["attribution_complete"] is False
    assert report["groups"][0]["role"] is None
    assert report["groups"][0]["attempt_duration_seconds"] == {
        "known_sum": 0.0, "known_max": 0.0,
        "unknown_started_attempts": 1}
    assert "must-not-echo" not in json.dumps(report)


def test_contradictory_unstarted_attempt_is_unknown_not_zero(tmp_path):
    usage = tmp_path / "usage.jsonl"
    _write(usage, [_usage(
        started=False, attempt_status="succeeded",
        usage_complete=True, usage_status="complete")])

    report = analysis.analyze(usage)

    assert report["cost_complete"] is False
    group, = report["groups"]
    assert group["started_attempt_count"] == 1
    assert group["tokens"]["input_tokens"]["unknown_started_attempts"] == 1
    assert "invalid_unstarted_attempt:1" in report["cost_findings"]


def test_malformed_rows_and_unhashable_fields_fail_closed_without_raw_echo(
        tmp_path):
    usage = tmp_path / "usage.jsonl"
    spans = Path(str(usage) + ".spans.jsonl")
    poisoned = _usage(attempt_status={"secret": "candidate-body"})
    usage.write_text(json.dumps(poisoned) + "\n{broken\n", encoding="utf-8")
    _write(spans, [{
        "record_type": {"secret": "span-body"},
        "span_schema_version": 1,
    }])

    report = analysis.analyze(usage)

    assert report["cost_complete"] is False
    assert report["attribution_complete"] is False
    assert report["groups"][0]["attempt_statuses"] == {"invalid": 1}
    encoded = json.dumps(report)
    assert "candidate-body" not in encoded
    assert "span-body" not in encoded
    assert "invalid_json:usage.jsonl:2" in report["cost_findings"]
    assert "unknown_span_record:1" in report["attribution_findings"]


def test_json_integer_digit_limit_is_reported_not_raised(tmp_path):
    usage = tmp_path / "usage.jsonl"
    usage.write_text(
        '{"input_tokens":' + ("9" * 5000) + '}\n', encoding="utf-8")

    report = analysis.analyze(usage)

    assert report["cost_complete"] is False
    assert report["attribution_complete"] is False
    assert "invalid_json:usage.jsonl:1" in report["cost_findings"]


def test_jsonl_resource_caps_fail_closed(tmp_path, monkeypatch):
    usage = tmp_path / "usage.jsonl"
    usage.write_text("{}\n{}\n", encoding="utf-8")
    monkeypatch.setattr(analysis, "MAX_LEDGER_BYTES", 4)
    report = analysis.analyze(usage)
    assert "ledger_too_large:usage.jsonl" in report["cost_findings"]

    monkeypatch.setattr(analysis, "MAX_LEDGER_BYTES", 100)
    monkeypatch.setattr(analysis, "MAX_LEDGER_ROWS", 1)
    report = analysis.analyze(usage)
    assert "too_many_rows:usage.jsonl" in report["cost_findings"]

    monkeypatch.setattr(analysis, "MAX_LEDGER_ROWS", 10)
    monkeypatch.setattr(analysis, "MAX_LINE_CHARS", 1)
    report = analysis.analyze(usage)
    assert "line_too_large:usage.jsonl:1" in report["cost_findings"]


def test_cli_writes_report_without_changing_gates(tmp_path):
    usage = tmp_path / "legacy.jsonl"
    output = tmp_path / "report.json"
    _write(usage, [{
        "call_id": "8" * 32, "role": "extract",
        "model": "gpt-5.6-luna", "input_tokens": 9,
        "cached_input_tokens": 4, "uncached_input_tokens": 5,
        "output_tokens": 2,
    }])

    assert analysis.main([str(usage), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cost_complete"] is True
    assert report["attribution_complete"] is False
