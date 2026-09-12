import copy
import json
import os
from pathlib import Path

import pytest

from experiments.terminal_bench import merge_setup_accounting as merge
from test_shared_setup import _attempt, _spans, _write_jsonl


def _component(tmp_path, component_id, calls):
    root = tmp_path / "components"
    usage = root / f"{component_id}.usage.jsonl"
    spans = root / f"{component_id}.spans.jsonl"
    attempts = [_attempt(number, role) for number, role in calls]
    for row in attempts:
        row.update({"setup_id": "setup-a", "backend": "codex-cli"})
    _write_jsonl(usage, attempts)
    _write_jsonl(spans, [row for number, role in calls
                         for row in _spans(number, role)])
    return {
        "component_id": component_id,
        "usage_path": str(usage.resolve()),
        "spans_path": str(spans.resolve()),
    }


def _components(tmp_path):
    return [
        _component(tmp_path, "extract", [(1, "extract")]),
        _component(tmp_path, "claim-a", [
            (2, "obligations"), (3, "critic")]),
    ]


def _run(tmp_path, components, *, destination="accounting", wall=7.5):
    return merge.merge_setup_accounting(
        tmp_path / destination,
        setup_id="setup-a",
        components=components,
        wall_seconds=wall,
    )


def test_exact_bytes_merge_once_and_yield_flat_setup_metrics(tmp_path):
    components = _components(tmp_path)
    expected_usage = b"".join(
        Path(row["usage_path"]).read_bytes() for row in components)
    expected_spans = b"".join(
        Path(row["spans_path"]).read_bytes() for row in components)

    result = _run(tmp_path, components)

    assert result["complete"] is True
    assert result["findings"] == []
    assert Path(result["usage_path"]).read_bytes() == expected_usage
    assert Path(result["spans_path"]).read_bytes() == expected_spans
    assert result["model_work"]["attempt_count"] == 3
    assert result["metrics"] == {
        "complete": True,
        "cost_complete": True,
        "wall_complete": True,
        "started_attempts": 3,
        "uncached_input_tokens": 120,
        "cached_input_tokens": 180,
        "output_tokens": 30,
        "wall_seconds": 7.5,
    }
    assert all(record[label]["included"] and record[label]["stable"]
               for record in result["source_records"]
               for label in ("usage", "spans"))
    assert not any(path.name.startswith(".")
                   for path in (tmp_path / "accounting").iterdir())


def test_provider_and_schema_failures_are_preserved_as_distinct_bills(
        tmp_path):
    extract = _component(tmp_path, "extract", [(1, "extract")])
    claim = _component(tmp_path, "claim-a", [
        (2, "obligations"), (3, "critic"), (4, "critic")])
    usage_path = Path(claim["usage_path"])
    rows = [json.loads(line) for line in
            usage_path.read_text(encoding="utf-8").splitlines()]
    provider_retry = copy.deepcopy(rows[0])
    provider_retry.update({
        "attempt_id": f"{902:032x}",
        "attempt_index": 1,
        "attempt_status": "provider_retry",
    })
    rows[0]["attempt_index"] = 2
    rows[1]["attempt_status"] = "schema_error"
    _write_jsonl(usage_path, [provider_retry, *rows])
    spans_path = Path(claim["spans_path"])
    spans = [row for number, role in ((2, "obligations"), (3, "critic"),
                                      (4, "critic"))
             for row in _spans(number, role)]
    for row in spans:
        if (row.get("call_id") == f"{3:032x}"
                and row["record_type"] == "model_call_finished"):
            row["status"] = "error"
        if (row.get("role_attempt_id") == f"{303:032x}"
                and row["record_type"] == "role_attempt_finished"):
            row["status"] = "error"
        if (row.get("role_span_id") == f"{203:032x}"
                and row["record_type"] == "role_span_finished"):
            row["status"] = "error"
    _write_jsonl(spans_path, spans)

    result = _run(tmp_path, [extract, claim])

    assert result["complete"] is True
    assert result["model_work"]["totals"]["attempt_statuses"] == {
        "provider_retry": 1, "schema_error": 1, "succeeded": 3}
    assert result["metrics"]["started_attempts"] == 5
    assert result["metrics"]["uncached_input_tokens"] == 200


def test_empty_component_plan_is_incomplete_and_unknown(tmp_path):
    result = _run(tmp_path, [])

    assert result["complete"] is False
    assert result["source_records"] == []
    assert result["model_work"]["cost_complete"] is False
    assert result["model_work"]["totals"]["started_attempt_count"] == 0
    assert result["metrics"]["cost_complete"] is False
    assert result["metrics"]["started_attempts"] is None
    assert Path(result["usage_path"]).read_bytes() == b""
    assert Path(result["spans_path"]).read_bytes() == b""


def test_missing_component_ledger_keeps_other_known_rows_but_not_a_charge(
        tmp_path):
    components = _components(tmp_path)
    Path(components[1]["usage_path"]).unlink()

    result = _run(tmp_path, components)

    assert result["complete"] is False
    assert "unavailable_usage:1" in result["findings"]
    assert result["model_work"]["totals"]["started_attempt_count"] == 1
    assert result["model_work"]["totals"]["tokens"][
        "input_tokens"]["known_sum"] == 100
    assert result["metrics"]["started_attempts"] is None


@pytest.mark.parametrize("duplicate", ["component_id", "source_path"])
def test_duplicate_component_identity_or_source_is_never_double_charged(
        tmp_path, duplicate):
    components = _components(tmp_path)
    if duplicate == "component_id":
        extra = _component(tmp_path, "claim-b", [(4, "critic")])
        extra["component_id"] = components[1]["component_id"]
        expected_attempts = 4
    else:
        extra = {
            "component_id": "duplicate-extract",
            "usage_path": components[0]["usage_path"],
            "spans_path": components[0]["spans_path"],
        }
        expected_attempts = 3

    result = _run(tmp_path, [*components, extra])

    assert result["complete"] is False
    assert any(duplicate in finding for finding in result["findings"])
    assert result["model_work"]["totals"][
        "started_attempt_count"] == expected_attempts
    assert result["metrics"]["started_attempts"] is None


def test_unterminated_component_is_omitted_without_hiding_other_cost(
        tmp_path):
    components = _components(tmp_path)
    usage = Path(components[1]["usage_path"])
    usage.write_bytes(usage.read_bytes().rstrip(b"\n"))

    result = _run(tmp_path, components)

    assert result["complete"] is False
    assert "unterminated_usage:1" in result["findings"]
    assert result["model_work"]["totals"]["started_attempt_count"] == 1
    assert result["metrics"]["cost_complete"] is False


def test_source_change_between_capture_and_recheck_is_fail_closed(
        tmp_path, monkeypatch):
    components = _components(tmp_path)
    victim = Path(components[1]["usage_path"])
    original = merge._private_read
    mutated = False

    def mutate_after_capture(path, cap):
        nonlocal mutated
        raw = original(path, cap)
        if path == victim and not mutated:
            mutated = True
            victim.write_bytes(raw + b"\n")
        return raw

    monkeypatch.setattr(merge, "_private_read", mutate_after_capture)
    result = _run(tmp_path, components)

    assert result["complete"] is False
    assert "changed_usage:1" in result["findings"]
    assert result["model_work"]["totals"]["started_attempt_count"] == 3
    assert result["metrics"]["started_attempts"] is None


@pytest.mark.parametrize("target", ["source", "aggregate"])
def test_change_during_model_audit_is_detected_on_final_recheck(
        tmp_path, monkeypatch, target):
    components = _components(tmp_path)
    original = merge.shared_setup.audit_model_work

    def mutate_after_audit(*args, **kwargs):
        result = original(*args, **kwargs)
        if target == "source":
            path = Path(components[1]["usage_path"])
        else:
            path = Path(args[0])
            path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(
        merge.shared_setup, "audit_model_work", mutate_after_audit)
    result = _run(tmp_path, components)

    assert result["complete"] is False
    expected = ("changed_usage:1" if target == "source"
                else "changed_aggregate_usage")
    assert expected in result["findings"]
    assert result["metrics"]["started_attempts"] is None


@pytest.mark.parametrize("field", ["setup_id", "backend"])
def test_every_usage_row_is_bound_to_setup_and_codex_backend(
        tmp_path, field):
    components = _components(tmp_path)
    path = Path(components[1]["usage_path"])
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines()]
    rows[-1][field] = "different"
    _write_jsonl(path, rows)

    result = _run(tmp_path, components)

    assert result["complete"] is False
    expected = ("usage_setup_mismatch" if field == "setup_id"
                else "usage_backend_mismatch")
    assert any(expected in finding for finding in result["findings"])
    assert result["model_work"]["cost_complete"] is True
    assert result["metrics"]["started_attempts"] is None


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_nonprivate_source_is_rejected_while_other_rows_remain(kind, tmp_path):
    components = _components(tmp_path)
    original = Path(components[1]["usage_path"])
    replacement = original.with_name("replacement.jsonl")
    if kind == "symlink":
        replacement.symlink_to(original)
    else:
        replacement.hardlink_to(original)
    components[1]["usage_path"] = str(replacement)

    result = _run(tmp_path, components)

    assert result["complete"] is False
    assert "unavailable_usage:1" in result["findings"]
    assert result["model_work"]["totals"]["started_attempt_count"] == 1
    assert result["metrics"]["started_attempts"] is None


def test_total_aggregate_bytes_are_bounded(tmp_path, monkeypatch):
    components = _components(tmp_path)
    first_size = Path(components[0]["usage_path"]).stat().st_size
    monkeypatch.setattr(merge.analyze_model_spans, "MAX_LEDGER_BYTES",
                        first_size)

    result = _run(tmp_path, components)

    assert result["complete"] is False
    assert "unavailable_usage:1" in result["findings"]
    assert result["metrics"]["started_attempts"] is None


def test_fifo_source_is_rejected_without_blocking(tmp_path):
    components = _components(tmp_path)
    fifo = tmp_path / "components" / "usage.fifo"
    os.mkfifo(fifo)
    components[1]["usage_path"] = str(fifo)

    result = _run(tmp_path, components)

    assert result["complete"] is False
    assert "unavailable_usage:1" in result["findings"]
    assert result["model_work"]["totals"]["started_attempt_count"] == 1
