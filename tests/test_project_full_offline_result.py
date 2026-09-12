import json

import pytest

from experiments.terminal_bench import project_full_offline_result as projection


def _obligation(oid="o1", text="first", **changes):
    value = {
        "id": oid, "text": text, "premise_scope": [],
        "required_basis": "record", "severity": "must", "clause": text,
        "refutable_by_absence": False,
    }
    value.update(changes)
    return value


def _event(obligations, claim="final"):
    return {"seq": 7, "type": "obligations_compiled",
            "payload": {"claim": claim, "obligations": obligations}}


def _certificate(obligations=None):
    if obligations is None:
        obligations = [{"id": "o1", "status": "verified", "reason": "ok",
                        "basis": "record", "strength": "decisive"}]
    return {"final": "final", "verdict": "verified",
            "slice": [{"id": "final", "obligations": obligations}]}


def _write(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    return path


def test_projects_exact_contract_and_marks_unsettled_tail_unevaluated(tmp_path):
    obligations = [_obligation(), _obligation(
        "o2", "second", premise_scope=["p"], required_basis="world",
        severity="should", clause="literal clause", refutable_by_absence=True)]
    path = _write(tmp_path / "events.jsonl", [_event(obligations)])

    report = projection.project_full_offline_result(path, _certificate())

    assert report["complete"] is True
    assert report["findings"] == []
    assert report["final_claim_id"] == "final"
    assert report["ordered_contract"] == [
        {"ordinal": index, **obligation}
        for index, obligation in enumerate(obligations)]
    assert report["ordered_outcomes"] == [
        {"ordinal": 0, "obligation_id": "o1", "evaluated": True,
         "status": "verified", "reason": "ok", "basis": "record",
         "strength": "decisive"},
        {"ordinal": 1, "obligation_id": "o2", "evaluated": False,
         "status": None, "reason": None, "basis": None, "strength": None},
    ]
    assert len(report["events_sha256"]) == 64


def test_last_compiled_event_for_final_is_the_audit_source(tmp_path):
    path = _write(tmp_path / "events.jsonl", [
        _event([_obligation("old", "old")]),
        _event([_obligation("other", "other")], claim="support"),
        _event([_obligation("o1", "new")]),
    ])

    report = projection.project_full_offline_result(path, _certificate())

    assert report["complete"] is True
    assert report["ordered_contract"][0]["text"] == "new"


@pytest.mark.parametrize(("events", "finding"), [
    ([], "events_empty_or_partial"),
    ([{"type": "unrelated", "payload": {}}], "final_contract_missing"),
    ([_event([])], "final_contract_empty"),
    ([_event("not-a-list")], "final_contract_malformed"),
    ([_event([_obligation(), _obligation()])], "contract_item_malformed:1"),
    ([_event([_obligation(required_basis="guess")])], "contract_item_malformed:0"),
])
def test_missing_empty_duplicate_or_malformed_contract_fails_closed(
    tmp_path, events, finding,
):
    path = tmp_path / "events.jsonl"
    if events:
        _write(path, events)
    else:
        path.write_bytes(b"")

    report = projection.project_full_offline_result(path, _certificate())

    assert report["complete"] is False
    assert report["findings"] == [finding]
    assert report["ordered_contract"] == []
    assert report["ordered_outcomes"] == []


def test_duplicate_final_entry_fails_closed(tmp_path):
    path = _write(tmp_path / "events.jsonl", [_event([_obligation()])])
    certificate = _certificate()
    certificate["slice"].append(dict(certificate["slice"][0]))

    report = projection.project_full_offline_result(path, certificate)

    assert report["complete"] is False
    assert report["findings"] == ["certificate_final_not_unique"]


def test_non_object_certificate_slice_entry_fails_closed(tmp_path):
    path = _write(tmp_path / "events.jsonl", [_event([_obligation()])])
    certificate = _certificate()
    certificate["slice"].append("not-an-entry")

    report = projection.project_full_offline_result(path, certificate)

    assert report["complete"] is False
    assert report["findings"] == ["certificate_slice_malformed"]


@pytest.mark.parametrize(("outcomes", "finding"), [
    ([{"id": "o1", "status": "verified", "reason": "ok", "basis": "record",
       "strength": "decisive"},
      {"id": "o1", "status": "refuted", "reason": "no", "basis": "record",
       "strength": "decisive"}], "outcome_id_invalid_or_duplicate:1"),
    ([{"id": "outside", "status": "verified", "reason": "ok",
       "basis": "record", "strength": "decisive"}],
     "certificate_has_unexpected_obligation"),
    ([{"id": "o1", "status": "invented", "reason": "ok", "basis": "record",
       "strength": "decisive"}], "outcome_item_malformed:0"),
])
def test_bad_certificate_outcomes_fail_closed(tmp_path, outcomes, finding):
    path = _write(tmp_path / "events.jsonl", [_event([_obligation()])])

    report = projection.project_full_offline_result(path, _certificate(outcomes))

    assert report["complete"] is False
    assert report["findings"] == [finding]


def test_duplicate_json_key_nonfinite_and_partial_files_fail_closed(tmp_path):
    cases = (
        (b'{"type":"x","type":"y"}\n', "events_duplicate_json_key"),
        (b'{"value":NaN}\n', "events_nonfinite_number"),
        (b'{"type":"x"}', "events_empty_or_partial"),
    )
    for index, (raw, finding) in enumerate(cases):
        path = tmp_path / f"events-{index}.jsonl"
        path.write_bytes(raw)
        report = projection.project_full_offline_result(path, _certificate())
        assert report["complete"] is False
        assert report["findings"] == [finding]


def test_reader_caps_rows_lines_and_total_bytes(tmp_path, monkeypatch):
    path = _write(tmp_path / "events.jsonl", [_event([_obligation()])])

    monkeypatch.setattr(projection, "MAX_EVENT_ROWS", 0)
    assert projection.project_full_offline_result(
        path, _certificate())["findings"] == ["events_too_many_rows"]
    monkeypatch.setattr(projection, "MAX_EVENT_ROWS", 10)
    monkeypatch.setattr(projection, "MAX_EVENT_LINE_BYTES", 1)
    assert projection.project_full_offline_result(
        path, _certificate())["findings"] == ["events_invalid_line:1"]
    monkeypatch.setattr(projection, "MAX_EVENT_LINE_BYTES", 2 * 1024 * 1024)
    monkeypatch.setattr(projection, "MAX_EVENTS_BYTES", 1)
    assert projection.project_full_offline_result(
        path, _certificate())["findings"] == ["events_unsafe_file"]


def test_reader_rejects_symlink_and_hardlink_aliases(tmp_path):
    path = _write(tmp_path / "events.jsonl", [_event([_obligation()])])
    symlink = tmp_path / "events-symlink.jsonl"
    symlink.symlink_to(path)
    hardlink = tmp_path / "events-hardlink.jsonl"
    hardlink.hardlink_to(path)

    assert projection.project_full_offline_result(
        symlink, _certificate())["findings"] == ["events_noncanonical_path"]
    assert projection.project_full_offline_result(
        hardlink, _certificate())["findings"] == ["events_unsafe_file"]


def test_projection_does_not_alias_or_mutate_inputs(tmp_path):
    obligation = _obligation(premise_scope=["p"])
    certificate = _certificate()
    original = json.loads(json.dumps(certificate))
    path = _write(tmp_path / "events.jsonl", [_event([obligation])])

    report = projection.project_full_offline_result(path, certificate)
    report["ordered_contract"][0]["premise_scope"].append("caller-change")

    assert certificate == original
    assert obligation["premise_scope"] == ["p"]
