"""Local tests for the task-neutral synthetic record-semantics gate."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.terminal_bench import prepare_record_semantic_gate as gate
from experiments.terminal_bench import probe_record_game as probe
from verifier import offline
from verifier.llm.schema import FixtureBackend


REQUIRED = "instrument.delta.v2"
OPAQUE = "instrument.theta.v9"
JUDGE = {
    "sufficient": "yes",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "The complete ledger receipt settles this synthetic condition.",
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _feed(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _method(authority: str, required_label: str) -> dict:
    arguments = {
        "selector": "any",
        "value_pointer": "/input/keystrokes",
        "through_seq": 8,
        "predicate": "nonblank",
    }
    if authority == "exact_identity":
        arguments["tool_names"] = [required_label]
    return {
        "procedure": "Apply the literal synthetic record predicate.",
        "artifact": "A compact authenticated ledger receipt.",
        "level": "exact",
        "basis": "record",
        "strength": "decisive",
        "bridges": [],
        "adapter": "ledger.tool-call-string",
        "adapter_version": 2,
        "adapter_arguments": json.dumps(arguments),
        "adapter_inputs": [{"ref": "ledger:events", "provenance": "ledger",
                            "role": "subject"}],
    }


def _discharge(outcome: str) -> dict:
    return {
        "outcome": outcome,
        "witness": ("" if outcome == "holds" else
                    "The complete ledger contains no qualifying record."),
        "artifact": "synthetic executor receipt",
        "assumed_premises": [],
        "witness_kind": "none" if outcome == "holds" else "absence",
        "inputs": [{"ref": "ledger:events", "provenance": "ledger",
                    "role": "subject"}],
    }


def _run_fixture(record: dict, output: Path, *, typed: bool):
    fixtures = {"oracles": {"methods": [
                    _method(record["authority"],
                            record["required_tool_identity"])]},
                "judger": JUDGE}
    if not typed:
        fixtures["discharge"] = _discharge(record["expected_binding_outcome"])
    backend = FixtureBackend(fixtures)
    result = probe.run_probe(
        Path(record["source"]), Path(record["case"]), output,
        typed_adapters=typed,
        expected_source_seal=record["source_seal"],
        expected_source_fingerprint=probe._source_fingerprint(probe.SOURCE_ROOT),
        backend_factory=lambda _state: backend,
    )
    return result, backend


def test_prepare_generates_four_minimal_sealed_inputs(tmp_path):
    output = tmp_path / "gate"
    manifest = gate.prepare(
        output, required_label=REQUIRED, opaque_label=OPAQUE)

    assert manifest == _json(output / "manifest.json")
    assert manifest["diagnostic_only"] is True
    assert manifest["full_e2e_acceptance"] is False
    assert manifest["synthetic_gold"] is True
    assert manifest["terminal_bench_input"] is False
    assert manifest["model_calls"] == 0
    assert manifest["typed_adapter_required"] is False
    assert manifest["probe_fixed_config"] == {
        "compact_inspection": False, "method_cache": False,
        "live_world": False,
    }
    assert len(manifest["records"]) == 4
    assert {(r["authority"], r["variant"], r["expected_binding_outcome"])
            for r in manifest["records"]} == {
        ("exact_identity", "required", "holds"),
        ("exact_identity", "opaque", "refuted"),
        ("any_identity", "required", "holds"),
        ("any_identity", "opaque", "holds"),
    }

    for record in manifest["records"]:
        source = Path(record["source"])
        case = Path(record["case"])
        assert offline.verify_immutable_input_anchor(str(source)) \
            == record["source_seal"]
        assert hashlib.sha256(case.read_bytes()).hexdigest() \
            == record["case_sha256"]
        assert hashlib.sha256((source / "hook_feed.jsonl").read_bytes()).hexdigest() \
            == record["feed_sha256"]
        row, obligation, _ = probe._case(case)
        assert row["id"] == record["claim_id"]
        assert obligation.id == record["obligation_id"]
        assert {item.name for item in source.iterdir()} == {
            "task.md", "final.txt", "hook_feed.jsonl",
            offline.IMMUTABLE_INPUT_SEAL_FILE,
        }
        assert not any(source.glob("**/score.json"))
        assert not any(source.glob("**/extract.json"))


@pytest.mark.parametrize("authority", ["exact_identity", "any_identity"])
def test_pair_diff_is_only_the_declared_target_identity(tmp_path, authority):
    manifest = gate.prepare(
        tmp_path / "gate", required_label=REQUIRED, opaque_label=OPAQUE)
    indexed = {(r["authority"], r["variant"]): r
               for r in manifest["records"]}
    required = indexed[(authority, "required")]
    opaque = indexed[(authority, "opaque")]
    left = _feed(Path(required["source"]) / "hook_feed.jsonl")
    right = _feed(Path(opaque["source"]) / "hook_feed.jsonl")

    normalized = copy.deepcopy(right)
    changed = 0
    for source_event, event in zip(left, normalized, strict=True):
        if event["payload"]["tool"] == OPAQUE:
            event["payload"]["tool"] = REQUIRED
            changed += 1
        assert event == source_event
    assert changed == 6  # call + result for each of the three target records
    calls = [event for event in left if event["type"] == "tool_call"]
    results = [event for event in left if event["type"] == "tool_result"]
    assert [event["payload"]["input"]["keystrokes"] for event in calls] == [
        "printf synthetic-record", "   \t", "", "unrelated nonblank input"]
    for event in results:
        result_text = event["payload"]["excerpt"]
        assert result_text == "synthetic result"
        assert event["payload"]["response_chars"] == len(result_text)
        assert event["payload"]["response_sha256"] == \
            hashlib.sha256(result_text.encode()).hexdigest()
    assert calls[-1]["payload"]["tool"] == gate.NONMATCHING_LABEL
    assert Path(required["case"]).read_bytes() == Path(opaque["case"]).read_bytes()
    assert (Path(required["source"]) / "task.md").read_bytes() == \
        (Path(opaque["source"]) / "task.md").read_bytes()
    assert (Path(required["source"]) / "final.txt").read_bytes() == \
        (Path(opaque["source"]) / "final.txt").read_bytes()


def test_explicit_authorities_are_independent(tmp_path):
    manifest = gate.prepare(
        tmp_path / "gate", required_label=REQUIRED, opaque_label=OPAQUE)
    by_authority = {}
    for record in manifest["records"]:
        by_authority.setdefault(record["authority"], record)
    exact = by_authority["exact_identity"]
    any_identity = by_authority["any_identity"]
    assert exact["claim_id"] != any_identity["claim_id"]
    assert exact["obligation_id"] != any_identity["obligation_id"]
    assert Path(exact["case"]).read_bytes() != Path(any_identity["case"]).read_bytes()
    assert f'exactly equal to "{REQUIRED}"' in \
        (Path(exact["source"]) / "task.md").read_text()
    assert "Tool identity is irrelevant" in \
        (Path(any_identity["source"]) / "task.md").read_text()


@pytest.mark.parametrize(("required", "opaque"), [
    ("instrument.delta.v2", "instrument.theta.v9"),
    ("engine.alpha-3", "engine.beta-4"),
])
def test_both_tool_identities_are_operator_parameters(
        tmp_path, required, opaque):
    manifest = gate.prepare(
        tmp_path / f"gate-{required}", required_label=required,
        opaque_label=opaque)
    assert "bash" not in {manifest["required_label"], manifest["opaque_label"]}
    assert {record["literal_tool_identity"] for record in manifest["records"]} \
        == {required, opaque}
    exact = [record for record in manifest["records"]
             if record["authority"] == "exact_identity"]
    assert {record["expected_binding_outcome"] for record in exact} \
        == {"holds", "refuted"}
    task = (Path(exact[0]["source"]) / "task.md").read_text()
    assert json.dumps(required) in task
    assert opaque not in task


@pytest.mark.parametrize("typed", [False, True])
def test_all_cases_use_existing_binding_game_without_cache(tmp_path, typed):
    manifest = gate.prepare(
        tmp_path / "gate", required_label=REQUIRED, opaque_label=OPAQUE)
    for index, record in enumerate(manifest["records"]):
        result, backend = _run_fixture(
            record, tmp_path / "runs" / f"{index}-{typed}", typed=typed)
        assert result["binding_result"]["outcome"] \
            == record["expected_binding_outcome"]
        assert result["scope"] == probe.SCOPE
        assert result["diagnostic_only"] is True
        assert result["full_e2e_acceptance"] is False
        assert result["test_only"] is True
        assert result["compact_inspection"] is False
        assert result["complete"] is False  # FixtureBackend has no paid ledger.
        assert result["cost_report"]["complete"] is False
        assert backend.calls[0]["role"] == "oracles"
        assert backend.calls[-1]["role"] == "judger"
        assert [call["role"] for call in backend.calls].count("judger") == 1
        if typed:
            assert [call["role"] for call in backend.calls] == [
                "oracles", "judger"]
            assert len(result["events"]["adapter_discharges"]) == 1
            assert not result["events"]["adapter_fallbacks"]
        else:
            assert [call["role"] for call in backend.calls] == [
                "oracles", "discharge", "judger"]
            assert not result["events"]["adapter_discharges"]


@pytest.mark.parametrize("label", ["", " white", "slash/name", "x" * 129])
def test_invalid_labels_fail_before_output(tmp_path, label):
    output = tmp_path / "gate"
    with pytest.raises(ValueError, match="labels"):
        gate.prepare(output, required_label=REQUIRED, opaque_label=label)
    assert not output.exists()


@pytest.mark.parametrize(("required", "opaque"), [
    (REQUIRED, REQUIRED),
    (gate.NONMATCHING_LABEL, OPAQUE),
    (REQUIRED, gate.NONMATCHING_LABEL),
])
def test_identity_collisions_fail_before_output(tmp_path, required, opaque):
    output = tmp_path / "gate"
    with pytest.raises(ValueError, match="must differ"):
        gate.prepare(
            output, required_label=required, opaque_label=opaque)
    assert not output.exists()


def test_output_is_absolute_fresh_and_never_overwritten(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        gate.prepare(
            Path("relative-gate"), required_label=REQUIRED,
            opaque_label=OPAQUE)
    output = tmp_path / "gate"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("untouched", encoding="utf-8")
    with pytest.raises(ValueError, match="must be new"):
        gate.prepare(output, required_label=REQUIRED, opaque_label=OPAQUE)
    assert marker.read_text(encoding="utf-8") == "untouched"
