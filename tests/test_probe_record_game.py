"""Local-only tests for the paid one-obligation diagnostic wrapper."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.terminal_bench import probe_record_game as probe
from test_contract_replay import source_state
from verifier.core.model import content_hash
from verifier.engine.typed_judger_packet import TYPED_JUDGER_PACKET_PROTOCOL
from verifier.llm.codex.runtime import CodexRuntime
from verifier.llm.schema import FixtureBackend
from verifier.llm.oracle_execution import raise_oracle_execution_payload


USAGE = {"input_tokens": 100, "cached_input_tokens": 30,
         "output_tokens": 20, "reasoning_output_tokens": 10}
METHOD = {
    "procedure": "Check that one Bash call has a nonblank command string.",
    "artifact": "A compact authenticated ledger receipt.",
    "level": "exact",
    "basis": "record",
    "strength": "decisive",
    "bridges": [],
    "adapter": "ledger.tool-call-string",
    "adapter_version": 2,
    "adapter_arguments": json.dumps({
        "tool_names": ["Bash"], "value_pointer": "/input/command",
        "selector": "any", "through_seq": 2, "predicate": "nonblank",
    }),
    "adapter_inputs": [{"ref": "ledger:events", "provenance": "ledger",
                        "role": "subject"}],
}
DISCHARGE = {
    "outcome": "holds", "witness": "command is nonblank",
    "artifact": "executor receipt", "assumed_premises": [],
    "witness_kind": "none",
    "inputs": [{"ref": "ledger:events", "provenance": "ledger",
                "role": "subject"}],
}
JUDGE = {"sufficient": "yes", "faithful": "yes", "degradation": "n/a",
         "reason": "The exact ledger check settles the obligation."}


def _inputs(tmp_path):
    source = source_state(tmp_path, extract=False)
    row_id = content_hash({"fixture": "final row"})
    case = tmp_path / "case.json"
    case.write_text(json.dumps({
        "row": {"id": row_id, "kind": "final", "text": "The record is valid"},
        "obligation": {
            "claim_id": row_id,
            "text": "At least one Bash call has a nonblank command",
            "required_basis": "record", "severity": "must",
            "clause": "A command must have been recorded.",
            "refutable_by_absence": True, "premise_scope": [],
        },
    }))
    return source, case


def _backend_factory(monkeypatch, responses, calls, *, compact_inspection=False,
                     judger_prefetch=False):
    queue = copy.deepcopy(responses)

    def run(cmd, **kwargs):
        calls.append(list(cmd))
        assert queue, "unexpected model call"
        response = queue.pop(0)
        text = json.dumps(response)
        Path(cmd[cmd.index("-o") + 1]).write_text(text)
        stream = [
            {"type": "thread.started", "thread_id": f"stub-{len(calls)}"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {
                "id": "answer", "type": "agent_message", "text": text}},
            {"type": "turn.completed", "usage": USAGE},
        ]
        return SimpleNamespace(returncode=0,
                               stdout="\n".join(map(json.dumps, stream)),
                               stderr="")

    monkeypatch.setattr("verifier.llm.codex.runtime._run_codex_process", run)

    def factory(state):
        return CodexRuntime(cwd=str(state), model="gpt-5.6-luna",
                            effort="medium", service_tier="standard",
                            compact_inspection=compact_inspection,
                            judger_prefetch=judger_prefetch)

    return factory


def _run(tmp_path, monkeypatch, typed, *, compact_inspection=False,
         judger_prefetch=False, typed_judger_packet=False):
    source, case = _inputs(tmp_path)
    calls = []
    responses = [raise_oracle_execution_payload({"methods": [METHOD]})]
    if not typed:
        responses.append(DISCHARGE)
    responses.append(JUDGE)
    result = probe.run_probe(
        source, case, (tmp_path / "typed") if typed else (tmp_path / "control"),
        typed_adapters=typed,
        expected_source_seal=probe.offline.verify_immutable_input_anchor(str(source)),
        expected_source_fingerprint=probe._source_fingerprint(probe.SOURCE_ROOT),
        compact_inspection=compact_inspection,
        judger_prefetch=judger_prefetch,
        typed_judger_packet=typed_judger_packet,
        backend_factory=_backend_factory(
            monkeypatch, responses, calls, compact_inspection=compact_inspection,
            judger_prefetch=judger_prefetch),
    )
    return result, calls, source


def test_probe_can_exercise_production_typed_judger_transport(
        tmp_path, monkeypatch):
    result, calls, _ = _run(
        tmp_path, monkeypatch, True, typed_judger_packet=True)

    assert result["complete"] is True
    assert result["typed_judger_packet"] is True
    assert result["typed_judger_attempts"] == 1
    assert len(calls) == 2
    rows = [json.loads(line) for line in
            (tmp_path / "typed" / "usage.jsonl").read_text().splitlines()]
    assert rows[0]["typed_judger_transport"] is False
    assert rows[1]["typed_judger_transport"] is True
    assert rows[1]["typed_judger_protocol"] \
        == TYPED_JUDGER_PACKET_PROTOCOL


@pytest.mark.parametrize("typed,expected_calls", [(False, 3), (True, 2)])
@pytest.mark.parametrize("enabled", [False, True])
def test_prefetch_profile_is_explicit_and_judger_only(
        tmp_path, monkeypatch, typed, expected_calls, enabled):
    result, calls, _ = _run(tmp_path, monkeypatch, typed, judger_prefetch=enabled)
    assert result["complete"] and result["runtime_profile_accounting_complete"]
    assert result["judger_prefetch"] is enabled
    assert result["prefetch_attached_attempts"] == int(enabled)
    assert len(calls) == expected_calls
    assert result["cost_report"]["metrics"]["started_attempts"] == expected_calls
    assert result["binding_result"]["outcome"] == "holds"
    root = tmp_path / ("typed" if typed else "control")
    rows = [json.loads(line) for line in (root / "usage.jsonl").read_text().splitlines()]
    assert [row["role"] for row in rows] == (
        ["oracles", "judger"] if typed else ["oracles", "discharge", "judger"])
    for row in rows:
        assert row["judger_prefetch"] is enabled
        assert row["judger_prefetch_attached"] is (enabled and row["role"] == "judger")
    assert result["excluded_stage_costs"] == {
        "extract": None, "obligations": None, "critic": None, "setup": None}


@pytest.mark.parametrize("bad", [None, 0, 1, "1", [], {}])
def test_invalid_prefetch_flag_fails_before_creating_output(tmp_path, bad):
    with pytest.raises(ValueError, match="judger_prefetch must be a boolean"):
        probe.run_probe(tmp_path / "missing", tmp_path / "missing-case",
                        tmp_path / "out", typed_adapters=True,
                        expected_source_seal="f" * 64,
                        expected_source_fingerprint="f" * 64,
                        judger_prefetch=bad)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field,value", [
    ("judger_prefetch", False),
    ("judger_prefetch_protocol", "wrong"),
    ("model", "different-model"),
    ("effort", "different-effort"),
    ("judger_prefetch_attached", True),
])
def test_mislabeled_attempt_profile_is_inconclusive_not_free(
        tmp_path, monkeypatch, field, value):
    original_snapshots = probe._ledger_snapshots

    def tampered(usage, spans):
        snapshots = original_snapshots(usage, spans)
        rows = [json.loads(line) for line in snapshots["usage"].splitlines()]
        # The first attempt is Oracle, so attached=True is invalid there.
        rows[0][field] = value
        snapshots["usage"] = ("\n".join(map(json.dumps, rows)) + "\n").encode()
        return snapshots

    monkeypatch.setattr(probe, "_ledger_snapshots", tampered)
    result, calls, _ = _run(tmp_path, monkeypatch, True, judger_prefetch=True)
    assert result["complete"] is False
    assert result["runtime_profile_accounting_complete"] is False
    assert result["cost_report"]["complete"] is True
    assert result["cost_report"]["metrics"]["started_attempts"] == len(calls) == 2


def test_mislabeled_prefetch_backend_never_calls_model(tmp_path, monkeypatch):
    source, case = _inputs(tmp_path)
    calls = []
    result = probe.run_probe(
        source, case, tmp_path / "mismatch", typed_adapters=True,
        expected_source_seal=probe.offline.verify_immutable_input_anchor(str(source)),
        expected_source_fingerprint=probe._source_fingerprint(probe.SOURCE_ROOT),
        judger_prefetch=True,
        backend_factory=_backend_factory(monkeypatch, [], calls))
    assert not calls
    assert result["complete"] is False
    assert result["error_class"] == "ReplaySetupError"


@pytest.mark.parametrize("field", [
    "judger_prefetch", "judger_prefetch_protocol",
    "judger_prefetch_packet_sha256", "judger_prefetch_attached",
    "compact_inspection",
])
def test_missing_disabled_profile_field_is_not_inferred_as_default(
        tmp_path, monkeypatch, field):
    original_snapshots = probe._ledger_snapshots

    def missing(usage, spans):
        snapshots = original_snapshots(usage, spans)
        rows = [json.loads(line) for line in snapshots["usage"].splitlines()]
        del rows[0][field]
        snapshots["usage"] = ("\n".join(map(json.dumps, rows)) + "\n").encode()
        return snapshots

    monkeypatch.setattr(probe, "_ledger_snapshots", missing)
    result, calls, _ = _run(tmp_path, monkeypatch, True)
    assert result["complete"] is False
    assert result["runtime_profile_accounting_complete"] is False
    assert result["cost_report"]["complete"] is True
    assert len(calls) == 2


def test_inspection_profile_is_explicit_and_each_attempt_matches(tmp_path, monkeypatch):
    result, calls, _ = _run(tmp_path, monkeypatch, True, compact_inspection=True)
    assert result["compact_inspection"] is True
    assert result["complete"] is True and result["test_only"] is True
    assert result["runtime"] == probe.RUNTIME
    assert len(calls) == 2
    rows = [json.loads(line) for line in
            (tmp_path / "typed" / "usage.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["compact_inspection"] is True for row in rows)
    assert len({row["runtime_instruction_identity"] for row in rows}) == 1


def test_probe_rejects_mislabelled_backend_profile(tmp_path, monkeypatch):
    source, case = _inputs(tmp_path)
    calls = []
    result = probe.run_probe(
        source, case, tmp_path / "mismatch", typed_adapters=True,
        expected_source_seal=probe.offline.verify_immutable_input_anchor(str(source)),
        expected_source_fingerprint=probe._source_fingerprint(probe.SOURCE_ROOT),
        compact_inspection=True,
        backend_factory=_backend_factory(monkeypatch, [], calls),
    )
    assert not calls
    assert result["complete"] is False
    assert result["error_class"] == "ReplaySetupError"


@pytest.mark.parametrize("typed,expected_calls", [
    (False, 3),
    (True, 2),
])
def test_real_binding_game_local_stub_retains_complete_bills(
        tmp_path, monkeypatch, typed, expected_calls):
    result, calls, source = _run(tmp_path, monkeypatch, typed)
    assert result["complete"] is True
    assert result["test_only"] is True
    assert result["binding_result"]["outcome"] == "holds"
    assert result["binding_result"]["instrument"] == METHOD["procedure"]
    assert result["cost_report"]["complete"] is True
    assert result["cost_report"]["metrics"]["started_attempts"] == expected_calls
    assert len(calls) == expected_calls
    assert result["source_unchanged"] is True
    assert probe.offline.verify_immutable_input_anchor(str(source)) \
        == result["source_seal"]
    root = Path(result["source"]).parent / ("typed" if typed else "control")
    assert not (root / "state" / "score.json").exists()
    assert not (root / "state" / "claims.sqlite").exists()
    assert not (root / "state" / "extract.json").exists()
    assert result["excluded_stage_costs"] == {
        "extract": None, "obligations": None, "critic": None, "setup": None}
    if typed:
        assert result["events"]["adapter_discharges"][0]["adapter"] \
            == "ledger.tool-call-string@2"
        assert result["binding_result"]["artifact_bytes"] > 0


def test_plan_and_paid_confirmation_do_not_start_model(tmp_path, monkeypatch, capsys):
    source, case = _inputs(tmp_path)
    fingerprint = probe._source_fingerprint(probe.SOURCE_ROOT)
    common = ["--source", str(source), "--case", str(case),
              "--output-root", str(tmp_path / "out"),
              "--typed-adapters", "1", "--source-seal",
              probe.offline.verify_immutable_input_anchor(str(source)),
              "--source-fingerprint", fingerprint]
    monkeypatch.setattr(probe, "run_probe",
                        lambda *args, **kwargs: pytest.fail("model path started"))
    assert probe.main(common) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["diagnostic_only"] is True
    assert planned["judger_prefetch"] is False
    assert planned["judger_prefetch_protocol"] is None
    assert probe.main([*common, "--judger-prefetch", "1"]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["judger_prefetch"] is True
    assert planned["judger_prefetch_protocol"] == probe.PREFETCH_PROTOCOL
    with pytest.raises(SystemExit):
        probe.main([*common, "--execute"])


def test_invalid_case_is_rejected_before_output(tmp_path):
    source, case = _inputs(tmp_path)
    payload = json.loads(case.read_text())
    payload["obligation"]["required_basis"] = "world"
    case.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="record obligation"):
        probe.run_probe(
            source, case, tmp_path / "out", typed_adapters=True,
            expected_source_seal=probe.offline.verify_immutable_input_anchor(str(source)),
            expected_source_fingerprint=probe._source_fingerprint(probe.SOURCE_ROOT),
            backend_factory=lambda _state: FixtureBackend({}),
        )
    assert not (tmp_path / "out").exists()


def test_interrupt_writes_terminal_record_and_keeps_raw_ledgers(tmp_path):
    source, case = _inputs(tmp_path)

    class InterruptBackend(FixtureBackend):
        def complete(self, *args, **kwargs):
            raise KeyboardInterrupt

    output = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        probe.run_probe(
            source, case, output, typed_adapters=False,
            expected_source_seal=probe.offline.verify_immutable_input_anchor(str(source)),
            expected_source_fingerprint=probe._source_fingerprint(probe.SOURCE_ROOT),
            backend_factory=lambda _state: InterruptBackend({}),
        )
    terminal = json.loads((output / "probe-result.json").read_text())
    assert terminal["diagnostic_status"] == "inconclusive"
    assert terminal["error_class"] == "KeyboardInterrupt"
    assert (output / "usage.jsonl").exists()
    assert (output / "usage.jsonl.spans.jsonl").exists()


def test_postprocessing_failure_is_terminal_and_cannot_look_complete(
        tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "_binding",
                        lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")))
    result, calls, _source = _run(tmp_path, monkeypatch, typed=True)
    assert len(calls) == 2
    assert result["complete"] is False
    assert result["error_class"] == "RuntimeError"
    assert result["cost_report"]["complete"] is True
    assert result["binding_result"] is None


def test_output_must_be_fresh_and_operator_pins_are_enforced(tmp_path):
    source, case = _inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    arguments = dict(
        typed_adapters=True,
        expected_source_seal=probe.offline.verify_immutable_input_anchor(str(source)),
        expected_source_fingerprint=probe._source_fingerprint(probe.SOURCE_ROOT),
        backend_factory=lambda _state: FixtureBackend({}),
    )
    with pytest.raises(ValueError, match="new absolute"):
        probe.run_probe(source, case, output, **arguments)
    arguments["expected_source_seal"] = "f" * 64
    with pytest.raises(ValueError, match="operator seal"):
        probe.run_probe(source, case, tmp_path / "new", **arguments)
