"""Local-only tests for the synthetic Judger direction probe."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.terminal_bench import probe_judger_direction as probe
from verifier.core.events import EventLog
from verifier.llm.codex.runtime import CodexRuntime
from verifier.llm.schema import FixtureBackend


USAGE = {
    "input_tokens": 90,
    "cached_input_tokens": 20,
    "output_tokens": 12,
    "reasoning_output_tokens": 5,
}
EXPECTED = ("yes", "no", "no", "yes")


def _verdict(sufficient: str):
    return {
        "sufficient": sufficient,
        "faithful": "yes",
        "degradation": "n/a",
        "reason": "The real adapter evidence has the stated directional scope.",
    }


def _codex_factory(monkeypatch, responses, calls, *, on_call=None,
                   judger_prefetch=False):
    queue = list(responses)

    def run(cmd, **kwargs):
        calls.append(list(cmd))
        if on_call is not None:
            on_call(Path(kwargs["cwd"]), len(calls), cmd)
        assert queue, "unexpected model call"
        response = queue.pop(0)
        text = json.dumps(response)
        Path(cmd[cmd.index("-o") + 1]).write_text(text, encoding="utf-8")
        stream = [
            {"type": "thread.started", "thread_id": f"stub-{len(calls)}"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {
                "id": "answer", "type": "agent_message", "text": text,
            }},
            {"type": "turn.completed", "usage": USAGE},
        ]
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(json.dumps(row) for row in stream),
            stderr="",
        )

    monkeypatch.setattr("verifier.llm.codex.runtime._run_codex_process", run)

    def factory(state):
        return CodexRuntime(
            cwd=str(state), model="gpt-5.6-luna", effort="medium",
            service_tier="standard", compact_inspection=False,
            judger_prefetch=judger_prefetch,
        )

    return factory


def _run(
    tmp_path, monkeypatch, case_id="exists-alpha", responses=None,
    name="probe", *, on_call=None, judger_prefetch=False,
):
    calls = []
    expected = probe._case_spec(case_id)["expected_sufficient"]
    responses = responses or [_verdict(expected)]
    result = probe.run_probe(
        tmp_path / name,
        case_id=case_id,
        expected_source_fingerprint=probe._source_fingerprint(probe.SOURCE_ROOT),
        backend_factory=_codex_factory(
            monkeypatch, responses, calls, on_call=on_call,
            judger_prefetch=judger_prefetch,
        ),
        judger_prefetch=judger_prefetch,
    )
    return result, calls


@pytest.mark.parametrize("case_id", probe.CASES)
def test_prefetch_profile_runs_same_real_adapter_and_records_actual_exposure(
        tmp_path, monkeypatch, case_id):
    result, calls = _run(tmp_path, monkeypatch, case_id,
                         judger_prefetch=True)
    assert result["complete"] and result["semantic_pass"]
    assert result["judger_prefetch"] is True
    assert result["runtime_profile_accounting_complete"] is True
    assert result["prefetch_attached_attempts"] == 1
    assert result["fixture_adapter_executions"] == 1
    assert len(calls) == 1
    assert "VERIFIER-BOUND RAW EVIDENCE TRANSPORT" in calls[0][-1]
    assert result["full_e2e_acceptance"] is False
    assert all(value is None for value in result["excluded_stage_costs"].values())


def test_wrong_prefetch_profile_bill_is_inconclusive(tmp_path, monkeypatch):
    from verifier.llm.codex import attempts as cb

    original = cb._record_codex_attempt

    def wrong_bill(*args, **kwargs):
        kwargs["base"] = {**kwargs["base"], "judger_prefetch": False}
        return original(*args, **kwargs)

    monkeypatch.setattr(cb, "_record_codex_attempt", wrong_bill)
    result, calls = _run(tmp_path, monkeypatch, judger_prefetch=True)
    assert len(calls) == 1
    assert result["cost_report"]["complete"] is True
    assert result["runtime_profile_accounting_complete"] is False
    assert result["complete"] is False and result["semantic_pass"] is False


@pytest.mark.parametrize("invalid", [None, 0, 1, "1", [], {}])
def test_invalid_prefetch_profile_fails_before_creating_state(tmp_path, invalid):
    output = tmp_path / "probe"
    with pytest.raises(probe.ReplaySetupError, match="must be a bool"):
        probe.run_probe(output, case_id="exists-alpha",
                        expected_source_fingerprint="a" * 64,
                        judger_prefetch=invalid)
    assert not output.exists()


def test_injected_backend_prefetch_profile_cannot_disagree(tmp_path, monkeypatch):
    calls = []
    factory = _codex_factory(monkeypatch, [_verdict("yes")], calls)
    with pytest.raises(probe.ReplaySetupError, match="prefetch profile"):
        probe.run_probe(
            tmp_path / "probe", case_id="exists-alpha",
            expected_source_fingerprint=probe._source_fingerprint(probe.SOURCE_ROOT),
            backend_factory=factory, judger_prefetch=True)
    assert calls == []


def test_case_matrix_fixes_one_prefix_and_both_outcome_directions():
    specs = probe._case_specs()

    assert [spec["case_id"] for spec in specs] == [
        "exists-alpha", "exists-beta", "forall-alpha", "forall-beta",
    ]
    assert [spec["expected_sufficient"] for spec in specs] == list(EXPECTED)
    assert {spec["method"]["adapter"] for spec in specs} == {
        "ledger.tool-call-string"
    }
    assert {spec["method"]["adapter_version"] for spec in specs} == {2}
    assert {spec["method"]["adapter_inputs"][0]["ref"] for spec in specs} \
        == {"ledger:events"}
    assert all("adapter_input_binding" not in spec["method"] for spec in specs)
    arguments = [json.loads(spec["method"]["adapter_arguments"])
                 for spec in specs]
    assert {item["through_seq"] for item in arguments} == {1}
    assert [item["selector"] for item in arguments] == [
        "any", "any", "all", "all",
    ]
    assert all(item["predicate"] == "equals" for item in arguments)
    assert "alpha" not in probe._TASK and "beta" not in probe._TASK


@pytest.mark.parametrize("case_id,expected,outcome,witness_kind", [
    ("exists-alpha", "yes", "holds", "none"),
    ("exists-beta", "no", "refuted", "absence"),
    ("forall-alpha", "no", "holds", "none"),
    ("forall-beta", "yes", "refuted", "record_contradiction"),
])
def test_complete_local_codex_stub_retains_real_evidence_and_bills(
        tmp_path, monkeypatch, case_id, expected, outcome, witness_kind):
    def before_model(state, _index, cmd):
        start = (state.parent / "probe-start.json").read_text(encoding="utf-8")
        assert "expected_sufficient" not in start
        assert not (state.parent / "probe-result.json").exists()
        assert "expected_sufficient" not in " ".join(cmd)

    result, calls = _run(
        tmp_path, monkeypatch, case_id, name=case_id, on_call=before_model)

    assert result["complete"] is True
    assert result["semantic_pass"] is True
    assert result["observed_expectations_match"] is True
    assert result["test_only"] is True
    assert result["scope"] == probe.SCOPE
    assert result["full_e2e_acceptance"] is False
    assert result["compact_inspection"] is False
    assert result["excluded_stage_costs"] == {
        name: None for name in probe._EXCLUDED
    }
    assert len(calls) == 1
    assert result["cost_report"]["complete"] is True
    assert result["cost_report"]["metrics"]["started_attempts"] == 1
    assert result["cases"][0]["actual"]["sufficient"] == expected
    assert result["cases"][0]["adapter_outcome"] == outcome
    assert result["cases"][0]["witness_kind"] == witness_kind
    assert result["fixture_adapter_executions"] == 1
    assert result["input_unchanged"] is True
    assert result["subject_events_unchanged"] is True
    assert result["evidence_artifacts_unchanged"] is True

    state = tmp_path / case_id / "state"
    events = list(EventLog(str(state / "events.jsonl")).replay())
    assert [event["type"] for event in events[:2]] == ["tool_call", "tool_call"]
    assert [event["payload"]["input"]["value"] for event in events[:2]] \
        == ["alpha", "beta"]
    assert [event["type"] for event in events[2:]] == ["judger_verdict"]
    assert len(list((state / "artifacts").glob("*.json"))) == 1
    assert all("expected_sufficient" not in path.read_text(encoding="utf-8")
               for path in state.rglob("*") if path.is_file())


def test_method_and_adapter_artifact_hashes_are_deterministic_between_runs(
        tmp_path, monkeypatch):
    first, _ = _run(tmp_path, monkeypatch, name="first")
    first_pair = (first["cases"][0]["method_sha256"],
                  first["cases"][0]["evidence_artifact"])
    second, _ = _run(tmp_path, monkeypatch, name="second")

    assert (second["cases"][0]["method_sha256"],
            second["cases"][0]["evidence_artifact"]) == first_pair
    assert second["source_seal"] == first["source_seal"]
    assert second["subject_event_digests"] == first["subject_event_digests"]


def test_schema_and_outer_judger_retry_attempts_are_all_billed(
        tmp_path, monkeypatch):
    malformed = {
        "sufficient": "maybe", "faithful": "yes",
        "degradation": "n/a", "reason": "invalid enum",
    }
    # The first logical AgentRole exhausts both schema attempts. BindingGame's
    # independent Judger retry then succeeds against the unchanged artifact.
    responses = [malformed, malformed, _verdict("yes")]
    result, calls = _run(tmp_path, monkeypatch, responses=responses)

    assert result["complete"] is True and result["semantic_pass"] is True
    assert len(calls) == 3
    assert result["cost_report"]["metrics"]["started_attempts"] == 3
    events = list(EventLog(str(
        tmp_path / "probe" / "state" / "events.jsonl"
    )).replay())
    failures = [event for event in events
                if event["type"] == "judger_call_failed"]
    assert len(failures) == 1
    assert failures[0]["payload"]["will_retry"] is True


def test_positive_direction_requires_faithful_yes_not_only_sufficient_yes(
        tmp_path, monkeypatch):
    unfaithful = _verdict("yes") | {"faithful": "no"}
    result, calls = _run(tmp_path, monkeypatch, responses=[unfaithful])

    assert len(calls) == 1
    assert result["complete"] is True
    assert result["cases"][0]["actual"]["sufficient"] == "yes"
    assert result["cases"][0]["actual"]["faithful"] == "no"
    assert result["cases"][0]["direction_matches"] is False
    assert result["observed_expectations_match"] is False
    assert result["semantic_pass"] is False


def test_fixture_backend_is_allowed_but_unknown_cost_is_not_zero(tmp_path):
    backend = FixtureBackend({"judger": _verdict("yes")})
    result = probe.run_probe(
        tmp_path / "fixture",
        case_id="exists-alpha",
        expected_source_fingerprint=probe._source_fingerprint(probe.SOURCE_ROOT),
        backend_factory=lambda _state: backend,
    )

    assert result["test_only"] is True
    assert result["observed_expectations_match"] is True
    assert result["complete"] is False
    assert result["semantic_pass"] is False
    assert result["cost_report"]["complete"] is False
    metrics = result["cost_report"]["metrics"]
    assert metrics["started_attempts"] is None
    assert metrics["uncached_input_tokens"] is None


def test_subject_event_mutation_is_detected_outside_input_anchor(tmp_path):
    class TamperingBackend(FixtureBackend):
        def complete(self, role_name, prompt, variables):
            answer = super().complete(role_name, prompt, variables)
            if len(self.calls) == 1:
                path = tmp_path / "tampered" / "state" / "events.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0]["payload"]["input"]["value"] = "changed"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return answer

    backend = TamperingBackend({"judger": _verdict("yes")})
    result = probe.run_probe(
        tmp_path / "tampered",
        case_id="exists-alpha",
        expected_source_fingerprint=probe._source_fingerprint(probe.SOURCE_ROOT),
        backend_factory=lambda _state: backend,
    )

    assert result["subject_events_unchanged"] is False
    assert result["input_unchanged"] is False
    assert result["complete"] is False


def test_plan_and_paid_confirmation_are_nonexecuting_by_default(
        tmp_path, monkeypatch, capsys):
    output = tmp_path / "never-created"
    fingerprint = probe._source_fingerprint(probe.SOURCE_ROOT)
    args = ["--output-root", str(output),
            "--case", "exists-alpha",
            "--source-fingerprint", fingerprint]
    monkeypatch.setattr(
        probe, "run_probe",
        lambda *_args, **_kwargs: pytest.fail("model path started"),
    )

    assert probe.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["diagnostic_only"] is True
    assert plan["full_e2e_acceptance"] is False
    assert output.exists() is False
    with pytest.raises(SystemExit):
        probe.main([*args, "--execute"])


def test_wrong_source_fingerprint_rejected_before_output(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="operator pin"):
        probe.run_probe(
            output, case_id="exists-alpha",
            expected_source_fingerprint="f" * 64,
            backend_factory=lambda _state: FixtureBackend({}),
        )
    assert not output.exists()


def test_interrupt_preserves_terminal_record_and_raw_ledgers(tmp_path):
    class InterruptBackend(FixtureBackend):
        def complete(self, *_args, **_kwargs):
            raise KeyboardInterrupt

    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        probe.run_probe(
            output,
            case_id="exists-alpha",
            expected_source_fingerprint=probe._source_fingerprint(probe.SOURCE_ROOT),
            backend_factory=lambda _state: InterruptBackend({}),
        )
    terminal = json.loads((output / "probe-result.json").read_text())
    assert terminal["diagnostic_status"] == "inconclusive"
    assert terminal["error_class"] == "KeyboardInterrupt"
    assert (output / "usage.jsonl").exists()
    assert (output / "usage.jsonl.spans.jsonl").exists()
