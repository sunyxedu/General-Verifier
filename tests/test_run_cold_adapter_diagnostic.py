import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.terminal_bench import run_cold_adapter_diagnostic as runner
from experiments.terminal_bench import prepare_shared_contracts as shared_preparer
from experiments.terminal_bench.run_typed_adapter_eval import _source_fingerprint
from test_contract_replay import COMMITS
from test_prepare_contract_entry import PASS, PAYLOAD, harness
from verifier import offline


def _metrics(attempts, *, uncached=None, cached=None, output=None, wall=1.0,
             complete=True):
    return {
        "complete": complete,
        "cost_complete": complete,
        "wall_complete": True,
        "started_attempts": attempts if complete else None,
        "uncached_input_tokens": (
            attempts * 10 if uncached is None and complete else uncached),
        "cached_input_tokens": (
            attempts * 20 if cached is None and complete else cached),
        "output_tokens": attempts * 3 if output is None and complete else output,
        "wall_seconds": wall,
    }


def _request(tmp_path, *, source=None, source_seal="a" * 64,
             source_fingerprint="b" * 64):
    if source is None:
        source = tmp_path / "source"
        source.mkdir()
    source = Path(source)
    return {
        "schema": runner.SCHEMA,
        "run_id": "cold-local",
        "setup_id": "cold-local-setup",
        "source": str(source.resolve()),
        "source_seal": source_seal,
        "source_root": str(runner.SOURCE_ROOT),
        "source_fingerprint": source_fingerprint,
        "output": str((tmp_path / "output").resolve()),
        "runtime_path": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "runtime_read_roots": [
            "/opt/homebrew/bin", "/opt/homebrew/Cellar",
            "/opt/homebrew/opt",
        ],
        "max_concurrent_seats": 12,
        "model": "gpt-5.6-luna",
        "effort": "medium",
        "service_tier": "standard",
    }


def test_default_is_plan_only_and_execution_requires_exact_confirmation(
        tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    source.mkdir()
    fingerprint = "b" * 64
    monkeypatch.setattr(runner, "_validate_execute_output_root", lambda _p: None)
    monkeypatch.setattr(runner, "_source_fingerprint", lambda _p: fingerprint)
    monkeypatch.setattr(
        runner.offline, "verify_immutable_input_anchor", lambda _p: "a" * 64)
    monkeypatch.setattr(
        runner, "offline_runtime_read_roots",
        lambda _env: ("/opt/homebrew/bin",))
    called = []
    monkeypatch.setattr(
        runner, "run", lambda request: called.append(request) or {
            "comparison_complete": True, "result_equivalent": True})

    common = [
        "--source", str(source),
        "--source-fingerprint", fingerprint,
        "--runtime-path", "/opt/homebrew/bin:/usr/bin:/bin",
        "--runtime-read-root", "/opt/homebrew/bin",
    ]
    plan_output = tmp_path / "plan-output"
    assert runner.main([*common, "--output", str(plan_output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["schema"] == runner.SCHEMA
    assert plan["output"] == str(plan_output)
    assert called == []
    assert not plan_output.exists()

    with pytest.raises(SystemExit):
        runner.main([
            *common, "--output", str(tmp_path / "unconfirmed"), "--execute"])
    assert called == []

    execute_output = tmp_path / "confirmed"
    assert runner.main([
        *common, "--output", str(execute_output), "--execute",
        "--confirm-paid", "GPT-5.6-Luna",
    ]) == 0
    assert len(called) == 1


def test_worker_environments_clear_ambient_flags_and_arms_differ_only_by_adapter(
        tmp_path, monkeypatch):
    request = _request(tmp_path)
    monkeypatch.setenv("GV_SHARED_DIR", "/ambient/cache")
    monkeypatch.setenv("GV_BROWSER_OBSERVER", "/ambient/browser")
    monkeypatch.setenv("GV_PARALLEL_CLAIMS", "99")
    monkeypatch.setenv("PYTHONPATH", "/ambient/python")

    setup = runner._environment(request, "setup")
    control = runner._environment(request, "control")
    treatment = runner._environment(request, "treatment")

    assert setup == treatment
    differing = {
        key for key in set(control) | set(treatment)
        if control.get(key) != treatment.get(key)
    }
    assert differing == {"GV_TYPED_ADAPTERS"}
    assert control["GV_TYPED_ADAPTERS"] == "0"
    assert treatment["GV_TYPED_ADAPTERS"] == "1"
    assert control["GV_CONTRACT_CACHE"] == "1"
    assert control["TZ"] == "Europe/London"
    assert control["GV_OFFLINE_RUNTIME_READ_ROOTS"] == json.dumps(
        request["runtime_read_roots"])
    assert not any(key in control for key in (
        "GV_SHARED_DIR", "GV_BROWSER_OBSERVER", "GV_PARALLEL_CLAIMS",
        "PYTHONPATH"))


def test_three_real_stage_wrappers_charge_process_makespans(
        tmp_path, monkeypatch):
    request = _request(tmp_path)
    root = Path(request["output"])
    stage_cost = {"setup": 5, "control": 4, "treatment": 2}
    environments = {}

    def process(command, **kwargs):
        stage = command[command.index("--worker") + 1]
        assert kwargs["start_new_session"] is True
        environments[stage] = kwargs["env"]
        destination = root / stage
        destination.mkdir()
        if stage == "setup":
            value = {"complete": True, "metrics": _metrics(stage_cost[stage])}
            name = "setup-result.json"
        else:
            value = {
                "complete": True, "verification_complete": True,
                "accounting_complete": True, "diagnostic_status": "complete",
                "cost_report": {"metrics": _metrics(stage_cost[stage])},
            }
            name = "replay-result.json"
        (destination / name).write_text(json.dumps(value), encoding="utf-8")
        return SimpleNamespace(pid=321, wait=lambda timeout=None: 0)

    ticks = iter((0, 1_000_000_000, 3_000_000_000,
                  4_000_000_000, 9_000_000_000,
                  10_000_000_000, 14_000_000_000, 15_000_000_000))
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: next(ticks))
    monkeypatch.setattr(runner.subprocess, "Popen", process)
    monkeypatch.setattr(
        runner, "_source_fingerprint", lambda _p: request["source_fingerprint"])
    monkeypatch.setattr(
        runner.offline, "verify_immutable_input_anchor",
        lambda _p: request["source_seal"])
    monkeypatch.setattr(
        runner, "_compare_features", lambda _output, _records: {
            "available": True, "same_verdict": True,
            "same_verdict_strength": True, "same_semantic_features": True,
            "same_rank_key": True, "same_final_contract": True,
            "same_ordered_final_outcomes": True, "zero_loss_proven": False})
    monkeypatch.setattr(
        runner, "_recorded_config", lambda _request, _stage: {})
    monkeypatch.setattr(
        runner, "_admit_setup", lambda _request, **_kwargs: ({}, b"bundle"))
    monkeypatch.setattr(
        runner, "_check_arm_bills", lambda _request, _stage, _result: None)

    result = runner.run(request)

    assert result["comparison_complete"] is True
    assert result["result_equivalent"] is True
    assert [result["stages"][stage]["wall_seconds"]
            for stage in runner.STAGES] == [2.0, 5.0, 4.0]
    assert result["coordinator_wall_seconds"] == 15.0
    assert result["charges"]["charged"]["control"][
        "started_attempts"] == 9
    assert result["charges"]["charged"]["treatment"][
        "started_attempts"] == 7
    assert result["charges"]["physical"]["started_attempts"] == 11
    assert environments["setup"] == environments["treatment"]
    assert environments["control"]["GV_TYPED_ADAPTERS"] == "0"
    assert environments["treatment"]["GV_TYPED_ADAPTERS"] == "1"


@pytest.mark.parametrize("accounting_complete", [None, False])
def test_complete_replay_without_complete_accounting_is_inconclusive(
        tmp_path, monkeypatch, accounting_complete):
    request = _request(tmp_path)
    root = Path(request["output"])
    root.mkdir()

    def process(command, **kwargs):
        assert command[command.index("--worker") + 1] == "control"
        result = {
            "complete": True, "verification_complete": True,
            "diagnostic_status": "complete",
            "cost_report": {"metrics": _metrics(4)},
        }
        if accounting_complete is not None:
            result["accounting_complete"] = accounting_complete
        destination = root / "control"
        destination.mkdir()
        (destination / "replay-result.json").write_text(
            json.dumps(result), encoding="utf-8")
        return SimpleNamespace(pid=322, wait=lambda timeout=None: 0)

    ticks = iter((0, 2_000_000_000))
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: next(ticks))
    monkeypatch.setattr(runner.subprocess, "Popen", process)
    monkeypatch.setattr(
        runner, "_source_fingerprint", lambda _p: request["source_fingerprint"])
    monkeypatch.setattr(
        runner.offline, "verify_immutable_input_anchor",
        lambda _p: request["source_seal"])
    monkeypatch.setattr(
        runner, "_check_arm_bills",
        lambda *_args, **_kwargs: pytest.fail("incomplete accounting was audited as complete"))

    record = runner._run_stage(request, "control")

    assert record["status"] == "inconclusive"
    assert record["worker_result_sha256"] is None
    assert "worker_result_unavailable" in record["findings"]
    assert record["metrics"]["started_attempts"] is None


def test_failed_setup_does_not_launch_arms_and_retains_its_cost(
        tmp_path, monkeypatch):
    request = _request(tmp_path)
    launched = []

    def stage(_request, name):
        launched.append(name)
        assert name == "setup"
        return {"stage": name, "status": "inconclusive",
                "metrics": _metrics(7), "findings": ["setup_failed"]}

    monkeypatch.setattr(runner, "_run_stage", stage)
    result = runner.run(request)

    assert launched == ["setup"]
    assert result["comparison_complete"] is False
    assert result["stages"]["control"]["status"] == "not_started"
    assert result["stages"]["treatment"]["status"] == "not_started"
    assert result["charges"]["physical"]["started_attempts"] == 7
    assert result["charges"]["charged"]["control"][
        "started_attempts"] == 7
    assert result["charges"]["charged"]["treatment"][
        "started_attempts"] == 7


def test_control_failure_still_runs_treatment_and_is_overall_inconclusive(
        tmp_path, monkeypatch):
    request = _request(tmp_path)
    launched = []

    def stage(_request, name):
        launched.append(name)
        status = "inconclusive" if name == "control" else "complete"
        return {"stage": name, "status": status,
                "metrics": _metrics({"setup": 5, "control": 3,
                                      "treatment": 2}[name])}

    monkeypatch.setattr(runner, "_run_stage", stage)
    result = runner.run(request)

    assert launched == ["setup", "control", "treatment"]
    assert result["comparison_complete"] is False
    assert result["stages"]["treatment"]["status"] == "complete"
    assert result["charges"]["physical"]["started_attempts"] == 10


def test_unknown_arm_ledger_is_never_coerced_to_zero(tmp_path, monkeypatch):
    request = _request(tmp_path)

    def stage(_request, name):
        if name == "control":
            return {"stage": name, "status": "inconclusive",
                    "metrics": _metrics(None, uncached=None, cached=None,
                                        output=None, complete=False)}
        return {"stage": name, "status": "complete",
                "metrics": _metrics(5 if name == "setup" else 2)}

    monkeypatch.setattr(runner, "_run_stage", stage)
    result = runner.run(request)

    assert result["comparison_complete"] is False
    assert result["stages"]["control"]["metrics"][
        "started_attempts"] is None
    assert result["charges"]["complete"] is False
    assert result["charges"]["charged"]["control"][
        "started_attempts"] is None
    assert result["charges"]["physical"]["started_attempts"] is None


def _admission_fixture(tmp_path, monkeypatch):
    local = harness(
        tmp_path, monkeypatch,
        responses=[{"commits": COMMITS}, PAYLOAD, PASS, PAYLOAD, PASS])
    config = offline.config_fields(
        model="gpt-5.6-luna", seats="codex", effort="medium",
        service_tier="standard")
    fingerprint = _source_fingerprint(shared_preparer.SOURCE_ROOT)
    seal = offline.verify_immutable_input_anchor(str(local.source))
    request = _request(
        tmp_path, source=local.source, source_seal=seal,
        source_fingerprint=fingerprint)
    root = Path(request["output"]) / "setup"
    result = shared_preparer.prepare_shared_contracts(
        local.source, root, setup_id=request["setup_id"], config=config,
        expected_source_fingerprint=fingerprint,
        backend_factory=lambda state: shared_preparer.codex_backend(state, config),
        service_factory=lambda state: shared_preparer.codex_service(state, config))
    assert result["complete"], result["findings"]
    assert len(local.calls) == 5
    # Factory injection deliberately marks local fixtures test-only. Convert
    # only those three terminal operator markers so the admission happy path
    # is exercised without weakening production's test-only refusal.
    terminal = {}
    for name in ("setup-state.json", "setup-result.json", "bundle-pin.json"):
        path = root / name
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["test_only"] is True
        value["test_only"] = False
        offline._write_json_atomic(str(path), value)
        terminal[name] = value
    monkeypatch.setattr(runner, "_config", lambda: copy.deepcopy(config))
    return (request, root, terminal["setup-state.json"],
            terminal["setup-result.json"], terminal["bundle-pin.json"])


def test_setup_admission_accepts_complete_independently_pinned_local_fixture(
        tmp_path, monkeypatch):
    request, _root, _state, result, _pin = _admission_fixture(
        tmp_path, monkeypatch)
    admitted, bundle = runner._admit_setup(request)

    assert admitted == result
    assert bundle


@pytest.mark.parametrize("attack", ["test_only", "missing_pin", "mismatch"])
def test_setup_admission_rejects_untrusted_or_incomplete_terminal_records(
        tmp_path, monkeypatch, attack):
    request, root, state, result, pin = _admission_fixture(
        tmp_path, monkeypatch)
    if attack == "test_only":
        result["test_only"] = True
        offline._write_json_atomic(str(root / "setup-result.json"), result)
    elif attack == "missing_pin":
        (root / "bundle-pin.json").unlink()
    else:
        pin["bundle_sha256"] = "f" * 64
        offline._write_json_atomic(str(root / "bundle-pin.json"), pin)

    with pytest.raises((OSError, ValueError)):
        runner._admit_setup(request)
