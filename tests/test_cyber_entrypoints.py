"""Runtime selection and MCP contracts, without launching any model."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.common import solver
from verifier import mcp_server
from verifier.profiles import get_profile
from verifier.profiles.cyber.evaluator import EvaluatorSpec


def tool_names(tmp_path, profile, evaluator_spec=None):
    state = tmp_path / profile
    state.mkdir()
    (state / "task.md").write_text("Check the task-defined security property.")
    service = mcp_server.Service(str(state), profile=profile,
        project_root=str(tmp_path), evaluator_spec=evaluator_spec,
        kernel_factory=lambda sc: SimpleNamespace())
    server = mcp_server.build_server(service)
    return {tool.name for tool in asyncio.run(server.list_tools())}


def test_mcp_exposes_one_original_verifier_with_optional_evidence_tools(tmp_path):
    ordinary = {"commit", "verify", "certificate", "status"}
    evidence = {"cyber_checkpoint", "cyber_register_candidate",
                "cyber_candidates", "cyber_finalize_candidate",
                "verification_job", "verification_jobs", "verification_retry"}
    assert tool_names(tmp_path, "default") == ordinary
    assert tool_names(tmp_path, "cyber") == ordinary | evidence
    second = tmp_path / "with-evaluator"
    second.mkdir()
    spec = EvaluatorSpec((str(Path(sys.executable).resolve()), "{candidate}"))
    assert tool_names(second, "cyber", spec) == ordinary | evidence | {
        "cyber_evaluate_candidate"}


def test_cyber_solver_protocol_preserves_revision_and_effect_mechanics():
    original = Path(mcp_server._PROTOCOL_PATH).read_text()
    assert mcp_server._solver_protocol(get_profile()) == original
    cyber = mcp_server._solver_protocol(get_profile("cyber"))
    assert cyber.startswith(original + "\n\n" + get_profile("cyber").solver_instructions)
    assert "stable job_id" in cyber and "verification_job(job_id)" in cyber
    assert "revises" in cyber and "effect:" in cyber


@pytest.mark.parametrize("changed", ["MODEL", "ANTHROPIC_BASE_URL"])
def test_cyber_state_pins_effective_sdk_defaults(tmp_path, monkeypatch, changed):
    from verifier.llm import provider
    args = dict(profile="cyber", project_root=str(tmp_path),
                kernel_factory=lambda sc: SimpleNamespace())
    service = mcp_server.Service(str(tmp_path / "state"), **args)
    assert service._profile_pin["runtime"]["effective"]["model"] == provider.MODEL
    service.store.db.close()
    monkeypatch.setattr(provider, changed, getattr(provider, changed) + "-changed")
    with pytest.raises(ValueError, match="different verification profile/runtime"):
        mcp_server.Service(str(tmp_path / "state"), **args)


def test_mount_config_preserves_default_and_selects_cyber_runtime(tmp_path):
    expected = ["run", "--project", solver.ROOT, "python", "-m",
        "verifier.mcp_server", "--state-dir", ".gv", "--toolchain-dir",
        solver.TOOLCHAIN]
    assert solver.mcp_json()["mcpServers"]["gv"]["args"] == expected
    command = [sys.executable, "{candidate}"]
    solver.mount_gv(str(tmp_path), "fixture-model", profile="cyber",
        seats="codex", effort="max", service_tier="fast",
        final_destination="answer.bin", evaluator_command=command,
        evaluator_pins=["/absolute/entrypoint.py"])
    config = json.loads((tmp_path / ".mcp.json").read_text())
    args = config["mcpServers"]["gv"]["args"]
    assert args[:len(expected)] == expected
    assert args[args.index("--model") + 1] == "fixture-model"
    assert "--cyber" in args
    assert args[args.index("--seats") + 1] == "codex"
    assert args[args.index("--service-tier") + 1] == "fast"
    assert args[args.index("--cyber-final-destination") + 1] == "answer.bin"
    assert json.loads(args[args.index("--cyber-evaluator-command-json") + 1]) == command
    assert args[args.index("--cyber-evaluator-pin") + 1] == "/absolute/entrypoint.py"
    assert not (tmp_path / "CLAUDE.md").exists()


@pytest.mark.parametrize("kwargs", [
    {"final_destination": "answer.bin"},
    {"evaluator_command": []}, {"seats": "other"}, {"effort": "max"},
    {"service_tier": "fast"},
    {"profile": "cyber", "evaluator_pins": ["/absolute/entrypoint.py"]},
])
def test_mount_rejects_invalid_configuration_before_writing(tmp_path, kwargs):
    with pytest.raises(ValueError):
        solver.mount_gv(str(tmp_path), **kwargs)
    assert not (tmp_path / ".mcp.json").exists()
    assert not (tmp_path / ".claude").exists()


def test_cli_selects_host_configuration_without_model_calls(tmp_path, monkeypatch):
    seen = {}
    def fake_service(state_dir, **kwargs):
        seen.update(state_dir=state_dir, **kwargs)
        return object()
    monkeypatch.setattr(mcp_server, "Service", fake_service)
    monkeypatch.setattr(mcp_server, "build_server", lambda service:
        SimpleNamespace(run=lambda: seen.update(ran=True)))
    monkeypatch.setattr(sys, "argv", ["gv", "--state-dir", str(tmp_path),
        "--cyber", "--seats", "codex", "--model", "fixture-model",
        "--effort", "max", "--service-tier", "fast",
        "--project-root", str(tmp_path), "--cyber-final-destination", "answer.bin",
        "--cyber-evaluator-command-json", json.dumps([
            str(Path(sys.executable).resolve()), "{candidate}"])])
    mcp_server.main()
    assert seen["profile"] == "cyber" and seen["seats"] == "codex"
    assert seen["effort"] == "max" and seen["service_tier"] == "fast"
    assert seen["evaluator_spec"].timeout is None
    assert seen["final_destination"] == "answer.bin" and seen["ran"] is True


@pytest.mark.parametrize("flags", [
    ["--effort", "max"], ["--service-tier", "fast"],
    ["--cyber-final-destination", "answer.bin"],
    ["--cyber-evaluator-command-json", "[]"],
    ["--cyber", "--cyber-evaluator-command-json", '{"command":"sh"}'],
])
def test_cli_rejects_incompatible_or_invalid_configuration(tmp_path, monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["gv", "--state-dir", str(tmp_path), *flags])
    with pytest.raises(SystemExit) as error:
        mcp_server.main()
    assert error.value.code == 2
