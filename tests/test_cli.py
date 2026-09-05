"""Exercise the installed entry point without any model credentials."""
import json
import os
import subprocess
import sys

import pytest


def credential_free_env():
    return {**os.environ, "GV_PROVIDER": "deepseek", "GV_MODEL": "test-model",
            "DEEPSEEK_API_KEY": "", "OPENROUTER_API_KEY": "", "GV_API_KEY": ""}


@pytest.mark.parametrize("command", ["help", "replay", "codex-config"])
def test_model_free_entry_points_need_no_provider_key(tmp_path, command):
    events = tmp_path / "events.jsonl"
    events.write_text("")
    args = {
        "help": ["-m", "verifier.cli", "--help"],
        "replay": ["-m", "verifier.cli", "replay", "--events", str(events)],
        "codex-config": ["-c", "from verifier.offline import config_fields; "
                         "assert config_fields(seats='codex')['seats'] == 'codex'"],
    }[command]
    result = subprocess.run([sys.executable, *args], env=credential_free_env(),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    if command == "replay":
        assert json.loads(result.stdout) == {}


def test_sdk_requires_credentials_when_it_starts():
    result = subprocess.run(
        [sys.executable, "-c", "from verifier.llm import provider; "
         "print('imported'); provider.claude_env()"],
        env=credential_free_env(), capture_output=True, text=True)
    assert result.stdout.strip() == "imported"
    assert result.returncode != 0
    assert "needs DEEPSEEK_API_KEY" in result.stderr


@pytest.mark.parametrize("verdict,exit_code", [
    ("verified", 0), ("supported", 1), ("unscoreable", 1),
])
def test_score_cli_forwards_codex_configuration(monkeypatch, capsys, verdict, exit_code):
    from verifier import cli, offline

    received = {}

    def score(state_dir, **kwargs):
        received.update(state_dir=state_dir, **kwargs)
        return {"verdict": verdict}

    monkeypatch.setattr(offline, "score", score)
    assert cli.main([
        "score", "--state-dir", "/tmp/example", "--seats", "codex",
        "--model", "test-model", "--effort", "high", "--service-tier", "fast",
        "--max-rounds", "2", "--critic-rounds", "4",
    ]) == exit_code
    assert received == {"state_dir": "/tmp/example", "seats": "codex",
                        "model": "test-model", "effort": "high",
                        "service_tier": "fast", "max_rounds": 2,
                        "critic_rounds": 4}
    assert json.loads(capsys.readouterr().out)["verdict"] == verdict


@pytest.mark.parametrize("option", [["--effort", "high"], ["--service-tier", "fast"]])
def test_score_cli_rejects_codex_options_for_sdk(option, capsys):
    from verifier import cli

    with pytest.raises(SystemExit) as error:
        cli.main(["score", "--state-dir", "/tmp/example", *option])
    assert error.value.code == 2
    assert "require --seats codex" in capsys.readouterr().err
