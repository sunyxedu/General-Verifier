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
