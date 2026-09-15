"""Anthropic first-party provider wiring; subprocess isolation for import-time env."""
import json
import os
import subprocess
import sys

import pytest

PROVIDER_PROBE = ("import json; from verifier.llm import provider; "
                  "print(json.dumps({'PROVIDER': provider.PROVIDER, 'MODEL': provider.MODEL,"
                  " 'LITELLM_MODEL': provider.LITELLM_MODEL,"
                  " 'CONFIG_DIR': provider.CONFIG_DIR, 'CLAUDE_ENV': provider.CLAUDE_ENV}))")
CLAUDE_ENV_PROBE = ("import json; from verifier.llm import provider; "
                    "print(json.dumps(provider.claude_env(base={})))")


def anthropic_env(tmp_path, **extra):
    env = {**os.environ, "GV_PROVIDER": "anthropic", "GV_MODEL": "",
           "GV_CLAUDE_CONFIG_DIR": "", "CLAUDE_CONFIG_DIR": "",
           "DEEPSEEK_API_KEY": "", "OPENROUTER_API_KEY": "", "GV_API_KEY": "",
           "ANTHROPIC_BASE_URL": "", "ANTHROPIC_AUTH_TOKEN": "", "ANTHROPIC_API_KEY": ""}
    env.update(extra)
    return env


def login(tmp_path, name="config"):
    config = tmp_path / name
    config.mkdir()
    (config / ".credentials.json").write_text('{"fixture":"oauth-sentinel"}')
    return config


def probe(env, code=PROVIDER_PROBE):
    result = subprocess.run([sys.executable, "-B", "-c", code], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_anthropic_env_carries_no_base_url_or_token_overrides(tmp_path):
    env = anthropic_env(tmp_path, GV_CLAUDE_CONFIG_DIR=str(login(tmp_path)))
    values = probe(env)
    assert values["PROVIDER"] == "anthropic"
    assert values["MODEL"] == "claude-sonnet-5"
    assert values["LITELLM_MODEL"] == "anthropic/claude-sonnet-5"
    claude_env = values["CLAUDE_ENV"]
    assert "ANTHROPIC_BASE_URL" not in claude_env
    assert "ANTHROPIC_AUTH_TOKEN" not in claude_env
    assert "ANTHROPIC_API_KEY" not in claude_env
    assert claude_env["ANTHROPIC_MODEL"] == "claude-sonnet-5"
    assert claude_env["CLAUDE_CODE_SUBAGENT_MODEL"] == "claude-sonnet-5"
    assert claude_env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


def test_config_dir_precedence_is_override_then_ambient_then_login(tmp_path):
    override = login(tmp_path, "override")
    ambient = login(tmp_path, "ambient")
    env = anthropic_env(tmp_path, GV_CLAUDE_CONFIG_DIR=str(override),
                        CLAUDE_CONFIG_DIR=str(ambient))
    assert probe(env)["CONFIG_DIR"] == str(override)
    env = anthropic_env(tmp_path, CLAUDE_CONFIG_DIR=str(ambient))
    assert probe(env)["CONFIG_DIR"] == str(ambient)
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    env = anthropic_env(tmp_path, HOME=str(home))
    assert probe(env)["CONFIG_DIR"] == str(home / ".claude")


def test_claude_env_requires_the_oauth_credentials_file(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    env = anthropic_env(tmp_path, GV_CLAUDE_CONFIG_DIR=str(empty))
    result = subprocess.run([sys.executable, "-B", "-c", CLAUDE_ENV_PROBE], env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert ".credentials.json" in result.stderr
    assert "oauth-sentinel" not in result.stderr


def test_claude_env_exports_the_login_config_dir_without_secrets(tmp_path):
    config = login(tmp_path)
    env = anthropic_env(tmp_path, GV_CLAUDE_CONFIG_DIR=str(config))
    spawned = probe(env, CLAUDE_ENV_PROBE)
    assert spawned["CLAUDE_CONFIG_DIR"] == str(config)
    assert spawned["ANTHROPIC_MODEL"] == "claude-sonnet-5"
    assert "ANTHROPIC_BASE_URL" not in spawned and "ANTHROPIC_AUTH_TOKEN" not in spawned
    assert "oauth-sentinel" not in json.dumps(spawned)


def test_explicit_api_key_remains_an_opt_in_fallback(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    env = anthropic_env(tmp_path, GV_CLAUDE_CONFIG_DIR=str(empty), GV_API_KEY="fixture-key")
    spawned = probe(env, CLAUDE_ENV_PROBE)
    assert spawned["ANTHROPIC_API_KEY"] == "fixture-key"
