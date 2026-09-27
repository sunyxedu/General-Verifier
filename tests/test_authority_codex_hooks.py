"""Codex authority sessions cannot bypass their candidate/tool boundary."""
import json
from types import SimpleNamespace

import pytest

from verifier.llm.authority_backend import (
    AuthorityBackendUnavailable,
    authority_session,
)
from verifier.llm.codex_backend import CodexRuntime, codex_usage_ledger


def _stream(*items, message='{"ok": true}'):
    events = [
        {"type": "thread.started", "thread_id": "authority-thread"},
        {"type": "turn.started"},
        *[{"type": "item.completed", "item": item} for item in items],
        {"type": "item.completed", "item": {
            "type": "agent_message", "text": message,
        }},
        {"type": "turn.completed", "usage": {
            "input_tokens": 100,
            "cached_input_tokens": 60,
            "output_tokens": 12,
            "reasoning_output_tokens": 7,
        }},
    ]
    return "\n".join(json.dumps(event) for event in events) + "\n"


def _backend(original_cwd, scratch):
    backend = CodexRuntime(
        roles={"obligations": SimpleNamespace(
            system="", schema={"ok": bool})},
        cwd=str(original_cwd),
        session=authority_session(str(scratch)),
    )
    return backend


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_forced_json_falls_back_to_validated_agent_message_without_ledger(
    tmp_path, monkeypatch,
):
    original = tmp_path / "candidate-state"
    scratch = tmp_path / "empty-authority-scratch"
    original.mkdir()
    scratch.mkdir()
    backend = _backend(original, scratch)
    observed = {}

    def fake_run(cmd, **kwargs):
        observed["cmd"] = cmd
        observed["cwd"] = kwargs["cwd"]
        # Deliberately leave -o empty: the validated JSONL agent message is
        # the only acceptable fallback, never the full event stream.
        return SimpleNamespace(returncode=0, stderr="", stdout=_stream())

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)

    assert backend.complete("obligations", "authority only", {}) == {"ok": True}
    assert "--json" in observed["cmd"]
    assert observed["cwd"] == str(scratch)
    assert str(original) not in "\n".join(observed["cmd"])


def test_valid_output_file_cannot_bypass_stream_gate(tmp_path, monkeypatch):
    original = tmp_path / "candidate-state"
    scratch = tmp_path / "empty-authority-scratch"
    original.mkdir()
    scratch.mkdir()
    backend = _backend(original, scratch)

    def fake_run(cmd, **_kwargs):
        with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(returncode=0, stderr="", stdout=_stream({
            "type": "command_execution", "command": "read candidate",
        }))

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)

    with pytest.raises(
        AuthorityBackendUnavailable, match="tool or unknown item",
    ):
        backend.complete("obligations", "authority only", {})


def test_rejected_stream_is_charged_against_original_usage_context(
    tmp_path, monkeypatch,
):
    original = tmp_path / "candidate-state"
    scratch = tmp_path / "empty-authority-scratch"
    usage_path = tmp_path / "usage.jsonl"
    original.mkdir()
    scratch.mkdir()
    backend = _backend(original, scratch)
    observed = {}

    def fake_run(cmd, **kwargs):
        observed["cwd"] = kwargs["cwd"]
        with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(returncode=0, stderr="", stdout=_stream({
            "type": "web_search", "query": "candidate answer",
        }))

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    with codex_usage_ledger(
        str(usage_path), {str(original): {"trial": "trial-authority"}},
    ):
        with pytest.raises(AuthorityBackendUnavailable):
            backend.complete("obligations", "authority only", {})

    row, = _rows(usage_path)
    assert observed["cwd"] == str(scratch)
    assert row["trial"] == "trial-authority"
    assert row["state_dir"] == str(original)
    assert row["attempt_status"] == "output_error"
    assert row["error_kind"] == "attempt_stream_policy"
    assert row["usage_complete"] is True
    assert row["input_tokens"] == 100


def test_authority_command_and_prompt_do_not_enable_or_advertise_web(
    tmp_path, monkeypatch,
):
    original = tmp_path / "candidate-state"
    scratch = tmp_path / "empty-authority-scratch"
    original.mkdir()
    scratch.mkdir()
    backend = _backend(original, scratch)
    observed = {}

    def fake_run(cmd, **_kwargs):
        observed["cmd"] = cmd
        with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(returncode=0, stderr="", stdout=_stream())

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)

    assert backend.complete("obligations", "authority only", {}) == {"ok": True}
    command = observed["cmd"]
    instruction = command[-1]
    config_values = [
        command[index + 1] for index, value in enumerate(command[:-1])
        if value == "-c"
    ]
    assert "WEB SEARCH (knowledge channel only)" not in instruction
    assert "tools.web_search=true" not in config_values
    assert "tools.web_search=false" in config_values
    assert 'web_search="disabled"' in config_values

