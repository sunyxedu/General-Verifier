"""Every codex seat call leaves a timing row in <state>/role_calls.jsonl,
beside the direct seats' rows, so a latency breakdown times the oracle,
executor and judger runs instead of inferring them from ledger gaps."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import verifier.llm.codex_backend as cb


def rows(tmp_path):
    path = tmp_path / "role_calls.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def backend(tmp_path, monkeypatch, fail=False):
    role = SimpleNamespace(system="fixed role rules", schema={"ok": bool})
    instance = cb.CodexRuntime(roles={"judger": role}, cwd=str(tmp_path), model="gpt-5.6-luna")

    def run(command, **kwargs):
        if fail:
            raise RuntimeError("seat process died")
        Path(command[command.index("-o") + 1]).write_text('{"ok":true}')
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(cb, "_run_codex_process", run)
    return instance


def test_a_seat_call_is_timed(tmp_path, monkeypatch):
    instance = backend(tmp_path, monkeypatch)
    assert instance.complete("judger", "content", {"claim_text": "the claim",
                                                   "obligation_text": "the obligation"}) == {"ok": True}
    row, = rows(tmp_path)
    assert row["role"] == "judger" and row["backend"] == "codex" and row["model"] == "gpt-5.6-luna"
    assert row["finished"] >= row["started"] and row["seconds"] >= 0 and row["error"] is None
    assert row["claim_text"] == "the claim" and row["obligation_text"] == "the obligation"


def test_a_failed_seat_call_is_timed_with_its_error(tmp_path, monkeypatch):
    instance = backend(tmp_path, monkeypatch, fail=True)
    with pytest.raises(Exception):
        instance.complete("judger", "content", {})
    row, = rows(tmp_path)
    assert row["role"] == "judger" and row["error"]
