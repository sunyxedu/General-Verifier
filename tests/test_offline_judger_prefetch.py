"""Offline Judger-prefetch plumbing tests; all model calls are local stubs."""
import sys

import pytest

from experiments.common import scoring
from experiments.terminal_bench import score as tb_score
from verifier import offline
from verifier.llm import codex_backend as cb


def test_config_default_protocol_and_identity(monkeypatch):
    from verifier.engine.judger_prefetch import PREFETCH_PROTOCOL

    monkeypatch.setattr(offline, "pipeline_fingerprint", lambda: "fixed")
    default = offline.config_fields(seats="codex")
    explicit = offline.config_fields(seats="codex", judger_prefetch=False)
    enabled = offline.config_fields(seats="codex", judger_prefetch=True)

    assert default == explicit
    assert default["judger_prefetch"] is False
    assert default["judger_prefetch_protocol"] is None
    assert enabled["judger_prefetch"] is True
    assert enabled["judger_prefetch_protocol"] == PREFETCH_PROTOCOL
    assert default["config"] != enabled["config"]
    assert offline.config_id(
        seats="codex", judger_prefetch=True) == enabled["config"]
    ignored = {"config", "judger_prefetch", "judger_prefetch_protocol"}
    assert {k: v for k, v in default.items() if k not in ignored} == {
        k: v for k, v in enabled.items() if k not in ignored}


@pytest.mark.parametrize("invalid", [None, 0, 1, "1", [], {}])
def test_config_rejects_non_bool(invalid):
    with pytest.raises(ValueError, match="judger_prefetch must be a bool"):
        offline.config_fields(seats="codex", judger_prefetch=invalid)


def test_enabled_profile_requires_codex_seats():
    with pytest.raises(ValueError, match="only valid for Codex"):
        offline.config_fields(judger_prefetch=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_injected_backend_mismatch_fails_before_scoring(tmp_path, enabled):
    backend = cb.CodexBackend(cwd=str(tmp_path), judger_prefetch=not enabled)
    with pytest.raises(ValueError, match="judger_prefetch does not match"):
        offline.score(str(tmp_path), seats="codex", backend=backend,
                      judger_prefetch=enabled)
    assert not (tmp_path / "events.jsonl").exists()


def test_enabled_backend_cannot_hide_behind_sdk_config(tmp_path):
    backend = cb.CodexBackend(cwd=str(tmp_path), judger_prefetch=True)
    with pytest.raises(ValueError, match="judger_prefetch does not match"):
        offline.score(str(tmp_path), backend=backend)


@pytest.mark.parametrize("enabled", [False, True])
def test_extractor_and_kernel_backends_receive_same_flag(
        tmp_path, monkeypatch, enabled):
    from verifier import mcp_server

    (tmp_path / "task.md").write_text("Report the synthetic answer.")
    (tmp_path / "final.txt").write_text("synthetic answer")
    (tmp_path / "hook_feed.jsonl").write_text("")
    seen = []

    def complete(self, role_name, user, variables):
        assert role_name == "extract"
        seen.append(self)
        return {"commits": [{"label": "F", "kind": "final",
                             "text": "synthetic answer", "deps": [],
                             "revises": ""}]}

    class ReachedKernel(Exception):
        pass

    def kernel(sc, model, **kwargs):
        seen.append(kwargs["backend"])
        raise ReachedKernel()

    monkeypatch.setattr(cb.CodexBackend, "complete", complete)
    monkeypatch.setattr(mcp_server, "_default_kernel", kernel)
    with pytest.raises(ReachedKernel):
        offline.score(
            str(tmp_path), seats="codex", model="gpt-5.6-luna",
            effort="medium", service_tier="standard",
            compact_inspection=False, judger_prefetch=enabled)

    assert len(seen) == 2 and seen[0] is not seen[1]
    for backend in seen:
        assert backend.judger_prefetch is enabled
        assert backend.compact_inspection is False
        assert backend.model == "gpt-5.6-luna"
        assert backend.effort == "medium"
        assert backend.service_tier == "standard"
    assert seen[0].runtime_instruction_identity == \
        seen[1].runtime_instruction_identity


def test_score_many_forwards_flag(tmp_path, monkeypatch):
    seen = []

    def fake_score(state, **kwargs):
        seen.append((state, kwargs))
        return {"verdict": "unscoreable"}

    monkeypatch.setattr(offline, "score", fake_score)
    record = {"trial": "synthetic"}
    scoring.score_many(
        [(record, str(tmp_path / "state"))], str(tmp_path / "scores.jsonl"),
        workers=1, heartbeat=0, judger_prefetch=True)
    assert len(seen) == 1
    assert seen[0][1]["judger_prefetch"] is True


def test_terminal_bench_cli_rejects_enabled_profile_without_codex(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["score.py", "--judger-prefetch", "1"])
    with pytest.raises(SystemExit) as error:
        tb_score.main()
    assert error.value.code == 2
