"""Offline profile plumbing; all model calls are local stubs."""
from types import SimpleNamespace

import pytest

from verifier import offline
from verifier.llm import codex_backend as cb
from verifier.llm.agents import COMPILE_OBLIGATIONS_AGENT, CRITIC_AGENT
from verifier.llm.authority_backend import authority_compilation_backend


def test_config_explicit_default_and_profile_separation(monkeypatch):
    monkeypatch.setattr(offline, "pipeline_fingerprint", lambda: "fixed")
    default = offline.config_fields(seats="codex")
    explicit = offline.config_fields(seats="codex", compact_inspection=False)
    enabled = offline.config_fields(seats="codex", compact_inspection=True)
    assert default == explicit
    assert default["compact_inspection"] is False
    assert enabled["compact_inspection"] is True
    assert default["config"] != enabled["config"]
    assert offline.config_id(seats="codex", compact_inspection=True) == enabled["config"]
    assert {k: v for k, v in default.items() if k not in {"config", "compact_inspection"}} == {
        k: v for k, v in enabled.items() if k not in {"config", "compact_inspection"}}


@pytest.mark.parametrize("invalid", [None, 0, 1, "1", [], {}])
def test_config_rejects_non_bool(invalid):
    with pytest.raises(ValueError, match="must be a bool"):
        offline.config_fields(seats="codex", compact_inspection=invalid)


def test_profile_cannot_be_advertised_for_sdk():
    with pytest.raises(ValueError, match="only valid for Codex"):
        offline.config_fields(compact_inspection=True)


def test_injected_backend_cannot_hide_enabled_profile_behind_sdk_stamp(tmp_path):
    with pytest.raises(ValueError, match="compact_inspection does not match"):
        offline.score(str(tmp_path), backend=cb.CodexBackend(compact_inspection=True))


@pytest.mark.parametrize("enabled", [False, True])
def test_injected_backend_profile_mismatch_fails_before_call(tmp_path, enabled):
    backend = cb.CodexBackend(cwd=str(tmp_path), compact_inspection=not enabled)
    with pytest.raises(ValueError, match="compact_inspection does not match"):
        offline.score(str(tmp_path), seats="codex", backend=backend,
                      compact_inspection=enabled)
    assert not (tmp_path / "events.jsonl").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_extractor_and_kernel_receive_same_explicit_profile(tmp_path, monkeypatch, enabled):
    from verifier import mcp_server

    (tmp_path / "task.md").write_text("Report the synthetic answer.")
    (tmp_path / "final.txt").write_text("synthetic answer")
    (tmp_path / "hook_feed.jsonl").write_text("")
    seen = []

    def complete(self, role_name, user, variables):
        assert role_name == "extract"
        seen.append(self)
        return {"commits": [{"label": "F", "kind": "final",
                             "text": "synthetic answer", "deps": [], "revises": ""}]}

    class ReachedKernel(Exception):
        pass

    def kernel(sc, model, **kwargs):
        seen.append(kwargs["backend"])
        assert kwargs["critic_rounds"] == 2
        raise ReachedKernel()

    monkeypatch.setattr(cb.CodexBackend, "complete", complete)
    monkeypatch.setattr(mcp_server, "_default_kernel", kernel)
    with pytest.raises(ReachedKernel):
        offline.score(str(tmp_path), seats="codex", model="gpt-5.6-luna",
                      effort="medium", service_tier="standard", critic_rounds=2,
                      compact_inspection=enabled)
    assert len(seen) == 2
    assert seen[0] is not seen[1]
    for backend in seen:
        assert backend.compact_inspection is enabled
        assert backend.model == "gpt-5.6-luna"
        assert backend.effort == "medium"
        assert backend.service_tier == "standard"
    assert seen[0].runtime_instruction_identity == seen[1].runtime_instruction_identity


@pytest.mark.parametrize("enabled", [False, True])
def test_authority_clone_preserves_profile_and_frozen_identity(tmp_path, monkeypatch, enabled):
    import verifier.llm.authority_backend as ab

    backend = cb.CodexBackend(cwd=str(tmp_path), compact_inspection=enabled)
    identity = backend.runtime_instruction_identity
    monkeypatch.setattr(cb, "_COMPACT_INSPECTION_NOTE", "later module default")
    monkeypatch.setattr(cb, "_ARTIFACT_ADDRESS_NOTE", "later artifact note")
    monkeypatch.setattr(ab.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="codex-cli test"))
    with authority_compilation_backend(backend, COMPILE_OBLIGATIONS_AGENT, CRITIC_AGENT) as session:
        assert session.backend.compact_inspection is enabled
        assert session.backend.runtime_instruction_identity == identity
        assert session.obligations_role.tools == []
        assert session.critic_role.tools == []
