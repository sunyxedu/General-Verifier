"""Offline profile plumbing; all model calls are local stubs."""
import os
from types import SimpleNamespace

import pytest

from verifier import offline
from verifier.core.events import EventLog
from verifier.core.store import ClaimStore
from verifier.engine.kernel import Kernel
from verifier.execution.instruments import ArtifactStore
from verifier.llm import prompts
from verifier.llm.codex import command as codex_command
from verifier.llm.codex import runtime as cb
from verifier.llm.agents import COMPILE_OBLIGATIONS_AGENT, CRITIC_AGENT
from verifier.llm.authority_backend import authority_compilation_backend
from verifier.llm.schema import FixtureBackend


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
        offline.score(str(tmp_path), backend=cb.CodexRuntime(compact_inspection=True))


@pytest.mark.parametrize("enabled", [False, True])
def test_injected_backend_profile_mismatch_fails_before_call(tmp_path, enabled):
    backend = cb.CodexRuntime(cwd=str(tmp_path), compact_inspection=not enabled)
    with pytest.raises(ValueError, match="compact_inspection does not match"):
        offline.score(str(tmp_path), seats="codex", backend=backend,
                      compact_inspection=enabled)
    assert not (tmp_path / "events.jsonl").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_extractor_and_kernel_receive_same_explicit_profile(tmp_path, monkeypatch, enabled):
    from verifier.app import assembly

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

    monkeypatch.setattr(cb.CodexRuntime, "complete", complete)
    monkeypatch.setattr(assembly, "build_kernel", kernel)
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
    backend = cb.CodexRuntime(cwd=str(tmp_path), compact_inspection=enabled)
    identity = backend.runtime_instruction_identity
    monkeypatch.setattr(codex_command, "_COMPACT_INSPECTION_NOTE", "later module default")
    monkeypatch.setattr(codex_command, "_ARTIFACT_ADDRESS_NOTE", "later artifact note")
    monkeypatch.setattr(cb.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="codex-cli test"))
    with authority_compilation_backend(backend, COMPILE_OBLIGATIONS_AGENT, CRITIC_AGENT) as session:
        assert session.backend.compact_inspection is enabled
        assert session.backend.runtime_instruction_identity == identity
        assert session.obligations_role.tools == []
        assert session.critic_role.tools == []


def _prompt_kernel(tmp_path, compact):
    backend = FixtureBackend({})
    backend.compact_inspection = compact
    return Kernel(
        ClaimStore(),
        EventLog(os.path.join(tmp_path, "events.jsonl")),
        backend,
        ArtifactStore(os.path.join(tmp_path, "artifacts")),
        # Construction is all this prompt-wiring test needs.  Avoid creating
        # an execution seat that could obscure which two roles changed.
        discharger=SimpleNamespace(),
    )


def test_compact_profile_clones_only_compiler_and_critic_prompts(tmp_path):
    control = _prompt_kernel(tmp_path / "control", False)
    treatment = _prompt_kernel(tmp_path / "treatment", True)

    assert control.obligations_role is COMPILE_OBLIGATIONS_AGENT
    assert control.critic_role is CRITIC_AGENT
    assert treatment.obligations_role is not COMPILE_OBLIGATIONS_AGENT
    assert treatment.critic_role is not CRITIC_AGENT
    assert treatment.obligations_role.system == (
        COMPILE_OBLIGATIONS_AGENT.system
        + prompts.COMPACT_COMPLETENESS_COMPILER_ADDENDUM)
    assert treatment.critic_role.system == (
        CRITIC_AGENT.system + prompts.COMPACT_COMPLETENESS_CRITIC_ADDENDUM)

    # The treatment is a prompt representation change only.  Seat identity,
    # output schema, tools, and retry/turn limits remain exactly symmetric.
    for field in ("name", "template", "schema", "tools", "model", "max_turns"):
        assert getattr(treatment.obligations_role, field) == getattr(
            COMPILE_OBLIGATIONS_AGENT, field)
        assert getattr(treatment.critic_role, field) == getattr(
            CRITIC_AGENT, field)
    assert COMPILE_OBLIGATIONS_AGENT.system == prompts.OBLIGATIONS_SYSTEM
    assert CRITIC_AGENT.system == prompts.CRITIC_SYSTEM


def test_compact_contract_checklists_are_general_and_complete():
    compiler = prompts.COMPACT_COMPLETENESS_COMPILER_ADDENDUM
    critic = prompts.COMPACT_COMPLETENESS_CRITIC_ADDENDUM
    for text in (compiler, critic):
        for dimension in (
            "carrier", "membership", "item identity", "per-item",
            "quantity", "aggregation",
        ):
            assert dimension in text
        assert "terminal authority or addressing scheme" in text
        assert "SQLite" not in text
    assert "Emit every dimension the source requires and invent none" in compiler
    assert "Finish the necessity attack and the sufficiency attack independently" in critic
    assert "Finding one defect never ends the search" in critic
    assert "every independently surviving defect" in critic
