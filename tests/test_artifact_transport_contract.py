"""The native transport description must match the real content address."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verifier.core.model import canonical, content_hash
from verifier.engine.method_cache import _backend_identity
from verifier.execution.instruments import ArtifactStore
from verifier.llm import codex_backend as cb


@pytest.mark.parametrize("value", [
    {}, {"z": 1, "a": 2}, {"text": "\u4e2d\U0001f600\n\t"},
    {"values": [None, True, False, 1, 1.0, -0.0, 1e-7, 1e20]},
    {"nested": {"b": "\\\"/", "a": ["\u2028", {"y": 3, "x": 4}]}},
])
def test_documented_calculation_matches_real_artifact_store(tmp_path, value):
    # Evaluate only a fixed verifier-owned expression, never model/user code.
    reference = eval(cb._ARTIFACT_DIGEST_EXPRESSION, {"__builtins__": {}},
                     {"hashlib": hashlib, "json": json, "value": value})
    store = ArtifactStore(str(tmp_path / "artifacts"))
    digest = store.put(value)
    assert reference == digest == content_hash(value)
    parsed = json.loads((Path(store.root) / (digest + ".json")).read_bytes())
    assert content_hash(parsed) == digest
    assert hashlib.sha256((canonical(parsed) + "\n").encode()).hexdigest() != digest


@pytest.mark.parametrize("role_name", ["judger", "oracles", "discharge", "critic", "obligations"])
def test_only_judger_receives_format_note_without_tool_or_schema_changes(tmp_path, monkeypatch, role_name):
    role = SimpleNamespace(system="fixed role rules", schema={"ok": bool})
    instance = cb.CodexBackend(roles={role_name: role}, cwd=str(tmp_path), model="gpt-5.6-luna")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        Path(command[command.index("-o") + 1]).write_text('{"ok":true}')
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(cb, "_run_codex_process", run)
    assert instance.complete(role_name, "fixed user content", {}) == {"ok": True}
    prompt = calls[0][-1]
    assert (cb._ARTIFACT_ADDRESS_NOTE in prompt) is (role_name == "judger")
    assert "fixed role rules\n\nfixed user content" in prompt
    assert "matching this JSON Schema" in prompt
    assert calls[0][calls[0].index("--model") + 1] == "gpt-5.6-luna"
    assert instance.effort == "medium"
    assert role.system == "fixed role rules"
    assert role.schema == {"ok": bool}


def test_runtime_transport_identity_is_frozen_per_backend_and_separates_cache(monkeypatch):
    before = cb.CodexBackend(model="gpt-5.6-luna")
    first = _backend_identity(before)
    monkeypatch.setattr(cb, "_ARTIFACT_ADDRESS_NOTE", cb._ARTIFACT_ADDRESS_NOTE + " revision")
    after = cb.CodexBackend(model="gpt-5.6-luna")
    assert _backend_identity(before) == first
    assert _backend_identity(after) != first
    assert set(_backend_identity(after)) == set(first)
    assert first["runtime_instruction_identity"] == before.runtime_instruction_identity


def test_format_note_does_not_authorize_semantic_shortcuts():
    assert "independently audit method sufficiency" in cb._ARTIFACT_ADDRESS_NOTE
    assert "A genuine mismatch remains a defect" in cb._ARTIFACT_ADDRESS_NOTE
    assert "never proves semantic correctness" in cb._ARTIFACT_ADDRESS_NOTE
