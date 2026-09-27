"""The native transport description must match the real content address."""
import base64
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from verifier.execution import instruments as instrument_module
from verifier.core.model import canonical, content_hash
from verifier.llm.identity import backend_identity
from verifier.execution.instruments import ArtifactStore, validated_image_info
from verifier.llm.codex import command as codex_command
from verifier.llm.codex import runtime as cb


VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII=")


@pytest.mark.parametrize("value", [
    {}, {"z": 1, "a": 2}, {"text": "\u4e2d\U0001f600\n\t"},
    {"values": [None, True, False, 1, 1.0, -0.0, 1e-7, 1e20]},
    {"nested": {"b": "\\\"/", "a": ["\u2028", {"y": 3, "x": 4}]}},
])
def test_documented_calculation_matches_real_artifact_store(tmp_path, value):
    # Evaluate only a fixed verifier-owned expression, never model/user code.
    reference = eval(codex_command._ARTIFACT_DIGEST_EXPRESSION, {"__builtins__": {}},
                     {"hashlib": hashlib, "json": json, "value": value})
    store = ArtifactStore(str(tmp_path / "artifacts"))
    digest = store.put(value)
    assert reference == digest == content_hash(value)
    parsed = json.loads((Path(store.root) / (digest + ".json")).read_bytes())
    assert content_hash(parsed) == digest
    assert hashlib.sha256((canonical(parsed) + "\n").encode()).hexdigest() != digest


@pytest.mark.parametrize(("raw", "expected"), [
    (
        VALID_PNG,
        ("image/png", 1, 1),
    ),
    (
        b"\xff\xd8\xff\xc0\x00\x0b\x08\x00\x01\x00\x01"
        b"\x01\x01\x11\x00\xff\xd9",
        ("image/jpeg", 1, 1),
    ),
    (b"GIF89a\x01\x00\x01\x00\x00\x00\x00", ("image/gif", 1, 1)),
    (
        b"RIFF\x16\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00"
        + b"\x00" * 10,
        ("image/webp", 1, 1),
    ),
])
def test_allowlisted_image_headers_have_bounded_dimensions(raw, expected):
    assert validated_image_info(raw) == expected


def test_blob_publication_pins_directories_during_a_path_swap(
        tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    store = ArtifactStore(str(root))
    original_link = os.link
    raced = False

    def swap_then_link(source, destination, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            (root / "blobs").rename(root / "detached-blobs")
            (root / "blobs").symlink_to(attacker, target_is_directory=True)
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(instrument_module.os, "link", swap_then_link)

    with pytest.raises(ValueError, match="directories changed"):
        store.put_blob(b"fixed raw evidence")
    assert list(attacker.iterdir()) == []


@pytest.mark.parametrize("role_name", ["judger", "oracles", "discharge", "critic", "obligations"])
def test_only_judger_receives_format_note_without_tool_or_schema_changes(tmp_path, monkeypatch, role_name):
    role = SimpleNamespace(system="fixed role rules", schema={"ok": bool})
    instance = cb.CodexRuntime(roles={role_name: role}, cwd=str(tmp_path), model="gpt-5.6-luna")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        Path(command[command.index("-o") + 1]).write_text('{"ok":true}')
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(cb, "_run_codex_process", run)
    assert instance.complete(role_name, "fixed user content", {}) == {"ok": True}
    prompt = calls[0][-1]
    assert (codex_command._ARTIFACT_ADDRESS_NOTE in prompt) is (role_name == "judger")
    assert "fixed role rules\n\nfixed user content" in prompt
    assert "matching this JSON Schema" in prompt
    assert calls[0][calls[0].index("--model") + 1] == "gpt-5.6-luna"
    assert instance.effort == "medium"
    assert role.system == "fixed role rules"
    assert role.schema == {"ok": bool}


def test_runtime_transport_identity_is_frozen_per_backend_and_separates_cache(monkeypatch):
    before = cb.CodexRuntime(model="gpt-5.6-luna")
    first = backend_identity(before)
    monkeypatch.setattr(codex_command, "_ARTIFACT_ADDRESS_NOTE", codex_command._ARTIFACT_ADDRESS_NOTE + " revision")
    after = cb.CodexRuntime(model="gpt-5.6-luna")
    assert backend_identity(before) == first
    assert backend_identity(after) != first
    assert set(backend_identity(after)) == set(first)
    assert first["runtime_instruction_identity"] == before.runtime_instruction_identity


def test_format_note_does_not_authorize_semantic_shortcuts():
    assert "independently audit method sufficiency" in codex_command._ARTIFACT_ADDRESS_NOTE
    assert "A genuine mismatch remains a defect" in codex_command._ARTIFACT_ADDRESS_NOTE
    assert "never proves semantic correctness" in codex_command._ARTIFACT_ADDRESS_NOTE
