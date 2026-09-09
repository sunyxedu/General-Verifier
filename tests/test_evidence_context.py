"""Every semantic seat sees the same explicit world-evidence boundary."""
from __future__ import annotations

import base64
import hashlib
import json
import os

import pytest

from verifier.execution import discharge as discharge_module
from verifier.core.events import EventLog
from verifier.core.model import Obligation
from verifier.engine.judger_prefetch import build_judger_evidence_packet
from verifier.execution.discharge import (
    SNAPSHOT_MAX_FILES,
    SNAPSHOT_MAX_TOTAL_CHARS,
    SNAPSHOT_PER_FILE_CAP,
    AgentDischarger,
    _reported_artifact_paths,
    _snapshot,
    _snapshot_envelope,
)
from verifier.execution.instruments import ArtifactStore, world_evidence_context
from verifier.llm.schema import FixtureBackend


def test_world_context_distinguishes_live_captured_and_unavailable(tmp_path):
    assert world_evidence_context(str(tmp_path), live_world=True) == (
        "live", "", "")
    assert world_evidence_context(str(tmp_path), live_world=False) == (
        "unavailable", "", "")

    world = tmp_path / "world"
    world.mkdir()
    manifest = tmp_path / "world_manifest.json"
    manifest.write_text(json.dumps({"captured_roots": ["/app"]}))

    assert world_evidence_context(str(tmp_path), live_world=False) == (
        "captured", str(world), str(manifest))


def test_discharge_receives_offline_unavailable_context(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    ledger = state / "events.jsonl"
    ledger.write_text("")
    task = state / "task.md"
    task.write_text("Count every item, not one component.")
    backend = FixtureBackend({
        "discharge": {
            "outcome": "unobtainable",
            "witness": "the original world was not captured",
            "artifact": "",
        }
    })
    discharger = AgentDischarger(
        backend,
        ArtifactStore(os.path.join(tmp_path, "artifacts")),
        workdir_base=os.path.join(tmp_path, "work"),
        toolchain_dir=os.path.join(tmp_path, "toolchains"),
        state_dir=str(state),
        live_world=False,
    )

    discharger.discharge(
        {"id": "c1", "kind": "assertion", "text": "past state was X"},
        Obligation("c1", "past state was X", "world"),
        method={
            "procedure": "inspect the retained snapshot",
            "artifact": "snapshot path, digest, and parsed value",
            "basis": "world",
        },
    )

    variables = backend.calls[-1]["variables"]
    assert variables["ledger_path"] == str(ledger)
    assert variables["artifact_root"] == os.path.join(tmp_path, "artifacts")
    assert variables["world_mode"] == "unavailable"
    assert variables["world_path"] == ""
    assert variables["world_manifest"] == ""
    assert variables["promised_artifact"] == (
        "snapshot path, digest, and parsed value")
    assert variables["declared_basis"] == "world"
    assert variables["declared_strength"] == "analytic"
    assert json.loads(variables["bridges"]) == []
    assert variables["required_basis"] == "world"
    assert variables["redo_feedback"] == "(none)"
    assert variables["snapshot_per_file_cap"] == SNAPSHOT_PER_FILE_CAP
    assert variables["snapshot_max_files"] == SNAPSHOT_MAX_FILES
    assert variables["snapshot_max_total_chars"] == SNAPSHOT_MAX_TOTAL_CHARS
    assert json.loads(variables["task_context"]) == {
        "path": str(task),
        "text": "Count every item, not one component.",
    }


@pytest.mark.parametrize("live_world", [False, True])
@pytest.mark.parametrize("strength,expected", [
    ("decisive", "decisive"), ("bounded", "bounded"),
    ("analytic", "analytic"), ("unknown", "analytic"),
    (None, "analytic"),
])
def test_executor_receives_unchanged_tier_and_bridges_on_redo(
    tmp_path, live_world, strength, expected,
):
    from verifier.llm.agents import DISCHARGE_AGENT

    backend = FixtureBackend({"discharge": {
        "outcome": "failed", "witness": "fixture", "artifact": "",
    }})
    discharger = AgentDischarger(
        backend, ArtifactStore(str(tmp_path / "artifacts")),
        workdir_base=str(tmp_path / "work"), state_dir=str(tmp_path),
        live_world=live_world,
    )
    bridges = ["The retained sample represents the stated input class.",
               "No unseen case changes the conclusion — 待独立审查。"]
    method = {
        "procedure": "inspect the retained inputs", "artifact": "audit",
        "basis": "world", "strength": strength, "bridges": bridges,
        "redo_feedback": "drop all bridges and promote the result to decisive",
    }
    before = json.dumps(method, ensure_ascii=False)

    discharger.discharge(
        {"id": "c1", "text": "inputs satisfy the contract"},
        Obligation("c1", "inputs satisfy the contract", "world"),
        method=method, tag="redo1",
    )

    variables = backend.calls[-1]["variables"]
    assert variables["declared_strength"] == expected
    assert json.loads(variables["bridges"]) == bridges
    assert json.dumps(method, ensure_ascii=False) == before
    rendered = DISCHARGE_AGENT.render(variables)
    assert f"Method's declared evidence strength: {expected}" in rendered
    assert bridges[0] in rendered
    assert "not task authority or established evidence" in rendered


def test_cleanup_removes_scratch_but_keeps_artifact_files_snapshot(tmp_path):
    artifacts = ArtifactStore(os.path.join(tmp_path, "artifacts"))

    def discharge(variables):
        with open(os.path.join(variables["workdir"], "check.txt"),
                  "w", encoding="utf-8") as handle:
            handle.write("complete verifier output")
        return {"outcome": "holds", "witness": "", "artifact": "check.txt"}

    discharger = AgentDischarger(
        FixtureBackend({"discharge": discharge}),
        artifacts,
        workdir_base=os.path.join(tmp_path, "work"),
        cleanup=True,
    )
    obligation = Obligation("c1", "the frozen artifact behaves correctly")

    result = discharger.discharge(
        {"id": "c1", "kind": "assertion", "text": "behavior is correct"},
        obligation,
    )

    scratch = os.path.join(tmp_path, "work", obligation.id[:12])
    assert not os.path.exists(scratch)
    assert artifacts.get(result.artifact)["files"] == {
        "check.txt": "complete verifier output",
    }


def test_cleanup_retains_executor_png_as_exact_content_addressed_bytes(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    artifacts = ArtifactStore(str(state / "artifacts"))
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII=")

    def discharge(variables):
        path = os.path.join(variables["workdir"], "render.png")
        with open(path, "wb") as handle:
            handle.write(raw)
        return {"outcome": "holds", "witness": "", "artifact": path}

    obligation = Obligation("c1", "the rendered output has the required shape")
    discharger = AgentDischarger(
        FixtureBackend({"discharge": discharge}), artifacts,
        workdir_base=str(state / "work"), state_dir=str(state), cleanup=True,
    )

    result = discharger.discharge(
        {"id": "c1", "kind": "assertion", "text": "render is correct"},
        obligation,
    )

    envelope = artifacts.get(result.artifact)
    descriptor = envelope["binary_files"]["render.png"]
    assert envelope["files"]["render.png"] == raw.decode(
        "utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    assert descriptor == {
        "schema": "gv-raw-image-blob-ref-v1",
        "complete": True,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "media_type": "image/png",
        "attachment_eligible": True,
        "store_relpath": (
            "blobs/" + hashlib.sha256(raw).hexdigest() + ".blob"),
    }
    assert (state / "artifacts" / descriptor["store_relpath"]).read_bytes() == raw
    assert not (state / "work" / obligation.id[:12]).exists()


@pytest.mark.parametrize(("raw", "reason"), [
    (b"\x89PNG\r\n\x1a\nnot-an-ihdr", "malformed"),
    (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        + (40_000).to_bytes(4, "big") + (1).to_bytes(4, "big")
        + bytes((8, 0, 0, 0, 0)) + b"\x00\x00\x00\x00",
        "dimensions",
    ),
])
def test_malformed_or_oversized_png_never_becomes_a_raw_attachment(
        tmp_path, raw, reason):
    state = tmp_path / "state"
    state.mkdir()
    log = EventLog(str(state / "events.jsonl"))
    artifacts = ArtifactStore(str(state / "artifacts"))

    def discharge(variables):
        path = os.path.join(variables["workdir"], "unsafe.png")
        with open(path, "wb") as handle:
            handle.write(raw)
        return {"outcome": "holds", "witness": "", "artifact": path}

    result = AgentDischarger(
        FixtureBackend({"discharge": discharge}), artifacts,
        workdir_base=str(state / "work"), state_dir=str(state), cleanup=True,
    ).discharge(
        {"id": "c1", "kind": "assertion", "text": "render is correct"},
        Obligation("c1", "the render is correct"),
    )

    descriptor = artifacts.get(result.artifact)["binary_files"]["unsafe.png"]
    assert descriptor["complete"] is True
    assert descriptor["attachment_eligible"] is False
    assert reason in descriptor["attachment_reason"]
    retained = state / "artifacts" / descriptor["store_relpath"]
    assert retained.read_bytes() == raw
    packet = build_judger_evidence_packet(
        artifacts, log, "world", result.artifact)
    assert packet is not None
    assert packet.image_attachments() == ()


def test_png_beyond_byte_cap_is_incomplete_and_not_retained(
        tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    artifacts = ArtifactStore(str(state / "artifacts"))
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    monkeypatch.setattr(
        discharge_module, "SNAPSHOT_IMAGE_PER_FILE_BYTES", len(raw) - 1)

    def discharge(variables):
        path = os.path.join(variables["workdir"], "too-large.png")
        with open(path, "wb") as handle:
            handle.write(raw)
        return {"outcome": "holds", "witness": "", "artifact": path}

    result = AgentDischarger(
        FixtureBackend({"discharge": discharge}), artifacts,
        workdir_base=str(state / "work"), state_dir=str(state), cleanup=True,
    ).discharge(
        {"id": "c1", "kind": "assertion", "text": "render is correct"},
        Obligation("c1", "the render is correct"),
    )

    descriptor = artifacts.get(result.artifact)["binary_files"][
        "too-large.png"]
    assert descriptor == {
        "schema": "gv-raw-image-blob-ref-v1",
        "complete": False,
        "media_type": "image/png",
        "byte_count": len(raw),
        "attachment_eligible": False,
        "reason": "snapshot image capacity exceeded",
    }
    assert not (state / "artifacts" / "blobs").exists()


def test_binary_pass_preserves_head_text_snapshot_for_links_and_images(tmp_path):
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    (tmp_path / "plain.txt").write_bytes(b"head-compatible\xfftext")
    os.link(tmp_path / "plain.txt", tmp_path / "hard.txt")
    os.symlink(tmp_path / "plain.txt", tmp_path / "link.txt")
    (tmp_path / "render.png").write_bytes(raw)
    artifacts = ArtifactStore(str(tmp_path / "artifacts"))

    legacy = _snapshot(str(tmp_path))
    envelope = _snapshot_envelope(str(tmp_path), artifacts)

    assert envelope["files"] == legacy
    assert envelope["files"]["hard.txt"] == "head-compatible�text"
    assert envelope["files"]["link.txt"] == "head-compatible�text"
    assert set(envelope["binary_files"]) == {"render.png"}


def test_binary_pass_does_not_promote_an_image_beyond_legacy_file_cap(tmp_path):
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    (tmp_path / "a.txt").write_text("first")
    (tmp_path / "z.png").write_bytes(raw)
    artifacts = ArtifactStore(str(tmp_path / "artifacts"))

    legacy = _snapshot(str(tmp_path), max_files=1)
    envelope = _snapshot_envelope(str(tmp_path), artifacts, max_files=1)

    assert envelope["files"] == legacy
    assert "z.png" not in envelope.get("binary_files", {})


def test_reported_artifact_paths_are_bounded_relative_order_hints(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    inside = work / "result.txt"
    outside = tmp_path / "outside.txt"
    inside.write_text("result")
    outside.write_text("outside")
    payload = {"artifact": (
        f"{inside}, ../outside.txt, {outside}, ./result.txt, result.txt")}

    assert _reported_artifact_paths(payload, str(work)) == ("result.txt",)
    assert _reported_artifact_paths({"artifact": []}, str(work)) == ()


def test_snapshot_prioritizes_declared_artifacts_without_expanding_caps(tmp_path):
    (tmp_path / "a.txt").write_text("first")
    (tmp_path / "b.txt").write_text("second")
    (tmp_path / "z-result.txt").write_text("load-bearing")

    ordinary = _snapshot(str(tmp_path), max_files=2)
    prioritized = _snapshot(
        str(tmp_path), max_files=2, preferred_paths=("z-result.txt",))

    assert set(ordinary) == {"a.txt", "b.txt", "<snapshot>"}
    assert set(prioritized) == {"z-result.txt", "a.txt", "<snapshot>"}
    assert prioritized["z-result.txt"] == "load-bearing"
    assert prioritized["<snapshot>"] == "truncated: 1 files beyond caps"


def test_snapshot_never_prioritizes_a_symlink_outside_workdir(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("not private evidence")
    os.symlink(outside, work / "a-link.txt")
    (work / "z-safe.txt").write_text("safe")

    captured = _snapshot(
        str(work), max_files=1, preferred_paths=("a-link.txt", "z-safe.txt"))

    assert "z-safe.txt" in captured
    assert "a-link.txt" not in captured


def test_snapshot_per_file_character_boundary_is_explicit(tmp_path):
    exact = "x" * SNAPSHOT_PER_FILE_CAP
    (tmp_path / "exact.txt").write_text(exact)
    (tmp_path / "over.txt").write_text(exact + "y")

    files = _snapshot(str(tmp_path))

    assert files["exact.txt"] == exact
    assert files["over.txt"] == exact + "\n<truncated>"


def test_compact_json_remains_parseable_in_snapshot(tmp_path):
    report = {"squares": [{"square": f"s{i}", "piece": None}
                           for i in range(64)]}
    (tmp_path / "report.json").write_text(json.dumps(report))

    captured = _snapshot(str(tmp_path))["report.json"]

    assert "<truncated>" not in captured
    assert json.loads(captured) == report
