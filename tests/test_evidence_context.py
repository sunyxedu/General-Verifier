"""Every semantic seat sees the same explicit world-evidence boundary."""
from __future__ import annotations

import json
import os

import pytest

from verifier.core.model import Obligation
from verifier.execution.discharge import (
    SNAPSHOT_PER_FILE_CAP,
    AgentDischarger,
    _snapshot,
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
