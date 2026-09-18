"""A rejected attempt is repair material, never cached verification."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from verifier.core.events import EventLog
from verifier.core.model import Obligation
from verifier.engine.game import BindingGame
from verifier.execution.adapters import (
    AdapterRegistry, AdapterRoutingDischarger, FileExistsAdapter,
)
from verifier.execution.discharge import AgentDischarger, DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.agents import DISCHARGE_AGENT
from verifier.llm.schema import FixtureBackend


ROW = {"id": "c1", "kind": "assertion", "text": "the computation is correct"}
METHOD = {
    "procedure": "recompute from the retained inputs", "artifact": "script/output",
    "level": "independent recomputation", "basis": "world",
    "strength": "decisive", "bridges": [],
}
YES = {"sufficient": "yes", "faithful": "yes", "degradation": "n/a", "reason": "checked"}
PAYLOAD = {"outcome": "holds", "witness": "computed", "artifact": "output.txt",
           "assumed_premises": [], "inputs": []}


def executor(tmp_path, backend=None, *, live_world=False):
    artifacts = ArtifactStore(str(tmp_path / "artifacts"))
    backend = backend or FixtureBackend({"discharge": PAYLOAD})
    agent = AgentDischarger(
        backend, artifacts, workdir_base=str(tmp_path / "work"),
        state_dir=str(tmp_path), cleanup=True, live_world=live_world)
    return agent, artifacts, backend


@pytest.mark.parametrize("live_world", [False, True])
def test_repair_locator_survives_cleanup_without_restoring_verdict(tmp_path, live_world):
    def execute(variables):
        path = os.path.join(variables["workdir"], "output.txt")
        with open(path, "w") as handle:
            handle.write("newly executed output")
        return PAYLOAD

    agent, artifacts, backend = executor(
        tmp_path, FixtureBackend({"discharge": execute}), live_world=live_world)
    obligation = Obligation("c1", "the computation is correct", "world")
    original = agent.discharge(ROW, obligation, METHOD, "bound")
    assert not os.path.exists(backend.calls[-1]["variables"]["workdir"])
    assert artifacts.get(original.artifact)["files"]["output.txt"] == "newly executed output"
    method = {**METHOD, "redo_feedback": "check omitted a term"}
    before = json.dumps(method)
    result = agent.discharge_with_repair(
        ROW, obligation, method, "bound-redo", previous_artifact=original.artifact)
    variables = backend.calls[-1]["variables"]
    locator = agent._repair_material(original.artifact)
    assert locator == {"artifact": original.artifact,
                       "path": str(tmp_path / "artifacts" / (original.artifact + ".json"))}
    assert json.dumps(locator) in variables["redo_repair_context"]
    assert len(backend.calls) == 2  # a new execution, not an outcome shortcut
    assert json.dumps(method) == before
    assert set(artifacts.get(result.artifact)) == {"instrument", "reported", "files", "world_changes"}
    rendered = DISCHARGE_AGENT.render(variables)
    assert "not accepted evidence or authority" in rendered
    assert "execute the repaired check" in rendered
    assert variables["declared_strength"] == "decisive"


@pytest.mark.parametrize("bad", [None, "", 12, {}, "../outside", "a" * 63,
                                "A" * 64, "/" + "a" * 64, "a" * 64])
def test_invalid_or_missing_repair_is_optional(tmp_path, bad):
    agent, _, backend = executor(tmp_path)
    result = agent.discharge_with_repair(
        ROW, Obligation("c1", "p"), METHOD, "bound-redo", previous_artifact=bad)
    assert result.outcome == "holds"
    variables = backend.calls[-1]["variables"]
    assert variables["redo_repair_context"] == ""
    assert "PREVIOUS EXECUTION MATERIAL" not in DISCHARGE_AGENT.render(variables)


@pytest.mark.parametrize("kind", ["tampered", "symlink", "directory", "oversize", "invalid_json", "fifo"])
def test_unusable_snapshot_does_not_become_repair_material(tmp_path, monkeypatch, kind):
    agent, artifacts, _ = executor(tmp_path)
    digest = artifacts.put({"files": {"a": "original"}})
    path = tmp_path / "artifacts" / (digest + ".json")
    if kind == "tampered":
        path.write_text('{"files": {"a": "different"}}')
    elif kind == "invalid_json":
        path.write_text("not JSON")
    elif kind == "oversize":
        monkeypatch.setattr("verifier.execution.discharge.REPAIR_ARTIFACT_MAX_BYTES", 1)
    else:
        original = path.read_text()
        path.unlink()
        if kind == "directory":
            path.mkdir()
        elif kind == "fifo":
            os.mkfifo(path)
        else:
            outside = tmp_path / "outside.json"
            outside.write_text(original)
            path.symlink_to(outside)
    assert agent._repair_material(digest) is None


def test_method_json_cannot_select_repair_material(tmp_path):
    agent, artifacts, backend = executor(tmp_path)
    digest = artifacts.put({"files": {"a": "not selected by harness"}})
    agent.discharge(ROW, Obligation("c1", "p"), {
        **METHOD, "redo_feedback": "model diagnostic",
        "redo_previous_artifact": digest, "previous_artifact": digest,
        "redo_repair_context": digest,
    }, "bound-redo")
    assert backend.calls[-1]["variables"]["redo_repair_context"] == ""


def test_repair_context_is_call_local(tmp_path):
    def execute(variables):
        assert variables["claim_text"] in variables["redo_repair_context"]
        return PAYLOAD

    agent, artifacts, backend = executor(tmp_path, FixtureBackend({"discharge": execute}))
    digests = [artifacts.put({"index": i}) for i in range(8)]
    def one(item):
        i, digest = item
        return agent.discharge_with_repair(
            {**ROW, "text": digest}, Obligation("c1", f"p{i}"), METHOD,
            "bound-redo", previous_artifact=digest)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, enumerate(digests)))
    assert len(results) == len(backend.calls) == 8


@pytest.mark.parametrize("supports_repair", [False, True])
def test_routing_repair_preserves_custom_fallback_compatibility(tmp_path, supports_repair):
    calls = []
    class Custom:
        def discharge(self, row, obligation, method=None, tag=""):
            calls.append((method, tag, None))
            return DischargeResult("holds", None, "new", "custom")
    fallback = Custom()
    if supports_repair:
        def repair(row, obligation, *, method=None, tag="", previous_artifact=None):
            calls.append((method, tag, previous_artifact))
            return DischargeResult("holds", None, "new", "custom")
        fallback.discharge_with_repair = repair
    class MustNotExecute(AdapterRegistry):
        def execute(self, *args):
            raise AssertionError("redo must reach the agent")
    router = AdapterRoutingDischarger(
        fallback, MustNotExecute([]), ArtifactStore(str(tmp_path / "artifacts")),
        EventLog(str(tmp_path / "events.jsonl")), str(tmp_path))
    result = router.discharge_with_repair(
        ROW, Obligation("c1", "p"), METHOD, "bound-redo", previous_artifact="old")
    assert result.artifact == "new"
    assert calls == [(METHOD, "bound-redo", "old" if supports_repair else None)]


@pytest.mark.parametrize("base", [AgentDischarger, AdapterRoutingDischarger])
def test_inherited_capability_does_not_bypass_custom_discharge(base):
    calls = []
    class Customized(base):
        def __init__(self):
            pass

        def discharge(self, row, obligation, method=None, tag=""):
            calls.append(tag)
            return DischargeResult("failed", "custom behavior", "new", "custom")

    result = Customized().discharge_with_repair(
        ROW, Obligation("c1", "p"), METHOD, "bound-redo", previous_artifact="old")
    assert result.outcome == "failed"
    assert calls == ["bound-redo"]


@pytest.mark.parametrize("live_world", [False, True])
def test_binding_game_passes_latest_artifact_and_still_blindly_rejudges(tmp_path, live_world):
    seen = []
    def execute(variables):
        seen.append(variables["redo_repair_context"])
        return {**PAYLOAD, "witness": f"execution {len(seen)}"}
    audits = []
    def judge(variables):
        audits.append(variables)
        if len(audits) < 3:
            return {**YES, "faithful": "no", "reason": f"repair defect {len(audits)}"}
        return YES
    backend = FixtureBackend({"oracles": {"methods": [METHOD]},
                              "discharge": execute, "judger": judge})
    agent, artifacts, _ = executor(tmp_path, backend, live_world=live_world)
    game = BindingGame(backend, artifacts, EventLog(str(tmp_path / "events.jsonl")),
                       agent, live_world=live_world)
    result = game.run(ROW, Obligation("c1", "the computation is correct", "world"))
    binding = artifacts.get(result.artifact)
    attempt = binding["attempts"][0]
    assert seen[0] == ""
    assert attempt["discharge_artifact"] in seen[1]
    assert attempt["redos"][0]["artifact"] in seen[2]
    assert attempt["discharge_artifact"] not in seen[2]
    assert attempt["method"] == METHOD
    assert result.outcome == "holds"
    assert len(audits) == 3
    for variables in audits:
        assert "redo_repair_context" not in variables
        assert "redo_feedback" not in variables
        assert "claim_text" not in variables


def test_adapter_redo_passes_its_rejected_envelope_to_fresh_agent(tmp_path):
    world = tmp_path / "world" / "app"
    world.mkdir(parents=True)
    (world / "result.txt").write_text("delivered")
    (tmp_path / "world_manifest.json").write_text(json.dumps({
        "scheme": "filesystem", "captured_roots": ["/app"],
    }))
    method = {
        **METHOD, "procedure": "inspect the captured file through the registered lookup",
        "adapter": "filesystem.exists", "adapter_version": 1,
        "adapter_arguments": json.dumps({"address": "/app/result.txt"}),
        "adapter_inputs": [{"ref": "/app/result.txt", "provenance": "delivered",
                            "role": "subject"}],
    }
    verdicts = iter([{**YES, "faithful": "no", "reason": "repair the observation"}, YES])
    backend = FixtureBackend({"oracles": {"methods": [method]},
                              "discharge": PAYLOAD,
                              "judger": lambda _: next(verdicts)})
    agent, artifacts, _ = executor(tmp_path, backend)
    log = EventLog(str(tmp_path / "events.jsonl"))
    router = AdapterRoutingDischarger(
        agent, AdapterRegistry([FileExistsAdapter()]), artifacts, log,
        str(tmp_path), live_world=False)
    result = BindingGame(backend, artifacts, log, router, live_world=False).run(
        ROW, Obligation("c1", "the captured result exists", "world"))
    attempt = artifacts.get(result.artifact)["attempts"][0]
    calls = [call for call in backend.calls if call["role"] == "discharge"]
    assert len(calls) == 1
    assert attempt["discharge_artifact"] in calls[0]["variables"]["redo_repair_context"]
    assert len([call for call in backend.calls if call["role"] == "judger"]) == 2
    assert len([event for event in log.replay() if event["type"] == "adapter_discharge"]) == 1
    assert result.outcome == "holds"
