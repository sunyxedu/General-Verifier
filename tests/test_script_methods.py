"""Methods in EXECUTABLE FORM: the oracle supplies a script, the executor
runs it verbatim (no agent), and the script judger signs sufficiency and
reads the outcome off the retained output. The same axioms gate the
result as on the agent path; a script that did not run is a failed
attempt, never a verdict.
"""
from __future__ import annotations

import json
import os

from verifier.core.events import EventLog
from verifier.core.model import Obligation
from verifier.engine.game import BindingGame
from verifier.execution.discharge import AgentDischarger, DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend, RoleError


ROW = {"id": "c1", "kind": "assertion", "text": "the file holds three lines"}


def script_method(script, procedure="count the lines of the file",
                  strength="decisive", basis="world"):
    return {"procedure": procedure, "artifact": "the printed count",
            "level": "direct observation", "basis": basis,
            "strength": strength, "bridges": [], "script": script}


def agent_method(procedure="count the lines by hand"):
    return {"procedure": procedure, "artifact": "notes", "level": "manual",
            "basis": "world", "strength": "decisive", "bridges": []}


def reader(outcome, sufficient="yes", witness="", kind="none",
           degradation="n/a"):
    return {"sufficient": sufficient, "outcome": outcome, "witness": witness,
            "witness_kind": kind,
            "inputs": [{"ref": "target.txt", "provenance": "task_given",
                        "role": "subject"}],
            "degradation": degradation, "reason": "read the count"}


class AgentStub:
    """The agent executor, which must NOT be consulted for a script method."""
    def __init__(self, discharger):
        self.inner = discharger
        self.agent_calls = 0

    def discharge(self, row, obligation, method=None, tag=""):
        self.agent_calls += 1
        return DischargeResult(witness_kind="execution", outcome="holds",
                               witness=None, artifact="agent", instrument="stub",
                               basis="world")

    def run_script(self, row, obligation, method, tag=""):
        return self.inner.run_script(row, obligation, method, tag)


def build(tmp_path, roles, methods):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    artifacts = ArtifactStore(os.path.join(tmp_path, "art"))
    backend = FixtureBackend({"oracles": {"methods": methods}, **roles})
    real = AgentDischarger(backend, artifacts,
                           workdir_base=os.path.join(tmp_path, "work"),
                           state_dir=str(tmp_path))
    discharger = AgentStub(real)
    game = BindingGame(backend, artifacts, log, discharger)
    return game, backend, discharger, artifacts, log


def test_script_runs_verbatim_and_the_reader_settles_it(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("a\nb\nc\n")
    script = f"wc -l < {target}\necho workdir=$GV_WORKDIR"
    game, backend, discharger, artifacts, log = build(
        tmp_path, {"script_judger": reader("holds")}, [script_method(script)])

    result = game.run(ROW, Obligation("c1", "target.txt has three lines",
                                      required_basis="world"))

    assert result.outcome == "holds"
    assert result.basis == "world" and result.strength == "decisive"
    assert discharger.agent_calls == 0
    assert [c["role"] for c in backend.calls] == ["oracles", "script_judger"]
    seen = backend.calls[1]["variables"]
    assert seen["script"] == script
    assert seen["exit_code"] == 0
    assert seen["stdout"].strip().startswith("3")
    assert "workdir=" in seen["stdout"]
    # the retained run is the artifact the binding record points at
    bound = artifacts.get(result.artifact)["bound"]
    run = artifacts.get(bound["discharge_artifact"])
    assert run["instrument"] == "script" and run["script"] == script
    assert run["files"]["check.sh"].strip() == script
    verdicts = [e for e in log.replay() if e["type"] == "judger_verdict"]
    assert verdicts[0]["payload"]["source"] == "script"
    assert result.inputs[0]["ref"] == "target.txt"


COUNTERSIGN = {"sufficient": "yes", "faithful": "yes", "degradation": "n/a",
               "reason": "the printed count contradicts the obligation"}


def test_reader_may_refute_at_decisive_once_the_judger_countersigns(tmp_path):
    game, backend, discharger, artifacts, log = build(
        tmp_path,
        {"script_judger": reader("refuted", witness="printed 2, not 3",
                                 kind="execution"),
         "judger": COUNTERSIGN},
        [script_method("printf 'a\\nb\\n' | wc -l")])
    result = game.run(ROW, Obligation("c1", "three lines",
                                      required_basis="world"))
    assert result.outcome == "refuted"
    assert result.witness == "printed 2, not 3"
    assert result.witness_kind == "execution"
    roles = [c["role"] for c in backend.calls]
    assert roles == ["oracles", "script_judger", "judger"]
    # the countersigning judger was handed the retained script run
    seen = backend.calls[2]["variables"]
    evidence = json.loads(seen["evidence"])
    assert evidence["outcome"] == "refuted"
    assert artifacts.get(evidence["artifact_digest"])["instrument"] == "script"
    verdicts = [e["payload"] for e in log.replay()
                if e["type"] == "judger_verdict"]
    assert [v.get("source") for v in verdicts] == [
        "script", "refutation countersign"]


def test_a_disputed_refutation_is_a_failed_attempt(tmp_path):
    game, backend, discharger, artifacts, log = build(
        tmp_path,
        {"script_judger": reader("refuted", witness="printed 2, not 3",
                                 kind="execution"),
         "judger": {"sufficient": "yes", "faithful": "no",
                    "degradation": "n/a",
                    "reason": "the file at the task address has 3 lines; "
                              "the script read a stale copy"}},
        [script_method("printf 'a\\nb\\n' | wc -l")])
    result = game.run(ROW, Obligation("c1", "three lines",
                                      required_basis="world"))
    assert result.outcome == "failed"
    rejected = [e["payload"]["reason"] for e in log.replay()
                if e["type"] == "method_rejected"]
    assert any("not countersigned" in r for r in rejected)
    # a holds never needs the countersign
    assert [c["role"] for c in backend.calls].count("judger") == 1


def test_refutation_below_decisive_is_a_contrary_signal_not_a_verdict(tmp_path):
    game, backend, discharger, artifacts, log = build(
        tmp_path,
        {"script_judger": reader("refuted", witness="2", kind="execution",
                                 degradation="justified")},
        [script_method("echo 2", strength="bounded")])
    result = game.run(ROW, Obligation("c1", "three lines",
                                      required_basis="world"))
    assert result.outcome == "failed"
    rejected = [e["payload"]["reason"] for e in log.replay()
                if e["type"] == "method_rejected"]
    assert any("below decisive" in r for r in rejected)


def test_absence_witness_needs_a_presence_requirement(tmp_path):
    game, backend, discharger, artifacts, log = build(
        tmp_path,
        {"script_judger": reader("refuted", witness="no such file",
                                 kind="absence")},
        [script_method("ls nowhere 2>&1 || true")])
    result = game.run(ROW, Obligation("c1", "the file has three lines",
                                      required_basis="world"))
    assert result.outcome == "failed"
    rejected = [e["payload"]["reason"] for e in log.replay()
                if e["type"] == "method_rejected"]
    assert any("inadmissible" in r for r in rejected)


def test_insufficient_script_is_rejected_and_the_agent_method_runs(tmp_path):
    game, backend, discharger, artifacts, log = build(
        tmp_path,
        {"script_judger": reader("holds", sufficient="no"),
         "judger": {"sufficient": "yes", "faithful": "yes",
                    "degradation": "n/a", "reason": "ok"}},
        [script_method("echo 3"), agent_method()])
    result = game.run(ROW, Obligation("c1", "three lines",
                                      required_basis="world"))
    assert result.outcome == "holds"
    assert result.instrument.startswith("count the lines by hand")
    assert discharger.agent_calls == 1
    # the oracle's repair pass sees the insufficiency as a method defect
    rejected = [e["payload"]["reason"] for e in log.replay()
                if e["type"] == "method_rejected"]
    assert any("insufficient" in r for r in rejected)


def test_timeout_is_a_failed_attempt_never_a_verdict(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_SCRIPT_TIMEOUT", "1")
    game, backend, discharger, artifacts, log = build(
        tmp_path, {"script_judger": reader("holds")},
        [script_method("sleep 5; echo 3")])
    result = game.run(ROW, Obligation("c1", "three lines",
                                      required_basis="world"))
    assert result.outcome == "failed"
    # the reader was never consulted about a run that did not finish
    assert [c["role"] for c in backend.calls] == ["oracles", "oracles"]
    rejected = [e["payload"]["reason"] for e in log.replay()
                if e["type"] == "method_rejected"]
    assert any("timed out" in r for r in rejected)


def test_unobtainable_read_off_the_output_unlocks_weaker_tiers(tmp_path):
    calls = {"n": 0}

    def read(variables):
        calls["n"] += 1
        if calls["n"] == 1:
            return reader("unobtainable", witness="tool absent")
        return reader("holds", degradation="justified")

    game, backend, discharger, artifacts, log = build(
        tmp_path, {"script_judger": read},
        [script_method("echo missing", procedure="strong"),
         script_method("echo 3", procedure="weak", strength="bounded")])
    result = game.run(ROW, Obligation("c1", "three lines",
                                      required_basis="world"))
    assert result.outcome == "holds"
    assert result.strength == "bounded"


def test_reader_failure_fails_closed(tmp_path):
    def broken(variables):
        raise RoleError("seat down")
    game, backend, discharger, artifacts, log = build(
        tmp_path, {"script_judger": broken}, [script_method("echo 3")])
    result = game.run(ROW, Obligation("c1", "three lines",
                                      required_basis="world"))
    assert result.outcome == "failed"
    failed = [e for e in log.replay() if e["type"] == "judger_call_failed"]
    assert failed and failed[0]["payload"]["source"] == "script"


def test_agent_method_without_a_script_takes_the_agent_path(tmp_path):
    game, backend, discharger, artifacts, log = build(
        tmp_path,
        {"judger": {"sufficient": "yes", "faithful": "yes",
                    "degradation": "n/a", "reason": "ok"}},
        [agent_method()])
    result = game.run(ROW, Obligation("c1", "three lines",
                                      required_basis="world"))
    assert result.outcome == "holds"
    assert discharger.agent_calls == 1
    assert "script_judger" not in [c["role"] for c in backend.calls]
