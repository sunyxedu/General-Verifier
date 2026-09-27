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
from verifier.engine.binding.game import BindingGame
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
    assert "REFUTATION COUNTERSIGN" in seen["countersign"]
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


def test_a_shebang_selects_the_interpreter(tmp_path):
    script = "#!/usr/bin/env python3\nimport sys\nprint('lines', 3, sys.argv[0].endswith('check'))\n"
    game, backend, discharger, artifacts, log = build(
        tmp_path, {"script_judger": reader("holds")}, [script_method(script)])
    result = game.run(ROW, Obligation("c1", "three lines",
                                      required_basis="world"))
    assert result.outcome == "holds"
    seen = backend.calls[1]["variables"]
    assert seen["exit_code"] == 0
    assert seen["stdout"].strip() == "lines 3 True"


def test_sibling_unobtainable_findings_reach_the_next_oracle(tmp_path):
    seen_feedback = []

    def oracle(variables):
        seen_feedback.append(variables["sibling_boundary"])
        return {"methods": [script_method("echo missing")]}

    game, backend, discharger, artifacts, log = build(
        tmp_path,
        {"script_judger": reader("unobtainable", witness="the log file "
                                 "the task names does not exist")},
        [])
    backend.fixtures["oracles"] = oracle
    game.run(ROW, Obligation("c1", "first check", required_basis="world"))
    game.run(ROW, Obligation("c1", "second check", required_basis="world"))
    assert seen_feedback[0] == "[]"
    assert "does not exist" in seen_feedback[-1]
    assert "first check" in seen_feedback[-1]


def test_a_script_that_alters_the_delivered_world_is_rejected(tmp_path):
    import json as _json
    world = tmp_path / "world"
    world.mkdir()
    (world / "keep.txt").write_text("as delivered\n")
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline" / "manifest.json").write_text(
        _json.dumps({"root": str(world), "files": {}}))
    game, backend, discharger, artifacts, log = build(
        tmp_path, {"script_judger": reader("holds")},
        [script_method(f"echo probe > {world}/probe.txt; cat {world}/keep.txt")])

    result = game.run(ROW, Obligation("c1", "keep.txt says as delivered",
                                      required_basis="world"))

    assert result.outcome == "failed"
    rejected = [e["payload"]["reason"] for e in log.replay()
                if e["type"] == "method_rejected"]
    assert any("altered the delivered world" in r and "probe.txt" in r
               for r in rejected)
    # the reader was never asked about a run that corrupted the subject
    assert "script_judger" not in [c["role"] for c in backend.calls]


def test_a_read_only_script_leaves_no_world_changes(tmp_path):
    import json as _json
    world = tmp_path / "world"
    world.mkdir()
    (world / "keep.txt").write_text("as delivered\n")
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline" / "manifest.json").write_text(
        _json.dumps({"root": str(world), "files": {}}))
    game, backend, discharger, artifacts, log = build(
        tmp_path, {"script_judger": reader("holds")},
        [script_method(f"cat {world}/keep.txt; echo scratch > $GV_WORKDIR/s.txt")])
    result = game.run(ROW, Obligation("c1", "keep.txt says as delivered",
                                      required_basis="world"))
    assert result.outcome == "holds"


def test_a_relative_state_dir_still_finds_the_script(tmp_path, monkeypatch):
    """The script is launched with cwd=workdir; a relative workdir_base must
    not make the script's own path resolve against that new cwd (the
    host-side mounts pass --state-dir .gv)."""
    monkeypatch.chdir(tmp_path)
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    artifacts = ArtifactStore(os.path.join(tmp_path, "art"))
    backend = FixtureBackend({"oracles": {"methods": [script_method("echo three")]},
                              "script_judger": reader("holds")})
    real = AgentDischarger(backend, artifacts, workdir_base=".gv/work",
                           state_dir=".gv")
    game = BindingGame(backend, artifacts, log, AgentStub(real))
    result = game.run(ROW, Obligation("c1", "three lines", required_basis="world"))
    assert result.outcome == "holds"
    seen = backend.calls[1]["variables"]
    assert seen["exit_code"] == 0 and seen["stdout"].strip() == "three"


def test_a_shebang_interpreter_missing_from_the_task_path_comes_from_the_toolchain(tmp_path, monkeypatch):
    """A verification script is the verifier's instrument: when the task
    environment lacks the interpreter its shebang names, the verifier's
    toolchain supplies it; the script's commands still run under the
    task's PATH."""
    toolchain = tmp_path / "toolchain"
    (toolchain / "bin").mkdir(parents=True)
    interp = toolchain / "bin" / "gvtestinterp"
    interp.write_text("#!/bin/sh\necho interpreted-by-toolchain \"$1\"\n")
    interp.chmod(0o755)
    empty = tmp_path / "emptypath"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    artifacts = ArtifactStore(os.path.join(tmp_path, "art"))
    script = "#!/usr/bin/env gvtestinterp\nignored body\n"
    backend = FixtureBackend({"oracles": {"methods": [script_method(script)]},
                              "script_judger": reader("holds")})
    real = AgentDischarger(backend, artifacts,
                           workdir_base=os.path.join(tmp_path, "work"),
                           state_dir=str(tmp_path), toolchain_dir=str(toolchain))
    game = BindingGame(backend, artifacts, log, AgentStub(real))
    result = game.run(ROW, Obligation("c1", "three lines", required_basis="world"))
    assert result.outcome == "holds"
    seen = backend.calls[1]["variables"]
    assert seen["exit_code"] == 0
    assert seen["stdout"].startswith("interpreted-by-toolchain")


def test_an_interpreter_the_script_body_calls_comes_from_the_toolchain_when_the_task_lacks_it(tmp_path, monkeypatch):
    """A check whose body invokes an interpreter by name (not via its
    shebang) must still launch when the task environment lacks it: the
    verifier's instrument directories are reachable after the task's."""
    toolchain = tmp_path / "toolchain"
    (toolchain / "bin").mkdir(parents=True)
    tool = toolchain / "bin" / "gvtesttool"
    tool.write_text("#!/bin/sh\necho from-toolchain\n")
    tool.chmod(0o755)
    empty = tmp_path / "taskpath"
    empty.mkdir()
    monkeypatch.setenv("PATH", f"{empty}:/usr/bin:/bin")
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    artifacts = ArtifactStore(os.path.join(tmp_path, "art"))
    backend = FixtureBackend({"oracles": {"methods": [script_method("gvtesttool")]},
                              "script_judger": reader("holds")})
    real = AgentDischarger(backend, artifacts,
                           workdir_base=os.path.join(tmp_path, "work"),
                           state_dir=str(tmp_path), toolchain_dir=str(toolchain))
    game = BindingGame(backend, artifacts, log, AgentStub(real))
    result = game.run(ROW, Obligation("c1", "three lines", required_basis="world"))
    assert result.outcome == "holds"
    seen = backend.calls[1]["variables"]
    assert seen["exit_code"] == 0 and seen["stdout"].strip() == "from-toolchain"


def test_the_task_environment_wins_over_the_verifier_toolchain(tmp_path, monkeypatch):
    """When the task has the program, the script sees the task's copy: a
    check of the delivery's runnability must not be answered by ours."""
    toolchain = tmp_path / "toolchain"
    (toolchain / "bin").mkdir(parents=True)
    ours = toolchain / "bin" / "gvtesttool"
    ours.write_text("#!/bin/sh\necho from-toolchain\n")
    ours.chmod(0o755)
    taskbin = tmp_path / "taskbin"
    taskbin.mkdir()
    theirs = taskbin / "gvtesttool"
    theirs.write_text("#!/bin/sh\necho from-task\n")
    theirs.chmod(0o755)
    monkeypatch.setenv("PATH", f"{taskbin}:/usr/bin:/bin")
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    artifacts = ArtifactStore(os.path.join(tmp_path, "art"))
    backend = FixtureBackend({"oracles": {"methods": [script_method("gvtesttool")]},
                              "script_judger": reader("holds")})
    real = AgentDischarger(backend, artifacts,
                           workdir_base=os.path.join(tmp_path, "work"),
                           state_dir=str(tmp_path), toolchain_dir=str(toolchain))
    game = BindingGame(backend, artifacts, log, AgentStub(real))
    game.run(ROW, Obligation("c1", "three lines", required_basis="world"))
    assert backend.calls[1]["variables"]["stdout"].strip() == "from-task"
