"""A check's world delta is charged to the check only when the record says
the solver was not the one changing the world.

Online the solver keeps working while the verifier checks, so a file the
solver writes during a check shows up in the check's before/after delta.
The hook feed (every tool call stamped before it runs, every result after
it finishes) splits the delta: changes made while no world-mutating
solver tool was running are the check's (a structural defect, as
before); changes made during a solver run are "moved" — the check saw no
single state, settles nothing, and the method is run again rather than
rejected with a demand for isolation it never needed.
"""
from __future__ import annotations

import json
import os
import time

from verifier.core.events import EventLog
from verifier.core.model import Claim, CLAIM_ASSERTION, Obligation
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.engine.game import (
    Attempt, BindingGame, MOVED_RETRIES, prior_attempt_briefing,
)
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import AgentDischarger, DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.execution.world_watch import (
    HOOK_FEED, feed_mark, solver_activity, split_changes,
)
from verifier.llm.schema import FixtureBackend


def feed_line(kind, tool, uid, ts=None, session="s1"):
    payload = {"tool": tool, "tool_use_id": uid, "session": session}
    if ts is not None:
        payload["ts"] = ts
    return json.dumps({"type": kind, "payload": payload}) + "\n"


def write_feed(state, *lines):
    with open(os.path.join(state, HOOK_FEED), "a", encoding="utf-8") as f:
        f.writelines(lines)


# -- solver_activity ------------------------------------------------------

def test_no_feed_means_no_activity(tmp_path):
    assert feed_mark(str(tmp_path)) == 0
    assert solver_activity(str(tmp_path), 0, time.time())["acted"] is False


def test_a_mutating_run_after_the_mark_is_activity(tmp_path):
    state = str(tmp_path)
    write_feed(state, feed_line("tool_call", "Bash", "old", 1.0),
               feed_line("tool_result", "Bash", "old", 2.0))
    mark = feed_mark(state)
    write_feed(state, feed_line("tool_call", "apply_patch", "p1", 10.0),
               feed_line("tool_result", "apply_patch", "p1", 11.0))
    act = solver_activity(state, mark, 20.0)
    assert act["acted"] and act["timed"]
    assert act["intervals"] == [(10.0, 11.0)]
    assert act["tools"] == ["apply_patch"]


def test_verifier_verbs_and_reads_are_not_activity(tmp_path):
    state = str(tmp_path)
    mark = feed_mark(state)
    write_feed(state, feed_line("tool_call", "mcp__gv__status", "m1", 10.0),
               feed_line("tool_call", "Read", "r1", 11.0),
               feed_line("tool_result", "Read", "r1", 12.0))
    assert solver_activity(state, mark, 20.0)["acted"] is False


def test_a_run_in_flight_when_the_check_began_is_activity(tmp_path):
    state = str(tmp_path)
    write_feed(state, feed_line("tool_call", "Bash", "build", 5.0))
    mark = feed_mark(state)
    act = solver_activity(state, mark, 20.0)
    # still running when the check ended
    assert act["acted"] and act["intervals"] == [(5.0, 20.0)]
    write_feed(state, feed_line("tool_result", "Bash", "build", 15.0))
    assert solver_activity(state, mark, 20.0)["intervals"] == [(5.0, 15.0)]


def test_an_abandoned_call_the_session_moved_past_is_not_running(tmp_path):
    state = str(tmp_path)
    write_feed(state, feed_line("tool_call", "Bash", "lost", 1.0),
               feed_line("tool_call", "mcp__gv__commit", "c1", 2.0))
    mark = feed_mark(state)
    assert solver_activity(state, mark, 20.0)["acted"] is False


def test_unstamped_activity_is_untimed(tmp_path):
    state = str(tmp_path)
    mark = feed_mark(state)
    write_feed(state, feed_line("tool_call", "Bash", "b1"),
               feed_line("tool_result", "Bash", "b1"))
    act = solver_activity(state, mark, 20.0)
    assert act["acted"] and act["timed"] is False


# -- split_changes --------------------------------------------------------

def _world(tmp_path):
    root = tmp_path / "world"
    root.mkdir()
    (root / "kept.txt").write_text("kept\n")
    (root / "gone.txt").write_text("gone\n")
    return root


def test_without_solver_activity_every_change_is_charged(tmp_path):
    from verifier.baseline import tree_manifest
    root = _world(tmp_path)
    before = tree_manifest(str(root))
    (root / "new.txt").write_text("new\n")
    after = tree_manifest(str(root))
    quiet = {"acted": False, "timed": True, "intervals": []}
    charged, moved = split_changes(str(root), before, after, quiet,
                                   (time.time() - 5, time.time() + 5))
    assert charged["added"] == ["new.txt"] and moved == {}


def test_a_change_inside_a_solver_run_is_moved(tmp_path):
    from verifier.baseline import tree_manifest
    root = _world(tmp_path)
    before = tree_manifest(str(root))
    (root / "new.txt").write_text("new\n")
    (root / "gone.txt").unlink()
    after = tree_manifest(str(root))
    at = os.lstat(root / "new.txt").st_ctime
    act = {"acted": True, "timed": True, "intervals": [(at - 0.5, at + 0.5)]}
    charged, moved = split_changes(str(root), before, after, act,
                                   (at - 5, at + 5))
    assert charged == {}
    assert moved["added"] == ["new.txt"]
    # a deletion leaves no change time to place: never charged when the
    # solver was running
    assert moved["removed"] == ["gone.txt"]


def test_a_change_no_solver_run_spans_is_charged(tmp_path):
    from verifier.baseline import tree_manifest
    root = _world(tmp_path)
    before = tree_manifest(str(root))
    (root / "probe.txt").write_text("the check wrote this\n")
    after = tree_manifest(str(root))
    at = os.lstat(root / "probe.txt").st_ctime
    # the solver ran a tool during the window, but well before the write
    act = {"acted": True, "timed": True, "intervals": [(at - 30, at - 20)]}
    charged, moved = split_changes(str(root), before, after, act,
                                   (at - 40, at + 5))
    assert charged["added"] == ["probe.txt"] and moved == {}


def test_untimed_activity_moves_everything(tmp_path):
    from verifier.baseline import tree_manifest
    root = _world(tmp_path)
    before = tree_manifest(str(root))
    (root / "new.txt").write_text("new\n")
    after = tree_manifest(str(root))
    act = {"acted": True, "timed": False, "intervals": []}
    charged, moved = split_changes(str(root), before, after, act,
                                   (time.time() - 5, time.time() + 5))
    assert charged == {} and moved["added"] == ["new.txt"]


# -- the binding game -----------------------------------------------------

ROW = {"id": "c1", "kind": "assertion", "text": "keep.txt says as delivered"}


def script_method(script):
    return {"procedure": "read keep.txt", "artifact": "the printed text",
            "level": "direct observation", "basis": "world",
            "strength": "decisive", "bridges": [], "script": script}


def reader_holds(_variables):
    return {"sufficient": "yes", "outcome": "holds", "witness": "",
            "witness_kind": "none",
            "inputs": [{"ref": "keep.txt", "provenance": "task_given",
                        "role": "subject"}],
            "degradation": "n/a", "reason": "read the text"}


def delivered_world(tmp_path):
    world = tmp_path / "world"
    world.mkdir()
    (world / "keep.txt").write_text("as delivered\n")
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline" / "manifest.json").write_text(
        json.dumps({"root": str(world), "files": {}}))
    return world


def script_game(tmp_path, script):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    artifacts = ArtifactStore(os.path.join(tmp_path, "art"))
    backend = FixtureBackend({"oracles": {"methods": [script_method(script)]},
                              "script_judger": reader_holds})
    discharger = AgentDischarger(backend, artifacts,
                                 workdir_base=os.path.join(tmp_path, "work"),
                                 state_dir=str(tmp_path))
    return BindingGame(backend, artifacts, log, discharger), backend, log


def solver_edit_during_check(state, path, text, uid, offset=0.0):
    """Python source that plays the solver: a stamped tool call, a write
    into the delivered world, a stamped result."""
    return (
        "import json, time\n"
        f"feed = {os.path.join(state, HOOK_FEED)!r}\n"
        "def line(kind, ts):\n"
        f"    return json.dumps({{'type': kind, 'payload': {{'tool': 'apply_patch', "
        f"'tool_use_id': {uid!r}, 'session': 's1', 'ts': ts}}}}) + '\\n'\n"
        f"open(feed, 'a').write(line('tool_call', time.time() + {offset}))\n"
        f"open({path!r}, 'w').write({text!r})\n"
        f"open(feed, 'a').write(line('tool_result', time.time() + {offset}))\n"
    )


def test_a_solver_edit_during_the_check_requeues_instead_of_rejecting(tmp_path):
    world = delivered_world(tmp_path)
    marker = tmp_path / "solver-done"
    edit = solver_edit_during_check(str(tmp_path), str(world / "out.txt"),
                                    "the solver's deliverable\n", "p1")
    (tmp_path / "solver.py").write_text(edit)
    # first run: the solver writes its deliverable while the check reads;
    # second run: the solver is done and the world holds still
    script = (f"if [ ! -e {marker} ]; then touch {marker}; "
              f"python3 {tmp_path / 'solver.py'}; fi; cat {world}/keep.txt")
    game, backend, log = script_game(tmp_path, script)

    result = game.run(ROW, Obligation("c1", "keep.txt says as delivered",
                                      required_basis="world"))

    assert result.outcome == "holds"
    events = log.replay()
    moved = [e["payload"] for e in events if e["type"] == "method_moved"]
    assert len(moved) == 1 and moved[0]["requeued"] is True
    assert "apply_patch" in moved[0]["reason"]
    assert not [e for e in events if e["type"] == "method_rejected"]
    # the reader only saw the run that observed a single state
    assert [c["role"] for c in backend.calls].count("script_judger") == 1


def test_a_check_write_outside_every_solver_run_is_still_a_defect(tmp_path):
    world = delivered_world(tmp_path)
    # the solver ran (and finished) a tool during the window, half a
    # minute before the check's own write
    edit = solver_edit_during_check(str(tmp_path), str(tmp_path / "elsewhere"),
                                    "x", "p0", offset=-30.0)
    (tmp_path / "solver.py").write_text(edit)
    script = (f"python3 {tmp_path / 'solver.py'}; "
              f"echo probe > {world}/probe.txt; cat {world}/keep.txt")
    game, backend, log = script_game(tmp_path, script)

    result = game.run(ROW, Obligation("c1", "keep.txt says as delivered",
                                      required_basis="world"))

    assert result.outcome == "failed"
    rejected = [e["payload"]["reason"] for e in log.replay()
                if e["type"] == "method_rejected"]
    assert any("altered the delivered world" in r and "probe.txt" in r
               for r in rejected)


class MovingThenStill:
    """Agent-path executor: the first run reports a moved world."""
    def __init__(self):
        self.calls = 0

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls += 1
        moved = {"added": ["out.txt"], "added_count": 1,
                 "solver_tools": ["Bash"], "solver_runs": 1, "timed": True}
        return DischargeResult(
            witness_kind="execution", outcome="refuted" if self.calls == 1
            else "holds", witness="out.txt missing" if self.calls == 1
            else None, artifact=f"d{self.calls}", instrument="stub",
            basis="world", world_moved=moved if self.calls == 1 else {})


def test_a_moved_agent_run_settles_nothing_and_is_rerun(tmp_path):
    oracle = {"methods": [{
        "procedure": "read the delivered file", "artifact": "notes",
        "level": "direct", "basis": "world", "strength": "decisive",
        "bridges": []}]}
    judge = {"sufficient": "yes", "faithful": "yes", "degradation": "n/a",
             "reason": "artifact checks out"}
    backend = FixtureBackend({"oracles": oracle, "judger": judge})
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    executor = MovingThenStill()
    game = BindingGame(backend, ArtifactStore(os.path.join(tmp_path, "art")),
                       log, executor)

    result = game.run(ROW, Obligation("c1", "the delivered file exists",
                                      required_basis="world"))

    # the refutation seen on a moving world never reached the judger
    assert result.outcome == "holds" and executor.calls == 2
    assert [c["role"] for c in backend.calls].count("judger") == 1
    assert not [e for e in log.replay() if e["type"] == "method_rejected"]


def test_moved_retries_are_bounded(tmp_path):
    class AlwaysMoving:
        calls = 0

        def discharge(self, row, obligation, method=None, tag=""):
            AlwaysMoving.calls += 1
            return DischargeResult(
                witness_kind="execution", outcome="holds", witness=None,
                artifact="d", instrument="stub", basis="world",
                world_moved={"modified": ["a"], "modified_count": 1,
                             "solver_tools": ["Bash"]})

    oracle = {"methods": [{
        "procedure": "read the delivered file", "artifact": "notes",
        "level": "direct", "basis": "world", "strength": "decisive",
        "bridges": []}]}
    backend = FixtureBackend({"oracles": oracle, "judger": {}})
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    game = BindingGame(backend, ArtifactStore(os.path.join(tmp_path, "art")),
                       log, AlwaysMoving())

    result = game.run(ROW, Obligation("c1", "the delivered file exists",
                                      required_basis="world"))

    assert result.outcome == "failed"
    assert AlwaysMoving.calls == 1 + MOVED_RETRIES
    assert "changed during every check" in str(result.witness)
    # nothing was judged and no repair was asked for: there was nothing
    # about any method to repair
    roles = [c["role"] for c in backend.calls]
    assert "judger" not in roles and roles.count("oracles") == 1


def test_briefings_skip_runs_that_say_nothing_about_their_method():
    moved = Attempt(method={"procedure": "read it"}, moved=True,
                    rejected="the delivered world changed during the check")
    judged = Attempt(method={"procedure": "diff it"},
                     rejected="judger method insufficient: no diff printed",
                     discharge_outcome="holds")
    text = prior_attempt_briefing([moved, judged])
    assert "no diff printed" in text and "changed during the check" not in text


# -- the kernel stamps the epoch the check began in -----------------------

def test_evidence_carries_the_epoch_its_check_began_in(tmp_path):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))

    class RecordsSolverActsMidCheck:
        seen = 0

        def discharge(self, row, obligation, method=None, tag=""):
            RecordsSolverActsMidCheck.seen += 1
            if RecordsSolverActsMidCheck.seen == 1:
                # the server ingests the solver's edits while this check
                # is still running
                log.append("tool_call", {"tool": "Bash",
                                         "input": {"command": "sed -i"}})
            return DischargeResult(witness_kind="execution", outcome="holds",
                                   witness=None, artifact="", instrument="stub",
                                   basis="world")

    def compile_role(_v):
        return {"obligations": [{"text": "the delivered file is right",
                                 "required_basis": "world"}]}

    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    Kernel(store, log, FixtureBackend({"obligations": compile_role}),
           ArtifactStore(os.path.join(tmp_path, "art")), critic_role=None,
           discharger=RecordsSolverActsMidCheck(), live_world=True)

    first = store.add_claim(Claim(CLAIM_ASSERTION, "first"))
    node = store.obligation_children(first)[0]
    assert store.evidence_for(node["id"])[-1]["world_epoch"] == 0
    # the world moved while that check ran: the next claim on the same
    # obligation checks afresh instead of reusing the verdict
    store.add_claim(Claim(CLAIM_ASSERTION, "second"))
    assert RecordsSolverActsMidCheck.seen == 2
