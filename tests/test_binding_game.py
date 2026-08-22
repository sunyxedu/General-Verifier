"""The binding game's audit surface: what the judger actually sees.

Fixture-backed. The provenance fix hangs on the judger auditing the
oracle's declared evidence basis — so the declaration must reach the
judger's variables, and the settled result must carry the audited basis.
"""
from __future__ import annotations

import os

from verifier.core.events import EventLog
from verifier.core.model import BASIS_WORLD, Obligation
from verifier.engine.game import BindingGame
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend


ROW = {"id": "c1", "kind": "assertion", "text": "the answer is 42"}

ORACLE = {"methods": [{
    "procedure": "recompute the answer from the task's own inputs",
    "artifact": "check script and output",
    "level": "independent recomputation",
    "basis": "world",
}]}

JUDGE_YES = {"faithful": "yes", "reason": "artifact checks out"}


class StubDischarger:
    def discharge(self, row, obligation, method=None, tag=""):
        return DischargeResult(outcome="holds", witness=None,
                               artifact="d1", instrument="stub")


def game(tmp_path, roles):
    backend = FixtureBackend(roles)
    return BindingGame(
        backend,
        ArtifactStore(os.path.join(tmp_path, "art")),
        EventLog(os.path.join(tmp_path, "events.jsonl")),
        StubDischarger(),
    ), backend


def test_judger_sees_the_declared_basis(tmp_path):
    g, backend = game(tmp_path, {"oracles": ORACLE, "judger": JUDGE_YES})

    result = g.run(ROW, Obligation("c1", "answer equals 42"))

    judger_calls = [c for c in backend.calls if c["role"] == "judger"]
    assert judger_calls, "the holds outcome must face the judger"
    assert judger_calls[-1]["variables"]["basis"] == "world"
    # and the audited declaration is what the settled result carries
    assert result.outcome == "holds"
    assert result.basis == BASIS_WORLD


def test_missing_declaration_fails_closed_to_record(tmp_path):
    legacy = {"methods": [{"procedure": "p", "artifact": "a", "level": "l",
                           "basis": "record"}]}
    g, backend = game(tmp_path, {"oracles": legacy, "judger": JUDGE_YES})

    result = g.run(ROW, Obligation("c1", "answer equals 42"))

    assert result.basis == "record"
    assert backend.calls[-1]["variables"]["basis"] == "record"
