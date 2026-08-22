"""The binding game's audit surface: what the judger actually sees.

Fixture-backed. The provenance fix hangs on the judger auditing the
oracle's declared evidence basis — so the declaration must reach the
judger's variables, and the settled result must carry the audited basis.
"""
from __future__ import annotations

import hashlib
import json
import os

from verifier.core.events import EventLog
from verifier.core.model import BASIS_WORLD, Obligation, content_hash
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


class SequenceDischarger:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def discharge(self, row, obligation, method=None, tag=""):
        outcome, witness = next(self.outcomes)
        return DischargeResult(outcome=outcome, witness=witness,
                               artifact="d1", instrument="stub")


def game(tmp_path, roles, live_world=True):
    backend = FixtureBackend(roles)
    return BindingGame(
        backend,
        ArtifactStore(os.path.join(tmp_path, "art")),
        EventLog(os.path.join(tmp_path, "events.jsonl")),
        StubDischarger(),
        live_world=live_world,
    ), backend


def test_artifact_store_digest_hashes_canonical_parsed_json(tmp_path):
    """The pretty on-disk envelope intentionally has a different byte hash."""
    store = ArtifactStore(os.path.join(tmp_path, "art"))
    digest = store.put({"z": "λ", "nested": {"b": 2, "a": 1}})
    path = tmp_path / "art" / f"{digest}.json"

    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert content_hash(parsed) == digest
    assert hashlib.sha256(path.read_bytes()).hexdigest() != digest


def test_judger_sees_the_declared_basis(tmp_path):
    g, backend = game(tmp_path, {"oracles": ORACLE, "judger": JUDGE_YES})

    result = g.run(ROW, Obligation("c1", "answer equals 42"))

    judger_calls = [c for c in backend.calls if c["role"] == "judger"]
    assert judger_calls, "the holds outcome must face the judger"
    assert judger_calls[-1]["variables"]["basis"] == "world"
    # and the audited declaration is what the settled result carries
    assert result.outcome == "holds"
    assert result.basis == BASIS_WORLD


def test_judger_sees_world_and_manifest_when_present(tmp_path):
    """The judger audits refutations against the delivered world and its
    capture manifest — both paths must reach its variables (derived from
    the ledger's directory, same as the kernel and the executor)."""
    os.makedirs(os.path.join(tmp_path, "world", "app"))
    manifest = os.path.join(tmp_path, "world_manifest.json")
    with open(manifest, "w") as f:
        json.dump({"captured_roots": ["/app"]}, f)
    g, backend = game(tmp_path, {"oracles": ORACLE, "judger": JUDGE_YES})

    g.run(ROW, Obligation("c1", "answer equals 42"))

    judger_vars = [c for c in backend.calls if c["role"] == "judger"][-1][
        "variables"]
    assert judger_vars["world_path"] == os.path.join(str(tmp_path), "world")
    assert judger_vars["world_manifest"] == manifest
    assert judger_vars["world_mode"] == "captured"
    assert judger_vars["ledger_path"] == os.path.join(str(tmp_path),
                                                       "events.jsonl")


def test_judger_world_vars_empty_when_absent(tmp_path):
    g, backend = game(tmp_path, {"oracles": ORACLE, "judger": JUDGE_YES})

    g.run(ROW, Obligation("c1", "answer equals 42"))

    judger_vars = [c for c in backend.calls if c["role"] == "judger"][-1][
        "variables"]
    assert judger_vars["world_path"] == ""
    assert judger_vars["world_manifest"] == ""
    assert judger_vars["world_mode"] == "live"


def test_judger_knows_when_offline_world_was_not_captured(tmp_path):
    g, backend = game(
        tmp_path, {"oracles": ORACLE, "judger": JUDGE_YES},
        live_world=False,
    )

    g.run(ROW, Obligation("c1", "answer equals 42"))

    variables = [c for c in backend.calls if c["role"] == "judger"][-1][
        "variables"]
    assert variables["world_mode"] == "unavailable"
    assert variables["world_path"] == ""


def test_oracle_sees_floor_ledger_and_fixed_world_context(tmp_path):
    os.makedirs(os.path.join(tmp_path, "world", "app"))
    manifest = os.path.join(tmp_path, "world_manifest.json")
    with open(manifest, "w") as f:
        json.dump({"captured_roots": ["/app"]}, f)
    g, backend = game(
        tmp_path, {"oracles": ORACLE, "judger": JUDGE_YES},
        live_world=False,
    )

    g.run(ROW, Obligation("c1", "answer equals 42", "world"))

    variables = [c for c in backend.calls if c["role"] == "oracles"][-1][
        "variables"]
    assert variables["required_basis"] == "world"
    assert variables["ledger_path"] == os.path.join(str(tmp_path),
                                                       "events.jsonl")
    assert variables["world_mode"] == "captured"
    assert variables["world_path"] == os.path.join(str(tmp_path), "world")
    assert variables["world_manifest"] == manifest


def test_explicit_empty_oracle_is_unobtainable_offline(tmp_path):
    g, _ = game(tmp_path, {"oracles": {"methods": []}}, live_world=False)

    result = g.run(ROW, Obligation("c1", "past state was X", "world"))

    assert result.outcome == "unobtainable"
    assert "no executable sufficient method" in result.witness


def test_explicit_empty_oracle_stays_failed_live(tmp_path):
    g, _ = game(tmp_path, {"oracles": {"methods": []}})

    result = g.run(ROW, Obligation("c1", "answer equals 42", "world"))

    assert result.outcome == "failed"
    assert "no executable sufficient method" in result.witness


def test_oracle_role_error_stays_failed_offline(tmp_path):
    g, _ = game(tmp_path, {}, live_world=False)

    result = g.run(ROW, Obligation("c1", "past state was X", "world"))

    assert result.outcome == "failed"
    assert "oracle enumeration failed" in result.witness


def test_schema_invalid_oracle_output_stays_failed_offline(tmp_path):
    g, backend = game(
        tmp_path, {"oracles": {"methods": "not-a-list"}},
        live_world=False,
    )

    result = g.run(ROW, Obligation("c1", "past state was X", "world"))

    assert result.outcome == "failed"
    assert "oracle enumeration failed" in result.witness
    assert len([c for c in backend.calls if c["role"] == "oracles"]) == 2


def test_only_below_floor_methods_are_protocol_failure_offline(tmp_path):
    methods = {"methods": [{
        "procedure": "grep the ledger",
        "artifact": "ledger excerpt",
        "level": "record inspection",
        "basis": "record",
    }]}
    g, _ = game(tmp_path, {"oracles": methods}, live_world=False)

    result = g.run(ROW, Obligation("c1", "answer equals 42", "world"))

    assert result.outcome == "failed"
    assert "oracle protocol error" in result.witness
    assert "required basis 'world'" in result.witness


def test_missing_declaration_fails_closed_to_record(tmp_path):
    legacy = {"methods": [{"procedure": "p", "artifact": "a", "level": "l",
                           "basis": "record"}]}
    g, backend = game(tmp_path, {"oracles": legacy, "judger": JUDGE_YES})

    result = g.run(ROW, Obligation("c1", "answer equals 42"))

    assert result.basis == "record"
    assert backend.calls[-1]["variables"]["basis"] == "record"


def test_all_unavailable_methods_preserve_the_concrete_gap(tmp_path):
    methods = {"methods": [
        {"procedure": "read the retained output", "artifact": "output",
         "level": "direct", "basis": "world"},
        {"procedure": "read the contemporaneous snapshot",
         "artifact": "snapshot", "level": "direct", "basis": "world"},
    ]}
    backend = FixtureBackend({"oracles": methods})
    game = BindingGame(
        backend,
        ArtifactStore(os.path.join(tmp_path, "art")),
        EventLog(os.path.join(tmp_path, "events.jsonl")),
        SequenceDischarger([
            ("unobtainable", "result excerpt was truncated"),
            ("unobtainable", "no contemporaneous snapshot was retained"),
        ]),
    )

    result = game.run(ROW, Obligation("c1", "past state was X", "world"))

    assert result.outcome == "unobtainable"
    assert "result excerpt was truncated" in result.witness
    assert "no contemporaneous snapshot was retained" in result.witness


def test_retryable_failure_is_not_hidden_by_an_unavailable_method(tmp_path):
    methods = {"methods": [
        {"procedure": "read the retained output", "artifact": "output",
         "level": "direct", "basis": "world"},
        {"procedure": "run a valid independent checker",
         "artifact": "checker output", "level": "direct", "basis": "world"},
    ]}
    backend = FixtureBackend({"oracles": methods})
    game = BindingGame(
        backend,
        ArtifactStore(os.path.join(tmp_path, "art")),
        EventLog(os.path.join(tmp_path, "events.jsonl")),
        SequenceDischarger([
            ("unobtainable", "result excerpt was truncated"),
            ("failed", "checker timed out"),
        ]),
    )

    result = game.run(ROW, Obligation("c1", "past state was X", "world"))

    assert result.outcome == "failed"
