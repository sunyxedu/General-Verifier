"""Evidence-basis provenance: what a "verified" entitles others to rely on.

world  — content established independently of the solver's reporting.
record — the ledger attests the solver did/reported it; proves reporting.

The claim-level basis is a taint fold (any record component -> record,
verified assertion premises included, effect premises exempt because they
reach the compiler opaque). The reusable atoms are the obligation NODES,
each carrying the basis of the method that discharged it.
"""
from __future__ import annotations

import os

from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_RECORD, BASIS_WORLD, Claim, CLAIM_ASSERTION,
)
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend


COMPILE = lambda v: {"obligations": [{"text": f"O of: {v['claim_text']}", "required_basis": "record"}]}  # noqa: E731


class BasisDischarger:
    """Stub answering holds with a fixed or per-obligation basis."""

    def __init__(self, basis=BASIS_WORLD):
        self.basis = basis

    def discharge(self, row, obligation, method=None, tag=""):
        basis = (self.basis(obligation.text) if callable(self.basis)
                 else self.basis)
        return DischargeResult(witness_kind="execution", outcome="holds", witness=None,
                               artifact="", instrument="stub", basis=basis)


def build(tmp_path, compile_fixture=COMPILE, basis=BASIS_WORLD):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": compile_fixture})
    Kernel(store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
           critic_role=None, discharger=BasisDischarger(basis))
    return store, backend


def test_world_discharge_yields_world_claim(tmp_path):
    store, _ = build(tmp_path, basis=BASIS_WORLD)
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 42"))
    assert store.status(cid) == "verified"
    assert store.get(cid)["basis"] == BASIS_WORLD


def test_record_discharge_yields_record_claim(tmp_path):
    store, _ = build(tmp_path, basis=BASIS_RECORD)
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "as of effect:2, it ran"))
    assert store.status(cid) == "verified"
    assert store.get(cid)["basis"] == BASIS_RECORD


def test_mixed_obligations_taint_to_record(tmp_path):
    """One record leg in the sufficiency set taints the whole claim — but
    the world-based obligation NODE keeps its own basis for reuse."""
    two = lambda v: {"obligations": [                              # noqa: E731
        {"text": "independent recomputation gives 42",
         "required_basis": "record"},
        {"text": "the ledger records the solver printing 42",
         "required_basis": "record"},
    ]}
    per_text = lambda text: (                                      # noqa: E731
        BASIS_RECORD if "ledger" in text else BASIS_WORLD)
    store, _ = build(tmp_path, compile_fixture=two, basis=per_text)

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 42"))
    assert store.status(cid) == "verified"
    assert store.get(cid)["basis"] == BASIS_RECORD

    bases = {o["text"]: o["basis"] for o in store.obligation_children(cid)}
    assert bases["independent recomputation gives 42"] == BASIS_WORLD
    assert bases["the ledger records the solver printing 42"] == BASIS_RECORD


def test_record_premise_taints_dependent_claim(tmp_path):
    """A chain of reliance must bottom out in world checks. Under
    conditional semantics the taint is charged at read time: the LOCAL
    basis stays the claim's own (world — its obligations were world
    discharged), while the EFFECTIVE fold caps the chain at record
    because the premise's content only ever entered via the solver's
    reporting."""
    per_text = lambda text: (                                      # noqa: E731
        BASIS_RECORD if "reported" in text else BASIS_WORLD)
    store, _ = build(tmp_path, basis=per_text)

    a = store.add_claim(Claim(CLAIM_ASSERTION, "the solver reported 42"))
    assert store.get(a)["basis"] == BASIS_RECORD

    b = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 42", deps=(a,)))
    assert store.status(b) == "verified"
    assert store.get(b)["basis"] == BASIS_WORLD
    effective = store.effective(b)
    assert effective["effective_status"] == "verified"
    assert effective["effective_basis"] == BASIS_RECORD


def test_effect_premise_does_not_taint(tmp_path):
    """Effect premises reach the compiler opaque, so they cannot have
    contributed semantic content — a world-discharged claim citing one
    stays world. The effect claim itself is record by construction."""
    from verifier.stream.builder import Builder

    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": COMPILE})
    Kernel(store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
           critic_role=None, discharger=BasisDischarger(BASIS_WORLD))
    builder = Builder(log, store)

    builder.feed(log.append("tool_call", {"tool": "Bash"}))
    builder.feed(log.append("tool_result", {"stdout": "42"}))
    builder.feed(log.append("commit", {
        "label": "A1", "kind": "assertion",
        "text": "the answer is 42", "deps": ["effect:2"]}))

    effect = builder.labels["effect:2"]
    assert store.get(effect)["basis"] == BASIS_RECORD
    assert store.get(builder.labels["A1"])["basis"] == BASIS_WORLD


def test_premise_labels_carry_basis(tmp_path):
    """The compiler sees [world-level]/[record-level] tags on verified
    assertion premises."""
    per_text = lambda text: (                                      # noqa: E731
        BASIS_RECORD if "reported" in text else BASIS_WORLD)
    store, backend = build(tmp_path, basis=per_text)

    a = store.add_claim(Claim(CLAIM_ASSERTION, "the solver reported 42"))
    b = store.add_claim(Claim(CLAIM_ASSERTION, "the count is 42"))
    store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 42", deps=(a, b)))

    variables = backend.calls[-1]["variables"]
    assert "[record-level] the solver reported 42" in variables["verified_deps"]
    assert "[world-level] the count is 42" in variables["verified_deps"]


def test_inheritance_transfers_basis(tmp_path):
    """An obligation node inheriting a same-text claim's verdict inherits
    its basis too — changing identity space does not launder provenance."""
    fixtures = {
        # first claim: restates itself; second claim: demands the first's text
        "the solver reported 42":
            {"obligations": [{"text": "the solver reported 42",
                              "required_basis": "record"}]},
        "the answer is 42":
            {"obligations": [{"text": "the solver reported 42",
                              "required_basis": "record"}]},
    }
    compile_fixture = lambda v: fixtures[v["claim_text"]]          # noqa: E731
    store, _ = build(tmp_path, compile_fixture=compile_fixture,
                     basis=BASIS_RECORD)

    store.add_claim(Claim(CLAIM_ASSERTION, "the solver reported 42"))
    b = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 42"))

    assert store.status(b) == "verified"
    # the shared obligation node carries the inherited record basis, and it
    # (not a world check) is what settled claim b -> b is record too
    assert store.get(b)["basis"] == BASIS_RECORD


def test_store_migration_adds_basis_column(tmp_path):
    """A store created before basis tracking opens cleanly and reports
    basis=None for its rows."""
    import sqlite3

    path = os.path.join(tmp_path, "claims.sqlite")
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE claims (id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
        "text TEXT NOT NULL, revises TEXT, "
        "status TEXT NOT NULL DEFAULT 'unverified', reason TEXT);"
        "CREATE TABLE deps (child TEXT NOT NULL, parent TEXT NOT NULL, "
        "PRIMARY KEY (child, parent));"
        "CREATE TABLE obligation_edges (claim TEXT NOT NULL, "
        "obligation TEXT NOT NULL, PRIMARY KEY (claim, obligation));"
        "CREATE TABLE evidence (claim_id TEXT NOT NULL, payload TEXT NOT NULL);"
    )
    db.execute("INSERT INTO claims (id, kind, text) VALUES ('x', 'assertion', 'old row')")
    db.commit()
    db.close()

    store = ClaimStore(path)
    assert store.get("x")["basis"] is None


def test_an_effect_is_named_by_either_sequence_of_its_call_result_pair(tmp_path):
    """The record shows two numbered lines per effect; a reader citing the
    call's number means the same effect as one citing the result's."""
    from verifier.core.events import EventLog
    from verifier.core.store import ClaimStore
    from verifier.stream.builder import Builder, TraceError
    import pytest
    log = EventLog(str(tmp_path / "events.jsonl"))
    store = ClaimStore(str(tmp_path / "claims.sqlite"))
    b = Builder(log, store)
    call = log.append("tool_call", {"tool": "exec", "tool_use_id": "c1", "input": "ls"})
    result = log.append("tool_result", {"tool": "exec", "tool_use_id": "c1", "excerpt": "a.txt"})
    b.feed(call); b.feed(result)
    assert b.resolve(f"effect:{call['seq']}") == b.resolve(f"effect:{result['seq']}")
    with pytest.raises(TraceError):
        b.resolve(f"effect:{result['seq'] + 1}")


def test_two_event_logs_on_one_file_never_reuse_a_sequence_number(tmp_path):
    """Sequence numbers come from the file, under a lock: a second EventLog
    on the same path (another thread's instance, a helper process) cannot
    hand out a number the first already used, so replay never sees a
    duplicate."""
    from verifier.core.events import EventLog
    path = str(tmp_path / "events.jsonl")
    a = EventLog(path)
    b = EventLog(path)          # both start believing the ledger is empty
    a.append("message", {"role": "assistant", "text": "one"})
    b.append("message", {"role": "assistant", "text": "two"})
    a.append("message", {"role": "assistant", "text": "three"})
    b.append("message", {"role": "assistant", "text": "four"})
    seqs = [e["seq"] for e in EventLog(path).replay()]
    assert seqs == [1, 2, 3, 4]



def test_environment_record_names_the_task_path_not_the_verifier_venv(tmp_path, monkeypatch):
    """The baseline environment is taken from inside the verifier's venv;
    the record must describe the task's PATH (venv removed) and where the
    task's interpreters resolve on it, so a runnability check reproduces
    the task's invocation rather than one of its own."""
    import os as _os
    from verifier import baseline
    from verifier.llm import provider
    fake_bin = tmp_path / "taskbin"; fake_bin.mkdir()
    py = fake_bin / "python"; py.write_text("#!/bin/sh\nexit 1\n"); py.chmod(0o755)
    venv_bin = _os.path.join(provider.ROOT, ".venv", "bin")
    monkeypatch.setenv("PATH", venv_bin + _os.pathsep + str(fake_bin))
    rec = baseline.environment_fingerprint()
    assert rec["path"] == str(fake_bin)
    assert rec["interpreters"]["python"] == str(py)
    assert "python" in rec["executables"]


def test_world_view_names_the_task_root_for_bare_file_addresses(tmp_path):
    """A task that says "the file chess_board.png" addresses it relative to
    its working root; the world description must say what that root is
    (and where it lives when captured), or a seat resolves the bare name
    against its own state directory and reports the file absent."""
    import json as _json
    from verifier.core.world import load_world_view
    state = tmp_path / "gv"; (state / "baseline").mkdir(parents=True)
    (state / "baseline" / "manifest.json").write_text(_json.dumps(
        {"root": "/app", "files": {}}))
    live = load_world_view(str(state), live_world=True).describe()
    assert live["task_root"] == "/app"
    assert "bare file name" in live["relative_addresses"]
    (state / "world" / "app").mkdir(parents=True)
    (state / "world_manifest.json").write_text(_json.dumps(
        {"captured_roots": ["/app"]}))
    captured = load_world_view(str(state), live_world=False).describe()
    assert captured["task_root"] == "/app"
    assert str(state / "world" / "app") in captured["relative_addresses"]

