"""An explicit revision invalidates the evidence used by its consumers."""
import json

import pytest


@pytest.fixture(autouse=True)
def blocking_commits(monkeypatch):
    """These tests read each verdict off the commit call itself (the final's
    path, and every claim's with GV_BACKGROUND_CHECKS=0); the background
    path's revision race is covered in tests/test_background_checks.py."""
    monkeypatch.setenv("GV_BACKGROUND_CHECKS", "0")

from verifier.core.model import Claim, CLAIM_ASSERTION, CLAIM_FINAL
from verifier.core.store import ClaimStore, TRIGGER_LAZY, TRIGGER_ON_COMMIT
from verifier.engine.certificate import request_certificate
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.llm.schema import FixtureBackend
from verifier.mcp_server import Service


def service(tmp_path, *, inherit=False, trigger=TRIGGER_LAZY):
    answer = tmp_path / "answer.txt"
    calls = []

    def factory(sc):
        class Check:
            def discharge(self, row, obligation, method=None, tag=""):
                calls.append(row["id"])
                observed = answer.read_text()
                witness = {"expected": "15", "observed": observed}
                return DischargeResult(
                    outcome="holds" if observed == "15" else "refuted",
                    witness=witness, artifact=sc.artifacts.put(witness),
                    instrument="read-file", basis="world", strength="decisive",
                    witness_kind="world_state")

        backend = FixtureBackend({"obligations": lambda v: {"obligations": [{
            "text": ("the answer is right" if inherit and v["claim_text"] == "final"
                     else "the answer file contains 15"),
            "required_basis": "world"}]}})
        return Kernel(sc.store, sc.log, backend, sc.artifacts,
                      critic_role=None, discharger=Check())

    return Service(str(tmp_path / "state"), kernel_factory=factory, trigger=trigger), answer, calls, factory


@pytest.mark.parametrize("original,revised", [("15", "14"), ("14", "15")])
@pytest.mark.parametrize("inherit", [False, True])
def test_revision_invalidates_shared_and_inherited_verdicts(tmp_path, original, revised, inherit):
    sc, answer, calls, _ = service(tmp_path, inherit=inherit)
    answer.write_text(original)
    source = sc.store.add_claim(Claim(CLAIM_ASSERTION, "the answer is right"))
    final = sc.store.add_claim(Claim(CLAIM_FINAL, "final"))
    first_status = "verified" if original == "15" else "refuted"
    assert sc.kernel.verify(source) == sc.kernel.verify(final) == first_status
    assert len(calls) == 1

    answer.write_text(revised)
    replacement = sc.store.add_claim(Claim(
        CLAIM_ASSERTION, "the corrected answer is right", revises=source))
    assert sc.store.status(final) == "stale"
    expected = "verified" if revised == "15" else "refuted"
    assert sc.kernel.verify(replacement) == expected
    assert len(calls) == 2
    cert = request_certificate(sc.store, sc.kernel, sc.log, sc.artifacts, final)
    assert cert["verdict"] == expected
    node, = sc.store.obligation_children(final)
    latest = sc.store.evidence_for(node["id"])[-1]
    assert latest["outcome"] == ("holds" if revised == "15" else "refuted")
    evidence_path = tmp_path / "state/artifacts" / (latest["artifact"] + ".json")
    assert json.loads(evidence_path.read_text())["observed"] == revised


def test_legacy_revision_judgments_are_invalidated_once(tmp_path):
    path = str(tmp_path / "legacy.sqlite")
    store = ClaimStore(path, trigger=TRIGGER_LAZY)
    base = store.add_claim(Claim(CLAIM_ASSERTION, "original"))
    legacy = store.add_claim(Claim(CLAIM_ASSERTION, "revised"))
    consumer = store.add_claim(Claim(CLAIM_ASSERTION, "consumer", deps=(legacy,)))
    for cid in (base, legacy, consumer):
        store.set_verified(cid, basis="world")
    # Reproduce a pre-migration store: revised text used the same identity
    # as a non-revising claim, with lineage kept only in the row.
    store.db.execute("UPDATE claims SET revises=? WHERE id=?", (base, legacy))
    store.db.execute("DELETE FROM store_metadata")
    store.db.commit()
    store.db.close()

    migrated = ClaimStore(path, trigger=TRIGGER_LAZY)
    assert migrated.status(legacy) == migrated.status(consumer) == "stale"
    assert migrated.status(base) == "verified"
    replacement = migrated.add_claim(Claim(CLAIM_ASSERTION, "revised", revises=legacy))
    assert replacement != legacy
    migrated.set_verified(replacement, basis="world")
    migrated.db.close()
    reopened = ClaimStore(path, trigger=TRIGGER_LAZY)
    assert reopened.status(replacement) == "verified"


def test_same_statement_can_be_revised_and_replayed_without_rechecking(tmp_path):
    sc, answer, calls, factory = service(tmp_path, trigger=TRIGGER_ON_COMMIT)
    answer.write_text("14")
    first = sc.commit("A1", "assertion", "the answer is right", [])
    assert first["status"] == "refuted"
    assert sc.commit("A1", "assertion", "the answer is right", []) == first
    assert len(calls) == 1

    answer.write_text("15")
    revised = sc.commit("A2", "assertion", "the answer is right", [], revises="A1")
    assert revised["status"] == "verified"
    assert revised["claim_id"] != first["claim_id"]
    assert sc.store.status(first["claim_id"]) == "stale"
    assert len(calls) == 2
    restarted = Service(sc.state_dir, kernel_factory=factory)
    assert restarted.builder.resolve("A2") == revised["claim_id"]
    assert restarted.store.status(revised["claim_id"]) == "verified"
    assert len(calls) == 2
