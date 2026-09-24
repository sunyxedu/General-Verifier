"""A claim whose complete check could not settle it is not checked again on
the same world: every queued certificate request used to rerun the whole
check (30-100 minutes each) on unchanged files and got the same answer. Any
change to the delivered files — content, mode, a new entry — reopens it, and
an explicit gv.verify always reruns it."""
from __future__ import annotations

import json
import os

import pytest

from verifier.core.model import Claim, CLAIM_ASSERTION
from verifier.core.store import TRIGGER_LAZY
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.llm.schema import FixtureBackend
from verifier.mcp_server import Service


@pytest.fixture(autouse=True)
def blocking_commits(monkeypatch):
    monkeypatch.setenv("GV_BACKGROUND_CHECKS", "0")


def build(tmp_path, *, live_world=True, with_baseline=True):
    state = tmp_path / "state"
    world = tmp_path / "world"
    world.mkdir()
    (world / "out.txt").write_text("partial\n")
    if with_baseline:
        (state / "baseline").mkdir(parents=True)
        (state / "baseline" / "manifest.json").write_text(
            json.dumps({"root": str(world), "files": {}}))
    calls = []

    def factory(sc):
        class Unsettled:
            def discharge(self, row, obligation, method=None, tag=""):
                calls.append(obligation.text)
                witness = {"why": "no instrument reached the condition"}
                return DischargeResult(
                    outcome="failed", witness=witness,
                    artifact=sc.artifacts.put(witness), instrument="none",
                    basis="world", strength="decisive")

        backend = FixtureBackend({"obligations": lambda v: {"obligations": [{
            "text": "out.txt holds the finished report",
            "required_basis": "world"}]}})
        return Kernel(sc.store, sc.log, backend, sc.artifacts,
                      critic_role=None, discharger=Unsettled(),
                      live_world=live_world)

    svc = Service(str(state), kernel_factory=factory, trigger=TRIGGER_LAZY,
                  live_world=live_world)
    return svc, world, calls


def skipped(svc):
    return [e for e in svc.log.replay() if e["type"] == "verify_skipped"]


def test_an_unsettled_claim_is_not_rechecked_on_the_same_world(tmp_path):
    svc, world, calls = build(tmp_path)
    claim = svc.store.add_claim(Claim(CLAIM_ASSERTION, "the report is finished"))
    assert svc.kernel.verify(claim) == "unverified"
    ran = len(calls)
    assert ran >= 1
    assert svc.kernel.verify(claim) == "unverified"
    assert len(calls) == ran
    assert claim in svc.kernel.not_rechecked and len(skipped(svc)) == 1


@pytest.mark.parametrize("change", ["content", "mode", "new-entry"])
def test_any_change_to_the_files_reopens_it(tmp_path, change):
    svc, world, calls = build(tmp_path)
    claim = svc.store.add_claim(Claim(CLAIM_ASSERTION, "the report is finished"))
    svc.kernel.verify(claim)
    ran = len(calls)
    if change == "content":
        (world / "out.txt").write_text("finished\n")
    elif change == "mode":
        os.chmod(world / "out.txt", 0o755)
    else:
        (world / "logs").mkdir()
    svc.kernel.verify(claim)
    assert len(calls) > ran
    assert claim not in svc.kernel.not_rechecked and not skipped(svc)


def test_an_explicit_verify_always_reruns(tmp_path):
    svc, world, calls = build(tmp_path)
    svc.commit("A1", "assertion", "the report is finished", [])
    ran = len(calls)
    out = svc.verify("A1")
    assert out["status"] == "unverified" and len(calls) > ran


def test_the_verifiers_own_state_under_the_root_is_not_the_world(tmp_path):
    svc, world, calls = build(tmp_path)
    claim = svc.store.add_claim(Claim(CLAIM_ASSERTION, "the report is finished"))
    svc.kernel.verify(claim)
    ran = len(calls)
    (world / ".gv").mkdir()
    (world / ".gv" / "hook_feed.jsonl").write_text("{}\n")
    (world / "__pycache__").mkdir()
    (world / "__pycache__" / "m.cpython-312.pyc").write_bytes(b"\0")
    svc.kernel.verify(claim)
    assert len(calls) == ran


def test_the_certificate_marks_a_claim_it_did_not_recheck(tmp_path):
    svc, world, calls = build(tmp_path)
    svc.commit("F", "final", "the report is finished", [])
    svc.kernel.verify(svc.builder.labels["F"])
    ran = len(calls)
    assert ran >= 1
    cert = svc.certificate("F")
    assert len(calls) == ran
    entry, = [e for e in cert["slice"] if e["text"] == "the report is finished"]
    assert "not_rechecked" in entry


@pytest.mark.parametrize("live_world,with_baseline", [(False, True), (True, False)])
def test_without_a_world_to_fingerprint_every_request_checks(tmp_path, live_world,
                                                            with_baseline):
    svc, world, calls = build(tmp_path, live_world=live_world,
                              with_baseline=with_baseline)
    claim = svc.store.add_claim(Claim(CLAIM_ASSERTION, "the report is finished"))
    svc.kernel.verify(claim)
    ran = len(calls)
    svc.kernel.verify(claim)
    assert len(calls) > ran and not skipped(svc)


def test_a_superseded_claim_stays_invalidated_when_a_late_check_settles(tmp_path):
    svc, world, calls = build(tmp_path)
    old = svc.store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 14"))
    svc.store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 15", revises=old))
    assert svc.store.status(old) == "stale"
    svc.store.set_verified(old)
    svc.store.set_refuted(old, reason="late")
    assert svc.store.status(old) == "stale"


def test_a_superseded_claim_is_not_checked(tmp_path):
    svc, world, calls = build(tmp_path)
    old = svc.store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 14"))
    svc.store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 15", revises=old))
    assert svc.kernel.verify(old) == "stale" and not calls
