"""Online, an intermediate claim's commit returns once the claim is recorded
and its checks run in the background; the final's commit still waits, the
certificate waits for every check in its slice without redoing any, and a
refutation found in the background reaches the solver as `news` exactly
once, with the failing obligation and its counterexample."""
from __future__ import annotations

import collections
import threading
import time

from verifier.core.model import Obligation
from verifier.core.store import TRIGGER_ON_COMMIT
from verifier.mcp_server import Service

CHECK = 0.8


class SlowKernel:
    """Idempotent and serialized per claim, like the real kernel; every
    check takes CHECK seconds; claims whose text is listed are refuted."""
    def __init__(self, service, refute=()):
        self.service = service
        self.refute = set(refute)
        self.calls = collections.Counter()
        self._locks = collections.defaultdict(threading.Lock)
        service.store.on_trigger(self.verify)

    def verify(self, claim_id, force=False):
        store = self.service.store
        with self._locks[claim_id]:
            if store.status(claim_id) not in ("unverified", "stale"):
                return store.status(claim_id)
            self.calls[claim_id] += 1
            time.sleep(CHECK)
            row = store.get(claim_id)
            if row["text"] in self.refute:
                oid = store.add_obligation(claim_id, Obligation(claim_id, "the decode equals abc"))
                store.record_evidence(oid, {"outcome": "refuted",
                                            "witness": "an independent render reads abd"})
                store.set_refuted(oid, reason="counterexample")
                store.set_refuted(claim_id, reason="obligation refuted")
            else:
                store.set_verified(claim_id)
            return store.status(claim_id)


def service(tmp_path, refute=(), live_world=True):
    return Service(str(tmp_path), live_world=live_world, trigger=TRIGGER_ON_COMMIT,
                   kernel_factory=lambda sc: SlowKernel(sc, refute))


def test_an_assertion_commit_returns_before_its_check(tmp_path):
    svc = service(tmp_path)
    t0 = time.time()
    out = svc.commit("A1", "assertion", "the file has three lines", [])
    assert time.time() - t0 < CHECK / 2
    assert out["status"] == "unverified" and out["checking"] is True
    # a later commit may cite it at once
    svc.commit("A2", "assertion", "the file is sorted", ["A1"])
    time.sleep(CHECK + 0.5)
    assert svc.status("A1")["status"] == "verified"


def test_a_final_commit_waits_for_its_check(tmp_path):
    svc = service(tmp_path)
    t0 = time.time()
    out = svc.commit("F", "final", "the task is complete", [])
    assert time.time() - t0 >= CHECK
    assert out["status"] == "verified" and "checking" not in out


def test_a_background_refutation_arrives_as_news_once(tmp_path):
    svc = service(tmp_path, refute={"the decoded text is abc"})
    svc.commit("A1", "assertion", "the decoded text is abc", [])
    time.sleep(CHECK + 0.4)
    first = svc.commit("A2", "assertion", "the output file holds it", [])
    news = first.get("news")
    assert news and news[0]["label"] == "A1" and news[0]["status"] == "refuted"
    assert "an independent render reads abd" in news[0]["why"]
    time.sleep(CHECK + 0.4)
    second = svc.status("A2")
    assert "news" not in second or all(n["label"] != "A1" for n in second["news"])


def test_the_certificate_waits_for_background_checks_without_redoing_them(tmp_path):
    svc = service(tmp_path)
    svc.commit("A1", "assertion", "the root cause is the stale cache", [])
    svc.commit("F", "final", "the task is complete", ["A1"])
    cert = svc.certificate("F")
    assert cert["verdict"] and "error" not in cert
    a1 = svc.builder.labels["A1"]
    assert svc.store.status(a1) == "verified"
    assert svc.kernel.calls[a1] == 1


def test_the_switch_restores_blocking_commits(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_BACKGROUND_CHECKS", "0")
    svc = service(tmp_path)
    t0 = time.time()
    out = svc.commit("A1", "assertion", "the file has three lines", [])
    assert time.time() - t0 >= CHECK and out["status"] == "verified"


def test_a_background_check_that_raises_is_logged_and_the_server_goes_on(tmp_path):
    class Exploding(SlowKernel):
        def verify(self, claim_id, force=False):
            if self.service.store.get(claim_id)["text"] == "boom":
                raise RuntimeError("seat crashed")
            return super().verify(claim_id, force)

    svc = Service(str(tmp_path), live_world=True, trigger=TRIGGER_ON_COMMIT,
                  kernel_factory=lambda sc: Exploding(sc))
    out = svc.commit("A1", "assertion", "boom", [])
    assert out["checking"] is True
    time.sleep(0.3)
    failed = [e["payload"] for e in svc.log.replay() if e["type"] == "background_check_failed"]
    assert failed and "seat crashed" in failed[0]["error"]
    assert svc.status("A1")["status"] == "unverified"
    assert svc.commit("F", "final", "the task is complete", [])["status"] == "verified"


def test_a_revision_returns_at_once_and_the_replaced_version_stays_invalidated(tmp_path):
    svc = service(tmp_path, refute={"the answer is 14"})
    svc.commit("A1", "assertion", "the answer is 14", [])
    # revised while A1's check is still running: the revision does not wait
    t0 = time.time()
    out = svc.commit("A2", "assertion", "the answer is 15", [], revises="A1")
    assert time.time() - t0 < CHECK / 2 and out["checking"] is True
    time.sleep(2 * CHECK + 0.5)
    a1, a2 = svc.builder.labels["A1"], svc.builder.labels["A2"]
    # A1's check finished after the revision arrived and settled nothing
    assert svc.kernel.calls[a1] == 1
    assert svc.store.status(a1) == "stale"
    assert svc.store.status(a2) == "verified"
    assert not any(n["label"] == "A1" for n in svc.status("A2").get("news", []))


def test_a_final_check_does_not_wait_for_the_checks_of_what_it_cites(tmp_path):
    svc = service(tmp_path)
    svc.commit("A1", "assertion", "the root cause is the stale cache", [])
    t0 = time.time()
    out = svc.commit("F", "final", "the task is complete", ["A1"])
    # A1's check (still running) and F's own ran side by side
    assert time.time() - t0 < 1.7 * CHECK
    assert out["status"] == "verified"
    cert = svc.certificate("F")
    a1 = svc.builder.labels["A1"]
    assert svc.store.status(a1) == "verified" and svc.kernel.calls[a1] == 1
    assert "error" not in cert


class Checking(SlowKernel):
    def checking(self, claim_id):
        return self._locks[claim_id].locked()


def test_status_marks_a_claim_whose_check_is_running(tmp_path):
    svc = Service(str(tmp_path), live_world=True, trigger=TRIGGER_ON_COMMIT,
                  kernel_factory=lambda sc: Checking(sc))
    svc.commit("A1", "assertion", "the file has three lines", [])
    time.sleep(0.1)
    assert svc.status("A1")["checking"] is True
    assert svc.status()["A1"] == "unverified (checking)"
    time.sleep(CHECK + 0.3)
    assert "checking" not in svc.status("A1")
    assert svc.status()["A1"] == "verified"
