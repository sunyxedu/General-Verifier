"""A final a revision made stale is checked again at once: the solver only
polls gv.status for its verdict, and nothing else reopened it."""
from __future__ import annotations

import time

from verifier.core.store import TRIGGER_LAZY
from verifier.mcp_server import Service


class Recorder:
    def __init__(self, service):
        self.service = service
        self.calls = []
        service.store.on_trigger(self.verify)

    def verify(self, claim_id, force=False):
        self.calls.append(claim_id)
        return self.service.store.status(claim_id)


def test_a_final_made_stale_is_rechecked_without_a_certificate_request(
        tmp_path, monkeypatch):
    svc = Service(str(tmp_path), trigger=TRIGGER_LAZY,
                  kernel_factory=lambda sc: Recorder(sc))
    svc.background_checks = True
    svc.store.on_stale(svc._recheck_stale_finals)
    svc.commit("A1", "assertion", "helper holds", [])
    svc.commit("FINAL", "final", "task done", ["A1"])
    final = svc.builder.labels["FINAL"]
    svc.kernel.calls.clear()
    svc.commit("A1", "assertion", "helper holds, revised", [], revises="A1")
    deadline = time.time() + 10
    while final not in svc.kernel.calls and time.time() < deadline:
        time.sleep(0.05)
    assert final in svc.kernel.calls
    scheduled = [e for e in svc.log.replay() if e["type"] == "recheck_scheduled"]
    assert scheduled and scheduled[0]["payload"]["claim"] == final
