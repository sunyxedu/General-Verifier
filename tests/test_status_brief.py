"""The status map says why a claim was refuted, not only that it was: a
client may abandon the pending commit that carried the residue, and the
solver's next look at the record is the status map."""
from __future__ import annotations

from verifier.core.model import Obligation
from verifier.core.store import TRIGGER_LAZY
from verifier.mcp_server import Service


class IdleKernel:
    """Never verifies on its own; the test settles the record by hand."""
    def __init__(self, service):
        self.service = service

    def verify(self, claim_id, force=False):
        return self.service.store.status(claim_id)


def test_the_status_map_carries_the_counterexample(tmp_path):
    service = Service(str(tmp_path), live_world=False, trigger=TRIGGER_LAZY,
                      kernel_factory=IdleKernel)
    service.commit("A1", "assertion", "the decoded text is abc", [])
    service.commit("A2", "assertion", "the file has three lines", [])
    store = service.store
    a1 = service.builder.labels["A1"]
    oid = store.add_obligation(a1, Obligation(a1, "the decode equals abc"))
    store.record_evidence(oid, {"outcome": "refuted",
                                "witness": "an independent render reads abd"})
    store.set_refuted(oid, reason="counterexample")
    store.set_refuted(a1, reason="obligation refuted")

    status = service.status()

    assert status["A2"] == "unverified"
    assert status["A1"].startswith("refuted: the decode equals abc")
    assert "an independent render reads abd" in status["A1"]
    # the single-label view still returns the full residue
    detail = service.status("A1")
    assert detail["obligations"][0]["reason"]
