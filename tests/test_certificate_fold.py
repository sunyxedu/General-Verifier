"""The certificate verdict is the FINAL's own adjudication.

The final's obligations are the task contract, so its status answers the
question the certificate exists to answer. History claims stay in the
exported slice (and in the verified-fraction features) but no longer cap
the verdict: a refuted or unverifiable narration step must not decide an
independently established deliverable — and vice versa, a healthy
narration must not lift an unestablished final.
"""
from __future__ import annotations

import os

import pytest

from verifier.core.events import EventLog
from verifier.core.model import Claim, CLAIM_ASSERTION, CLAIM_EFFECT, CLAIM_FINAL
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.engine.certificate import request_certificate
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend


COMPILE = lambda v: {"obligations": [{"text": f"O of: {v['claim_text']}", "required_basis": "record"}]}  # noqa: E731


class PerClaimDischarger:
    """Outcome chosen by the claim's text (holds unless told otherwise)."""

    def __init__(self, outcomes):
        self.outcomes = outcomes

    def discharge(self, row, obligation, method=None, tag=""):
        outcome = self.outcomes.get(row["text"], "holds")
        return DischargeResult(witness_kind="execution", outcome=outcome, witness="w" if outcome == "refuted" else None,
                               artifact="", instrument="stub", basis="world")


def build(tmp_path, outcomes):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": COMPILE})
    artifacts = ArtifactStore(os.path.join(tmp_path, "art"))
    kernel = Kernel(store, log, backend, artifacts, critic_role=None,
                    discharger=PerClaimDischarger(outcomes))
    return store, kernel, log, artifacts


def certify(tmp_path, outcomes):
    store, kernel, log, artifacts = build(tmp_path, outcomes)
    a = store.add_claim(Claim(CLAIM_ASSERTION, "the solver saw 63841 mid-run"))
    f = store.add_claim(Claim(CLAIM_FINAL, "the answer is 79586", deps=(a,)))
    cert = request_certificate(store, kernel, log, artifacts, f)
    return store, a, f, cert


def test_refuted_history_does_not_cap_a_verified_final(tmp_path):
    store, a, f, cert = certify(
        tmp_path, {"the solver saw 63841 mid-run": "refuted"})

    assert store.status(a) == "refuted"          # still on the record
    assert store.status(f) == "verified"         # adjudicated on its own contract
    assert cert["verdict"] == "verified"
    by_text = {e["text"]: e["status"] for e in cert["slice"]}
    assert by_text["the solver saw 63841 mid-run"] == "refuted"


def test_unverifiable_history_does_not_drag_a_verified_final(tmp_path):
    _, a, f, cert = certify(
        tmp_path, {"the solver saw 63841 mid-run": "failed"})

    assert cert["verdict"] == "verified"


def test_a_refuted_final_is_still_refuted(tmp_path):
    _, a, f, cert = certify(tmp_path, {"the answer is 79586": "refuted"})

    assert cert["verdict"] == "refuted"


def test_a_healthy_argument_cannot_lift_an_unestablished_final(tmp_path):
    _, a, f, cert = certify(tmp_path, {"the answer is 79586": "failed"})

    assert cert["verdict"] == "unverified"


def test_final_is_adjudicated_even_on_a_refuted_premise(tmp_path):
    """The old early-out ('depends on refuted claim') no longer spares the
    final: its task-contract obligations are compiled and discharged."""
    store, a, f, cert = certify(
        tmp_path, {"the solver saw 63841 mid-run": "refuted"})

    # the final's own obligation node exists and was settled
    children = store.obligation_children(f)
    assert [o["status"] for o in children] == ["verified"]


def test_assertions_keep_the_refuted_premise_early_out(tmp_path):
    """Non-final claims still fail closed on a collapsed premise — the
    exemption is for the deliverable's adjudication only."""
    store, kernel, log, artifacts = build(
        tmp_path, {"premise": "refuted"})
    a = store.add_claim(Claim(CLAIM_ASSERTION, "premise"))
    b = store.add_claim(Claim(CLAIM_ASSERTION, "conclusion", deps=(a,)))

    assert store.status(b) == "unverified"
    assert store.evidence_for(b)[-1]["error"].startswith("premise chain refuted")


@pytest.mark.parametrize("kind", [CLAIM_ASSERTION, CLAIM_EFFECT])
def test_certificate_requires_a_final_before_recording_a_request(tmp_path, kind):
    store, kernel, log, artifacts = build(tmp_path, {})
    target = store.add_claim(Claim(kind, "an intermediate observation"))
    before = list(log.replay())

    with pytest.raises(ValueError, match="final claim"):
        request_certificate(store, kernel, log, artifacts, target)

    assert list(log.replay()) == before
