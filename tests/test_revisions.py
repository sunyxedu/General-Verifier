"""An explicit revision invalidates the evidence used by its consumers."""
import json

import pytest

from verifier.core.model import Claim, CLAIM_ASSERTION, CLAIM_FINAL
from verifier.core.store import TRIGGER_LAZY
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
    assert json.loads((tmp_path / "state/artifacts" / (latest["artifact"] + ".json")).read_text())["observed"] == revised
