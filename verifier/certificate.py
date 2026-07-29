"""Certificate (components 13, 23, 33).

A certificate request is the final claim gaining a dependent — the same single
rule. The slicer walks the dependency closure, the kernel settles residual
unverified claims, and the freezer exports the slice with its evidence chain.
Verdicts never collapse dishonestly: any refuted -> refuted; any unresolved ->
unverified; else verified at the weakest grade on the slice.
"""
from __future__ import annotations

from typing import Dict

from .events import EventLog
from .instruments import ArtifactStore
from .kernel import Kernel
from .model import UNVERIFIED, STALE, VERIFIED, REFUTED, weakest_grade
from .store import ClaimStore


def request_certificate(
    store: ClaimStore, kernel: Kernel, log: EventLog,
    artifacts: ArtifactStore, final_id: str,
) -> Dict:
    log.append("verify_request", {"target": final_id, "cause": "certificate"})

    slice_ids = store.backward_slice(final_id)
    for claim_id in reversed(slice_ids):  # parents first: settle leaves upward
        if store.status(claim_id) in (UNVERIFIED, STALE):
            kernel.verify(claim_id)

    entries, statuses, grades = [], [], []
    for claim_id in slice_ids:
        row = store.get(claim_id)
        statuses.append(row["status"])
        grades.append(row["grade"])
        entries.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "text": row["text"],
                "status": row["status"],
                "grade": row["grade"],
                "reason": row["reason"],
                "evidence": store.evidence_for(claim_id),
            }
        )

    if any(s == REFUTED for s in statuses):
        verdict = "refuted"
    elif any(s in (UNVERIFIED, STALE) for s in statuses):
        verdict = "unverified"
    else:
        verdict = "verified"

    certificate = {
        "final": final_id,
        "verdict": verdict,
        "overall_grade": weakest_grade(grades) if verdict == "verified" else None,
        "slice": entries,
    }
    digest = artifacts.put(certificate)
    certificate["certificate_id"] = digest
    log.append("certificate", {"final": final_id, "verdict": verdict, "digest": digest})
    return certificate
