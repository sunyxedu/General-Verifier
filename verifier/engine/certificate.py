"""Certificate (components 13, 23, 33).

A certificate request is the final claim gaining a dependent — the same single
rule. The slicer walks the dependency closure, the kernel settles residual
unverified claims, and the freezer exports the slice with its evidence chain.
Verdicts never collapse dishonestly: any refuted -> refuted; any unresolved ->
unverified; else verified.
"""
from __future__ import annotations

from typing import Dict

from ..core.events import EventLog
from ..execution.instruments import ArtifactStore
from .kernel import Kernel
from ..core.model import UNVERIFIED, STALE, REFUTED, PARTIAL
from ..core.store import ClaimStore


def request_certificate(
    store: ClaimStore, kernel: Kernel, log: EventLog,
    artifacts: ArtifactStore, final_id: str,
) -> Dict:
    log.append("verify_request", {"target": final_id, "cause": "certificate"})

    slice_ids = store.backward_slice(final_id)
    for claim_id in reversed(slice_ids):
        if store.status(claim_id) in (UNVERIFIED, STALE):
            kernel.verify(claim_id)

    entries, statuses = [], []
    for claim_id in slice_ids:
        row = store.get(claim_id)
        statuses.append(row["status"])
        entries.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "text": row["text"],
                "status": row["status"],
                "reason": row["reason"],
                "evidence": store.evidence_for(claim_id),
            }
        )

    if any(s == REFUTED for s in statuses):
        verdict = "refuted"
    elif any(s in (UNVERIFIED, STALE) for s in statuses):
        verdict = "unverified"
    elif any(s == PARTIAL for s in statuses):
        verdict = "partial"
    else:
        verdict = "verified"

    certificate = {
        "final": final_id,
        "verdict": verdict,
        "slice": entries,
    }
    digest = artifacts.put(certificate)
    certificate["certificate_id"] = digest
    log.append("certificate", {"final": final_id, "verdict": verdict, "digest": digest})
    return certificate
