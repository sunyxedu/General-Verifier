"""
Certificate.

A certificate request is the backstop, not the schedule: the slicer walks the
dependency closure, the kernel settles whatever is still unverified, and the
freezer exports the slice with its evidence chain. 
Under the on-commit rule that residue is normally empty (every claim was checked as it was made). 
Under the lazy policy the request is what settles the slice.
Verdicts never collapse dishonestly: any refuted -> refuted; any unresolved -> unverified; else verified.
"""
from __future__ import annotations

from typing import Dict

from ..core.events import EventLog
from ..execution.instruments import ArtifactStore
from .kernel import Kernel
from ..core.model import STATUS_UNVERIFIED, STATUS_STALE, STATUS_REFUTED, STATUS_PARTIAL
from ..core.store import ClaimStore


def request_certificate(
    store: ClaimStore, kernel: Kernel, log: EventLog,
    artifacts: ArtifactStore, final_id: str,
) -> Dict:
    log.append("verify_request", {"target": final_id, "cause": "certificate"})

    slice_ids = store.backward_slice(final_id)
    for claim_id in reversed(slice_ids):
        if store.status(claim_id) in (STATUS_UNVERIFIED, STATUS_STALE):
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
                # evidence basis of a verified node (world | record): what
                # its verdict entitles reliance to; disclosed for audit
                "basis": row["basis"],
                "evidence": store.evidence_for(claim_id),
                # obligations are first-class graph nodes: the certificate
                # carries each one with its own status and evidence. The
                # structured residue is native, not a summary bolted on
                "obligations": [
                    {"id": o["id"], "text": o["text"],
                     "status": o["status"], "reason": o["reason"],
                     "basis": o["basis"],
                     "evidence": store.evidence_for(o["id"])}
                    for o in store.obligation_children(claim_id)
                ],
            }
        )

    if any(s == STATUS_REFUTED for s in statuses):
        verdict = "refuted"
    elif any(s in (STATUS_UNVERIFIED, STATUS_STALE) for s in statuses):
        verdict = "unverified"
    elif any(s == STATUS_PARTIAL for s in statuses):
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
