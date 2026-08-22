"""
Certificate.

A certificate request is the backstop, not the schedule: the slicer walks the
dependency closure, the kernel settles whatever is still unverified, and the
freezer exports the slice with its evidence chain.
Under the on-commit rule that residue is normally empty (every claim was checked as it was made).
Under the lazy policy the request is what settles the slice.

The VERDICT is the final claim's own adjudication. The final's obligations
are the task contract (compiled against the authoritative task statement,
independently established), so its status answers the question the
certificate exists to answer: is the deliverable right? The rest of the
slice is the audited argument — exported in full, scored as features
(verified fraction, obligation counts) — but a support claim failing
eliminates that line of support, it does not decide the deliverable: the
solver's narration is evidence about the solver, not about the world.
Fail-closed is preserved where it matters: an unverified/failed task
obligation still leaves the verdict unverified, and a refuted one refutes.
"""
from __future__ import annotations

from typing import Dict

from ..core.events import EventLog
from ..execution.instruments import ArtifactStore
from .kernel import Kernel
from ..core.model import (
    STATUS_UNVERIFIED, STATUS_VERIFIED, STATUS_STALE, STATUS_REFUTED,
    STATUS_PARTIAL,
)
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

    entries = []
    for claim_id in slice_ids:
        row = store.get(claim_id)
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

    # the final's own status IS the verdict (statuses collected above are
    # exported per-entry; the fold over them was retired — one refuted
    # history claim must not cap an independently verified deliverable,
    # nor one unverifiable narration drag it to unverified)
    verdict = {
        STATUS_VERIFIED: "verified",
        STATUS_REFUTED: "refuted",
        STATUS_PARTIAL: "partial",
    }.get(store.status(final_id), "unverified")

    certificate = {
        "final": final_id,
        "verdict": verdict,
        "slice": entries,
    }
    digest = artifacts.put(certificate)
    certificate["certificate_id"] = digest
    log.append("certificate", {"final": final_id, "verdict": verdict, "digest": digest})
    return certificate
