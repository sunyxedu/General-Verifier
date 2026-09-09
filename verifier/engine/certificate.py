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
solver's narration is evidence about the solver, not about the world. Its
whole-slice counts remain diagnostic features, while BoN tie-breaking uses
only the final claim's own task-contract obligations.
Fail-closed is preserved where it matters: an unverified/failed task
obligation still leaves the verdict unverified, and a refuted one refutes.

Slice entries carry two verdicts per claim: the LOCAL status (conditional —
does the statement hold assuming its premises) and the EFFECTIVE read-time
fold over the premise chain (what a consumer may rely on), with the open
premises and any poisoning refuted premise disclosed for audit.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from ..core.events import EventLog
from ..execution.instruments import ArtifactStore
from ..llm.model_spans import submit_with_context
from .kernel import Kernel
from ..core.model import (
    CLAIM_FINAL,
    STATUS_UNVERIFIED, STATUS_VERIFIED, STATUS_STALE, STATUS_REFUTED,
    STATUS_PARTIAL, STRENGTH_DECISIVE, strength_rank,
)
from ..core.store import ClaimStore

# What a certificate request settles.  ``full`` verifies every open claim in
# the final's dependency closure; ``final`` settles only the final claim,
# whose task contract is compiled independently of the solver's narration
# and whose status is the verdict.  The slice is exported whole either way,
# so support claims left open are visible as such.
CERTIFICATE_SCOPE_FULL = "full"
CERTIFICATE_SCOPE_FINAL = "final"
CERTIFICATE_SCOPES = (CERTIFICATE_SCOPE_FULL, CERTIFICATE_SCOPE_FINAL)


def request_certificate(
    store: ClaimStore, kernel: Kernel, log: EventLog,
    artifacts: ArtifactStore, final_id: str,
    scope: str = CERTIFICATE_SCOPE_FULL,
) -> Dict:
    if store.get(final_id)["kind"] != CLAIM_FINAL:
        raise ValueError("a certificate requires a final claim")
    if scope not in CERTIFICATE_SCOPES:
        raise ValueError(f"unknown certificate scope: {scope!r}")
    log.append("verify_request", {"target": final_id, "cause": "certificate",
                                  "scope": scope})

    slice_ids = store.backward_slice(final_id)
    pending = [cid for cid in reversed(slice_ids)
               if store.status(cid) in (STATUS_UNVERIFIED, STATUS_STALE)
               and (scope == CERTIFICATE_SCOPE_FULL or cid == final_id)]
    parallel = getattr(kernel, "parallel_claims", 1) or 1
    if parallel > 1 and len(pending) > 1:
        # The final contract is compiled independently of solver-derived
        # premises and its local verdict never waits on them.  Submit it first
        # so a bounded worker pool does not put the task verdict behind a long
        # support-claim queue; every slice claim still runs and the certificate
        # still waits for all of them below.
        immutable_compact_run = (
            not getattr(kernel, "live_world", True)
            and getattr(kernel, "memory", None) is None
            and getattr(kernel.backend, "compact_inspection", False) is True
        )
        if immutable_compact_run and final_id in pending:
            pending.remove(final_id)
            pending.insert(0, final_id)
        # every claim's LOCAL verdict is conditional on its premises and
        # never blocks on them (the read-time fold charges premise state
        # to consumers), so the whole slice verifies concurrently; the
        # kernel serializes per claim and per obligation node, so shared
        # work is still done once
        with ThreadPoolExecutor(max_workers=parallel,
                                thread_name_prefix="gv-claim") as pool:
            futures = [submit_with_context(pool, kernel.verify, claim_id)
                       for claim_id in pending]
            for future in futures:
                future.result()
    else:
        for claim_id in pending:
            kernel.verify(claim_id)

    entries = []
    effective_memo = {}
    for claim_id in slice_ids:
        row = store.get(claim_id)
        effective = store.effective(claim_id, effective_memo)
        entries.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "text": row["text"],
                # status is LOCAL — the conditional verdict (holds assuming
                # every premise). effective_* is the read-time fold over the
                # premise chain: what a consumer may actually rely on.
                # open_premises lists the transitive premises still
                # unsettled; premise_refuted names the poisoning node.
                "status": row["status"],
                "effective_status": effective["effective_status"],
                "effective_basis": effective["effective_basis"],
                "effective_strength": effective["effective_strength"],
                "open_premises": effective["open_premises"],
                "premise_refuted": effective["premise_refuted"],
                "reason": row["reason"],
                # evidence basis of a verified node (world | record): what
                # its verdict entitles reliance to; disclosed for audit
                "basis": row["basis"],
                # evidence strength (decisive | bounded | analytic; None =
                # settled outside the ladder, reads decisive): how the
                # verdict was won; disclosed for audit
                "strength": row["strength"],
                "evidence": store.evidence_for(claim_id),
                # obligations are first-class graph nodes: the certificate
                # carries each one with its own status and evidence. The
                # structured residue is native, not a summary bolted on
                "obligations": [
                    {"id": o["id"], "text": o["text"],
                     "status": o["status"], "reason": o["reason"],
                     "basis": o["basis"], "strength": o["strength"],
                     # edge attributes: the clause's modality (only a
                     # must refutes), the clause itself, and whether an
                     # absence witness may refute it
                     "severity": o.get("severity", "must"),
                     "clause": o.get("clause", ""),
                     "refutable_by_absence":
                         bool(o.get("refutable_by_absence", False)),
                     "evidence": store.evidence_for(o["id"])}
                    for o in store.obligation_children(claim_id)
                ],
            }
        )

    # the final's own status IS the verdict (statuses collected above are
    # exported per-entry; the fold over them was retired — one refuted
    # history claim must not cap an independently verified deliverable,
    # nor one unverifiable narration drag it to unverified)
    final_row = store.get(final_id)
    verdict = {
        STATUS_VERIFIED: "verified",
        STATUS_REFUTED: "refuted",
        STATUS_PARTIAL: "partial",
    }.get(final_row["status"], "unverified")

    # the strength ladder's read-out: a final established entirely at
    # decisive strength is verified; established but resting on bounded/
    # analytic evidence somewhere is SUPPORTED — everything holds, some
    # of it on declared-and-audited weaker instruments, with the bridges
    # disclosed in the evidence. Distinct from partial, where a necessary
    # check never got established at all. None reads as decisive
    # (pre-ladder rows, direct discharge).
    verdict_strength = None
    if verdict == "verified":
        verdict_strength = final_row["strength"] or STRENGTH_DECISIVE
        if strength_rank(verdict_strength) \
                < strength_rank(STRENGTH_DECISIVE):
            verdict = "supported"

    certificate = {
        "final": final_id,
        "verdict": verdict,
        "verdict_strength": verdict_strength,
        "slice": entries,
    }
    digest = artifacts.put(certificate)
    certificate["certificate_id"] = digest
    log.append("certificate", {"final": final_id, "verdict": verdict,
                               "verdict_strength": verdict_strength,
                               "digest": digest})
    return certificate
