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

import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Dict

from ..core.events import EventLog
from ..execution.instruments import ArtifactStore
from ..llm.model_spans import submit_with_context
from ..llm.schema import RoleError
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
    try:
        return _certify(store, kernel, log, artifacts, final_id, scope)
    except Exception as err:
        # the request is on the ledger; without its end a reader (the stop
        # hook) cannot tell a check that errored from one still running.
        # The error still reaches the caller.
        log.append("certificate_failed", {
            "final": final_id, "error": f"{type(err).__name__}: {err}"[:500],
            "infrastructure": infrastructure_failure(err)})
        raise


def _certify(store: ClaimStore, kernel: Kernel, log: EventLog,
             artifacts: ArtifactStore, final_id: str, scope: str) -> Dict:
    slice_ids = store.backward_slice(final_id)
    pending = [
        cid for cid in reversed(slice_ids)
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
            and kernel.run_options.compact_inspection
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
                # admission flag, orthogonal to status: the screener found
                # the claim settled by the retained record and the
                # verifier assumed it without a check (never on a final)
                "confident": bool(row.get("confident")),
                "confidence_reason": row.get("confidence_reason"),
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
                     "task_bearing": o.get("task_bearing", "unclear"),
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
    # a final the screener settled carries the screener's verdict like any
    # other status; the certificate discloses where it came from
    settled_by = "screener" if final_row.get("confident") else "checks"

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

    integrity = verifier_integrity(log)
    void = integrity.pop("void", None)
    if void:
        # the verifier that produced this slice is not the one the session
        # started with: nothing it settled is a verdict of that verifier
        verdict, verdict_strength = "unverified", None
    certificate = {
        "final": final_id,
        "verdict": verdict,
        "verdict_strength": verdict_strength,
        "settled_by": settled_by,
        "integrity": integrity,
        **({"void": void} if void else {}),
        "slice": entries,
    }
    digest = artifacts.put(certificate)
    certificate["certificate_id"] = digest
    log.append("certificate", {"final": final_id, "verdict": verdict,
                               "verdict_strength": verdict_strength,
                               **({"void": void} if void else {}),
                               "digest": digest})
    return certificate


# Failures of the verifier's own means of checking — its model seats, their
# provider and transport, its host — as opposed to an internal error of the
# verifier's logic. Neither is a defect of the delivery; the flag tells the
# reader whether repeating the same request is the remedy.
_INFRASTRUCTURE_NAMES = frozenset({
    "TransientTransportError", "AwaitingCredit", "OutputBudgetExhausted",
    "HTTPError", "HTTPStatusError", "TransportError", "TimeoutException",
    "APIError", "APIStatusError", "APIConnectionError", "APITimeoutError",
    "RateLimitError", "InternalServerError",
})
_HTTP_STATUS = re.compile(r"\bHTTP [45]\d\d\b")


def infrastructure_failure(err: BaseException) -> bool:
    """Whether ``err`` (or an exception it was raised from) is a seat,
    provider, transport, or host failure rather than a verifier bug."""
    seen = 0
    while err is not None and seen < 8:
        if isinstance(err, (RoleError, OSError, TimeoutError, FutureTimeout)):
            return True
        if any(k.__name__ in _INFRASTRUCTURE_NAMES
               for k in type(err).__mro__):
            return True
        if _HTTP_STATUS.search(str(err)):
            return True
        err = err.__cause__ or err.__context__
        seen += 1
    return False


def verifier_integrity(log: EventLog) -> Dict:
    """How the verifier ran during the session, from its own launch records:
    how many times its server started, and whether any start changed its
    settings or code or ran beside a live server (then ``void`` says why)."""
    starts, changes, concurrent = 0, [], 0
    for event in log.replay():
        kind = event.get("type")
        if kind == "server_started":
            starts += 1
        elif kind == "server_config_changed":
            changes += list((event.get("payload") or {}).get("changed") or [])
        elif kind == "server_concurrent_start":
            concurrent += 1
    out = {"server_starts": starts}
    reasons = []
    if changes:
        out["changed"] = sorted(set(changes))
        reasons.append("the verifier was restarted with other settings or "
                       "code (" + ", ".join(out["changed"][:6]) + ")")
    if concurrent:
        out["concurrent_starts"] = concurrent
        reasons.append("a second verifier server ran on the same state")
    if reasons:
        out["void"] = "; ".join(reasons)
    return out
