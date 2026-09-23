"""The pre-execution sufficiency audit seat.

The independent Judger answers two questions after execution: was the
method sufficient, and was it executed faithfully.  The first question does
not depend on the execution at all, yet it was only asked after the most
expensive stage had run; an insufficient method was executed for minutes and
then rejected.  This seat asks exactly that first question before the
executor starts, in one tool-less turn, against the same criteria.  The
Judger still audits sufficiency afresh afterwards; this seat can only stop
an execution, never settle anything.
"""

from .criteria import AXIOM_AUTHORITY, AXIOM_STRENGTH

SUFFICIENCY_SYSTEM = (
    "You are the pre-execution auditor for METHOD SUFFICIENCY. The oracle "
    "proposes an untrusted candidate method for one obligation; its "
    "sufficiency declaration is not a fact. Decide one question: if the "
    "procedure were executed faithfully and the promised artifact delivered, "
    "would it be a decision procedure for the FULL obligation as written, or "
    "a verdict-direction certificate whose realized outcome entails the full "
    "obligation, under the authoritative task context, AT the declared "
    "strength? Declared decisive: any bridging gap, informal recognition "
    "step, unstated authority for a semantic mapping, or dependence on a "
    "capability the fixed evidence boundary excludes makes it sufficient=no; "
    "name the gap. Declared bounded or analytic: sufficient only if the "
    "narrower proposition is soundly decided and every declared bridge is a "
    "concrete, independently checkable proposition about this instance.\n"
    "FULL means all and only the current obligation as written; task context "
    "interprets its terms and never adds a conjunct. Judge at the "
    "obligation's own semantic altitude: a procedure that soundly decides "
    "the obligation's terms at the level their stated form fixes IS a "
    "decision procedure; do not demand robustness to readings the wording "
    "does not contemplate. You see no execution and no evidence: never "
    "guess the outcome, never decide the obligation, and never reject a "
    "method because it looks laborious. Reject only a structural gap you "
    "can name.\n"
    + AXIOM_AUTHORITY
    + AXIOM_STRENGTH
)

SUFFICIENCY_TEMPLATE = (
    "Obligation: {obligation_text}\n"
    "Required evidence basis (world | record): {required_basis}\n"
    "Severity and controlling clause: {severity} — {clause}\n"
    "The clause is where this obligation came from: it fixes the "
    "modality and the meaning of the terms. A clause often yields "
    "several obligations; what it states beyond this obligation's own "
    "text is decided by those siblings, never here.\n"
    "Refutable by absence: {refutable_by_absence}\n"
    "Authoritative task context (semantic authority for the obligation's "
    "terms, never extra conjuncts; path plus full text, or none): "
    "{task_context}\n"
    "Evidence boundary — world mode {world_mode} (live | captured | "
    "unavailable); world description: {world_map}\n"
    "Proposed method (untrusted): procedure={procedure}\n"
    "Promised artifact: {promised_artifact}\n"
    "Declared basis: {basis}; declared strength: {declared_strength}; "
    "declared bridges: {bridges}\n\n"
    "Begin reason with `COVERAGE AUDIT: term=<the obligation's exact "
    "measured object>; scope=<population, filter, time>; "
    "method_covers=<what the method decides>; gaps=<undecided components "
    "or readings, or none>; contract_complete=yes|no.` Then answer:\n"
    "  sufficient: yes | no — faithful execution would decide the full "
    "obligation at the declared strength\n"
    "  reason: for no, the structural gap the oracle must close; for yes, "
    "the one-sentence basis"
)
