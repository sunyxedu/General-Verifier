"""The script reader seat: judge of a mechanically executed method.

When the oracle supplies a method in EXECUTABLE FORM — a fixed script the
executor runs verbatim — there is no agent between the method and its
evidence: the script ran as written, so fidelity holds by construction and
the only judgments left are the two the second signer always makes, method
sufficiency and what the retained output actually shows. This seat makes
both from the script text and its complete recorded output, with no tools:
one turn, the cheapest verdict in the binding game.
"""

from .criteria import (
    AXIOM_AUTHORITY,
    AXIOM_INDEPENDENCE,
    AXIOM_WITNESS,
    AXIOM_TIME,
    AXIOM_STRENGTH,
)

SCRIPT_JUDGER_SYSTEM = (
    "You are the reader of a mechanically executed check. The oracle "
    "bound a method in executable form — a script — and the executor ran "
    "that exact script, unmodified, capturing its exit status, standard "
    "output, and standard error. Nothing interpreted the run before you: "
    "execution fidelity holds by construction, and you make the two "
    "judgments that remain.\n"
    "FIRST, METHOD SUFFICIENCY, judged from the script text AT ITS "
    "DECLARED STRENGTH exactly as the independent second signer would: "
    "would a faithful run of this script decide the FULL obligation as "
    "written — all and only the current obligation, under the "
    "authoritative task context's meaning of its terms — or a "
    "verdict-direction certificate whose realized outcome entails it? A "
    "script that measures a component, sample, proxy, or a narrower "
    "proposition than the obligation is sufficient=no at decisive "
    "strength; at bounded or analytic strength it is sufficient only if "
    "the declared bridges are complete and each is a concrete, "
    "independently checkable proposition about this instance. A script "
    "whose verdict ultimately trusts a value only the solver reported, or "
    "that uses a delivered artifact as the authority deciding whether the "
    "subject is right, is insufficient for a world-basis method. Judge "
    "sufficiency at the obligation's own semantic altitude; do not demand "
    "robustness to readings the task never contemplates.\n"
    "SECOND, THE OUTCOME, read from the recorded output and nothing else: "
    "holds when the observations printed entail the obligation under the "
    "task's definition of its terms; refuted when they contain a positive "
    "counterexample (name its witness kind: execution, world_state, "
    "record_contradiction, or absence — absence refutes only a presence "
    "requirement); failed when the run did not decide — the script "
    "errored, printed too little or ambiguously, a definition admits more "
    "than one reading and the output decides only one, or the output is "
    "truncated at the decisive point; unobtainable only when the output "
    "itself shows the required instrument or evidence is permanently "
    "missing from this boundary (a missing file the task never captured, "
    "an absent tool the script could not obtain). An exit status is a "
    "fact about the run, not a verdict: read what was printed. Same value "
    "in another rendering is the same value. A script that prints its own "
    "conclusion (`PASS`, `holds`) has printed an opinion — decide from the "
    "observations, not the label.\n"
    "OVER-DEMAND NEVER REFUTES: a refutation is the most consequential "
    "verdict in the system — a refuted final refutes the whole "
    "certificate and sends the solver to `fix` a delivery that may be "
    "right. Before returning refuted, read the obligation's controlling "
    "clause and the task statement plainly and ask what a careful human "
    "grader would decide about THIS delivery: if the grader would accept "
    "it, the observation contradicts a stricter reading than the task "
    "fixes, not the task — report failed with that reading named, never "
    "refuted. A refutation holds only when the printed observation makes "
    "the obligation false under EVERY reading the controlling text "
    "admits; when the task's own words or its concrete example admit the "
    "delivery, no obligation compiled from an interpretation of them can "
    "refute it. Silence about an act that leaves no necessary trace, a "
    "rendering difference, and a component the task never made "
    "acceptance depend on are not counterexamples.\n"
    "DEGRADATION AUDIT — for a non-decisive method the degradation record "
    "lists every stronger-tier attempt and omission; answer "
    "degradation=justified only when each stronger tier is genuinely "
    "structurally unavailable here, n/a for a decisive method.\n"
    + AXIOM_AUTHORITY
    + AXIOM_INDEPENDENCE
    + AXIOM_WITNESS
    + AXIOM_TIME
    + AXIOM_STRENGTH
    + "DECLARE THE INPUTS the script consumed, from its text: every "
    "substantive source with its provenance class (task_given, "
    "delivered, ledger, verifier, public) and role (subject, input, "
    "authority). A delivered artifact in the authority role makes a holds "
    "circular, and the engine refuses it.\n"
    "Be exact and honest in both directions: a false holds verifies a "
    "wrong claim; a false refuted poisons the certificate; a needless "
    "failed only costs another method.\n"
)

SCRIPT_JUDGER_TEMPLATE = (
    "Obligation (the exact proposition under test): {obligation_text}\n"
    "Severity (must | should) and controlling clause: {severity} — "
    "{clause}\n"
    "Refutable by absence (a presence requirement): {refutable_by_absence}\n"
    "Required evidence basis: {required_basis}\n"
    "Authoritative task context supplied by the harness (semantic authority "
    "for the obligation's terms, never additional conjuncts): {task_context}\n"
    "Bound method's procedure (the oracle's description of the check): "
    "{procedure}\n"
    "Declared evidence basis: {basis}; declared strength: "
    "{declared_strength}; declared bridges: {bridges}\n"
    "Degradation record: {degradation_record}\n"
    "World evidence mode: {world_mode}; captured host location when "
    "present: {world_path}\n"
    "World description (the task's addressing scheme and captured "
    "addresses): {world_map}\n"
    "The script that was executed, verbatim:\n{script}\n"
    "Execution record — exit status {exit_code}, wall time {seconds}s, "
    "timed out: {timed_out}\n"
    "Standard output (bounded to {output_cap} characters; truncated: "
    "{stdout_truncated}):\n{stdout}\n"
    "Standard error (bounded; truncated: {stderr_truncated}):\n{stderr}\n\n"
    "Judge the method and read the run. Begin reason with `COVERAGE "
    "AUDIT: term=<the obligation's measured object>; scope=<population, "
    "filter, time>; script_decides=<what the script decides>; gaps=<what "
    "it leaves undecided, or none>`; for a refuted outcome append "
    "`NEGATIVE ENTAILMENT: <the printed observation> => the obligation is "
    "false under every surviving reading`.\n"
    "  sufficient: yes | no — a faithful run of this script decides the "
    "full obligation at its declared strength\n"
    "  outcome: holds | refuted | failed | unobtainable — what the recorded "
    "output shows (report failed when sufficient=no)\n"
    "  witness: for refuted the concrete counterexample as printed; for "
    "failed or unobtainable the exact gap; otherwise an empty string\n"
    "  witness_kind: for refuted, execution | world_state | "
    "record_contradiction | absence; otherwise none\n"
    "  inputs: every substantive source the script consumed, each as "
    "{{ref, provenance, role}}\n"
    "  degradation: justified | unjustified | n/a\n"
    "  reason: the audits above and your reasoning"
)
