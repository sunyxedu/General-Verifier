"""The verdict fold: a claim's local verdict from its obligations' outcomes.

A pure function of what the checks found — no store, no ledger. The kernel
applies the result (the store takes it only while the claim is still the
one the check began on).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..core.model import (
    BASIS_RECORD, BASIS_WORLD,
    OUTCOME_FAILED, OUTCOME_HOLDS, OUTCOME_REFUTED, OUTCOME_UNOBTAINABLE,
    SEVERITY_MUST,
    STATUS_PARTIAL, STATUS_REFUTED, STATUS_UNVERIFIED, STATUS_VERIFIED,
    weakest_strength,
)

ABSTENTION = "no world-basis evidence for a completion claim"


@dataclass(frozen=True)
class Fold:
    """A folded verdict: the status and what it carries, the store reason,
    and evidence to record on the claim before the status is set."""
    status: str
    basis: Optional[str] = None
    strength: Optional[str] = None
    reason: Optional[str] = None
    evidence: Optional[Dict] = None


def fold(outcomes: Sequence[str], bases: Sequence[str] = (),
         strengths: Sequence[Optional[str]] = (),
         severities: Sequence[str] = (), *,
         completion_claim: bool = False) -> Fold:
    """Fold obligation outcomes (in compile order) into a local verdict."""
    outcomes = list(outcomes)
    severities = list(severities) or [SEVERITY_MUST] * len(outcomes)
    must_refuted = [i for i, o in enumerate(outcomes)
                    if o == OUTCOME_REFUTED and severities[i] == SEVERITY_MUST]
    # NECESSITY AXIOM at fold time: a refuted SHOULD obligation is a
    # contrary signal, not a counterexample — the task did not make
    # acceptance depend on that clause. It lowers the claim (to partial,
    # with the violation on record) but never refutes it.
    should_violated = [i for i, o in enumerate(outcomes)
                       if o == OUTCOME_REFUTED
                       and severities[i] != SEVERITY_MUST]
    settled: List[str] = [OUTCOME_UNOBTAINABLE if i in should_violated else o
                          for i, o in enumerate(outcomes)]
    if must_refuted:
        # Parallel checks may finish after the refutation. Provenance
        # belongs to the deciding check, never an unrelated later one.
        index = must_refuted[0]
        return Fold(STATUS_REFUTED,
                    basis=bases[index] if bases else None,
                    strength=strengths[index] if strengths else None,
                    reason="counterexample on record")
    if should_violated and all(
            o in (OUTCOME_HOLDS, OUTCOME_UNOBTAINABLE) for o in settled):
        return Fold(STATUS_PARTIAL, evidence={
            "outcome": OUTCOME_UNOBTAINABLE,
            "contrary_signals": len(should_violated),
            "error": "advisory (should) clause violated; every "
                     "acceptance (must) check that ran held"})
    all_hold = bool(outcomes) and all(o == OUTCOME_HOLDS for o in outcomes)
    if all_hold and completion_claim and BASIS_WORLD not in bases:
        # conservative ABSTENTION POLICY, not a logical theorem: an
        # all-record final is the laundering signature (every check
        # passed, none ever left the solver's own account), so it
        # abstains. A purely process task could legitimately rest on
        # record evidence alone — revisit if real false-unverifieds of
        # that shape appear.
        return Fold(STATUS_UNVERIFIED, reason=ABSTENTION,
                    evidence={"outcome": OUTCOME_FAILED, "error": ABSTENTION})
    if all_hold:
        # LOCAL provenance fold: world only when every obligation was
        # world-based. This is the conditional verdict's own basis —
        # premise taint (residual obligations may have relied on any
        # premise) is charged at read time by store.effective, which folds
        # every non-effect premise's effective basis into the chain. Any
        # record component -> record: a chain of reliance must bottom out
        # in world checks or say so. Strength is a weakest-link fold;
        # below decisive the certificate reads the verified status out as
        # "supported".
        basis = (BASIS_WORLD if bases and all(b == BASIS_WORLD for b in bases)
                 else BASIS_RECORD)
        return Fold(STATUS_VERIFIED, basis=basis,
                    strength=weakest_strength(strengths))
    if any(o == OUTCOME_UNOBTAINABLE for o in settled) and all(
            o in (OUTCOME_HOLDS, OUTCOME_UNOBTAINABLE) for o in settled):
        # Partial has one narrow, terminal meaning: every check that did
        # run held, but at least one necessary check is structurally
        # unavailable inside this fixed evidence boundary. A retryable
        # failed check is not partial progress and takes the branch below.
        return Fold(STATUS_PARTIAL)
    return Fold(STATUS_UNVERIFIED)
