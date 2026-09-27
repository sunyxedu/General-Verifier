"""The strength ladder and the record of a binding game's attempts: what a
method declares, the order methods run in, when a weaker tier may run, and
what earlier attempts tell later seats."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from ...core.model import (
    BASIS_RECORD, STRENGTH_ANALYTIC, STRENGTH_DECISIVE, STRENGTHS,
    strength_rank,
)


def method_strength(method: Dict[str, str]) -> str:
    """The method's declared strength, failing closed to analytic when the
    declaration is absent or unrecognized (mirrors basis -> record: an
    undeclared method must not execute as if it were a decision
    procedure)."""
    strength = method.get("strength", STRENGTH_ANALYTIC)
    return strength if strength in STRENGTHS else STRENGTH_ANALYTIC


def tier_order(method: Dict[str, str]) -> int:
    """Sort key for strictness-descending execution; sorted() is stable,
    so the oracle's cost-ascending order survives within a tier."""
    return -strength_rank(method_strength(method))


@dataclass
class Attempt:
    method: Dict[str, str]
    rejected: Optional[str] = None          # reason, if the method was rejected
    discharge_artifact: Optional[str] = None
    discharge_outcome: Optional[str] = None
    audit: Optional[Dict[str, str]] = None  # judger's first verdict
    redos: List[Dict] = None                # one {reason, outcome, artifact,
                                            # audit} per redo round
    briefing: str = ""                      # prior attempts' judged gaps,
                                            # as handed to this executor
    moved: bool = False                     # the delivered world changed
                                            # under the check (the solver
                                            # was working): it says nothing
                                            # about the method

    def __post_init__(self):
        if self.redos is None:
            self.redos = []


BRIEFING_MAX_ATTEMPTS = 3
BRIEFING_PROCEDURE_CHARS = 240
BRIEFING_REASON_CHARS = 600
# a method whose check ran while the solver changed the delivered world is
# run again this many times before it is set aside: the method was not at
# fault, and the solver is often done by the time its turn comes again
MOVED_RETRIES = 1


def moved_reason(moved: Dict) -> str:
    counts = ", ".join(f"{k[:-len('_count')]} {v}" for k, v in moved.items()
                       if k.endswith("_count"))
    tools = ", ".join(moved.get("solver_tools") or []) or "unknown"
    return ("the delivered world changed during the check while the solver "
            f"was running world-mutating tools ({tools}; {counts or 'changes'}"
            "): the check observed no single state and settles nothing; "
            "this says nothing about the method")


def prior_attempt_briefing(attempts: List["Attempt"]) -> str:
    """Untrusted diagnostics from this obligation's earlier attempts.

    A fresh method's executor otherwise starts blind: the Oracle saw why
    the previous method was rejected, but the seat that has to implement
    the replacement did not, and re-made the rejected artifact (a checker
    hard-coding one database page was executed twice, 300 s each, and
    rejected twice for the same gap).  The redo loop already hands the
    Judger's reason to the same method's re-execution; this extends that
    to the obligation's earlier methods.  It is diagnostic only: it cannot
    amend the bound procedure and the fresh Judger audits as before.
    """
    lines = []
    judged = [a for a in attempts if not a.moved]
    for index, attempt in enumerate(judged[-BRIEFING_MAX_ATTEMPTS:], 1):
        reason = attempt.rejected or ""
        if attempt.redos and attempt.redos[-1].get("reason"):
            reason = reason or ""
            reason = (reason + " | last redo: "
                      + str(attempt.redos[-1]["reason"])).strip(" |")
        if not reason:
            continue
        procedure = str(attempt.method.get("procedure", ""))
        procedure = procedure[:BRIEFING_PROCEDURE_CHARS]
        if attempt.discharge_outcome is None:
            state = "was rejected before execution"
        else:
            state = f"was executed (outcome {attempt.discharge_outcome}) and rejected"
        lines.append(
            f"Prior attempt {index} on this obligation used a different method "
            f"(procedure: {procedure}) and {state}: "
            f"{reason[:BRIEFING_REASON_CHARS]}")
    return "\n\n".join(lines)


def blocking_tiers(method: Dict[str, str], tier_listed: set,
              tier_unobtainable: set) -> List[str]:
    """The stronger tiers this method is parked behind: every LISTED
    stronger tier not yet audited structurally unavailable (>=1
    unobtainable attempt; failed attempts do not clear a tier — nor
    do they hold it closed once an audited gap is on record). Empty
    means executable. An unlisted stronger tier never blocks: the
    omission itself goes into the degradation record for the judger
    to audit."""
    strength = method_strength(method)
    return [tier for tier in STRENGTHS
            if strength_rank(tier) > strength_rank(strength)
            and tier in tier_listed and tier not in tier_unobtainable]


def degradation_record(strength: str, attempts: List[Attempt],
                        tier_listed: set) -> str:
    """What the judger audits before countersigning a non-decisive
    binding: every stronger-tier attempt with its terminal state, plus
    stronger tiers the oracle never listed (an omission is itself a
    signed declaration that no such method exists in principle)."""
    if strength == STRENGTH_DECISIVE:
        return "(none — decisive method, no degradation to audit)"
    stronger = [a for a in attempts
                if not a.moved
                and strength_rank(method_strength(a.method))
                > strength_rank(strength)]
    return json.dumps(
        {
            "stronger_attempts": [
                {"tier": method_strength(a.method),
                 "procedure": a.method.get("procedure", "")[:80],
                 "outcome": a.discharge_outcome,
                 "terminal_reason": a.rejected}
                for a in stronger
            ],
            "omitted_tiers": [
                tier for tier in STRENGTHS
                if strength_rank(tier) > strength_rank(strength)
                and tier not in tier_listed
            ],
        },
        ensure_ascii=False,
    )


def features(row: Dict) -> str:
    """Mechanical claim features"""
    text = row["text"]
    return json.dumps(
        {
            "kind": row.get("kind", ""),
            "length": len(text),
            "has_digits": any(c.isdigit() for c in text),
            "has_quantifier_words": any(
                w in text for w in ("all", "any", "every", "each", "exists")
            ),
        },
        ensure_ascii=False,
    )


def attempt_diagnostic(attempt: Attempt) -> Dict:
    """Bounded method diagnostics for the Oracle's single repair pass.

    Artifact contents and digests are deliberately absent: this feedback says
    why an implementation was rejected, never whether the obligation is true.
    """
    method = {
        "procedure": attempt.method.get("procedure", ""),
        "promised_artifact": attempt.method.get("artifact", ""),
        "level": attempt.method.get("level", ""),
        "basis": attempt.method.get("basis", BASIS_RECORD),
        "strength": method_strength(attempt.method),
        "bridges": attempt.method.get("bridges", []),
    }
    if attempt.method.get("script"):
        method["script"] = str(attempt.method["script"])[:4000]

    def public(audit):
        if audit is None:
            return None
        return {k: v for k, v in audit.items() if not k.startswith("_")}

    return {
        "method": method,
        "initial_outcome": attempt.discharge_outcome,
        "initial_audit": public(attempt.audit),
        "redos": [
            {
                "reason": redo.get("reason"),
                "outcome": redo.get("outcome"),
                "audit": public(redo.get("audit")),
            }
            for redo in attempt.redos
        ],
        "terminal_reason": attempt.rejected,
    }
