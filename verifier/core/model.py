"""
Data contracts. 
Everything content-addressed; 
statuses are a small state machine enforced by the store, not here.
"""

from __future__ import annotations
from typing import Literal

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

# canonical hashing
def canonical(obj: Any) -> str:
    """Deterministic JSON encoding used for every content hash."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


# claim kinds
CLAIM_ASSERTION = "assertion"
CLAIM_EFFECT = "effect"
CLAIM_FINAL = "final"
KINDS = (CLAIM_ASSERTION, CLAIM_EFFECT, CLAIM_FINAL)
CLAIM_KINDS = Literal["assertion", "effect", "final"]

# status machine: unverified -> verified | refuted | partial; anything -> stale
STATUS_UNVERIFIED = "unverified"
STATUS_VERIFIED = "verified"
STATUS_REFUTED = "refuted"
STATUS_STALE = "stale"
STATUS_PARTIAL = "partial"

# verify outcome
OUTCOME_HOLDS = "holds"
OUTCOME_REFUTED = "refuted"
OUTCOME_FAILED = "failed"
OUTCOME_UNOBTAINABLE = "unobtainable"
OUTCOME_KINDS = Literal["holds", "refuted", "failed", "unobtainable"]

# evidence basis of a verified node: what its "verified" entitles others to.
# world  — the statement's content was established independently of the
#          solver's own reporting (recomputation from task inputs, checks of
#          the delivered world); downstream compilation may rely on it as fact.
# record — the ledger attests the solver did/reported this; the event is real
#          but the content's truth was never independently established, so it
#          proves reporting, not truth. Effects are record by construction.
# The claim-level basis is a taint fold (any record component -> record); the
# reusable atoms are the obligation NODES, each carrying its own basis.
BASIS_WORLD = "world"
BASIS_RECORD = "record"

# internal graph node kind for obligations; deliberately NOT in KINDS so a
# solver can never commit one — only the kernel materializes them
OBLIGATION_KINDS = "obligation"

@dataclass(frozen=True)
class Claim:
    kind: CLAIM_KINDS
    text: str
    deps: Tuple[str, ...] = ()
    revises: Optional[str] = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown claim kind: {self.kind}")
        object.__setattr__(self, "deps", tuple(self.deps))

    @property
    def id(self) -> str:
        # identity = content + position in the graph (sorted dep set).
        # revises is lineage metadata: resubmitting identical content dedups.
        return content_hash(
            {
                "kind": self.kind,
                "text": self.text,
                "deps": sorted(self.deps),
            }
        )


@dataclass(frozen=True)
class Obligation:
    claim_id: str  # context only — excluded from identity
    text: str
    # the minimum evidence basis that can settle this obligation: world
    # for conditions about what is TRUE (the solver's own reporting can
    # never settle those), record for conditions about what the solver
    # did or reported (the ledger settles those). Compiled by the
    # obligation compiler; enforced mechanically at binding and settle.
    required_basis: str = BASIS_RECORD

    @property
    def id(self) -> str:
        """
        identity is the content of the check AND its floor: the same text
        demanded at world level is a different check than at record level
        — sharing one node would let a record-satisfied verdict slip into
        a world slot through dedup. Same text + same floor still transfers
        settled verdicts across claims for free.
        """
        return content_hash({"text": self.text,
                             "required": self.required_basis})


def basis_satisfies(achieved: Optional[str], required: str) -> bool:
    """world evidence satisfies any floor; record satisfies only record."""
    return required != BASIS_WORLD or achieved == BASIS_WORLD


@dataclass(frozen=True)
class Evidence:
    obligation_id: str
    instrument: str
    outcome: OUTCOME_KINDS
    artifacts: dict = field(default_factory=dict)
    witness: Optional[Any] = None

    @property
    def id(self) -> str:
        return content_hash(
            {
                "obligation": self.obligation_id,
                "instrument": self.instrument,
                "outcome": self.outcome,
                "artifacts": self.artifacts,
                "witness": self.witness,
            }
        )