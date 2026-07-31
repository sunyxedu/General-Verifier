"""Data contracts (report §14). Everything content-addressed; statuses are a
small state machine enforced by the store, not here."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

# ---- canonical hashing -------------------------------------------------------

def canonical(obj: Any) -> str:
    """Deterministic JSON encoding used for every content hash."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


# ---- claim -------------------------------------------------------------------

KIND_ASSERTION = "assertion"
KIND_EFFECT = "effect"
KIND_FINAL = "final"
KINDS = (KIND_ASSERTION, KIND_EFFECT, KIND_FINAL)

# status machine: unverified -> verified(grade) | refuted ; anything -> stale
UNVERIFIED = "unverified"
VERIFIED = "verified"
REFUTED = "refuted"
STALE = "stale"
PARTIAL = "partial"

# evidence grades, strongest first (§14: overall = weakest link)
GRADES = ("proven", "demonstrated", "corroborated")


def grade_rank(grade: str) -> int:
    return GRADES.index(grade)


def weakest_grade(grades) -> Optional[str]:
    grades = [g for g in grades if g is not None]
    if not grades:
        return None
    return max(grades, key=grade_rank)


@dataclass(frozen=True)
class Claim:
    kind: str
    text: str
    deps: Tuple[str, ...] = ()
    branch: Optional[str] = None
    revises: Optional[str] = None  # lineage only: excluded from identity

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown claim kind: {self.kind}")
        object.__setattr__(self, "deps", tuple(self.deps))

    @property
    def id(self) -> str:
        # identity = content + position in the graph (sorted dep set) + branch.
        # revises is lineage metadata: resubmitting identical content dedups.
        return content_hash(
            {
                "kind": self.kind,
                "text": self.text,
                "deps": sorted(self.deps),
                "branch": self.branch,
            }
        )


@dataclass(frozen=True)
class Obligation:
    claim_id: str
    text: str
    failure_looks_like: str  # empty string == vacuous == must be filtered out

    @property
    def id(self) -> str:
        return content_hash(
            {"claim": self.claim_id, "text": self.text, "fail": self.failure_looks_like}
        )

    @property
    def falsifiable(self) -> bool:
        return bool(self.failure_looks_like.strip())


@dataclass(frozen=True)
class Evidence:
    obligation_id: str
    instrument: str
    outcome: str            # "holds" | "refuted" | "failed"  (failed == fail-closed)
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


OUTCOME_HOLDS = "holds"
OUTCOME_REFUTED = "refuted"
OUTCOME_FAILED = "failed"
