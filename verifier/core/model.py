"""Data contracts. Everything content-addressed; statuses are a
small state machine enforced by the store, not here."""
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

# status machine: unverified -> verified(grade) | refuted ; anything -> stale
UNVERIFIED = "unverified"
VERIFIED = "verified"
REFUTED = "refuted"
STALE = "stale"
PARTIAL = "partial"


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


# internal graph node kind for obligations; deliberately NOT in KINDS so a
# solver can never commit one — only the kernel materializes them
KIND_OBLIGATION = "obligation"


@dataclass(frozen=True)
class Obligation:
    claim_id: str            # context only — excluded from identity
    text: str

    @property
    def id(self) -> str:
        """
        identity is the CONTENT of the check alone: the same
        world-condition compiled under different claims is one graph node,
        so settled verification work transfers across claims for free
        """
        return content_hash({"text": self.text})


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
# the method is right but its instrument cannot be obtained in this
# environment; settles the claim at partial, never silently upgraded
OUTCOME_UNOBTAINABLE = "unobtainable"
