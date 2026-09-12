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

# evidence strength of a settled verdict: HOW decisive the winning method
# was, orthogonal to basis (whose account the evidence rests on).
# decisive — a decision procedure for the obligation exactly as written;
#            the only strength entitled to settle a refutation.
# bounded  — a sound instrument for a strictly narrower proposition plus a
#            declared, finite list of bridging assumptions that transport
#            its positive verdict to the obligation (a unit test, a
#            depth-bounded search). Establishes only; never refutes.
# analytic — a disciplined reading of pinned world artifacts with a
#            concrete witness trace. Weakest; establishes only.
# The claim-level strength is a weakest-link fold; a verified claim whose
# fold is below decisive surfaces on the certificate as "supported".
STRENGTH_DECISIVE = "decisive"
STRENGTH_BOUNDED = "bounded"
STRENGTH_ANALYTIC = "analytic"
STRENGTHS = (STRENGTH_DECISIVE, STRENGTH_BOUNDED, STRENGTH_ANALYTIC)

_STRENGTH_RANK = {STRENGTH_ANALYTIC: 0, STRENGTH_BOUNDED: 1,
                  STRENGTH_DECISIVE: 2}


def strength_rank(strength: Optional[str]) -> int:
    """Ordering rank, higher is stronger. None means the ladder was never
    in play (direct discharge, ledger checks, pre-ladder rows) and reads
    as decisive, preserving pre-ladder semantics verbatim. A DECLARED but
    unrecognized value reads as the weakest tier — an unvalidated typo
    must not grant strength (the LLM boundary is schema-gated; this
    guards the code paths, mirroring basis_satisfies)."""
    if strength is None:
        return _STRENGTH_RANK[STRENGTH_DECISIVE]
    return _STRENGTH_RANK.get(strength, _STRENGTH_RANK[STRENGTH_ANALYTIC])


def weakest_strength(strengths) -> str:
    """Fold a claim's strength over its obligations': the weakest link
    wins. Empty folds to decisive (nothing present to demote); unknown
    declared values normalize to analytic."""
    worst = STRENGTH_DECISIVE
    worst_rank = _STRENGTH_RANK[STRENGTH_DECISIVE]
    for strength in strengths:
        rank = strength_rank(strength)
        if rank < worst_rank:
            worst = (strength if strength in _STRENGTH_RANK
                     else STRENGTH_ANALYTIC)
            worst_rank = rank
    return worst


# obligation severity — the modality of the task clause an obligation is
# compiled from, NOT its topic. MUST: a deliverable, postcondition, or an
# explicit hard constraint whose violation makes the task incomplete; its
# refutation refutes the claim. SHOULD: advice, preference, or a soft
# constraint the task states without making acceptance depend on it; its
# refutation is a contrary signal that can only lower a verdict (to
# partial), never refute. Defaults to MUST: an undeclared clause is read
# in the fail-closed direction.
SEVERITY_MUST = "must"
SEVERITY_SHOULD = "should"
SEVERITIES = (SEVERITY_MUST, SEVERITY_SHOULD)

# witness kinds — what a refutation is made of. Only a POSITIVE, retained,
# replayable observation may refute: an execution whose recorded output
# contradicts the obligation, a world artifact whose recorded content
# contradicts it, or two ledger records that contradict each other.
# ABSENCE (nothing found, nothing printed, nothing captured) refutes only
# an obligation compiled as a presence requirement (refutable_by_absence);
# everywhere else it is an evidence gap. NONE = no witness was delivered.
WITNESS_EXECUTION = "execution"
WITNESS_WORLD_STATE = "world_state"
WITNESS_RECORD_CONTRADICTION = "record_contradiction"
WITNESS_ABSENCE = "absence"
WITNESS_NONE = "none"
WITNESS_KINDS = (WITNESS_EXECUTION, WITNESS_WORLD_STATE,
                 WITNESS_RECORD_CONTRADICTION, WITNESS_ABSENCE, WITNESS_NONE)
POSITIVE_WITNESS_KINDS = (WITNESS_EXECUTION, WITNESS_WORLD_STATE,
                          WITNESS_RECORD_CONTRADICTION)


def witness_admissible(kind, refutable_by_absence: bool) -> bool:
    """Whether a refutation carried by a witness of this kind may settle.
    Unknown or missing kinds read as NONE (fail closed)."""
    if kind in POSITIVE_WITNESS_KINDS:
        return True
    return kind == WITNESS_ABSENCE and bool(refutable_by_absence)


# provenance classes of an input a check consumed — where the bytes came
# from, independent of what the check did with them. The independence
# axiom is enforced on the (provenance, role) pair: a DELIVERED artifact
# (anything the solver produced or changed) may be the SUBJECT of a check
# or an INPUT fed to it, but never the AUTHORITY that decides whether the
# subject is right — that is the circular certificate.
PROVENANCE_TASK_GIVEN = "task_given"     # supplied by the task, unchanged
PROVENANCE_DELIVERED = "delivered"       # produced or modified by the solver
PROVENANCE_LEDGER = "ledger"             # the recorded action stream
PROVENANCE_VERIFIER = "verifier"         # computed by the verifier itself
PROVENANCE_PUBLIC = "public"             # a public standard / reference
PROVENANCES = (PROVENANCE_TASK_GIVEN, PROVENANCE_DELIVERED,
               PROVENANCE_LEDGER, PROVENANCE_VERIFIER, PROVENANCE_PUBLIC)
ROLE_SUBJECT = "subject"       # the thing being checked
ROLE_INPUT = "input"           # fed into the check
ROLE_AUTHORITY = "authority"   # decides what the right answer is
INPUT_ROLES = (ROLE_SUBJECT, ROLE_INPUT, ROLE_AUTHORITY)


def circular_inputs(inputs) -> list:
    """The declared inputs that make a positive verdict circular: a
    delivered artifact standing as authority for the check."""
    out = []
    for item in inputs or ():
        if not isinstance(item, dict):
            continue
        if item.get("provenance") == PROVENANCE_DELIVERED \
                and item.get("role") == ROLE_AUTHORITY:
            out.append(item.get("ref", "?"))
    return out


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
        # Ordinary repeated commits still dedup. An explicit revision is
        # a new verification epoch even when the statement is unchanged:
        # the solver may have repaired the world rather than its wording.
        identity = {"kind": self.kind, "text": self.text,
                    "deps": sorted(self.deps)}
        if self.revises is not None:
            identity["revises"] = self.revises
        return content_hash(identity)


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
    # Reliance metadata: severity and clause are edge attributes.
    # severity — must | should (see SEVERITY_*): decides whether a
    #   refutation of the node refutes the claim or only lowers it;
    # clause — the verbatim source span the obligation is compiled from,
    #   retained so necessity can be audited against the controlling text;
    # refutable_by_absence — the obligation is a presence requirement
    #   (a named deliverable must exist), so an absence witness may refute
    #   it; everywhere else absence is an evidence gap.
    severity: str = SEVERITY_MUST
    clause: str = ""
    refutable_by_absence: bool = False
    # Set by the kernel, never by a model. Conditional checks may share a
    # verdict only within the same set of assumable premises. Finals use ().
    premise_scope: Tuple[str, ...] = ()
    # Host-owned verification context (profile, subject and evidence epoch).
    # Empty keeps the original identity exactly; models cannot set this field.
    verification_context: str = ""

    def __post_init__(self):
        # basis_satisfies reads any unknown value as a low floor — an
        # unvalidated typo would silently disable the gate. The LLM
        # boundary is schema-gated; this guards the code paths.
        if self.required_basis not in (BASIS_WORLD, BASIS_RECORD):
            raise ValueError(
                f"unknown required_basis: {self.required_basis!r}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity: {self.severity!r}")
        object.__setattr__(self, "premise_scope",
                           tuple(sorted(set(self.premise_scope))))
        if not isinstance(self.verification_context, str):
            raise ValueError("verification_context must be a string")

    @property
    def id(self) -> str:
        """
        Identity includes every condition on evidence reuse. Different
        assumption scopes or absence rules need separate verdicts and
        evidence histories, even when the statement and floor coincide.
        Versioning also prevents reuse of legacy unscoped nodes.
        """
        identity = {"schema": "gv-obligation-v2", "text": self.text,
                    "required": self.required_basis,
                    "refutable_by_absence": self.refutable_by_absence,
                    "premise_scope": self.premise_scope}
        if self.verification_context:
            identity["verification_context"] = self.verification_context
        return content_hash(identity)


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
