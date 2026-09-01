"""Best-of-N selection over offline-scored traces.

Pure functions over rows of the shape {"answer": ..., "features": ...},
where features is a score.json dict from verifier.offline. Selection
strategies return the CHOSEN ROW (so callers can read whatever ground
truth rides along); metrics helpers return floats.

argmax uses offline.rank_key directly. The voting strategies weight by
scalar() — the banded flattening of rank_key — with refuted and
unscoreable voting 0 (a trace whose answer the verifier believes WRONG
must not push that answer's group up). Still a stand-in: to be replaced
by weights fitted (logistic regression on a dev split with benchmark
ground truth) rather than hand-tuned.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .offline import VERDICT_RANK, rank_key

Row = Dict[str, Any]


def scalar(features) -> float:
    """Flatten rank_key's lexicographic tuple into ONE absolute score in
    [0, 1]: six equal verdict bands that can never cross —
    verified [5/6, 1) · supported [4/6, 5/6) · partial [3/6, 4/6) ·
    unverified [2/6, 3/6) · refuted [1/6, 2/6) · unscoreable exactly
    0.0. Within a band, only the count of contrary signals on the final
    claim's contract moves the value (zero signals sits at the top of the
    band; each extra signal lowers it, never out of the band). Selection
    keeps comparing the exact tuple; this is a view, not the decision
    variable."""
    rank, neg_contrary = rank_key(features)
    if rank < 0:
        return 0.0
    return (rank + 1 + 0.999 / (1 - neg_contrary)) / 6


def argmax(rows: List[Row]) -> Optional[Row]:
    """The single best trace by lexicographic rank; first wins ties
    (stable, so reruns select the same trace). Callers that must not
    pretend a tie was decided use argmax_set."""
    if not rows:
        return None
    best = rows[0]
    for row in rows[1:]:
        if rank_key(row.get("features")) > rank_key(best.get("features")):
            best = row
    return best


def argmax_set(rows: List[Row]) -> List[Row]:
    """Every trace sharing the top rank key — the honest read-out when
    the verifier has no information separating them. A singleton means
    the selection was decided; a longer list is an explicit tie, to be
    scored as an abstention (or by expected credit), never as a pick."""
    if not rows:
        return []
    top = max(rank_key(row.get("features")) for row in rows)
    return [row for row in rows if rank_key(row.get("features")) == top]


def _vote(rows: List[Row], weight) -> Optional[Row]:
    """Group by answer, sum weights, return the best row of the winning
    group. Rows without a parseable answer cannot vote; if none can,
    fall back to argmax."""
    groups: Dict[Any, List[Row]] = {}
    for row in rows:
        if row.get("answer") is not None:
            groups.setdefault(row["answer"], []).append(row)
    if not groups:
        return argmax(rows)
    # ties between groups break by the strongest member's rank (then by
    # first appearance, since sort is stable over insertion order)
    def group_key(item):
        _, members = item
        return (sum(weight(r) for r in members),
                max(rank_key(r.get("features")) for r in members))
    _, winners = max(groups.items(), key=group_key)
    return argmax(winners)


def majority_vote(rows: List[Row]) -> Optional[Row]:
    """Self-consistency baseline: one trace, one vote."""
    return _vote(rows, lambda row: 1)


def weighted_vote(rows: List[Row]) -> Optional[Row]:
    """Verifier-weighted voting; weight = the banded scalar, with refuted
    and unscoreable zeroed (see module docstring — fitted weights replace
    this on a dev split). supported traces vote between verified and
    partial, exactly the ladder's read-out."""
    def weight(row):
        features = row.get("features")
        if not features or features.get("unscoreable"):
            return 0.0
        if VERDICT_RANK.get(features.get("verdict"), 0) == 0:
            return 0.0    # a refuted trace must not push its answer up
        return scalar(features)
    return _vote(rows, weight)


# ---- metrics ----------------------------------------------------------------

def pass_at_1(rows: List[Row]) -> float:
    """Mean solved rate over independent samples."""
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get("solved")) / len(rows)


def oracle(rows: List[Row]) -> bool:
    """pass@N ceiling: did ANY sample solve it."""
    return any(r.get("solved") for r in rows)
