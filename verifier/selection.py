"""Best-of-N selection over offline-scored traces.

Pure functions over rows of the shape {"answer": ..., "features": ...},
where features is a score.json dict from verifier.offline. Selection
strategies return the CHOSEN ROW (so callers can read whatever ground
truth rides along); metrics helpers return floats.

argmax uses offline.rank_key directly. The voting strategies need a
scalar weight per trace; verdict rank is the provisional stand-in — to
be replaced by weights fitted (logistic regression on a dev split with
benchmark ground truth) rather than hand-tuned.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .offline import VERDICT_RANK, rank_key

Row = Dict[str, Any]


def argmax(rows: List[Row]) -> Optional[Row]:
    """The single best trace by lexicographic rank; first wins ties
    (stable, so reruns select the same trace)."""
    if not rows:
        return None
    best = rows[0]
    for row in rows[1:]:
        if rank_key(row.get("features")) > rank_key(best.get("features")):
            best = row
    return best


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
    """Verifier-weighted voting; weight = verdict rank for now (see
    module docstring — fitted weights replace this on a dev split)."""
    def weight(row):
        features = row.get("features")
        if not features or features.get("unscoreable"):
            return 0
        return VERDICT_RANK.get(features.get("verdict"), 0)
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
