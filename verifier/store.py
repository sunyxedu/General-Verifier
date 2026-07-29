"""Claim store (component 10) + the single trigger rule (component 12) +
backward slicer (component 13). SQLite-backed materialized view of the stream.

The one scheduling rule of the whole system (report §9):
    when an UNVERIFIED claim gains its FIRST dependent, verify it.
The store owns detecting that moment; what "verify" means belongs to the kernel.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Callable, Dict, List, Optional, Set

from .model import (
    Claim, UNVERIFIED, VERIFIED, REFUTED, STALE, KINDS,
)

TriggerCallback = Callable[[str], None]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    id      TEXT PRIMARY KEY,
    kind    TEXT NOT NULL,
    text    TEXT NOT NULL,
    branch  TEXT,
    revises TEXT,
    status  TEXT NOT NULL DEFAULT 'unverified',
    grade   TEXT,
    reason  TEXT
);
CREATE TABLE IF NOT EXISTS deps (
    child  TEXT NOT NULL,
    parent TEXT NOT NULL,
    PRIMARY KEY (child, parent)
);
CREATE TABLE IF NOT EXISTS evidence (
    claim_id TEXT NOT NULL,
    payload  TEXT NOT NULL
);
"""


class UnknownDependency(ValueError):
    pass


class IllegalTransition(ValueError):
    pass


class ClaimStore:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.executescript(_SCHEMA)
        self._trigger_callbacks: List[TriggerCallback] = []

    # ---- wiring --------------------------------------------------------------

    def on_trigger(self, cb: TriggerCallback) -> None:
        """Register the kernel. cb(claim_id) fires when an unverified claim
        gains its first dependent — the system's only scheduling event."""
        self._trigger_callbacks.append(cb)

    # ---- graph construction --------------------------------------------------

    def add_claim(self, claim: Claim) -> str:
        cur = self.db.execute("SELECT 1 FROM claims WHERE id=?", (claim.id,))
        if cur.fetchone():
            return claim.id  # content-addressed dedup: identical claim is a no-op

        for dep in claim.deps:
            if not self._exists(dep):
                raise UnknownDependency(f"dep {dep[:12]}… not in store")
        if claim.revises is not None and not self._exists(claim.revises):
            raise UnknownDependency(f"revises {claim.revises[:12]}… not in store")

        self.db.execute(
            "INSERT INTO claims (id, kind, text, branch, revises) VALUES (?,?,?,?,?)",
            (claim.id, claim.kind, claim.text, claim.branch, claim.revises),
        )

        fired: List[str] = []
        for dep in set(claim.deps):
            had_dependents = self._dependent_count(dep) > 0
            self.db.execute(
                "INSERT OR IGNORE INTO deps (child, parent) VALUES (?,?)",
                (claim.id, dep),
            )
            if not had_dependents and self.status(dep) == UNVERIFIED:
                fired.append(dep)

        if claim.revises is not None:
            self._mark_stale_cone(claim.revises)

        self.db.commit()
        for claim_id in fired:  # fire after commit: callbacks see consistent state
            for cb in self._trigger_callbacks:
                cb(claim_id)
        return claim.id

    # ---- status machine ------------------------------------------------------

    def status(self, claim_id: str) -> str:
        row = self._row(claim_id)
        return row["status"]

    def grade(self, claim_id: str) -> Optional[str]:
        return self._row(claim_id)["grade"]

    def set_verified(self, claim_id: str, grade: str) -> None:
        self._transition(claim_id, VERIFIED, allowed={UNVERIFIED, STALE}, grade=grade)

    def set_refuted(self, claim_id: str, reason: str) -> None:
        self._transition(claim_id, REFUTED, allowed={UNVERIFIED, STALE}, reason=reason)

    def _transition(self, claim_id, new, allowed, grade=None, reason=None):
        current = self.status(claim_id)
        if current not in allowed:
            raise IllegalTransition(f"{current} -> {new} not allowed")
        self.db.execute(
            "UPDATE claims SET status=?, grade=?, reason=? WHERE id=?",
            (new, grade, reason, claim_id),
        )
        self.db.commit()

    def _mark_stale_cone(self, claim_id: str) -> None:
        """Revision semantics (§14): predecessor and everything downstream of it
        lose their verification status — the mechanical form of 'blast radius'."""
        for cid in [claim_id, *self.downstream(claim_id)]:
            self.db.execute("UPDATE claims SET status=?, grade=NULL WHERE id=?", (STALE, cid))

    # ---- evidence ------------------------------------------------------------

    def record_evidence(self, claim_id: str, payload: dict) -> None:
        self.db.execute(
            "INSERT INTO evidence (claim_id, payload) VALUES (?,?)",
            (claim_id, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        )
        self.db.commit()

    def evidence_for(self, claim_id: str) -> List[dict]:
        cur = self.db.execute("SELECT payload FROM evidence WHERE claim_id=?", (claim_id,))
        return [json.loads(r[0]) for r in cur.fetchall()]

    # ---- graph queries -------------------------------------------------------

    def backward_slice(self, claim_id: str) -> List[str]:
        """The dependency closure including the claim itself (cert scope)."""
        return self._walk(claim_id, "SELECT parent FROM deps WHERE child=?")

    def downstream(self, claim_id: str) -> List[str]:
        """Everything that (transitively) depends on claim_id, excluding it."""
        return self._walk(claim_id, "SELECT child FROM deps WHERE parent=?")[1:]

    def _walk(self, start: str, query: str) -> List[str]:
        if not self._exists(start):
            raise UnknownDependency(f"{start[:12]}… not in store")
        seen: Set[str] = {start}
        order: List[str] = [start]
        frontier = [start]
        while frontier:
            next_frontier = []
            for node in frontier:
                for (neighbor,) in self.db.execute(query, (node,)):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        order.append(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        return order

    def get(self, claim_id: str) -> Dict:
        return self._row(claim_id)

    # ---- internals -----------------------------------------------------------

    def _exists(self, claim_id: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM claims WHERE id=?", (claim_id,)
        ).fetchone() is not None

    def _dependent_count(self, claim_id: str) -> int:
        (n,) = self.db.execute(
            "SELECT COUNT(*) FROM deps WHERE parent=?", (claim_id,)
        ).fetchone()
        return n

    def _row(self, claim_id: str) -> Dict:
        cur = self.db.execute(
            "SELECT id, kind, text, branch, revises, status, grade, reason "
            "FROM claims WHERE id=?",
            (claim_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise UnknownDependency(f"{claim_id[:12]}… not in store")
        keys = ("id", "kind", "text", "branch", "revises", "status", "grade", "reason")
        return dict(zip(keys, row))

    def parents(self, claim_id: str) -> List[str]:
        cur = self.db.execute("SELECT parent FROM deps WHERE child=?", (claim_id,))
        return [r[0] for r in cur.fetchall()]

    # ---- search support (M3) -------------------------------------------------

    def find_equivalent(self, claim_id: str) -> Optional[str]:
        """A settled claim with identical (kind, text, dep set) on ANOTHER
        branch. Claim identity includes the branch, so parallel hypotheses that
        re-derive the same lemma are distinct nodes — but verification work is
        content-addressed and transfers."""
        row = self._row(claim_id)
        my_deps = set(self.parents(claim_id))
        cur = self.db.execute(
            "SELECT id FROM claims WHERE kind=? AND text=? AND id!=? "
            "AND status IN (?, ?)",
            (row["kind"], row["text"], claim_id, VERIFIED, REFUTED),
        )
        for (other,) in cur.fetchall():
            if set(self.parents(other)) == my_deps:
                return other
        return None

    def branches(self) -> Dict[str, List[Dict]]:
        cur = self.db.execute(
            "SELECT branch, id, status FROM claims WHERE branch IS NOT NULL"
        )
        out: Dict[str, List[Dict]] = {}
        for branch, cid, status in cur.fetchall():
            out.setdefault(branch, []).append({"id": cid, "status": status})
        return out

    def dead_branches(self) -> List[str]:
        """Branches touched by a refutation or its downstream cone — pruning
        is advisory bookkeeping for the searcher; soundness never relied on it
        (the taint guard already blocks anything built on a refuted claim)."""
        cur = self.db.execute("SELECT id FROM claims WHERE status=?", (REFUTED,))
        dead_ids: Set[str] = set()
        for (cid,) in cur.fetchall():
            dead_ids.add(cid)
            dead_ids.update(self.downstream(cid))
        dead: Set[str] = set()
        for branch, members in self.branches().items():
            if any(m["id"] in dead_ids for m in members):
                dead.add(branch)
        return sorted(dead)
