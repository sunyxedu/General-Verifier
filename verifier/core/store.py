"""
Claim store + the single trigger rule + backward slicer. SQLite-backed materialized view of the stream.

Two mechanical carve-outs keep the rule cheap and total:
  * EFFECT claims stay lazy. They are the ledger — hundreds per run, mechanical
    digest checks, and meaningful only once some claim leans on one.
  * The first-dependent moment still fires for anything left unverified, so a
    claim whose check failed closed gets retried when it becomes load-bearing.
The certificate remains the backstop: it settles the final's whole slice.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Callable, Dict, List, Optional, Set

from .model import (
    Claim, CLAIM_EFFECT, OBLIGATION_KINDS,
    STATUS_UNVERIFIED, STATUS_VERIFIED, STATUS_REFUTED, STATUS_STALE, STATUS_PARTIAL,
)

TriggerCallback = Callable[[str], None]

TRIGGER_ON_COMMIT = "on-commit"
TRIGGER_ON_FIRST_DEPENDENT = "on-first-dependent"
TRIGGERS = (TRIGGER_ON_COMMIT, TRIGGER_ON_FIRST_DEPENDENT)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    id      TEXT PRIMARY KEY,
    kind    TEXT NOT NULL,
    text    TEXT NOT NULL,
    revises TEXT,
    status  TEXT NOT NULL DEFAULT 'unverified',
    reason  TEXT,
    -- evidence basis of a settled node: 'world' (independently established;
    -- downstream may rely on the content) or 'record' (the ledger attests the
    -- solver reported/did it; proves reporting, not truth). For refuted nodes
    -- it is the refuting evidence's basis. NULL until settled.
    basis   TEXT,
    -- evidence strength of a settled node (decisive | bounded | analytic);
    -- NULL = settled outside the ladder (direct discharge, ledger checks,
    -- pre-ladder rows), read as decisive downstream.
    strength TEXT
);
CREATE TABLE IF NOT EXISTS deps (
    child  TEXT NOT NULL,
    parent TEXT NOT NULL,
    PRIMARY KEY (child, parent)
);
-- obligation edges live OUTSIDE claim identity: a claim's deps are fixed at
-- commit time, but its obligations are materialized later by the kernel
CREATE TABLE IF NOT EXISTS obligation_edges (
    claim      TEXT NOT NULL,
    obligation TEXT NOT NULL,
    PRIMARY KEY (claim, obligation)
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
    def __init__(self, path: str = ":memory:", trigger: str = TRIGGER_ON_COMMIT):
        if trigger not in TRIGGERS:
            raise ValueError(f"unknown trigger policy: {trigger}")
        self.db = sqlite3.connect(path)
        self.db.executescript(_SCHEMA)
        # migration: stores created before evidence-basis tracking lack the
        # column (CREATE TABLE IF NOT EXISTS never alters an existing table)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(claims)")}
        if "basis" not in cols:
            self.db.execute("ALTER TABLE claims ADD COLUMN basis TEXT")
            self.db.commit()
        if "strength" not in cols:
            self.db.execute("ALTER TABLE claims ADD COLUMN strength TEXT")
            self.db.commit()
        self.trigger = trigger
        self._trigger_callbacks: List[TriggerCallback] = []

    # wiring
    def on_trigger(self, cb: TriggerCallback) -> None:
        """
        Register the kernel. 
        cb(claim_id) fires when a claim is committed (or has afirst dependent), the system's only scheduling event.
        """
        self._trigger_callbacks.append(cb)

    # graph construction
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
            "INSERT INTO claims (id, kind, text, revises) VALUES (?,?,?,?)",
            (claim.id, claim.kind, claim.text, claim.revises),
        )

        fired: List[str] = []
        for dep in set(claim.deps):
            had_dependents = self._dependent_count(dep) > 0
            self.db.execute(
                "INSERT OR IGNORE INTO deps (child, parent) VALUES (?,?)",
                (claim.id, dep),
            )
            if not had_dependents and self.status(dep) == STATUS_UNVERIFIED:
                fired.append(dep)

        if claim.revises is not None:
            self._mark_stale_cone(claim.revises)

        self.db.commit()
        
        if self.trigger == TRIGGER_ON_COMMIT and claim.kind != CLAIM_EFFECT:
            fired.append(claim.id)
        for claim_id in fired:
            for cb in self._trigger_callbacks:
                cb(claim_id)
        return claim.id

    # status machine
    def status(self, claim_id: str) -> str:
        row = self._row(claim_id)
        return row["status"]

    def set_verified(self, claim_id: str, basis: Optional[str] = None,
                     strength: Optional[str] = None) -> None:
        self._transition(claim_id, STATUS_VERIFIED,
                         allowed={STATUS_UNVERIFIED, STATUS_STALE},
                         basis=basis, strength=strength)

    def set_refuted(self, claim_id: str, reason: str,
                    basis: Optional[str] = None,
                    strength: Optional[str] = None) -> None:
        # basis = the refuting evidence's achieved basis: it decides
        # whether the refutation may transfer by inheritance (a record
        # counterexample cannot settle a world demand) and is disclosed
        # on the certificate. strength is disclosure only — mechanics
        # upstream admit no refutation below decisive.
        self._transition(claim_id, STATUS_REFUTED,
                         allowed={STATUS_UNVERIFIED, STATUS_STALE},
                         reason=reason, basis=basis, strength=strength)

    def set_partial(self, claim_id: str) -> None:
        self._transition(claim_id, STATUS_PARTIAL,
                         allowed={STATUS_UNVERIFIED, STATUS_STALE})

    def set_unverified(self, claim_id: str,
                       reason: Optional[str] = None) -> None:
        """Finish a stale/failed recheck without claiming new evidence."""
        self._transition(claim_id, STATUS_UNVERIFIED,
                         allowed={STATUS_UNVERIFIED, STATUS_STALE},
                         reason=reason)

    def _transition(self, claim_id, new, allowed, reason=None, basis=None,
                    strength=None):
        current = self.status(claim_id)
        if current not in allowed:
            raise IllegalTransition(f"{current} -> {new} not allowed")
        self.db.execute(
            "UPDATE claims SET status=?, reason=?, basis=?, strength=? "
            "WHERE id=?",
            (new, reason, basis, strength, claim_id),
        )
        self.db.commit()

    def _mark_stale_cone(self, claim_id: str) -> None:
        for cid in [claim_id, *self.downstream(claim_id)]:
            self.db.execute("UPDATE claims SET status=? WHERE id=?", (STATUS_STALE, cid))

    # evidence
    def record_evidence(self, claim_id: str, payload: dict) -> None:
        self.db.execute(
            "INSERT INTO evidence (claim_id, payload) VALUES (?,?)",
            (claim_id, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        )
        self.db.commit()

    def evidence_for(self, claim_id: str) -> List[dict]:
        cur = self.db.execute("SELECT payload FROM evidence WHERE claim_id=?", (claim_id,))
        return [json.loads(r[0]) for r in cur.fetchall()]

    # graph queries
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

    # internals
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
            "SELECT id, kind, text, revises, status, reason, basis, strength "
            "FROM claims WHERE id=?",
            (claim_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise UnknownDependency(f"{claim_id[:12]}… not in store")
        keys = ("id", "kind", "text", "revises", "status", "reason", "basis",
                "strength")
        return dict(zip(keys, row))

    def parents(self, claim_id: str) -> List[str]:
        cur = self.db.execute("SELECT parent FROM deps WHERE child=?", (claim_id,))
        return [r[0] for r in cur.fetchall()]

    # obligation nodes
    def add_obligation(self, claim_id: str, obligation) -> str:
        """Materialize an obligation as a first-class graph node
        (kind='obligation') and link it to its claim. Content-addressed:
        the same check compiled under another claim is the SAME node, so a
        settled verdict transfers for free. No trigger fires — obligation
        scheduling is the kernel's job, not the store's."""
        oid = obligation.id
        cur = self.db.execute("SELECT 1 FROM claims WHERE id=?", (oid,))
        if not cur.fetchone():
            self.db.execute(
                "INSERT INTO claims (id, kind, text) VALUES (?,?,?)",
                (oid, OBLIGATION_KINDS, obligation.text),
            )
        self.db.execute(
            "INSERT OR IGNORE INTO obligation_edges (claim, obligation) "
            "VALUES (?,?)",
            (claim_id, oid),
        )
        self.db.commit()
        return oid

    def obligation_children(self, claim_id: str) -> List[Dict]:
        """The obligation nodes materialized under a claim, full rows."""
        cur = self.db.execute(
            "SELECT obligation FROM obligation_edges WHERE claim=?",
            (claim_id,),
        )
        return [self._row(oid) for (oid,) in cur.fetchall()]

    def settled_statement(self, text: str,
                          exclude: Optional[str] = None) -> Optional[Dict]:
        """A currently-trusted CLAIM (assertion/final) that settled exactly
        this statement — verified or refuted, identical text. Lets an
        obligation node inherit the verdict of a same-text claim: the two
        live in different identity spaces (claim id hashes kind/deps
        too), so plain node dedup cannot see across. Stale and partial
        claims never transfer; effects are ledger refs, not statements."""
        cur = self.db.execute(
            "SELECT id, status FROM claims "
            "WHERE text=? AND kind IN ('assertion', 'final') "
            "AND status IN (?, ?) AND id != ? LIMIT 1",
            (text, STATUS_VERIFIED, STATUS_REFUTED, exclude or ""),
        )
        row = cur.fetchone()
        return {"id": row[0], "status": row[1]} if row else None

    def graph_snapshot(self, exclude: Optional[str] = None,
                       cap: int = 150) -> List[Dict]:
        """A compact view of the WHOLE graph for the obligation compiler:
        every assertion/final/obligation node with its status, so already-
        proven statements can be found and relied on instead of re-demanded.
        Effects are omitted (their text is a ledger hash, useless to read)."""
        cur = self.db.execute(
            "SELECT id, kind, status, text, basis, strength FROM claims "
            "WHERE kind != 'effect'"
        )
        out: List[Dict] = []
        for cid, kind, status, text, basis, strength in cur.fetchall():
            if cid == exclude:
                continue
            entry = {"kind": kind, "status": status, "text": text[:300]}
            if basis:
                entry["basis"] = basis
            if strength:
                entry["strength"] = strength
            out.append(entry)
            if len(out) >= cap:
                out.append({"note": f"...truncated at {cap} entries"})
                break
        return out
