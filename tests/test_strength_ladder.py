"""The evidence-strength ladder (docs/evidence-strength-ladder.md).

Strength is the second evidence axis, orthogonal to basis: decisive >
bounded > analytic. The binding game executes strictness-descending and
only degrades under audit; a verified claim whose weakest evidence is
below decisive surfaces as "supported" on the certificate; nothing below
decisive may refute.
"""
from __future__ import annotations

import os
import sqlite3

from verifier.core.model import (
    Claim, CLAIM_ASSERTION,
    STRENGTH_ANALYTIC, STRENGTH_BOUNDED, STRENGTH_DECISIVE,
    strength_rank, weakest_strength,
)
from verifier.core.store import ClaimStore


# ---- the lattice ----------------------------------------------------------

def test_strength_rank_orders_the_ladder():
    assert (strength_rank(STRENGTH_DECISIVE)
            > strength_rank(STRENGTH_BOUNDED)
            > strength_rank(STRENGTH_ANALYTIC))


def test_no_declaration_reads_as_decisive():
    """None = the ladder was never in play (direct discharge, ledger
    checks, pre-ladder rows): pre-ladder semantics are preserved."""
    assert strength_rank(None) == strength_rank(STRENGTH_DECISIVE)


def test_unknown_declared_value_reads_as_weakest():
    """A declared-but-unrecognized value must not grant strength."""
    assert strength_rank("formal-ish") == strength_rank(STRENGTH_ANALYTIC)


def test_weakest_strength_is_a_weakest_link_fold():
    assert weakest_strength(
        [STRENGTH_DECISIVE, STRENGTH_BOUNDED, STRENGTH_DECISIVE]
    ) == STRENGTH_BOUNDED
    assert weakest_strength(
        [STRENGTH_BOUNDED, STRENGTH_ANALYTIC]) == STRENGTH_ANALYTIC
    assert weakest_strength(
        [None, STRENGTH_DECISIVE]) == STRENGTH_DECISIVE


def test_weakest_strength_empty_folds_to_decisive():
    assert weakest_strength([]) == STRENGTH_DECISIVE


def test_weakest_strength_normalizes_unknown_values():
    assert weakest_strength([STRENGTH_DECISIVE, "vibes"]) == STRENGTH_ANALYTIC


# ---- persistence ----------------------------------------------------------

def test_store_persists_strength_on_verified(tmp_path):
    store = ClaimStore()
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 42"))
    store.set_verified(cid, basis="world", strength=STRENGTH_BOUNDED)
    row = store.get(cid)
    assert row["strength"] == STRENGTH_BOUNDED
    assert row["basis"] == "world"


def test_store_strength_defaults_to_null(tmp_path):
    store = ClaimStore()
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 42"))
    store.set_verified(cid, basis="world")
    assert store.get(cid)["strength"] is None


def test_store_migrates_pre_strength_databases(tmp_path):
    """A store created before strength tracking gains the column on open
    (CREATE TABLE IF NOT EXISTS never alters an existing table)."""
    path = os.path.join(tmp_path, "claims.sqlite")
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE claims ("
        " id TEXT PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL,"
        " revises TEXT, status TEXT NOT NULL DEFAULT 'unverified',"
        " reason TEXT, basis TEXT);"
        "CREATE TABLE deps (child TEXT NOT NULL, parent TEXT NOT NULL,"
        " PRIMARY KEY (child, parent));"
    )
    db.execute(
        "INSERT INTO claims (id, kind, text, status) VALUES (?,?,?,?)",
        ("old", "assertion", "legacy row", "verified"),
    )
    db.commit()
    db.close()

    store = ClaimStore(path)
    assert store.get("old")["strength"] is None
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "new row"))
    store.set_verified(cid, strength=STRENGTH_DECISIVE)
    assert store.get(cid)["strength"] == STRENGTH_DECISIVE


def test_graph_snapshot_discloses_strength(tmp_path):
    store = ClaimStore()
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 42"))
    store.set_verified(cid, basis="world", strength=STRENGTH_ANALYTIC)
    (entry,) = store.graph_snapshot()
    assert entry["strength"] == STRENGTH_ANALYTIC
