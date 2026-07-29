import pytest

from verifier.model import (
    Claim, Obligation, Evidence, canonical, content_hash, weakest_grade,
    KIND_ASSERTION, KIND_EFFECT,
)


def test_canonical_is_deterministic_and_unicode_stable():
    a = {"b": 1, "a": "中文", "c": [3, 2]}
    b = {"c": [3, 2], "a": "中文", "b": 1}
    assert canonical(a) == canonical(b)
    assert content_hash(a) == content_hash(b)


def test_claim_id_stable_and_dep_order_insensitive():
    c1 = Claim(KIND_ASSERTION, "6 | n^3 - n", deps=("x", "y"))
    c2 = Claim(KIND_ASSERTION, "6 | n^3 - n", deps=("y", "x"))
    assert c1.id == c2.id  # dep *set* defines position, not order


def test_claim_id_sensitive_to_content_deps_branch():
    base = Claim(KIND_ASSERTION, "t", deps=("a",))
    assert base.id != Claim(KIND_ASSERTION, "t2", deps=("a",)).id
    assert base.id != Claim(KIND_ASSERTION, "t", deps=("b",)).id
    assert base.id != Claim(KIND_ASSERTION, "t", deps=("a",), branch="b1").id
    assert base.id != Claim(KIND_EFFECT, "t", deps=("a",)).id


def test_revises_is_lineage_not_identity():
    a = Claim(KIND_ASSERTION, "t")
    b = Claim(KIND_ASSERTION, "t", revises="deadbeef")
    assert a.id == b.id  # identical content dedups regardless of lineage


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        Claim("banana", "t")


def test_obligation_falsifiability():
    good = Obligation("c", "every n in [1,100] passes", "an n with nonzero remainder")
    bad = Obligation("c", "proof is elegant", "   ")
    assert good.falsifiable and not bad.falsifiable
    assert good.id != bad.id


def test_weakest_grade():
    assert weakest_grade(["proven", "demonstrated"]) == "demonstrated"
    assert weakest_grade(["proven", "corroborated", "demonstrated"]) == "corroborated"
    assert weakest_grade([]) is None


def test_evidence_id_covers_outcome():
    e1 = Evidence("o", "python", "holds", {"exit": 0})
    e2 = Evidence("o", "python", "refuted", {"exit": 0}, witness={"n": 2})
    assert e1.id != e2.id
