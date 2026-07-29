import pytest

from verifier.model import Claim, UNVERIFIED, VERIFIED, REFUTED, STALE
from verifier.store import ClaimStore, UnknownDependency, IllegalTransition


def make(store, text, kind="assertion", deps=(), **kw):
    claim = Claim(kind, text, deps=deps, **kw)
    store.add_claim(claim)
    return claim.id


def test_dedup_and_unknown_dep():
    s = ClaimStore()
    a1 = make(s, "A")
    a2 = make(s, "A")
    assert a1 == a2
    with pytest.raises(UnknownDependency):
        make(s, "B", deps=("f" * 64,))


def test_trigger_fires_on_first_dependent_only():
    s = ClaimStore()
    fired = []
    s.on_trigger(fired.append)
    a = make(s, "A")
    assert fired == []                     # 出生不触发
    make(s, "B", deps=(a,))
    assert fired == [a]                    # 首个依赖者触发
    make(s, "C", deps=(a,))
    assert fired == [a]                    # 第二个依赖者不触发


def test_trigger_skips_already_verified_parent():
    s = ClaimStore()
    fired = []
    a = make(s, "A")
    s.set_verified(a, "demonstrated")
    s.on_trigger(fired.append)
    make(s, "B", deps=(a,))
    assert fired == []


def test_trigger_fires_for_each_new_parent_once():
    s = ClaimStore()
    fired = []
    s.on_trigger(fired.append)
    a, b = make(s, "A"), make(s, "B")
    make(s, "C", deps=(a, b))
    assert sorted(fired) == sorted([a, b])


def test_status_machine_guards():
    s = ClaimStore()
    a = make(s, "A")
    s.set_refuted(a, "counterexample n=2")
    with pytest.raises(IllegalTransition):
        s.set_verified(a, "demonstrated")   # refuted 不可直接翻成 verified


def test_backward_slice_diamond():
    s = ClaimStore()
    a = make(s, "A")
    b = make(s, "B", deps=(a,))
    c = make(s, "C", deps=(a,))
    f = make(s, "F", kind="final", deps=(b, c))
    assert set(s.backward_slice(f)) == {f, b, c, a}
    orphan = make(s, "orphan")
    assert orphan not in s.backward_slice(f)   # 死分支不在切片


def test_revision_marks_stale_cone():
    s = ClaimStore()
    a = make(s, "A")
    b = make(s, "B", deps=(a,))
    f = make(s, "F", kind="final", deps=(b,))
    s.set_verified(a, "demonstrated")
    s.set_verified(b, "demonstrated")
    a2 = make(s, "A-fixed", revises=a)
    assert s.status(a) == STALE
    assert s.status(b) == STALE                 # 爆炸半径：下游锥全部失效
    assert s.status(f) == STALE
    assert s.status(a2) == UNVERIFIED           # 新 claim 不受影响
    s.set_verified(b, "demonstrated")           # stale 可以被重验


def test_persistence_across_reopen(tmp_path):
    path = str(tmp_path / "store.db")
    s = ClaimStore(path)
    a = make(s, "A")
    s.set_verified(a, "proven")
    s.record_evidence(a, {"instrument": "lean", "outcome": "holds"})
    s2 = ClaimStore(path)
    assert s2.status(a) == VERIFIED and s2.grade(a) == "proven"
    assert s2.evidence_for(a)[0]["instrument"] == "lean"


def test_trigger_fires_after_state_is_consistent():
    s = ClaimStore()
    seen_status = []
    a_holder = {}

    def cb(claim_id):
        # 回调时新 claim 必须已入库、边已建好（依赖者数为 1）
        seen_status.append(s._dependent_count(claim_id))

    s.on_trigger(cb)
    a = make(s, "A")
    a_holder["a"] = a
    make(s, "B", deps=(a,))
    assert seen_status == [1]
