"""Binding game — every path offline, roles via FixtureBackend, scripted
discharger so each scenario controls exactly what the 'instrument' sees."""
import pytest

from verifier.agents import COMPILE_OBLIGATIONS_AGENT
from verifier.builder import Builder
from verifier.discharge import DischargeResult
from verifier.events import EventLog
from verifier.game import BindingGame, KIND_RANK
from verifier.instruments import ArtifactStore
from verifier.kernel import Kernel
from verifier.model import Obligation
from verifier.roles import FixtureBackend
from verifier.store import ClaimStore


ROW = {"id": "c1", "kind": "assertion", "text": "对所有 n≤100，P(n) 成立"}
OB = Obligation("c1", "范围内每个 n 通过检查", "一个具体反例记录")

METHODS = {"methods": [
    {"procedure": "读一遍", "instrument_id": "weak", "evidence_kind": "text",
     "artifact": "笔记", "failure_looks_like": "矛盾句"},
    {"procedure": "逐个执行", "instrument_id": "strong", "evidence_kind": "execute",
     "artifact": "运行日志", "failure_looks_like": "非零余数"},
]}
VARIANTS = {"variants": [
    {"text": "V1（假）", "disproof": "跑过，确实假"},
    {"text": "V2（假）", "disproof": "跑过，确实假"},
]}
AUDIT_YES = {"entails": "yes", "grade": "demonstrated", "reason": "重跑一致"}


class Scripted:
    """discharge(row, obligation, method, tag) -> scripted DischargeResult."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append({"text": row["text"], "tag": tag,
                           "instrument": method["instrument_id"] if method else None})
        outcome, witness = self.fn(row["text"], method, tag)
        return DischargeResult(outcome, witness, "d" * 64, "scripted", "corroborated")


def make(tmp_path, fixtures, fn, **kw):
    log = EventLog(str(tmp_path / "e.jsonl"))
    artifacts = ArtifactStore(str(tmp_path / "a"))
    backend = FixtureBackend(fixtures)
    game = BindingGame(backend, artifacts, log, Scripted(fn), n_variants=2, **kw)
    return game, log, artifacts


def refute_variants_hold_original(text, method, tag):
    return ("refuted", "变体反例") if text.startswith("V") else ("holds", "")


def test_happy_path_binds_strongest_method(tmp_path):
    game, log, artifacts = make(
        tmp_path,
        {"oracles": METHODS, "variants": VARIANTS, "entailment": AUDIT_YES},
        refute_variants_hold_original,
    )
    result = game.run(ROW, OB)
    assert result.outcome == "holds"
    assert result.instrument == "strong"          # execute outranks text
    assert result.grade == "demonstrated"         # min(ceiling execute, audit)
    record = artifacts.get(result.artifact)
    assert record["bound"]["method"]["instrument_id"] == "strong"
    assert len(record["attempts"]) == 1
    assert len(record["attempts"][0]["discrimination"]) == 2
    types = [e["type"] for e in log.replay()]
    assert types == ["methods_enumerated", "variants_generated",
                     "discrimination", "discrimination",
                     "entailment_audit", "method_bound"]


def test_false_passing_method_rejected_then_downgrade(tmp_path):
    def fn(text, method, tag):
        if text.startswith("V"):
            # strong instrument false-passes the exam; weak one refutes
            return ("holds", "") if method["instrument_id"] == "strong" else ("refuted", "")
        return ("holds", "")

    game, log, artifacts = make(
        tmp_path,
        {"oracles": METHODS, "variants": VARIANTS, "entailment": AUDIT_YES},
        fn,
    )
    result = game.run(ROW, OB)
    assert result.outcome == "holds"
    assert result.instrument == "weak"            # downgraded to the survivor
    assert result.grade == "corroborated"         # ceiling of text < audit grade
    rejected = [e for e in log.replay() if e["type"] == "method_rejected"]
    assert rejected[0]["payload"]["instrument_id"] == "strong"
    assert "false pass" in rejected[0]["payload"]["reason"]
    record = artifacts.get(result.artifact)
    assert record["attempts"][0]["rejected"].startswith("false pass")


def test_entailment_no_downgrades_then_exhausts(tmp_path):
    audits = iter([
        {"entails": "no", "grade": "", "reason": "抽样蕴含不了全称"},
        {"entails": "no", "grade": "", "reason": "证据只是转述"},
    ])
    game, log, _ = make(
        tmp_path,
        {"oracles": METHODS, "variants": VARIANTS,
         "entailment": lambda v: next(audits)},
        refute_variants_hold_original,
    )
    result = game.run(ROW, OB)
    assert result.outcome == "failed"
    assert result.witness == "all candidate methods rejected"
    reasons = [e["payload"]["reason"] for e in log.replay()
               if e["type"] == "method_rejected"]
    assert "抽样蕴含不了全称" in reasons[0]


def test_no_exam_fails_closed(tmp_path):
    no_disproof = {"variants": [{"text": "V1", "disproof": "  "}]}
    game, log, _ = make(
        tmp_path,
        {"oracles": METHODS, "variants": no_disproof, "entailment": AUDIT_YES},
        refute_variants_hold_original,
    )
    result = game.run(ROW, OB)
    assert result.outcome == "failed"
    assert result.witness == "no discrimination exam"


def test_no_methods_fails_closed(tmp_path):
    game, _, _ = make(
        tmp_path,
        {"oracles": {"methods": []}, "variants": VARIANTS, "entailment": AUDIT_YES},
        refute_variants_hold_original,
    )
    assert game.run(ROW, OB).witness == "no candidate methods"


def test_refuted_short_circuits_without_audit(tmp_path):
    def fn(text, method, tag):
        return ("refuted", "n=7 不成立") if tag == "bound" else ("refuted", "")

    calls = {"audit": 0}

    def audit(v):
        calls["audit"] += 1
        return AUDIT_YES

    game, log, _ = make(
        tmp_path,
        {"oracles": METHODS, "variants": VARIANTS, "entailment": audit},
        fn,
    )
    result = game.run(ROW, OB)
    assert result.outcome == "refuted" and result.witness == "n=7 不成立"
    assert calls["audit"] == 0                     # asymmetric: no audit on refute
    assert result.grade == "demonstrated"          # ceiling of execute


def test_execution_failure_downgrades(tmp_path):
    def fn(text, method, tag):
        if text.startswith("V"):
            return ("refuted", "")
        if method["instrument_id"] == "strong":
            return ("failed", "timeout")           # unreachable at retry limit
        return ("holds", "")

    game, log, _ = make(
        tmp_path,
        {"oracles": METHODS, "variants": VARIANTS, "entailment": AUDIT_YES},
        fn,
    )
    result = game.run(ROW, OB)
    assert result.instrument == "weak"
    reasons = [e["payload"]["reason"] for e in log.replay()
               if e["type"] == "method_rejected"]
    assert reasons == ["execution failure at retry limit"]


def test_methods_mechanically_resorted_and_capped(tmp_path):
    many = {"methods": [
        {"procedure": "p", "instrument_id": f"i{k}", "evidence_kind": kind,
         "artifact": "a", "failure_looks_like": "f"}
        for k, kind in enumerate(["text", "search", "formal", "rederive", "execute"])
    ]}
    game, log, _ = make(
        tmp_path,
        {"oracles": many, "variants": VARIANTS, "entailment": AUDIT_YES},
        refute_variants_hold_original,
        max_methods=3,
    )
    game.run(ROW, OB)
    enumerated = next(e for e in log.replay() if e["type"] == "methods_enumerated")
    kinds = [m["evidence_kind"] for m in enumerated["payload"]["methods"]]
    assert kinds == ["formal", "execute", "rederive"]   # strongest 3, in order
    assert sorted(KIND_RANK, key=KIND_RANK.get)[0] == "formal"


def test_bogus_exam_question_disqualified_not_the_method(tmp_path):
    """All candidates hold on V1; the audit finds V1 is not actually false ->
    V1 is disqualified, methods are judged on V2 only, strongest binds."""
    def fn(text, method, tag):
        if text == "V1（假）":
            return ("holds", "")          # every method holds on the bogus one
        if text == "V2（假）":
            return ("refuted", "")
        return ("holds", "")

    audits = []

    def validate(v):
        audits.append(v["variant_text"])
        return {"actually_false": "no", "reason": "复跑发现它其实为真"}

    game, log, artifacts = make(
        tmp_path,
        {"oracles": METHODS, "variants": VARIANTS, "entailment": AUDIT_YES,
         "validate_variant": validate},
        fn,
    )
    result = game.run(ROW, OB)
    assert result.outcome == "holds"
    assert result.instrument == "strong"          # method NOT blamed
    assert audits == ["V1（假）"]                  # audited exactly once (memo)
    types = [e["type"] for e in log.replay()]
    assert "variant_disqualified" in types
    assert "method_rejected" not in types


def test_confirmed_exam_question_still_rejects_method(tmp_path):
    """Audit says the variant IS false -> the hold was a real blind spot."""
    def fn(text, method, tag):
        if text.startswith("V"):
            return ("holds", "") if method["instrument_id"] == "strong" else ("refuted", "")
        return ("holds", "")

    game, log, _ = make(
        tmp_path,
        {"oracles": METHODS, "variants": VARIANTS, "entailment": AUDIT_YES,
         "validate_variant": {"actually_false": "yes", "reason": "确实假"}},
        fn,
    )
    result = game.run(ROW, OB)
    assert result.instrument == "weak"            # strong rejected as before
    rejected = [e for e in log.replay() if e["type"] == "method_rejected"]
    assert "false pass" in rejected[0]["payload"]["reason"]


def test_all_questions_disqualified_fails_closed(tmp_path):
    game, log, _ = make(
        tmp_path,
        {"oracles": METHODS, "variants": VARIANTS, "entailment": AUDIT_YES,
         "validate_variant": {"actually_false": "no", "reason": "都不假"}},
        lambda text, method, tag: ("holds", ""),
    )
    result = game.run(ROW, OB)
    assert result.outcome == "failed"
    assert result.witness == "no usable discrimination exam"


def test_audit_failure_keeps_question_fail_closed(tmp_path):
    """No validate_variant fixture -> RoleError -> question kept -> reject."""
    game, log, _ = make(
        tmp_path,
        {"oracles": METHODS, "variants": VARIANTS, "entailment": AUDIT_YES},
        lambda text, method, tag: ("holds", "") if text.startswith("V")
        else ("holds", ""),
    )
    result = game.run(ROW, OB)
    assert result.outcome == "failed"             # every method rejected
    assert result.witness == "all candidate methods rejected"


def test_kernel_with_binder_seat_end_to_end(tmp_path):
    log = EventLog(str(tmp_path / "e.jsonl"))
    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": "对所有 n≤100，P(n) 成立", "deps": []})
    log.append("commit", {"label": "F", "kind": "final",
                          "text": "结论成立", "deps": ["A"]})

    backend = FixtureBackend({
        "obligations": {"obligations": [
            {"text": "范围内每个 n 通过检查",
             "failure_looks_like": "一个具体反例记录"}]},
        "oracles": METHODS, "variants": VARIANTS, "entailment": AUDIT_YES,
    })
    store = ClaimStore()
    artifacts = ArtifactStore(str(tmp_path / "a"))
    game = BindingGame(backend, artifacts, log,
                       Scripted(refute_variants_hold_original), n_variants=2)
    kernel = Kernel(store, log, backend, artifacts,
                    obligations_role=COMPILE_OBLIGATIONS_AGENT, binder=game)
    builder = Builder(log, store).replay()

    a = builder.labels["A"]
    assert store.status(a) == "verified"
    assert store.grade(a) == "demonstrated"
    ev = store.evidence_for(a)[0]
    assert ev["instrument"] == "strong" and ev["grade"] == "demonstrated"
