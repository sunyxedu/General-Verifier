"""Coverage audit (验漏): claim-level variants exam the obligation set."""
from verifier.agents import COMPILE_OBLIGATIONS_AGENT
from verifier.builder import Builder
from verifier.coverage import CoverageAudit
from verifier.discharge import DischargeResult
from verifier.events import EventLog
from verifier.instruments import ArtifactStore
from verifier.kernel import Kernel
from verifier.model import Obligation
from verifier.roles import FixtureBackend
from verifier.store import ClaimStore


ROW = {"id": "c1", "kind": "assertion", "text": "对所有 n≤100，P(n) 成立"}
OBS = [Obligation("c1", "范围检查", "反例记录")]

VARIANTS = {"variants": [{"text": "V-漏网（假）", "disproof": "跑过，为假"}]}
PATCH = {"text": "输入忠实性检查", "failure_looks_like": "与原始输入不一致的字段"}


class Scripted:
    def __init__(self, fn):
        self.fn = fn

    def discharge(self, row, obligation, method=None, tag=""):
        outcome = self.fn(row["text"], obligation)
        return DischargeResult(outcome, None, "d" * 64, "scripted", "corroborated")


def make(tmp_path, fixtures):
    log = EventLog(str(tmp_path / "e.jsonl"))
    return CoverageAudit(FixtureBackend(fixtures), log), log


def test_covered_variant_adds_nothing(tmp_path):
    audit, log = make(tmp_path, {"variants": VARIANTS, "coverage_patch": PATCH})
    added = audit.audit(ROW, OBS, Scripted(lambda t, o: "refuted"))
    assert added == []
    exam = next(e for e in log.replay() if e["type"] == "coverage_exam")
    assert exam["payload"]["covered"] is True
    assert exam["payload"]["covered_by"] == "范围检查"


def test_hole_yields_patched_obligation(tmp_path):
    audit, log = make(tmp_path, {"variants": VARIANTS, "coverage_patch": PATCH})
    added = audit.audit(ROW, OBS, Scripted(lambda t, o: "holds"))  # nothing catches it
    assert len(added) == 1
    assert added[0].text == "输入忠实性检查" and added[0].falsifiable
    types = [e["type"] for e in log.replay()]
    assert "obligations_added" in types


def test_vacuous_patch_and_duplicates_dropped(tmp_path):
    vacuous = {"text": "优雅", "failure_looks_like": ""}
    audit, _ = make(tmp_path, {"variants": VARIANTS, "coverage_patch": vacuous})
    assert audit.audit(ROW, OBS, Scripted(lambda t, o: "holds")) == []

    dup = {"text": "范围检查", "failure_looks_like": "反例"}   # already an obligation
    audit2, _ = make(tmp_path, {"variants": VARIANTS, "coverage_patch": dup})
    assert audit2.audit(ROW, OBS, Scripted(lambda t, o: "holds")) == []


def test_variant_failure_is_fail_open_with_trace(tmp_path):
    audit, log = make(tmp_path, {"coverage_patch": PATCH})  # no variants fixture
    added = audit.audit(ROW, OBS, Scripted(lambda t, o: "refuted"))
    assert added == []
    exam = next(e for e in log.replay() if e["type"] == "coverage_exam")
    assert "error" in exam["payload"]


def test_kernel_discharges_patched_obligation_too(tmp_path):
    log = EventLog(str(tmp_path / "e.jsonl"))
    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": ROW["text"], "deps": []})
    log.append("commit", {"label": "F", "kind": "final",
                          "text": "结论成立", "deps": ["A"]})

    backend = FixtureBackend({
        "obligations": {"obligations": [
            {"text": "范围检查", "failure_looks_like": "反例记录"}]},
        "variants": VARIANTS,
        "coverage_patch": PATCH,
        "discharge": lambda v: {
            # exam runs (variant text) pass -> hole; real runs hold
            "outcome": "holds", "witness": "", "artifact": ""},
    })
    store = ClaimStore()
    artifacts = ArtifactStore(str(tmp_path / "a"))
    from verifier.discharge import AgentDischarger
    discharger = AgentDischarger(backend, artifacts,
                                 workdir_base=str(tmp_path / "w"))
    kernel = Kernel(store, log, backend, artifacts,
                    obligations_role=COMPILE_OBLIGATIONS_AGENT,
                    discharger=discharger,
                    coverage=CoverageAudit(backend, log))
    builder = Builder(log, store).replay()

    a = builder.labels["A"]
    assert store.status(a) == "verified"
    evidence = store.evidence_for(a)
    texts = {e["obligation"] for e in evidence}
    assert texts == {"范围检查", "输入忠实性检查"}   # the patch was discharged too
