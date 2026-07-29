"""M3 search semantics: cross-branch dedup, asymmetric pruning, OR resolution."""
import json

from verifier.builder import Builder
from verifier.certificate import request_certificate
from verifier.events import EventLog
from verifier.instruments import ArtifactStore
from verifier.kernel import Kernel
from verifier.roles import FixtureBackend
from verifier.sidecar import Sidecar
from verifier.store import ClaimStore

HOLDS = 'import json\nprint(json.dumps({"holds": True}))'
REFUTES = 'import json\nprint(json.dumps({"holds": False, "witness": {"n": 2}}))'


def world(tmp_path, check_by_text):
    log = EventLog(str(tmp_path / "e.jsonl"))
    store = ClaimStore()
    backend = FixtureBackend({
        "compile_obligations": {"obligations": [
            {"text": "O1", "failure_looks_like": "反例"}]},
        "write_check": lambda v: {"check_source": check_by_text(v["claim_text"])},
    })
    kernel = Kernel(store, log, backend,
                    ArtifactStore(str(tmp_path / "a")))
    return log, store, backend, kernel


def test_cross_branch_dedup_transfers_verdict(tmp_path):
    log, store, backend, kernel = world(tmp_path, lambda t: HOLDS)
    log.append("commit", {"label": "L1", "kind": "assertion", "text": "共享引理",
                          "deps": [], "branch": "b1"})
    log.append("commit", {"label": "C1", "kind": "assertion", "text": "分支1结论",
                          "deps": ["L1"], "branch": "b1"})
    log.append("commit", {"label": "L2", "kind": "assertion", "text": "共享引理",
                          "deps": [], "branch": "b2"})
    log.append("commit", {"label": "C2", "kind": "assertion", "text": "分支2结论",
                          "deps": ["L2"], "branch": "b2"})
    builder = Builder(log, store).replay()

    assert builder.labels["L1"] != builder.labels["L2"]   # branch in identity
    assert store.status(builder.labels["L1"]) == "verified"
    assert store.status(builder.labels["L2"]) == "verified"
    # second branch paid nothing: verdict transferred, evidence says dedup
    hits = [e for e in log.replay() if e["type"] == "dedup_hit"]
    assert len(hits) == 1 and hits[0]["payload"]["source"] == builder.labels["L1"]
    ev = store.evidence_for(builder.labels["L2"])
    assert ev[0]["instrument"] == "dedup"
    # the instruments ran exactly once (one write_check for L1 only)
    checks = [c for c in backend.calls if c["role"] == "write_check"]
    assert len(checks) == 1


def test_refuted_dedup_transfers_too(tmp_path):
    log, store, backend, kernel = world(tmp_path, lambda t: REFUTES)
    log.append("commit", {"label": "L1", "kind": "assertion", "text": "假引理",
                          "deps": [], "branch": "b1"})
    log.append("commit", {"label": "C1", "kind": "assertion", "text": "c1",
                          "deps": ["L1"], "branch": "b1"})
    log.append("commit", {"label": "L2", "kind": "assertion", "text": "假引理",
                          "deps": [], "branch": "b2"})
    log.append("commit", {"label": "C2", "kind": "assertion", "text": "c2",
                          "deps": ["L2"], "branch": "b2"})
    builder = Builder(log, store).replay()
    assert store.status(builder.labels["L2"]) == "refuted"
    assert store.get(builder.labels["L2"])["reason"].startswith("dedup of")


def test_refutation_prunes_cone_and_marks_branch_dead(tmp_path):
    log, store, backend, kernel = world(
        tmp_path, lambda t: REFUTES if "假设X" in t else HOLDS)
    log.append("commit", {"label": "H", "kind": "assertion", "text": "假设X成立",
                          "deps": [], "branch": "bx"})
    log.append("commit", {"label": "D1", "kind": "assertion", "text": "推论1",
                          "deps": ["H"], "branch": "bx"})
    log.append("commit", {"label": "D2", "kind": "assertion", "text": "推论2",
                          "deps": ["D1"], "branch": "bx"})
    builder = Builder(log, store).replay()

    h = builder.labels["H"]
    assert store.status(h) == "refuted"
    pruned = [e for e in log.replay() if e["type"] == "branch_pruned"]
    assert pruned and pruned[0]["payload"]["refuted"] == h
    assert builder.labels["D1"] in pruned[0]["payload"]["dead_cone"]
    assert pruned[0]["payload"]["branches"] == ["bx"]
    assert store.dead_branches() == ["bx"]
    # taint guard: nothing downstream can ever verify
    assert store.status(builder.labels["D1"]) == "unverified"


def test_or_resolution_certificate_ignores_dead_branch(tmp_path):
    log, store, backend, kernel = world(
        tmp_path, lambda t: REFUTES if "死路" in t else HOLDS)
    log.append("commit", {"label": "H1", "kind": "assertion", "text": "死路假设",
                          "deps": [], "branch": "b1"})
    log.append("commit", {"label": "X", "kind": "assertion", "text": "b1的推论",
                          "deps": ["H1"], "branch": "b1"})
    log.append("commit", {"label": "H2", "kind": "assertion", "text": "活路假设",
                          "deps": [], "branch": "b2"})
    log.append("commit", {"label": "F", "kind": "final", "text": "结论走b2",
                          "deps": ["H2"]})
    builder = Builder(log, store).replay()

    cert = request_certificate(store, kernel, log,
                               ArtifactStore(str(tmp_path / "a2")),
                               builder.labels["F"])
    assert cert["verdict"] == "verified"
    texts = {e["text"] for e in cert["slice"]}
    assert "死路假设" not in texts       # backward slice never touches b1


def test_sidecar_branches_tool(tmp_path):
    def factory(sc):
        backend = FixtureBackend({
            "compile_obligations": {"obligations": [
                {"text": "O1", "failure_looks_like": "反例"}]},
            "write_check": lambda v: {
                "check_source": REFUTES if "b1" in v["claim_text"] else HOLDS},
        })
        return Kernel(sc.store, sc.log, backend, sc.artifacts)

    sc = Sidecar(str(tmp_path), kernel_factory=factory)
    sc.commit("A1", "assertion", "b1 的假设", [], branch="b1")
    sc.commit("C1", "assertion", "b1 推论", ["A1"], branch="b1")
    sc.commit("A2", "assertion", "b2 的假设", [], branch="b2")
    sc.commit("C2", "assertion", "b2 推论", ["A2"], branch="b2")

    out = sc.branches()
    assert set(out["branches"]) == {"b1", "b2"}
    assert out["dead"] == ["b1"]
