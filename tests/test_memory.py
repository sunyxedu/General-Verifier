"""Memory three stores: dedup, persistence, mechanical retrieval, wiring."""
import json

from verifier.builder import Builder
from verifier.discharge import DischargeResult
from verifier.events import EventLog
from verifier.game import BindingGame
from verifier.instruments import ArtifactStore
from verifier.kernel import Kernel
from verifier.memory import Memory, _similarity
from verifier.model import Obligation
from verifier.roles import FixtureBackend
from verifier.store import ClaimStore


def test_record_dedup_and_reopen(tmp_path):
    m = Memory(str(tmp_path))
    assert m.record_obligations("整除 claim", ["逐个检查"], "verified")
    assert not m.record_obligations("整除 claim", ["逐个检查"], "verified")
    assert m.record_instrument("python", "execute", "跑一遍")

    m2 = Memory(str(tmp_path))                       # reopen from disk
    assert not m2.record_obligations("整除 claim", ["逐个检查"], "verified")
    assert "python" in m2.registry_summary()


def test_retrieval_ranks_by_similarity(tmp_path):
    m = Memory(str(tmp_path))
    m.record_obligations("对所有 n≤100，6 整除 n^3-n", ["整除逐检"], "verified")
    m.record_obligations("2^13-1 是素数", ["素性检查"], "verified")

    hits = json.loads(m.hits_for_obligations("对所有 n≤500，30 整除 n^5-n", k=1))
    assert len(hits) == 1 and "整除" in hits[0]["claim"]
    assert _similarity("整除性质", "整除性质") == 1.0


def test_no_hits_for_unrelated_query(tmp_path):
    m = Memory(str(tmp_path))
    m.record_attack("整除 claim", "变体", "被抓", "false_pass_caught")
    assert json.loads(m.hits_for_attacks("qwerty", k=3)) == []


def test_kernel_records_and_reuses_patterns(tmp_path):
    memory = Memory(str(tmp_path / "mem"))
    log = EventLog(str(tmp_path / "e.jsonl"))
    store = ClaimStore()
    backend = FixtureBackend({
        "compile_obligations": {"obligations": [
            {"text": "范围整除检查", "failure_looks_like": "反例"}]},
        "write_check": {"check_source":
                        'import json\nprint(json.dumps({"holds": True}))'},
    })
    kernel = Kernel(store, log, backend, ArtifactStore(str(tmp_path / "a")),
                    memory=memory)
    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": "对所有 n≤100，6 整除 n^3-n", "deps": []})
    log.append("commit", {"label": "F", "kind": "final",
                          "text": "结论", "deps": ["A"]})
    Builder(log, store).replay()

    stored = json.loads(memory.hits_for_obligations("整除"))
    assert stored and stored[0]["obligations"] == ["范围整除检查"]
    assert any(e["type"] == "memory_recorded" for e in log.replay())

    # a later similar claim compiles WITH the retrieved pattern in its prompt
    log2 = EventLog(str(tmp_path / "e2.jsonl"))
    store2 = ClaimStore()
    kernel2 = Kernel(store2, log2, backend, ArtifactStore(str(tmp_path / "a2")),
                     memory=memory)
    log2.append("commit", {"label": "B", "kind": "assertion",
                           "text": "对所有 n≤300，30 整除 n^5-n", "deps": []})
    log2.append("commit", {"label": "F", "kind": "final",
                           "text": "结论", "deps": ["B"]})
    Builder(log2, store2).replay()
    compile_calls = [c for c in backend.calls if c["role"] == "compile_obligations"]
    assert "范围整除检查" in compile_calls[-1]["variables"]["memory_hits"]


def test_refutation_recorded_as_attack(tmp_path):
    memory = Memory(str(tmp_path / "mem"))
    log = EventLog(str(tmp_path / "e.jsonl"))
    store = ClaimStore()
    backend = FixtureBackend({
        "compile_obligations": {"obligations": [
            {"text": "O1", "failure_looks_like": "反例"}]},
        "write_check": {"check_source":
                        'import json\nprint(json.dumps({"holds": False, "witness": {"n": 2}}))'},
    })
    Kernel(store, log, backend, ArtifactStore(str(tmp_path / "a")),
           memory=memory)
    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": "假 claim", "deps": []})
    log.append("commit", {"label": "F", "kind": "final",
                          "text": "结论", "deps": ["A"]})
    Builder(log, store).replay()

    attacks = json.loads(memory.hits_for_attacks("假 claim"))
    assert attacks and attacks[0]["kind"] == "refutation"
    assert "n" in attacks[0]["evidence"]


def test_game_feeds_registry_and_records_bind_and_escape(tmp_path):
    memory = Memory(str(tmp_path / "mem"))
    memory.record_instrument("sympy", "execute", "符号复算")

    log = EventLog(str(tmp_path / "e.jsonl"))
    backend = FixtureBackend({
        "oracles": {"methods": [
            {"procedure": "跑数值", "instrument_id": "weak",
             "evidence_kind": "text", "artifact": "a", "failure_looks_like": "f"},
            {"procedure": "逐个执行", "instrument_id": "strong",
             "evidence_kind": "execute", "artifact": "a", "failure_looks_like": "f"},
        ]},
        "variants": {"variants": [{"text": "V1", "disproof": "确实假"}]},
        "entailment": {"entails": "yes", "grade": "demonstrated", "reason": "ok"},
    })

    class D:
        def discharge(self, row, obligation, method=None, tag=""):
            if row["text"].startswith("V"):
                # strong false-passes -> escape recorded; weak refutes
                outcome = "holds" if method["instrument_id"] == "strong" else "refuted"
                return DischargeResult(outcome, None, "d" * 64, "s", "corroborated")
            return DischargeResult("holds", None, "d" * 64, "s", "corroborated")

    game = BindingGame(backend, ArtifactStore(str(tmp_path / "a")), log, D(),
                       n_variants=1, memory=memory)
    row = {"id": "c1", "kind": "assertion", "text": "某 claim"}
    result = game.run(row, Obligation("c1", "O1", "反例"))

    oracle_call = next(c for c in backend.calls if c["role"] == "oracles")
    assert "sympy" in oracle_call["variables"]["registry"]   # warm registry fed in
    assert result.instrument == "weak"
    attacks = json.loads(memory.hits_for_attacks("某 claim"))
    assert attacks[0]["kind"] == "false_pass_caught"          # escape remembered
    registry = memory.registry_summary()
    assert "weak" in registry                                 # bound instrument saved
