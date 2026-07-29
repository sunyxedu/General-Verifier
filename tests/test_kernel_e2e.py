"""Kernel + builder + certificate, wired end to end over an offline trace.

Scenario (M0 acceptance, report §16): replay a recorded solver trace, the
single trigger rule drives verification, a certificate request settles the
residue, and the frozen certificate carries the whole evidence chain.
"""
import json

import pytest

from verifier.builder import Builder, TraceError
from verifier.certificate import request_certificate
from verifier.events import EventLog
from verifier.instruments import ArtifactStore
from verifier.kernel import Kernel
from verifier.model import UNVERIFIED
from verifier.roles import FixtureBackend
from verifier.store import ClaimStore


CLAIM_6 = "对所有 1≤n≤100，6 整除 n^3-n"
CLAIM_4 = "对所有 1≤n≤100，4 整除 n^3-n"   # false: n=2 -> 6, 6 % 4 != 0
FINAL_TEXT = "结论：范围内的整除性质成立"

CHECK_TEMPLATE = (
    "import json\n"
    "bad = next((n for n in range(1, 101) if (n**3 - n) % {d} != 0), None)\n"
    'print(json.dumps({{"holds": bad is None, "witness": {{"n": bad}}}}))\n'
)
TRIVIAL_HOLDS = 'import json\nprint(json.dumps({"holds": True}))\n'


def obligations_fixture(variables):
    text = variables["claim_text"]
    if "整除" in text:
        return {
            "obligations": [
                {"text": "范围内每个 n 通过整除检查",
                 "failure_looks_like": "一个具体 n 与非零余数的记录"},
                {"text": "论证优雅", "failure_looks_like": ""},   # vacuous -> filtered
            ]
        }
    return {
        "obligations": [
            {"text": "结论与其依赖一致",
             "failure_looks_like": "依赖中存在与结论矛盾的记录"},
        ]
    }


def check_fixture(variables):
    text = variables["claim_text"]
    if "6 整除" in text:
        return {"check_source": CHECK_TEMPLATE.format(d=6)}
    if "4 整除" in text:
        return {"check_source": CHECK_TEMPLATE.format(d=4)}
    return {"check_source": TRIVIAL_HOLDS}


def build_world(tmp_path, claim_text):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("session_start", {})
    log.append("tool_call", {"tool": "python", "cmd": "env probe"})
    log.append("tool_result", {"exit": 0, "stdout": "3.9"})
    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": claim_text, "deps": ["effect:3"]})
    log.append("commit", {"label": "F", "kind": "final",
                          "text": FINAL_TEXT, "deps": ["A"]})
    log.append("stop", {})

    store = ClaimStore()
    artifacts = ArtifactStore(str(tmp_path / "artifacts"))
    backend = FixtureBackend(
        {"compile_obligations": obligations_fixture, "write_check": check_fixture}
    )
    kernel = Kernel(store, log, backend, artifacts)
    builder = Builder(log, store).replay()
    return log, store, artifacts, kernel, builder


def test_replay_triggers_verify_lazily(tmp_path):
    log, store, artifacts, kernel, builder = build_world(tmp_path, CLAIM_6)
    effect, a, f = builder.labels["effect:3"], builder.labels["A"], builder.labels["F"]
    # 触发时序：A 入图 -> 效应被首次依赖 -> 查账验证；F 入图 -> A 被首次依赖 -> 内核验证
    assert store.status(effect) == "verified" and store.grade(effect) == "demonstrated"
    assert store.status(a) == "verified" and store.grade(a) == "demonstrated"
    # F 从未被依赖 -> 一分钱不花
    assert store.status(f) == UNVERIFIED


def test_certificate_settles_residue_and_freezes(tmp_path):
    log, store, artifacts, kernel, builder = build_world(tmp_path, CLAIM_6)
    f = builder.labels["F"]
    cert = request_certificate(store, kernel, log, artifacts, f)

    assert cert["verdict"] == "verified"
    assert cert["overall_grade"] == "demonstrated"
    assert len(cert["slice"]) == 3                      # F, A, effect
    # 证据链完整：断言的 artifact 可从内容寻址库取回，含检查程序哈希
    a_entry = next(e for e in cert["slice"] if e["text"] == CLAIM_6)
    artifact = artifacts.get(a_entry["evidence"][0]["artifact"])
    assert artifact["exit_code"] == 0 and "check_src_sha256" in artifact
    # 义务过滤留痕：空洞义务被丢弃且记录在事件流里
    compiled = [e for e in log.replay() if e["type"] == "obligations_compiled"]
    assert any("论证优雅" in e["payload"]["dropped"] for e in compiled)
    # 内核活动与出证全部进了同一条事件流
    types = {e["type"] for e in log.replay()}
    assert {"discharge", "status_change", "certificate"} <= types


def test_false_claim_yields_refuted_certificate_with_witness(tmp_path):
    log, store, artifacts, kernel, builder = build_world(tmp_path, CLAIM_4)
    a, f = builder.labels["A"], builder.labels["F"]
    assert store.status(a) == "refuted"
    cert = request_certificate(store, kernel, log, artifacts, f)
    assert cert["verdict"] == "refuted"
    a_entry = next(e for e in cert["slice"] if e["text"] == CLAIM_4)
    assert a_entry["evidence"][0]["witness"] == {"n": 2}   # 反例入证
    # taint 守卫：final 不得建立在 refuted 之上，保持 unverified 并留痕
    f_entry = next(e for e in cert["slice"] if e["id"] == f)
    assert f_entry["status"] == UNVERIFIED
    assert any(
        ev.get("error") == "depends on refuted claim"
        for ev in f_entry["evidence"]
    )


def test_fail_closed_check_leaves_claim_unverified(tmp_path):
    log = EventLog(str(tmp_path / "e.jsonl"))
    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": "某个 claim", "deps": []})
    log.append("commit", {"label": "F", "kind": "final",
                          "text": "结论", "deps": ["A"]})
    store = ClaimStore()
    artifacts = ArtifactStore(str(tmp_path / "artifacts"))
    backend = FixtureBackend({
        "compile_obligations": {"obligations": [
            {"text": "O1", "failure_looks_like": "反例"}]},
        "write_check": {"check_source": "raise RuntimeError('boom')"},
    })
    kernel = Kernel(store, log, backend, artifacts)
    builder = Builder(log, store).replay()
    a = builder.labels["A"]
    assert store.status(a) == UNVERIFIED                # 崩溃不折算成任何结论
    cert = request_certificate(store, kernel, log, artifacts, builder.labels["F"])
    assert cert["verdict"] == "unverified"


def test_vacuous_only_obligations_fail_closed(tmp_path):
    log = EventLog(str(tmp_path / "e.jsonl"))
    store = ClaimStore()
    artifacts = ArtifactStore(str(tmp_path / "artifacts"))
    backend = FixtureBackend({
        "compile_obligations": {"obligations": [
            {"text": "优雅", "failure_looks_like": ""}]},
        "write_check": {"check_source": TRIVIAL_HOLDS},
    })
    kernel = Kernel(store, log, backend, artifacts)
    # 手动构图（跳过 trace 的 kind 校验路径）
    from verifier.model import Claim
    a = store.add_claim(Claim("assertion", "只有空洞义务的 claim"))
    store.add_claim(Claim("final", "F", deps=(a,)))     # 触发 verify(A)
    assert store.status(a) == UNVERIFIED
    reasons = [ev.get("error") for ev in store.evidence_for(a)]
    assert "no falsifiable obligations" in reasons


def test_builder_rejects_unknown_reference(tmp_path):
    log = EventLog(str(tmp_path / "e.jsonl"))
    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": "t", "deps": ["nope"]})
    with pytest.raises(TraceError, match="unknown claim reference"):
        Builder(log, ClaimStore()).replay()
