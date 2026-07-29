"""Live SDK round-trips — opt-in (GV_LIVE=1), spends real tokens.

Two smokes: a pure-text role (plumbing: query -> output_format ->
structured_output -> our validate), and a Bash-palette role (grounding: the
agent must actually execute something and report what it saw).
"""
import os

import pytest

from verifier.agents import AgentBackend, AgentRole

pytestmark = pytest.mark.skipif(
    os.environ.get("GV_LIVE") != "1", reason="live SDK test; set GV_LIVE=1"
)

MODEL = "claude-haiku-4-5"   # plumbing smoke only — real roles inherit stronger defaults


def test_live_pure_text_role(tmp_path):
    role = AgentRole(
        name="echo",
        template="计算 {a} * {b}，把结果数字作为字符串填入 answer。",
        schema={"answer": str},
        tools=[],
        model=MODEL,
        max_turns=4,
    )
    backend = AgentBackend(roles={"echo": role}, cwd=str(tmp_path))
    payload = role.call(backend, {"a": 12, "b": 9})
    assert payload["answer"].strip() == "108"
    t = backend.transcripts[-1]
    assert t["subtype"] == "success" and t["session_id"]


def test_live_agent_seated_kernel_end_to_end(tmp_path):
    """The real M0.5 acceptance: obligations compiled by an agent, obligations
    discharged by an agent that actually executes a check, certificate issued.
    """
    from verifier.agents import (
        AgentBackend, ALL_AGENT_ROLES, COMPILE_OBLIGATIONS_AGENT,
    )
    from verifier.builder import Builder
    from verifier.certificate import request_certificate
    from verifier.discharge import AgentDischarger
    from verifier.events import EventLog
    from verifier.instruments import ArtifactStore
    from verifier.kernel import Kernel
    from verifier.store import ClaimStore

    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": "对所有 1≤n≤100，6 整除 n^3-n", "deps": []})
    log.append("commit", {"label": "F", "kind": "final",
                          "text": "结论：范围内的整除性质成立", "deps": ["A"]})

    backend = AgentBackend(
        roles=ALL_AGENT_ROLES, cwd=str(tmp_path), model="claude-sonnet-5"
    )
    store = ClaimStore()
    artifacts = ArtifactStore(str(tmp_path / "artifacts"))
    kernel = Kernel(
        store, log, backend, artifacts,
        obligations_role=COMPILE_OBLIGATIONS_AGENT,
        discharger=AgentDischarger(backend, artifacts,
                                   workdir_base=str(tmp_path / "work")),
    )
    builder = Builder(log, store).replay()

    a = builder.labels["A"]
    assert store.status(a) == "verified", store.evidence_for(a)
    assert store.grade(a) == "corroborated"

    cert = request_certificate(store, kernel, log, artifacts, builder.labels["F"])
    assert cert["verdict"] == "verified", cert
    assert cert["overall_grade"] == "corroborated"
    # the discharge agent left re-checkable artifacts on disk, snapshotted
    a_entry = next(e for e in cert["slice"] if "整除" in e["text"])
    stored = artifacts.get(a_entry["evidence"][0]["artifact"])
    assert stored["reported"]["outcome"] == "holds"


def test_live_bash_palette_role(tmp_path):
    role = AgentRole(
        name="prober",
        template=(
            "用 Bash 运行 `echo $((6*7))`，把它 stdout 里的数字作为字符串填入 "
            "answer；把你实际运行的命令原样填入 command。"
        ),
        schema={"answer": str, "command": str},
        tools=["Bash"],
        model=MODEL,
        max_turns=6,
    )
    backend = AgentBackend(roles={"prober": role}, cwd=str(tmp_path))
    payload = role.call(backend, {})
    assert payload["answer"].strip() == "42"
    assert "echo" in payload["command"]
