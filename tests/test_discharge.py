"""Discharge strategies + agent-seated kernel, all offline via FixtureBackend.

FixtureBackend speaks the same complete() protocol as AgentBackend, so agent
roles (with their enum schemas) are exercised end to end without the SDK.
"""
import os

import pytest

from verifier.agents import COMPILE_OBLIGATIONS_AGENT, DISCHARGE_AGENT
from verifier.builder import Builder
from verifier.certificate import request_certificate
from verifier.discharge import AgentDischarger, DischargeResult
from verifier.events import EventLog
from verifier.instruments import ArtifactStore
from verifier.kernel import Kernel
from verifier.model import Obligation, OUTCOME_FAILED
from verifier.roles import FixtureBackend
from verifier.store import ClaimStore


ROW = {"id": "c1", "text": "某个 claim"}
OB = Obligation("c1", "范围内每个元素通过检查", "一个具体反例记录")


def make(tmp_path, fixture):
    backend = FixtureBackend({"discharge": fixture})
    artifacts = ArtifactStore(str(tmp_path / "artifacts"))
    d = AgentDischarger(backend, artifacts, workdir_base=str(tmp_path / "work"))
    return d, artifacts


def test_agent_discharge_snapshots_workdir_and_reports(tmp_path):
    def fixture(variables):
        # the "agent" leaves artifacts on disk, exactly as the real one would
        with open(os.path.join(variables["workdir"], "check.py"), "w") as f:
            f.write("print('ok')\n")
        return {"outcome": "holds", "witness": "", "artifact": "check.py"}

    d, artifacts = make(tmp_path, fixture)
    result = d.discharge(ROW, OB)
    assert result.outcome == "holds" and result.grade == "corroborated"
    stored = artifacts.get(result.artifact)
    assert stored["reported"]["outcome"] == "holds"
    assert stored["files"]["check.py"] == "print('ok')\n"


def test_agent_discharge_passes_witness_through(tmp_path):
    d, _ = make(tmp_path, {"outcome": "refuted", "witness": "n=2 余 2",
                           "artifact": ""})
    result = d.discharge(ROW, OB)
    assert result.outcome == "refuted" and result.witness == "n=2 余 2"


def test_agent_discharge_fails_closed_when_role_fails(tmp_path):
    d, artifacts = make(tmp_path, None)      # no fixture -> RoleError inside
    d.backend = FixtureBackend({})           # truly no fixture registered
    result = d.discharge(ROW, OB)
    assert result.outcome == OUTCOME_FAILED
    assert "no fixture" in artifacts.get(result.artifact)["error"]


def test_enum_gate_rejects_invented_outcome(tmp_path):
    d, _ = make(tmp_path, {"outcome": "passed", "witness": "", "artifact": ""})
    result = d.discharge(ROW, OB)            # "passed" is not in the enum
    assert result.outcome == OUTCOME_FAILED  # schema violation == failed, on record


def test_snapshot_skips_toolchain_dirs_and_caps_totals(tmp_path):
    from verifier.discharge import _snapshot

    w = tmp_path / "w"
    (w / ".lake" / "deep").mkdir(parents=True)
    (w / ".lake" / "deep" / "huge.olean").write_text("x" * 100000)
    (w / "check.py").write_text("print(1)")
    for i in range(250):
        (w / f"f{i:03d}.txt").write_text("y")

    snap = _snapshot(str(w))
    assert "check.py" in snap
    assert not any(".lake" in k for k in snap)          # vendored tree excluded
    assert "<snapshot>" in snap and "truncated" in snap["<snapshot>"]
    assert len(snap) <= 202                              # max_files + markers


def test_explicit_toolchain_dir_is_shared_and_survives(tmp_path):
    shared = str(tmp_path / "toolchains")
    seen = []

    def fixture(variables):
        seen.append(variables["toolchain_dir"])
        return {"outcome": "holds", "witness": "", "artifact": ""}

    backend = FixtureBackend({"discharge": fixture})
    artifacts = ArtifactStore(str(tmp_path / "a"))
    d1 = AgentDischarger(backend, artifacts, workdir_base=str(tmp_path / "w1"),
                         cleanup=True, toolchain_dir=shared)
    d2 = AgentDischarger(backend, artifacts, workdir_base=str(tmp_path / "w2"),
                         toolchain_dir=shared)
    d1.discharge(ROW, OB)
    d2.discharge(ROW, OB)
    assert seen == [shared, shared]           # both point at the same cache
    assert os.path.isdir(shared)              # survives d1's cleanup


def test_cleanup_flag_removes_workdir_after_snapshot(tmp_path):
    def fixture(variables):
        with open(os.path.join(variables["workdir"], "log.txt"), "w") as f:
            f.write("ran")
        assert os.path.isdir(variables["toolchain_dir"])  # shared dir offered
        return {"outcome": "holds", "witness": "", "artifact": "log.txt"}

    backend = FixtureBackend({"discharge": fixture})
    artifacts = ArtifactStore(str(tmp_path / "a"))
    d = AgentDischarger(backend, artifacts,
                        workdir_base=str(tmp_path / "w"), cleanup=True)
    result = d.discharge(ROW, OB)
    assert artifacts.get(result.artifact)["files"]["log.txt"] == "ran"
    assert not os.path.exists(os.path.join(str(tmp_path / "w"), OB.id[:12]))
    assert os.path.isdir(d.toolchain_dir)                # toolchains survive


def test_method_binding_reaches_prompt(tmp_path):
    seen = {}

    def fixture(variables):
        seen.update(variables)
        return {"outcome": "holds", "witness": "", "artifact": ""}

    d, _ = make(tmp_path, fixture)
    d.discharge(ROW, OB, method={"procedure": "逐个复算", "instrument_id": "sympy"})
    assert seen["procedure"] == "逐个复算" and seen["instrument_id"] == "sympy"
    assert os.path.isdir(seen["workdir"])


# ---- agent-seated kernel, offline e2e ----------------------------------------

CLAIM = "对所有 1≤n≤100，6 整除 n^3-n"
FINAL = "结论：范围内的整除性质成立"


def obligations_fixture(variables):
    if "整除" in variables["claim_text"]:
        return {"obligations": [
            {"text": "范围内每个 n 通过整除检查",
             "failure_looks_like": "一个具体 n 与非零余数的记录"},
        ]}
    return {"obligations": [
        {"text": "结论与其依赖一致",
         "failure_looks_like": "依赖中与结论矛盾的记录"},
    ]}


def test_kernel_runs_on_agent_roles_offline(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("tool_call", {"tool": "python", "cmd": "probe"})
    log.append("tool_result", {"exit": 0})
    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": CLAIM, "deps": ["effect:2"]})
    log.append("commit", {"label": "F", "kind": "final",
                          "text": FINAL, "deps": ["A"]})

    backend = FixtureBackend({
        "obligations": obligations_fixture,
        "discharge": {"outcome": "holds", "witness": "", "artifact": "run.log"},
    })
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
    assert store.status(a) == "verified"
    assert store.grade(a) == "corroborated"          # agent-reported, graded honestly

    cert = request_certificate(store, kernel, log, artifacts, builder.labels["F"])
    assert cert["verdict"] == "verified"
    assert cert["overall_grade"] == "corroborated"   # weakest link on the slice
    a_entry = next(e for e in cert["slice"] if e["text"] == CLAIM)
    assert a_entry["evidence"][0]["instrument"] == "agent:discharge"
    assert a_entry["evidence"][0]["grade"] == "corroborated"
