"""Termination controller (component 27) + the hook scripts themselves."""
import json
import os
import subprocess
import sys

from verifier.builder import Builder
from verifier.events import EventLog
from verifier.instruments import ArtifactStore
from verifier.kernel import Kernel
from verifier.roles import FixtureBackend
from verifier.store import ClaimStore

HOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "hooks")


def test_obligation_cap_yields_unverified_bounds(tmp_path):
    """Beyond the cap the kernel refuses honestly — checking a subset and
    passing would dishonestly imply full verification."""
    many = {"obligations": [
        {"text": f"O{i}", "failure_looks_like": "反例"} for i in range(20)
    ]}
    log = EventLog(str(tmp_path / "e.jsonl"))
    store = ClaimStore()
    backend = FixtureBackend({
        "compile_obligations": many,
        "write_check": {"check_source":
                        'import json\nprint(json.dumps({"holds": True}))'},
    })
    kernel = Kernel(store, log, backend, ArtifactStore(str(tmp_path / "a")),
                    max_obligations=16)
    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": "病态 claim", "deps": []})
    log.append("commit", {"label": "F", "kind": "final",
                          "text": "结论", "deps": ["A"]})
    builder = Builder(log, store).replay()

    a = builder.labels["A"]
    assert store.status(a) == "unverified"
    assert any("bounds" in (ev.get("error") or "")
               for ev in store.evidence_for(a))
    changes = [e for e in log.replay() if e["type"] == "status_change"]
    assert changes[-1]["payload"]["reason"] == "bounds: obligations cap"
    # not a cent was spent on discharges
    assert not [e for e in log.replay() if e["type"] == "discharge"]


def test_cap_is_a_breadth_bound_not_a_verdict(tmp_path):
    """At the cap exactly, verification proceeds normally."""
    ok = {"obligations": [
        {"text": f"O{i}", "failure_looks_like": "反例"} for i in range(3)
    ]}
    log = EventLog(str(tmp_path / "e.jsonl"))
    store = ClaimStore()
    backend = FixtureBackend({
        "compile_obligations": ok,
        "write_check": {"check_source":
                        'import json\nprint(json.dumps({"holds": True}))'},
    })
    Kernel(store, log, backend, ArtifactStore(str(tmp_path / "a")),
           max_obligations=3)
    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": "正常 claim", "deps": []})
    log.append("commit", {"label": "F", "kind": "final",
                          "text": "结论", "deps": ["A"]})
    builder = Builder(log, store).replay()
    assert store.status(builder.labels["A"]) == "verified"


def _run_hook(script, payload, feed):
    return subprocess.run(
        [sys.executable, os.path.join(HOOKS_DIR, script)],
        input=json.dumps(payload), capture_output=True, text=True,
        env={**os.environ, "GV_HOOK_FEED": feed}, timeout=10,
    )


def test_hook_scripts_write_feed_lines(tmp_path):
    feed = str(tmp_path / "sub" / "feed.jsonl")   # dir must be auto-created

    proc = _run_hook("pre_tool_use.py", {
        "session_id": "s1", "tool_name": "Bash",
        "tool_input": {"command": "ls"}, "agent_id": "b1",
    }, feed)
    assert proc.returncode == 0, proc.stderr

    proc = _run_hook("post_tool_use.py", {
        "session_id": "s1", "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_response": {"stdout": "x" * 5000},
    }, feed)
    assert proc.returncode == 0, proc.stderr

    lines = [json.loads(line) for line in open(feed, encoding="utf-8")]
    assert [l["type"] for l in lines] == ["tool_call", "tool_result"]
    assert lines[0]["payload"]["tool"] == "Bash"
    assert lines[0]["payload"]["agent"] == "b1"
    assert len(lines[1]["payload"]["response_sha256"]) == 64
    assert len(lines[1]["payload"]["excerpt"]) <= 2000


def test_hook_feed_flows_into_sidecar(tmp_path):
    """Scripts -> raw feed -> sidecar ingestion -> referenceable effect."""
    from verifier.sidecar import Sidecar

    feed = os.path.join(str(tmp_path), "hook_feed.jsonl")
    _run_hook("pre_tool_use.py", {"tool_name": "python"}, feed)
    _run_hook("post_tool_use.py", {"tool_name": "python",
                                   "tool_response": {"out": 1}}, feed)

    def factory(sc):
        backend = FixtureBackend({
            "compile_obligations": {"obligations": [
                {"text": "O1", "failure_looks_like": "反例"}]},
            "write_check": {"check_source":
                            'import json\nprint(json.dumps({"holds": True}))'},
        })
        return Kernel(sc.store, sc.log, backend, sc.artifacts)

    sc = Sidecar(str(tmp_path), kernel_factory=factory)
    out = sc.commit("A", "assertion", "基于工具输出", ["effect:2"])
    assert "claim_id" in out, out
