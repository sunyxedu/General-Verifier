"""Sidecar verbs, hook-feed ingestion, restart persistence, CLI gate."""
import json
import os

from verifier.cli import certify, replay
from verifier.instruments import ArtifactStore
from verifier.kernel import Kernel
from verifier.roles import FixtureBackend
from verifier.sidecar import Sidecar, build_server

FIXTURES = {
    "compile_obligations": {"obligations": [
        {"text": "O1", "failure_looks_like": "反例"}]},
    "write_check": {"check_source":
                    'import json\nprint(json.dumps({"holds": True}))'},
}
REFUTING = {
    "compile_obligations": {"obligations": [
        {"text": "O1", "failure_looks_like": "反例"}]},
    "write_check": {"check_source":
                    'import json\nprint(json.dumps({"holds": False, "witness": 7}))'},
}


def fixture_factory(fixtures=FIXTURES):
    def factory(sc):
        return Kernel(sc.store, sc.log, FixtureBackend(fixtures), sc.artifacts)
    return factory


def test_commit_verify_certificate_roundtrip(tmp_path):
    sc = Sidecar(str(tmp_path), kernel_factory=fixture_factory())
    out = sc.commit("A", "assertion", "某 claim", [])
    assert out["status"] == "unverified"          # no dependent yet

    out = sc.commit("F", "final", "结论", ["A"])
    assert sc.status("A")["status"] == "verified"  # trigger fired inline

    cert = sc.certificate("F")
    assert cert["verdict"] == "verified"
    assert len(cert["slice"]) == 2
    assert sc.status()["F"] == "verified"


def test_unknown_dep_and_unknown_label_are_errors_not_crashes(tmp_path):
    sc = Sidecar(str(tmp_path), kernel_factory=fixture_factory())
    assert "error" in sc.commit("A", "assertion", "t", ["nope"])
    assert "error" in sc.verify("missing")
    assert "error" in sc.certificate("missing")


def test_hook_feed_ingested_and_referenceable(tmp_path):
    sc = Sidecar(str(tmp_path), kernel_factory=fixture_factory())
    feed = os.path.join(str(tmp_path), "hook_feed.jsonl")
    with open(feed, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "tool_call",
                            "payload": {"tool": "Bash"}}) + "\n")
        f.write(json.dumps({"type": "tool_result",
                            "payload": {"tool": "Bash",
                                        "response_sha256": "x"}}) + "\n")
        f.write(json.dumps({"type": "garbage", "payload": {}}) + "\n")

    out = sc.commit("A", "assertion", "基于探测", ["effect:2"])
    assert "claim_id" in out                     # effect:2 existed at commit time
    cert_labels = sc.status()
    assert "effect:2" in cert_labels

    sc.commit("F", "final", "结论", ["A"])
    assert sc.status("A")["status"] == "verified"


def test_restart_rebuilds_view_from_own_ledger(tmp_path):
    sc = Sidecar(str(tmp_path), kernel_factory=fixture_factory())
    sc.commit("A", "assertion", "t", [])
    sc.commit("F", "final", "结论", ["A"])

    sc2 = Sidecar(str(tmp_path), kernel_factory=fixture_factory())
    assert sc2.status("A")["status"] == "verified"   # persisted, not re-verified
    assert set(sc2.builder.labels) == {"A", "F"}


def test_build_server_smoke(tmp_path):
    sc = Sidecar(str(tmp_path), kernel_factory=fixture_factory())
    server = build_server(sc)
    assert server.name == "gv"


def test_cli_certify_gate_exit_codes(tmp_path, capsys):
    from verifier.events import EventLog

    events = str(tmp_path / "trace.jsonl")
    log = EventLog(events)
    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": "t", "deps": []})
    log.append("commit", {"label": "F", "kind": "final",
                          "text": "结论", "deps": ["A"]})

    code = certify(events, str(tmp_path / "s1"),
                   kernel_factory=fixture_factory())
    assert code == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "verified"

    code = certify(events, str(tmp_path / "s2"),
                   kernel_factory=fixture_factory(REFUTING))
    assert code == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "refuted"


def test_cli_replay_is_llm_free(tmp_path, capsys):
    from verifier.events import EventLog

    events = str(tmp_path / "trace.jsonl")
    log = EventLog(events)
    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": "t", "deps": []})
    assert replay(events) == 0
    assert json.loads(capsys.readouterr().out) == {"A": "unverified"}
