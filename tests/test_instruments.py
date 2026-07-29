import json

from verifier.events import EventLog
from verifier.instruments import ArtifactStore, LedgerInstrument, PythonSandbox


def test_sandbox_holds():
    r = PythonSandbox().run(
        'import json\n'
        'ok = all((n**3 - n) % 6 == 0 for n in range(1, 101))\n'
        'print(json.dumps({"holds": ok}))\n'
    )
    assert r.outcome == "holds"
    assert r.artifacts["exit_code"] == 0


def test_sandbox_refuted_with_witness():
    r = PythonSandbox().run(
        'import json\n'
        'bad = next((n for n in range(1, 101) if (n**3 - n) % 4 != 0), None)\n'
        'print(json.dumps({"holds": bad is None, "witness": {"n": bad}}))\n'
    )
    assert r.outcome == "refuted"
    assert r.witness == {"n": 2}


def test_sandbox_fail_closed_on_crash():
    r = PythonSandbox().run("raise RuntimeError('boom')")
    assert r.outcome == "failed" and r.artifacts["error"] == "nonzero exit"


def test_sandbox_fail_closed_on_garbage_output():
    r = PythonSandbox().run("print('looks fine, trust me')")
    assert r.outcome == "failed" and r.artifacts["error"] == "unparseable verdict"


def test_sandbox_fail_closed_on_timeout():
    r = PythonSandbox(timeout=0.5).run("while True: pass")
    assert r.outcome == "failed" and r.artifacts["error"] == "timeout"


def test_ledger_holds_and_refutes(tmp_path):
    log = EventLog(str(tmp_path / "e.jsonl"))
    log.append("tool_call", {"tool": "python", "cmd": "sum"})
    e = log.append("tool_result", {"exit": 0, "stdout": "118.8"})
    ledger = LedgerInstrument(log)

    good = json.dumps(
        {"event_seq": 2, "tool": "python", "digest": EventLog.event_digest(e)},
        sort_keys=True,
    )
    assert ledger.check(good).outcome == "holds"

    fabricated = json.dumps(
        {"event_seq": 2, "tool": "python", "digest": "f" * 64}, sort_keys=True
    )
    r = ledger.check(fabricated)
    assert r.outcome == "refuted" and "mismatch" in r.witness["reason"]

    missing = json.dumps({"event_seq": 99, "digest": "a" * 64}, sort_keys=True)
    assert ledger.check(missing).outcome == "refuted"

    assert ledger.check("not json at all").outcome == "failed"


def test_artifact_store_roundtrip(tmp_path):
    store = ArtifactStore(str(tmp_path / "artifacts"))
    digest = store.put({"stdout": "ok", "exit_code": 0})
    assert store.get(digest)["stdout"] == "ok"
    assert store.put({"stdout": "ok", "exit_code": 0}) == digest  # 内容寻址幂等
