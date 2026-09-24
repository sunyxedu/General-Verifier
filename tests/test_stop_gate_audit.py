"""The stop hook judges the LATEST final by its own certificate, tells a
running check from a failed verifier from a void or refuted delivery, and
never lets a stop through without a trace in the ledger."""
import importlib.util
import io
import json
import os
import threading
import time

import pytest

from verifier.core.events import EventLog
from verifier.core.model import Claim
from verifier.engine.certificate import infrastructure_failure
from verifier.llm.schema import RoleError, SeatTimeout

from test_certificate_scope import _fixtures, _service
from verifier.llm.schema import FixtureBackend

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "gv_stop_hook_audit", os.path.join(HERE, "hooks", "stop.py"))
stop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stop)


def _final(label, text, deps=(), revises=None):
    payload = {"kind": "final", "label": label, "text": text,
               "deps": list(deps)}
    if revises:
        payload["revises"] = revises
    return payload


def _id(text):
    return Claim("final", text).id


def _ledger(tmp_path, events):
    """A real ledger (the EventLog writer's encoding) in a state dir."""
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    log = EventLog(str(state / "events.jsonl"))
    for etype, payload in events:
        log.append(etype, payload)
    return state, log


def _run_hook(state, monkeypatch, goal="partial"):
    monkeypatch.setenv("GV_STATE_DIR", str(state))
    monkeypatch.setenv("GV_GOAL", goal)
    monkeypatch.delenv("GV_GOAL_MAX_BLOCKS", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    stop.main()
    text = out.getvalue()
    return json.loads(text) if text else None


def _types(state):
    return [e["type"] for e in EventLog(str(state / "events.jsonl")).replay()]


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    monkeypatch.setattr(stop, "POLL_SECONDS", 0.0)
    monkeypatch.setattr(stop, "POLL_INTERVAL", 0.01)


# 1. the certificate must belong to the latest final

def test_a_refuted_certificate_of_an_older_final_does_not_block_the_new_one():
    events = [
        {"seq": 1, "type": "commit", "payload": _final("F1", "old")},
        {"seq": 2, "type": "certificate",
         "payload": {"final": _id("old"), "verdict": "refuted"}},
        {"seq": 3, "type": "commit", "payload": _final("F2", "new")},
        {"seq": 4, "type": "verify_request",
         "payload": {"target": _id("new"), "cause": "certificate"}},
        {"seq": 5, "type": "certificate",
         "payload": {"final": _id("new"), "verdict": "verified"}},
    ]
    assert stop.final_state(events, "partial")[0] == "met"
    assert stop.assess(events, "partial") is None


def test_an_old_good_certificate_does_not_pass_a_new_unchecked_final():
    events = [
        {"seq": 1, "type": "commit", "payload": _final("F1", "old")},
        {"seq": 2, "type": "certificate",
         "payload": {"final": _id("old"), "verdict": "verified"}},
        {"seq": 3, "type": "commit", "payload": _final("F2", "new")},
    ]
    assert stop.final_state(events, "partial")[0] == "unrequested"
    # nor does a later certificate that names another final
    events.append({"seq": 4, "type": "certificate",
                   "payload": {"final": _id("old"), "verdict": "verified"}})
    assert stop.final_state(events, "partial")[0] == "unrequested"


def test_a_final_commit_that_did_not_resolve_is_not_the_latest_final():
    events = [
        {"seq": 1, "type": "commit", "payload": _final("F1", "good")},
        {"seq": 2, "type": "certificate",
         "payload": {"final": _id("good"), "verdict": "verified"}},
        # cites a label that was never committed: the server refused it
        {"seq": 3, "type": "commit",
         "payload": _final("F2", "bad", deps=["nope"])},
    ]
    state, final, _ = stop.final_state(events, "partial")
    assert state == "met" and final["label"] == "F1"


def test_revisions_and_effect_references_resolve_to_the_servers_claim_id():
    events = [
        {"seq": 1, "type": "tool_call",
         "payload": {"tool": "bash", "tool_use_id": "t1"}},
        {"seq": 2, "type": "tool_result",
         "payload": {"tool": "bash", "tool_use_id": "t1", "excerpt": ""}},
        {"seq": 3, "type": "commit", "payload": _final("F1", "v1")},
        {"seq": 4, "type": "commit",
         "payload": _final("F2", "v1", deps=["effect:1"], revises="F1")},
    ]
    final = stop.latest_final(events)
    assert final["label"] == "F2" and final["id"] != _id("v1")


def test_without_the_verifier_package_the_hook_matches_by_ledger_order(
        monkeypatch):
    monkeypatch.setattr(stop, "_verifier_import", lambda *a: None)
    events = [
        {"seq": 1, "type": "commit", "payload": _final("F1", "good")},
        {"seq": 2, "type": "certificate",
         "payload": {"final": "x", "verdict": "refuted"}},
        {"seq": 3, "type": "commit",
         "payload": _final("F2", "bad", deps=["nope"])},
        {"seq": 4, "type": "commit", "payload": _final("F3", "new")},
    ]
    state, final, _ = stop.final_state(events, "partial")
    assert final == {"seq": 4, "label": "F3", "id": None}
    assert state == "unrequested"


def test_the_hook_reads_the_claim_id_the_service_certified(
        tmp_path, monkeypatch):
    service = _service(tmp_path, FixtureBackend(_fixtures()), monkeypatch)
    service.certificate("f")
    events = list(service.log.replay())
    state, final, decided = stop.final_state(events, "partial")
    assert final["id"] == service.builder.finals[-1]
    assert decided["payload"]["final"] == final["id"]
    assert state == "met"


# 2. a certificate request that raises leaves its end on the ledger

def test_a_raising_certificate_request_is_logged_and_still_raises(
        tmp_path, monkeypatch):
    service = _service(tmp_path, FixtureBackend(_fixtures()), monkeypatch)

    def broken(claim_id, *args, **kwargs):
        raise RoleError("direct call failed: HTTP 503: overloaded")
    monkeypatch.setattr(service.kernel, "verify", broken)
    reply = service.certificate("f")
    assert "HTTP 503" in reply["error"]
    events = list(service.log.replay())
    failed = [e for e in events if e["type"] == "certificate_failed"]
    assert len(failed) == 1
    assert failed[0]["payload"]["final"] == service.builder.finals[-1]
    assert failed[0]["payload"]["infrastructure"] is True
    state, _, _ = stop.final_state(events, "partial")
    assert state == "failed"
    reason = stop.assess(events, "partial")
    assert "not a defect" in reason and "do not rewrite the final" in reason


def test_infrastructure_is_told_apart_from_a_verifier_bug():
    assert infrastructure_failure(SeatTimeout("seat ran out of time"))
    assert infrastructure_failure(TimeoutError())
    assert infrastructure_failure(RuntimeError("HTTP 429: rate limited"))
    try:
        try:
            raise ConnectionResetError("peer")
        except ConnectionResetError as inner:
            raise ValueError("wrapped") from inner
    except ValueError as outer:
        assert infrastructure_failure(outer)
    assert not infrastructure_failure(ValueError("illegal transition"))
    assert not infrastructure_failure(KeyError("obligation"))


# 3 and 4. a running check is waited for, not charged

def _requested(text="done"):
    return [("commit", _final("F", text)),
            ("verify_request", {"target": _id(text), "cause": "certificate"})]


def test_a_gv_verify_on_the_final_is_not_a_certificate_request():
    events = [
        {"seq": 1, "type": "commit", "payload": _final("F", "done")},
        {"seq": 2, "type": "verify_request",
         "payload": {"target": _id("done"), "cause": "mcp"}},
        {"seq": 3, "type": "verify_request",
         "payload": {"target": "some-other-claim", "cause": "certificate"}},
    ]
    assert stop.final_state(events, "partial")[0] == "unrequested"


def test_a_verdict_arriving_during_the_poll_is_used(tmp_path, monkeypatch):
    state, log = _ledger(tmp_path, _requested())
    monkeypatch.setattr(stop, "POLL_SECONDS", 10.0)

    def finish():
        time.sleep(0.2)
        log.append("certificate", {"final": _id("done"), "verdict": "partial"})
    worker = threading.Thread(target=finish)
    worker.start()
    try:
        assert _run_hook(state, monkeypatch) is None
    finally:
        worker.join()


def test_a_running_check_waits_without_spending_blocks_and_waits_are_capped(
        tmp_path, monkeypatch):
    state, _ = _ledger(tmp_path, _requested())
    monkeypatch.setattr(stop, "MAX_WAITS", 2)
    for _ in range(2):
        out = _run_hook(state, monkeypatch)
        assert out["decision"] == "block"
        assert "still being checked" in out["reason"]
        assert "gv.status" in out["reason"]
    record = json.loads((state / stop.BLOCKS_FILE).read_text())
    assert record["blocks"] == [] and len(record["waits"]) == 2
    assert _run_hook(state, monkeypatch) is None
    assert _types(state).count("goal_released") == 1


def test_a_server_restart_during_the_check_is_a_verifier_failure(
        tmp_path, monkeypatch):
    state, _ = _ledger(tmp_path, _requested() + [("server_started", {})])
    out = _run_hook(state, monkeypatch)
    assert "restarted while the check was running" in out["reason"]
    assert "not a defect" in out["reason"]


def test_a_second_verifier_failure_releases_the_stop(tmp_path, monkeypatch):
    fail = ("certificate_failed", {"final": _id("done"), "error": "HTTP 502",
                                   "infrastructure": True})
    state, log = _ledger(tmp_path, _requested() + [fail])
    out = _run_hook(state, monkeypatch)
    assert "model provider or transport failed" in out["reason"]
    # the same failure, no retry: still owed the retry
    assert _run_hook(state, monkeypatch)["decision"] == "block"
    log.append(*_requested()[1])
    log.append(*fail)
    assert _run_hook(state, monkeypatch) is None
    assert "goal_released" in _types(state)


# 5. a void certificate is reported once, then released

def test_a_void_certificate_is_named_void_and_released_after_one_notice(
        tmp_path, monkeypatch):
    state, _ = _ledger(tmp_path, _requested() + [
        ("certificate", {"final": _id("done"), "verdict": "unverified",
                         "void": "the verifier was restarted with other "
                                 "settings or code (model)"})])
    out = _run_hook(state, monkeypatch)
    assert "void" in out["reason"]
    assert "Address these in the delivery" not in out["reason"]
    assert _run_hook(state, monkeypatch) is None
    assert "goal_released" in _types(state)


# 6. only terminal discharge outcomes are findings

def test_a_script_that_ran_is_not_a_failed_obligation():
    fid = _id("done")
    events = [
        {"seq": 1, "type": "commit", "payload": _final("F", "done")},
        {"seq": 2, "type": "obligations_compiled", "payload": {
            "claim": fid, "obligations": [
                {"id": "o1", "text": "The server answers on its port."},
                {"id": "o2", "text": "The log file is rotated."}]}},
        {"seq": 3, "type": "discharge",
         "payload": {"claim": fid, "obligation": "o1", "outcome": "ran"}},
        {"seq": 4, "type": "discharge",
         "payload": {"claim": fid, "obligation": "o1", "outcome": "holds"}},
        {"seq": 5, "type": "discharge",
         "payload": {"claim": fid, "obligation": "o2", "outcome": "ran"}},
        {"seq": 6, "type": "certificate",
         "payload": {"final": fid, "verdict": "unverified"}},
    ]
    reason = stop.assess(events, "partial")
    assert "[ran]" not in reason
    assert "The server answers" not in reason
    assert "[not decided] The log file is rotated." in reason


# 7. budgets per final, a global cap, atomic record, visible release

def _refuted(text):
    return [("commit", _final("F-" + text, text)),
            ("verify_request", {"target": _id(text), "cause": "certificate"}),
            ("certificate", {"final": _id(text), "verdict": "refuted"})]


def test_blocks_are_budgeted_per_final_and_the_release_is_recorded(
        tmp_path, monkeypatch):
    state, log = _ledger(tmp_path, _refuted("one"))
    for _ in range(3):
        assert _run_hook(state, monkeypatch)["decision"] == "block"
    assert _run_hook(state, monkeypatch) is None
    released = [e for e in log.replay() if e["type"] == "goal_released"]
    assert len(released) == 1 and "block limit" in released[0]["payload"]["reason"]
    assert not [n for n in os.listdir(state) if n.endswith(".tmp")]
    # a new final is new work: a fresh budget
    for etype, payload in _refuted("two"):
        log.append(etype, payload)
    assert _run_hook(state, monkeypatch)["decision"] == "block"


def test_the_run_wide_cap_bounds_a_solver_cycling_through_finals(
        tmp_path, monkeypatch):
    state, log = _ledger(tmp_path, [])
    blocked = 0
    for n in range(stop.GLOBAL_BLOCK_FACTOR * 3 + 2):
        for etype, payload in _refuted(f"v{n}"):
            log.append(etype, payload)
        blocked += _run_hook(state, monkeypatch) is not None
    assert blocked == stop.GLOBAL_BLOCK_FACTOR * 3


# 8. a broken hook leaves a trace and blocks at most once

def test_an_unexpected_hook_error_is_logged_and_blocks_once(
        tmp_path, monkeypatch):
    state, _ = _ledger(tmp_path, _refuted("one"))

    def broken(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(stop, "run", broken)
    out = _run_hook(state, monkeypatch)
    assert out["decision"] == "block" and "boom" in out["reason"]
    assert _run_hook(state, monkeypatch) is None
    assert _types(state).count("goal_hook_error") == 2


# 9. the protocol and the hook say the same thing

def test_the_protocol_tells_the_solver_to_wait_and_not_to_repair_for_the_verifier():
    with open(os.path.join(HERE, "protocol", "solver-protocol.md"),
              encoding="utf-8") as f:
        protocol = f.read()
    assert "poll `gv.status`" in protocol
    assert "not a\n  defect of your delivery" in protocol \
        or "not a defect of your delivery" in protocol
