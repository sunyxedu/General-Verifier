"""The Stop hook: a certificate below the goal blocks the solver's stop."""
import importlib.util
import os

from verifier.core.model import Claim

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "gv_stop_hook", os.path.join(HERE, "hooks", "stop.py"))
stop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stop)

# commit events carry what gv.commit records; the claim id is derived
FINAL = {"kind": "final", "label": "F", "text": "The task is complete.",
         "deps": []}
C1 = Claim("final", FINAL["text"]).id


def _ev(seq, type_, payload):
    return {"seq": seq, "type": type_, "payload": payload}


def test_no_final_claim_blocks_with_the_protocol_step():
    reason = stop.assess([_ev(1, "commit", {"kind": "assertion", "label": "A1",
                                            "text": "x", "deps": []})],
                         "partial")
    assert reason and "no final claim" in reason


def test_final_without_certificate_blocks():
    reason = stop.assess([_ev(1, "commit", FINAL)], "partial")
    assert reason and "no certificate was requested" in reason


def test_requested_certificate_without_a_verdict_is_named_as_such():
    events = [_ev(1, "commit", FINAL),
              _ev(2, "verify_request", {"target": C1, "cause": "certificate"})]
    reason = stop.assess(events, "partial")
    assert reason and "still being checked" in reason
    assert "no certificate was requested" not in reason
    # a request that predates the latest final does not count for it
    events = [_ev(1, "commit", FINAL),
              _ev(2, "verify_request", {"target": C1, "cause": "certificate"}),
              _ev(3, "commit", dict(FINAL, label="F2", text="Done, again."))]
    assert "no certificate was requested" in stop.assess(events, "partial")


def test_certificate_at_or_above_goal_allows_the_stop():
    events = [_ev(1, "commit", FINAL),
              _ev(2, "certificate", {"final": C1, "verdict": "partial"})]
    assert stop.assess(events, "partial") is None
    events[-1]["payload"]["verdict"] = "verified"
    assert stop.assess(events, "verified") is None


def test_certificate_below_goal_blocks_with_the_verifier_findings():
    events = [
        _ev(1, "commit", FINAL),
        _ev(2, "obligations_compiled", {"claim": C1, "obligations": [
            {"id": "o1", "text": "The output file lists every winning move."},
            {"id": "o2", "text": "The file exists."}]}),
        _ev(3, "discharge", {"claim": C1, "obligation": "o2", "outcome": "holds"}),
        _ev(4, "judger_verdict", {"obligation": "o1", "outcome": "refuted",
                                  "witness": "g2g4 is winning and is absent"}),
        _ev(5, "discharge", {"claim": C1, "obligation": "o1", "outcome": "refuted",
                             "witness": "g2g4 is winning and is absent"}),
        _ev(6, "certificate", {"final": C1, "verdict": "refuted"}),
    ]
    reason = stop.assess(events, "partial")
    assert reason and "`refuted`" in reason
    assert "[refuted] The output file lists every winning move" in reason
    assert "g2g4" in reason and "The file exists" not in reason
    # a stricter goal blocks a partial certificate too
    events[-1]["payload"]["verdict"] = "partial"
    assert stop.assess(events, "verified")
