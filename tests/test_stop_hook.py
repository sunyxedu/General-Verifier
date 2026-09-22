"""The Stop hook: a certificate below the goal blocks the solver's stop."""
import importlib.util
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "gv_stop_hook", os.path.join(HERE, "hooks", "stop.py"))
stop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stop)


def _ev(seq, type_, payload):
    return {"seq": seq, "type": type_, "payload": payload}


def test_no_final_claim_blocks_with_the_protocol_step():
    reason = stop.assess([_ev(1, "commit", {"kind": "assertion", "label": "A1"})],
                         "partial")
    assert reason and "no final claim" in reason


def test_final_without_certificate_blocks():
    reason = stop.assess([_ev(1, "commit", {"kind": "final", "label": "F"})],
                         "partial")
    assert reason and "no certificate was requested" in reason


def test_certificate_at_or_above_goal_allows_the_stop():
    events = [_ev(1, "commit", {"kind": "final", "label": "F"}),
              _ev(2, "certificate", {"final": "c1", "verdict": "partial"})]
    assert stop.assess(events, "partial") is None
    events[-1]["payload"]["verdict"] = "verified"
    assert stop.assess(events, "verified") is None


def test_certificate_below_goal_blocks_with_the_verifier_findings():
    events = [
        _ev(1, "commit", {"kind": "final", "label": "F"}),
        _ev(2, "obligations_compiled", {"claim": "c1", "obligations": [
            {"id": "o1", "text": "The output file lists every winning move."},
            {"id": "o2", "text": "The file exists."}]}),
        _ev(3, "discharge", {"claim": "c1", "obligation": "o2", "outcome": "holds"}),
        _ev(4, "judger_verdict", {"obligation": "o1", "outcome": "refuted",
                                  "witness": "g2g4 is winning and is absent"}),
        _ev(5, "discharge", {"claim": "c1", "obligation": "o1", "outcome": "refuted",
                             "witness": "g2g4 is winning and is absent"}),
        _ev(6, "certificate", {"final": "c1", "verdict": "refuted"}),
    ]
    reason = stop.assess(events, "partial")
    assert reason and "`refuted`" in reason
    assert "[refuted] The output file lists every winning move" in reason
    assert "g2g4" in reason and "The file exists" not in reason
    # a stricter goal blocks a partial certificate too
    events[-1]["payload"]["verdict"] = "partial"
    assert stop.assess(events, "verified")
