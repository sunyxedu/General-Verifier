"""Scoring logic of the false-pass experiment (pure functions only)."""
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "false_pass",
    os.path.join(os.path.dirname(__file__), "..", "experiments", "false_pass.py"),
)
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


def test_benchmark_is_balanced_pairs():
    assert len(fp.BENCHMARK) % 2 == 0
    trues = sum(1 for _, t in fp.BENCHMARK if t)
    assert trues == len(fp.BENCHMARK) // 2


def test_classify_judge():
    assert fp.classify(True, judge_verdict="true") == "ok"
    assert fp.classify(False, judge_verdict="true") == "false_pass"
    assert fp.classify(True, judge_verdict="false") == "false_alarm"
    assert fp.classify(False, judge_verdict="false") == "ok"


def test_classify_verifier_counts_unverified_as_abstain():
    assert fp.classify(False, verifier_status="verified") == "false_pass"
    assert fp.classify(True, verifier_status="verified") == "ok"
    assert fp.classify(True, verifier_status="refuted") == "false_alarm"
    assert fp.classify(False, verifier_status="refuted") == "ok"
    assert fp.classify(True, verifier_status="unverified") == "abstain"
    assert fp.classify(False, verifier_status="unverified") == "abstain"


def test_summarize():
    rows = [
        {"arm": "judge", "outcome": "false_pass"},
        {"arm": "judge", "outcome": "ok"},
        {"arm": "verifier", "outcome": "abstain"},
        {"arm": "verifier", "outcome": "ok"},
    ]
    s = fp.summarize(rows)
    assert s["judge"]["false_pass"] == 1 and s["judge"]["n"] == 2
    assert s["verifier"]["abstain"] == 1 and s["verifier"]["false_pass"] == 0
