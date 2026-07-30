"""Scoring logic of the false-pass experiment (pure functions only)."""
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "false_pass",
    os.path.join(os.path.dirname(__file__), "..", "experiments", "false_pass.py"),
)
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


def test_benchmarks_are_balanced_pairs():
    for entries in fp.BENCHMARK_SETS.values():
        assert len(entries) % 2 == 0
        trues = sum(1 for _, t, _c in entries if t)
        assert trues == len(entries) // 2


def test_all_labels_pass_the_mechanical_gate():
    """Every claim's labelled truth equals its computed truth — the exam
    itself is validated before it grades anything."""
    for name, entries in fp.BENCHMARK_SETS.items():
        fp.validate_labels(entries)


def test_label_gate_catches_a_wrong_label():
    import pytest
    with pytest.raises(AssertionError, match="label wrong"):
        fp.validate_labels([("假标签", True, "1 == 2")])


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
