"""The offline ORM path: feed ingestion, outcome-blind extraction with
structural retry, certify-based scoring, and best-of-N selection.

Fixture-backed and instrument-free, like the rest of the suite: the
extractor is a FixtureBackend role, the kernel seats are fixtures/stubs.
"""
from __future__ import annotations

import json
import os

import pytest

from verifier import offline, selection
from verifier.core.events import EventLog
from verifier.core.store import ClaimStore
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.llm.schema import FixtureBackend
from verifier.stream.builder import Builder

COMPILE = lambda v: {"obligations": [{"text": f"O of: {v['claim_text']}"}]}  # noqa: E731

FEED = [
    {"type": "tool_call", "payload": {
        "tool": "Bash", "tool_use_id": "t1", "input": {"command": "echo 42"}}},
    {"type": "tool_result", "payload": {
        "tool": "Bash", "tool_use_id": "t1",
        "response_sha256": "abc", "excerpt": "42"}},
    {"type": "not_a_hook_event", "payload": {"x": 1}},   # untrusted line: ignored
]

GOOD_COMMITS = {"commits": [
    {"label": "C1", "kind": "assertion",
     "text": "as of effect:2, echo printed 42", "deps": ["effect:2"],
     "revises": ""},
    {"label": "F", "kind": "final", "text": "the answer is 42",
     "deps": ["C1"], "revises": ""},
]}


class StubDischarger:
    def __init__(self, outcome="holds"):
        self.outcome = outcome
        self.seen = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.seen.append(obligation.text)
        return DischargeResult(outcome=self.outcome, witness=None,
                               artifact="", instrument="stub")


def state_dir(tmp_path, feed=FEED):
    state = str(tmp_path / "gv")
    os.makedirs(state, exist_ok=True)
    with open(os.path.join(state, "hook_feed.jsonl"), "w") as f:
        for line in feed:
            f.write(json.dumps(line) + "\n")
    with open(os.path.join(state, "task.md"), "w") as f:
        f.write("print the answer")
    with open(os.path.join(state, "final.txt"), "w") as f:
        f.write("FINAL ANSWER: 42")
    return state


def kernel_factory(sc):
    return Kernel(sc.store, sc.log, FixtureBackend({"obligations": COMPILE}),
                  sc.artifacts, critic_role=None, discharger=StubDischarger())


# ---- ingestion ---------------------------------------------------------------

def test_feed_lines_become_referenceable_effects(tmp_path):
    state = state_dir(tmp_path)
    log = EventLog(str(tmp_path / "events.jsonl"))

    n = offline.ingest_feed(os.path.join(state, "hook_feed.jsonl"), log)

    assert n == 2                                # the unknown line was dropped
    builder = Builder(log, ClaimStore())
    for event in log.replay():
        builder.feed(event)
    assert "effect:2" in builder.labels          # tool_result -> effect claim


# ---- extraction --------------------------------------------------------------

def test_extract_writes_a_replayable_graph(tmp_path):
    state = state_dir(tmp_path)
    backend = FixtureBackend({"extract": lambda v: GOOD_COMMITS})

    meta = offline.extract(state, backend=backend)

    assert meta == {"rounds": 1, "commits": 2}
    log = EventLog(os.path.join(state, "events.jsonl"))
    builder = Builder(log, ClaimStore())
    for event in log.replay():
        builder.feed(event)
    assert len(builder.finals) == 1
    assert builder.labels["F"] == builder.finals[0]
    # the extractor saw the trace, not the verdicts
    variables = backend.calls[0]["variables"]
    assert variables["task"] == "print the answer"
    assert "echo 42" in variables["events_index"]
    assert variables["feedback"] == ""


def test_structural_error_feeds_back_and_retries(tmp_path):
    state = state_dir(tmp_path)
    bad = {"commits": [{"label": "F", "kind": "final", "text": "done",
                        "deps": ["nope"], "revises": ""}]}
    responses = iter([bad, GOOD_COMMITS])
    backend = FixtureBackend({"extract": lambda v: next(responses)})

    meta = offline.extract(state, backend=backend)

    assert meta["rounds"] == 2
    feedback = backend.calls[1]["variables"]["feedback"]
    assert "'F'" in feedback and "nope" in feedback


def test_not_exactly_one_final_is_a_structural_error(tmp_path):
    state = state_dir(tmp_path)
    no_final = {"commits": [{"label": "C1", "kind": "assertion", "text": "x",
                             "deps": [], "revises": ""}]}
    backend = FixtureBackend({"extract": lambda v: no_final})

    with pytest.raises(offline.ExtractionError, match="exactly one final"):
        offline.extract(state, backend=backend, max_rounds=2)
    # every failed round rebuilt the candidate from scratch
    assert not os.path.exists(os.path.join(state, "events.jsonl"))


def test_extraction_failure_scores_unscoreable_and_ranks_last(tmp_path):
    state = state_dir(tmp_path)
    backend = FixtureBackend({})    # no extract fixture -> RoleError every round

    features = offline.score(state, backend=backend,
                             kernel_factory=kernel_factory, max_rounds=2)

    assert features["unscoreable"] is True
    assert offline.rank_key(features) < offline.rank_key(
        {"unscoreable": False, "verdict": "refuted",
         "slice_size": 1, "claims": {}, "obligations": {}})


# ---- scoring -----------------------------------------------------------------

def test_score_certifies_the_extracted_graph(tmp_path):
    state = state_dir(tmp_path)
    backend = FixtureBackend({"extract": lambda v: GOOD_COMMITS})

    features = offline.score(state, backend=backend,
                             kernel_factory=kernel_factory)

    assert features["verdict"] == "verified"
    assert features["slice_size"] == 3           # effect + C1 + F
    assert features["claims"]["verified"] == 3
    assert features["obligations"]["verified"] == features["obligations_total"]
    assert features["extract_rounds"] == 1
    # persisted for the experiment drivers
    on_disk = json.load(open(os.path.join(state, "score.json")))
    assert on_disk == features


def test_score_resumes_from_an_extracted_graph(tmp_path):
    state = state_dir(tmp_path)
    backend = FixtureBackend({"extract": lambda v: GOOD_COMMITS})
    offline.extract(state, backend=backend)

    # no extract fixture available: score must not need one on resume
    features = offline.score(state, backend=FixtureBackend({}),
                             kernel_factory=kernel_factory)

    assert features["verdict"] == "verified"


# ---- ranking & selection -----------------------------------------------------

def F(verdict, verified=0, size=1, obligations=0):
    return {"unscoreable": False, "verdict": verdict, "slice_size": size,
            "claims": {"verified": verified},
            "obligations": {"verified": obligations}}


def test_rank_is_lexicographic():
    # verdict dominates absolutely, whatever the lower components say
    assert offline.rank_key(F("verified", verified=1, size=9, obligations=1)) \
        > offline.rank_key(F("partial", verified=9, size=9, obligations=99))
    # same verdict: verified fraction breaks the tie
    assert offline.rank_key(F("partial", verified=8, size=9)) \
        > offline.rank_key(F("partial", verified=7, size=9))
    # same fraction: obligation count breaks it (granularity-resistant)
    assert offline.rank_key(F("partial", verified=1, size=2, obligations=9)) \
        > offline.rank_key(F("partial", verified=1, size=2, obligations=3))
    assert offline.rank_key(None) == (-1, 0.0, 0)


def test_selection_strategies():
    rows = [
        {"answer": "7", "solved": False, "features": F("unverified")},
        {"answer": "7", "solved": False, "features": F("unverified")},
        {"answer": "42", "solved": True,
         "features": F("verified", verified=1, size=1, obligations=4)},
    ]
    assert selection.argmax(rows)["answer"] == "42"
    # one trace one vote: the wrong majority wins — the baseline's failure mode
    assert selection.majority_vote(rows)["answer"] == "7"
    # verdict-weighted: verified (3) outvotes two unverified (1+1)
    assert selection.weighted_vote(rows)["answer"] == "42"
    assert selection.oracle(rows) is True
    assert selection.pass_at_1(rows) == pytest.approx(1 / 3)


def test_voting_falls_back_to_argmax_without_answers():
    rows = [{"answer": None, "solved": False, "features": F("partial")},
            {"answer": None, "solved": False,
             "features": F("verified", verified=1)}]
    assert selection.majority_vote(rows)["features"]["verdict"] == "verified"
