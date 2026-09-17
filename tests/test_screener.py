"""The admission screener: a non-final claim the retained record already
settles is admitted `confident` (assumed, unverified, no check spent);
anything below `high` takes the full pipeline; a final is never screened;
an explicit verify runs the full check; revision clears the flag.

Fixture-backed and instrument-free.
"""
from __future__ import annotations

import json
import os

from verifier.core.events import EventLog
from verifier.core.model import Claim, CLAIM_ASSERTION, CLAIM_FINAL
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.engine.certificate import request_certificate
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.agents import SCREENER_AGENT
from verifier.llm.schema import FixtureBackend
from verifier.stream.builder import Builder


COMPILE = lambda v: {"obligations": [  # noqa: E731
    {"text": f"O of: {v['claim_text']}", "required_basis": "world"}]}
HIGH = {"confidence": "high", "grounds": "seq 3 shows it", "reason": "restated"}
MEDIUM = {"confidence": "medium", "grounds": "seq 3", "reason": "one inference"}


class StubDischarger:
    def __init__(self):
        self.seen = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.seen.append(obligation.text)
        return DischargeResult(witness_kind="execution", outcome="holds",
                               witness=None, artifact="", instrument="stub",
                               basis="world")


def build(tmp_path, roles, screener=SCREENER_AGENT):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend(roles)
    discharger = StubDischarger()
    kernel = Kernel(store, log, backend,
                    ArtifactStore(os.path.join(tmp_path, "art")),
                    critic_role=None, discharger=discharger,
                    screener_role=screener)
    return store, log, backend, discharger, kernel


def record_a_call(log, builder):
    call = log.append("tool_call", {"tool": "Bash", "tool_use_id": "t1",
                                    "input": {"command": "pytest -q"}})
    builder.feed(call)
    result = log.append("tool_result", {"tool": "Bash", "tool_use_id": "t1",
                                        "excerpt": "3 passed",
                                        "response_chars": 8,
                                        "excerpt_truncated": False})
    builder.feed(result)
    return result["seq"]


def test_high_confidence_admits_without_a_check(tmp_path):
    store, log, backend, discharger, kernel = build(
        tmp_path, {"obligations": COMPILE, "screener": HIGH})
    builder = Builder(log, store)
    seq = record_a_call(log, builder)
    effect = builder.labels[f"effect:{seq}"]

    cid = store.add_claim(Claim(CLAIM_ASSERTION,
                                "as of the pytest run, 3 tests passed",
                                deps=(effect,)))

    row = store.get(cid)
    assert row["status"] == "unverified"
    assert row["confident"] is True
    assert "restated" in row["confidence_reason"]
    assert [c["role"] for c in backend.calls] == ["screener"]
    assert discharger.seen == []
    # the screener saw the cited event in full and the recent record
    variables = backend.calls[0]["variables"]
    cited = json.loads(variables["cited_events"])
    assert cited and cited[0]["seq"] == seq
    assert cited[0]["excerpt"]["value"] == "3 passed"
    assert cited[0]["call"]["input"]["value"] == {"command": "pytest -q"}
    assert json.loads(variables["call_index"])[0]["tool"] == "Bash"
    screened = [e for e in log.replay() if e["type"] == "claim_screened"]
    assert screened and screened[0]["payload"]["confidence"] == "high"
    # the flag is disclosed to consumers: graph snapshot and effective fold
    assert store.graph_snapshot()[0]["confident"] is True
    assert store.effective(cid)["effective_status"] == "unverified"


def test_lower_confidence_takes_the_full_pipeline(tmp_path):
    store, log, backend, discharger, kernel = build(
        tmp_path, {"obligations": COMPILE, "screener": MEDIUM})

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the parser handles every input"))

    assert store.status(cid) == "verified"
    assert store.get(cid)["confident"] is False
    assert [c["role"] for c in backend.calls] == ["screener", "obligations"]
    assert discharger.seen == ["O of: the parser handles every input"]


def test_a_final_is_never_screened(tmp_path):
    store, log, backend, discharger, kernel = build(
        tmp_path, {"obligations": COMPILE, "screener": HIGH})

    cid = store.add_claim(Claim(CLAIM_FINAL, "the task is complete"))

    assert store.status(cid) == "verified"
    assert [c["role"] for c in backend.calls] == ["obligations"]


def test_screener_failure_never_blocks(tmp_path):
    def broken(variables):
        raise RuntimeError("seat down")
    store, log, backend, discharger, kernel = build(
        tmp_path, {"obligations": COMPILE, "screener": broken})

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "x holds"))

    assert store.status(cid) == "verified"
    screened = [e for e in log.replay() if e["type"] == "claim_screened"]
    assert screened[0]["payload"]["confidence"] == "error"


def test_explicit_verify_runs_the_full_check(tmp_path):
    store, log, backend, discharger, kernel = build(
        tmp_path, {"obligations": COMPILE, "screener": HIGH})
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "x holds"))
    assert store.get(cid)["confident"] is True

    # a second implicit verify keeps the screener's word (no new calls)
    assert kernel.verify(cid) == "unverified"
    assert [c["role"] for c in backend.calls] == ["screener"]
    # an explicit one does not
    assert kernel.verify(cid, force=True) == "verified"
    assert [c["role"] for c in backend.calls] == ["screener", "obligations"]


def test_confident_premise_is_assumed_by_consumers_and_disclosed(tmp_path):
    store, log, backend, discharger, kernel = build(
        tmp_path, {"obligations": COMPILE,
                   "screener": lambda v: HIGH if "premise" in v["claim_text"]
                   else MEDIUM})
    premise = store.add_claim(Claim(CLAIM_ASSERTION, "the premise holds"))
    consumer = store.add_claim(Claim(CLAIM_ASSERTION, "the consequence holds",
                                     deps=(premise,)))

    assert store.get(premise)["confident"] is True
    assert store.status(consumer) == "verified"
    compile_call = [c for c in backend.calls if c["role"] == "obligations"][0]
    assert "admitted confident" in compile_call["variables"]["pending_deps"]
    # the premise stays open in the fold: confident is not verified
    assert store.effective(consumer)["effective_status"] == "unverified"


def test_revision_clears_the_flag_and_certificate_discloses_it(tmp_path):
    store, log, backend, discharger, kernel = build(
        tmp_path, {"obligations": COMPILE,
                   "screener": lambda v: HIGH if "old" in v["claim_text"]
                   else MEDIUM})
    old = store.add_claim(Claim(CLAIM_ASSERTION, "the old reading"))
    assert store.get(old)["confident"] is True
    new = store.add_claim(Claim(CLAIM_ASSERTION, "the new reading", revises=old))
    assert store.get(old)["status"] == "stale"
    assert store.get(old)["confident"] is False
    assert store.status(new) == "verified"

    final = store.add_claim(Claim(CLAIM_FINAL, "done", deps=(new,)))
    cert = request_certificate(store, kernel, log,
                               ArtifactStore(os.path.join(tmp_path, "art2")),
                               final)
    by_text = {e["text"]: e for e in cert["slice"]}
    assert by_text["the new reading"]["confident"] is False
    assert cert["verdict"] == "verified"


def test_no_screener_seated_checks_everything(tmp_path):
    store, log, backend, discharger, kernel = build(
        tmp_path, {"obligations": COMPILE}, screener=None)
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "x holds"))
    assert store.status(cid) == "verified"
    assert [c["role"] for c in backend.calls] == ["obligations"]
