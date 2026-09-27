"""The screener: a high-confidence verified / supported / refuted read off
the retained record becomes the claim's status, no check spent, marked
`confident`; anything below `high`, or `open`, takes the full pipeline; a
refutation needs a positive witness; a final is screened only when finals
are unified, and never verified on the record alone; an explicit verify
reopens the screener's verdict for the full check; revision clears the
flag.

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
HIGH = {"verdict": "verified", "confidence": "high", "basis": "world",
        "grounds": "seq 3 shows it", "reason": "restated"}
SUPPORTED = dict(HIGH, verdict="supported", grounds="seq 3 plus a bridge")
REFUTED = dict(HIGH, verdict="refuted", grounds="seq 3 contradicts it",
               witness="seq 3 printed 2 passed", witness_kind="execution")
MEDIUM = dict(HIGH, confidence="medium", reason="one inference")
OPEN = dict(HIGH, verdict="open", reason="the record is silent")


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
    assert row["status"] == "verified"
    assert row["strength"] == "decisive" and row["basis"] == "world"
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
    # the status is real for consumers; the flag discloses its source
    assert store.graph_snapshot()[0]["confident"] is True
    assert store.effective(cid)["effective_status"] == "verified"


def test_lower_confidence_or_open_takes_the_full_pipeline(tmp_path):
    for screen in (MEDIUM, OPEN, dict(REFUTED, confidence="low")):
        store, log, backend, discharger, kernel = build(
            tmp_path, {"obligations": COMPILE, "screener": screen})

        cid = store.add_claim(Claim(CLAIM_ASSERTION,
                                    "the parser handles every input"))

        assert store.status(cid) == "verified"
        assert store.get(cid)["confident"] is False
        assert [c["role"] for c in backend.calls] == ["screener",
                                                      "obligations"]
        assert discharger.seen == ["O of: the parser handles every input"]


def test_supported_settles_below_decisive(tmp_path):
    store, log, backend, discharger, kernel = build(
        tmp_path, {"obligations": COMPILE, "screener": SUPPORTED})
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "x holds"))

    row = store.get(cid)
    assert row["status"] == "verified" and row["confident"] is True
    assert row["strength"] == "analytic"
    assert discharger.seen == []


def test_a_refutation_settles_with_its_witness(tmp_path):
    store, log, backend, discharger, kernel = build(
        tmp_path, {"obligations": COMPILE, "screener": REFUTED})
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "3 tests passed"))

    row = store.get(cid)
    assert row["status"] == "refuted" and row["confident"] is True
    assert row["strength"] == "decisive"
    assert "2 passed" in row["reason"]
    assert discharger.seen == []
    ev = store.evidence_for(cid)[-1]
    assert ev["instrument"] == "screener"
    assert ev["witness_kind"] == "execution"


def test_a_refutation_without_a_positive_witness_is_checked(tmp_path):
    for bad in (dict(REFUTED, witness_kind="absence"),
                dict(REFUTED, witness=""),
                {k: v for k, v in REFUTED.items() if k != "witness_kind"}):
        store, log, backend, discharger, kernel = build(
            tmp_path, {"obligations": COMPILE, "screener": bad})
        cid = store.add_claim(Claim(CLAIM_ASSERTION, "3 tests passed"))

        assert store.status(cid) == "verified"
        assert store.get(cid)["confident"] is False
        screened = [e for e in log.replay()
                    if e["type"] == "claim_screened"][-1]["payload"]
        assert "positive witness" in screened["refused"]


def test_a_final_is_not_screened_unless_finals_are_unified(tmp_path):
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

    assert store.status(cid) == "verified"
    # a second implicit verify keeps the screener's verdict (no new calls)
    assert kernel.verify(cid) == "verified"
    assert [c["role"] for c in backend.calls] == ["screener"]
    # an explicit one reopens it for the full check
    assert kernel.verify(cid, force=True) == "verified"
    assert [c["role"] for c in backend.calls] == ["screener", "obligations"]
    assert store.get(cid)["confident"] is False
    assert discharger.seen == ["O of: x holds"]


def test_explicit_verify_can_overturn_a_screener_refutation(tmp_path):
    store, log, backend, discharger, kernel = build(
        tmp_path, {"obligations": COMPILE, "screener": REFUTED})
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "3 tests passed"))
    assert store.status(cid) == "refuted"

    assert kernel.verify(cid, force=True) == "verified"
    assert store.get(cid)["confident"] is False


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
    assert "settled by the screener" in \
        compile_call["variables"]["verified_deps"]
    # the screener's verdict is the premise's status in the fold
    assert store.effective(consumer)["effective_status"] == "verified"


def test_a_screener_verdict_never_transfers_to_an_obligation_node(tmp_path):
    store, log, backend, discharger, kernel = build(
        tmp_path, {"obligations": lambda v: {"obligations": [
                       {"text": "the premise holds",
                        "required_basis": "record"}]},
                   "screener": lambda v: HIGH if v["claim_text"]
                   == "the premise holds" else MEDIUM})
    premise = store.add_claim(Claim(CLAIM_ASSERTION, "the premise holds"))
    assert store.get(premise)["confident"] is True
    assert store.settled_statement("the premise holds") is None

    store.add_claim(Claim(CLAIM_ASSERTION, "another claim"))
    # the same-text obligation was checked, not inherited
    assert discharger.seen == ["the premise holds"]


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


def test_compile_starts_alongside_the_screener_when_speculating(tmp_path):
    import threading
    seen = []
    gate = threading.Event()

    def compile_role(variables):
        seen.append("compile")
        gate.wait(5)
        return COMPILE(variables)

    def screener(variables):
        seen.append("screener")
        gate.set()
        return MEDIUM

    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": compile_role,
                              "screener": screener})
    discharger = StubDischarger()
    Kernel(store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
           critic_role=None, discharger=discharger, binder=discharger,
           screener_role=SCREENER_AGENT, speculative_discharge=True)
    discharger.run = lambda row, obligation: discharger.discharge(
        row, obligation)

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "x holds"))

    assert store.status(cid) == "verified"
    # the compiler was running while the screener answered, and its
    # answer was used: exactly one compile call
    assert seen == ["compile", "screener"]
    assert [c["role"] for c in backend.calls].count("obligations") == 1
    assert [e for e in log.replay() if e["type"] == "speculative_compile"]


def test_a_high_screen_drops_the_speculative_compile(tmp_path):
    seen = []

    def compile_role(variables):
        seen.append("compile")
        return COMPILE(variables)

    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": compile_role, "screener": HIGH})
    discharger = StubDischarger()
    Kernel(store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
           critic_role=None, discharger=discharger, binder=discharger,
           screener_role=SCREENER_AGENT, speculative_discharge=True)

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "x holds"))

    assert store.get(cid)["confident"] is True
    assert discharger.seen == []


def test_a_unified_final_settled_by_the_screener_certifies_its_verdict(
        tmp_path):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": COMPILE, "screener": HIGH})
    discharger = StubDischarger()
    artifacts = ArtifactStore(os.path.join(tmp_path, "art"))
    kernel = Kernel(store, log, backend, artifacts, critic_role=None,
                    discharger=discharger, screener_role=SCREENER_AGENT,
                    unify_final=True)
    cid = store.add_claim(Claim(CLAIM_FINAL, "the task is complete"))

    assert store.status(cid) == "verified"
    assert discharger.seen == []
    cert = request_certificate(store, kernel, log, artifacts, cid)
    assert cert["verdict"] == "verified"
    assert cert["settled_by"] == "screener"
    assert "admitted_final_check" not in [e["type"] for e in log.replay()]


def test_a_unified_final_is_never_verified_on_the_record_alone(tmp_path):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": COMPILE,
                              "screener": dict(HIGH, basis="record")})
    discharger = StubDischarger()
    artifacts = ArtifactStore(os.path.join(tmp_path, "art"))
    Kernel(store, log, backend, artifacts, critic_role=None,
           discharger=discharger, screener_role=SCREENER_AGENT,
           unify_final=True)
    cid = store.add_claim(Claim(CLAIM_FINAL, "the task is complete"))

    # the screen is refused; the contract is checked formally
    assert store.get(cid)["confident"] is False
    assert discharger.seen == ["O of: the task is complete"]


def test_a_unified_final_refuted_by_the_screener(tmp_path):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": COMPILE, "screener": REFUTED})
    discharger = StubDischarger()
    artifacts = ArtifactStore(os.path.join(tmp_path, "art"))
    kernel = Kernel(store, log, backend, artifacts, critic_role=None,
                    discharger=discharger, screener_role=SCREENER_AGENT,
                    unify_final=True)
    cid = store.add_claim(Claim(CLAIM_FINAL, "the task is complete"))

    cert = request_certificate(store, kernel, log, artifacts, cid)
    assert cert["verdict"] == "refuted"
    assert cert["settled_by"] == "screener"


def test_the_stop_gate_has_no_screened_verdict():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stop_hook", os.path.join(os.path.dirname(__file__), "..", "hooks",
                                  "stop.py"))
    stop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stop)
    assert "screened" not in stop.VERDICT_RANK
