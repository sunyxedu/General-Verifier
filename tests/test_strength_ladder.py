"""The evidence-strength ladder (docs/evidence-strength-ladder.md).

Strength is the second evidence axis, orthogonal to basis: decisive >
bounded > analytic. The binding game executes strictness-descending and
only degrades under audit; a verified claim whose weakest evidence is
below decisive surfaces as "supported" on the certificate; nothing below
decisive may refute.
"""
from __future__ import annotations

import os
import sqlite3

from verifier.core.model import (
    Claim, CLAIM_ASSERTION,
    STRENGTH_ANALYTIC, STRENGTH_BOUNDED, STRENGTH_DECISIVE,
    strength_rank, weakest_strength,
)
from verifier.core.store import ClaimStore


# ---- the lattice ----------------------------------------------------------

def test_strength_rank_orders_the_ladder():
    assert (strength_rank(STRENGTH_DECISIVE)
            > strength_rank(STRENGTH_BOUNDED)
            > strength_rank(STRENGTH_ANALYTIC))


def test_no_declaration_reads_as_decisive():
    """None = the ladder was never in play (direct discharge, ledger
    checks, pre-ladder rows): pre-ladder semantics are preserved."""
    assert strength_rank(None) == strength_rank(STRENGTH_DECISIVE)


def test_unknown_declared_value_reads_as_weakest():
    """A declared-but-unrecognized value must not grant strength."""
    assert strength_rank("formal-ish") == strength_rank(STRENGTH_ANALYTIC)


def test_weakest_strength_is_a_weakest_link_fold():
    assert weakest_strength(
        [STRENGTH_DECISIVE, STRENGTH_BOUNDED, STRENGTH_DECISIVE]
    ) == STRENGTH_BOUNDED
    assert weakest_strength(
        [STRENGTH_BOUNDED, STRENGTH_ANALYTIC]) == STRENGTH_ANALYTIC
    assert weakest_strength(
        [None, STRENGTH_DECISIVE]) == STRENGTH_DECISIVE


def test_weakest_strength_empty_folds_to_decisive():
    assert weakest_strength([]) == STRENGTH_DECISIVE


def test_weakest_strength_normalizes_unknown_values():
    assert weakest_strength([STRENGTH_DECISIVE, "vibes"]) == STRENGTH_ANALYTIC


# ---- persistence ----------------------------------------------------------

def test_store_persists_strength_on_verified(tmp_path):
    store = ClaimStore()
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 42"))
    store.set_verified(cid, basis="world", strength=STRENGTH_BOUNDED)
    row = store.get(cid)
    assert row["strength"] == STRENGTH_BOUNDED
    assert row["basis"] == "world"


def test_store_strength_defaults_to_null(tmp_path):
    store = ClaimStore()
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 42"))
    store.set_verified(cid, basis="world")
    assert store.get(cid)["strength"] is None


def test_store_migrates_pre_strength_databases(tmp_path):
    """A store created before strength tracking gains the column on open
    (CREATE TABLE IF NOT EXISTS never alters an existing table)."""
    path = os.path.join(tmp_path, "claims.sqlite")
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE claims ("
        " id TEXT PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL,"
        " revises TEXT, status TEXT NOT NULL DEFAULT 'unverified',"
        " reason TEXT, basis TEXT);"
        "CREATE TABLE deps (child TEXT NOT NULL, parent TEXT NOT NULL,"
        " PRIMARY KEY (child, parent));"
    )
    db.execute(
        "INSERT INTO claims (id, kind, text, status) VALUES (?,?,?,?)",
        ("old", "assertion", "legacy row", "verified"),
    )
    db.commit()
    db.close()

    store = ClaimStore(path)
    assert store.get("old")["strength"] is None
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "new row"))
    store.set_verified(cid, strength=STRENGTH_DECISIVE)
    assert store.get(cid)["strength"] == STRENGTH_DECISIVE


def test_graph_snapshot_discloses_strength(tmp_path):
    store = ClaimStore()
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the answer is 42"))
    store.set_verified(cid, basis="world", strength=STRENGTH_ANALYTIC)
    (entry,) = store.graph_snapshot()
    assert entry["strength"] == STRENGTH_ANALYTIC


# ---- the binding game's ladder mechanics ----------------------------------

import json as _json

from verifier.core.events import EventLog
from verifier.core.model import Obligation
from verifier.engine.game import BindingGame
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend

ROW = {"id": "c1", "kind": "assertion", "text": "there is a bug xxx"}


def method(procedure, strength=STRENGTH_DECISIVE, bridges=(), basis="world"):
    return {"procedure": procedure, "artifact": "check output",
            "level": "direct", "basis": basis, "strength": strength,
            "bridges": list(bridges)}


SYSTEM_TEST = method("system test: run the full reproduction scenario")
UNIT_TEST = method(
    "unit test: drive f with input I against its local contract",
    strength=STRENGTH_BOUNDED,
    bridges=["call path P reaches f(I)", "no upstream sanitizer rejects I"])
CODE_READING = method(
    "read src/m.c:120-180 (sha256 pinned) and trace input I step by step",
    strength=STRENGTH_ANALYTIC,
    bridges=["the code as read is the code as built"])

JUDGE_OK = {"sufficient": "yes", "faithful": "yes", "degradation": "n/a",
            "reason": "checks out"}
JUDGE_OK_DEGRADED = {"sufficient": "yes", "faithful": "yes",
                     "degradation": "justified", "reason": "gap is real"}


class ScriptedDischarger:
    """outcome per procedure, recording execution order."""

    def __init__(self, outcomes):
        self.outcomes = dict(outcomes)
        self.executed = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.executed.append(method["procedure"])
        outcome, witness = self.outcomes[method["procedure"]]
        return DischargeResult(outcome=outcome, witness=witness,
                               artifact="d1", instrument="stub")


def ladder_game(tmp_path, roles, discharger, live_world=True):
    backend = FixtureBackend(roles)
    return BindingGame(
        backend,
        ArtifactStore(os.path.join(tmp_path, "art")),
        EventLog(os.path.join(tmp_path, "events.jsonl")),
        discharger,
        live_world=live_world,
    ), backend


def test_execution_is_strictness_descending(tmp_path):
    """The engine sorts tiers strongest-first regardless of the oracle's
    listing order; within a tier the oracle's order survives."""
    discharger = ScriptedDischarger({
        SYSTEM_TEST["procedure"]: ("holds", None),
    })
    g, _ = ladder_game(
        tmp_path,
        {"oracles": {"methods": [CODE_READING, UNIT_TEST, SYSTEM_TEST]},
         "judger": JUDGE_OK},
        discharger,
    )

    result = g.run(ROW, Obligation("c1", "the bug is present", "world"))

    assert result.outcome == "holds"
    assert result.strength == STRENGTH_DECISIVE
    assert discharger.executed == [SYSTEM_TEST["procedure"]]


def test_audited_gap_unlocks_the_next_tier(tmp_path):
    """system test structurally unavailable -> unit test binds; the judger
    receives the degradation record and the settled result carries the
    bounded strength (-> supported at the certificate)."""
    discharger = ScriptedDischarger({
        SYSTEM_TEST["procedure"]: ("unobtainable", "no API key in the env"),
        UNIT_TEST["procedure"]: ("holds", None),
    })
    g, backend = ladder_game(
        tmp_path,
        {"oracles": {"methods": [SYSTEM_TEST, UNIT_TEST]},
         "judger": JUDGE_OK_DEGRADED},
        discharger,
    )

    result = g.run(ROW, Obligation("c1", "the bug is present", "world"))

    assert result.outcome == "holds"
    assert result.strength == STRENGTH_BOUNDED
    assert discharger.executed == [SYSTEM_TEST["procedure"],
                                   UNIT_TEST["procedure"]]
    judger_vars = [c for c in backend.calls
                   if c["role"] == "judger"][-1]["variables"]
    assert judger_vars["declared_strength"] == STRENGTH_BOUNDED
    assert "no upstream sanitizer rejects I" in judger_vars["bridges"]
    record = _json.loads(judger_vars["degradation_record"])
    assert record["stronger_attempts"][0]["tier"] == STRENGTH_DECISIVE
    assert record["stronger_attempts"][0]["outcome"] == "unobtainable"
    assert record["omitted_tiers"] == []


def test_failed_decisive_does_not_unlock_weaker_tiers(tmp_path):
    """A retryable failure is not a structural gap: the weaker method
    stays deferred and the obligation fails (unverified upstream)."""
    discharger = ScriptedDischarger({
        SYSTEM_TEST["procedure"]: ("failed", "harness crashed"),
    })

    def enumerate_methods(variables):
        if variables["method_feedback"] == "(none)":
            return {"methods": [SYSTEM_TEST, UNIT_TEST]}
        feedback = _json.loads(variables["method_feedback"])
        assert feedback["deferred_methods"][0]["strength"] == STRENGTH_BOUNDED
        assert feedback["deferred_methods"][0]["waiting_on"] == [
            STRENGTH_DECISIVE]
        return {"methods": []}

    g, _ = ladder_game(
        tmp_path,
        {"oracles": enumerate_methods, "judger": JUDGE_OK},
        discharger,
    )

    result = g.run(ROW, Obligation("c1", "the bug is present", "world"))

    assert result.outcome == "failed"
    assert "deferred behind a stronger tier" in result.witness
    assert discharger.executed == [SYSTEM_TEST["procedure"]]


def test_omitted_decisive_tier_reaches_the_judger_for_audit(tmp_path):
    """Listing no decisive method is a signed declaration, not a fast
    path: the weaker method executes but its degradation record shows
    the omission for the judger to audit."""
    discharger = ScriptedDischarger({
        UNIT_TEST["procedure"]: ("holds", None),
    })
    g, backend = ladder_game(
        tmp_path,
        {"oracles": {"methods": [UNIT_TEST]}, "judger": JUDGE_OK_DEGRADED},
        discharger,
    )

    result = g.run(ROW, Obligation("c1", "the bug is present", "world"))

    assert result.outcome == "holds"
    record = _json.loads(
        [c for c in backend.calls if c["role"] == "judger"][-1]
        ["variables"]["degradation_record"])
    assert record["omitted_tiers"] == [STRENGTH_DECISIVE]


def test_unjustified_degradation_rejects_and_disputes_the_gap(tmp_path):
    """The third signature: when the judger disputes that the stronger
    tier was genuinely unavailable, the weaker evidence may not settle
    AND the obligation's unobtainable story falls with it (failed, not
    unobtainable — fail-closed)."""
    discharger = ScriptedDischarger({
        SYSTEM_TEST["procedure"]: ("unobtainable", "claims no API key"),
        UNIT_TEST["procedure"]: ("holds", None),
    })
    unjustified = {"sufficient": "yes", "faithful": "yes",
                   "degradation": "unjustified",
                   "reason": "the ledger shows a usable offline replayer"}
    g, _ = ladder_game(
        tmp_path,
        {"oracles": oracle_once({"methods": [SYSTEM_TEST, UNIT_TEST]}),
         "judger": unjustified},
        discharger,
    )

    result = g.run(ROW, Obligation("c1", "the bug is present", "world"))

    assert result.outcome == "failed"
    assert any("degradation unjustified" in (e["payload"].get("reason") or "")
               for e in g.log.replay() if e["type"] == "method_rejected")


def oracle_once(initial):
    """Initial batch once, empty repair afterwards."""
    def enumerate_methods(variables):
        return initial if variables["method_feedback"] == "(none)" \
            else {"methods": []}
    return enumerate_methods


def test_refutation_below_decisive_converts_to_failed(tmp_path):
    """A failing unit test does not prove the system-level bug absent (or
    present): nothing below decisive may refute. The negative finding
    stays on record as a diagnostic; the judger is never consulted."""
    discharger = ScriptedDischarger({
        SYSTEM_TEST["procedure"]: ("unobtainable", "no API key"),
        UNIT_TEST["procedure"]: ("refuted", "assertion failed at step 3"),
    })
    g, backend = ladder_game(
        tmp_path,
        {"oracles": oracle_once({"methods": [SYSTEM_TEST, UNIT_TEST]})},
        discharger,
    )

    result = g.run(ROW, Obligation("c1", "the bug is present", "world"))

    assert result.outcome != "refuted"
    assert not [c for c in backend.calls if c["role"] == "judger"]
    rejected = [e for e in g.log.replay() if e["type"] == "method_rejected"]
    assert any("refutation below decisive strength is inadmissible"
               in e["payload"]["reason"] for e in rejected)
    assert any("assertion failed at step 3" in e["payload"]["reason"]
               for e in rejected)


def test_repair_can_unlock_deferred_weaker_methods(tmp_path):
    """A repaired decisive method that lands unobtainable clears the tier
    after the fact: parked weaker methods re-queue and may bind."""
    replacement = method("system test via the recorded fixture server")
    discharger = ScriptedDischarger({
        SYSTEM_TEST["procedure"]: ("failed", "harness crashed"),
        replacement["procedure"]: ("unobtainable", "fixture server gone"),
        UNIT_TEST["procedure"]: ("holds", None),
    })

    def enumerate_methods(variables):
        if variables["method_feedback"] == "(none)":
            return {"methods": [SYSTEM_TEST, UNIT_TEST]}
        return {"methods": [replacement]}

    g, _ = ladder_game(
        tmp_path,
        {"oracles": enumerate_methods, "judger": JUDGE_OK_DEGRADED},
        discharger,
    )

    result = g.run(ROW, Obligation("c1", "the bug is present", "world"))

    assert result.outcome == "holds"
    assert result.strength == STRENGTH_BOUNDED
    assert discharger.executed == [SYSTEM_TEST["procedure"],
                                   replacement["procedure"],
                                   UNIT_TEST["procedure"]]


def test_full_ladder_descends_to_analytic(tmp_path):
    """system test and unit test both structurally unavailable -> the
    disciplined code reading binds at analytic strength, with both
    stronger attempts in its degradation record."""
    discharger = ScriptedDischarger({
        SYSTEM_TEST["procedure"]: ("unobtainable", "no API key"),
        UNIT_TEST["procedure"]: ("unobtainable", "cannot isolate the unit"),
        CODE_READING["procedure"]: ("holds", None),
    })
    g, backend = ladder_game(
        tmp_path,
        {"oracles": {"methods": [SYSTEM_TEST, UNIT_TEST, CODE_READING]},
         "judger": JUDGE_OK_DEGRADED},
        discharger,
    )

    result = g.run(ROW, Obligation("c1", "the bug is present", "world"))

    assert result.outcome == "holds"
    assert result.strength == STRENGTH_ANALYTIC
    record = _json.loads(
        [c for c in backend.calls if c["role"] == "judger"][-1]
        ["variables"]["degradation_record"])
    assert [a["tier"] for a in record["stronger_attempts"]] == [
        STRENGTH_DECISIVE, STRENGTH_BOUNDED]


# ---- prompt and schema pins ----------------------------------------------

def test_prompts_pin_the_ladder():
    from verifier.llm import prompts
    assert "STRENGTH LADDER" in prompts.ORACLES_SYSTEM
    assert "strength=analytic" in prompts.ORACLES_SYSTEM
    assert "FALLBACKS, not substitutes" in prompts.ORACLES_SYSTEM
    assert "strength: decisive | bounded | analytic" in prompts.ORACLES_TEMPLATE
    assert "STRENGTH IS PART OF SUFFICIENCY" in prompts.JUDGER_SYSTEM
    assert "DEGRADATION AUDIT" in prompts.JUDGER_SYSTEM
    assert "{declared_strength}" in prompts.JUDGER_TEMPLATE
    assert "{bridges}" in prompts.JUDGER_TEMPLATE
    assert "{degradation_record}" in prompts.JUDGER_TEMPLATE
    assert "STRENGTH AUDIT" in prompts.JUDGER_TEMPLATE


def test_role_schemas_pin_the_ladder_fields():
    from verifier.llm.agents import ENUMERATE_ORACLES_AGENT, JUDGER_AGENT
    method_schema = ENUMERATE_ORACLES_AGENT.schema["methods"][0]
    assert method_schema["strength"] == ("decisive", "bounded", "analytic")
    assert method_schema["bridges"] == [str]
    assert JUDGER_AGENT.schema["degradation"] == (
        "justified", "unjustified", "n/a")


# ---- settlement fold and the certificate read-out -------------------------

from verifier.core.model import CLAIM_FINAL
from verifier.core.store import ClaimStore as _Store, TRIGGER_ON_COMMIT
from verifier.engine.certificate import request_certificate
from verifier.engine.kernel import Kernel


class StrengthDischarger:
    """(outcome, strength) per obligation text; world basis throughout."""

    def __init__(self, plan):
        self.plan = plan

    def discharge(self, row, obligation, method=None, tag=""):
        outcome, strength = self.plan.get(
            obligation.text, ("holds", STRENGTH_DECISIVE))
        return DischargeResult(
            outcome=outcome,
            witness="w" if outcome != "holds" else None,
            artifact="", instrument="stub", basis="world",
            strength=strength)


def kernel_fixture(tmp_path, obligations, plan):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = _Store(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend(
        {"obligations": lambda v: {"obligations": obligations}})
    artifacts = ArtifactStore(os.path.join(tmp_path, "art"))
    kernel = Kernel(store, log, backend, artifacts, critic_role=None,
                    discharger=StrengthDischarger(plan))
    return store, kernel, log, artifacts


def test_claim_strength_is_the_weakest_link(tmp_path):
    store, _, _, _ = kernel_fixture(
        tmp_path,
        [{"text": "system-level check", "required_basis": "world"},
         {"text": "unit-level check", "required_basis": "world"}],
        {"system-level check": ("holds", STRENGTH_DECISIVE),
         "unit-level check": ("holds", STRENGTH_BOUNDED)},
    )

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "there is a bug xxx"))

    assert store.status(cid) == "verified"
    assert store.get(cid)["strength"] == STRENGTH_BOUNDED


def test_supported_verdict_on_the_certificate(tmp_path):
    """verified status + below-decisive strength reads out as supported;
    the strength and per-node disclosures ride the certificate."""
    store, kernel, log, artifacts = kernel_fixture(
        tmp_path,
        [{"text": "the bug reproduces", "required_basis": "world"}],
        {"the bug reproduces": ("holds", STRENGTH_BOUNDED)},
    )

    f = store.add_claim(Claim(CLAIM_FINAL, "there is a bug xxx"))
    cert = request_certificate(store, kernel, log, artifacts, f)

    assert store.status(f) == "verified"     # status machine untouched
    assert cert["verdict"] == "supported"    # the read-out is the ladder's
    assert cert["verdict_strength"] == STRENGTH_BOUNDED
    (entry,) = [e for e in cert["slice"] if e["id"] == f]
    assert entry["strength"] == STRENGTH_BOUNDED
    assert entry["obligations"][0]["strength"] == STRENGTH_BOUNDED


def test_all_decisive_final_stays_verified(tmp_path):
    store, kernel, log, artifacts = kernel_fixture(
        tmp_path,
        [{"text": "the bug reproduces", "required_basis": "world"}],
        {"the bug reproduces": ("holds", STRENGTH_DECISIVE)},
    )

    f = store.add_claim(Claim(CLAIM_FINAL, "there is a bug xxx"))
    cert = request_certificate(store, kernel, log, artifacts, f)

    assert cert["verdict"] == "verified"
    assert cert["verdict_strength"] == STRENGTH_DECISIVE


def test_missing_evidence_still_beats_weak_evidence(tmp_path):
    """holds@bounded + unobtainable folds to partial: everything
    established on weak evidence (supported) outranks a claim where a
    necessary check never got established at all."""
    store, _, _, _ = kernel_fixture(
        tmp_path,
        [{"text": "unit-level check", "required_basis": "world"},
         {"text": "irreproducible historical check",
          "required_basis": "world"}],
        {"unit-level check": ("holds", STRENGTH_BOUNDED),
         "irreproducible historical check":
             ("unobtainable", STRENGTH_DECISIVE)},
    )

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "there is a bug xxx"))

    assert store.status(cid) == "partial"


def test_inherited_verdicts_carry_their_strength(tmp_path):
    """Node dedup must not launder strength: a bounded-won verdict
    transfers as bounded into every claim that shares the obligation."""
    store, _, _, _ = kernel_fixture(
        tmp_path,
        [{"text": "shared check", "required_basis": "world"}],
        {"shared check": ("holds", STRENGTH_ANALYTIC)},
    )

    first = store.add_claim(Claim(CLAIM_ASSERTION, "claim one"))
    second = store.add_claim(Claim(CLAIM_ASSERTION, "claim two"))

    assert store.get(first)["strength"] == STRENGTH_ANALYTIC
    assert store.get(second)["strength"] == STRENGTH_ANALYTIC


def test_kernel_backstop_blocks_non_decisive_refutations(tmp_path):
    """Direct-path mirror of the game's conversion: a custom discharger
    reporting refuted at bounded strength cannot refute."""
    store, _, _, _ = kernel_fixture(
        tmp_path,
        [{"text": "unit-level check", "required_basis": "world"}],
        {"unit-level check": ("refuted", STRENGTH_BOUNDED)},
    )

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "there is a bug xxx"))

    assert store.status(cid) == "unverified"
    (node,) = store.obligation_children(cid)
    ev = store.evidence_for(node["id"])[-1]
    assert ev["outcome"] == "failed"
    assert "refutation below decisive strength is inadmissible" \
        in ev["witness"]


def test_decisive_refutation_still_refutes(tmp_path):
    store, _, _, _ = kernel_fixture(
        tmp_path,
        [{"text": "system-level check", "required_basis": "world"}],
        {"system-level check": ("refuted", STRENGTH_DECISIVE)},
    )

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "there is a bug xxx"))

    assert store.status(cid) == "refuted"


# ---- ranking and selection ------------------------------------------------

from verifier.offline import VERDICT_RANK, rank_key
from verifier.selection import scalar, weighted_vote


def _features(verdict, frac=0.5, total=2, obligations=1):
    verified = round(frac * total)
    return {"unscoreable": False, "verdict": verdict,
            "slice_size": total, "claims": {"verified": verified},
            "obligations": {"verified": obligations},
            "obligations_total": obligations}


def test_supported_ranks_between_verified_and_partial():
    assert (VERDICT_RANK["verified"] > VERDICT_RANK["supported"]
            > VERDICT_RANK["partial"] > VERDICT_RANK["unverified"]
            > VERDICT_RANK["refuted"])
    assert rank_key(_features("supported"))[0] == 3


def test_scalar_bands_never_cross():
    """Six bands: the best trace of a lower band stays under the worst
    trace of the band above, at the extremes of frac and obligations."""
    best_supported = scalar(_features("supported", frac=1.0, total=44,
                                      obligations=10_000))
    worst_verified = scalar(_features("verified", frac=0.0,
                                      obligations=0))
    assert best_supported < worst_verified
    order = [scalar(_features(v, frac=1.0)) for v in
             ("refuted", "unverified", "partial", "supported", "verified")]
    assert order == sorted(order)
    assert scalar(None) == 0.0                       # unscoreable
    assert scalar(_features("refuted", frac=0.0)) > 0.0


def test_weighted_vote_zeroes_refuted_traces():
    """Three refuted traces agreeing on an answer must not outvote one
    unverified trace: a verdict of WRONG is not a vote for the answer."""
    rows = [
        {"answer": "A", "features": _features("refuted")},
        {"answer": "A", "features": _features("refuted")},
        {"answer": "A", "features": _features("refuted")},
        {"answer": "B", "features": _features("unverified")},
    ]
    assert weighted_vote(rows)["answer"] == "B"


def test_weighted_vote_supported_sits_between_the_bands():
    """One supported trace outvotes one partial trace and loses to one
    verified trace."""
    rows = [
        {"answer": "A", "features": _features("supported")},
        {"answer": "B", "features": _features("partial")},
    ]
    assert weighted_vote(rows)["answer"] == "A"
    rows = [
        {"answer": "A", "features": _features("supported")},
        {"answer": "B", "features": _features("verified")},
    ]
    assert weighted_vote(rows)["answer"] == "B"


# ---- acceptance: the bug-scenario walk (docs/evidence-strength-ladder.md §0)

BUG_CLAIM = "there is a bug: f mis-handles input I in the delivered code"


def acceptance_pipeline(tmp_path, outcomes, judger=None):
    """The full pipeline with the binding game seated as the kernel's
    binder: claim text == obligation text (the degenerate decomposition
    is legal and stable), three methods on the ladder."""
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = _Store(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({
        "obligations": lambda v: {"obligations": [
            {"text": BUG_CLAIM, "required_basis": "world"}]},
        "oracles": {"methods": [SYSTEM_TEST, UNIT_TEST, CODE_READING]},
        "judger": judger or JUDGE_OK_DEGRADED,
    })
    artifacts = ArtifactStore(os.path.join(tmp_path, "art"))
    discharger = ScriptedDischarger(outcomes)
    binder = BindingGame(backend, artifacts, log, discharger,
                         live_world=False)
    kernel = Kernel(store, log, backend, artifacts, critic_role=None,
                    discharger=discharger, binder=binder, live_world=False)
    f = store.add_claim(Claim(CLAIM_FINAL, BUG_CLAIM))
    cert = request_certificate(store, kernel, log, artifacts, f)
    return store, f, cert, discharger


def test_acceptance_system_test_verifies(tmp_path):
    _, f, cert, d = acceptance_pipeline(tmp_path, {
        SYSTEM_TEST["procedure"]: ("holds", None),
    })
    assert cert["verdict"] == "verified"
    assert cert["verdict_strength"] == STRENGTH_DECISIVE
    assert d.executed == [SYSTEM_TEST["procedure"]]


def test_acceptance_degrades_to_supported_at_bounded(tmp_path):
    """system test structurally unavailable -> unit test binds ->
    supported@bounded, with the unverified bridges disclosed on the
    certificate's evidence."""
    _, f, cert, d = acceptance_pipeline(tmp_path, {
        SYSTEM_TEST["procedure"]: ("unobtainable", "no API key in the env"),
        UNIT_TEST["procedure"]: ("holds", None),
    })
    assert cert["verdict"] == "supported"
    assert cert["verdict_strength"] == STRENGTH_BOUNDED
    (entry,) = [e for e in cert["slice"] if e["id"] == f]
    node = entry["obligations"][0]
    assert node["status"] == "verified"
    assert node["strength"] == STRENGTH_BOUNDED
    assert node["evidence"][-1]["bridges"] == UNIT_TEST["bridges"]


def test_acceptance_descends_to_supported_at_analytic(tmp_path):
    _, f, cert, d = acceptance_pipeline(tmp_path, {
        SYSTEM_TEST["procedure"]: ("unobtainable", "no API key"),
        UNIT_TEST["procedure"]: ("unobtainable", "cannot isolate the unit"),
        CODE_READING["procedure"]: ("holds", None),
    })
    assert cert["verdict"] == "supported"
    assert cert["verdict_strength"] == STRENGTH_ANALYTIC
    assert d.executed == [SYSTEM_TEST["procedure"], UNIT_TEST["procedure"],
                          CODE_READING["procedure"]]


def test_acceptance_system_test_refutes(tmp_path):
    _, f, cert, _ = acceptance_pipeline(tmp_path, {
        SYSTEM_TEST["procedure"]: ("refuted", "scenario runs correctly"),
    })
    assert cert["verdict"] == "refuted"
    assert cert["verdict_strength"] is None


def test_acceptance_negative_unit_signal_blocks_weaker_holds(tmp_path):
    """A bounded instrument's negative result cannot refute — but it does
    keep the analytic tier locked (the bounded tier ended failed, not
    structurally unavailable), so a code reading cannot assert holds over
    a live contrary signal. Fail-closed to unverified."""
    store, f, cert, d = acceptance_pipeline(tmp_path, {
        SYSTEM_TEST["procedure"]: ("unobtainable", "no API key"),
        UNIT_TEST["procedure"]: ("refuted", "assertion failed at step 3"),
        CODE_READING["procedure"]: ("holds", None),
    })
    assert cert["verdict"] == "unverified"
    assert CODE_READING["procedure"] not in d.executed


def test_acceptance_mixed_failure_and_gap_settles_partial(tmp_path):
    """The chess shape end to end: an audited structural gap beside a
    construction failure settles the obligation unobtainable -> the
    claim partial, not unverified."""
    _, f, cert, _ = acceptance_pipeline(tmp_path, {
        SYSTEM_TEST["procedure"]: ("unobtainable", "no API key"),
        UNIT_TEST["procedure"]: ("failed", "harness crashed"),
        CODE_READING["procedure"]: ("unobtainable", "generated code, "
                                    "no stable spans to pin"),
    })
    assert cert["verdict"] == "partial"


def test_prompts_pin_refutation_soundness_gates():
    from verifier.llm import prompts
    assert "REFUTATION SOUNDNESS GATES" in prompts.JUDGER_SYSTEM
    assert "OPERATIONAL-DOMAIN WITNESS" in prompts.JUDGER_SYSTEM
    assert "REJECTION EQUIVALENCE" in prompts.JUDGER_SYSTEM
    assert "CANONICAL-RENDERING EQUALITY" in prompts.JUDGER_SYSTEM
    assert "SELF-CONSISTENT READING" in prompts.JUDGER_SYSTEM
    assert "NON-VACUOUS DOMAIN" in prompts.JUDGER_SYSTEM
    assert "NEGATIVE-METHOD SOUNDNESS" in prompts.ORACLES_SYSTEM
    assert "OPTIMALITY-REFUTATION PREFLIGHT" in prompts.ORACLES_SYSTEM
    assert "VERDICT-DIRECTION AT EXECUTION" in prompts.DISCHARGE_SYSTEM
