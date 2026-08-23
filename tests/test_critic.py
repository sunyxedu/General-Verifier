"""The decomposition critic: adversarial second signature on the compiler.

Fixture-backed and instrument-free — the point under test is the loop: a
pass admits the decomposition, an insufficient verdict sends it back for
repair with the attack as feedback, and a counterexample still standing
after the last round fails closed (claim stays unverified).
"""
from __future__ import annotations

import json
import os

from verifier.core.events import EventLog
from verifier.core.model import Claim, CLAIM_ASSERTION, CLAIM_FINAL
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm import prompts
from verifier.llm.agents import CRITIC_AGENT
from verifier.llm.schema import FixtureBackend


COMPILE = lambda v: {"obligations": [{"text": f"O of: {v['claim_text']}", "required_basis": "record"}]}  # noqa: E731

PASS = {"verdict": "pass",
        "counterexample_world": "n/a", "missing_condition": "n/a"}
ATTACK = {"verdict": "insufficient",
          "counterexample_world": "P holds for 2, 3 and 5 but fails at 7",
          "missing_condition": "P must hold for every prime, not a sample"}


class StubDischarger:
    def __init__(self):
        self.seen: list[str] = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.seen.append(obligation.text)
        return DischargeResult(outcome="holds", witness=None,
                               artifact="", instrument="stub", basis="world")


def build(tmp_path, roles):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend(roles)
    discharger = StubDischarger()
    critic = CRITIC_AGENT if "critic" in roles else None
    Kernel(store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
           critic_role=critic, discharger=discharger)
    return store, log, backend, discharger


def ledger_events(log, etype):
    return [e for e in log.replay() if e["type"] == etype]


def test_pass_on_the_first_round_admits_the_decomposition(tmp_path):
    store, log, backend, discharger = build(
        tmp_path, {"obligations": COMPILE, "critic": PASS})

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the tree is clean"))

    assert store.status(cid) == "verified"
    assert [c["role"] for c in backend.calls] == ["obligations", "critic"]
    assert discharger.seen == ["O of: the tree is clean"]
    assert backend.calls[0]["variables"]["task_context"] == "(none supplied)"
    # the critic saw the claim, its kind, the task context, the obligations,
    # and the premises — never the solver's argument
    critic_vars = backend.calls[1]["variables"]
    assert set(critic_vars) == {
        "claim_text", "claim_kind", "task_context", "ledger_path",
        "world_mode", "world_path", "obligations_json", "verified_deps",
    }
    assert critic_vars["claim_kind"] == "assertion"
    assert critic_vars["task_context"] == "(none supplied)"
    # floors ride along so the critic attacks the decomposition AS FLOORED
    assert json.loads(critic_vars["obligations_json"]) == [
        {"text": "O of: the tree is clean", "required_basis": "record"}]


def test_a_standing_attack_forces_one_repair_with_the_attack_as_feedback(tmp_path):
    verdicts = iter([ATTACK, PASS])
    store, log, backend, _ = build(
        tmp_path, {"obligations": COMPILE, "critic": lambda v: next(verdicts)})

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "P holds for every prime"))

    assert store.status(cid) == "verified"
    assert [c["role"] for c in backend.calls] == [
        "obligations", "critic", "obligations", "critic"]
    # the repair round carries the counterexample back to the compiler
    feedback = backend.calls[2]["variables"]["critic_feedback"]
    assert ATTACK["counterexample_world"] in feedback
    assert ATTACK["missing_condition"] in feedback
    assert backend.calls[0]["variables"]["critic_feedback"] == ""


def test_a_counterexample_standing_after_the_last_round_fails_closed(tmp_path):
    store, log, backend, discharger = build(
        tmp_path, {"obligations": COMPILE, "critic": ATTACK})

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "P holds for every prime"))

    assert store.status(cid) == "unverified"
    assert discharger.seen == []          # nothing was ever discharged
    ev = store.evidence_for(cid)[-1]
    assert ev["error"] == "decomposition failed critic"
    assert ev["counterexample_world"] == ATTACK["counterexample_world"]
    attacks = ledger_events(log, "decomposition_attacked")
    assert [a["payload"]["round"] for a in attacks] == [0, 1]
    assert all(a["payload"]["verdict"] == "insufficient" for a in attacks)
    # exactly one repair was attempted; the post-deadline repair never ran
    assert [c["role"] for c in backend.calls] == [
        "obligations", "critic", "obligations", "critic"]


def test_critic_gets_the_task_statement_when_the_harness_wrote_one(tmp_path):
    """Compiler and critic get task truth inline, not as a skippable path."""
    task_text = "Print ALL winning moves, one per line."
    (tmp_path / "task.md").write_text(task_text)
    store, log, backend, _ = build(
        tmp_path, {"obligations": COMPILE, "critic": PASS})

    store.add_claim(Claim(CLAIM_ASSERTION, "the best move is g2g4"))

    compiler_context = json.loads(backend.calls[0]["variables"]["task_context"])
    critic_context = json.loads(backend.calls[1]["variables"]["task_context"])
    assert compiler_context == critic_context
    assert compiler_context == {
        "path": str(tmp_path / "task.md"),
        "text": task_text,
    }


def test_harness_final_contract_is_isolated_from_solver_graph(tmp_path):
    """A verified solver definition cannot erase a task requirement."""
    (tmp_path / "task.md").write_text("Return the complete total.")
    store, _, backend, _ = build(
        tmp_path, {"obligations": COMPILE, "critic": PASS})

    premise = store.add_claim(Claim(
        CLAIM_ASSERTION, "the complete total means component A only"))
    store.add_claim(Claim(
        CLAIM_FINAL, "the complete total is 42", deps=(premise,)))

    compiler = [
        c["variables"] for c in backend.calls
        if c["role"] == "obligations"
        and c["variables"]["claim_kind"] == CLAIM_FINAL
    ][-1]
    critic = [
        c["variables"] for c in backend.calls
        if c["role"] == "critic"
        and c["variables"]["claim_kind"] == CLAIM_FINAL
    ][-1]

    assert json.loads(compiler["verified_deps"]) == []
    assert json.loads(compiler["pending_deps"]) == []
    assert json.loads(compiler["graph"]) == [{
        "note": "withheld: harness final contract is compiled "
                "independently from authoritative task context"
    }]
    assert json.loads(compiler["task_context"])["text"] \
        == "Return the complete total."
    assert json.loads(critic["verified_deps"]) == []


def test_prompts_gate_task_definitions_and_attack_scope_projection():
    """Pin the general contract behind the VVE regression without naming it."""
    assert "DEFINITION GATE" in prompts.OBLIGATIONS_SYSTEM
    assert (
        "NEVER define the controlling task or claim's terminology"
        in prompts.OBLIGATIONS_SYSTEM
    )
    assert (
        "preserve the controlling statement's source-relative term"
        in prompts.OBLIGATIONS_SYSTEM
    )
    assert "MANDATORY DEFINITION CHECK" in prompts.OBLIGATIONS_TEMPLATE
    assert "MANDATORY SCOPE-PRESERVATION ATTACK" in prompts.CRITIC_SYSTEM
    assert "proper subset" in prompts.CRITIC_SYSTEM
    assert "proper-subset/component/proxy world" in prompts.CRITIC_TEMPLATE
    assert "EVIDENCE METHOD IS NOT OBLIGATION CONTENT" \
        in prompts.OBLIGATIONS_SYSTEM
    assert "fresh recomputation, captured" in prompts.OBLIGATIONS_SYSTEM
    assert "State the condition, not a preferred verification procedure" \
        in prompts.OBLIGATIONS_HARNESS_ADDENDUM


def test_extractor_preserves_component_and_aggregate_labels():
    assert "Preserve the SEMANTIC LABEL" in prompts.EXTRACT_SYSTEM
    assert "component versus aggregate" in prompts.EXTRACT_SYSTEM
    assert "Never promote a labeled component or subtotal" \
        in prompts.EXTRACT_SYSTEM


def test_prompts_preserve_temporal_targets_across_every_semantic_seat():
    """Pin the no-replay contract where each role can accidentally undo it."""
    assert "Every obligation must be a world-level condition" \
        not in prompts.OBLIGATIONS_TEMPLATE
    assert "MANDATORY TEMPORAL CHECK" in prompts.OBLIGATIONS_TEMPLATE
    assert "required_basis=record; never replay" \
        in prompts.OBLIGATIONS_TEMPLATE
    assert "actual past world truth stays" in prompts.OBLIGATIONS_TEMPLATE
    assert "MANDATORY TEMPORAL ATTACK" in prompts.CRITIC_SYSTEM
    assert "record-floor paraphrase" in prompts.CRITIC_SYSTEM
    assert "TEMPORAL TARGET IS PART OF SUFFICIENCY" \
        in prompts.ORACLES_SYSTEM
    assert "Do NOT re-run the historical action" in prompts.DISCHARGE_SYSTEM
    assert "HISTORICAL EVIDENCE AVAILABILITY" in prompts.DISCHARGE_SYSTEM
    assert "permanently lacks" in prompts.DISCHARGE_SYSTEM
    assert "retryable execution failure" in prompts.DISCHARGE_TEMPLATE
    assert "Never re-execute the historical tool" in prompts.JUDGER_SYSTEM


def test_compiler_does_not_strengthen_historical_reports_into_world_truth():
    assert "REPORT CONTENT IS INTENSIONAL" in prompts.OBLIGATIONS_SYSTEM
    assert "correctly implements an external definition" \
        in prompts.OBLIGATIONS_SYSTEM
    assert "reported Y for S" in prompts.OBLIGATIONS_TEMPLATE
    assert "additional world claims" in prompts.OBLIGATIONS_TEMPLATE


def test_execution_roles_distinguish_targets_from_input_provenance():
    """Every seat that handles a method preserves the same trust boundary."""
    for role in (
        prompts.ORACLES_SYSTEM,
        prompts.DISCHARGE_SYSTEM,
        prompts.JUDGER_SYSTEM,
    ):
        assert "TARGETS ARE NOT INPUTS" in role
        assert "expected, accepted, or comparison target" in role
        assert "even if its literal text also appears in the obligation" in role
        assert "newly introduced by the method" in role
    assert "trusted check specification, not evidence" \
        in prompts.JUDGER_TEMPLATE


def test_semantic_roles_share_proof_carrying_effect_boundary():
    """Ledger location alone neither downgrades nor upgrades evidence."""
    for role in (
        prompts.OBLIGATIONS_SYSTEM,
        prompts.ORACLES_SYSTEM,
        prompts.DISCHARGE_SYSTEM,
        prompts.JUDGER_SYSTEM,
    ):
        assert prompts.PROOF_CARRYING_EFFECTS in role
        assert "ledger is a transport, not automatically" in role
        assert "individually complete, correlated tool call/result pairs" \
            in role
        assert "whole ledger for earlier definition/provenance links" in role
        assert "absence of a revision reduces later replayability" in role
        assert "Do not demand a revision, raw source snapshot" in role
        assert "benchmark gold, tests, or expected-output files" in role
        assert "does not upgrade the effect claim itself" in role
    for role in (
        prompts.ORACLES_SYSTEM,
        prompts.DISCHARGE_SYSTEM,
        prompts.JUDGER_SYSTEM,
    ):
        assert "proof-carrying effect" in role
        assert "not a bare transcription" in role \
            or "not a transcription" in role


def test_oracle_cannot_hide_missing_evidence_behind_an_empty_list():
    assert "only when no method could in principle establish" \
        in prompts.ORACLES_TEMPLATE
    assert "list it; the executor must inspect the boundary" \
        in prompts.ORACLES_TEMPLATE
    assert "report the exact unobtainable gap" in prompts.ORACLES_TEMPLATE


def test_oracle_gets_one_bounded_structural_repair_prompt():
    assert "Prior method feedback" in prompts.ORACLES_TEMPLATE
    assert "one bounded repair pass" in prompts.ORACLES_TEMPLATE
    assert "return only materially new methods" in prompts.ORACLES_TEMPLATE
    assert "Do not repeat a rejected procedure" in prompts.ORACLES_TEMPLATE
    assert "diagnostic, NOT exhaustive" in prompts.ORACLES_TEMPLATE
    assert "Do not merely patch the stated rejection reason" \
        in prompts.ORACLES_TEMPLATE
    assert "every object, axis, sibling component, filter, quantifier" \
        in prompts.ORACLES_TEMPLATE


def test_oracle_methods_are_decisions_bound_to_task_semantics():
    """A target literal cannot turn Oracle enumeration into confirmation."""
    assert "DECISION PROCEDURE, not a confirmation recipe" \
        in prompts.ORACLES_SYSTEM
    assert "either holds or refuted" in prompts.ORACLES_SYSTEM
    assert "retained evidence appears to contradict" in prompts.ORACLES_SYSTEM
    assert "SEMANTIC BINDING GATE" in prompts.ORACLES_SYSTEM
    assert "field/component/proxy may narrow" in prompts.ORACLES_SYSTEM
    assert "proper component or proxy is insufficient" \
        in prompts.ORACLES_SYSTEM
    assert "SEMANTIC AXES MATTER" in prompts.ORACLES_SYSTEM
    assert "every same-axis sibling component" in prompts.ORACLES_SYSTEM
    assert "source field is not a competing sibling" \
        in prompts.ORACLES_SYSTEM
    assert "classification/category/domain condition does not also" \
        in prompts.ORACLES_SYSTEM
    assert "does not waive provenance for the classified data itself" \
        in prompts.ORACLES_SYSTEM
    assert "FAMILY-COVERAGE GATE" in prompts.ORACLES_SYSTEM
    assert "disjoint-additive" in prompts.ORACLES_SYSTEM
    assert "overlapping/alias/cached/materialized" in prompts.ORACLES_SYSTEM
    assert "derived aggregate" in prompts.ORACLES_SYSTEM
    assert "orthogonal metadata" in prompts.ORACLES_SYSTEM
    assert "relation remains unresolved" in prompts.ORACLES_SYSTEM
    assert "task-authoritative representation and population" \
        in prompts.ORACLES_SYSTEM
    assert "caches, subsets, and materialized views" \
        in prompts.ORACLES_SYSTEM
    assert "Merely reading the ledger does not determine basis" \
        in prompts.ORACLES_SYSTEM


def test_offline_oracle_prefers_complete_retained_evidence():
    assert "TARGET-BLIND SEMANTIC COVERAGE" in prompts.ORACLES_TEMPLATE
    assert "Every procedure string MUST contain" in prompts.ORACLES_TEMPLATE
    assert "SEMANTIC COVERAGE: term=" in prompts.ORACLES_TEMPLATE
    assert "artifact string MUST" in prompts.ORACLES_TEMPLATE
    assert "a target match is never exclusion evidence" \
        in prompts.ORACLES_TEMPLATE
    assert "realized coverage report sufficient for the stated verdict" \
        in prompts.ORACLES_TEMPLATE
    assert "plausibly in-scope same-axis component type" \
        in prompts.ORACLES_TEMPLATE
    assert "monotonicity/disjointness" in prompts.ORACLES_TEMPLATE
    assert "SEMANTIC COVERAGE: not-applicable" in prompts.ORACLES_TEMPLATE
    assert "MANDATORY RETAINED-EVIDENCE PREFLIGHT" in prompts.ORACLES_SYSTEM
    assert "establish or refute the obligation" in prompts.ORACLES_SYSTEM
    assert "before any method requiring fresh retrieval or replay" \
        in prompts.ORACLES_SYSTEM
    assert "every task-relevant component and aggregate" \
        in prompts.ORACLES_SYSTEM
    assert "never cherry-pick a target-matching component or subtotal" \
        in prompts.ORACLES_SYSTEM
    assert "VERDICT-DIRECTION SUFFICIENCY" in prompts.ORACLES_SYSTEM
    assert "exact equality negatively with a rigorous monotone bound" \
        in prompts.ORACLES_SYSTEM
    assert "one in-scope counterexample" in prompts.ORACLES_SYSTEM


def test_judger_independently_countersigns_method_sufficiency():
    assert "independent second signer for METHOD SUFFICIENCY" \
        in prompts.JUDGER_SYSTEM
    assert "untrusted candidate method" in prompts.JUDGER_SYSTEM
    assert "report ALL independently visible scope defects" \
        in prompts.JUDGER_SYSTEM
    assert "nonzero same-axis sibling or contradictory aggregate" \
        in prompts.JUDGER_SYSTEM
    assert "do not demand an additional row-level provenance/source join" \
        in prompts.JUDGER_SYSTEM
    assert "SEMANTIC COVERAGE IS NOT SELF-AUTHENTICATING" \
        in prompts.JUDGER_SYSTEM
    assert "alternative aggregate found in raw evidence" \
        in prompts.JUDGER_SYSTEM
    assert "Never return both yes" in prompts.JUDGER_SYSTEM
    assert "COVERAGE AUDIT: term=" in prompts.JUDGER_TEMPLATE
    assert "contract_complete=yes|no" in prompts.JUDGER_TEMPLATE
    assert "execution_complete=yes|no" in prompts.JUDGER_TEMPLATE
    assert "all relevant fields" in prompts.JUDGER_TEMPLATE
    assert "complete for the stated verdict direction" \
        in prompts.JUDGER_TEMPLATE
    assert "COVERAGE AUDIT: applicable=no" in prompts.JUDGER_TEMPLATE
    assert "component, subtotal, sample" in prompts.JUDGER_SYSTEM
    assert "authoritative equivalence or exclusion" in prompts.JUDGER_SYSTEM
    assert "same-axis sibling component" in prompts.JUDGER_SYSTEM
    assert "provenance, identifier, or source fields do not compete" \
        in prompts.JUDGER_SYSTEM
    assert "monotone bound may be sufficient to refute" \
        in prompts.JUDGER_SYSTEM
    assert "a bad run does not make an otherwise sufficient" \
        in prompts.JUDGER_SYSTEM
    assert "If they do not, faithful=no" in prompts.JUDGER_SYSTEM
    assert "writing 'thus' or 'means' does not supply" \
        in prompts.JUDGER_SYSTEM
    assert "sufficient=no" in prompts.JUDGER_SYSTEM
    assert "sufficient: yes | no" in prompts.JUDGER_TEMPLATE
    assert "untrusted candidate check" in prompts.JUDGER_TEMPLATE


def test_compiler_and_critic_preserve_task_term_modifiers():
    assert "Preserve every noun modifier, namespace, producer qualifier" \
        in prompts.OBLIGATIONS_SYSTEM
    assert "Do not broaden `X items` into `items`" \
        in prompts.OBLIGATIONS_SYSTEM
    assert "noun modifier/namespace/producer qualifier" \
        in prompts.OBLIGATIONS_TEMPLATE
    assert "without an authority-backed equivalence or expansion" \
        in prompts.CRITIC_SYSTEM


def test_judger_uses_artifact_store_content_address_semantics():
    assert "ARTIFACT CONTENT ADDRESSING" in prompts.JUDGER_SYSTEM
    assert "content address of the PARSED JSON value" in prompts.JUDGER_SYSTEM
    assert "pretty-printed file's raw-byte" in prompts.JUDGER_SYSTEM
    assert "canonical parsed-JSON content address" in prompts.JUDGER_TEMPLATE
    assert "not the raw-file SHA-256" in prompts.JUDGER_TEMPLATE


def test_compiler_and_critic_receive_explicit_world_evidence_mode(tmp_path):
    store, _, backend, _ = build(
        tmp_path, {"obligations": COMPILE, "critic": PASS})

    store.add_claim(Claim(CLAIM_ASSERTION, "the historical report exists"))

    compiler = backend.calls[0]["variables"]
    critic = backend.calls[1]["variables"]
    assert compiler["world_mode"] == critic["world_mode"] == "live"
    assert compiler["world_path"] == critic["world_path"] == ""
    assert compiler["ledger_path"] == critic["ledger_path"]


def test_offline_without_capture_is_not_mislabeled_live(tmp_path):
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": COMPILE, "critic": PASS})
    Kernel(
        store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
        critic_role=CRITIC_AGENT, discharger=StubDischarger(),
        live_world=False,
    )

    store.add_claim(Claim(CLAIM_ASSERTION, "the historical report exists"))

    assert backend.calls[0]["variables"]["world_mode"] == "unavailable"
    assert backend.calls[1]["variables"]["world_mode"] == "unavailable"


def test_task_defined_aggregate_projection_is_repaired(tmp_path):
    """An abstract A+B task exercises the general repair path, not one task."""
    task_text = "Under the named specification, TOTAL is component A plus B."
    (tmp_path / "task.md").write_text(task_text)

    def compile_aggregate(variables):
        if not variables["critic_feedback"]:
            return {"obligations": [{
                "text": "Component A alone equals N.",
                "required_basis": "world",
            }]}
        context = json.loads(variables["task_context"])
        assert context["text"] == task_text
        return {"obligations": [{
            "text": "The independently established sum of A and B equals N.",
            "required_basis": "world",
        }]}

    def attack_component(variables):
        (obligation,) = json.loads(variables["obligations_json"])
        if "A alone" in obligation["text"]:
            return {
                "verdict": "insufficient",
                "counterexample_world": (
                    "A equals N and B is nonzero, so the obligation holds "
                    "while the task-defined TOTAL does not equal N."
                ),
                "missing_condition": "The obligation must cover A plus B.",
            }
        return PASS

    store, _, backend, discharger = build(
        tmp_path,
        {"obligations": compile_aggregate, "critic": attack_component},
    )

    cid = store.add_claim(Claim(CLAIM_FINAL, "The task-defined TOTAL is N."))

    assert store.status(cid) == "verified"
    assert discharger.seen == [
        "The independently established sum of A and B equals N."
    ]
    assert [call["role"] for call in backend.calls] == [
        "obligations", "critic", "obligations", "critic"
    ]


def test_disabled_critic_restores_the_old_behavior(tmp_path):
    store, log, backend, _ = build(tmp_path, {"obligations": COMPILE})

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the tree is clean"))

    assert store.status(cid) == "verified"
    assert [c["role"] for c in backend.calls] == ["obligations"]
    assert ledger_events(log, "decomposition_attacked") == []
