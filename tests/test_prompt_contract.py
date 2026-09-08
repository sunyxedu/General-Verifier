"""Pins of the prompt CONTRACT — which axioms each seat composes, which
variables each template renders, and the phrases the mechanisms rely on.
Wording pins beyond these are deliberately absent: the prompts are
principles, not a rule catalogue, and tests should not freeze prose."""
from __future__ import annotations

import re

from verifier.llm import prompts, agents


def _vars(template: str) -> set:
    return set(re.findall(r"{(\w+)}", template))


def test_axioms_are_composed_where_they_apply():
    A = prompts
    # the compiler and critic own the contract: necessity and sufficiency
    for system in (A.OBLIGATIONS_SYSTEM, A.CRITIC_SYSTEM):
        for axiom in (A.AXIOM_AUTHORITY, A.AXIOM_NECESSITY,
                      A.AXIOM_SUFFICIENCY, A.AXIOM_TIME, A.AXIOM_REFERENT):
            assert system.count(axiom) == 1
        assert A.AXIOM_WITNESS not in system      # evidence-side only
        assert A.AXIOM_STRENGTH not in system
    # the evidence seats own witnesses and strength
    for system in (A.ORACLES_SYSTEM, A.DISCHARGE_SYSTEM, A.JUDGER_SYSTEM):
        for axiom in (A.AXIOM_INDEPENDENCE, A.AXIOM_WITNESS,
                      A.AXIOM_TIME, A.AXIOM_STRENGTH):
            assert system.count(axiom) == 1
        assert A.AXIOM_NECESSITY not in system    # contract-side only
        assert A.AXIOM_SUFFICIENCY not in system
    # the compiler also carries independence (it phrases world floors)
    assert A.AXIOM_INDEPENDENCE in A.OBLIGATIONS_SYSTEM
    # the extractor judges nothing and carries no axiom
    assert not any(ax in A.EXTRACT_SYSTEM for ax in (
        A.AXIOM_AUTHORITY, A.AXIOM_WITNESS, A.AXIOM_STRENGTH))


def test_templates_render_exactly_the_variables_the_engine_supplies():
    assert _vars(prompts.OBLIGATIONS_TEMPLATE) == {
        "claim_text", "claim_kind", "contract_locus", "task_context",
        "ledger_path", "world_mode", "world_path", "world_map",
        "verified_deps", "pending_deps", "dependency_referent_context",
        "graph", "memory_hits", "critic_feedback"}
    assert _vars(prompts.CRITIC_TEMPLATE) == {
        "claim_text", "claim_kind", "contract_locus", "task_context",
        "ledger_path", "world_mode", "world_path", "world_map",
        "obligations_json", "verified_deps", "pending_deps",
        "dependency_referent_context", "game_history"}
    assert _vars(prompts.ORACLES_TEMPLATE) >= {
        "obligation_text", "required_basis", "severity", "clause",
        "refutable_by_absence", "task_context", "method_feedback",
        "world_map", "adapter_catalog"}
    assert _vars(prompts.DISCHARGE_TEMPLATE) >= {
        "obligation_text", "procedure", "promised_artifact",
        "declared_basis", "declared_strength", "bridges", "required_basis",
        "severity", "clause",
        "refutable_by_absence", "workdir", "toolchain_dir",
        "workspace_dir", "workspace_index", "world_map",
        "snapshot_per_file_cap", "redo_feedback", "adapter_method"}
    assert _vars(prompts.JUDGER_TEMPLATE) >= {
        "obligation_text", "severity", "clause", "refutable_by_absence",
        "procedure", "basis", "declared_strength", "bridges",
        "degradation_record", "evidence", "world_map", "adapter_method"}
    assert "{claim_text}" not in prompts.JUDGER_TEMPLATE   # blind judger
    assert _vars(prompts.EXTRACT_TEMPLATE) == {
        "task", "final_answer", "events_index", "ledger_path", "feedback"}
    for role in agents.ALL_AGENT_ROLES.values():
        role.template.format(**{n: "x" for n in _vars(role.template)})


def test_mechanism_fields_are_asked_for_by_name():
    # compiler emits the edge attributes the kernel folds on
    for field in ("severity: must | should", "clause:",
                  "refutable_by_absence:"):
        assert field in prompts.OBLIGATIONS_TEMPLATE
    assert "ACCEPTANCE COUNTERFACTUAL" in prompts.OBLIGATIONS_TEMPLATE
    # executor declares what the kernel gates on
    assert "witness_kind: for refuted" in prompts.DISCHARGE_TEMPLATE
    assert "inputs: every substantive source consumed" \
        in prompts.DISCHARGE_TEMPLATE
    assert "DECLARE YOUR INPUTS" in prompts.DISCHARGE_SYSTEM
    assert "DECLARE YOUR WITNESS KIND" in prompts.DISCHARGE_SYSTEM
    # judger audits both declarations
    assert "INPUT-PROVENANCE AUDIT" in prompts.JUDGER_SYSTEM
    assert "WITNESS-KIND AUDIT" in prompts.JUDGER_SYSTEM
    assert "PROVENANCE AUDIT:" in prompts.JUDGER_TEMPLATE
    assert "delivered_as_authority" in prompts.JUDGER_TEMPLATE
    # critic audits severity in both directions
    assert "SEVERITY IS PART OF THE DECOMPOSITION" in prompts.CRITIC_SYSTEM
    assert "SEVERITY AUDIT" in prompts.CRITIC_TEMPLATE
    # oracle: refutation methods promise a positive witness
    assert "REFUTATION METHODS PROMISE A WITNESS" in prompts.ORACLES_SYSTEM
    assert "INPUT PROVENANCE IS PART OF THE METHOD" in prompts.ORACLES_SYSTEM
    method_schema = agents.ENUMERATE_ORACLES_AGENT.schema["methods"][0]
    assert {"adapter?", "adapter_version?", "adapter_arguments?",
            "adapter_inputs?"} <= set(method_schema)
    assert "adapter_catalog" in _vars(prompts.ORACLES_TEMPLATE)


def test_adapter_preference_is_conditional_and_preserves_verification_scope():
    # Prompt contract guard only; model behavior still needs paired evals.
    text = prompts.ORACLES_TEMPLATE
    assert "equally sufficient method" in text
    assert "otherwise use the custom method" in text
    assert "preserve the obligation's full scope, provenance, absence rules" in text
    assert "retain every enumeration or observation the obligation actually requires" in text
    assert "omit all four fields unless" in text
    assert "never method sufficiency" in text


def test_game_and_locus_contract_survive():
    assert "{game_history}" in prompts.CRITIC_TEMPLATE
    assert "GAME CONSISTENCY RULE" in prompts.CRITIC_TEMPLATE
    assert "explicit retraction" in prompts.CRITIC_TEMPLATE
    assert "Never silently re-issue an overruled demand" \
        in prompts.CRITIC_TEMPLATE
    assert "{game_history}" in prompts.CRITIC_FEEDBACK_TEMPLATE
    assert "bound by, the same record" in prompts.CRITIC_FEEDBACK_TEMPLATE
    assert "{contract_locus}" in prompts.CRITIC_TEMPLATE
    assert "{contract_locus}" in prompts.OBLIGATIONS_TEMPLATE
    assert "Whole-task acceptance is adjudicated ONLY at a final claim" \
        in prompts.CONTRACT_LOCUS_NONFINAL
    assert "false sufficiency attack" in prompts.CONTRACT_LOCUS_NONFINAL
    assert "genuine necessity violation" in prompts.CONTRACT_LOCUS_NONFINAL
    assert "neither narrows nor expands" in prompts.CONTRACT_LOCUS_FINAL
    assert "A record floor is not a deficiency there" in prompts.CRITIC_SYSTEM
    assert "covers the action's occurrence only" in prompts.CRITIC_SYSTEM
    assert "APPEND OBLIGATION:" in prompts.CRITIC_SYSTEM
    assert "{violation}" in prompts.APPEND_REPAIR_GUARD_FEEDBACK


def test_strength_ladder_contract_survives():
    assert "STRENGTH LADDER" in prompts.ORACLES_SYSTEM
    assert "strength=analytic" in prompts.ORACLES_SYSTEM
    assert "FALLBACKS, not substitutes" in prompts.ORACLES_SYSTEM
    assert "strength: decisive | bounded | analytic" in prompts.ORACLES_TEMPLATE
    assert "STRENGTH IS PART OF SUFFICIENCY" in prompts.JUDGER_SYSTEM
    assert "DEGRADATION AUDIT" in prompts.JUDGER_SYSTEM
    assert "STRENGTH AUDIT" in prompts.JUDGER_TEMPLATE
    assert "OPTIMALITY-REFUTATION PREFLIGHT" in prompts.ORACLES_SYSTEM
    assert "VERDICT-DIRECTION AT EXECUTION" in prompts.DISCHARGE_SYSTEM


def test_plain_reading_gate_is_refutation_only():
    """The careful-reader gate downgrades refutations only: discharge owns
    it, the judger mirrors it, and it must not spread to the seats that
    compile or enumerate (where it would soften sufficiency)."""
    assert "careful human grader" in prompts.DISCHARGE_TEMPLATE.lower()
    assert "careful human grader" in prompts.JUDGER_SYSTEM.lower()
    assert "careful human grader" not in prompts.OBLIGATIONS_SYSTEM.lower()
    assert "careful human grader" not in prompts.ORACLES_SYSTEM.lower()
    assert "NEGATIVE ENTAILMENT AUDIT" in prompts.DISCHARGE_TEMPLATE
    assert "NEGATIVE ENTAILMENT AUDIT" in prompts.JUDGER_TEMPLATE
    assert "retained witness => exact obligation is false" \
        in prompts.JUDGER_TEMPLATE


def test_harness_addendum_is_generic():
    add = prompts.OBLIGATIONS_HARNESS_ADDENDUM
    assert "State the condition, not a preferred verification procedure" in add
    assert "declarative pointers to named documentation" in add
    assert "MINIMAL-CHANGE" in add
    assert "emit NO minimal-change" in add
