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
        "snapshot_per_file_cap", "snapshot_max_files",
        "snapshot_max_total_chars", "redo_feedback", "adapter_method"}
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
    assert "emit only the adapter-bound method" in text
    assert "binding that adapter is mandatory" in text
    assert "do not put an equivalent unbound executor first" in text
    assert "decision procedure or evidence coverage materially differs" in text
    assert "lists only retained guarantees" in text
    assert "never infer further guarantees" in text
    assert "must not claim original bytes, complete text, listings" in text
    assert "sufficiency still depends on the same" in text
    assert "different available evidence" in text
    assert "independently validated replacement" in text
    assert "otherwise return methods=[]" in text


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
    assert "E→O stays record-floor" in prompts.CRITIC_SYSTEM
    assert "record decides occurrence, not reported content's truth" \
        in prompts.CRITIC_SYSTEM
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
    assert "IRREPLACEABLE-PREREQUISITE SHORT-CIRCUIT" \
        in prompts.DISCHARGE_SYSTEM
    assert "at most one bounded direct obtain attempt" \
        in prompts.DISCHARGE_SYSTEM
    assert "Install only a local instrument explicitly named or permitted" \
        in prompts.DISCHARGE_SYSTEM
    assert "missing physical device, external endpoint, credential" \
        in prompts.DISCHARGE_SYSTEM
    assert "terminal result ends this exact method" \
        in prompts.DISCHARGE_SYSTEM
    assert "authoritative fixed-boundary capability facts" \
        in prompts.DISCHARGE_SYSTEM
    assert "attempt failure alone never proves a permanent gap" \
        in prompts.DISCHARGE_SYSTEM
    assert "do not parse, render, install, or inspect downstream material" \
        in prompts.DISCHARGE_SYSTEM
    assert "LARGE-INPUT DISCIPLINE" in prompts.DISCHARGE_TEMPLATE
    assert "Never print or read an entire large source" \
        in prompts.DISCHARGE_TEMPLATE
    assert "rerun it from fixed inputs" in prompts.DISCHARGE_TEMPLATE


def test_semantic_methods_are_operational_and_target_is_comparison_only():
    assert "OPERATIONAL-PRIMITIVE AUDIT" in prompts.ORACLES_SYSTEM
    assert "DERIVED-PERCEPTION BOTTLENECK" in prompts.ORACLES_SYSTEM
    assert "SEMANTIC-SIGNATURE REPAIR" in prompts.ORACLES_TEMPLATE
    assert "otherwise return methods=[]" in prompts.ORACLES_TEMPLATE

    assert "TARGET-TAINT BOUNDARY" in prompts.DISCHARGE_SYSTEM
    assert "report failed" in prompts.DISCHARGE_SYSTEM

    assert "CONCRETE-STRONGER-ROUTE TEST" in prompts.JUDGER_SYSTEM
    assert "TARGET-TAINT AUDIT" in prompts.JUDGER_SYSTEM
    assert "faithful=no" in prompts.JUDGER_SYSTEM


def test_oracle_compiles_methods_without_executing_their_evidence_checks():
    system = prompts.ORACLES_SYSTEM
    template = prompts.ORACLES_TEMPLATE
    assert "ORACLE COMPILES; THE ADAPTER/EXECUTOR OBSERVES" in system
    assert "compile that audit without performing it" in system
    assert "inspect the whole record before listing methods" not in system
    assert "must not determine the method's predicates or expected values" \
        in system
    assert "omit the adapter binding" in system
    assert "a repair must never guess a new version" in template
    assert "holds-only adapter replaces only" in template
    assert "never refuted" in template


def test_oracle_adapter_dispatch_is_basis_and_primitive_driven():
    template = prompts.ORACLES_TEMPLATE
    assert "TYPED-ADAPTER DISPATCH ORDER" in template
    assert "record.query first" in template
    assert "event-at, exists, paired-exchange" in template
    assert "record.program only for bounded pure computation" in template
    assert "filesystem, document, and evidence-program families" in template
    assert "never by task name, domain, expected outcome" in template
    assert "does not cover the whole obligation" in template
    assert "DIRECT EFFECT-OUTPUT RECIPE" in template
    assert "record.query@1 event-at" in template
    assert "/excerpt_truncated equals false" in template
    assert "Any producing verb/noun or described subject" in template
    assert "paired-exchange with exact result seq=N" in template
    assert "full correlated call, never event-at" in template
    assert "already-selected call is fresh Judger evidence audit" in template
    assert "not outside-world truth" in template


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
    assert "CANDIDATE-CARRIER NORMAL FORM" in add
    assert "Never compare with a value copied from the final claim" in add
    assert "the response is independently selected as a carrier" \
        in prompts.CRITIC_SYSTEM


def test_compact_profile_requires_one_pass_completion_coverage_and_repairs():
    compiler = prompts.COMPACT_COMPLETENESS_COMPILER_ADDENDUM
    critic = prompts.COMPACT_COMPLETENESS_CRITIC_ADDENDUM

    assert "INITIAL COMPLETION COVERAGE LEDGER" in compiler
    assert "critic feedback is empty" in compiler
    assert "independently falsifiable semantic and process conditions" in compiler
    assert "apply the ACCEPTANCE COUNTERFACTUAL separately" in compiler
    assert "no acceptance condition is unmapped" in compiler
    assert "Emit only the existing obligations schema" in compiler
    assert "GOVERNING-SCOPE DISTRIBUTION" in compiler
    assert "subject, quantifier, modality" in compiler
    assert "Reading a child span in isolation never cancels inherited" in compiler
    assert "only after the acceptance counterfactual" in compiler
    assert "cannot turn runner protocol into a candidate requirement" in compiler
    assert "RUNNER-INTERFACE QUOTIENT" in compiler
    assert "CAPTURE-PRESERVING NON-CIRCULARITY GATE" in compiler
    assert "successfully captures and hands off that candidate" in compiler
    assert "despite the challenged protocol violation" in compiler
    assert "task-owned content, form, provenance, or non-runner-history" \
        in compiler
    assert "task-owned acceptance consequence" in compiler
    assert "source says it is mandatory" in compiler
    assert "not task-owned merely because the runner records it" in compiler
    assert "delivered or independently evaluated object" in compiler
    assert "do not convert it to a SHOULD or a narrower" in compiler
    assert "PRE-CRITIC MATERIAL NORMAL FORM" in compiler
    assert compiler.index("ZERO-ORDER RUNNER-INTERFACE QUOTIENT") \
        < compiler.index("INITIAL COMPLETION COVERAGE LEDGER")
    assert "term, floor, severity, absence" in compiler
    assert "Remove only exact semantic duplicates" in compiler
    assert "never merge, weaken, or omit independent conditions" in compiler

    assert "COMPLETE DEFECT CENSUS" in critic
    assert critic.index("ZERO-ORDER RUNNER-INTERFACE QUOTIENT") \
        < critic.index("COMPLETE DEFECT CENSUS")
    assert "two source-order sweeps" in critic
    assert "Do not stop at the first witness" in critic
    assert "every independently visible missing and overstrong condition" in critic
    assert "do not defer a visible defect to a later round" in critic
    assert "one `APPEND OBLIGATION:` prefix" in critic
    assert "use `REWRITE BOTH:`" in critic
    assert "MATERIAL DEFECT GATE" in critic
    assert "RUNNER-INTERFACE QUOTIENT" in critic
    assert "CAPTURE-PRESERVING NON-CIRCULARITY GATE" in critic
    assert "before searching for missing conditions" in critic
    assert "apply this as a NECESSITY attack" in critic
    assert "successfully captures and hands off that candidate" in critic
    assert "despite the challenged protocol violation" in critic
    assert "task-owned content, form, provenance, or non-runner-history" \
        in critic
    assert "task-owned acceptance consequence" in critic
    assert "source says it is mandatory" in critic
    assert "not task-owned merely because the runner records it" in critic
    assert "require removal in this same round" in critic
    assert "never append, refine, downgrade to SHOULD, or narrow" in critic
    assert "different wording, ordering, or granularity" in critic
    assert "do not demand a duplicate obligation" in critic
    assert "never excuses a surviving necessity or sufficiency" in critic
    assert "COMPACT AUDIT REPORT" in critic
    assert "for output formatting only" in critic
    assert "supersedes the CRITIC_TEMPLATE" in critic
    assert "does not supersede performing those audits" in critic
    assert "passed gates need no prose" in critic
    assert "Brevity never licenses" in critic


def test_compiler_separates_runner_protocol_from_task_acceptance():
    prompt = prompts.OBLIGATIONS_SYSTEM

    assert "RUNNER-INTERFACE QUOTIENT" in prompt
    assert "CAPTURE-PRESERVING NON-CIRCULARITY GATE" in prompt
    assert "successful runner capture and handoff" in prompt
    assert "source says it is mandatory" in prompt
    assert "Remove failures before every later audit" in prompt
    assert "not even a SHOULD or narrower" in prompt
    assert "semantic function, never by tool name" in prompt
    assert "never infer `the task is incomplete, invalid, or rejected`" \
        in prompt
    assert "Response cadence, invocation/result handshakes" in prompt
    assert "bans on work after submission are runner interface" in prompt
    assert prompts.OBLIGATIONS_TEMPLATE.index(
        "ZERO-ORDER RUNNER-INTERFACE QUOTIENT") \
        < prompts.OBLIGATIONS_TEMPLATE.index("MANDATORY DEFINITION CHECK")


def test_critic_attacks_both_sides_of_runner_boundary():
    prompt = prompts.CRITIC_SYSTEM

    assert "RUNNER-INTERFACE QUOTIENT" in prompt
    assert "CAPTURE-PRESERVING NON-CIRCULARITY ATTACK" in prompt
    assert "apply a NECESSITY attack" in prompt
    assert "successful runner capture and handoff" in prompt
    assert "It survives only if both a non-circular task-owned property" \
        in prompt
    assert "require removal in this same round" in prompt
    assert "never append, refine, downgrade to SHOULD, or narrow" in prompt
    assert "Imperatives, `must`, headings, tool names" in prompt
    assert "never infer `the task is incomplete, invalid, or rejected`" \
        in prompt
    assert "Response cadence, invocation/result handshakes" in prompt
    assert prompts.CRITIC_TEMPLATE.index(
        "ZERO-ORDER RUNNER-INTERFACE QUOTIENT") \
        < prompts.CRITIC_TEMPLATE.index("GAME CONSISTENCY RULE")
    assert "bind you only where they survive that quotient" \
        in prompts.CRITIC_TEMPLATE


def test_compiler_and_critic_freeze_past_report_intension():
    compiler = prompts.OBLIGATIONS_SYSTEM + prompts.OBLIGATIONS_TEMPLATE
    critic = prompts.CRITIC_SYSTEM

    for prompt in (compiler, critic):
        assert "E→O" in prompt
        assert "P@time" in prompt
    assert "HISTORICAL EVENT-CONTENT GATE" in compiler
    assert "trusted ledger's direct event predicate" in compiler
    assert "Missing record availability is a downstream evidence gap" \
        in compiler
    assert "HISTORICAL EVENT-CONTENT INVARIANT" in critic
    assert "not another reporter" in critic
    assert "downstream evidence availability" in critic
    assert "adding accuracy or an external property" in critic
