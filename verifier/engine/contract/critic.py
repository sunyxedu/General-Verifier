"""The critic game: the adversarial second signature on a decomposition.

The critic tries to construct a world where every obligation and every
verified premise holds and the claim is still false, or the claim holds
and an obligation fails. A standing attack sends the decomposition back to
the compiler for one repair; one still standing after the budget fails the
claim closed. Both seats see the game's record of earlier rounds, so the
critic is bound by its own prior rulings (memoryless rounds oscillated).
For a non-final claim under an advisory budget the last repair stands
unreviewed instead: whole-task acceptance is adjudicated at the final.
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Callable, Dict, List, Optional

from ...core.model import BASIS_RECORD, CLAIM_FINAL, OUTCOME_FAILED, SEVERITY_MUST
from ...llm import prompts
from ...llm.model_spans import model_scope
from ..source_evidence import bind_source_evidence_variables


def _repair_oscillation(history: List[Dict]) -> bool:
    """Best-effort cross-round contradiction heuristic over the critic's
    mechanical repair labels: one round demanded removal while another
    demanded (re)addition. Exact-state revisits cannot be relied on (a
    repair usually also rewords surviving obligations), so this reads the
    labels the protocol already mandates. Triage metadata only — it never
    changes the fail-closed verdict."""
    seen: set = set()
    for entry in history:
        mc = entry.get("missing_condition") or ""
        kinds = set()
        if "REMOVE OR WEAKEN:" in mc:
            kinds.add("remove")
        if "APPEND OBLIGATION:" in mc or "ADD:" in mc:
            kinds.add("add")
        if ("remove" in seen and "add" in kinds) \
                or ("add" in seen and "remove" in kinds):
            return True
        seen |= kinds
    return False



class CriticGame:
    """One claim's review: the seats, the budgets, and where the game's
    events and fail-closed evidence go. ``conclude_unverified(claim_id,
    reason)`` fails a claim closed (None: nothing to conclude, as for a
    precompile that critiques into a buffer)."""

    def __init__(self, *, log, store, backend, obligations_role, critic_role,
                 critic_rounds: int, run_options,
                 critic_rounds_nonfinal: Optional[int] = None,
                 unify_final: bool = False,
                 append_only_deltas: bool = False,
                 conclude_unverified: Optional[Callable] = None):
        self.log = log
        self.store = store
        self.backend = backend
        self.obligations_role = obligations_role
        self.critic_role = critic_role
        self.critic_rounds = critic_rounds
        self.run_options = run_options
        self.critic_rounds_nonfinal = critic_rounds_nonfinal
        self.unify_final = unify_final
        self.append_only_deltas = append_only_deltas
        self.conclude_unverified = conclude_unverified or (lambda *a: False)

    def review(self, row: Dict, payload: Dict, variables: Dict, *,
                  backend=None, obligations_role=None, critic_role=None,
                  record: bool = True):
        """
        Adversarial second signature on the decomposition: the critic
        tries to construct a world where every obligation and every verified
        premise holds and the claim is still false. It never sees the
        compiler's reasoning. Only the claim, the obligations, the
        verified premises, and the game record of its own prior rounds.
        An attack that stands sends the decomposition back for repair;
        one still standing after critic_rounds fails closed (returns
        None, claim stays unverified).
        """
        backend = self.backend if backend is None else backend
        obligations_role = obligations_role or self.obligations_role
        critic_role = critic_role or self.critic_role
        if critic_role is None:
            return payload
        # with finals handled like any claim (unify_final) the critic's
        # budget and advisory review apply to the final and its precompile
        # too: the review no longer holds a final's check back
        nonfinal = self.critic_rounds_nonfinal
        advisory = nonfinal is not None and (
            row["kind"] != CLAIM_FINAL or self.unify_final)
        rounds = (nonfinal if advisory
                  else self.critic_rounds)
        if rounds <= 0:
            return payload
        # The compiler and critic receive the same inlined harness truth.
        # A path-only pointer let a model skip the Read and then infer task
        # semantics from a solver-authored graph node instead.
        task_context = variables["task_context"]
        verdict = None
        failure_counterexample = ""
        failure_missing_condition = ""
        history: List[Dict] = []  # reviewed rounds, shown to BOTH seats: the
        critic_calls = 0          # critic is bound by its own prior rulings
        repair_calls = 0          # (memoryless rounds oscillated: a demand
        while critic_calls < rounds:              # overruled in round n was
            round_no = critic_calls               # re-issued verbatim in n+2)
            with model_scope(stage="contract_critic", round=round_no):
                critic_variables = bind_source_evidence_variables(
                    self.log,
                    {
                    "claim_text": variables["claim_text"],
                    "claim_kind": variables["claim_kind"],
                    "contract_locus": variables["contract_locus"],
                    "task_context": task_context,
                    "ledger_path": variables.get(
                        "_mixed_ledger_path", variables["ledger_path"]),
                    "world_mode": variables["world_mode"],
                    "world_path": variables["world_path"],
                    "world_map": variables["world_map"],
                    # Audit the complete contract, including which
                    # violations refute and when absence is admissible.
                    "obligations_json": json.dumps(
                        payload["obligations"],
                        ensure_ascii=False
                    ),
                    "excluded_json": json.dumps(
                        payload.get("excluded", []), ensure_ascii=False),
                    "verified_deps": variables["verified_deps"],
                    "pending_deps": variables["pending_deps"],
                    "dependency_referent_context":
                        variables["dependency_referent_context"],
                    "game_history": json.dumps(history, ensure_ascii=False),
                    },
                    enabled=self.run_options.compact_inspection,
                )
                verdict = critic_role.call(
                    backend,
                    critic_variables,
                )
            self.log.append(
                "decomposition_attacked",
                {"claim": row["id"], "round": round_no,
                 "verdict": verdict["verdict"],
                 "counterexample_world": verdict["counterexample_world"],
                 "missing_condition": verdict["missing_condition"]},
            )
            history.append({
                "round": round_no,
                "reviewed_obligations": payload["obligations"],
                "verdict": verdict["verdict"],
                "counterexample_world": verdict["counterexample_world"],
                "missing_condition": verdict["missing_condition"],
            })
            critic_calls += 1
            if verdict["verdict"] == "pass":
                return payload
            failure_counterexample = verdict["counterexample_world"]
            failure_missing_condition = verdict["missing_condition"]
            if critic_calls == rounds:
                break  # last attack stands; a repair would go unreviewed
            append_delta = (
                self.append_only_deltas
                and verdict["missing_condition"].lstrip().startswith(
                    "APPEND OBLIGATION:"))
            feedback_template = (
                prompts.APPEND_DELTA_FEEDBACK_TEMPLATE
                if append_delta else prompts.CRITIC_FEEDBACK_TEMPLATE)
            semantic_feedback = feedback_template.format(
                previous_obligations=json.dumps(
                    payload["obligations"], ensure_ascii=False),
                game_history=json.dumps(history, ensure_ascii=False),
                counterexample_world=verdict["counterexample_world"],
                missing_condition=verdict["missing_condition"],
            )
            repair_accepted = False
            declined = False
            guard_feedback = ""
            while repair_calls < rounds - 1:
                repair_variables = dict(
                    variables,
                    critic_feedback=semantic_feedback + guard_feedback,
                )
                with model_scope(
                        stage="contract_compile", round=repair_calls + 1):
                    repair_role = obligations_role
                    if append_delta:
                        repair_role = replace(
                            obligations_role,
                            template=(obligations_role.template
                                      + prompts.APPEND_DELTA_OUTPUT_CONTRACT))
                    candidate = repair_role.call(
                        backend, repair_variables)
                repair_calls += 1
                if append_delta:
                    violation = self.append_delta_violation(
                        payload, candidate)
                    if violation is None:
                        candidate = self.merge_append_delta(
                            payload, candidate)
                else:
                    violation = None
                if (not append_delta
                        and verdict["missing_condition"].lstrip().startswith(
                            "APPEND OBLIGATION:")):
                    violation = self.append_repair_violation(payload, candidate)
                if violation is None:
                    payload = candidate
                    variables = repair_variables
                    repair_accepted = True
                    break
                if (not append_delta
                        and self.same_contract(payload, candidate)):
                    # The compiler returned the contract unchanged: a
                    # principled refusal of the demanded append (for example
                    # a clause its own gates classify as runner interface),
                    # not a malformed repair.  Burning the remaining repairs
                    # on the mechanical guard could only deadlock; hand the
                    # unchanged contract and the refusal back to the Critic,
                    # which still fails the claim closed if it stands.
                    declined = True
                    self.log.append(
                        "contract_repair_declined",
                        {"claim": row["id"], "round": round_no,
                         "repair_attempt": repair_calls,
                         "missing_condition": verdict["missing_condition"]},
                    )
                    break

                failure_counterexample = (
                    "MECHANICAL APPEND-ONLY VIOLATION: " + violation)
                failure_missing_condition = verdict["missing_condition"]
                self.log.append(
                    "decomposition_attacked",
                    {
                        "claim": row["id"],
                        "round": round_no,
                        "verdict": "insufficient",
                        "counterexample_world": failure_counterexample,
                        "missing_condition": failure_missing_condition,
                        "source": "mechanical_append_guard",
                        "repair_attempt": repair_calls,
                    },
                )
                guard_template = (
                    prompts.APPEND_DELTA_GUARD_FEEDBACK
                    if append_delta else prompts.APPEND_REPAIR_GUARD_FEEDBACK)
                guard_feedback = guard_template.format(
                    violation=violation,
                    rejected_obligations=json.dumps(
                        candidate["obligations"], ensure_ascii=False),
                )
            if declined:
                history[-1]["repair"] = (
                    "declined: the compiler returned the contract unchanged, "
                    "judging the demanded append outside task acceptance; "
                    "re-review the unchanged contract or name a different "
                    "deficiency")
                continue
            if not repair_accepted:
                break
        if advisory and verdict is not None \
                and verdict["verdict"] != "pass":
            # advisory review: the last standing attack becomes one
            # repair instruction the compiler answers, and that repair
            # stands unreviewed — a non-final claim is adjudicated on a
            # best-effort decomposition rather than failing closed
            repair_variables = dict(
                variables,
                critic_feedback=prompts.CRITIC_FEEDBACK_TEMPLATE.format(
                    previous_obligations=json.dumps(
                        payload["obligations"], ensure_ascii=False),
                    game_history=json.dumps(history, ensure_ascii=False),
                    counterexample_world=failure_counterexample,
                    missing_condition=failure_missing_condition))
            try:
                payload = self.obligations_role.call(
                    self.backend, repair_variables)
            except Exception as err:  # noqa: BLE001 - keep the reviewed one
                self.log.append(
                    "decomposition_attacked",
                    {"claim": row["id"], "round": critic_calls,
                     "verdict": "insufficient",
                     "counterexample_world": failure_counterexample,
                     "missing_condition": failure_missing_condition,
                     "resolution": "advisory: repair failed, the reviewed "
                                   f"decomposition stands ({str(err)[:120]})"})
                return payload
            self.log.append(
                "decomposition_attacked",
                {"claim": row["id"], "round": critic_calls,
                 "verdict": "insufficient",
                 "counterexample_world": failure_counterexample,
                 "missing_condition": failure_missing_condition,
                 "resolution": "advisory: unreviewed repair accepted for a "
                               "non-final claim"})
            return payload
        # triage metadata only — an oscillating critic still fails closed
        # (auto-passing on critic incoherence would be an incentive to
        # induce it), but the run's analysis can count verifier-internal
        # deadlocks apart from genuinely unverifiable decompositions
        reason = "decomposition failed critic"
        if _repair_oscillation(history):
            reason += " (oscillation)"
        if not record:
            return None   # a precompile has no store row to fail closed on
        self.store.record_evidence(
            row["id"],
            {
                "outcome": OUTCOME_FAILED,
                "error": reason,
                "counterexample_world": failure_counterexample,
                "missing_condition": failure_missing_condition,
            },
        )
        self.conclude_unverified(row["id"], reason)
        return None

    @staticmethod
    def same_contract(previous: Dict, candidate: Dict) -> bool:
        """Whether a repair returned exactly the previous obligation set."""
        def contract(item):
            return (item["text"], item.get("required_basis", BASIS_RECORD),
                    item.get("severity", SEVERITY_MUST),
                    item.get("clause", ""),
                    item.get("refutable_by_absence", False))

        return ([contract(item) for item in previous["obligations"]]
                == [contract(item) for item in candidate["obligations"]])

    @staticmethod
    def append_repair_violation(previous: Dict, candidate: Dict):
        """Return why an ``APPEND OBLIGATION:`` repair is not a tail append.

        The critic's label is a mechanical contract: every existing
        obligation contract remains unchanged and in the same order,
        and at least one new obligation is appended. Other repair labels retain the
        compiler's ordinary freedom and never call this helper.
        """
        def contract(item):
            return (item["text"], item.get("required_basis", BASIS_RECORD),
                    item.get("severity", SEVERITY_MUST),
                    item.get("clause", ""),
                    item.get("refutable_by_absence", False))

        old = [contract(item) for item in previous["obligations"]]
        new = [contract(item) for item in candidate["obligations"]]
        if len(new) <= len(old):
            return (
                f"expected more than {len(old)} obligations after append, "
                f"got {len(new)}")
        for index, expected in enumerate(old):
            actual = new[index]
            if actual != expected:
                return (
                    f"obligation {index} changed: expected {expected!r}, "
                    f"got {actual!r}")
        return None

    @staticmethod
    def append_delta_violation(previous: Dict, candidate: Dict):
        """Return why an additions-only response is not an admissible delta.

        Arbitrary new obligation text cannot reveal whether a model intended
        to edit an old item.  The sound mechanical guarantee is instead that
        no response can replace old bytes: they are deep-copied below.  This
        guard additionally rejects an empty delta and the unambiguous legacy
        mistake of echoing the complete old contract as its prefix.  It does
        not use fuzzy text or clause matching.
        """
        additions = candidate["obligations"]
        if not additions:
            return "the additions-only obligation array is empty"
        old = previous["obligations"]
        if old and len(additions) >= len(old) \
                and additions[:len(old)] == old:
            return (
                "the additions-only obligation array repeats the complete "
                "existing contract as a prefix")
        return None

    @staticmethod
    def merge_append_delta(previous: Dict, delta: Dict) -> Dict:
        """Construct a full contract without exposing old items to rewrite."""
        merged = {
            "obligations": (
                copy.deepcopy(previous["obligations"])
                + copy.deepcopy(delta["obligations"]))
        }
        excluded = (list(previous.get("excluded") or [])
                    + list(delta.get("excluded") or []))
        if excluded:
            merged["excluded"] = copy.deepcopy(excluded)
        return merged
