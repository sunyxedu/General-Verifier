"""The evidence axioms as gates: each rule a result must pass before it may
settle anything, defined once.

A gate answers one question about one discharge result — does it break
this axiom? — and says why in the words the ledger records. Models judge
meaning; these gates enforce types, whatever a seat argued:

  DecisiveRefutation  nothing below decisive strength refutes
  AdmissibleWitness   a refutation rests on a positive, retained
                      observation; absence refutes only a presence
                      requirement
  Independence        a holds never rests on a delivered artifact standing
                      as authority (the circular certificate)
  BasisFloor          evidence below the obligation's floor settles nothing,
                      in either direction
  PremiseScope        evidence assumes only the premises in force for the
                      claim (none for a completion claim)
  WorldUnchanged      a check leaves the delivered world as it found it

Two chains apply them. ``BACKSTOP`` runs in the kernel behind every
discharge (a custom binder, direct discharge, or a mislabel that slipped an
audit): each violated gate converts the result to failed and the later
gates see the converted result. ``PREFLIGHT`` runs in the binding game on
the executor's own declaration before any semantic audit spends a judger
call; it reports the first violation. The two chains phrase a violation
for their own reader and are recorded that way.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from ..core.model import (
    OUTCOME_FAILED, OUTCOME_HOLDS, OUTCOME_REFUTED, OUTCOME_UNOBTAINABLE,
    STRENGTH_DECISIVE, WITNESS_NONE,
    basis_satisfies, circular_inputs, strength_rank, witness_admissible,
)
from ..execution.discharge import DischargeResult


@dataclass(frozen=True)
class GateScope:
    """What a gate may need beyond the result and the obligation: the
    premises whose content the claim's evidence may assume, and whether
    the claim is a completion claim (whose contract admits none)."""
    premises_in_force: frozenset = frozenset()
    completion_claim: bool = False


def below_decisive(outcome: str, strength: Optional[str]) -> bool:
    """A refutation from an instrument below decisive strength. None reads
    as decisive (direct discharge has no ladder)."""
    return (outcome == OUTCOME_REFUTED
            and strength_rank(strength) < strength_rank(STRENGTH_DECISIVE))


def below_decisive_reason(strength: Optional[str], witness,
                          redo: Optional[int] = None) -> str:
    """Why a refutation below decisive strength settles nothing."""
    where = f"redo {redo}; " if redo is not None else ""
    return (f"refutation below decisive strength is inadmissible "
            f"({where}{strength} instrument reported: {witness})")


def admissible_refutation(witness, witness_kind,
                          refutable_by_absence: bool = False) -> bool:
    """A refutation carried by this witness may settle: one was delivered
    and its kind is admissible for the obligation."""
    return bool(witness) and witness_admissible(witness_kind,
                                                refutable_by_absence)


def transferable_refutation(witness_kind, strength, obligation) -> bool:
    """A settled refutation may carry over to this obligation: its witness
    kind is admissible here and it was won at decisive strength."""
    return (witness_admissible(witness_kind, obligation.refutable_by_absence)
            and strength_rank(strength) >= strength_rank(STRENGTH_DECISIVE))


class Gate:
    """One axiom. ``violated`` decides; ``backstop`` / ``preflight`` word
    the violation for their chain. ``keeps_declaration``: the failed result
    the backstop makes keeps the executor's witness kind and inputs for
    audit."""
    axiom = ""
    keeps_declaration = False

    def violated(self, result, obligation, scope: GateScope) -> bool:
        raise NotImplementedError

    def backstop(self, result, obligation, scope: GateScope) -> str:
        raise NotImplementedError

    def preflight(self, result, obligation, scope: GateScope) -> str:
        raise NotImplementedError


class DecisiveRefutation(Gate):
    axiom = "strength"

    def violated(self, result, obligation, scope):
        return below_decisive(result.outcome, result.strength)

    def backstop(self, result, obligation, scope):
        return below_decisive_reason(result.strength, result.witness)


class AdmissibleWitness(Gate):
    axiom = "witness"
    keeps_declaration = True

    def violated(self, result, obligation, scope):
        return (result.outcome == OUTCOME_REFUTED
                and not witness_admissible(result.witness_kind,
                                           obligation.refutable_by_absence))

    def backstop(self, result, obligation, scope):
        kind = result.witness_kind or WITNESS_NONE
        return (f"refutation witness of kind {kind!r} is "
                f"inadmissible for this obligation (absence and "
                f"silence never refute unless the obligation is a "
                f"presence requirement); reported: "
                f"{result.witness}")

    def preflight(self, result, obligation, scope):
        kind = result.witness_kind or WITNESS_NONE
        return (f"refutation witness of kind {kind!r} is inadmissible: "
                f"absence, silence, and undeclared witnesses never "
                f"refute unless the obligation is a presence "
                f"requirement (reported: {result.witness})")


class Independence(Gate):
    axiom = "independence"
    keeps_declaration = True

    def violated(self, result, obligation, scope):
        return (result.outcome == OUTCOME_HOLDS
                and bool(circular_inputs(result.inputs)))

    def backstop(self, result, obligation, scope):
        return (f"circular evidence: delivered artifact(s) "
                f"{circular_inputs(result.inputs)} declared as authority "
                f"for the check; "
                f"a positive verdict grounded on the solver's own "
                f"output establishes consistency, not truth")

    def preflight(self, result, obligation, scope):
        return (f"circular evidence: delivered artifact(s) "
                f"{circular_inputs(result.inputs)} declared as authority; "
                f"a holds "
                f"grounded on the solver's own output establishes "
                f"consistency, not truth")


class BasisFloor(Gate):
    axiom = "basis_floor"

    def violated(self, result, obligation, scope):
        return (result.outcome in (OUTCOME_HOLDS, OUTCOME_REFUTED)
                and not basis_satisfies(result.basis,
                                        obligation.required_basis))

    def backstop(self, result, obligation, scope):
        detail = (f" (reported counterexample: {result.witness})"
                  if result.outcome == OUTCOME_REFUTED
                  and result.witness else "")
        return (f"evidence basis {result.basis!r} below the "
                f"obligation's floor "
                f"{obligation.required_basis!r}{detail}")


class PremiseScope(Gate):
    axiom = "premise_scope"

    @staticmethod
    def _illegal(result, scope):
        return sorted(set(result.assumed_premises) - scope.premises_in_force)

    def violated(self, result, obligation, scope):
        return (result.outcome in (OUTCOME_HOLDS, OUTCOME_REFUTED,
                                   OUTCOME_UNOBTAINABLE)
                and bool(result.assumed_premises)
                and bool(self._illegal(result, scope)))

    def backstop(self, result, obligation, scope):
        why = ("a completion claim's contract admits no assumed "
               "premises" if scope.completion_claim
               else "assumed premises outside the claim's "
                    "dependency closure")
        return (f"{why}: {self._illegal(result, scope)} "
                f"(reported {result.outcome}: "
                f"{result.witness})")


class WorldUnchanged(Gate):
    axiom = "world_unchanged"

    def violated(self, result, obligation, scope):
        return bool(getattr(result, "world_changes", None))

    def preflight(self, result, obligation, scope):
        return ("check altered the delivered world: "
                + json.dumps(result.world_changes, ensure_ascii=False)[:400]
                + " — a check must leave the delivery as it found it "
                "(exercise mutable behaviour on a copy, or restore the "
                "state it changed before finishing)")


BACKSTOP: Tuple[Gate, ...] = (DecisiveRefutation(), AdmissibleWitness(),
                              Independence(), BasisFloor(), PremiseScope())
PREFLIGHT: Tuple[Gate, ...] = (WorldUnchanged(), AdmissibleWitness(),
                               Independence())


def apply_backstop(result: DischargeResult, obligation, scope: GateScope,
                   gates: Iterable[Gate] = BACKSTOP) -> DischargeResult:
    """The result as it may settle: every violated gate converts it to
    failed, its finding kept in the witness for audit."""
    for gate in gates:
        if not gate.violated(result, obligation, scope):
            continue
        declaration = ({"witness_kind": result.witness_kind,
                        "inputs": result.inputs}
                       if gate.keeps_declaration else {})
        result = DischargeResult(
            outcome=OUTCOME_FAILED,
            witness=gate.backstop(result, obligation, scope),
            artifact=result.artifact,
            instrument=result.instrument,
            basis=result.basis,
            strength=result.strength,
            assumed_premises=result.assumed_premises,
            **declaration,
        )
    return result


def preflight_defect(result, obligation,
                     gates: Iterable[Gate] = PREFLIGHT) -> Optional[str]:
    """The first axiom the executor's declaration breaks, worded for the
    game's record; None when it breaks none."""
    scope = GateScope()
    for gate in gates:
        if gate.violated(result, obligation, scope):
            return gate.preflight(result, obligation, scope)
    return None
