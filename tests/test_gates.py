"""The evidence axioms, each a gate defined once (engine.gates): what each
one rejects, what the backstop chain turns a violating result into, and
which violation the game's preflight reports first."""
from __future__ import annotations

from verifier.core.model import Obligation
from verifier.engine import gates
from verifier.engine.gates import GateScope, apply_backstop, preflight_defect
from verifier.execution.discharge import DischargeResult

WORLD = Obligation("c", "the file holds 15", required_basis="world")
PRESENCE = Obligation("c", "the file exists", required_basis="world",
                      refutable_by_absence=True)
DELIVERED_AUTHORITY = ({"ref": "/app/out", "provenance": "delivered",
                        "role": "authority"},)


def result(**fields):
    base = dict(outcome="holds", witness=None, artifact="a", instrument="i",
                basis="world", strength="decisive")
    base.update(fields)
    return DischargeResult(**base)


def test_a_clean_result_passes_every_gate_unchanged():
    clean = result(inputs=({"ref": "/app/out", "provenance": "delivered",
                            "role": "subject"},))
    assert apply_backstop(clean, WORLD, GateScope()) is clean
    assert preflight_defect(clean, WORLD) is None


def test_nothing_below_decisive_refutes():
    bounded = result(outcome="refuted", strength="bounded", witness="w",
                     witness_kind="execution")
    out = apply_backstop(bounded, WORLD, GateScope())
    assert out.outcome == "failed"
    assert out.witness.startswith("refutation below decisive strength")
    # the strength gate keeps no declaration; None reads as decisive
    assert out.witness_kind is None
    assert not gates.below_decisive("refuted", None)
    assert gates.below_decisive_reason("analytic", "w", redo=2) == (
        "refutation below decisive strength is inadmissible "
        "(redo 2; analytic instrument reported: w)")


def test_absence_refutes_only_a_presence_requirement():
    absent = result(outcome="refuted", witness="nothing found",
                    witness_kind="absence")
    out = apply_backstop(absent, WORLD, GateScope())
    assert out.outcome == "failed" and out.witness_kind == "absence"
    assert "inadmissible for this obligation" in out.witness
    assert "inadmissible: absence, silence" in preflight_defect(absent, WORLD)
    assert apply_backstop(absent, PRESENCE, GateScope()) is absent
    assert gates.admissible_refutation("w", "execution")
    assert not gates.admissible_refutation("", "execution")
    assert not gates.admissible_refutation("w", "absence")


def test_a_delivered_authority_makes_a_holds_circular():
    circular = result(inputs=DELIVERED_AUTHORITY)
    out = apply_backstop(circular, WORLD, GateScope())
    assert out.outcome == "failed" and out.inputs == DELIVERED_AUTHORITY
    assert "declared as authority for the check" in out.witness
    assert "declared as authority; a holds" in preflight_defect(circular,
                                                                WORLD)


def test_evidence_below_the_floor_settles_neither_direction():
    for outcome in ("holds", "refuted"):
        low = result(outcome=outcome, basis="record", witness="w",
                     witness_kind="record_contradiction")
        out = apply_backstop(low, WORLD, GateScope())
        assert out.outcome == "failed"
        assert "below the obligation's floor 'world'" in out.witness


def test_assumed_premises_must_be_in_force():
    assuming = result(assumed_premises=("p1", "p2"))
    ok = apply_backstop(assuming, WORLD,
                        GateScope(premises_in_force=frozenset({"p1", "p2"})))
    assert ok is assuming
    out = apply_backstop(assuming, WORLD,
                         GateScope(premises_in_force=frozenset({"p1"})))
    assert out.witness.startswith(
        "assumed premises outside the claim's dependency closure: ['p2']")
    final = apply_backstop(assuming, WORLD, GateScope(completion_claim=True))
    assert final.witness.startswith(
        "a completion claim's contract admits no assumed premises")


def test_the_backstop_converts_in_order_and_later_gates_see_the_conversion():
    # a weak refutation is failed by the strength gate; the witness gate
    # then sees a failed result and adds nothing
    weak = result(outcome="refuted", strength="analytic",
                  witness_kind="absence", witness="w")
    out = apply_backstop(weak, WORLD, GateScope())
    assert out.witness.startswith("refutation below decisive strength")


def test_the_preflight_reports_a_changed_world_first():
    changed = DischargeResult(outcome="refuted", witness="w",
                              witness_kind="absence",
                              world_changes={"modified": ["/app/out"]})
    assert preflight_defect(changed, WORLD).startswith(
        "check altered the delivered world")


def test_a_refutation_transfers_only_decisive_with_an_admissible_witness():
    assert gates.transferable_refutation("execution", "decisive", WORLD)
    assert gates.transferable_refutation("execution", None, WORLD)
    assert not gates.transferable_refutation("execution", "bounded", WORLD)
    assert not gates.transferable_refutation("absence", "decisive", WORLD)
    assert gates.transferable_refutation("absence", "decisive", PRESENCE)
