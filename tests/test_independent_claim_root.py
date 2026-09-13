"""Offline host-root authority, premise isolation, export and replay tests."""
from __future__ import annotations

import copy
import json

import pytest

from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_RECORD, BASIS_WORLD, CLAIM_ASSERTION, CLAIM_EFFECT, CLAIM_FINAL,
    STRENGTH_BOUNDED, STRENGTH_DECISIVE, Claim, Obligation, content_hash,
)
from verifier.core.store import ClaimStore, TRIGGER_LAZY
from verifier.engine.certificate import request_certificate, request_independent_certificate
from verifier.engine.kernel import Kernel
from verifier.engine.game import BindingGame
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm import prompts
from verifier.llm.schema import FixtureBackend


HOST = {"scope": "candidate_authorization", "contract_locus":
        "HOST QUALIFICATION CONTRACT: establish every authoritative content, security "
        "and historical requirement for the immutable subject; only the declared "
        "future host byte-delivery postconditions remain pending. This is not completion."}


class RecordingDischarger:
    def __init__(self, *, outcome="holds", basis=BASIS_WORLD,
                 strength=STRENGTH_DECISIVE, assumed=()):
        self.calls = []
        self.outcome, self.basis, self.strength, self.assumed = outcome, basis, strength, assumed

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append((copy.deepcopy(row), obligation))
        return DischargeResult(
            outcome=self.outcome, witness="synthetic retained witness",
            witness_kind="execution", artifact="", instrument="offline fixture",
            basis=self.basis, strength=self.strength,
            assumed_premises=list(self.assumed))


def build(tmp_path, *, registry=None, discharger=None, db="claims.sqlite"):
    tmp_path.mkdir(exist_ok=True)
    log = EventLog(str(tmp_path / "events.jsonl"))
    store = ClaimStore(str(tmp_path / db), trigger=TRIGGER_LAZY)
    artifacts = ArtifactStore(str(tmp_path / "artifacts"))
    backend = FixtureBackend({
        "obligations": {"obligations": [{"text": "synthetic exact qualification condition",
                            "required_basis": "world", "severity": "must"}]},
        "critic": {"verdict": "pass", "counterexample_world": "", "missing_condition": ""},
    })
    executor = discharger or RecordingDischarger()
    kernel = Kernel(
        store, log, backend, artifacts, discharger=executor, critic_rounds=1,
        independent_root_provider=(registry.get if registry is not None else None),
        context_id_provider=lambda cid: "host-context")
    return store, kernel, log, artifacts, backend, executor


def add_root(store, registry, *, deps=(), text="host-generated complete candidate qualification"):
    claim = Claim(CLAIM_ASSERTION, text, deps=deps)
    registry[claim.id] = copy.deepcopy(HOST)
    return store.add_claim(claim)


def certify(parts, root, **kwargs):
    return request_independent_certificate(*parts[:4], root, **kwargs)


def test_only_callback_can_grant_authority_not_text_or_certificate_metadata(tmp_path):
    registry = {}
    parts = build(tmp_path, registry=registry)
    store, kernel, log, _, backend, _ = parts
    ordinary = store.add_claim(Claim(CLAIM_ASSERTION, json.dumps(HOST)))
    before = list(log.replay())
    assert not kernel.is_independent_root(ordinary)
    with pytest.raises(ValueError, match="not a registered"):
        certify(parts, ordinary, profile_metadata=HOST)
    assert backend.calls == [] and list(log.replay()) == before


@pytest.mark.parametrize("kind", [CLAIM_EFFECT, CLAIM_FINAL])
def test_host_callback_never_grants_other_claim_kinds_authority(tmp_path, kind):
    registry = {}
    parts = build(tmp_path, registry=registry)
    target = parts[0].add_claim(Claim(kind, "synthetic other kind"))
    registry[target] = HOST
    assert not parts[1].is_independent_root(target)
    with pytest.raises(ValueError, match="host assertion root"):
        certify(parts, target)


@pytest.mark.parametrize("bad", [True, "candidate_authorization", {},
    {"scope": "task", "contract_locus": "synthetic"},
    {"scope": "candidate_authorization", "contract_locus": "  "},
    {"scope": "candidate_authorization", "contract_locus": True},
    {**HOST, "extra": "untrusted"}])
def test_invalid_descriptor_fails_closed_without_echo(tmp_path, bad):
    registry = {}
    parts = build(tmp_path, registry=registry)
    root = parts[0].add_claim(Claim(CLAIM_ASSERTION, "synthetic assertion"))
    registry[root] = bad
    with pytest.raises(ValueError, match="invalid independent root descriptor"):
        certify(parts, root)
    assert not parts[4].calls and not list(parts[2].replay())


def test_provider_error_is_static_and_no_secret_echo(tmp_path):
    parts = build(tmp_path)
    root = parts[0].add_claim(Claim(CLAIM_ASSERTION, "synthetic assertion"))
    def broken(cid):
        raise RuntimeError("synthetic private provider contents")
    parts[1]._independent_root_provider = broken
    with pytest.raises(ValueError, match="^trusted independent root provider failed$"):
        certify(parts, root)


def test_compiler_critic_and_obligation_are_independent_of_solver_premises(tmp_path):
    registry = {}
    parts = build(tmp_path, registry=registry)
    store, kernel, _, _, backend, executor = parts
    poison = store.add_claim(Claim(CLAIM_ASSERTION, "solver private hypothesis"))
    store.set_refuted(poison, "synthetic false premise", basis=BASIS_WORLD)
    pending = store.add_claim(Claim(CLAIM_ASSERTION, "solver unknown premise"))
    root = add_root(store, registry, deps=(pending, poison))
    cert = certify(parts, root)
    assert cert["verdict"] == "verified"
    assert store.status(pending) == "unverified"
    assert [call["role"] for call in backend.calls] == ["obligations", "critic"]
    for call in backend.calls:
        variables = call["variables"]
        assert variables["verified_deps"] == variables["pending_deps"] == "[]"
        assert "solver private hypothesis" not in variables.get("graph", "")
        assert "solver unknown premise" not in variables["dependency_referent_context"]
        assert variables["contract_locus"] == HOST["contract_locus"]
    assert executor.calls[0][1].premise_scope == ()
    assert kernel._premises_in_force(store.get(root)) == set()
    assert cert["effective_status"] == "verified"
    assert cert["effective_basis"] == BASIS_WORLD
    assert cert["open_premises"] == [] and cert["premise_refuted"] is None
    exported = {entry["id"]: entry for entry in cert["slice"]}
    assert exported[poison]["effective_status"] == "refuted"
    assert exported[pending]["status"] == "unverified"
    assert exported[root]["deps"] == sorted([pending, poison])
    node = exported[root]["obligations"][0]
    assert node["premise_scope"] == []
    assert node["verification_context"] == kernel.verification_context_for(root)
    assert store.obligation_verification_context(node["id"]) == node["verification_context"]


def test_ordinary_assertion_retains_refuted_premise_early_out_and_effective_fold(tmp_path):
    parts = build(tmp_path, registry={})
    store, kernel = parts[:2]
    poison = store.add_claim(Claim(CLAIM_ASSERTION, "synthetic poison"))
    store.set_refuted(poison, "synthetic false premise", basis=BASIS_WORLD)
    ordinary = store.add_claim(Claim(CLAIM_ASSERTION, "synthetic ordinary", deps=(poison,)))
    assert kernel.verify(ordinary) == "unverified"
    assert parts[4].calls == []
    assert store.effective(ordinary)["effective_status"] == "refuted"


def test_root_keeps_exact_premise_restatements_in_full_qualification(tmp_path):
    registry = {}
    parts = build(tmp_path, registry=registry)
    store, kernel = parts[:2]
    parent = store.add_claim(Claim(CLAIM_ASSERTION, "synthetic parent condition"))
    root = add_root(store, registry, deps=(parent,))
    payload = {"obligations": [{"text": "synthetic parent condition", "required_basis": "world"},
                               {"text": "synthetic extra condition", "required_basis": "world"}]}
    assert len(kernel._materialize_obligations(store.get(root), payload)) == 2
    ordinary = store.add_claim(Claim(CLAIM_ASSERTION, "synthetic ordinary conjunction", deps=(parent,)))
    assert len(kernel._materialize_obligations(store.get(ordinary), payload)) == 1


def test_real_oracle_binding_judger_path_cannot_assume_root_dependencies(tmp_path):
    registry = {}
    parts = build(tmp_path, registry=registry)
    store, kernel, log, artifacts, backend, executor = parts
    parent = store.add_claim(Claim(CLAIM_ASSERTION, "synthetic recorded solver conclusion"))
    store.set_verified(parent, basis=BASIS_RECORD, strength=STRENGTH_BOUNDED)
    backend.fixtures.update({
        "oracles": {"methods": [{"procedure": "synthetic independent check",
            "artifact": "synthetic retained observation", "level": "synthetic",
            "basis": "world", "strength": "decisive", "bridges": []}]},
        "judger": {"sufficient": "yes", "faithful": "yes", "degradation": "n/a",
                   "reason": "synthetic exact observation holds"},
    })
    kernel.binder = BindingGame(backend, artifacts, log, executor)
    root = add_root(store, registry, deps=(parent,))
    cert = certify(parts, root)
    assert cert["verdict"] == "verified"
    assert cert["effective_basis"] == BASIS_WORLD
    assert cert["effective_strength"] == STRENGTH_DECISIVE
    assert [call["role"] for call in backend.calls] == ["obligations", "critic", "oracles", "judger"]
    assert executor.calls[0][1].premise_scope == ()
    node = cert["slice"][0]["obligations"][0]
    bound = artifacts.get(node["evidence"][-1]["artifact"])
    assert bound["instrument"] == "binding-game"
    assert node["evidence"][-1]["assumed_premises"] == []
    assert bound["bound"]["audit"]["sufficient"] == "yes"


@pytest.mark.parametrize("outcome", ["holds", "refuted", "unobtainable"])
def test_discharge_cannot_assume_even_an_explicit_dependency(tmp_path, outcome):
    registry = {}
    parts = build(tmp_path, registry=registry)
    store = parts[0]
    parent = store.add_claim(Claim(CLAIM_ASSERTION, "synthetic independent content premise"))
    store.set_verified(parent, basis=BASIS_WORLD)
    parts[5].outcome, parts[5].assumed = outcome, [parent]
    root = add_root(store, registry, deps=(parent,))
    cert = certify(parts, root)
    assert cert["verdict"] == "unverified"
    node = store.obligation_children(root)[0]
    evidence = store.evidence_for(node["id"])[-1]
    assert evidence["outcome"] == "failed"
    assert "admits no assumed premises" in evidence["witness"]


def test_context_is_explicit_and_cannot_share_phase_nodes(tmp_path):
    registry = {}
    parts = build(tmp_path, registry=registry)
    store, kernel = parts[:2]
    root = add_root(store, registry)
    ordinary = store.add_claim(Claim(CLAIM_ASSERTION, "synthetic ordinary"))
    final = store.add_claim(Claim(CLAIM_FINAL, "synthetic completion"))
    assert kernel.verification_context_for(root) == content_hash({
        "context_id": "host-context", "independent_root": HOST})
    assert kernel.verification_context_for(ordinary) == "host-context"
    assert kernel.verification_context_for(final) == "host-context"
    payload = parts[4].fixtures["obligations"]
    auth_node = kernel._materialize_obligations(store.get(root), payload)[0]
    ordinary_node = kernel._materialize_obligations(store.get(ordinary), payload)[0]
    final_node = kernel._materialize_obligations(store.get(final), payload)[0]
    assert auth_node.id != ordinary_node.id == final_node.id


def test_phase_context_does_not_silently_accept_a_malformed_base_context(tmp_path):
    registry = {}
    parts = build(tmp_path, registry=registry)
    root = add_root(parts[0], registry)
    parts[1].context_id_provider = lambda cid: {"untyped": "synthetic"}
    with pytest.raises(ValueError, match="verification context must be a string"):
        parts[1].verification_context_for(root)


def test_same_text_ordinary_claim_cannot_launder_a_different_scope_verdict(tmp_path):
    registry = {}
    parts = build(tmp_path, registry=registry, discharger=RecordingDischarger(outcome="refuted"))
    store = parts[0]
    earlier = store.add_claim(Claim(CLAIM_ASSERTION, "synthetic exact qualification condition"))
    store.set_verified(earlier, basis=BASIS_WORLD, strength=STRENGTH_DECISIVE)
    root = add_root(store, registry)
    cert = certify(parts, root)
    assert cert["verdict"] == "refuted" and len(parts[5].calls) == 1


@pytest.mark.parametrize("mutation", ["revoke", "locus", "scope"])
@pytest.mark.parametrize("consumer", ["verify", "effective", "certificate"])
def test_registration_change_rejected_after_settled_fast_path(tmp_path, mutation, consumer):
    registry = {}
    parts = build(tmp_path, registry=registry)
    root = add_root(parts[0], registry)
    assert certify(parts, root)["verdict"] == "verified"
    if mutation == "revoke":
        registry.pop(root)
    elif mutation == "locus":
        registry[root] = {**HOST, "contract_locus": "different fixed host contract"}
    else:
        registry[root] = {**HOST, "scope": "task"}
    before = list(parts[2].replay())
    action = {"verify": lambda: parts[1].verify(root),
              "effective": lambda: parts[0].effective(root),
              "certificate": lambda: certify(parts, root)}[consumer]
    with pytest.raises(ValueError):
        action()
    assert list(parts[2].replay()) == before


def test_explicit_none_registration_cannot_be_promoted_later(tmp_path):
    registry = {}
    parts = build(tmp_path, registry=registry)
    root = parts[0].add_claim(Claim(CLAIM_ASSERTION, "synthetic originally ordinary"))
    assert not parts[1].is_independent_root(root)
    registry[root] = HOST
    with pytest.raises(ValueError, match="registration changed"):
        parts[1].verify(root)


def test_already_settled_ordinary_assertion_cannot_gain_authority_on_new_kernel(tmp_path):
    parts = build(tmp_path)
    root = parts[0].add_claim(Claim(CLAIM_ASSERTION, "synthetic settled ordinary"))
    parts[0].set_verified(root, basis=BASIS_WORLD)
    parts[0].db.close()
    resumed = build(tmp_path, registry={root: HOST})
    with pytest.raises(ValueError, match="cannot promote"):
        certify(resumed, root)


def test_restart_preserves_authority_and_rejects_descriptor_rebinding(tmp_path):
    registry = {}
    parts = build(tmp_path, registry=registry)
    root = add_root(parts[0], registry)
    first = certify(parts, root)
    parts[0].db.close()
    resumed = build(tmp_path, registry=registry)
    repeat = certify(resumed, root)
    assert repeat["certificate_id"] == first["certificate_id"]
    assert resumed[4].calls == []
    resumed[0].db.close()
    changed = build(tmp_path, registry={root: {**HOST, "contract_locus": "changed fixed contract"}})
    with pytest.raises(ValueError, match="registration changed"):
        certify(changed, root)


@pytest.mark.parametrize("strength,expected", [(STRENGTH_DECISIVE, "verified"),
                                             (STRENGTH_BOUNDED, "supported")])
def test_certificate_exports_effective_strength_basis_and_complete_evidence(tmp_path, strength, expected):
    registry = {}
    parts = build(tmp_path, registry=registry, discharger=RecordingDischarger(strength=strength))
    root = add_root(parts[0], registry)
    metadata = {"synthetic": "host metadata not authority"}
    cert = certify(parts, root, profile_metadata=metadata)
    assert cert["schema"] == "gv-independent-claim-root-v1" and "final" not in cert
    assert cert["root"] == root and cert["root_scope"] == "candidate_authorization"
    assert cert["verdict"] == expected
    assert cert["effective_strength"] == cert["verdict_strength"] == strength
    assert cert["effective_basis"] == cert["verdict_basis"] == BASIS_WORLD
    assert cert["profile_metadata"] == metadata
    assert cert["slice"][0]["obligations"][0]["evidence"]
    retained = parts[3].get(cert["certificate_id"])
    assert retained == {key: value for key, value in cert.items() if key != "certificate_id"}
    event = list(parts[2].replay())[-1]
    assert event["type"] == "independent_certificate"
    assert event["payload"]["root"] == root


def test_unknown_basis_cannot_be_read_out_as_verified(tmp_path):
    registry = {}
    parts = build(tmp_path, registry=registry)
    root = add_root(parts[0], registry)
    assert parts[1].is_independent_root(root)
    parts[0].set_verified(root, basis=None)
    assert certify(parts, root)["verdict"] == "unverified"


def test_scope_tampering_in_retained_obligation_is_not_inferred_away(tmp_path):
    registry = {}
    parts = build(tmp_path, registry=registry)
    root = add_root(parts[0], registry)
    certify(parts, root)
    node = parts[0].obligation_children(root)[0]
    parts[0].db.execute("DELETE FROM store_metadata WHERE key=?",
                       ("obligation_context:" + node["id"],))
    parts[0].db.commit()
    assert parts[0].obligation_verification_context(node["id"]) is None
    with pytest.raises(ValueError, match="scope mismatch"):
        certify(parts, root)


def test_original_public_final_certificate_api_and_shape_remain_unchanged(tmp_path):
    registry = {}
    parts = build(tmp_path, registry=registry)
    store, kernel, log, artifacts = parts[:4]
    root = add_root(store, registry)
    before = list(log.replay())
    with pytest.raises(ValueError, match="final claim"):
        request_certificate(store, kernel, log, artifacts, root)
    assert list(log.replay()) == before
    final = store.add_claim(Claim(CLAIM_FINAL, "synthetic actual task completion"))
    cert = request_certificate(store, kernel, log, artifacts, final)
    assert cert["verdict"] == "verified" and cert["final"] == final
    assert "root" not in cert and "effective_status" not in cert
    assert "deps" not in cert["slice"][0]
    assert "premise_scope" not in cert["slice"][0]["obligations"][0]
    variables = next(call["variables"] for call in parts[4].calls if call["role"] == "obligations")
    assert variables["contract_locus"] == prompts.CONTRACT_LOCUS_FINAL


def test_certificate_cannot_mix_kernel_and_foreign_store(tmp_path):
    registry = {}
    parts = build(tmp_path, registry=registry)
    root = add_root(parts[0], registry)
    foreign = ClaimStore(trigger=TRIGGER_LAZY)
    with pytest.raises(ValueError, match="bound stores"):
        request_independent_certificate(foreign, parts[1], parts[2], parts[3], root)
