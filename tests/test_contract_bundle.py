"""Sealed contract replay stays a strict, result-free diagnostic seam."""
from __future__ import annotations

import hashlib
import json

import pytest

from verifier.core.events import EventLog
from verifier.core.model import (
    CLAIM_ASSERTION, CLAIM_EFFECT, CLAIM_FINAL, Claim, Obligation, canonical,
)
from verifier.core.store import ClaimStore, TRIGGER_LAZY
from verifier.engine.contract_bundle import (
    ContractBundleIdentity,
    ContractReplayError,
    SealedContractBundle,
    build_contract_bundle_bytes,
    claim_graph_sha256,
    contract_record_sha256,
    failed_contract_entry,
    passed_contract_entry,
)
from verifier.engine.kernel import Kernel
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend
from verifier.contract_replay import compile_input_digests, fresh_graph_snapshot


PASS = {"verdict": "pass", "counterexample_world": "n/a",
        "missing_condition": "n/a"}


class _TestBoundReplay:
    """Explicit test capability; a raw parsed bundle is never injectable."""

    def __init__(self, bundle):
        self.bundle = bundle

    def require_replayable(self):
        return self.bundle.require_replayable()

    def validate_kernel(self, kernel):
        assert kernel.live_world is False

    def lookup(self, claim_id):
        return self.bundle.lookup(claim_id)

    def validate_materialized(self, entry, obligations):
        return self.bundle.validate_materialized(entry, obligations)


def _kernel(tmp_path, name, claims, backend=None, replay=None):
    state = tmp_path / name
    state.mkdir()
    (state / "task.md").write_text("Return a checked result.")
    store = ClaimStore(trigger=TRIGGER_LAZY)
    log = EventLog(str(state / "events.jsonl"))
    backend = backend or FixtureBackend({})
    machine = Kernel(
        store, log, backend, ArtifactStore(str(state / "artifacts")),
        live_world=False,
    )
    for claim in claims:
        assert store.add_claim(claim) == claim.id
    if replay is not None:
        bound = _TestBoundReplay(replay)
        bound.require_replayable()
        bound.validate_kernel(machine)
        machine.contract_replay = bound
    return machine, store, backend


def _identity(claims):
    return ContractBundleIdentity(
        pipeline_sha256="1" * 64,
        config_sha256="2" * 64,
        source_sha256="3" * 64,
        graph_sha256=claim_graph_sha256(claims),
    )


def _materialized(claim, payload, scope=()):
    return [
        Obligation(
            claim.id, item["text"], item["required_basis"],
            severity=item.get("severity", "must"),
            clause=item.get("clause", ""),
            refutable_by_absence=item.get("refutable_by_absence", False),
            premise_scope=scope,
        )
        for item in payload["obligations"]
    ]


def _bundle(tmp_path):
    effect = Claim(CLAIM_EFFECT, '{"event_seq":1,"tool":"Read","digest":"x"}')
    support = Claim(CLAIM_ASSERTION, "the source record is relevant",
                    deps=(effect.id,))
    final = Claim(CLAIM_FINAL, "the delivered answer is correct",
                  deps=(support.id,))
    claims = [effect, support, final]
    setup, store, _ = _kernel(tmp_path, "setup", claims)
    support_payload = {"obligations": [{
        "text": "the source record establishes the stated fact",
        "required_basis": "record", "severity": "must",
        "clause": "source record", "refutable_by_absence": False,
    }]}
    final_payload = {"obligations": [{
        "text": "the delivered answer satisfies every requested condition",
        "required_basis": "world", "severity": "must",
        "clause": "Return a checked result.", "refutable_by_absence": False,
    }]}
    inputs = compile_input_digests(
        setup, fresh_graph_snapshot(setup, final.id))
    entries = [
        passed_contract_entry(
            claim_id=support.id, compile_input_sha256=inputs[support.id],
            critic_receipt_sha256=contract_record_sha256(PASS),
            payload=support_payload,
            obligations=_materialized(
                support, support_payload, scope=(effect.id,))),
        passed_contract_entry(
            claim_id=final.id, compile_input_sha256=inputs[final.id],
            critic_receipt_sha256=contract_record_sha256(PASS),
            payload=final_payload, obligations=_materialized(final, final_payload)),
    ]
    identity = _identity(claims)
    raw = build_contract_bundle_bytes(
        identity=identity, claims=claims, entries=entries)
    digest = hashlib.sha256(raw).hexdigest()
    bundle = SealedContractBundle.from_bytes(
        raw, expected_sha256=digest, expected_identity=identity,
        expected_compile_inputs=inputs)
    return claims, support_payload, final_payload, inputs, identity, raw, bundle


def test_replay_uses_exact_contract_without_compiler_or_critic_calls(tmp_path):
    claims, support_payload, final_payload, inputs, _, _, bundle = _bundle(tmp_path)
    machine, store, backend = _kernel(tmp_path, "replay", claims, replay=bundle)

    support = machine._compile_obligations(store.get(claims[1].id))
    final = machine._compile_obligations(store.get(claims[2].id))

    assert not backend.calls
    assert [item.text for item in support] == [
        support_payload["obligations"][0]["text"]]
    assert support[0].premise_scope == (claims[0].id,)
    assert [item.text for item in final] == [
        final_payload["obligations"][0]["text"]]
    assert final[0].premise_scope == ()
    compiled = [event for event in machine.log.replay()
                if event["type"] == "obligations_compiled"]
    assert [event["payload"]["claim"] for event in compiled] == [
        claims[1].id, claims[2].id]


def test_runtime_premise_status_does_not_invalidate_preexecution_lookup(tmp_path):
    premise = Claim(CLAIM_ASSERTION, "the premise")
    dependent = Claim(CLAIM_ASSERTION, "the conclusion", deps=(premise.id,))
    final = Claim(CLAIM_FINAL, "the final", deps=(dependent.id,))
    claims = [premise, dependent, final]
    setup, setup_store, _ = _kernel(tmp_path, "status-setup", claims)
    payloads = {
        premise.id: {"obligations": [{"text": "premise check",
                                      "required_basis": "record"}]},
        dependent.id: {"obligations": [
            {"text": premise.text, "required_basis": "record"},
            {"text": "residual check", "required_basis": "world"},
        ]},
        final.id: {"obligations": [{"text": "final check",
                                    "required_basis": "world"}]},
    }
    inputs = compile_input_digests(
        setup, fresh_graph_snapshot(setup, final.id))
    entries = []
    for claim in claims:
        scope = () if claim.kind == CLAIM_FINAL else tuple(sorted(claim.deps))
        obligations = _materialized(claim, payloads[claim.id], scope=scope)
        if claim is dependent:
            obligations = obligations[1:]
        entries.append(passed_contract_entry(
            claim_id=claim.id, compile_input_sha256=inputs[claim.id],
            critic_receipt_sha256=contract_record_sha256(PASS),
            payload=payloads[claim.id], obligations=obligations))
    identity = _identity(claims)
    raw = build_contract_bundle_bytes(
        identity=identity, claims=claims, entries=entries)
    bundle = SealedContractBundle.from_bytes(
        raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_identity=identity, expected_compile_inputs=inputs)
    machine, store, backend = _kernel(
        tmp_path, "status-replay", claims, replay=bundle)

    store.set_verified(premise.id, basis="record")
    obligations = machine._compile_obligations(store.get(dependent.id))

    assert not backend.calls
    assert [item.text for item in obligations] == ["residual check"]
    assert obligations[0].premise_scope == (premise.id,)


def test_missing_or_materialization_mismatch_is_hard_failure_without_fallback(
    tmp_path,
):
    claims, _, _, _, _, _, bundle = _bundle(tmp_path)
    machine, store, backend = _kernel(tmp_path, "missing", claims, replay=bundle)
    extra = Claim(CLAIM_ASSERTION, "not in the sealed graph")
    store.add_claim(extra)
    with pytest.raises(ContractReplayError, match="missing"):
        machine._compile_obligations(store.get(extra.id))
    assert not backend.calls

    entry = bundle.lookup(claims[1].id)
    wrong = [Obligation(claims[1].id, "different", "record",
                        premise_scope=(claims[0].id,))]
    with pytest.raises(ContractReplayError, match="materialized obligations"):
        bundle.validate_materialized(entry, wrong)


@pytest.mark.parametrize("mutation", ["sha", "identity", "input", "extra"])
def test_bundle_requires_external_exact_identity_and_strict_schema(tmp_path, mutation):
    _, _, _, inputs, identity, raw, _ = _bundle(tmp_path)
    expected_sha = hashlib.sha256(raw).hexdigest()
    expected_identity = identity
    expected_inputs = dict(inputs)
    if mutation == "sha":
        expected_sha = "f" * 64
    elif mutation == "identity":
        expected_identity = ContractBundleIdentity(
            identity.pipeline_sha256, identity.config_sha256,
            "f" * 64, identity.graph_sha256)
    elif mutation == "input":
        expected_inputs[next(iter(expected_inputs))] = "f" * 64
    else:
        value = json.loads(raw)
        value["unexpected"] = True
        raw = canonical(value).encode()
        expected_sha = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ContractReplayError):
        SealedContractBundle.from_bytes(
            raw, expected_sha256=expected_sha,
            expected_identity=expected_identity,
            expected_compile_inputs=expected_inputs)


def test_failure_entry_is_reportable_but_never_replayable(tmp_path):
    _, _, _, inputs, identity, raw, _ = _bundle(tmp_path)
    value = json.loads(raw)
    failed_id = value["entries"][0]["claim_id"]
    value["entries"][0] = failed_contract_entry(
        claim_id=failed_id,
        compile_input_sha256=inputs[failed_id], stage="critic",
        detail_sha256=contract_record_sha256({"reason": "standing attack"}),
        critic_receipt_sha256=contract_record_sha256({"verdict": "insufficient"}),
    )
    failed_raw = canonical(value).encode()
    bundle = SealedContractBundle.from_bytes(
        failed_raw, expected_sha256=hashlib.sha256(failed_raw).hexdigest(),
        expected_identity=identity, expected_compile_inputs=inputs)

    assert not bundle.replayable
    assert bundle.failures[0].failure_stage == "critic"
    with pytest.raises(ContractReplayError, match="failed compilation"):
        bundle.require_replayable()
    with pytest.raises(ContractReplayError, match="not a replayable"):
        bundle.lookup(failed_id)
    # A successful sibling entry cannot hide a partial setup failure.
    state = tmp_path / "partial-failure"
    state.mkdir()
    with pytest.raises(ContractReplayError, match="failed compilation"):
        Kernel(
            ClaimStore(trigger=TRIGGER_LAZY),
            EventLog(str(state / "events.jsonl")), FixtureBackend({}),
            ArtifactStore(str(state / "artifacts")), live_world=False,
            contract_replay=_TestBoundReplay(bundle))


def test_whole_graph_is_bound_but_only_final_slice_requires_entries(tmp_path):
    unrelated = Claim(CLAIM_ASSERTION, "not in the selected certificate")
    support = Claim(CLAIM_ASSERTION, "selected support")
    final = Claim(CLAIM_FINAL, "selected final", deps=(support.id,))
    claims = [unrelated, support, final]
    setup, store, _ = _kernel(tmp_path, "slice-setup", claims)
    inputs = compile_input_digests(
        setup, fresh_graph_snapshot(setup, final.id))
    assert unrelated.id not in inputs
    payload = {"obligations": [{"text": "check", "required_basis": "world"}]}
    entries = [passed_contract_entry(
        claim_id=claim.id, compile_input_sha256=inputs[claim.id],
        critic_receipt_sha256=contract_record_sha256(PASS), payload=payload,
        obligations=_materialized(claim, payload))
               for claim in (final, support)]
    identity = _identity(claims)
    raw = build_contract_bundle_bytes(
        identity=identity, claims=claims, entries=entries)

    bundle = SealedContractBundle.from_bytes(
        raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_identity=identity, expected_compile_inputs=inputs)
    assert bundle.lookup(final.id).replayable
    with pytest.raises(ContractReplayError, match="missing"):
        bundle.lookup(unrelated.id)

    # The same unrelated graph node is bound by graph_sha256, but adding a
    # contract for it must fail closed instead of indexing a missing expected
    # compile input or silently expanding the selected certificate.
    value = json.loads(raw)
    value["entries"].append(passed_contract_entry(
        claim_id=unrelated.id, compile_input_sha256="a" * 64,
        critic_receipt_sha256=contract_record_sha256(PASS), payload=payload,
        obligations=_materialized(unrelated, payload)))
    expanded = canonical(value).encode()
    with pytest.raises(ContractReplayError, match="outside.*final slice"):
        SealedContractBundle.from_bytes(
            expanded, expected_sha256=hashlib.sha256(expanded).hexdigest(),
            expected_identity=identity, expected_compile_inputs=inputs)


@pytest.mark.parametrize(
    ("field", "bad"),
    [("claim_kind", []), ("entry_status", {}), ("failure_stage", [])],
)
def test_malformed_non_scalar_discriminators_fail_closed(tmp_path, field, bad):
    _, _, _, inputs, identity, raw, _ = _bundle(tmp_path)
    value = json.loads(raw)
    if field == "claim_kind":
        value["claims"][0]["kind"] = bad
    elif field == "entry_status":
        value["entries"][0]["status"] = bad
    else:
        first = value["entries"][0]
        value["entries"][0] = {
            "claim_id": first["claim_id"],
            "status": "compile_failure",
            "compile_input_sha256": first["compile_input_sha256"],
            "critic_receipt_sha256": None,
            "failure": {"stage": bad, "detail_sha256": "a" * 64},
        }
    malformed = canonical(value).encode()
    with pytest.raises(ContractReplayError):
        SealedContractBundle.from_bytes(
            malformed,
            expected_sha256=hashlib.sha256(malformed).hexdigest(),
            expected_identity=identity, expected_compile_inputs=inputs)


def test_live_kernel_cannot_inject_contract_replay(tmp_path):
    claims, _, _, _, _, _, bundle = _bundle(tmp_path)
    state = tmp_path / "live-rejected"
    state.mkdir()
    store = ClaimStore(trigger=TRIGGER_LAZY)
    with pytest.raises(ValueError, match="offline diagnostic"):
        Kernel(
            store, EventLog(str(state / "events.jsonl")), FixtureBackend({}),
            ArtifactStore(str(state / "artifacts")),
            contract_replay=bundle, live_world=True)

    # Mutating the optional seam after construction cannot bypass the same
    # boundary (the diagnostic coordinator attaches only to an idle kernel).
    machine, store, _ = _kernel(tmp_path, "late-live", claims)
    machine.contract_replay = _TestBoundReplay(bundle)
    machine.live_world = True
    with pytest.raises(ValueError, match="offline diagnostic"):
        machine._compile_obligations(store.get(claims[-1].id))

    machine.live_world = False
    machine.contract_replay = bundle
    with pytest.raises(ValueError, match="bound to this kernel"):
        machine._compile_obligations(store.get(claims[-1].id))


def test_bound_provider_revalidates_kernel_before_lookup(tmp_path):
    claims, _, _, _, _, _, bundle = _bundle(tmp_path)

    class BoundProvider:
        def __init__(self):
            self.lookup_called = False

        def require_replayable(self):
            bundle.require_replayable()

        def validate_kernel(self, kernel):
            raise ContractReplayError("bound graph changed")

        def lookup(self, claim_id):
            self.lookup_called = True
            return bundle.lookup(claim_id)

        def validate_materialized(self, entry, obligations):
            return bundle.validate_materialized(entry, obligations)

    machine, store, backend = _kernel(tmp_path, "bound", claims)
    provider = BoundProvider()
    machine.contract_replay = provider
    with pytest.raises(ContractReplayError, match="bound graph changed"):
        machine._compile_obligations(store.get(claims[-1].id))
    assert not provider.lookup_called
    assert not backend.calls
    assert not list(machine.log.replay())


def test_default_compiler_critic_variables_and_order_are_unchanged(tmp_path):
    claim = Claim(CLAIM_ASSERTION, "a normal production claim")
    payload = {"obligations": [{"text": "normal check",
                                "required_basis": "record"}]}
    backend = FixtureBackend({"obligations": payload, "critic": PASS})
    machine, store, _ = _kernel(tmp_path, "ordinary", [claim], backend=backend)
    row = store.get(claim.id)
    expected = machine._obligation_compile_variables(row)

    obligations = machine._compile_obligations(row)

    assert [call["role"] for call in backend.calls] == ["obligations", "critic"]
    assert backend.calls[0]["variables"] == expected
    assert obligations[0].text == "normal check"
    compiled = [event for event in machine.log.replay()
                if event["type"] == "obligations_compiled"]
    assert len(compiled) == 1
