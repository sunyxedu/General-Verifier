"""Pure host phase binding: no models, evaluator calls or filesystem truth."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from verifier.core.model import canonical, content_hash
from verifier.profiles import get_profile
from verifier.profiles.cyber.phases import (
    AUTHORIZATION_CONTRACT_LOCUS, AUTHORIZATION_SCHEMA, COMPLETION_SCHEMA,
    PhaseBindingError, authorization_statement, authorization_verification_context,
    completion_statement, make_authorization_binding, make_completion_binding,
    validate_phase_binding,
)
from verifier.profiles.cyber.prompts import ANSWER_BLIND, SCOPE, SYSTEMS


def candidate():
    digest = content_hash("synthetic immutable bytes")
    return {"candidate_id": digest, "sha256": digest, "size": 7,
            "blob_path": "/synthetic/private/blobs/" + digest,
            "source_path": "/synthetic/solver/proposal", "label": "inert solver hint"}


def observation():
    return {"candidate_id": candidate()["candidate_id"], "artifact_id": content_hash("observation artifact"),
            "artifact_path": "/synthetic/artifacts/observation.json", "effect_ref": "effect:4",
            "effect_claim_id": content_hash("record claim"), "event_seq": 4,
            "event_digest": content_hash("canonical event"), "evaluator_id": content_hash("host instrument"),
            "environment": "host-owned isolated build", "authority_scope": "host_evaluator_execution"}


def authorization(**changes):
    kwargs = {"profile_id": content_hash("profile"), "task_sha256": content_hash("task"),
              "capability_revision": content_hash("capability inventory"), "destination": "/synthetic/output/final",
              "destination_source": "host_required", "observations": [observation()], "verification_epoch": "auth-epoch"}
    kwargs.update(changes)
    return make_authorization_binding(candidate(), **kwargs)


def completion(auth=None, **changes):
    kwargs = {"authorization_root_id": content_hash("host independent assertion"),
              "authorization_certificate_id": content_hash("authenticated authorization certificate"),
              "receipt_artifact_id": content_hash("authenticated actual receipt"),
              "delivery_id": "0123456789abcdef0123456789abcdef", "verification_epoch": "completion-epoch"}
    kwargs.update(changes)
    return make_completion_binding(auth or authorization(), **kwargs)


def rehash(binding):
    binding["context_id"] = content_hash({key: value for key, value in binding.items() if key != "context_id"})
    return binding


def test_authorization_has_exact_immutable_subject_and_host_plan():
    binding = authorization()
    assert validate_phase_binding(binding) is None
    assert binding["phase"] == "candidate_authorization" and binding["certificate_scope"] == "candidate"
    assert binding["delivery_pending"] is True
    assert binding["subject"] == {**{key: candidate()[key] for key in ("candidate_id", "sha256", "size", "blob_path")},
                                  "address": candidate()["blob_path"], "address_source": "immutable_candidate"}
    assert binding["delivery_contract"] == {"kind": "single_file_bytes", "destination": "/synthetic/output/final",
           "destination_source": "host_required", "source_candidate_id": candidate()["candidate_id"],
           "source_sha256": candidate()["sha256"], "source_size": 7,
           "physical_postconditions": ["exists", "regular_file", "exact_bytes"], "writer": "host_finalizer"}
    assert "source_path" not in binding["subject"] and "inert solver hint" not in authorization_statement(binding)
    assert json.loads(authorization_statement(binding)) == {"schema": AUTHORIZATION_SCHEMA, "binding": binding}


@pytest.mark.parametrize("source", ["host_required", "candidate_proposal"])
def test_completion_preserves_authorization_evidence_and_binds_real_receipt(source):
    auth = authorization(destination_source=source)
    before = deepcopy(auth)
    result = completion(auth)
    assert auth == before
    assert result["phase"] == "task_completion" and result["certificate_scope"] == "task"
    assert result["delivery_pending"] is False and result["subject"]["address"] == auth["delivery_contract"]["destination"]
    assert result["subject"]["address_source"] == source
    for key in ("profile_id", "task_sha256", "capability_revision", "observations", "delivery_contract"):
        assert result[key] == auth[key]
    assert json.loads(completion_statement(result)) == {"schema": COMPLETION_SCHEMA, "binding": result}
    assert auth["context_id"] != result["context_id"]


def test_factory_copies_host_inputs_and_no_observation_is_assumed_true():
    subject, records = candidate(), [observation()]
    result = make_authorization_binding(subject, profile_id=content_hash("profile"), task_sha256=content_hash("task"),
        capability_revision=content_hash("caps"), destination="/synthetic/final", destination_source="host_required",
        observations=records, verification_epoch="auth")
    records[0]["environment"] = "changed"
    subject["blob_path"] = "/changed"
    assert result["observations"][0]["environment"] != "changed"
    assert result["subject"]["blob_path"] != "/changed"
    assert "verdict" not in result


@pytest.mark.parametrize("path", ["relative", "/", "//host/path", "/a/../final", "/a/./final", "/a//final", "/a/final/", "/a/\x00final"])
def test_only_canonical_absolute_single_file_destination(path):
    with pytest.raises(PhaseBindingError):
        authorization(destination=path)


@pytest.mark.parametrize("mutate", [
    lambda b: b.update(schema="gv-cyber-candidate-v1"),
    lambda b: b.pop("phase"),
    lambda b: b.update(certificate_scope="task"),
    lambda b: b.update(delivery_pending=1),
    lambda b: b.update(premises=[]),
    lambda b: b["subject"].update(address=b["delivery_contract"]["destination"]),
    lambda b: b["subject"].update(address_source="host_required"),
    lambda b: b["subject"].update(size=True),
    lambda b: b["subject"].update(size=-1),
    lambda b: b["delivery_contract"].update(source_size=True),
    lambda b: b["delivery_contract"].update(source_sha256=content_hash("different bytes")),
    lambda b: b["delivery_contract"].update(kind="deployment"),
    lambda b: b["delivery_contract"].update(writer="solver"),
    lambda b: b["delivery_contract"].update(destination_source=[]),
    lambda b: b["delivery_contract"].update(deferred_conditions=["runtime security"]),
    lambda b: b["delivery_contract"].update(physical_postconditions=["exists", "regular_file", "exact_bytes", "security"]),
    lambda b: b["delivery_contract"].update(physical_postconditions=["exact_bytes", "exists", "regular_file"]),
    lambda b: b.update(verification_epoch=""),
    lambda b: b.update(verification_epoch="auth\n"),
    lambda b: b.update(profile_id="A" * 64),
    lambda b: b["observations"][0].update(candidate_id=content_hash("different candidate")),
    lambda b: b["observations"][0].update(event_seq=True),
    lambda b: b["observations"][0].update(effect_ref="effect:5"),
    lambda b: b["observations"][0].update(verdict="verified"),
    lambda b: b["observations"].append(deepcopy(b["observations"][0])),
])
def test_strict_shape_and_semantics_fail_even_with_recomputed_context(mutate):
    binding = authorization()
    mutate(binding)
    with pytest.raises(PhaseBindingError):
        validate_phase_binding(rehash(binding))


@pytest.mark.parametrize("key,value", [("profile_id", content_hash("changed profile")),
    ("task_sha256", content_hash("changed task")), ("capability_revision", content_hash("changed caps")),
    ("verification_epoch", "changed epoch")])
def test_stale_context_cannot_replay_changed_authorization(key, value):
    binding = authorization()
    binding[key] = value
    with pytest.raises(PhaseBindingError, match="binding_context_mismatch"):
        validate_phase_binding(binding)
    assert rehash(binding)["context_id"] != authorization()["context_id"]


@pytest.mark.parametrize("identity", ["authorization_root_id", "authorization_certificate_id", "receipt_artifact_id"])
def test_completion_context_binds_each_authenticated_identity(identity):
    binding = completion()
    old = binding["context_id"]
    binding[identity] = content_hash("different artifact")
    with pytest.raises(PhaseBindingError, match="binding_context_mismatch"):
        validate_phase_binding(binding)
    assert rehash(binding)["context_id"] != old


def test_completion_requires_fresh_epoch_and_does_not_reauthorize_itself():
    with pytest.raises(PhaseBindingError, match="fresh_epoch"):
        completion(verification_epoch="auth-epoch")
    with pytest.raises(PhaseBindingError, match="not_an_authorization"):
        completion(completion())
    with pytest.raises(PhaseBindingError):
        authorization_statement(completion())
    with pytest.raises(PhaseBindingError):
        completion_statement(authorization())


def test_independent_authorization_context_is_not_ordinary_or_completion_scope():
    binding = authorization()
    expected = content_hash({"context_id": binding["context_id"], "independent_root": {
        "scope": "candidate_authorization", "contract_locus": AUTHORIZATION_CONTRACT_LOCUS}})
    assert authorization_verification_context(binding) == expected != binding["context_id"]
    with pytest.raises(PhaseBindingError):
        authorization_verification_context(completion())


def test_empty_observations_and_zero_bytes_are_structural_not_verdicts():
    binding = authorization(observations=[])
    assert binding["observations"] == []
    binding["subject"]["size"] = 0
    binding["delivery_contract"]["source_size"] = 0
    assert validate_phase_binding(rehash(binding)) is None


def test_host_contract_locus_preserves_whole_candidate_qualification():
    for clause in ("without solver premises", "not task completion", "content, runtime, security, prerequisite",
                   "historical-action", "No other file", "deployment-dependent security property",
                   "historical execution", "merely mentioning destination", "Identity alone never"):
        assert clause in AUTHORIZATION_CONTRACT_LOCUS
    assert "ordinary nonfinal claim compile only its residual" in ANSWER_BLIND
    assert "entire authorization contract locus even though it is not final" in ANSWER_BLIND
    assert all("checks every task content/runtime/security/prerequisite/history condition" in system
               for system in SYSTEMS.values())
    assert "defer nothing else, including files, deployment-security, history or destination-mentioning conditions" in SCOPE
    assert "phase=task_completion requires actual delivery" in SCOPE


def test_profile_pin_includes_actual_phase_boundary_bytes():
    from verifier.profiles.cyber import phases
    identity = get_profile("cyber").identity
    assert identity["cyber_boundary"]["phases.py"] == hashlib.sha256(Path(phases.__file__).read_bytes()).hexdigest()


def test_same_receipt_new_delivery_has_distinct_completion_context_and_statement():
    first = completion()
    second = completion(delivery_id="fedcba9876543210fedcba9876543210")
    assert first["receipt_artifact_id"] == second["receipt_artifact_id"]
    assert first["authorization_root_id"] == second["authorization_root_id"]
    assert first["authorization_certificate_id"] == second["authorization_certificate_id"]
    assert first["verification_epoch"] == second["verification_epoch"]
    assert first["context_id"] != second["context_id"]
    assert completion_statement(first) != completion_statement(second)


def test_changed_delivery_cannot_replay_stale_completion_context():
    binding = completion()
    binding["delivery_id"] = "fedcba9876543210fedcba9876543210"
    with pytest.raises(PhaseBindingError, match="binding_context_mismatch"):
        validate_phase_binding(binding)


def test_restored_same_delivery_reconstructs_identical_completion_binding():
    original = completion()
    replayed = completion(delivery_id=original["delivery_id"])
    assert original == replayed
    assert completion_statement(original) == completion_statement(replayed)


@pytest.mark.parametrize("delivery_id", [None, True, 42, "", "not-a-transaction", "A" * 32,
                                        "a" * 31, "a" * 33, "a" * 64, "g" * 32,
                                        "a" * 31 + "\n", "01234567-89ab-cdef-0123-456789abcdef"])
def test_delivery_identity_is_exact_lowercase_32hex(delivery_id):
    with pytest.raises(PhaseBindingError, match="invalid_delivery_transaction_identity"):
        completion(delivery_id=delivery_id)


def test_authorization_cannot_contain_delivery_identity():
    binding = authorization()
    binding["delivery_id"] = "0123456789abcdef0123456789abcdef"
    with pytest.raises(PhaseBindingError, match="invalid_phase_fields"):
        validate_phase_binding(rehash(binding))


def test_legacy_completion_without_transaction_identity_is_rejected():
    binding = completion()
    del binding["delivery_id"]
    with pytest.raises(PhaseBindingError, match="invalid_phase_fields"):
        validate_phase_binding(rehash(binding))
