"""Pure, host-only candidate authorization and actual task completion bindings.

These functions validate and bind structure, not truth or authority. The caller
must authenticate candidates, retained observations, certificates and receipt
artifacts, enforce the host destination policy, and check the live delivery.
No filesystem inspection, execution, evaluator verdict or solver premise occurs
here. A syntactically valid binding is not a certificate or proof of completion.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath
import re
from typing import Any

from verifier.core.model import canonical, content_hash


AUTHORIZATION_SCHEMA = "gv-cyber-candidate-authorization-v2"
COMPLETION_SCHEMA = "gv-cyber-task-completion-v2"
AUTHORIZATION_CONTRACT_LOCUS = (
    "CANDIDATE AUTHORIZATION CONTRACT. Independently establish, without solver "
    "premises, that these immutable candidate bytes satisfy every authoritative "
    "task content, runtime, security, prerequisite, and historical-action "
    "condition under its original modality and required conditions. This is "
    "candidate qualification, not task completion. Only the exact host-owned "
    "single-file delivery_contract may postpone its destination exists, "
    "regular_file, and exact_bytes physical postconditions until real delivery "
    "inspection. Evaluate candidate content at its immutable blob address; do "
    "not pretend the canonical destination already exists. No other file, "
    "deployment-dependent security property, historical execution, or condition "
    "merely mentioning destination may be deferred. Identity alone never "
    "replaces the task's complete substantive conditions."
)

_BASE_FIELDS = frozenset({"certificate_scope", "phase", "delivery_pending", "profile_id",
                          "task_sha256", "capability_revision", "verification_epoch", "subject",
                          "observations", "delivery_contract", "context_id"})
_COMPLETION_DIGEST_FIELDS = frozenset({"authorization_root_id", "authorization_certificate_id", "receipt_artifact_id"})
_COMPLETION_FIELDS = _COMPLETION_DIGEST_FIELDS | {"delivery_id"}
_SUBJECT_FIELDS = frozenset({"candidate_id", "sha256", "size", "blob_path", "address", "address_source"})
_CONTRACT_FIELDS = frozenset({"kind", "destination", "destination_source", "source_candidate_id",
                             "source_sha256", "source_size", "physical_postconditions", "writer"})
_OBSERVATION_FIELDS = frozenset({"candidate_id", "artifact_id", "artifact_path", "effect_ref",
                                "effect_claim_id", "event_seq", "event_digest", "evaluator_id",
                                "environment", "authority_scope"})
_DESTINATION_SOURCES = frozenset({"host_required", "candidate_proposal"})
_PHYSICAL_POSTCONDITIONS = ["exists", "regular_file", "exact_bytes"]


class PhaseBindingError(ValueError):
    """Static, non-sensitive validation reason codes."""

    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise PhaseBindingError(code)


def _fields(value: Any, fields: frozenset[str], code: str) -> None:
    _require(type(value) is dict and value.keys() == fields, code)


def _digest(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _text(value: Any) -> bool:
    if type(value) is not str or not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
        return False
    try:
        value.encode("utf-8")
    except UnicodeError:
        return False
    return True


def _absolute_file_path(value: Any) -> bool:
    if not _text(value) or not value.startswith("/") or value.startswith("//"):
        return False
    path = PurePosixPath(value)
    return value != "/" and ".." not in path.parts and path.as_posix() == value


def _context(binding: dict[str, Any]) -> str:
    try:
        return content_hash({key: value for key, value in binding.items() if key != "context_id"})
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise PhaseBindingError("invalid_binding_json_data") from None


def validate_phase_binding(binding: dict[str, Any]) -> None:
    """Strict v2 shape/identity validation; no legacy or extensible host plan.

    Authentication against the designated host stores is a separate required
    caller check. In particular, opaque certificate/receipt identities cannot
    be authenticated from a content digest alone by this pure validator.
    """
    _require(type(binding) is dict, "invalid_phase_binding")
    phase = binding.get("phase")
    _require(type(phase) is str and phase in ("candidate_authorization", "task_completion"), "invalid_phase")
    completion = phase == "task_completion"
    _fields(binding, _BASE_FIELDS | (_COMPLETION_FIELDS if completion else frozenset()), "invalid_phase_fields")
    _require(type(binding["certificate_scope"]) is str
             and binding["certificate_scope"] == ("task" if completion else "candidate")
             and binding["delivery_pending"] is (not completion), "invalid_phase_scope")
    for key in ("profile_id", "task_sha256", "capability_revision", "context_id"):
        _require(_digest(binding[key]), "invalid_binding_digest")
    _require(_text(binding["verification_epoch"]), "invalid_verification_epoch")
    subject, contract = binding["subject"], binding["delivery_contract"]
    _fields(subject, _SUBJECT_FIELDS, "invalid_subject_fields")
    for key in ("candidate_id", "sha256"):
        _require(_digest(subject[key]), "invalid_subject_digest")
    _require(type(subject["size"]) is int and subject["size"] >= 0, "invalid_subject_size")
    _require(_absolute_file_path(subject["blob_path"]) and _absolute_file_path(subject["address"]), "invalid_subject_address")
    _fields(contract, _CONTRACT_FIELDS, "invalid_delivery_contract_fields")
    _require(type(contract["kind"]) is str and contract["kind"] == "single_file_bytes"
             and type(contract["writer"]) is str and contract["writer"] == "host_finalizer"
             and type(contract["physical_postconditions"]) is list
             and contract["physical_postconditions"] == _PHYSICAL_POSTCONDITIONS,
             "invalid_host_delivery_plan")
    _require(_absolute_file_path(contract["destination"])
             and type(contract["destination_source"]) is str
             and contract["destination_source"] in _DESTINATION_SOURCES,
             "invalid_delivery_destination")
    _require(contract["source_candidate_id"] == subject["candidate_id"]
             and contract["source_sha256"] == subject["sha256"]
             and type(contract["source_size"]) is int and contract["source_size"] == subject["size"],
             "delivery_source_identity_mismatch")
    _require(type(subject["address_source"]) is str
             and subject["address"] == (contract["destination"] if completion else subject["blob_path"])
             and subject["address_source"] == (contract["destination_source"] if completion else "immutable_candidate"),
             "phase_address_mismatch")
    observations = binding["observations"]
    _require(type(observations) is list, "invalid_observations")
    seen = set()
    for observation in observations:
        _fields(observation, _OBSERVATION_FIELDS, "invalid_observation_fields")
        _require(observation["candidate_id"] == subject["candidate_id"], "observation_subject_mismatch")
        for key in ("artifact_id", "effect_claim_id", "event_digest", "evaluator_id"):
            _require(_digest(observation[key]), "invalid_observation_digest")
        sequence = observation["event_seq"]
        _require(type(sequence) is int and sequence > 0 and observation["effect_ref"] == f"effect:{sequence}",
                 "invalid_observation_event")
        _require(_absolute_file_path(observation["artifact_path"])
                 and _text(observation["environment"]) and _text(observation["authority_scope"]),
                 "invalid_observation_provenance")
        _require(observation["effect_claim_id"] not in seen, "duplicate_observation")
        seen.add(observation["effect_claim_id"])
    if completion:
        for key in _COMPLETION_DIGEST_FIELDS:
            _require(_digest(binding[key]), "invalid_completion_identity")
        _require(type(binding["delivery_id"]) is str
                 and re.fullmatch(r"[0-9a-f]{32}", binding["delivery_id"]) is not None,
                 "invalid_delivery_transaction_identity")
    _require(_context(binding) == binding["context_id"], "binding_context_mismatch")


def make_authorization_binding(candidate: dict[str, Any], *, profile_id: str, task_sha256: str,
                               capability_revision: str, destination: str, destination_source: str,
                               observations: list[dict[str, Any]], verification_epoch: str) -> dict[str, Any]:
    """Bind authenticated host candidate metadata; discard labels/source hints."""
    _require(type(candidate) is dict and {"candidate_id", "sha256", "size", "blob_path"} <= candidate.keys(),
             "invalid_candidate_metadata")
    subject = {key: deepcopy(candidate[key]) for key in ("candidate_id", "sha256", "size", "blob_path")}
    subject.update(address=subject["blob_path"], address_source="immutable_candidate")
    binding = {"certificate_scope": "candidate", "phase": "candidate_authorization", "delivery_pending": True,
               "profile_id": profile_id, "task_sha256": task_sha256, "capability_revision": capability_revision,
               "verification_epoch": verification_epoch, "subject": subject, "observations": deepcopy(observations),
               "delivery_contract": {"kind": "single_file_bytes", "destination": destination,
                   "destination_source": destination_source, "source_candidate_id": subject["candidate_id"],
                   "source_sha256": subject["sha256"], "source_size": subject["size"],
                   "physical_postconditions": list(_PHYSICAL_POSTCONDITIONS), "writer": "host_finalizer"}}
    binding["context_id"] = _context(binding)
    validate_phase_binding(binding)
    return binding


def make_completion_binding(authorization_binding: dict[str, Any], *, authorization_root_id: str,
                            authorization_certificate_id: str, receipt_artifact_id: str,
                            delivery_id: str, verification_epoch: str) -> dict[str, Any]:
    """Bind a separately authenticated authorization and ACTUAL receipt artifact.

    The caller still authenticates all supplied identities against their exact
    artifacts and retained host delivery transaction, and checks current bytes.
    The transaction identity prevents a new delivery with a reused inode or
    identical receipt digest from inheriting another delivery's completion.
    This function never assumes delivery or authenticates a UUID by its shape.
    """
    validate_phase_binding(authorization_binding)
    _require(authorization_binding["phase"] == "candidate_authorization", "not_an_authorization_binding")
    _require(verification_epoch != authorization_binding["verification_epoch"], "completion_requires_fresh_epoch")
    binding = deepcopy(authorization_binding)
    contract = binding["delivery_contract"]
    binding.update(certificate_scope="task", phase="task_completion", delivery_pending=False,
                   authorization_root_id=authorization_root_id, authorization_certificate_id=authorization_certificate_id,
                   receipt_artifact_id=receipt_artifact_id, delivery_id=delivery_id, verification_epoch=verification_epoch)
    binding["subject"].update(address=contract["destination"], address_source=contract["destination_source"])
    binding["context_id"] = _context(binding)
    validate_phase_binding(binding)
    return binding


def authorization_statement(binding: dict[str, Any]) -> str:
    validate_phase_binding(binding)
    _require(binding["phase"] == "candidate_authorization", "not_an_authorization_binding")
    return canonical({"schema": AUTHORIZATION_SCHEMA, "binding": binding})


def completion_statement(binding: dict[str, Any]) -> str:
    validate_phase_binding(binding)
    _require(binding["phase"] == "task_completion", "not_a_completion_binding")
    return canonical({"schema": COMPLETION_SCHEMA, "binding": binding})


def authorization_verification_context(binding: dict[str, Any]) -> str:
    validate_phase_binding(binding)
    _require(binding["phase"] == "candidate_authorization", "not_an_authorization_binding")
    return content_hash({"context_id": binding["context_id"], "independent_root": {
        "scope": "candidate_authorization", "contract_locus": AUTHORIZATION_CONTRACT_LOCUS}})
