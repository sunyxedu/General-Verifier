"""A narrow, controller-only acceptance projection of a settled GV certificate.

This does not change GV's status/strength fold or turn a positive evaluator
response into a verdict. Every obligation, including SHOULD, must still hold.
Only the evidence-strength fold for this additional acceptance scope excludes
advice: each MUST must have explicit, decisive world evidence. The original
certificate and all advisory evidence remain linked and unchanged. In
particular a refuted, partial, stale, or unverified SHOULD never authorizes
delivery through this module.

Threat boundary: inputs are controller-authenticated certificate data, not
solver text or an executor's self-declared artifact. The caller MUST verify
the full artifact hash against its designated certificate ID, authenticate the
host binding/retained observations, and compare the certificate with the
currently settled store (including complete evidence). It must repeat these
checks under the candidate lease immediately before materialization and
authenticate both certificate artifacts for receipt replay. This pure function
does not inspect the filesystem, settle the store, authenticate network data,
prove a Judger's reasoning, or establish that the live world is unchanged.

Missing/legacy evidence metadata fails closed. Inherited graph evidence lacks
the direct-discharge evidence identity used here and cannot be silently
upgraded by this projection. Error reasons contain static codes only, never
candidate contents, task text, private paths, or credential configuration.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from verifier.core.model import (
    BASIS_RECORD,
    BASIS_WORLD,
    CLAIM_FINAL,
    OUTCOME_HOLDS,
    SEVERITY_MUST,
    SEVERITY_SHOULD,
    STATUS_VERIFIED,
    STRENGTH_DECISIVE,
    STRENGTHS,
    Claim,
    Evidence,
    Obligation,
    canonical,
    circular_inputs,
    content_hash,
    weakest_strength,
)


SCHEMA = "gv-cyber-candidate-acceptance-v1"
ACCEPTANCE_SCOPE = "candidate_must_requirements"


class AcceptanceError(ValueError):
    """An unauthorizable projection, with a non-sensitive reason code."""

    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise AcceptanceError(code)


def _digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _build(full: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(full, dict) and isinstance(binding, dict), "invalid_shape")
    _require(binding.get("certificate_scope") == "candidate"
             and binding.get("delivery_pending") is True, "invalid_binding_scope")
    for key in ("profile_id", "task_sha256", "context_id"):
        _require(_digest(binding.get(key)), "invalid_binding_identity")
    _require(isinstance(binding.get("verification_epoch"), str)
             and bool(binding["verification_epoch"]), "missing_binding_epoch")
    _require(content_hash({key: value for key, value in binding.items()
                           if key != "context_id"}) == binding["context_id"],
             "binding_context_mismatch")
    subject = binding.get("subject")
    _require(isinstance(subject, dict), "invalid_subject")
    _require(_digest(subject.get("candidate_id")) and _digest(subject.get("sha256")),
             "invalid_subject_identity")
    _require(type(subject.get("size")) is int and subject["size"] >= 0,
             "invalid_subject_size")
    for key in ("blob_path", "address", "address_source"):
        _require(isinstance(subject.get(key), str) and bool(subject[key]),
                 "invalid_subject_address")
    _require(subject["address_source"] in ("host_required", "candidate_proposal"),
             "invalid_subject_address")
    observations = binding.get("observations")
    _require(isinstance(observations, list), "invalid_observations")
    _require(all(isinstance(item, dict) and _digest(item.get("effect_claim_id"))
                 for item in observations), "invalid_observations")
    expected_metadata = {
        "profile": "cyber", "profile_id": binding["profile_id"],
        "certificate_scope": "candidate", "task_sha256": binding["task_sha256"],
        "context_id": binding["context_id"], "binding": binding, "delivery_pending": True,
    }
    _require(full.get("profile_metadata") == expected_metadata, "binding_mismatch")
    _require(full.get("verdict") in ("verified", "supported"), "full_verdict_not_eligible")
    _require(full.get("verdict_strength") in STRENGTHS, "invalid_full_strength")
    statement = canonical({"schema": "gv-cyber-candidate-v1", "binding": binding})
    final_id = Claim(CLAIM_FINAL, statement,
                     tuple(item["effect_claim_id"] for item in observations)).id
    _require(full.get("final") == final_id, "final_identity_mismatch")
    entries = full.get("slice")
    _require(isinstance(entries, list) and all(isinstance(item, dict) for item in entries),
             "invalid_slice")
    finals = [item for item in entries if item.get("id") == final_id]
    _require(len(finals) == 1, "ambiguous_final")
    final = finals[0]
    _require(final.get("kind") == CLAIM_FINAL and final.get("text") == statement,
             "final_statement_mismatch")
    _require(final.get("status") == STATUS_VERIFIED, "final_not_verified")
    obligations = final.get("obligations")
    _require(isinstance(obligations, list) and bool(obligations), "missing_obligations")
    must, advice, strengths, bases, seen = [], [], [], [], set()
    for node in obligations:
        _require(isinstance(node, dict) and _digest(node.get("id")), "invalid_obligation")
        _require(node["id"] not in seen, "duplicate_obligation")
        seen.add(node["id"])
        _require(node.get("severity") in (SEVERITY_MUST, SEVERITY_SHOULD),
                 "invalid_severity")
        _require(node.get("status") == STATUS_VERIFIED, "obligation_not_verified")
        _require(node.get("strength") in STRENGTHS, "invalid_obligation_strength")
        _require(node.get("basis") in (BASIS_RECORD, BASIS_WORLD), "invalid_obligation_basis")
        _require(isinstance(node.get("text"), str) and bool(node["text"])
                 and isinstance(node.get("clause"), str), "invalid_obligation_statement")
        history = node.get("evidence")
        _require(isinstance(history, list) and bool(history)
                 and all(isinstance(item, dict) for item in history), "missing_settled_evidence")
        evidence = history[-1]
        floor = evidence.get("required_basis")
        _require(floor in (BASIS_RECORD, BASIS_WORLD), "invalid_evidence_floor")
        _require(evidence.get("outcome") == OUTCOME_HOLDS, "evidence_not_holds")
        _require(evidence.get("basis") == node["basis"]
                 and evidence.get("strength") == node["strength"], "evidence_grade_mismatch")
        _require(floor != BASIS_WORLD or node["basis"] == BASIS_WORLD, "evidence_below_floor")
        _require(evidence.get("obligation") == node["text"]
                 and evidence.get("severity") == node["severity"]
                 and evidence.get("clause") == node["clause"], "evidence_statement_mismatch")
        _require(evidence.get("assumed_premises") == [], "assumed_premises")
        _require(isinstance(evidence.get("bridges"), list), "invalid_bridges")
        inputs = evidence.get("inputs")
        _require(isinstance(inputs, list) and bool(inputs)
                 and all(isinstance(item, dict)
                         and item.get("provenance") in
                         ("task_given", "delivered", "ledger", "verifier", "public")
                         and item.get("role") in ("subject", "input", "authority")
                         and isinstance(item.get("ref"), str) and bool(item["ref"])
                         for item in inputs), "invalid_evidence_inputs")
        _require(not circular_inputs(inputs), "delivered_authority")
        artifact = evidence.get("artifact")
        _require(_digest(artifact) and _digest(evidence.get("evidence_id"))
                 and isinstance(evidence.get("instrument"), str)
                 and bool(evidence["instrument"]), "invalid_evidence_identity")
        expected_node = Obligation(
            final_id, node["text"], floor, severity=node["severity"], clause=node["clause"],
            refutable_by_absence=node.get("refutable_by_absence"), premise_scope=(),
            verification_context=binding["context_id"])
        _require(type(node.get("refutable_by_absence")) is bool,
                 "invalid_absence_rule")
        _require(expected_node.id == node["id"], "obligation_context_mismatch")
        expected_evidence = Evidence(node["id"], evidence["instrument"], OUTCOME_HOLDS,
                                     {"artifact": artifact}, evidence.get("witness"))
        _require(expected_evidence.id == evidence["evidence_id"], "evidence_identity_mismatch")
        link = {"obligation_id": node["id"], "evidence_id": evidence["evidence_id"],
                "artifact_id": artifact, "required_basis": floor,
                "basis": node["basis"], "strength": node["strength"]}
        if node["severity"] == SEVERITY_MUST:
            _require(node["basis"] == BASIS_WORLD and node["strength"] == STRENGTH_DECISIVE,
                     "must_not_decisive_world")
            _require(evidence["bridges"] == [], "decisive_must_has_bridges")
            must.append(link)
        else:
            advice.append(link)
        strengths.append(node["strength"])
        bases.append(node["basis"])
    _require(bool(must), "missing_must_obligation")
    folded = weakest_strength(strengths)
    _require(final.get("strength") == folded and full["verdict_strength"] == folded,
             "full_strength_mismatch")
    _require(final.get("basis") == (BASIS_WORLD if all(b == BASIS_WORLD for b in bases)
                                   else BASIS_RECORD), "full_basis_mismatch")
    _require(full["verdict"] == ("verified" if folded == STRENGTH_DECISIVE else "supported"),
             "full_verdict_strength_mismatch")
    return {"schema": SCHEMA, "final": final_id, "verdict": "verified",
            "verdict_strength": STRENGTH_DECISIVE,
            "full_certificate_id": content_hash(full), "full_verdict": full["verdict"],
            "profile_metadata": {**deepcopy(expected_metadata), "acceptance_scope": ACCEPTANCE_SCOPE},
            "must_evidence": must, "advisory_evidence": advice}


def build_candidate_acceptance(full_certificate: dict[str, Any],
                               expected_binding: dict[str, Any]) -> dict[str, Any]:
    """Project authenticated controller data, or raise a safe AcceptanceError.

    No caller-owned objects are mutated. Hash authentication, current-store
    comparison, complete artifact-chain auditing and delivery TOCTOU protection
    remain mandatory caller responsibilities; see the module threat boundary.
    """
    try:
        return _build(full_certificate, expected_binding)
    except AcceptanceError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError):
        raise AcceptanceError("invalid_shape") from None
