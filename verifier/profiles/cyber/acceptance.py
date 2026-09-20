"""Controller-only delivery authorization, never a task-completion certificate.

This does not change GV's status/strength fold or turn a positive evaluator
response into a verdict. Every obligation, including SHOULD, must still hold.
Only the evidence-strength fold for this additional acceptance scope excludes
advice: each MUST must have explicit, decisive world evidence. The original
independent ASSERTION certificate and all advisory evidence remain linked unchanged. In
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
    CLAIM_ASSERTION,
    CLAIM_EFFECT,
    OUTCOME_HOLDS,
    SEVERITY_MUST,
    SEVERITY_SHOULD,
    STATUS_VERIFIED,
    STRENGTH_DECISIVE,
    STRENGTHS,
    Claim,
    Evidence,
    Obligation,
    circular_inputs,
    content_hash,
    weakest_strength,
)


SCHEMA = "gv-cyber-candidate-acceptance-v2"
ACCEPTANCE_SCOPE = "candidate_must_requirements"
ROOT_SCOPE = "candidate_authorization"
INDEPENDENT_SCHEMA = "gv-independent-claim-root-v1"


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


def _validate_slice(entries, root, root_id, observations):
    """The host authorization root depends only on its authenticated Effects."""
    effect_ids = [item["effect_claim_id"] for item in observations]
    _require(len(effect_ids) == len(set(effect_ids)), "duplicate_observation")
    _require(all(_digest(item.get("id")) for item in entries), "invalid_slice_identity")
    identities = [item["id"] for item in entries]
    _require(len(identities) == len(set(identities)), "duplicate_slice_identity")
    _require(set(identities) == {root_id, *effect_ids}, "incomplete_authorization_slice")
    _require(isinstance(root.get("deps"), list)
             and all(_digest(dep) for dep in root["deps"])
             and sorted(root["deps"]) == sorted(effect_ids), "root_dependencies_mismatch")
    for entry in entries:
        _require(entry.get("status") == STATUS_VERIFIED
                 and entry.get("effective_status") == STATUS_VERIFIED
                 and entry.get("open_premises") == []
                 and "premise_refuted" in entry and entry["premise_refuted"] is None,
                 "unsettled_authorization_slice")
        if entry["id"] == root_id:
            continue
        _require(entry.get("kind") == CLAIM_EFFECT and entry.get("deps") == []
                 and isinstance(entry.get("text"), str)
                 and Claim(CLAIM_EFFECT, entry["text"]).id == entry["id"],
                 "invalid_authorization_effect")
        _require(entry.get("basis") in (BASIS_WORLD, BASIS_RECORD)
                 and entry.get("effective_basis") == entry["basis"]
                 and "strength" in entry
                 and (entry["strength"] is None or entry["strength"] in STRENGTHS)
                 and entry.get("effective_strength") == (entry["strength"] or STRENGTH_DECISIVE),
                 "invalid_effective_effect")
        history = entry.get("evidence")
        _require(isinstance(history, list) and bool(history)
                 and all(isinstance(item, dict) for item in history)
                 and history[-1].get("outcome") == OUTCOME_HOLDS
                 and entry.get("obligations") == [], "incomplete_effect_evidence")


def _build(full: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(full, dict) and isinstance(binding, dict), "invalid_shape")
    _require(binding.get("certificate_scope") == "candidate"
             and binding.get("phase") == ROOT_SCOPE
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
    _require(_digest(subject.get("candidate_id"))
             and subject.get("sha256") == subject["candidate_id"],
             "invalid_subject_identity")
    _require(type(subject.get("size")) is int and subject["size"] >= 0,
             "invalid_subject_size")
    for key in ("blob_path", "address", "address_source"):
        _require(isinstance(subject.get(key), str) and bool(subject[key]),
                 "invalid_subject_address")
    _require(subject["address_source"] == "immutable_candidate"
             and subject["address"] == subject["blob_path"],
             "invalid_subject_address")
    # The phase module owns the strict, host-only single-file delivery
    # contract. Do not duplicate or weaken it in this projection.
    from .phases import (
        authorization_statement, authorization_verification_context, validate_phase_binding,
    )
    try:
        validate_phase_binding(binding)
        verification_context = authorization_verification_context(binding)
        statement = authorization_statement(binding)
    except (TypeError, ValueError, KeyError, AttributeError):
        raise AcceptanceError("invalid_authorization_binding") from None
    _require(_digest(verification_context), "invalid_authorization_context")
    observations = binding.get("observations")
    _require(isinstance(observations, list), "invalid_observations")
    _require(all(isinstance(item, dict) and _digest(item.get("effect_claim_id"))
                 for item in observations), "invalid_observations")
    expected_metadata = {
        "profile": "cyber", "profile_id": binding["profile_id"],
        "certificate_scope": "candidate", "phase": ROOT_SCOPE,
        "task_sha256": binding["task_sha256"],
        "context_id": binding["context_id"], "binding": binding, "delivery_pending": True,
    }
    _require(full.get("schema") == INDEPENDENT_SCHEMA
             and full.get("root_scope") == ROOT_SCOPE and "final" not in full,
             "invalid_authorization_certificate")
    # Python dict equality treats True == 1 and 7 == 7.0. The host binding is
    # a typed content identity: compare canonical hashes, not loose equality.
    _require(content_hash(full.get("profile_metadata")) == content_hash(expected_metadata),
             "binding_mismatch")
    _require(full.get("verdict") in ("verified", "supported"), "full_verdict_not_eligible")
    _require(full.get("verdict_strength") in STRENGTHS, "invalid_full_strength")
    root_id = Claim(CLAIM_ASSERTION, statement,
                     tuple(item["effect_claim_id"] for item in observations)).id
    _require(full.get("root") == root_id, "root_identity_mismatch")
    entries = full.get("slice")
    _require(isinstance(entries, list) and all(isinstance(item, dict) for item in entries),
             "invalid_slice")
    roots = [item for item in entries if item.get("id") == root_id]
    _require(len(roots) == 1, "ambiguous_root")
    root = roots[0]
    _require(root.get("kind") == CLAIM_ASSERTION and root.get("text") == statement,
             "root_statement_mismatch")
    _require(root.get("status") == STATUS_VERIFIED, "root_not_verified")
    for source in (full, root):
        _require(source.get("effective_status") == STATUS_VERIFIED
                 and source.get("open_premises") == []
                 and "premise_refuted" in source and source["premise_refuted"] is None,
                 "root_not_independent")
    _require(full.get("effective_basis") == root.get("effective_basis")
             and full.get("effective_strength") == root.get("effective_strength"),
             "full_effective_mismatch")
    _validate_slice(entries, root, root_id, observations)
    obligations = root.get("obligations")
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
        _require(node.get("premise_scope") == [], "nonempty_premise_scope")
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
        _require(isinstance(evidence.get("bridges"), list)
                 and all(isinstance(bridge, str) and bool(bridge)
                         for bridge in evidence["bridges"]), "invalid_bridges")
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
            root_id, node["text"], floor, severity=node["severity"], clause=node["clause"],
            refutable_by_absence=node.get("refutable_by_absence"), premise_scope=(),
            verification_context=verification_context)
        _require(type(node.get("refutable_by_absence")) is bool,
                 "invalid_absence_rule")
        _require(expected_node.id == node["id"], "obligation_context_mismatch")
        _require(node.get("verification_context") == verification_context,
                 "obligation_context_mismatch")
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
    _require(root.get("strength") == folded and full["verdict_strength"] == folded
             and root.get("effective_strength") == folded,
             "full_strength_mismatch")
    folded_basis = BASIS_WORLD if all(b == BASIS_WORLD for b in bases) else BASIS_RECORD
    _require(root.get("basis") == folded_basis
             and root.get("effective_basis") == folded_basis
             and full.get("verdict_basis") == folded_basis, "full_basis_mismatch")
    _require(full["verdict"] == ("verified" if folded == STRENGTH_DECISIVE else "supported"),
             "full_verdict_strength_mismatch")
    return {"schema": SCHEMA, "root": root_id, "root_scope": ROOT_SCOPE,
            "verdict": "verified",
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
