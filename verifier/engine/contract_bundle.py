"""Strict, immutable replay of Critic-approved obligation contracts.

This module is deliberately only a data boundary.  It does not compile a
contract, decide that a bundle is authoritative, or reuse any verification
result.  The operator supplies both the expected bundle digest and the
independently captured source/configuration identities.  A kernel using the
bundle still materializes the obligations against its current ClaimStore and
executes every effect, Oracle method, discharge, and Judger afresh.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from ..core.model import (
    BASIS_RECORD,
    BASIS_WORLD,
    CLAIM_EFFECT,
    CLAIM_FINAL,
    KINDS,
    SEVERITIES,
    SEVERITY_MUST,
    Claim,
    Obligation,
    canonical,
)


CONTRACT_BUNDLE_SCHEMA = "gv-obligation-contract-bundle/v1"
MAX_CONTRACT_BUNDLE_BYTES = 16 * 1024 * 1024
MAX_CONTRACT_CLAIMS = 50_000
MAX_CONTRACT_ENTRIES = 50_000
MAX_OBLIGATIONS_PER_ENTRY = 4_096
MAX_TOTAL_OBLIGATIONS = 100_000
MAX_CONTRACT_TEXT_BYTES = 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTITY_KEYS = frozenset({
    "pipeline_sha256", "config_sha256", "source_sha256", "graph_sha256",
})
_TOP_KEYS = frozenset({"schema", "identity", "claims", "entries"})
_CLAIM_KEYS = frozenset({"id", "kind", "text", "deps", "revises"})
_PASSED_ENTRY_KEYS = frozenset({
    "claim_id", "status", "compile_input_sha256",
    "critic_receipt_sha256", "payload", "payload_sha256", "obligations",
})
_FAILED_ENTRY_KEYS = frozenset({
    "claim_id", "status", "compile_input_sha256",
    "critic_receipt_sha256", "failure",
})
_FAILURE_KEYS = frozenset({"stage", "detail_sha256"})
_MATERIALIZED_KEYS = frozenset({
    "id", "text", "premise_scope", "required_basis", "severity", "clause",
    "refutable_by_absence",
})
_RAW_OBLIGATION_KEYS = frozenset({
    "text", "required_basis", "severity", "clause", "refutable_by_absence",
})


class ContractReplayError(ValueError):
    """A sealed contract bundle is absent, malformed, or does not match."""


@dataclass(frozen=True)
class ContractBundleIdentity:
    pipeline_sha256: str
    config_sha256: str
    source_sha256: str
    graph_sha256: str

    @classmethod
    def from_value(cls, value: Any) -> "ContractBundleIdentity":
        if isinstance(value, cls):
            identity = value
        else:
            obj = _object(value, "expected_identity", _IDENTITY_KEYS)
            identity = cls(**obj)
        for key in _IDENTITY_KEYS:
            _digest(getattr(identity, key), f"expected_identity.{key}")
        return identity

    def as_dict(self) -> dict[str, str]:
        return {
            "pipeline_sha256": self.pipeline_sha256,
            "config_sha256": self.config_sha256,
            "source_sha256": self.source_sha256,
            "graph_sha256": self.graph_sha256,
        }


@dataclass(frozen=True)
class SealedClaim:
    id: str
    kind: str
    text: str
    deps: tuple[str, ...]
    revises: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "text": self.text,
            "deps": list(self.deps), "revises": self.revises,
        }


@dataclass(frozen=True)
class MaterializedContractObligation:
    id: str
    text: str
    premise_scope: tuple[str, ...]
    required_basis: str
    severity: str
    clause: str
    refutable_by_absence: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "premise_scope": list(self.premise_scope),
            "required_basis": self.required_basis,
            "severity": self.severity,
            "clause": self.clause,
            "refutable_by_absence": self.refutable_by_absence,
        }


@dataclass(frozen=True)
class ReplayContractEntry:
    claim_id: str
    status: str
    compile_input_sha256: str
    critic_receipt_sha256: Optional[str]
    payload_sha256: Optional[str]
    _payload_json: Optional[bytes]
    obligations: tuple[MaterializedContractObligation, ...]
    failure_stage: Optional[str] = None
    failure_detail_sha256: Optional[str] = None

    @property
    def replayable(self) -> bool:
        return self.status == "passed"

    @property
    def payload(self) -> dict[str, Any]:
        if self._payload_json is None:
            raise ContractReplayError(
                f"claim {self.claim_id} has no replayable contract")
        # A caller can never mutate the bundle through the returned payload.
        return json.loads(self._payload_json.decode("utf-8"))


@dataclass(frozen=True)
class SealedContractBundle:
    """A bounded bundle whose bytes and external identities are pinned."""

    _digest: str
    _identity: ContractBundleIdentity
    _claims: tuple[SealedClaim, ...]
    _entries: Mapping[str, ReplayContractEntry]

    def __post_init__(self):
        object.__setattr__(self, "_entries",
                           MappingProxyType(dict(self._entries)))

    @property
    def sha256(self) -> str:
        return self._digest

    @property
    def identity(self) -> ContractBundleIdentity:
        return self._identity

    @property
    def claims(self) -> tuple[SealedClaim, ...]:
        return self._claims

    @property
    def replayable(self) -> bool:
        return all(entry.replayable for entry in self._entries.values())

    @property
    def failures(self) -> tuple[ReplayContractEntry, ...]:
        return tuple(entry for entry in self._entries.values()
                     if not entry.replayable)

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        *,
        expected_sha256: str,
        expected_identity: ContractBundleIdentity | Mapping[str, str],
        expected_compile_inputs: Mapping[str, str],
    ) -> "SealedContractBundle":
        """Parse only a bundle independently pinned by its caller.

        ``expected_compile_inputs`` must have been captured from the same
        fresh, complete pre-execution graph as the bundle.  It is checked
        here once, before any premise can settle; runtime lookup intentionally
        does not recompute status-sensitive compiler inputs.
        """
        if not isinstance(raw, bytes):
            raise ContractReplayError("contract bundle must be bytes")
        if len(raw) > MAX_CONTRACT_BUNDLE_BYTES:
            raise ContractReplayError(
                f"contract bundle exceeds {MAX_CONTRACT_BUNDLE_BYTES} bytes")
        pinned = _digest(expected_sha256, "expected_sha256")
        actual = hashlib.sha256(raw).hexdigest()
        if actual != pinned:
            raise ContractReplayError("contract bundle sha256 does not match")
        root = _strict_json(raw)
        root = _object(root, "bundle", _TOP_KEYS)
        if root["schema"] != CONTRACT_BUNDLE_SCHEMA:
            raise ContractReplayError("unsupported contract bundle schema")

        expected = ContractBundleIdentity.from_value(expected_identity)
        identity_obj = _object(root["identity"], "bundle.identity",
                               _IDENTITY_KEYS)
        identity = ContractBundleIdentity.from_value(identity_obj)
        if identity != expected:
            raise ContractReplayError("contract bundle identity does not match")

        claims = _parse_claims(root["claims"])
        if claim_graph_sha256(claims) != identity.graph_sha256:
            raise ContractReplayError(
                "contract bundle claim graph does not match graph_sha256")
        claim_by_id = {claim.id: claim for claim in claims}
        expected_inputs = _compile_input_mapping(
            expected_compile_inputs, claim_by_id)
        entries = _parse_entries(root["entries"], claim_by_id,
                                 expected_inputs)
        return cls(_digest=actual, _identity=identity, _claims=claims,
                   _entries=entries)

    def require_replayable(self) -> None:
        if not self.replayable:
            claims = ", ".join(entry.claim_id[:12]
                               for entry in self.failures[:8])
            raise ContractReplayError(
                f"contract bundle contains failed compilation entries: {claims}")

    def lookup(self, claim_id: str) -> ReplayContractEntry:
        entry = self._entries.get(claim_id)
        if entry is None:
            raise ContractReplayError(
                f"sealed contract missing for claim {str(claim_id)[:12]}")
        if not entry.replayable:
            raise ContractReplayError(
                f"sealed contract for claim {claim_id[:12]} is a "
                f"{entry.status} report, not a replayable contract")
        return entry

    def validate_materialized(
        self, entry: ReplayContractEntry, obligations: Sequence[Obligation],
    ) -> None:
        if self._entries.get(entry.claim_id) is not entry:
            raise ContractReplayError("contract entry does not belong to bundle")
        actual = tuple(_materialized_from_obligation(item)
                       for item in obligations)
        if actual != entry.obligations:
            raise ContractReplayError(
                f"materialized obligations do not match sealed order for "
                f"claim {entry.claim_id[:12]}")


def claim_graph_sha256(
    claims: Sequence[Claim] | Sequence[SealedClaim] | Sequence[Mapping[str, Any]],
) -> str:
    """Canonical identity used by the bundle's explicit ordered graph."""
    records = []
    for claim in claims:
        if isinstance(claim, SealedClaim):
            record = claim.as_dict()
        elif isinstance(claim, Claim):
            record = {
                "id": claim.id, "kind": claim.kind, "text": claim.text,
                "deps": list(claim.deps), "revises": claim.revises,
            }
        else:
            record = copy.deepcopy(dict(claim))
        records.append(record)
    return hashlib.sha256(canonical(records).encode("utf-8")).hexdigest()


def contract_record_sha256(value: Any) -> str:
    """Digest a bounded setup receipt using the verifier's canonical JSON."""
    return _value_sha256(value)


def passed_contract_entry(
    *, claim_id: str, compile_input_sha256: str,
    critic_receipt_sha256: str, payload: Mapping[str, Any],
    obligations: Sequence[Obligation] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build, but do not authorize, one successful JSON entry.

    The returned mapping is only construction convenience.  It becomes
    usable after the complete bundle is independently pinned and accepted by
    :meth:`SealedContractBundle.from_bytes`.
    """
    normalized_payload = _parse_payload(copy.deepcopy(dict(payload)),
                                        "entry.payload")
    records = [
        _materialized_from_obligation(item).as_dict()
        if isinstance(item, Obligation) else copy.deepcopy(dict(item))
        for item in obligations
    ]
    return {
        "claim_id": _digest(claim_id, "entry.claim_id"),
        "status": "passed",
        "compile_input_sha256": _digest(
            compile_input_sha256, "entry.compile_input_sha256"),
        "critic_receipt_sha256": _digest(
            critic_receipt_sha256, "entry.critic_receipt_sha256"),
        "payload": normalized_payload,
        "payload_sha256": _value_sha256(normalized_payload),
        "obligations": records,
    }


def failed_contract_entry(
    *, claim_id: str, compile_input_sha256: str, stage: str,
    detail_sha256: str, critic_receipt_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Build an explicit non-replayable setup failure record."""
    if stage not in ("compiler", "critic"):
        raise ContractReplayError("failure stage must be compiler or critic")
    if critic_receipt_sha256 is not None:
        critic_receipt_sha256 = _digest(
            critic_receipt_sha256, "entry.critic_receipt_sha256")
    return {
        "claim_id": _digest(claim_id, "entry.claim_id"),
        "status": "compile_failure",
        "compile_input_sha256": _digest(
            compile_input_sha256, "entry.compile_input_sha256"),
        "critic_receipt_sha256": critic_receipt_sha256,
        "failure": {
            "stage": stage,
            "detail_sha256": _digest(detail_sha256,
                                      "entry.failure.detail_sha256"),
        },
    }


def build_contract_bundle_bytes(
    *, identity: ContractBundleIdentity | Mapping[str, str],
    claims: Sequence[Claim] | Sequence[SealedClaim] | Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
) -> bytes:
    """Canonically encode a bundle and mechanically self-check its shape.

    This helper is for the compile-only diagnostic setup.  Its output is not
    trusted until an operator records the returned bytes' sha256 outside the
    bundle and later supplies that digest to :meth:`from_bytes`.
    """
    normalized_identity = ContractBundleIdentity.from_value(identity)
    claim_rows = []
    for item in claims:
        if isinstance(item, SealedClaim):
            row = item.as_dict()
        elif isinstance(item, Claim):
            row = {
                "id": item.id, "kind": item.kind, "text": item.text,
                "deps": list(item.deps), "revises": item.revises,
            }
        else:
            row = copy.deepcopy(dict(item))
        claim_rows.append(row)
    if claim_graph_sha256(claim_rows) != normalized_identity.graph_sha256:
        raise ContractReplayError(
            "builder claims do not match identity.graph_sha256")
    entry_rows = [copy.deepcopy(dict(item)) for item in entries]
    value = {
        "schema": CONTRACT_BUNDLE_SCHEMA,
        "identity": normalized_identity.as_dict(),
        "claims": claim_rows,
        "entries": entry_rows,
    }
    raw = canonical(value).encode("utf-8")
    if len(raw) > MAX_CONTRACT_BUNDLE_BYTES:
        raise ContractReplayError(
            f"contract bundle exceeds {MAX_CONTRACT_BUNDLE_BYTES} bytes")
    expected_inputs = {
        item.get("claim_id"): item.get("compile_input_sha256")
        for item in entry_rows
    }
    SealedContractBundle.from_bytes(
        raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_identity=normalized_identity,
        expected_compile_inputs=expected_inputs)
    return raw


def _strict_json(raw: bytes) -> Any:
    def object_pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key!r}")
            out[key] = value
        return out

    def reject_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs,
                          parse_constant=reject_constant)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError,
            RecursionError) as error:
        raise ContractReplayError(
            f"contract bundle is not strict UTF-8 JSON: {error}") from error


def _object(value: Any, path: str, keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractReplayError(f"{path} must be an object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ContractReplayError(
            f"{path} fields mismatch; missing={missing}, extra={extra}")
    return value


def _list(value: Any, path: str, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise ContractReplayError(f"{path} must be an array")
    if len(value) > maximum:
        raise ContractReplayError(f"{path} exceeds {maximum} entries")
    return value


def _digest(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ContractReplayError(f"{path} must be a lowercase sha256")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ContractReplayError(f"{path} must be a string")
    if "\x00" in value:
        raise ContractReplayError(f"{path} contains a NUL byte")
    if len(value.encode("utf-8")) > MAX_CONTRACT_TEXT_BYTES:
        raise ContractReplayError(
            f"{path} exceeds {MAX_CONTRACT_TEXT_BYTES} UTF-8 bytes")
    return value


def _parse_claims(value: Any) -> tuple[SealedClaim, ...]:
    rows = _list(value, "bundle.claims", MAX_CONTRACT_CLAIMS)
    if not rows:
        raise ContractReplayError("bundle.claims must not be empty")
    claims = []
    seen = set()
    final_count = 0
    for index, value in enumerate(rows):
        path = f"bundle.claims[{index}]"
        obj = _object(value, path, _CLAIM_KEYS)
        cid = _digest(obj["id"], f"{path}.id")
        if cid in seen:
            raise ContractReplayError(f"duplicate claim id at {path}")
        kind = obj["kind"]
        if kind not in KINDS:
            raise ContractReplayError(f"{path}.kind is not a claim kind")
        text = _text(obj["text"], f"{path}.text")
        deps_raw = _list(obj["deps"], f"{path}.deps", MAX_CONTRACT_CLAIMS)
        deps = tuple(_digest(dep, f"{path}.deps") for dep in deps_raw)
        if len(set(deps)) != len(deps):
            raise ContractReplayError(f"{path}.deps contains duplicates")
        if any(dep not in seen for dep in deps):
            raise ContractReplayError(
                f"{path}.deps must refer to earlier sealed claims")
        revises = obj["revises"]
        if revises is not None:
            revises = _digest(revises, f"{path}.revises")
            if revises not in seen:
                raise ContractReplayError(
                    f"{path}.revises must refer to an earlier sealed claim")
        claim = Claim(kind, text, deps=deps, revises=revises)
        if claim.id != cid:
            raise ContractReplayError(f"{path}.id does not match claim content")
        if kind == CLAIM_FINAL:
            final_count += 1
        claims.append(SealedClaim(cid, kind, text, deps, revises))
        seen.add(cid)
    if final_count != 1:
        raise ContractReplayError("bundle graph must contain exactly one final claim")
    return tuple(claims)


def _compile_input_mapping(value: Mapping[str, str],
                           claims: Mapping[str, SealedClaim]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ContractReplayError("expected_compile_inputs must be a mapping")
    final = next(claim for claim in claims.values()
                 if claim.kind == CLAIM_FINAL)
    slice_ids = set()
    frontier = [final.id]
    while frontier:
        current = frontier.pop()
        if current in slice_ids:
            continue
        slice_ids.add(current)
        frontier.extend(claims[current].deps)
    expected_ids = {cid for cid in slice_ids
                    if claims[cid].kind != CLAIM_EFFECT}
    actual_ids = set(value)
    if actual_ids != expected_ids:
        raise ContractReplayError(
            "expected_compile_inputs must exactly cover non-effect claims")
    return {cid: _digest(value[cid],
                         f"expected_compile_inputs[{cid[:12]}]")
            for cid in expected_ids}


def _parse_entries(value: Any, claims: Mapping[str, SealedClaim],
                   expected_inputs: Mapping[str, str]) \
        -> Mapping[str, ReplayContractEntry]:
    rows = _list(value, "bundle.entries", MAX_CONTRACT_ENTRIES)
    entries: dict[str, ReplayContractEntry] = {}
    total_obligations = 0
    for index, value in enumerate(rows):
        path = f"bundle.entries[{index}]"
        if not isinstance(value, dict):
            raise ContractReplayError(f"{path} must be an object")
        status = value.get("status")
        keys = (_PASSED_ENTRY_KEYS if status == "passed"
                else _FAILED_ENTRY_KEYS if status == "compile_failure"
                else None)
        if keys is None:
            raise ContractReplayError(f"{path}.status is unsupported")
        obj = _object(value, path, keys)
        cid = _digest(obj["claim_id"], f"{path}.claim_id")
        claim = claims.get(cid)
        if claim is None:
            raise ContractReplayError(f"{path} refers to an unknown claim")
        if claim.kind == CLAIM_EFFECT:
            raise ContractReplayError("effect claims must not carry contracts")
        if cid not in expected_inputs:
            raise ContractReplayError(
                f"{path} claim is outside the selected final slice")
        if cid in entries:
            raise ContractReplayError(f"duplicate contract entry for {cid[:12]}")
        compile_digest = _digest(obj["compile_input_sha256"],
                                 f"{path}.compile_input_sha256")
        if compile_digest != expected_inputs[cid]:
            raise ContractReplayError(
                f"{path}.compile_input_sha256 does not match operator input")
        if status == "compile_failure":
            critic_digest = obj["critic_receipt_sha256"]
            if critic_digest is not None:
                critic_digest = _digest(critic_digest,
                                         f"{path}.critic_receipt_sha256")
            failure = _object(obj["failure"], f"{path}.failure", _FAILURE_KEYS)
            stage = failure["stage"]
            if stage not in ("compiler", "critic"):
                raise ContractReplayError(f"{path}.failure.stage is unsupported")
            if ((stage == "compiler" and critic_digest is not None)
                    or (stage == "critic" and critic_digest is None)):
                raise ContractReplayError(
                    f"{path}.critic_receipt_sha256 does not match failure stage")
            detail = _digest(failure["detail_sha256"],
                             f"{path}.failure.detail_sha256")
            entries[cid] = ReplayContractEntry(
                cid, status, compile_digest, critic_digest, None, None, (),
                failure_stage=stage, failure_detail_sha256=detail)
            continue

        critic_digest = _digest(obj["critic_receipt_sha256"],
                                 f"{path}.critic_receipt_sha256")
        payload = _parse_payload(obj["payload"], f"{path}.payload")
        payload_sha = _digest(obj["payload_sha256"],
                              f"{path}.payload_sha256")
        if _value_sha256(payload) != payload_sha:
            raise ContractReplayError(f"{path}.payload_sha256 does not match")
        materialized_rows = _list(obj["obligations"], f"{path}.obligations",
                                  MAX_OBLIGATIONS_PER_ENTRY)
        expected_materialized = _materialize_from_graph(claim, claims, payload)
        parsed_materialized = tuple(
            _parse_materialized(item, f"{path}.obligations[{offset}]", cid)
            for offset, item in enumerate(materialized_rows))
        if parsed_materialized != expected_materialized:
            raise ContractReplayError(
                f"{path}.obligations do not match raw payload and graph")
        total_obligations += len(parsed_materialized)
        if total_obligations > MAX_TOTAL_OBLIGATIONS:
            raise ContractReplayError(
                f"bundle exceeds {MAX_TOTAL_OBLIGATIONS} obligations")
        entries[cid] = ReplayContractEntry(
            cid, status, compile_digest, critic_digest, payload_sha,
            canonical(payload).encode("utf-8"), parsed_materialized)
    if set(entries) != set(expected_inputs):
        raise ContractReplayError(
            "bundle entries must exactly cover every non-effect claim")
    return entries


def _parse_payload(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path, frozenset({"obligations"}))
    rows = _list(obj["obligations"], f"{path}.obligations",
                 MAX_OBLIGATIONS_PER_ENTRY)
    if not rows:
        raise ContractReplayError(f"{path}.obligations must not be empty")
    obligations = []
    for index, value in enumerate(rows):
        item_path = f"{path}.obligations[{index}]"
        if not isinstance(value, dict):
            raise ContractReplayError(f"{item_path} must be an object")
        if not set(value) <= _RAW_OBLIGATION_KEYS:
            raise ContractReplayError(f"{item_path} has unsupported fields")
        if not {"text", "required_basis"} <= set(value):
            raise ContractReplayError(
                f"{item_path} requires text and required_basis")
        text = _text(value["text"], f"{item_path}.text")
        basis = value["required_basis"]
        if basis not in (BASIS_WORLD, BASIS_RECORD):
            raise ContractReplayError(f"{item_path}.required_basis is invalid")
        out = {"text": text, "required_basis": basis}
        if "severity" in value:
            if value["severity"] not in SEVERITIES:
                raise ContractReplayError(f"{item_path}.severity is invalid")
            out["severity"] = value["severity"]
        if "clause" in value:
            out["clause"] = _text(value["clause"], f"{item_path}.clause")
        if "refutable_by_absence" in value:
            if not isinstance(value["refutable_by_absence"], bool):
                raise ContractReplayError(
                    f"{item_path}.refutable_by_absence must be boolean")
            out["refutable_by_absence"] = value["refutable_by_absence"]
        obligations.append(out)
    return {"obligations": obligations}


def _transitive_premises(claim: SealedClaim,
                         claims: Mapping[str, SealedClaim]) -> tuple[str, ...]:
    if claim.kind == CLAIM_FINAL:
        return ()
    seen = set()
    frontier = list(claim.deps)
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(claims[current].deps)
    return tuple(sorted(seen))


def _materialize_from_graph(
    claim: SealedClaim, claims: Mapping[str, SealedClaim], payload: Mapping[str, Any],
) -> tuple[MaterializedContractObligation, ...]:
    scope = _transitive_premises(claim, claims)
    obligations = [
        Obligation(
            claim.id, item["text"], item.get("required_basis", BASIS_RECORD),
            severity=(item.get("severity")
                      if item.get("severity") in SEVERITIES else SEVERITY_MUST),
            clause=str(item.get("clause") or ""),
            refutable_by_absence=bool(item.get("refutable_by_absence", False)),
            premise_scope=scope,
        )
        for item in payload["obligations"]
    ]
    if claim.kind != CLAIM_FINAL:
        direct_premise_texts = {
            claims[dep].text for dep in claim.deps
            if claims[dep].kind != CLAIM_EFFECT
        }
        residual = [item for item in obligations
                    if item.text not in direct_premise_texts]
        dropped = [item for item in obligations
                   if item.text in direct_premise_texts]
        if dropped and residual:
            obligations = residual
    return tuple(_materialized_from_obligation(item) for item in obligations)


def _parse_materialized(value: Any, path: str,
                        claim_id: str) -> MaterializedContractObligation:
    obj = _object(value, path, _MATERIALIZED_KEYS)
    oid = _digest(obj["id"], f"{path}.id")
    text = _text(obj["text"], f"{path}.text")
    raw_scope = _list(obj["premise_scope"], f"{path}.premise_scope",
                      MAX_CONTRACT_CLAIMS)
    scope = tuple(_digest(item, f"{path}.premise_scope") for item in raw_scope)
    if scope != tuple(sorted(set(scope))):
        raise ContractReplayError(
            f"{path}.premise_scope must be sorted and unique")
    basis = obj["required_basis"]
    if basis not in (BASIS_WORLD, BASIS_RECORD):
        raise ContractReplayError(f"{path}.required_basis is invalid")
    severity = obj["severity"]
    if severity not in SEVERITIES:
        raise ContractReplayError(f"{path}.severity is invalid")
    clause = _text(obj["clause"], f"{path}.clause")
    absence = obj["refutable_by_absence"]
    if not isinstance(absence, bool):
        raise ContractReplayError(
            f"{path}.refutable_by_absence must be boolean")
    obligation = Obligation(
        claim_id, text, basis, severity=severity, clause=clause,
        refutable_by_absence=absence, premise_scope=scope)
    if obligation.id != oid:
        raise ContractReplayError(f"{path}.id does not match obligation content")
    return _materialized_from_obligation(obligation)


def _materialized_from_obligation(
    item: Obligation,
) -> MaterializedContractObligation:
    return MaterializedContractObligation(
        item.id, item.text, tuple(item.premise_scope), item.required_basis,
        item.severity, item.clause, item.refutable_by_absence)


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()
