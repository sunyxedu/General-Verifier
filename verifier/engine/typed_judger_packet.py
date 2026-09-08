"""Complete, replay-bound evidence for a fresh independent Judger.

This is a transport seam, not a verdict shortcut.  A packet is available
only for a small allowlist of verifier-owned deterministic adapters after
their ordinary evidence envelope has passed the registry validator and an
exact fresh replay.  The same checks and another exact replay run after the
Judger call.  Any uncertainty returns to the unchanged Judger path.

The packet deliberately contains no earlier Judger verdict or reasoning.
It gives the new Judger the task authority, exact obligation and MethodIR,
registered adapter contract, complete bounded *discharge evidence envelope*,
and the verifier's mechanical replay receipt.  It does not copy the original
world file or ledger prefix bytes. Semantic sufficiency remains solely the
fresh Judger's decision.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..core.model import Obligation, canonical, content_hash
from ..execution.adapters import (
    AdapterInvocation,
    AdapterRegistry,
    AdapterRoutingDischarger,
)
from ..execution.discharge import DischargeResult
from ..execution.instruments import authoritative_task_context
from .judger_prefetch import (
    JudgerEvidencePacket,
    build_judger_evidence_packet,
)


TYPED_JUDGER_PACKET_PROTOCOL = (
    "gv-typed-adapter-judger-packet-v2;"
    "evidence-cap=16384;render-cap=131072;fresh-replay=before+after;"
    "transport=no-tools-v1"
)
MAX_TYPED_JUDGER_PACKET_BYTES = 128 * 1024

# These are the built-in, bounded, deterministic adapters whose full
# observations are retained in their content-addressed evidence envelopes.
# Browser output, observed EOF, and extension adapters stay on the existing
# inspection path until separately justified.
ELIGIBLE_ADAPTERS = frozenset({
    ("filesystem.exists", 1),
    ("filesystem.sha256", 1),
    ("document.json-value", 1),
    ("document.text", 1),
    ("ledger.event", 1),
    ("ledger.tool-call-string", 1),
    ("ledger.tool-call-string", 2),
})


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json(raw: str) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )


def _result_document(result: DischargeResult) -> dict:
    """The same result facts the ordinary Judger receives."""
    return {
        "outcome": result.outcome,
        "witness": result.witness,
        "artifact_digest": result.artifact,
        "assumed_premises": list(result.assumed_premises),
        "witness_kind": result.witness_kind,
        "declared_inputs": list(result.inputs),
    }


def _result_identity(result: DischargeResult) -> dict:
    """Every executor-return field, used only for exact mechanical replay."""
    return {
        **_result_document(result),
        "instrument": result.instrument,
        "basis": result.basis,
        "strength": result.strength,
        "bridges": list(result.bridges),
    }


def _obligation_document(obligation: Obligation) -> dict:
    return {
        "id": obligation.id,
        "claim_id": obligation.claim_id,
        "text": obligation.text,
        "required_basis": obligation.required_basis,
        "severity": obligation.severity,
        "clause": obligation.clause,
        "refutable_by_absence": obligation.refutable_by_absence,
        "premise_scope": list(obligation.premise_scope),
    }


def _obligation_from_document(value: Mapping[str, Any]) -> Obligation:
    return Obligation(
        claim_id=value["claim_id"],
        text=value["text"],
        required_basis=value["required_basis"],
        severity=value["severity"],
        clause=value["clause"],
        refutable_by_absence=value["refutable_by_absence"],
        premise_scope=tuple(value["premise_scope"]),
    )


def _one_catalog_entry(catalog: list, name: str, version: int) -> dict:
    matches = [
        entry for entry in catalog
        if isinstance(entry, dict)
        and entry.get("name") == name
        and entry.get("version") == version
    ]
    if len(matches) != 1:
        raise ValueError("adapter catalog lacks one exact registered entry")
    return matches[0]


def _same_result(left: DischargeResult, right_document: Mapping[str, Any]) -> bool:
    return _result_identity(left) == right_document


def _source_slots_complete(invocation: AdapterInvocation, evidence: dict) -> bool:
    """Every declared source is read exactly once by this adapter receipt.

    Registry validation already rejects consumed-but-undeclared references.
    This stricter packet gate also rejects the opposite direction: unused or
    duplicate declared capabilities cannot enter a future tool-free Judger
    transport as though they had been mechanically replayed.
    """
    execution = evidence.get("execution")
    if not isinstance(execution, dict):
        return False
    declared = list(invocation.inputs)
    consumed = execution.get("consumed")
    if not isinstance(consumed, list):
        return False
    declared_refs = [item.get("ref") for item in declared]
    consumed_refs = [
        item.get("ref") if isinstance(item, dict) else None
        for item in consumed
    ]
    return (
        bool(declared_refs)
        and None not in declared_refs
        and None not in consumed_refs
        and len(declared_refs) == len(set(declared_refs))
        and len(consumed_refs) == len(set(consumed_refs))
        and consumed_refs == declared_refs
    )


def _fresh_replay(
    discharger: AdapterRoutingDischarger,
    artifacts,
    claim_id: str,
    obligation_document: Mapping[str, Any],
    method_document: Mapping[str, Any],
    expected_result: Mapping[str, Any],
) -> Optional[DischargeResult]:
    """Re-execute one exact registered adapter without writing a log event."""
    try:
        obligation = _obligation_from_document(obligation_document)
        result = discharger.registry.execute(
            method_document,
            {"id": claim_id},
            obligation,
            discharger.context,
            artifacts,
        )
        if result is None or not _same_result(result, expected_result):
            return None
        valid, _reason = discharger.registry.validate_evidence(
            method_document,
            {"id": claim_id},
            obligation,
            result,
            artifacts,
        )
        return result if valid else None
    except Exception:
        return None


@dataclass(frozen=True)
class TypedAdapterJudgerPacket:
    """Immutable text plus trusted handles needed for post-call replay."""

    adapter_identity: str
    artifact_digest: str
    replay_digest: str
    _discharger: AdapterRoutingDischarger
    _artifacts: Any
    _state_dir: str
    _claim_id: str
    _obligation_json: str
    _method_json: str
    _result_json: str
    _catalog_json: str
    _evidence_source: JudgerEvidencePacket
    _rendered: str

    def __post_init__(self) -> None:
        if (type(self._discharger) is not AdapterRoutingDischarger
                or type(self._discharger.registry) is not AdapterRegistry
                or type(self._evidence_source) is not JudgerEvidencePacket
                or not isinstance(self._rendered, str)
                or len(self._rendered.encode("utf-8"))
                > MAX_TYPED_JUDGER_PACKET_BYTES):
            raise ValueError("invalid typed-adapter Judger packet")

    def render(self) -> str:
        """Return one complete, bounded, inert JSON document."""
        return self._rendered

    def revalidate(self) -> bool:
        """Rebind authority/catalog/evidence and replay after the Judger."""
        try:
            if authoritative_task_context(self._state_dir) != _strict_json(
                    self._rendered)["task_context"]:
                return False
            if self._discharger.adapter_catalog() != self._catalog_json:
                return False
            if not self._evidence_source.revalidate():
                return False
            obligation = _strict_json(self._obligation_json)
            method = _strict_json(self._method_json)
            expected_result = _strict_json(self._result_json)
            replay = _fresh_replay(
                self._discharger,
                self._artifacts,
                self._claim_id,
                obligation,
                method,
                expected_result,
            )
            return (
                replay is not None
                and content_hash(_result_identity(replay)) == self.replay_digest
                and replay.artifact == self.artifact_digest
            )
        except (OSError, TypeError, ValueError, KeyError, UnicodeError,
                json.JSONDecodeError, RecursionError, OverflowError):
            return False


def build_typed_adapter_judger_packet(
    discharger,
    artifacts,
    obligation: Obligation,
    method: Mapping[str, Any],
    result: DischargeResult,
    degradation_record: str,
    *,
    captured_images_present: bool = False,
) -> Optional[TypedAdapterJudgerPacket]:
    """Build a complete packet or return ``None`` for the original path."""
    if captured_images_present or not isinstance(degradation_record, str):
        return None
    try:
        # Exact shipped routing/registry types are the trust boundary.  An
        # extension adapter can still use the ordinary independent Judger,
        # but cannot inherit this transport merely by matching an interface.
        if (type(discharger) is not AdapterRoutingDischarger
                or type(discharger.registry) is not AdapterRegistry):
            return None
        invocation = AdapterInvocation.from_method(method)
        if invocation is None \
                or (invocation.name, invocation.version) not in ELIGIBLE_ADAPTERS:
            return None
        expected_instrument = f"adapter:{invocation.name}@{invocation.version}"
        if result.instrument != expected_instrument or result.assumed_premises:
            return None

        state_dir = os.path.abspath(discharger.context.state_dir)
        claim_id = obligation.claim_id
        row = {"id": claim_id}
        valid, _reason = discharger.registry.validate_evidence(
            method, row, obligation, result, artifacts)
        if not valid:
            return None

        # Reuse the hardened content-addressed file reader, but never attach
        # a ledger here.  Ledger adapters retain their complete bounded source
        # receipt in the evidence envelope and are replayed against the exact
        # declared prefix at both endpoints.
        evidence_source = build_judger_evidence_packet(
            artifacts, discharger.log, "artifact-only", result.artifact)
        if type(evidence_source) is not JudgerEvidencePacket:
            return None
        evidence_source_document = _strict_json(evidence_source.render())
        sources = evidence_source_document.get("sources")
        if not isinstance(sources, dict) or set(sources) != {"artifact"}:
            return None
        raw_evidence = sources["artifact"].get("raw_json_text")
        if not isinstance(raw_evidence, str):
            return None
        evidence = _strict_json(raw_evidence)
        if not isinstance(evidence, dict) \
                or not _source_slots_complete(invocation, evidence):
            return None

        catalog_json = discharger.adapter_catalog()
        catalog = _strict_json(catalog_json)
        if not isinstance(catalog, list):
            return None
        catalog_entry = _one_catalog_entry(
            catalog, invocation.name, invocation.version)

        obligation_document = _obligation_document(obligation)
        method_document = _strict_json(canonical(dict(method)))
        result_document = _result_document(result)
        result_identity = _result_identity(result)
        replay = _fresh_replay(
            discharger,
            artifacts,
            claim_id,
            obligation_document,
            method_document,
            result_identity,
        )
        if replay is None or not evidence_source.revalidate():
            return None
        replay_digest = content_hash(_result_identity(replay))
        task_context = authoritative_task_context(state_dir)

        bindings = {
            "task_context": content_hash(task_context),
            "obligation": content_hash(obligation_document),
            "method_ir": content_hash(method_document),
            "degradation_record": content_hash(degradation_record),
            "result": content_hash(result_identity),
            "adapter_catalog": hashlib.sha256(
                catalog_json.encode("utf-8")).hexdigest(),
            "evidence": result.artifact,
            "fresh_replay": replay_digest,
        }
        document = {
            "schema": TYPED_JUDGER_PACKET_PROTOCOL,
            "semantics": (
                "Complete bounded discharge evidence envelope for a new "
                "independent Judger. Original world-file and ledger-prefix "
                "bytes are not attached. The packet authenticates transport "
                "and deterministic source re-reading, not method sufficiency, "
                "execution fidelity, or a verdict."
            ),
            "task_context": task_context,
            "obligation": obligation_document,
            "method_ir": method_document,
            "degradation_record": degradation_record,
            "adapter_contract": {
                "identity": f"{invocation.name}@{invocation.version}",
                "invocation": invocation.public(),
                "catalog_entry": catalog_entry,
                "catalog_sha256": bindings["adapter_catalog"],
            },
            "result": result_document,
            "source_slot_coverage": {
                "all_declared_inputs_freshly_read": True,
                "original_source_bytes_attached": False,
                "declared_inputs": list(invocation.inputs),
                "evidence_declared_inputs": evidence["execution"][
                    "declared_inputs"],
                "consumed_identities": evidence["execution"]["consumed"],
            },
            "content_addressed_evidence": {
                "digest": result.artifact,
                "raw_sha256": sources["artifact"]["raw_sha256"],
                "byte_count": sources["artifact"]["byte_count"],
                "raw_json_text": raw_evidence,
            },
            "verifier_execution_receipt": {
                "instrument": result.instrument,
                "observation": evidence["execution"]["observation"],
                "consumed_identities": evidence["execution"]["consumed"],
                "fresh_exact_replay_digest": replay_digest,
            },
            "bindings": bindings,
        }
        rendered = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(rendered.encode("utf-8")) > MAX_TYPED_JUDGER_PACKET_BYTES:
            return None
        return TypedAdapterJudgerPacket(
            adapter_identity=f"{invocation.name}@{invocation.version}",
            artifact_digest=result.artifact,
            replay_digest=replay_digest,
            _discharger=discharger,
            _artifacts=artifacts,
            _state_dir=state_dir,
            _claim_id=claim_id,
            _obligation_json=canonical(obligation_document),
            _method_json=canonical(method_document),
            _result_json=canonical(result_identity),
            _catalog_json=catalog_json,
            _evidence_source=evidence_source,
            _rendered=rendered,
        )
    except (OSError, TypeError, ValueError, KeyError, UnicodeError,
            json.JSONDecodeError, RecursionError, OverflowError):
        return None
