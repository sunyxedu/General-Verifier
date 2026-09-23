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
    FileExistsAdapter,
    FileSha256Adapter,
    JsonPredicateAdapter,
    JsonPredicateAdapterV2,
    JsonValueAdapter,
    LedgerEventAdapter,
    LedgerToolCallStringAdapter,
    LedgerToolCallStringAdapterV2,
    TextValueAdapter,
)
from ..execution.discharge import DischargeResult
from ..execution.evidence_program import (
    EvidenceProgramAdapter,
    LANGUAGE as EVIDENCE_PROGRAM_LANGUAGE,
)
from ..execution.evidence_program import (
    MAX_CUMULATIVE_ALLOCATION_BYTES as EVIDENCE_PROGRAM_MAX_ALLOCATION,
)
from ..execution.evidence_program import MAX_FUEL as EVIDENCE_PROGRAM_MAX_FUEL
from ..execution.instruments import authoritative_task_context
from .judger_prefetch import (
    JudgerEvidencePacket,
    build_judger_evidence_packet,
)


TYPED_JUDGER_PACKET_PROTOCOL = (
    "gv-typed-adapter-judger-packet-v6;"
    "evidence-cap=16384;render-cap=131072;fresh-replay=before+after;"
    "transport=no-tools-v1"
)
MAX_TYPED_JUDGER_PACKET_BYTES = 128 * 1024

# These are the built-in, bounded, deterministic adapters whose full
# observations are retained in their content-addressed evidence envelopes.
# Browser output and extension adapters stay on the existing inspection path.
# An observed-EOF receipt is eligible only where an adapter-specific gate can
# replay the exact stable source projection before and after the Judger.
ELIGIBLE_ADAPTERS = frozenset({
    ("filesystem.exists", 1),
    ("filesystem.sha256", 1),
    ("document.json-predicate", 1),
    ("document.json-predicate", 2),
    ("document.json-value", 1),
    ("document.text", 1),
    ("evidence.program", 1),
    ("ledger.event", 1),
    ("ledger.sequence", 1),
    ("ledger.sequence", 2),
    ("record.query", 1),
    ("ledger.tool-call-string", 1),
    ("ledger.tool-call-string", 2),
})


def _shipped_adapter_type(identity: tuple[str, int]) -> type | None:
    """Return the verifier-owned class for one restricted transport identity."""
    if identity == ("record.query", 1):
        # Keep the record normalizer and query algebra lazy at this boundary.
        from ..execution.record_query import RecordQueryAdapter
        return RecordQueryAdapter
    if identity in {("ledger.sequence", 1), ("ledger.sequence", 2)}:
        # Keep the larger ledger module lazy at this import boundary.
        from ..execution.ledger_sequence import (
            LedgerSequenceAdapter,
            LedgerSequenceCompleteAuditAdapter,
        )
        return {
            ("ledger.sequence", 1): LedgerSequenceAdapter,
            ("ledger.sequence", 2): LedgerSequenceCompleteAuditAdapter,
        }[identity]
    return {
        ("filesystem.exists", 1): FileExistsAdapter,
        ("filesystem.sha256", 1): FileSha256Adapter,
        ("document.json-predicate", 1): JsonPredicateAdapter,
        ("document.json-predicate", 2): JsonPredicateAdapterV2,
        ("document.json-value", 1): JsonValueAdapter,
        ("document.text", 1): TextValueAdapter,
        ("evidence.program", 1): EvidenceProgramAdapter,
        ("ledger.event", 1): LedgerEventAdapter,
        ("ledger.tool-call-string", 1): LedgerToolCallStringAdapter,
        ("ledger.tool-call-string", 2): LedgerToolCallStringAdapterV2,
    }.get(identity)


def _uses_exact_shipped_adapter(
    registry: AdapterRegistry, invocation: AdapterInvocation,
) -> bool:
    expected = _shipped_adapter_type((invocation.name, invocation.version))
    return bool(expected and registry.has_exact_adapter_type(
        invocation.name, invocation.version, expected))


def _contains_truncation(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("truncated") is True or any(
            _contains_truncation(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_truncation(item) for item in value)
    return False


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _is_slot_list(value: Any, count: int) -> bool:
    return (isinstance(value, list)
            and all(type(slot) is int and 0 <= slot < count for slot in value)
            and value == sorted(value)
            and len(value) == len(set(value)))


def _adapter_specific_gate(invocation: AdapterInvocation, evidence: dict,
                           result_identity: Mapping[str, Any]) -> bool:
    """Extra packet invariants for general instruments with a larger TCB."""
    identity = (invocation.name, invocation.version)
    if identity in {("ledger.sequence", 1), ("ledger.sequence", 2)}:
        # Lazy import avoids a module cycle: the adapter itself reuses the
        # registry protocol defined in adapters.py.
        from ..execution.ledger_sequence import (
            typed_sequence_observation_is_complete,
        )
        return typed_sequence_observation_is_complete(
            invocation, evidence, result_identity)
    if identity == ("record.query", 1):
        from ..execution.record_query import (
            typed_record_query_observation_is_complete,
        )
        return typed_record_query_observation_is_complete(
            invocation, evidence, result_identity)
    if identity != ("evidence.program", 1):
        return True
    if (result_identity.get("outcome") != "holds"
            or result_identity.get("witness") is not None
            or result_identity.get("witness_kind") != "none"):
        return False
    execution = evidence.get("execution")
    if not isinstance(execution, dict):
        return False
    observation = execution.get("observation")
    if not isinstance(observation, dict) or set(observation) != {
            "language", "source_sha256", "source_chars", "input_slots",
            "assertions", "observations", "fuel_limit", "fuel_used",
            "allocation_limit_bytes", "allocation_reserved_bytes",
            "used_input_slots",
    } or _contains_truncation(observation):
        return False
    source = invocation.arguments.get("source")
    if (not isinstance(source, str)
            or observation.get("language") != EVIDENCE_PROGRAM_LANGUAGE
            or observation.get("source_chars") != len(source)
            or observation.get("source_sha256") != hashlib.sha256(
                source.encode("utf-8")).hexdigest()
            or observation.get("fuel_limit") != EVIDENCE_PROGRAM_MAX_FUEL
            or type(observation.get("fuel_used")) is not int
            or not 0 < observation["fuel_used"] <= EVIDENCE_PROGRAM_MAX_FUEL
            or observation.get("allocation_limit_bytes")
            != EVIDENCE_PROGRAM_MAX_ALLOCATION
            or type(observation.get("allocation_reserved_bytes")) is not int
            or not 0 <= observation["allocation_reserved_bytes"]
            <= EVIDENCE_PROGRAM_MAX_ALLOCATION):
        return False

    execution_inputs = [
        item for item in invocation.inputs if item.get("role") != "authority"]
    subject_slots = {
        index for index, item in enumerate(execution_inputs)
        if item.get("role") == "subject"
    }
    input_slots = observation.get("input_slots")
    if (not isinstance(input_slots, list)
            or len(input_slots) != len(execution_inputs)
            or observation.get("used_input_slots") != list(range(len(execution_inputs)))):
        return False
    for index, (slot, declared) in enumerate(zip(input_slots, execution_inputs)):
        if (not isinstance(slot, dict) or set(slot) != {
                "slot", "ref", "provenance", "role", "bytes", "sha256"}
                or slot.get("slot") != index
                or {key: slot.get(key) for key in ("ref", "provenance", "role")}
                != declared
                or type(slot.get("bytes")) is not int or slot["bytes"] < 0
                or not _is_sha256(slot.get("sha256"))):
            return False
    assertions = observation.get("assertions")
    observations = observation.get("observations")
    if (not isinstance(assertions, list) or not assertions
            or not isinstance(observations, list)):
        return False
    for assertion in assertions:
        if (not isinstance(assertion, dict)
                or set(assertion) != {
                    "label", "expression_sha256", "holds", "input_slots",
                    "subject_slots", "observed", "observed_input_slots",
                    "observed_subject_slots"}
                or assertion.get("holds") is not True
                or not isinstance(assertion.get("label"), str)
                or not assertion["label"]
                or not _is_sha256(assertion.get("expression_sha256"))
                or not _is_slot_list(
                    assertion.get("input_slots"), len(execution_inputs))
                or not _is_slot_list(
                    assertion.get("subject_slots"), len(execution_inputs))
                or not _is_slot_list(
                    assertion.get("observed_input_slots"), len(execution_inputs))
                or not _is_slot_list(
                    assertion.get("observed_subject_slots"), len(execution_inputs))
                or not assertion["subject_slots"]
                or any(slot not in subject_slots for slot in assertion["subject_slots"])
                or any(slot not in subject_slots
                       for slot in assertion["observed_subject_slots"])
                or set(assertion["subject_slots"])
                != set(assertion["input_slots"]) & subject_slots
                or set(assertion["observed_subject_slots"])
                != set(assertion["observed_input_slots"]) & subject_slots):
            return False
        if (not set(assertion["subject_slots"])
                <= set(assertion["observed_subject_slots"])
                or not set(assertion["observed_subject_slots"])
                <= set(assertion["observed_input_slots"])):
            return False
    for observation_item in observations:
        if (not isinstance(observation_item, dict)
                or set(observation_item) != {"label", "value"}
                or not isinstance(observation_item.get("label"), str)
                or not observation_item["label"]):
            return False
    return True


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


def _task_context_document(raw: str, state_dir: str) -> dict:
    value = _strict_json(raw)
    expected_path = os.path.join(os.path.abspath(state_dir), "task.md")
    # the authority is the task statement; the harness may add facts about
    # the run beside it (the candidate response, the recorded environment),
    # which are context for the seats and never part of the statement
    harness_facts = {"candidate_response", "statement_note",
                     "environment_invocation",
                     "environment_changes_since_task_start",
                     "environment_note"}
    if (not isinstance(value, dict)
            or not {"path", "text"} <= set(value)
            or not set(value) <= {"path", "text"} | harness_facts
            or value.get("path") != expected_path
            or not isinstance(value.get("text"), str)):
        raise ValueError("typed-adapter task authority is invalid")
    return value


def _source_slot_coverage(invocation: AdapterInvocation, evidence: dict,
                          task_context: Mapping[str, Any]) -> Optional[dict]:
    """Execution sources are read; task authority is bound separately.

    Registry validation already rejects consumed-but-undeclared references.
    This packet gate rejects unused or duplicate execution capabilities.  The
    sole unused declaration it permits is the authoritative task context that
    the packet carries verbatim and binds independently; arbitrary external
    authorities still require the ordinary toolful Judger.
    """
    execution = evidence.get("execution")
    if not isinstance(execution, dict):
        return None
    declared = list(invocation.inputs)
    evidence_declared = execution.get("declared_inputs")
    consumed = execution.get("consumed")
    if evidence_declared != declared or not isinstance(consumed, list):
        return None
    task_path = task_context.get("path") if isinstance(task_context, dict) else None
    if not isinstance(task_path, str) or not task_path:
        return None
    execution_inputs = [item for item in declared if item.get("role") != "authority"]
    authorities = [item for item in declared if item.get("role") == "authority"]
    declared_refs = [item.get("ref") for item in declared]
    execution_refs = [item.get("ref") for item in execution_inputs]
    consumed_refs = [
        item.get("ref") if isinstance(item, dict) else None
        for item in consumed
    ]
    if (not execution_refs or None in declared_refs or None in consumed_refs
            or len(declared_refs) != len(set(declared_refs))
            or len(consumed_refs) != len(set(consumed_refs))
            or consumed_refs != execution_refs or len(authorities) > 1
            or any(item.get("provenance") != "task_given"
                   or item.get("ref") not in {"task.md", task_path}
                   for item in authorities)):
        return None
    return {
        "all_execution_inputs_freshly_read": True,
        "semantic_authorities_bound": True,
        "original_source_bytes_attached": False,
        "execution_inputs": execution_inputs,
        "semantic_authorities": authorities,
        "declared_inputs": declared,
        "evidence_declared_inputs": evidence_declared,
        "consumed_identities": consumed,
    }


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
    _task_context_json: str
    _obligation_json: str
    _method_json: str
    _result_json: str
    _catalog_json: str
    _degradation_record: str
    _evidence_source: JudgerEvidencePacket
    _rendered: str

    def __post_init__(self) -> None:
        if (type(self._discharger) is not AdapterRoutingDischarger
                or type(self._discharger.registry) is not AdapterRegistry
                or type(self._evidence_source) is not JudgerEvidencePacket
                or not all(isinstance(value, str) for value in (
                    self.adapter_identity,
                    self.artifact_digest,
                    self.replay_digest,
                    self._state_dir,
                    self._claim_id,
                    self._task_context_json,
                    self._obligation_json,
                    self._method_json,
                    self._result_json,
                    self._catalog_json,
                    self._degradation_record,
                ))
                or not isinstance(self._rendered, str)
                or len(self._rendered.encode("utf-8"))
                > MAX_TYPED_JUDGER_PACKET_BYTES):
            raise ValueError("invalid typed-adapter Judger packet")
        try:
            if self._canonical_rendered() != self._rendered:
                raise ValueError("typed-adapter Judger packet is not canonical")
        except (OSError, TypeError, ValueError, KeyError, UnicodeError,
                json.JSONDecodeError, RecursionError, OverflowError) as error:
            raise ValueError("invalid typed-adapter Judger packet") from error

    def _canonical_rendered(self) -> str:
        """Rebuild every model-visible byte from replay-owned packet state."""
        task_context = _task_context_document(
            self._task_context_json, self._state_dir)
        obligation = _strict_json(self._obligation_json)
        method = _strict_json(self._method_json)
        result_identity = _strict_json(self._result_json)
        catalog = _strict_json(self._catalog_json)
        evidence_source = _strict_json(self._evidence_source.render())
        if (not isinstance(obligation, dict)
                or not isinstance(method, dict)
                or not isinstance(result_identity, dict)
                or not isinstance(catalog, list)
                or not isinstance(evidence_source, dict)
                or canonical(obligation) != self._obligation_json
                or canonical(method) != self._method_json
                or canonical(result_identity) != self._result_json
                or canonical(catalog) != self._catalog_json):
            raise ValueError("non-canonical typed-adapter packet state")

        obligation_value = _obligation_from_document(obligation)
        if (set(obligation) != {
                "id", "claim_id", "text", "required_basis", "severity",
                "clause", "refutable_by_absence", "premise_scope",
            }
                or obligation_value.id != obligation["id"]
                or obligation_value.claim_id != self._claim_id):
            raise ValueError("typed-adapter obligation identity mismatch")
        if set(result_identity) != {
                "outcome", "witness", "artifact_digest", "assumed_premises",
                "witness_kind", "declared_inputs", "instrument", "basis",
                "strength", "bridges",
            }:
            raise ValueError("typed-adapter result identity mismatch")

        invocation = AdapterInvocation.from_method(method)
        if (invocation is None
                or (invocation.name, invocation.version) not in ELIGIBLE_ADAPTERS
                or not _uses_exact_shipped_adapter(
                    self._discharger.registry, invocation)):
            raise ValueError("typed-adapter invocation identity mismatch")
        adapter_identity = f"{invocation.name}@{invocation.version}"
        expected_instrument = f"adapter:{adapter_identity}"
        if (self.adapter_identity != adapter_identity
                or result_identity["instrument"] != expected_instrument
                or result_identity["artifact_digest"] != self.artifact_digest
                or self._evidence_source.artifact_digest != self.artifact_digest
                or result_identity["assumed_premises"]):
            raise ValueError("typed-adapter public identity mismatch")
        expected_replay_digest = content_hash(result_identity)
        if self.replay_digest != expected_replay_digest:
            raise ValueError("typed-adapter replay identity mismatch")

        catalog_entry = _one_catalog_entry(
            catalog, invocation.name, invocation.version)
        sources = evidence_source.get("sources")
        if not isinstance(sources, dict) or set(sources) != {"artifact"}:
            raise ValueError("typed-adapter evidence source mismatch")
        artifact_source = sources["artifact"]
        if not isinstance(artifact_source, dict):
            raise ValueError("typed-adapter artifact source mismatch")
        raw_evidence = artifact_source.get("raw_json_text")
        raw_sha256 = artifact_source.get("raw_sha256")
        byte_count = artifact_source.get("byte_count")
        if (not isinstance(raw_evidence, str)
                or not isinstance(raw_sha256, str)
                or not isinstance(byte_count, int)
                or byte_count != len(raw_evidence.encode("utf-8"))
                or raw_sha256 != hashlib.sha256(
                    raw_evidence.encode("utf-8")).hexdigest()):
            raise ValueError("typed-adapter artifact bytes mismatch")
        evidence = _strict_json(raw_evidence)
        coverage = (_source_slot_coverage(invocation, evidence, task_context)
                    if isinstance(evidence, dict) else None)
        if coverage is None or not _adapter_specific_gate(
                invocation, evidence, result_identity):
            raise ValueError("typed-adapter source coverage mismatch")

        catalog_sha256 = hashlib.sha256(
            self._catalog_json.encode("utf-8")).hexdigest()
        bindings = {
            "task_context": content_hash(self._task_context_json),
            "obligation": content_hash(obligation),
            "method_ir": content_hash(method),
            "degradation_record": content_hash(self._degradation_record),
            "result": expected_replay_digest,
            "adapter_catalog": catalog_sha256,
            "evidence": self.artifact_digest,
            "fresh_replay": self.replay_digest,
        }
        result_document = {
            key: result_identity[key]
            for key in (
                "outcome", "witness", "artifact_digest", "assumed_premises",
                "witness_kind", "declared_inputs",
            )
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
            "task_context": self._task_context_json,
            "obligation": obligation,
            "method_ir": method,
            "degradation_record": self._degradation_record,
            "adapter_contract": {
                "identity": adapter_identity,
                "invocation": invocation.public(),
                "catalog_entry": catalog_entry,
                "catalog_sha256": catalog_sha256,
            },
            "result": result_document,
            "source_slot_coverage": coverage,
            "content_addressed_evidence": {
                "digest": self.artifact_digest,
                "raw_sha256": raw_sha256,
                "byte_count": byte_count,
                "raw_json_text": raw_evidence,
            },
            "verifier_execution_receipt": {
                "instrument": expected_instrument,
                "observation": evidence["execution"]["observation"],
                "consumed_identities": evidence["execution"]["consumed"],
                "fresh_exact_replay_digest": self.replay_digest,
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
            raise ValueError("typed-adapter Judger packet exceeds its cap")
        return rendered

    def render(self) -> str:
        """Return one complete, bounded, inert JSON document."""
        try:
            rendered = self._canonical_rendered()
            if rendered != self._rendered:
                raise ValueError("typed-adapter Judger packet binding changed")
            return rendered
        except (OSError, TypeError, ValueError, KeyError, UnicodeError,
                json.JSONDecodeError, RecursionError, OverflowError) as error:
            raise ValueError("typed-adapter Judger packet binding changed") from error

    def revalidate(self) -> bool:
        """Rebind authority/catalog/evidence and replay after the Judger."""
        try:
            if self._canonical_rendered() != self._rendered:
                return False
            if authoritative_task_context(
                    self._state_dir) != self._task_context_json:
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
                or (invocation.name, invocation.version) not in ELIGIBLE_ADAPTERS \
                or not _uses_exact_shipped_adapter(
                    discharger.registry, invocation):
            return None
        expected_instrument = f"adapter:{invocation.name}@{invocation.version}"
        if result.instrument != expected_instrument or result.assumed_premises:
            return None
        result_identity = _result_identity(result)

        state_dir = os.path.abspath(discharger.context.state_dir)
        task_context = authoritative_task_context(state_dir)
        task_context_document = _task_context_document(task_context, state_dir)
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
        coverage = (_source_slot_coverage(
            invocation, evidence, task_context_document)
            if isinstance(evidence, dict) else None)
        if coverage is None or not _adapter_specific_gate(
                invocation, evidence, result_identity):
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
            "source_slot_coverage": coverage,
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
            _task_context_json=task_context,
            _obligation_json=canonical(obligation_document),
            _method_json=canonical(method_document),
            _result_json=canonical(result_identity),
            _catalog_json=catalog_json,
            _degradation_record=degradation_record,
            _evidence_source=evidence_source,
            _rendered=rendered,
        )
    except (OSError, TypeError, ValueError, KeyError, UnicodeError,
            json.JSONDecodeError, RecursionError, OverflowError):
        return None
