"""Read-only, independently pinned validation of one direct preparation.

This validates a candidate, not a replay bundle or a complete experiment.
All-claims/extraction lifecycle and fresh replay remain separate outer gates.
Payload provenance retains the producer's explicitly declared trust boundary:
the pinned Kernel's terminal-pass control flow, not a model-result signature.
"""
from __future__ import annotations

import copy
import hashlib
import os
import re
import sqlite3
import stat
from pathlib import Path

from experiments.terminal_bench import analyze_model_spans
from experiments.terminal_bench.prepare_contract_entry import RECEIPT_SCHEMA, _receipt_calls
from verifier import offline
from verifier.contract_replay import (
    MAX_SNAPSHOT_CLAIMS, MAX_SOURCE_EVENTS, MAX_SOURCE_LOG_BYTES, SNAPSHOT_POLICY, pipeline_sha256,
)
from verifier.core.model import CLAIM_EFFECT, CLAIM_FINAL, canonical, content_hash
from verifier.core.store import ClaimStore, TRIGGER_LAZY
from verifier.engine.contract.bundle import (
    ContractBundleIdentity, SealedContractBundle, _compile_input_mapping,
    _parse_claims, _parse_entries, _strict_json, build_contract_bundle_bytes,
    claim_graph_sha256,
)
from verifier.engine.contract.authority_view import build_authority_compilation_view
from verifier.stream.builder import Builder


MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PROVENANCE = frozenset({
    "schema", "setup_id", "claim_id", "claim_kind", "identity",
    "compile_input_sha256", "diagnostic_only", "mode", "binding_mode",
    "accounting_scope", "started_at_monotonic_ns",
})
_RECEIPT = _PROVENANCE | frozenset({
    "status", "finished_at_monotonic_ns", "findings", "error_class",
    "event_prefix_sha256", "event_ledger_sha256", "event_range", "calls",
    "terminal_critic", "ledger_sha256", "usage_complete", "attribution_complete",
    "payload_sha256", "obligations_sha256",
})


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _exact_equal(left, right):
    # Canonical JSON preserves the distinctions Python == erases (False/0).
    return content_hash(left) == content_hash(right)


def _private_read(path: Path, cap: int) -> bytes:
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("symlink in diagnostic path")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > cap:
            raise ValueError("nonprivate or oversized diagnostic file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(cap + 1)
        if len(raw) > cap:
            raise ValueError("diagnostic file grew beyond its cap")
        return raw
    finally:
        os.close(descriptor)


def _jsonl(raw, *, max_rows, max_line):
    lines = raw.splitlines(keepends=True)
    if len(lines) > max_rows:
        raise ValueError("too many diagnostic rows")
    rows = []
    for line in lines:
        if len(line) > max_line or not line.strip():
            raise ValueError("invalid diagnostic line")
        value = _strict_json(line)
        if not isinstance(value, dict):
            raise ValueError("nonobject diagnostic row")
        rows.append(value)
    return rows, lines


def _graph(prefix):
    """Reconstruct declarative graph in memory; never open a live Service."""
    store = ClaimStore(trigger=TRIGGER_LAZY)
    try:
        builder = Builder(None, store)  # feed-only: no log replay/writes.
        saw_commit = False
        saw_final_response = False
        for event in prefix:
            if event["type"] in ("commit", "extracted_commit"):
                saw_commit = True
            elif offline.is_offline_final_response(event) and not saw_commit \
                    and not saw_final_response:
                saw_final_response = True
            elif event["type"] not in offline.HOOK_EVENT_TYPES or saw_commit \
                    or saw_final_response:
                raise ValueError("prefix contains verifier-generated context")
            builder.feed(event)
        if len(builder.finals) != 1:
            raise ValueError("prefix must have one final")
        ids = [row[0] for row in store.db.execute(
            "SELECT id FROM claims ORDER BY rowid LIMIT ?", (MAX_SNAPSHOT_CLAIMS + 1,))]
        if len(ids) > MAX_SNAPSHOT_CLAIMS:
            raise ValueError("preparation graph exceeds its cap")
        claims = [{"id": cid, "kind": store.get(cid)["kind"],
                   "text": store.get(cid)["text"], "deps": sorted(store.parents(cid)),
                   "revises": store.get(cid).get("revises")} for cid in ids]
        db_rows = list(store.db.execute(
            "SELECT id,kind,text,revises,status,basis,strength FROM claims ORDER BY rowid"))
        deps = list(store.db.execute("SELECT child,parent FROM deps ORDER BY child,parent"))
        return claims, {cid for cid in store.backward_slice(builder.finals[0])}, db_rows, deps
    finally:
        store.db.close()


def _check_sealed_hooks(state, prefix):
    """Mirror ingest_feed without creating a log or invoking extraction."""
    path = state / "hook_feed.jsonl"
    expected = []
    if os.path.lexists(path):
        raw = _private_read(path, MAX_SOURCE_LOG_BYTES)
        lines = raw.splitlines()
        if len(lines) > MAX_SOURCE_EVENTS:
            raise ValueError("sealed hook feed exceeds its row cap")
        for line in lines:
            if not line.strip():
                continue
            event = _strict_json(line)
            if not isinstance(event, dict):
                raise ValueError("invalid sealed hook event")
            if event.get("type") in offline.HOOK_EVENT_TYPES:
                expected.append({"seq": len(expected) + 1, "type": event["type"],
                                 "payload": event.get("payload", {})})
    actual = [{key: event[key] for key in ("seq", "type", "payload")}
              for event in prefix if event["type"] not in (
                  "message", "commit", "extracted_commit")]
    if not _exact_equal(actual, expected):
        raise ValueError("prefix hook context differs from sealed solver input")
    responses = [event for event in prefix
                 if offline.is_offline_final_response(event)]
    if len(responses) != 1 or responses[0]["payload"]["text"] != \
            _private_read(state / "final.txt", MAX_SOURCE_LOG_BYTES).decode(
                "utf-8", errors="strict"):
        raise ValueError("prefix final response differs from sealed solver input")


def _check_store(path, expected_rows, expected_deps):
    # mode=ro/query_only prevents rebuilding or 'repairing' caller artifacts.
    if any(os.path.lexists(str(path) + suffix) for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("preparation store must be quiescent without journal sidecars")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA query_only=ON")
        for table in ("claims", "deps", "evidence", "obligation_edges"):
            kind = connection.execute(
                "SELECT type FROM sqlite_master WHERE name=?", (table,)).fetchone()
            if kind != ("table",):
                raise ValueError("unexpected preparation store schema")
        if any(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
               for table in ("evidence", "obligation_edges")):
            raise ValueError("preparation has evidence")
        actual_rows = connection.execute(
            "SELECT id,kind,text,revises,status,basis,strength FROM claims ORDER BY rowid LIMIT ?",
            (len(expected_rows) + 1,)).fetchall()
        actual_deps = connection.execute(
            "SELECT child,parent FROM deps ORDER BY child,parent LIMIT ?",
            (len(expected_deps) + 1,)).fetchall()
        if actual_rows != expected_rows or actual_deps != expected_deps:
            raise ValueError("stored graph differs from selected declarative graph")
    finally:
        connection.close()


def _direct_events(events, claim_id, kind, payload, state):
    allowed = {"decomposition_attacked", "contract_cache_miss", "contract_cache_published"}
    if not events or any(event["type"] not in allowed
                         or event["payload"].get("claim") != claim_id for event in events):
        raise ValueError("unexpected compilation delta")
    cache = [event for event in events if event["type"].startswith("contract_cache_")]
    if kind != CLAIM_FINAL:
        if cache or events[-1]["type"] != "decomposition_attacked" or events[-1]["payload"].get("verdict") != "pass":
            raise ValueError("support delta has no terminal review")
        return
    if ([event["type"] for event in cache] != ["contract_cache_miss", "contract_cache_published"]
            or events[0] != cache[0] or events[-1] != cache[-1]
            or events[-2]["type"] != "decomposition_attacked"
            or events[-2]["payload"].get("verdict") != "pass"):
        raise ValueError("authority delta is not a direct compilation")
    miss, published = [event["payload"] for event in cache]
    if (set(miss) != {"claim", "key", "authority", "leader", "waited"}
            or set(published) != {"claim", "key", "authority", "waited", "after_duplicate_fill", "source_digest"}
            or not isinstance(miss.get("key"), str) or _HEX64.fullmatch(miss["key"]) is None
            or miss.get("authority") != build_authority_compilation_view(str(state)).identity
            or miss.get("leader") is not True or miss.get("waited") is not False
            or published.get("waited") is not False or published.get("after_duplicate_fill") is not False
            or published.get("key") != miss.get("key")
            or not _exact_equal(published.get("authority"), miss.get("authority"))
            or published.get("source_digest") != content_hash(payload)):
        raise ValueError("authority publication does not match the direct result")


def validate_prepared_contract_entry(
    root: str | Path, *, expected_setup_id: str, expected_claim_id: str,
    expected_identity: ContractBundleIdentity, expected_compile_input_sha256: str,
    expected_receipt_sha256: str, candidate_entry: dict,
) -> dict:
    """Re-read artifacts and return only an independently checked candidate.

    The caller supplies all identity/hash pins; none are learned from the
    candidate or receipt. ``model_work`` is recomputed even for a rejected
    receipt, retaining failed attempts/known sums. Its observed cost gate is
    distinct from this preparation's complete lifecycle/lineage gate.

    This API does not validate all claims, extraction spend, warm reuse or
    fresh replay. Nor does it invent direct model-result signatures: the
    operator must trust the pinned Kernel control flow at preparation time.
    A final experiment must additionally pin the complete experiment source
    fingerprint, since verifier.pipeline_sha256 excludes experiment scripts.
    Independence here means re-reading artifacts and recomputing admission;
    the version-pinned call-lineage parser is shared with the producer. The
    compile-input semantic digest relies on that pinned producer's pre/post
    checks, while the source seal and complete graph are rechecked here.
    """
    raw_root = Path(root)
    root = raw_root.absolute()
    state = root / "state"
    findings, captured = [], {}
    if not raw_root.is_absolute() or ".." in raw_root.parts:
        findings.append("preparation_root_must_be_an_absolute_operator_pin")
    paths = {
        "start": root / "prepare-start.json", "receipt": root / "prepare-receipt.json",
        "usage": root / "prepare-usage.jsonl",
        "spans": root / "prepare-usage.jsonl.spans.jsonl",
        "events": state / "events.jsonl", "store": state / "claims.sqlite",
        "marker": state / offline.DIAGNOSTIC_ONLY_FILE,
    }
    for label, path in paths.items():
        cap = (analyze_model_spans.MAX_LEDGER_BYTES if label in {"usage", "spans", "store"}
               else MAX_SOURCE_LOG_BYTES if label == "events" else MAX_RECEIPT_BYTES)
        try:
            captured[label] = _private_read(path, cap)
        except (OSError, ValueError):
            findings.append(f"cannot_read_private_{label}")
    model_work = None
    if all(label in captured for label in ("usage", "spans")):
        # Read costs even when the receipt is missing or false. Missing rows
        # are never zero; the standard analyzer retains known/unknown bins.
        model_work = analyze_model_spans.analyze(paths["usage"], paths["spans"])
    validated_entry, identity = None, None
    try:
        identity = ContractBundleIdentity.from_value(expected_identity)
        if type(candidate_entry) is not dict:
            raise ValueError("candidate must be a plain JSON object")
        candidate_bytes = canonical(candidate_entry).encode("utf-8")
        if len(candidate_bytes) > MAX_RECEIPT_BYTES:
            raise ValueError("candidate exceeds its size cap")
        # Freeze caller-owned data ONCE. Returning a later read/deepcopy of
        # that object could return B after validating A under concurrency.
        candidate_snapshot = _strict_json(candidate_bytes)
        if (not isinstance(expected_setup_id, str) or not expected_setup_id
                or any(not isinstance(value, str) or _HEX64.fullmatch(value) is None for value in (
                    expected_claim_id, expected_compile_input_sha256, expected_receipt_sha256))):
            raise ValueError("invalid external pin")
        if len(captured) != len(paths):
            raise ValueError("incomplete preparation artifacts")
        start, receipt = _strict_json(captured["start"]), _strict_json(captured["receipt"])
        if (not isinstance(start, dict) or set(start) != _PROVENANCE
                or not isinstance(receipt, dict) or set(receipt) != _RECEIPT
                or content_hash(receipt) != expected_receipt_sha256
                or not _exact_equal(start, {key: receipt[key] for key in _PROVENANCE})):
            raise ValueError("receipt/start schema or external hash mismatch")
        pinned = {"schema": RECEIPT_SCHEMA, "setup_id": expected_setup_id,
                  "claim_id": expected_claim_id, "identity": identity.as_dict(),
                  "compile_input_sha256": expected_compile_input_sha256,
                  "mode": "direct_compile", "diagnostic_only": True,
                  "binding_mode": "trusted-kernel-terminal-pass-control-flow",
                  "accounting_scope": "one_claim_excludes_extraction_and_replay"}
        if any(not _exact_equal(receipt[key], value) for key, value in pinned.items()):
            raise ValueError("receipt provenance differs from external pins")
        begin, end = receipt["started_at_monotonic_ns"], receipt["finished_at_monotonic_ns"]
        if any(type(value) is not int or value < 0 or value.bit_length() > 128 for value in (begin, end)) or end < begin:
            raise ValueError("invalid preparation lifecycle")
        if (receipt["status"] != "passed" or receipt["findings"] != [] or receipt["error_class"] is not None
                or receipt["usage_complete"] is not True or receipt["attribution_complete"] is not True):
            raise ValueError("producer did not finish with a passed candidate")
        if _strict_json(captured["marker"]) != {"diagnostic_only": SNAPSHOT_POLICY}:
            raise ValueError("missing diagnostic isolation marker")
        if (pipeline_sha256() != identity.pipeline_sha256
                or offline.verify_immutable_input_anchor(str(state)) != identity.source_sha256):
            raise ValueError("current source/pipeline differs from external pin")
        events, lines = _jsonl(captured["events"], max_rows=MAX_SOURCE_EVENTS, max_line=MAX_SOURCE_LOG_BYTES)
        for index, event in enumerate(events, 1):
            if (set(event) != {"seq", "ts", "type", "payload"} or type(event["seq"]) is not int
                    or event["seq"] != index or not isinstance(event["payload"], dict)):
                raise ValueError("invalid event ledger")
        event_range = receipt["event_range"]
        if not isinstance(event_range, dict) or set(event_range) != {"before_seq", "after_seq", "events"}:
            raise ValueError("invalid event range")
        before = event_range["before_seq"]
        if (type(before) is not int or not 0 < before < len(events)
                or type(event_range["after_seq"]) is not int or event_range["after_seq"] != len(events)
                or _sha(b"".join(lines[:before])) != receipt["event_prefix_sha256"]
                or _sha(captured["events"]) != receipt["event_ledger_sha256"]):
            raise ValueError("event range/prefix digest mismatch")
        claims, slice_ids, db_rows, deps = _graph(events[:before])
        _check_sealed_hooks(state, events[:before])
        if claim_graph_sha256(claims) != identity.graph_sha256:
            raise ValueError("whole graph differs from external pin")
        parsed_claims = {claim.id: claim for claim in _parse_claims(claims)}
        if expected_claim_id not in slice_ids or parsed_claims[expected_claim_id].kind == CLAIM_EFFECT:
            raise ValueError("candidate claim is not a non-effect member of the final slice")
        kind = parsed_claims[expected_claim_id].kind
        if kind != receipt["claim_kind"]:
            raise ValueError("receipt claim kind mismatch")
        _check_store(paths["store"], db_rows, deps)
        # Reuse the strict entry/materialization parser with the ENTIRE
        # reconstructed graph. This validates one candidate only, creates no
        # partial SealedContractBundle, and does not weaken its coverage gate.
        parsed = _parse_entries([candidate_snapshot], parsed_claims,
                                {expected_claim_id: expected_compile_input_sha256})[expected_claim_id]
        if (not parsed.replayable or parsed.critic_receipt_sha256 != expected_receipt_sha256
                or parsed.payload_sha256 != receipt["payload_sha256"]
                or content_hash([item.as_dict() for item in parsed.obligations]) != receipt["obligations_sha256"]):
            raise ValueError("candidate payload/materialization is not bound to the pinned receipt")
        delta = [{key: event[key] for key in ("seq", "type", "payload")} for event in events[before:]]
        _direct_events(delta, expected_claim_id, kind, parsed.payload, state)
        expected_events = [{"seq": event["seq"], "type": event["type"], "sha256": content_hash(event)} for event in delta]
        if not _exact_equal(event_range["events"], expected_events):
            raise ValueError("event receipt disagrees with raw ledger")
        if not isinstance(receipt["ledger_sha256"], dict) or set(receipt["ledger_sha256"]) != {"usage", "spans"}:
            raise ValueError("invalid accounting hash map")
        for label in ("usage", "spans"):
            rows, _ = _jsonl(captured[label], max_rows=analyze_model_spans.MAX_LEDGER_ROWS,
                             max_line=analyze_model_spans.MAX_LINE_CHARS)
            if receipt["ledger_sha256"].get(label) != _sha(captured[label]):
                raise ValueError("accounting ledger digest mismatch")
            if label == "spans" and any(type(item.get("monotonic_ns")) is not int
                    or not begin <= item["monotonic_ns"] <= end for item in rows):
                raise ValueError("model work lies outside preparation lifecycle")
        calls, terminal, recomputed, audit_findings = _receipt_calls(
            paths["usage"], paths["spans"], expected_claim_id, kind, delta,
            expected_context={"setup_id": expected_setup_id, "state_dir": str(state.resolve()), "backend": "codex-cli"})
        model_work = recomputed
        findings.extend(audit_findings)
        if not _exact_equal(calls, receipt["calls"]) or not _exact_equal(terminal, receipt["terminal_critic"]):
            raise ValueError("call receipt disagrees with raw lifecycles")
        validated_entry = copy.deepcopy(candidate_snapshot)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, sqlite3.Error, RecursionError):
        findings.append("invalid_preparation_lineage")
    # Any later change invalidates admission, while observed costs remain
    # available for diagnosis. No artifact is deleted, fixed or rewritten.
    for label, raw in captured.items():
        try:
            if _private_read(paths[label], max(len(raw), 1)) != raw:
                findings.append(f"{label}_changed_during_validation")
        except (OSError, ValueError):
            findings.append(f"{label}_changed_during_validation")
    try:
        if (identity is None or offline.verify_immutable_input_anchor(str(state)) != identity.source_sha256
                or pipeline_sha256() != identity.pipeline_sha256):
            findings.append("source_changed_during_validation")
        if validated_entry is not None:
            _check_store(paths["store"], db_rows, deps)
    except (OSError, ValueError, AttributeError, sqlite3.Error):
        findings.append("source_cannot_revalidate")
    complete = (not findings and validated_entry is not None and model_work is not None
                and model_work["cost_complete"] is True and model_work["attribution_complete"] is True)
    return {"complete": complete, "findings": sorted(set(findings)),
            "entry": validated_entry if complete else None, "model_work": model_work}


def assemble_prepared_contracts(
    *, expected_setup_id: str, expected_identity: ContractBundleIdentity,
    expected_claims: list[dict], expected_compile_inputs: dict[str, str],
    preparations: list[dict],
) -> dict:
    """Read every predeclared direct preparation, then assemble the full graph.

    Each preparation has exactly root/claim_id/receipt_sha256/candidate_entry.
    These are operator declarations, not discovered successful directories.
    Missing, duplicated or failed claims block the whole bundle. No retries,
    fallback compilation, source selection or models are run here. Per-claim
    reports retain observed costs, but are NOT a full shared-setup bill: the
    parent must still account for extraction, all planned failures and replay.

    The returned bundle SHA must be written to the independent operator run
    manifest before a fresh arm loads it; returning bytes is not activation.
    """
    findings, reports, raw = [], [], None
    try:
        identity = ContractBundleIdentity.from_value(expected_identity)
        claims = _parse_claims(_strict_json(canonical(expected_claims).encode("utf-8")))
        if claim_graph_sha256(claims) != identity.graph_sha256:
            raise ValueError("external graph does not match its identity")
        inputs = _compile_input_mapping(dict(expected_compile_inputs), {claim.id: claim for claim in claims})
        if type(preparations) is not list or len(preparations) > MAX_SNAPSHOT_CLAIMS:
            raise ValueError("invalid or oversized preparation plan")
        if len(preparations) != len(inputs):
            findings.append("incomplete_preparation_coverage")
        declarations, seen_roots, seen_claims = [], set(), set()
        for row in preparations:
            if type(row) is not dict or set(row) != {"root", "claim_id", "receipt_sha256", "candidate_entry"}:
                raise ValueError("invalid preparation declaration")
            declared_root = Path(row["root"])
            if not declared_root.is_absolute() or ".." in declared_root.parts:
                raise ValueError("preparation root must be an absolute operator pin")
            root = str(declared_root)
            cid = row["claim_id"]
            if cid not in inputs or cid in seen_claims or root in seen_roots:
                raise ValueError("duplicate or unexpected preparation")
            # Candidate/receipt values remain untrusted and are frozen by the
            # per-claim consumer; pins and plan membership are immutable here.
            declarations.append((root, cid, row["receipt_sha256"], row["candidate_entry"]))
            seen_roots.add(root)
            seen_claims.add(cid)
        if seen_claims != set(inputs):
            findings.append("incomplete_preparation_coverage")
        entries = {}
        for root, cid, receipt_sha, candidate in declarations:
            report = validate_prepared_contract_entry(
                root, expected_setup_id=expected_setup_id, expected_claim_id=cid,
                expected_identity=identity, expected_compile_input_sha256=inputs[cid],
                expected_receipt_sha256=receipt_sha, candidate_entry=candidate)
            reports.append({"claim_id": cid, "root": root, **report})
            if report["complete"]:
                entries[cid] = report["entry"]
            else:
                findings.append("incomplete_preparation:" + cid)
        if not findings:
            raw = build_contract_bundle_bytes(
                identity=identity, claims=[claim.as_dict() for claim in claims],
                entries=[entries[claim.id] for claim in claims if claim.id in inputs])
            # Coverage is checked against the COMPLETE external input map,
            # not a smaller map inferred from the successful candidate list.
            bundle = SealedContractBundle.from_bytes(
                raw, expected_sha256=_sha(raw), expected_identity=identity,
                expected_compile_inputs=inputs)
            bundle.require_replayable()
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        findings.append("invalid_preparation_plan_or_bundle")
    complete = not findings and raw is not None
    return {"complete": complete, "findings": sorted(set(findings)),
            "preparations": reports, "bundle_bytes": raw if complete else None,
            "bundle_sha256": _sha(raw) if complete else None,
            "accounting_scope": "direct_claims_only_excludes_extraction_and_replay"}
