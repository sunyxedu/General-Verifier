"""Cold shared setup for the explicit offline adapter-isolation experiment.

This is not an alternate production scheduler. It extracts once, predeclares
the complete compilation plan, runs existing Compiler/Critic paths on fresh
private ledgers, consumes their receipts, and pins the full bundle. It never
selects a different candidate or resamples a failed setup until it passes.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from experiments.terminal_bench.prepare_contract_entry import prepare_contract_entry
from experiments.terminal_bench.run_typed_adapter_eval import (
    _source_fingerprint, _validate_execute_output_root,
)
from experiments.terminal_bench.validate_prepared_contract_entry import assemble_prepared_contracts
from verifier import offline
from verifier.contract_replay import (
    SNAPSHOT_POLICY, clone_sealed_inputs, compile_input_digests,
    fresh_graph_snapshot, replay_identity,
)
from verifier.core.store import TRIGGER_LAZY
from verifier.core.model import canonical, content_hash
from verifier.engine.method_cache import AcceptedMethodCache
from verifier.llm.codex_backend import CodexRuntime, codex_usage_ledger
from verifier.app.assembly import build_kernel
from verifier.mcp_server import Service


SETUP_SCHEMA = "gv-cold-shared-contract-preparation/v1"
SOURCE_ROOT = Path(__file__).resolve().parents[2]
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_RUNTIME = {"model": "gpt-5.6-luna", "seats": "codex", "effort": "medium", "service_tier": "standard"}


def codex_backend(state, config):
    backend = CodexRuntime(cwd=str(state), model=config["model"],
                           effort=config["effort"], service_tier=config["service_tier"])
    backend.validate_offline_tool_scope(str(state), None, None)
    return backend


def codex_service(state, config):
    """Same role construction and evidence engine as ordinary offline.score.

    Cold diagnostic state has a fresh method cache, never an ambient cache
    populated by the other arm. No shared toolchain or advisory memory.
    """
    def factory(sc):
        kernel = build_kernel(
            sc, config["model"], backend=codex_backend(state, config),
            critic_rounds=config["critic_rounds"], parallel_obligations=4, parallel_claims=4)
        if kernel.binder is not None and kernel.binder.method_cache is not None:
            kernel.binder.method_cache = AcceptedMethodCache()
        kernel.contract_cache = None  # Producer owns its direct cold cache.
        return kernel
    return Service(str(state), kernel_factory=factory, trigger=TRIGGER_LAZY,
                   live_world=False, critic_rounds=config["critic_rounds"],
                   parallel_obligations=4, parallel_claims=4)


def _new_bytes(path, raw):
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o400)


def prepare_shared_contracts(
    source: str | Path, destination: str | Path, *, setup_id: str, config: dict,
    expected_source_fingerprint: str, source_root: str | Path = SOURCE_ROOT,
    backend_factory=None, service_factory=None,
) -> dict:
    """Execute one cold setup, including all actual extraction/compile bills.

    Factory injection is an operator-trusted local testing seam, not a plugin
    or a paid-CLI option. Default construction uses the unchanged Codex role
    registry and fixed Luna medium/standard runtime. This API is cold only:
    no warm lineage, cross-candidate seed replacement or automatic resampling.
    The parent A/B runner must separately account for and validate both fresh
    replay arms; this function alone is not full-E2E acceptance evidence.
    """
    from experiments.terminal_bench.merge_setup_accounting import merge_setup_accounting

    if type(config) is not dict:
        raise ValueError("config must be a plain object")
    import json
    config = json.loads(json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False))
    if not isinstance(setup_id, str) or _IDENTIFIER.fullmatch(setup_id) is None:
        raise ValueError("invalid setup id")
    if any(config.get(key) != value for key, value in _RUNTIME.items()):
        raise ValueError("diagnostic requires GPT-5.6-Luna, Codex, medium, standard")
    if config.get("browser_observer") is not None:
        raise ValueError("this cold experiment has no browser observer")
    for field in ("max_rounds", "critic_rounds"):
        if type(config.get(field)) is not int or not 1 <= config[field] <= 100:
            raise ValueError("invalid pinned retry budget")
    source_root = Path(source_root).resolve(strict=True)
    if source_root != SOURCE_ROOT or _source_fingerprint(source_root) != expected_source_fingerprint:
        raise ValueError("running experiment source differs from operator pin")
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    if not destination.is_absolute() or ".." in destination.parts:
        raise ValueError("setup root must be an absolute operator pin")
    if os.path.commonpath((source, destination.resolve())) in (str(source), str(destination.resolve())):
        raise ValueError("source and setup must be disjoint")
    if os.path.lexists(destination):
        raise ValueError("setup root must be new")
    if backend_factory is None and service_factory is None:
        _validate_execute_output_root(destination)
    test_only = backend_factory is not None or service_factory is not None
    backend_factory = backend_factory or (lambda state: codex_backend(state, config))
    service_factory = service_factory or (lambda state: codex_service(state, config))
    started = time.monotonic_ns()
    destination.mkdir(parents=True, exist_ok=False)
    source_seal = offline.verify_immutable_input_anchor(str(source))
    extracted = destination / "extracted"
    extract_usage = destination / "extract-usage.jsonl"
    state_path = destination / "setup-state.json"
    state_record = {
        "schema": SETUP_SCHEMA, "setup_id": setup_id, "phase": "cold",
        "status": "running", "source": str(source), "source_seal": source_seal,
        "source_root": str(source_root), "source_fingerprint": expected_source_fingerprint,
        "config": dict(config), "started_at_monotonic_ns": started,
        "extraction": {"status": "not_started"}, "claims": [],
        "diagnostic_only": True,
        "test_only": test_only,
    }
    offline._write_json_atomic(str(state_path), state_record)
    components, declarations, outputs = [], [], []
    identity, inputs, claims, metadata, bundle = None, None, None, None, None
    plan_sha = None
    failure = None
    try:
        clone_sealed_inputs(source, extracted)
        offline._write_json_atomic(str(extracted / offline.DIAGNOSTIC_ONLY_FILE),
                                   {"diagnostic_only": SNAPSHOT_POLICY})
        state_record["extraction"]["status"] = "started"
        components.append({"component_id": "extract", "usage_path": str(extract_usage),
                           "spans_path": str(extract_usage) + ".spans.jsonl"})
        offline._write_json_atomic(str(state_path), state_record)
        with codex_usage_ledger(str(extract_usage), {str(extracted): {"setup_id": setup_id}}):
            metadata = offline.extract(
                str(extracted), backend=backend_factory(extracted), max_rounds=config["max_rounds"],
                stamp={"config": config.get("config"), "immutable_input_seal": source_seal})
        state_record["extraction"]["status"] = "finished"
        state_record["extraction"]["rounds"] = metadata["rounds"]
        prototype = service_factory(extracted)
        if prototype.store.trigger != TRIGGER_LAZY or prototype.kernel.live_world:
            raise ValueError("setup factory must be a lazy offline service")
        if len(prototype.builder.finals) != 1:
            raise ValueError("setup requires exactly one extracted final before any compilation")
        graph = fresh_graph_snapshot(prototype.kernel, prototype.builder.finals[-1])
        identity = replay_identity(prototype.kernel, graph, config)
        inputs = compile_input_digests(prototype.kernel, graph)
        claims = [{key: row[key] for key in ("id", "kind", "text", "deps", "revises")}
                  for row in graph["claims"]]
        # Declare EVERY claim before the first compilation. Fail-fast leaves
        # remaining claims explicitly not_started, never replaced by a seed.
        state_record["claims"] = [
            {"claim_id": row["id"], "root": str(destination / "claims" / row["id"]),
             "compile_input_sha256": inputs[row["id"]], "status": "not_started"}
            for row in claims if row["id"] in inputs]
        plan = {
            "setup_id": setup_id, "identity": identity.as_dict(), "claims": claims,
            "compile_inputs": inputs, "preparations": state_record["claims"],
            "source_fingerprint": expected_source_fingerprint,
        }
        plan_sha = content_hash(plan)
        state_record["contract_plan_sha256"] = plan_sha
        _new_bytes(destination / "contract-plan.json", canonical(plan).encode("utf-8") + b"\n")
        offline._write_json_atomic(str(state_path), state_record)
        for planned in state_record["claims"]:
            planned["status"] = "started"
            offline._write_json_atomic(str(state_path), state_record)
            root = Path(planned["root"])
            components.append({"component_id": planned["claim_id"],
                               "usage_path": str(root / "prepare-usage.jsonl"),
                               "spans_path": str(root / "prepare-usage.jsonl.spans.jsonl")})
            result = prepare_contract_entry(
                extracted, root, setup_id=setup_id, claim_id=planned["claim_id"],
                service_factory=service_factory, config=config, expected_identity=identity,
                expected_compile_input_sha256=planned["compile_input_sha256"])
            planned["status"] = "finished" if result["entry"] is not None else "failed"
            outputs.append({"root": str(root), "claim_id": planned["claim_id"],
                            "receipt_sha256": result["receipt_sha256"], "candidate_entry": result["entry"]})
            # Persist externally to each producer before independently consuming
            # its bytes. This also retains the candidate if the parent crashes.
            offline._write_json_atomic(str(destination / "prepared-candidates.json"), {"preparations": outputs})
            offline._write_json_atomic(str(state_path), state_record)
            if result["entry"] is None:
                raise ValueError("a planned compilation was inconclusive; setup not resampled")
            declarations.append(outputs[-1])
        bundle = assemble_prepared_contracts(
            expected_setup_id=setup_id, expected_identity=identity, expected_claims=claims,
            expected_compile_inputs=inputs, preparations=declarations)
        if not bundle["complete"]:
            raise ValueError("complete contract bundle failed independent admission")
    except BaseException as error:
        failure = error
        if state_record["extraction"]["status"] == "started":
            state_record["extraction"]["status"] = "failed"
        for planned in state_record["claims"]:
            if planned["status"] == "started":
                planned["status"] = "failed"
    # Always collect every started component, including a failed extraction
    # or compile. Explicit not_started items produced no model calls; they
    # still prevent successful setup admission and remain in the plan.
    wall_before_accounting = (time.monotonic_ns() - started) / 1e9
    try:
        accounting = merge_setup_accounting(
            destination / "accounting", setup_id=setup_id, components=components,
            wall_seconds=wall_before_accounting)
    except BaseException as error:
        failure = failure or error
        # Source bills remain on disk. An aborted aggregator is not a zero
        # bill, even if a subset of its output was successfully written.
        accounting = {"complete": False, "findings": ["accounting_finalization_failed"],
                      "metrics": {"complete": False, "cost_complete": False, "wall_complete": False,
                                  "started_attempts": None, "uncached_input_tokens": None,
                                  "cached_input_tokens": None, "output_tokens": None, "wall_seconds": None},
                      "model_work": None, "source_records": [], "usage_path": None, "spans_path": None}
    findings = list(accounting["findings"])
    if failure is not None:
        findings.append("setup_failed:" + type(failure).__name__)
    # Re-consume all successful receipts after concatenating their ledgers.
    # A changed bill cannot be admitted under an earlier consumer decision.
    if bundle is not None and bundle["complete"]:
        try:
            again = assemble_prepared_contracts(
                expected_setup_id=setup_id, expected_identity=identity, expected_claims=claims,
                expected_compile_inputs=inputs, preparations=declarations)
            if not again["complete"] or again["bundle_bytes"] != bundle["bundle_bytes"]:
                findings.append("preparations_changed_during_accounting")
        except BaseException as error:
            failure = failure or error
            findings.append("preparations_changed_during_accounting")
    complete = not findings and accounting["complete"] and bundle is not None and bundle["complete"]
    bundle_sha = None
    if complete:
        try:
            bundle_sha = bundle["bundle_sha256"]
            _new_bytes(destination / "contracts.json", bundle["bundle_bytes"])
        except BaseException as error:
            failure = failure or error
            findings.append("bundle_persistence_failed")
    fingerprint_after = None
    try:
        fingerprint_after = _source_fingerprint(source_root)
        if (fingerprint_after != expected_source_fingerprint
                or offline.verify_immutable_input_anchor(str(source)) != source_seal
                or offline.verify_immutable_input_anchor(str(extracted)) != source_seal):
            findings.append("setup_source_changed")
        if plan_sha is not None and content_hash(offline._read_regular_json(
                str(destination / "contract-plan.json"))) != plan_sha:
            findings.append("setup_plan_changed")
    except BaseException as error:
        failure = failure or error
        findings.append("setup_source_cannot_revalidate")
    complete = complete and not findings
    if not complete:
        bundle_sha = None  # Any retained partial files are not admission pins.
    finished = time.monotonic_ns()
    # Parent makespan includes extraction, clones, repairs, admission and
    # accounting. Initial preflight and final terminal-record/pin writes are
    # excluded; an eventual CLI must also report its full process makespan.
    # Never sum overlapping model-call durations.
    metrics = dict(accounting["metrics"])
    metrics["wall_seconds"] = (finished - started) / 1e9
    metrics["wall_complete"] = True
    state_record.update({"status": "complete" if complete else "inconclusive",
                         "finished_at_monotonic_ns": finished, "wall_seconds": metrics["wall_seconds"],
                         "source_fingerprint_after": fingerprint_after,
                         "findings": sorted(set(findings)), "bundle_sha256": bundle_sha})
    offline._write_json_atomic(str(state_path), state_record)
    result = {"complete": complete, "findings": state_record["findings"],
              "setup_id": setup_id, "extracted_source": str(extracted),
              "extracted_rounds": metadata["rounds"] if metadata else None,
              "identity": identity.as_dict() if identity else None,
              "bundle_sha256": bundle_sha, "bundle_path": str(destination / "contracts.json") if complete else None,
              "accounting": accounting, "metrics": metrics,
              "source_fingerprint": expected_source_fingerprint,
              "source_fingerprint_after": fingerprint_after, "contract_plan_sha256": plan_sha,
              "test_only": test_only,
              "accounting_scope": "complete_cold_setup_excludes_replay"}
    offline._write_json_atomic(str(destination / "setup-result.json"), result)
    # Publish the admission marker LAST: a retained candidate or complete
    # result alone is insufficient. Final source/plan checks and both terminal
    # records must have succeeded. Consumers must require all three records.
    if complete:
        try:
            offline._write_json_atomic(str(destination / "bundle-pin.json"), {
                "setup_id": setup_id, "bundle_sha256": bundle_sha, "identity": identity.as_dict(),
                "source_fingerprint": expected_source_fingerprint, "test_only": test_only,
                "contract_plan_sha256": plan_sha,
                "accounting_usage_path": accounting["usage_path"],
                "accounting_spans_path": accounting["spans_path"],
            })
        except BaseException as error:
            failure = failure or error
            findings.append("bundle_pin_persistence_failed")
            state_record.update(status="inconclusive", findings=sorted(set(findings)), bundle_sha256=None)
            result.update(complete=False, findings=state_record["findings"], bundle_sha256=None, bundle_path=None)
            offline._write_json_atomic(str(state_path), state_record)
            offline._write_json_atomic(str(destination / "setup-result.json"), result)
    if isinstance(failure, (KeyboardInterrupt, SystemExit)):
        raise failure
    return result
