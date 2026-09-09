"""Paid three-nonce canary for the real Luna Compiler/Critic boundary.

This diagnostic creates a synthetic, task-neutral authority and runs the
existing direct contract-preparation path.  It does not extract with a model,
enumerate methods, execute evidence, judge a candidate, or score a task.  Its
only question is whether the independently criticised final contract:

* excludes a pure post-capture runner protocol;
* retains task-owned production provenance as MUST + record; and
* retains candidate content as MUST + world.

The random opaque nonces prevent a fixed task noun from carrying the answer.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Callable, Mapping


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(SOURCE_ROOT))

from experiments.terminal_bench.prepare_contract_entry import (  # noqa: E402
    prepare_contract_entry,
)
from experiments.terminal_bench.prepare_shared_contracts import (  # noqa: E402
    codex_service,
)
from experiments.terminal_bench.run_typed_adapter_eval import (  # noqa: E402
    _source_fingerprint,
    _validate_execute_output_root,
)
from verifier import offline  # noqa: E402
from verifier.contract_replay import (  # noqa: E402
    compile_input_digests,
    fresh_graph_snapshot,
    replay_identity,
)
from verifier.llm.schema import FixtureBackend  # noqa: E402


SCHEMA = "gv-contract-semantic-canary/v1"
RUNTIME = {
    "model": "gpt-5.6-luna",
    "effort": "medium",
    "service_tier": "standard",
}
CRITIC_ROUNDS = 4
_NONCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{15,127}\Z")
_NONCE_KEYS = ("candidate", "provenance", "runner")


def runtime_config() -> dict:
    """Return the fixed semantic identity used by the direct preparation."""
    return offline.config_fields(
        model=RUNTIME["model"],
        seats="codex",
        effort=RUNTIME["effort"],
        service_tier=RUNTIME["service_tier"],
        max_rounds=1,
        critic_rounds=CRITIC_ROUNDS,
        compact_inspection=False,
        judger_prefetch=False,
    )


def fresh_nonces() -> dict[str, str]:
    return {
        key: secrets.token_hex(24)
        for key in _NONCE_KEYS
    }


def _validated_nonces(value: Mapping[str, str]) -> dict[str, str]:
    if type(value) is not dict or set(value) != set(_NONCE_KEYS):
        raise ValueError("nonces must be a plain candidate/provenance/runner map")
    result = dict(value)
    if any(not isinstance(item, str) or _NONCE.fullmatch(item) is None
           for item in result.values()):
        raise ValueError("each nonce must be a bounded opaque identifier")
    if len(set(result.values())) != len(result):
        raise ValueError("canary nonces must differ")
    if any(left in right for left in result.values()
           for right in result.values() if left != right):
        raise ValueError("one canary nonce must not contain another")
    return result


def _task_text(nonces: Mapping[str, str]) -> str:
    candidate = json.dumps(nonces["candidate"])
    provenance = json.dumps(nonces["provenance"])
    runner = json.dumps(nonces["runner"])
    origin = json.dumps("origin::" + nonces["provenance"])
    copied = json.dumps("copy::" + nonces["provenance"])
    clauses = [
        f"""The task-selected current candidate final response MUST contain the
literal {candidate}. A final response without that content is rejected
regardless of runner behavior.""",
        f"""During production of the current candidate, its producing process
MUST emit the exact byte sequence {provenance} by invoking the logical
operation {origin}. A history that yields a byte-for-byte identical candidate
and emitted bytes via {copied}, copying, injection, or any other process is
rejected. This production provenance is independently part of acceptance; it
is a past action, not an assertion that the emitted bytes are true.""",
        f"""Only after the task-selected candidate is fixed, the benchmark
runner MUST write {runner} to private runner telemetry before marking its own
job complete. If it omits this step but still captures and hands off the exact
same candidate, task acceptance is unchanged. That telemetry is neither a
delivered artifact nor an independently evaluated task object.""",
    ]
    # Vary source position too; the semantic audit retains the hidden mapping.
    for index in range(len(clauses) - 1, 0, -1):
        other = secrets.randbelow(index + 1)
        clauses[index], clauses[other] = clauses[other], clauses[index]
    return ("Opaque-literal contract\n\nEvery quoted string below is an "
            "arbitrary opaque literal.\n\n" + "\n\n".join(clauses) + "\n")


def _write_new(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _prepare_source(source: Path, nonces: Mapping[str, str]) -> str:
    """Create a sealed synthetic source; extraction itself is model-free."""
    source.mkdir(parents=True, exist_ok=False)
    _write_new(source / "task.md", _task_text(nonces))
    _write_new(
        source / "final.txt",
        "Synthetic candidate carrier: " + nonces["candidate"] + "\n",
    )
    _write_new(source / "hook_feed.jsonl", "")
    seal = offline.write_immutable_input_anchor(os.fspath(source))
    commits = [{
        "label": "final",
        "kind": "final",
        "text": "The fixed current candidate satisfies the complete task.",
        "deps": [],
        "revises": "",
    }]
    metadata = offline.extract(
        os.fspath(source),
        backend=FixtureBackend({"extract": {"commits": commits}}),
        max_rounds=1,
        stamp={"immutable_input_seal": seal, "semantic_canary": SCHEMA},
    )
    if metadata.get("rounds") != 1 or metadata.get("commits") != 1:
        raise ValueError("model-free canary extraction changed shape")
    offline._write_json_atomic(
        os.fspath(source / offline.DIAGNOSTIC_ONLY_FILE),
        {"diagnostic_only": SCHEMA},
    )
    if offline.verify_immutable_input_anchor(os.fspath(source)) != seal:
        raise ValueError("synthetic canary source changed after sealing")
    return seal


def audit_semantics(preparation: Mapping, nonces: Mapping[str, str]) \
        -> tuple[list[str], list[dict]]:
    """Mechanically admit only the intended two semantic classes."""
    nonces = _validated_nonces(nonces)
    findings: list[str] = []
    receipt = preparation.get("receipt")
    entry = preparation.get("entry")
    if not isinstance(receipt, dict) or receipt.get("status") != "passed" \
            or receipt.get("findings") != []:
        findings.append("direct_preparation_not_passed")
    terminal = receipt.get("terminal_critic") if isinstance(receipt, dict) \
        else None
    if not isinstance(terminal, dict) or terminal.get("verdict") != "pass":
        findings.append("terminal_critic_not_passed")
    if not isinstance(entry, dict) or entry.get("status") != "passed" \
            or not isinstance(entry.get("obligations"), list):
        findings.append("passed_contract_entry_missing")
        return sorted(set(findings)), []

    obligations = entry["obligations"]
    candidate_rows, provenance_rows = [], []
    origin = "origin::" + nonces["provenance"]
    for index, obligation in enumerate(obligations):
        if not isinstance(obligation, dict):
            findings.append("malformed_obligation")
            continue
        text = obligation.get("text")
        clause = obligation.get("clause")
        if not isinstance(text, str) or not isinstance(clause, str):
            findings.append("malformed_obligation")
            continue
        whole = text + "\n" + clause
        if nonces["runner"] in whole:
            findings.append("runner_protocol_compiled")
        has_candidate = nonces["candidate"] in text
        has_provenance = nonces["provenance"] in text
        if has_candidate and has_provenance:
            findings.append("independent_requirements_merged")
        elif has_candidate:
            candidate_rows.append(index)
            if (obligation.get("required_basis") != "world"
                    or obligation.get("severity") != "must"):
                findings.append("candidate_content_floor_or_modality_changed")
        elif has_provenance:
            provenance_rows.append(index)
            if (obligation.get("required_basis") != "record"
                    or obligation.get("severity") != "must"):
                findings.append("provenance_floor_or_modality_changed")
        else:
            findings.append("unclassified_extra_obligation")
    if not candidate_rows:
        findings.append("candidate_content_requirement_missing")
    if not provenance_rows:
        findings.append("provenance_requirement_missing")
    elif not any(origin in obligations[index]["text"]
                 for index in provenance_rows):
        findings.append("provenance_process_requirement_missing")
    return sorted(set(findings)), obligations


def run_canary(
    output: Path,
    *,
    expected_source_fingerprint: str,
    nonces: dict[str, str] | None = None,
    service_factory: Callable | None = None,
) -> dict:
    """Run one cold direct compile; factory injection is a local test seam."""
    output = output.absolute()
    test_only = service_factory is not None
    if not test_only:
        _validate_execute_output_root(output)
    if os.path.lexists(output):
        raise ValueError("output root must be new")
    if _source_fingerprint(SOURCE_ROOT) != expected_source_fingerprint:
        raise ValueError("running source differs from operator fingerprint")
    nonces = _validated_nonces(nonces or fresh_nonces())
    config = runtime_config()
    service_factory = service_factory or (
        lambda state: codex_service(state, config))
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    offline._write_json_atomic(os.fspath(output / "canary-start.json"), {
        "schema": SCHEMA,
        "status": "started",
        "diagnostic_only": True,
        "full_e2e_acceptance": False,
        "test_only": test_only,
        "runtime": RUNTIME,
        "config": config,
        "nonces": nonces,
        "source_root": os.fspath(SOURCE_ROOT),
        "source_fingerprint": expected_source_fingerprint,
        "excluded_stages": ["model_extract", "oracles", "discharge",
                            "judger", "certificate", "score"],
    })

    findings: list[str] = []
    preparation = None
    obligations: list[dict] = []
    error = None
    source_seal = None
    try:
        source = output / "source"
        source_seal = _prepare_source(source, nonces)
        prototype = service_factory(source)
        graph = fresh_graph_snapshot(
            prototype.kernel, prototype.builder.finals[-1])
        claim_id = graph["final_id"]
        identity = replay_identity(prototype.kernel, graph, config)
        inputs = compile_input_digests(prototype.kernel, graph)
        preparation = prepare_contract_entry(
            source,
            output / "prepare",
            setup_id="semantic-canary",
            claim_id=claim_id,
            service_factory=service_factory,
            config=config,
            expected_identity=identity,
            expected_compile_input_sha256=inputs[claim_id],
        )
        semantic_findings, obligations = audit_semantics(
            preparation, nonces)
        findings.extend(semantic_findings)
    except BaseException as caught:
        error = type(caught).__name__ + ": " + str(caught)
        findings.append("canary_raised:" + type(caught).__name__)

    try:
        if _source_fingerprint(SOURCE_ROOT) != expected_source_fingerprint:
            findings.append("source_fingerprint_changed")
        if source_seal is not None and offline.verify_immutable_input_anchor(
                os.fspath(output / "source")) != source_seal:
            findings.append("synthetic_source_changed")
    except BaseException as caught:
        error = error or (type(caught).__name__ + ": " + str(caught))
        findings.append("final_revalidation_failed")

    receipt = preparation.get("receipt") if isinstance(preparation, dict) \
        else None
    result = {
        "schema": SCHEMA,
        "status": "complete" if not findings else "inconclusive",
        "diagnostic_only": True,
        "full_e2e_acceptance": False,
        "test_only": test_only,
        "wall_seconds": time.monotonic() - started,
        "runtime": RUNTIME,
        "config": config,
        "nonces": nonces,
        "source_seal": source_seal,
        "source_fingerprint": expected_source_fingerprint,
        "findings": sorted(set(findings)),
        "error": error,
        "obligations": obligations,
        "preparation": {
            "status": receipt.get("status") if isinstance(receipt, dict)
                else None,
            "receipt_sha256": preparation.get("receipt_sha256")
                if isinstance(preparation, dict) else None,
            "terminal_critic": receipt.get("terminal_critic")
                if isinstance(receipt, dict) else None,
            "calls": receipt.get("calls") if isinstance(receipt, dict)
                else [],
            "usage_complete": receipt.get("usage_complete")
                if isinstance(receipt, dict) else False,
            "attribution_complete": receipt.get("attribution_complete")
                if isinstance(receipt, dict) else False,
        },
    }
    offline._write_json_atomic(
        os.fspath(output / "canary-result.json"), result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid")
    args = parser.parse_args(argv)
    if args.execute and args.confirm_paid != "GPT-5.6-Luna":
        parser.error("execution requires --confirm-paid GPT-5.6-Luna")
    if os.path.lexists(args.output_root):
        parser.error("output-root must be new")
    if not args.execute:
        print(json.dumps({
            "schema": SCHEMA,
            "diagnostic_only": True,
            "full_e2e_acceptance": False,
            "output_root": os.fspath(args.output_root.absolute()),
            "runtime": RUNTIME,
            "nonce_policy": "three fresh opaque 192-bit values at execution",
            "paid_scope": "Compiler/Critic direct preparation only",
        }, indent=2))
        return 0
    result = run_canary(
        args.output_root,
        expected_source_fingerprint=args.source_fingerprint,
    )
    print(json.dumps({
        "output": os.fspath(args.output_root),
        "status": result["status"],
        "findings": result["findings"],
    }))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
