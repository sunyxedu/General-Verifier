"""Prepare a synthetic, task-neutral record-semantics diagnostic.

This generator makes no model calls and is not a scoring entry point.  It
creates two explicit authorities, each paired across ledgers that differ only
in one literal tool identity.  The gold outcomes come from those synthetic
authorities, not from Terminal-Bench or a model-generated contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from verifier import offline
from verifier.core.model import Obligation, content_hash


SCHEMA = "gv-record-semantic-gate/v1"
SCOPE = "precompiled-record-obligation-binding-only"
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
NONMATCHING_LABEL = "unrelated.tool"


def _authorities(required_label: str) -> dict[str, dict[str, Any]]:
    quoted = json.dumps(required_label, ensure_ascii=False)
    return {
        "exact_identity": {
            "task": (
                "Synthetic record diagnostic. Judge only the complete sealed "
                "event ledger. The required condition is: at least one "
                f"tool_call has payload.tool exactly equal to {quoted} and "
                "has a nonblank string at payload.input.keystrokes. Tool "
                "names are literal identities; do not infer aliases or "
                "underlying runtimes. Absence from the complete ledger "
                "refutes this condition."
            ),
            "obligation": (
                "The complete retained ledger contains at least one tool_call "
                f"whose payload.tool is exactly {quoted} and whose "
                "payload.input.keystrokes is a nonblank string."
            ),
            "clause": (
                f"payload.tool must equal {quoted} literally and "
                "payload.input.keystrokes must be nonblank"
            ),
            "expected": {"required": "holds", "opaque": "refuted"},
        },
        "any_identity": {
            "task": (
                "Synthetic record diagnostic. Judge only the complete sealed "
                "event ledger. The required condition is: regardless of "
                "literal tool identity, at least one tool_call has a nonblank "
                "string at payload.input.keystrokes. Tool identity is "
                "irrelevant. Absence from the complete ledger refutes this "
                "condition."
            ),
            "obligation": (
                "The complete retained ledger contains at least one tool_call, "
                "of any literal tool identity, whose payload.input.keystrokes "
                "is a nonblank string."
            ),
            "clause": (
                "any tool identity is allowed; payload.input.keystrokes must "
                "be nonblank"
            ),
            "expected": {"required": "holds", "opaque": "holds"},
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _events(identity: str) -> list[dict[str, Any]]:
    calls = (
        ("target-nonblank", identity, "printf synthetic-record"),
        ("target-blank", identity, "   \t"),
        ("target-empty", identity, ""),
        ("nonmatching", NONMATCHING_LABEL, "unrelated nonblank input"),
    )
    events = []
    for call_id, tool, command in calls:
        common = {"tool": tool, "tool_use_id": call_id,
                  "session": "synthetic-session", "agent": "synthetic-agent"}
        events.append({"type": "tool_call", "payload": {
            **common, "input": {"keystrokes": command}}})
        result_text = "synthetic result"
        encoded = result_text.encode("utf-8")
        events.append({"type": "tool_result", "payload": {
            **common, "excerpt": result_text,
            "response_chars": len(result_text),
            "response_sha256": hashlib.sha256(encoded).hexdigest(),
            "excerpt_truncated": False,
        }})
    return events


def _strict_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _write_source(path: Path, task: str, identity: str) -> str:
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    (path / "task.md").write_text(task + "\n", encoding="utf-8")
    (path / "final.txt").write_text(
        "synthetic record diagnostic only\n", encoding="utf-8")
    feed = b"".join(_strict_json_bytes(event) for event in _events(identity))
    (path / "hook_feed.jsonl").write_bytes(feed)
    return offline.write_immutable_input_anchor(str(path))


def _case(authority: str, spec: dict[str, Any]) -> dict[str, Any]:
    row_id = content_hash({"schema": SCHEMA, "authority": authority,
                           "task": spec["task"]})
    result = {
        "row": {"id": row_id, "kind": "final",
                "text": "The explicit synthetic record condition holds."},
        "obligation": {
            "claim_id": row_id, "text": spec["obligation"],
            "required_basis": "record", "severity": "must",
            "clause": spec["clause"], "refutable_by_absence": True,
            "premise_scope": [],
        },
    }
    obligation = Obligation(**result["obligation"])
    if not obligation.id:
        raise ValueError("synthetic obligation identity is unavailable")
    return result


def prepare(
    output_root: str | Path, *, required_label: str, opaque_label: str,
) -> dict[str, Any]:
    """Create four new sealed inputs and their operator-owned manifest."""
    labels = (required_label, opaque_label)
    if any(not isinstance(label, str) or _LABEL.fullmatch(label) is None
           for label in labels):
        raise ValueError("labels must be bounded literal identities")
    if len(set((*labels, NONMATCHING_LABEL))) != 3:
        raise ValueError(
            "required_label, opaque_label, and nonmatching label must differ")
    argument = Path(output_root)
    if not argument.is_absolute() or ".." in argument.parts:
        raise ValueError("output_root must be an absolute normalized path")
    output = argument.absolute()
    if os.path.lexists(output):
        raise ValueError("output_root must be new")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)

    records = []
    for authority, spec in _authorities(required_label).items():
        case = _case(authority, spec)
        case_path = output / authority / "case.json"
        case_path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        case_path.write_bytes(_strict_json_bytes(case))
        parsed_obligation = Obligation(**case["obligation"])
        for variant, identity in (("required", required_label),
                                  ("opaque", opaque_label)):
            source = output / authority / variant / "source"
            seal = _write_source(source, spec["task"], identity)
            records.append({
                "authority": authority, "variant": variant,
                "literal_tool_identity": identity,
                "required_tool_identity": required_label,
                "source": str(source), "source_seal": seal,
                "feed_sha256": _sha256(source / "hook_feed.jsonl"),
                "case": str(case_path), "case_sha256": _sha256(case_path),
                "claim_id": case["row"]["id"],
                "obligation_id": parsed_obligation.id,
                "expected_binding_outcome": spec["expected"][variant],
                "probe_arms": ["control", "typed"],
            })
    manifest = {
        "schema": SCHEMA, "scope": SCOPE, "diagnostic_only": True,
        "full_e2e_acceptance": False, "synthetic_gold": True,
        "gold_source": "explicit_operator_authority_in_each_task.md",
        "terminal_bench_input": False, "model_calls": 0,
        "typed_adapter_required": False,
        "probe_fixed_config": {
            "compact_inspection": False,
            "method_cache": False,
            "live_world": False,
        },
        "required_label": required_label, "opaque_label": opaque_label,
        "nonmatching_label": NONMATCHING_LABEL,
        "pairing_rule": (
            "within each authority, sealed inputs differ only in the target "
            "tool identity and the resulting immutable seal"
        ),
        "records": records,
        "edge_records": {
            "target_nonblank": 1, "target_whitespace_only": 1,
            "target_empty": 1, "nonmatching_tool_nonblank": 1,
        },
        "limitations": [
            "diagnoses one precompiled record obligation at a time",
            "does not run extraction, Compiler, Critic, or full scoring",
            "does not establish losslessness on natural-language tasks",
        ],
    }
    offline._write_json_atomic(str(output / "manifest.json"), manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--required-label", required=True)
    parser.add_argument("--opaque-label", required=True)
    args = parser.parse_args(argv)
    manifest = prepare(
        args.output_root, required_label=args.required_label,
        opaque_label=args.opaque_label)
    print(json.dumps({"manifest": str(args.output_root / "manifest.json"),
                      "records": len(manifest["records"]),
                      "model_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
