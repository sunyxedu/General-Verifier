"""Offline scoring of a finished vanilla trace (the ORM path).

The solver ran with nothing mounted but the recording hooks: its
distribution is untouched, and all it leaves behind is a state dir with
`hook_feed.jsonl` (raw tool events), `task.md`, `baseline/`, and
`final.txt` (its final output). This module turns that into a scored
claim graph in three steps, each reusing the online machinery unchanged:

  1. ingest   — feed lines become seq-numbered ledger events (the same
                conversion the MCP server's drain_hooks performs);
  2. extract  — an LLM role reconstructs the commit events the solver
                never made; validated LLM-free by replaying through
                Builder, with structural errors (and ONLY structural
                errors) fed back for a bounded retry. The extractor is
                outcome-blind: verification verdicts never reach it, so
                extraction stays independent of scoring.
  3. score    — the certify path (Service + request_certificate, lazy
                trigger) settles the final's slice; the certificate is
                folded into a feature dict and a rank_key.

rank_key is lexicographic, not a weighted sum: the certificate verdict
dominates absolutely, slice verified-fraction breaks ties, discharged
obligation count breaks those (obligation nodes are roughly invariant
under extraction granularity, so they resist extractor noise; claim
counts are not). A trace whose extraction fails outright ranks below
everything — fail-closed, same as the rest of the system.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .core.events import EventLog
from .core.model import canonical
from .core.store import ClaimStore, TRIGGER_ON_FIRST_DEPENDENT, UnknownDependency
from .engine.certificate import request_certificate
from .llm.roles import RoleError
from .stream.builder import Builder, TraceError

HOOK_EVENT_TYPES = ("tool_call", "tool_result")
SNIPPET_CHARS = 160
INDEX_MAX_EVENTS = 500


class ExtractionError(RuntimeError):
    """Extraction exhausted its rounds without a structurally valid graph."""


def _read(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def ingest_feed(feed_path: str, log: EventLog) -> int:
    """Raw hook-feed lines -> canonical ledger events (seq/ts assigned).
    Mirrors Service._ingest_hook_event; unknown lines are ignored (the
    feed is untrusted input)."""
    count = 0
    if not os.path.exists(feed_path):
        return count
    with open(feed_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if raw.get("type") not in HOOK_EVENT_TYPES:
                continue
            log.append(raw["type"], raw.get("payload", {}))
            count += 1
    return count


def events_index(log: EventLog) -> str:
    """Compact one-line-per-event rendering for the extractor's prompt;
    the full ledger file stays available to Read/Grep."""
    lines: List[str] = []
    for event in log.replay():
        if len(lines) >= INDEX_MAX_EVENTS:
            lines.append("... (index truncated; full ledger has more events)")
            break
        payload = event["payload"]
        body = payload.get("input") if event["type"] == "tool_call" \
            else payload.get("excerpt")
        snippet = canonical(body)[:SNIPPET_CHARS] if body is not None else ""
        lines.append(
            f"{event['seq']} {event['type']} {payload.get('tool')} :: {snippet}")
    return "\n".join(lines) if lines else "(empty ledger)"


def _apply_commits(log: EventLog, commits: List[Dict[str, Any]]) -> Optional[str]:
    """Append the extractor's commits to the candidate log, validating
    LLM-free through a trigger-less Builder replay. Returns the first
    structural error as text, or None on success."""
    builder = Builder(log, ClaimStore())
    for event in log.replay():
        builder.feed(event)
    finals = 0
    for commit in commits:
        payload = {
            "label": commit["label"],
            "kind": commit["kind"],
            "text": commit["text"],
            "deps": commit["deps"],
        }
        if commit.get("revises"):
            payload["revises"] = commit["revises"]
        try:
            builder.feed(log.append("commit", payload))
        except (TraceError, UnknownDependency, ValueError) as err:
            return f"commit {commit['label']!r}: {err}"
        if commit["kind"] == "final":
            finals += 1
    if finals != 1:
        return f"expected exactly one final claim, got {finals}"
    if commits[-1]["kind"] != "final":
        return "the final claim must be the last commit"
    return None


def extract(state_dir: str, backend=None, model: Optional[str] = None,
            max_rounds: int = 3) -> Dict[str, Any]:
    """Reconstruct commit events for the trace in `state_dir` and write
    the full events.jsonl. Retries feed back structural errors only —
    never verification results. Raises ExtractionError when every round
    fails; on success returns (and persists to extract.json) the meta."""
    if backend is None:
        from .llm.agents import AgentBackend
        backend = AgentBackend(cwd=state_dir, model=model)
    from .llm.agents import EXTRACT_CLAIMS_AGENT

    feed = os.path.join(state_dir, "hook_feed.jsonl")
    candidate = os.path.join(state_dir, "events.extract.jsonl")
    task = _read(os.path.join(state_dir, "task.md"))
    final_answer = _read(os.path.join(state_dir, "final.txt"))

    feedback, last_error = "", None
    for round_no in range(1, max_rounds + 1):
        # fresh candidate per attempt: a failed round's partial commits
        # must not survive into the next
        if os.path.exists(candidate):
            os.remove(candidate)
        log = EventLog(candidate)
        ingest_feed(feed, log)
        variables = {
            "task": task,
            "final_answer": final_answer,
            "events_index": events_index(log),
            "ledger_path": os.path.abspath(candidate),
            "feedback": feedback,
        }
        try:
            payload = EXTRACT_CLAIMS_AGENT.call(backend, variables)
        except RoleError as err:
            last_error = feedback = str(err)
            continue
        error = _apply_commits(log, payload["commits"])
        if error is None:
            os.replace(candidate, os.path.join(state_dir, "events.jsonl"))
            meta = {"rounds": round_no, "commits": len(payload["commits"])}
            with open(os.path.join(state_dir, "extract.json"), "w",
                      encoding="utf-8") as f:
                json.dump(meta, f)
            return meta
        last_error = feedback = error
    raise ExtractionError(last_error or "extraction produced nothing")


def _has_commits(events_path: str) -> bool:
    if not os.path.exists(events_path):
        return False
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and json.loads(line)["type"] == "commit":
                return True
    return False


def _unscoreable(error: str, extract_rounds: int) -> Dict[str, Any]:
    return {
        "unscoreable": True,
        "error": error,
        "verdict": None,
        "slice_size": 0,
        "claims": {},
        "obligations": {},
        "obligations_total": 0,
        "extract_rounds": extract_rounds,
    }


def _features(cert: Dict[str, Any], extract_rounds: int) -> Dict[str, Any]:
    claims: Dict[str, int] = {}
    obligations: Dict[str, str] = {}   # obligation id -> status (dedup: shared
    for entry in cert["slice"]:        # nodes appear under several claims)
        claims[entry["status"]] = claims.get(entry["status"], 0) + 1
        for o in entry.get("obligations", []):
            obligations[o["id"]] = o["status"]
    ob_counts: Dict[str, int] = {}
    for status in obligations.values():
        ob_counts[status] = ob_counts.get(status, 0) + 1
    return {
        "unscoreable": False,
        "verdict": cert["verdict"],
        "certificate_id": cert["certificate_id"],
        "slice_size": len(cert["slice"]),
        "claims": claims,
        "obligations": ob_counts,
        "obligations_total": len(obligations),
        "extract_rounds": extract_rounds,
    }


def score(state_dir: str, model: Optional[str] = None, kernel_factory=None,
          backend=None, max_rounds: int = 3) -> Dict[str, Any]:
    """Extract (if not already done) + certify + fold into features.
    Always returns a feature dict (unscoreable on extraction failure)
    and persists it to score.json."""
    events_path = os.path.join(state_dir, "events.jsonl")
    extract_rounds = 0
    if not _has_commits(events_path):
        try:
            meta = extract(state_dir, backend=backend, model=model,
                           max_rounds=max_rounds)
            extract_rounds = meta["rounds"]
        except ExtractionError as err:
            return _write_score(state_dir, _unscoreable(str(err), max_rounds))
    else:
        meta = json.loads(_read(os.path.join(state_dir, "extract.json")) or "{}")
        extract_rounds = meta.get("rounds", 0)

    from .mcp_server import Service
    sc = Service(state_dir, kernel_factory=kernel_factory, model=model,
                 trigger=TRIGGER_ON_FIRST_DEPENDENT)
    if not sc.builder.finals:
        return _write_score(
            state_dir, _unscoreable("no final claim in trace", extract_rounds))
    cert = request_certificate(sc.store, sc.kernel, sc.log, sc.artifacts,
                               sc.builder.finals[-1])
    return _write_score(state_dir, _features(cert, extract_rounds))


def _write_score(state_dir: str, features: Dict[str, Any]) -> Dict[str, Any]:
    with open(os.path.join(state_dir, "score.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False)
    return features


# ---- ranking -----------------------------------------------------------------

VERDICT_RANK = {"verified": 3, "partial": 2, "unverified": 1, "refuted": 0}


def rank_key(features: Optional[Dict[str, Any]]) -> tuple:
    """Lexicographic trace score, higher-is-better: certificate verdict
    dominates absolutely; slice verified-fraction breaks ties; verified
    obligation count breaks those. No weights to tune — selection only
    needs a total order."""
    if not features or features.get("unscoreable"):
        return (-1, 0.0, 0)
    claims = features.get("claims") or {}
    total = features.get("slice_size") or 0
    fraction = claims.get("verified", 0) / total if total else 0.0
    verified_obligations = (features.get("obligations") or {}).get("verified", 0)
    return (VERDICT_RANK.get(features.get("verdict"), 0),
            fraction, verified_obligations)
