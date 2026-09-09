"""
Offline scoring of a finished vanilla trace (the ORM path).

The solver ran with nothing mounted but the recording hooks: its
distribution is untouched, and all it leaves behind is a state dir with
`hook_feed.jsonl` (raw tool events), `task.md`, `baseline/`, and
`final.txt` (its final output). This module turns that into a scored
claim graph in three steps, each reusing the online machinery unchanged:

  1. ingest   — feed lines become seq-numbered ledger events (the same
                conversion the MCP server's drain_hooks performs);
  2. extract  — an LLM role reconstructs declarative `extracted_commit`
                events that the solver
                never made; validated LLM-free by replaying through
                Builder, with structural errors (and ONLY structural
                errors) fed back for a bounded retry. The extractor is
                outcome-blind: verification verdicts never reach it, so
                extraction stays independent of scoring.
  3. score    — the certify path (Service + request_certificate, lazy
                trigger) settles the final's slice; the certificate is
                folded into a feature dict and a rank_key.

rank_key is lexicographic, not a weighted sum: the certificate verdict
dominates absolutely; fewer violated advisory conditions on the FINAL
claim's task contract break ties. Passed-check fractions and history claims
remain diagnostic features, but cannot improve a candidate's rank merely by
making the extractor's decomposition or narration more verbose. A trace whose
extraction fails outright, or whose feature record predates this ranking
schema, ranks below everything — fail-closed, same as the rest of the system.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from typing import Any, Dict, List, Optional

from .core.events import EventLog, SOURCE_EVENT_TYPES
from .core.model import canonical, content_hash
from .core.store import ClaimStore, TRIGGER_ON_FIRST_DEPENDENT, UnknownDependency
from .engine.certificate import request_certificate
from .llm.model_spans import model_scope
from .llm.schema import RoleError
from .stream.builder import Builder, TraceError

HOOK_EVENT_TYPES = ("tool_call", "tool_result")
OFFLINE_FINAL_MESSAGE_SOURCE = "offline_final"
SNIPPET_CHARS = 160
INDEX_MAX_EVENTS = 500

# Written by a trusted adapter only after every immutable verifier input has
# been published.  It is deliberately not part of the seal it authenticates.
IMMUTABLE_INPUT_SEAL_FILE = "immutable_input_seal.json"
IMMUTABLE_INPUT_TAINT_FILE = "immutable_input_tainted.json"
# Explicit offline experiments retain their artifacts for inspection, but
# those artifacts must never leak into an ordinary scoring invocation.
DIAGNOSTIC_ONLY_FILE = ".contract_replay_diagnostic.json"
IMMUTABLE_INPUT_SEAL_SCHEMA = "gv-immutable-input-seal-v1"
IMMUTABLE_INPUT_PATHS = (
    "world",
    "world_manifest.json",
    "task.md",
    "baseline",
    "hook_feed.jsonl",
    "final.txt",
)


class ImmutableInputError(OSError):
    """The adapter-sealed offline evidence boundary is absent or unsafe."""


def _regular_file_seal(path: str, st: os.stat_result) -> Dict[str, Any]:
    """Hash one unlinked regular file and reject an in-place sealing race."""
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_CLOEXEC", 0))
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (st.st_dev, st.st_ino)):
            raise ImmutableInputError(
                f"immutable input changed while sealing: {path}")
        # A writable alias outside the sealed tree would let a supposedly
        # read-only input change through a path the sandbox is allowed to
        # write.  Captured inputs are copies, so fail closed on hard links.
        if opened.st_nlink != 1:
            raise ImmutableInputError(
                f"hard-linked immutable input is forbidden: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        closed = os.fstat(fd)
        stable_fields = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
            "st_mtime_ns", "st_ctime_ns",
        )
        if any(getattr(opened, field) != getattr(closed, field)
               for field in stable_fields):
            raise ImmutableInputError(
                f"immutable input changed while hashing: {path}")
    finally:
        os.close(fd)
    return {"type": "file", "mode": stat.S_IMODE(opened.st_mode),
            "size": opened.st_size, "nlink": opened.st_nlink,
            "sha256": digest.hexdigest()}


def _sealed_tree(path: str) -> List[Dict[str, Any]]:
    """Deterministic snapshot of one immutable tree, failing closed.

    A symlink cannot be treated as frozen evidence merely by hashing its link
    text: its target may live in writable ``work/`` or elsewhere on the host.
    FIFOs, sockets and devices likewise have live semantics rather than fixed
    bytes.  Adapters must materialize such inputs before publishing a seal.
    """
    if not os.path.lexists(path):
        return [{"path": ".", "type": "absent"}]
    records: List[Dict[str, Any]] = []
    pending = [(".", path)]
    while pending:
        relative, current = pending.pop()
        st = os.lstat(current)
        base = {"path": relative, "mode": stat.S_IMODE(st.st_mode)}
        if stat.S_ISDIR(st.st_mode):
            records.append({**base, "type": "dir"})
            children = sorted(os.listdir(current), reverse=True)
            closed = os.lstat(current)
            if (not stat.S_ISDIR(closed.st_mode)
                    or (closed.st_dev, closed.st_ino)
                    != (st.st_dev, st.st_ino)
                    or closed.st_mtime_ns != st.st_mtime_ns
                    or closed.st_ctime_ns != st.st_ctime_ns):
                raise ImmutableInputError(
                    f"immutable directory changed while sealing: {current}")
            for name in children:
                child_relative = name if relative == "." \
                    else f"{relative}/{name}"
                pending.append((child_relative, os.path.join(current, name)))
        elif stat.S_ISREG(st.st_mode):
            records.append({"path": relative,
                            **_regular_file_seal(current, st)})
        elif stat.S_ISLNK(st.st_mode):
            # Seal the link as a pointer (git semantics): hash its target
            # string, never follow it. In-tree targets are sealed by the walk
            # anyway; out-of-tree targets stay outside the evidence boundary,
            # which the capture-manifest rules already make undecidable
            # rather than trusted. Refusing symlinks outright made every
            # build-style world (povray, cython, compcert) unscoreable.
            records.append({**base, "type": "symlink",
                            "target": os.readlink(current)})
        else:
            raise ImmutableInputError(
                f"special file forbidden in immutable input: {current}")
    records.sort(key=lambda row: row["path"])
    return records


def captured_world_seal(state_dir: str) -> str:
    """Content identity for the captured world and its manifest only."""
    state = os.path.abspath(state_dir)
    return content_hash({
        "world": _sealed_tree(os.path.join(state, "world")),
        "manifest": _sealed_tree(os.path.join(state, "world_manifest.json")),
    })


def immutable_input_seal(state_dir: str) -> str:
    """Content identity for every immutable input consumed by offline GV."""
    state = os.path.abspath(state_dir)
    return content_hash({
        relative: _sealed_tree(os.path.join(state, relative))
        for relative in IMMUTABLE_INPUT_PATHS
    })


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".gv-seal-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary:
            try:
                os.remove(temporary)
            except OSError:
                pass


def write_immutable_input_anchor(state_dir: str) -> str:
    """Publish the trusted adapter anchor after all source inputs exist."""
    state = os.path.abspath(state_dir)
    if os.path.lexists(os.path.join(state, IMMUTABLE_INPUT_TAINT_FILE)):
        os.remove(os.path.join(state, IMMUTABLE_INPUT_TAINT_FILE))
    seal = immutable_input_seal(state)
    _write_json_atomic(
        os.path.join(state, IMMUTABLE_INPUT_SEAL_FILE),
        {
            "schema": IMMUTABLE_INPUT_SEAL_SCHEMA,
            "seal": seal,
            "paths": list(IMMUTABLE_INPUT_PATHS),
        },
    )
    return seal


def _read_regular_json(path: str) -> Dict[str, Any]:
    try:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode):
            raise ImmutableInputError(
                f"immutable input seal is not a regular file: {path}")
        if st.st_size > 65_536:
            raise ImmutableInputError(
                f"immutable input seal is too large: {path}")
        flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_CLOEXEC", 0))
        fd = os.open(path, flags)
        try:
            with os.fdopen(fd, encoding="utf-8") as handle:
                value = json.load(handle)
        except Exception:
            # fdopen owns fd after construction.
            raise
    except (OSError, ValueError, json.JSONDecodeError) as err:
        if isinstance(err, ImmutableInputError):
            raise
        raise ImmutableInputError(
            f"cannot read immutable input seal {path}: {err}") from err
    if not isinstance(value, dict):
        raise ImmutableInputError(
            f"immutable input seal is not a JSON object: {path}")
    return value


def verify_immutable_input_anchor(state_dir: str) -> str:
    """Return the trusted seal only when the current source bytes match."""
    state = os.path.abspath(state_dir)
    taint = os.path.join(state, IMMUTABLE_INPUT_TAINT_FILE)
    if os.path.lexists(taint):
        raise ImmutableInputError(
            f"offline state is quarantined after immutable-input mutation: {taint}")
    anchor = _read_regular_json(
        os.path.join(state, IMMUTABLE_INPUT_SEAL_FILE))
    expected_paths = list(IMMUTABLE_INPUT_PATHS)
    seal = anchor.get("seal")
    if (anchor.get("schema") != IMMUTABLE_INPUT_SEAL_SCHEMA
            or anchor.get("paths") != expected_paths
            or not isinstance(seal, str) or len(seal) != 64):
        raise ImmutableInputError("invalid immutable input seal payload")
    actual = immutable_input_seal(state)
    if actual != seal:
        raise ImmutableInputError(
            f"immutable input seal mismatch: expected {seal}, got {actual}")
    return seal


def _quarantine_immutable_inputs(
    state_dir: str, *, expected: str, actual: str, reason: str,
) -> None:
    _write_json_atomic(
        os.path.join(os.path.abspath(state_dir), IMMUTABLE_INPUT_TAINT_FILE),
        {
            "schema": "gv-immutable-input-taint-v1",
            "expected": expected,
            "actual": actual,
            "reason": reason,
        },
    )

# Version 1 feature records did not distinguish the final claim's own contract
# obligations from obligations attached to narration/support claims. They
# therefore cannot safely be used by the current BoN ranker. Keep this version
# explicit so mixed/legacy result files fail closed instead of silently
# reverting to the old whole-slice heuristic.
FEATURE_SCHEMA_VERSION = 3


def pipeline_fingerprint() -> str:
    """Version key over the whole verifier implementation: every .py source
    in the package (prompts, schemas, kernel, certificate, game, discharge,
    this module — all of it). A cached extraction, a settled claim store,
    or a score record stamped with a different fingerprint was produced by
    a DIFFERENT verifier — reusing it silently would attribute the old
    pipeline's judgments to the new one, poisoning every A/B comparison."""
    root = os.path.dirname(os.path.abspath(__file__))
    sources: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                sources[os.path.relpath(path, root)] = _read(path)
    return content_hash(sources)[:16]


def config_fields(model: Optional[str] = None, seats: Optional[str] = None,
                  effort: Optional[str] = None,
                  max_rounds: int = 3,
                  critic_rounds: Optional[int] = None,
                  service_tier: Optional[str] = None,
                  compact_inspection: bool = False,
                  judger_prefetch: bool = False) -> Dict[str, Any]:
    """The full identity of a scoring run's judgment: the pipeline
    fingerprint plus the resolved run configuration, as plaintext fields
    with a content hash (`config`). Two configs must never share derived
    state or score records — two models scoring the same trace are two
    different verifiers. Experiment drivers key their resume on `config`
    and store the plaintext fields alongside for analysis.
    max_rounds bounds extraction retries; critic_rounds bounds the
    decomposition game (default: the kernel's MAX_CRITIC_ROUNDS)."""
    if type(compact_inspection) is not bool:
        raise ValueError("compact_inspection must be a bool")
    if compact_inspection and seats != "codex":
        raise ValueError("compact_inspection is only valid for Codex seats")
    if type(judger_prefetch) is not bool:
        raise ValueError("judger_prefetch must be a bool")
    if judger_prefetch and seats != "codex":
        raise ValueError("judger_prefetch is only valid for Codex seats")
    if not model and seats == "codex":
        from .llm.codex_backend import DEFAULT_MODEL
        model = DEFAULT_MODEL
    if seats == "codex":
        from .llm.codex_service import normalize_service_tier
        resolved_service_tier = normalize_service_tier(service_tier)
    else:
        if service_tier not in (None, "standard"):
            raise ValueError("service_tier is only valid for Codex seats")
        resolved_service_tier = "n/a"
    if critic_rounds is None:
        from .engine.kernel import MAX_CRITIC_ROUNDS
        critic_rounds = MAX_CRITIC_ROUNDS
    fields: Dict[str, Any] = {
        "model": model,
        "seats": seats or "sdk",
        "effort": effort or ("medium" if seats == "codex" else "n/a"),
        "service_tier": resolved_service_tier,
        "max_rounds": max_rounds,
        "critic_rounds": critic_rounds,
        "compact_inspection": compact_inspection,
        "judger_prefetch": judger_prefetch,
    }
    if judger_prefetch:
        from .engine.judger_prefetch import PREFETCH_PROTOCOL
        fields["judger_prefetch_protocol"] = PREFETCH_PROTOCOL
    else:
        fields["judger_prefetch_protocol"] = None
    # Optional verifier instruments are part of the experiment, not ambient
    # machine state.  Stamp the pinned public identity (never its host path)
    # so score resumption cannot mix runs with different browser observers.
    from .execution.browser import configured_browser_observer
    from .execution.config import (
        APPEND_ONLY_DELTA_PROTOCOL,
        append_only_deltas_enabled,
        offline_runtime_read_roots,
        typed_adapters_enabled,
        typed_judger_packet_enabled,
    )
    fields["typed_adapters"] = typed_adapters_enabled()
    fields["typed_adapter_judger_packet"] = typed_judger_packet_enabled()
    if fields["typed_adapter_judger_packet"] and not fields["typed_adapters"]:
        raise ValueError(
            "typed-adapter Judger packet requires typed adapters")
    if fields["typed_adapter_judger_packet"]:
        from .engine.typed_judger_packet import TYPED_JUDGER_PACKET_PROTOCOL
        fields["typed_adapter_judger_packet_protocol"] = (
            TYPED_JUDGER_PACKET_PROTOCOL)
    else:
        fields["typed_adapter_judger_packet_protocol"] = None
    # The shipped online/offline kernel never injects a contract cache: the
    # Obligation Compiler and Critic are part of every production run.
    fields["contract_cache"] = False
    fields["append_only_deltas"] = append_only_deltas_enabled()
    fields["append_only_delta_protocol"] = (
        APPEND_ONLY_DELTA_PROTOCOL
        if fields["append_only_deltas"] else None
    )
    fields["offline_runtime_read_roots"] = (
        list(offline_runtime_read_roots()) if fields["seats"] == "codex" else [])
    browser_observer = (configured_browser_observer()
                        if fields["typed_adapters"] else None)
    fields["browser_observer"] = (
        dict(browser_observer.identity) if browser_observer is not None
        else None
    )
    if fields["seats"] == "sdk":
        from .llm import provider

        fields.update({
            "model": model or provider.MODEL,
            "provider": provider.PROVIDER,
            # A custom URL can contain credentials. Track its identity
            # without copying it into public score records.
            "endpoint_fingerprint": content_hash(provider.ANTHROPIC_BASE_URL),
            "text_only": provider.TEXT_ONLY,
        })
    fields["config"] = content_hash(
        {"pipeline": pipeline_fingerprint(), **fields})[:16]
    return fields


def config_id(model: Optional[str] = None, seats: Optional[str] = None,
              effort: Optional[str] = None, max_rounds: int = 3,
              critic_rounds: Optional[int] = None,
              service_tier: Optional[str] = None,
              compact_inspection: bool = False,
              judger_prefetch: bool = False) -> str:
    return config_fields(
        model, seats, effort, max_rounds, critic_rounds,
        service_tier=service_tier,
        compact_inspection=compact_inspection,
        judger_prefetch=judger_prefetch,
    )["config"]


class ExtractionError(RuntimeError):
    """Extraction exhausted its rounds without a structurally valid graph."""


def _read(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def ingest_feed(feed_path: str, log: EventLog) -> int:
    """
    Raw hook-feed lines -> canonical ledger events (seq/ts assigned).
    Mirrors Service._ingest_hook_event; unknown lines are ignored (the
    feed is untrusted input).
    """
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


def append_final_response(log: EventLog, final_answer: str) -> Dict[str, Any]:
    """Retain the sealed vanilla solver response as ordinary source evidence."""
    if not isinstance(final_answer, str):
        raise ValueError("offline final response must be text")
    return log.append("message", {
        "role": "assistant",
        "text": final_answer,
        "source": OFFLINE_FINAL_MESSAGE_SOURCE,
    })


def is_offline_final_response(event: Dict[str, Any]) -> bool:
    if type(event) is not dict or event.get("type") != "message" \
            or type(event.get("payload")) is not dict:
        return False
    payload = event["payload"]
    return (
        isinstance(payload.get("text"), str)
        and payload == {
            "role": "assistant",
            "text": payload["text"],
            "source": OFFLINE_FINAL_MESSAGE_SOURCE,
        }
    )


def events_index(log: EventLog) -> str:
    """
    Compact one-line-per-event rendering for the extractor's prompt;
    the full ledger file stays available to Read/Grep.
    """
    lines: List[str] = []
    for event in log.replay():
        if len(lines) >= INDEX_MAX_EVENTS:
            lines.append("... (index truncated; full ledger has more events)")
            break
        payload = event["payload"]
        body = payload.get("input") if event["type"] == "tool_call" \
            else payload.get("text") if event["type"] == "message" \
            else payload.get("excerpt")
        snippet = canonical(body)[:SNIPPET_CHARS] if body is not None else ""
        lines.append(
            f"{event['seq']} {event['type']} {payload.get('tool')} :: {snippet}")
    return "\n".join(lines) if lines else "(empty ledger)"


def _apply_commits(log: EventLog, commits: List[Dict[str, Any]], *,
                   event_type: str = "extracted_commit") -> Optional[str]:
    """
    Append the extractor's commits to the candidate log, validating
    LLM-free through a trigger-less Builder replay. Returns the first
    structural error as text, or None on success.

    ``event_type`` exists only so diagnostic replay can retain the identity of
    legacy snapshots whose extractor output was recorded as ``commit``.
    """
    if event_type not in ("commit", "extracted_commit"):
        raise ValueError("invalid extracted commit event type")
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
            builder.feed(log.append(event_type, payload))
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
            max_rounds: int = 3,
            stamp: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Reconstruct extracted-commit events for the trace in `state_dir` and write
    the full events.jsonl. Retries feed back structural errors only.
    Never verification results. Raises ExtractionError when every round
    fails. On success returns (and persists to extract.json) the meta;
    `stamp` (e.g. the caller's config id) is merged into it.
    """
    if backend is None:
        from .llm.agents import AgentBackend
        backend = AgentBackend(cwd=state_dir, model=model)
    from .llm.agents import EXTRACT_CLAIMS_AGENT

    input_seal = ((stamp or {}).get("immutable_input_seal")
                  or immutable_input_seal(state_dir))
    feed = os.path.join(state_dir, "hook_feed.jsonl")
    candidate = os.path.join(state_dir, "events.extract.jsonl")
    task = _read(os.path.join(state_dir, "task.md"))
    final_answer = _read(os.path.join(state_dir, "final.txt"))

    feedback, last_error = "", None
    for round_no in range(1, max_rounds + 1):
        # Fresh candidate per attempt: a failed round's partial commits
        # must not survive into the next
        if os.path.exists(candidate):
            os.remove(candidate)
        log = EventLog(candidate)
        ingest_feed(feed, log)
        append_final_response(log, final_answer)
        variables = {
            "task": task,
            "final_answer": final_answer,
            "events_index": events_index(log),
            "ledger_path": os.path.abspath(candidate),
            "feedback": feedback,
        }
        try:
            with model_scope(
                    locus="candidate_extract", stage="extract",
                    round=round_no):
                payload = EXTRACT_CLAIMS_AGENT.call(backend, variables)
        except RoleError as err:
            last_error = feedback = str(err)
            continue
        error = _apply_commits(log, payload["commits"])
        if error is None:
            os.replace(candidate, os.path.join(state_dir, "events.jsonl"))
            meta = {"rounds": round_no, "commits": len(payload["commits"]),
                    "fingerprint": pipeline_fingerprint(), **(stamp or {}),
                    "immutable_input_seal": input_seal}
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
            if line and json.loads(line)["type"] in (
                    "commit", "extracted_commit"):
                return True
    return False


def _unscoreable(error: str, extract_rounds: int,
                 fingerprint: str = "") -> Dict[str, Any]:
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "unscoreable": True,
        "error": error,
        "verdict": None,
        "verdict_strength": None,
        "slice_size": 0,
        "claims": {},
        "claims_effective": {},
        "conditional_verified": 0,
        "obligations": {},
        "obligations_total": 0,
        "final_obligations": {},
        "final_obligations_total": 0,
        "final_contrary_signals": 0,
        "contract_audit_status": "unavailable_unscoreable",
        "final_obligation_contract": [],
        "final_obligation_outcomes_ordered": [],
        "extract_rounds": extract_rounds,
        "fingerprint": fingerprint,
    }


def _compiled_final_contract(
    log: EventLog, final_id: str,
) -> tuple[str, List[Dict[str, Any]]]:
    """Return the exact, ordered compiler output attached to the final claim.

    The certificate intentionally summarizes settled graph nodes and SQLite
    does not preserve compiler order or the required evidence floor.  The
    append-only event is therefore the audit source.  Keep wording and source
    clauses verbatim: hashing or normalizing them would hide quantifier drift.
    """
    absent = object()
    compiled: Any = absent
    for event in log.replay():
        if event.get("type") != "obligations_compiled":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("claim") == final_id:
            compiled = payload.get("obligations")
    if compiled is absent:
        return "missing", []
    if compiled == []:
        return "empty", []
    if not isinstance(compiled, list):
        return "malformed", []
    fields = (
        "id", "text", "premise_scope", "required_basis", "severity",
        "clause", "refutable_by_absence",
    )
    out: List[Dict[str, Any]] = []
    ids: set[str] = set()
    for ordinal, obligation in enumerate(compiled):
        if (not isinstance(obligation, dict)
                or any(field not in obligation for field in fields)
                or not isinstance(obligation.get("id"), str)
                or not obligation["id"] or obligation["id"] in ids
                or not isinstance(obligation.get("text"), str)
                or not obligation["text"]
                or not isinstance(obligation.get("premise_scope"), list)
                or any(not isinstance(value, str)
                       for value in obligation["premise_scope"])
                or obligation.get("required_basis") not in {"record", "world"}
                or obligation.get("severity") not in {"must", "should"}
                or not isinstance(obligation.get("clause"), str)
                or not isinstance(obligation.get("refutable_by_absence"), bool)):
            return "malformed", []
        ids.add(obligation["id"])
        out.append({
            "ordinal": ordinal,
            **{field: obligation[field] for field in fields},
        })
    return "available", out


def _ordered_final_outcomes(
    cert: Dict[str, Any], contract: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Align certificate outcomes to compiler order without dropping text."""
    final_entry = next(
        (entry for entry in cert["slice"] if entry["id"] == cert["final"]),
        None,
    )
    if final_entry is None:
        raise ValueError("certificate slice is missing its final claim")
    outcomes = {}
    for item in final_entry.get("obligations", []):
        oid = item.get("id")
        if not isinstance(oid, str) or not oid or oid in outcomes:
            raise ValueError("certificate final obligation outcomes are malformed")
        outcomes[oid] = item
    contract_ids = {item["id"] for item in contract}
    unexpected = sorted(set(outcomes) - contract_ids)
    if unexpected:
        raise ValueError(
            f"certificate contains obligations outside compiled contract: {unexpected}")
    ordered = []
    for item in contract:
        outcome = outcomes.get(item["id"])
        ordered.append({
            "ordinal": item["ordinal"],
            "obligation_id": item["id"],
            "evaluated": outcome is not None,
            "status": outcome.get("status") if outcome is not None else None,
            "reason": outcome.get("reason") if outcome is not None else None,
            "basis": outcome.get("basis") if outcome is not None else None,
            "strength": outcome.get("strength") if outcome is not None else None,
        })
    return ordered


def _final_contract_audit(
    log: EventLog, cert: Dict[str, Any],
) -> Dict[str, Any]:
    """Best-effort audit enrichment which never changes score semantics."""
    status, contract = _compiled_final_contract(log, cert["final"])
    outcomes: List[Dict[str, Any]] = []
    if status == "available":
        try:
            outcomes = _ordered_final_outcomes(cert, contract)
        except (KeyError, TypeError, ValueError):
            status = "outcomes_malformed"
    return {
        "contract_audit_status": status,
        "final_obligation_contract": contract,
        "final_obligation_outcomes_ordered": outcomes,
    }


def _features(cert: Dict[str, Any], extract_rounds: int,
              fingerprint: str = "") -> Dict[str, Any]:
    claims: Dict[str, int] = {}
    claims_effective: Dict[str, int] = {}
    conditional_verified = 0           # locally verified, chain still open —
    obligations: Dict[str, str] = {}   # obligation id -> status (dedup: shared
    final_entry: Optional[Dict[str, Any]] = None
    for entry in cert["slice"]:        # nodes appear under several claims)
        if entry["id"] == cert["final"]:
            final_entry = entry
        claims[entry["status"]] = claims.get(entry["status"], 0) + 1
        effective = entry.get("effective_status", entry["status"])
        claims_effective[effective] = claims_effective.get(effective, 0) + 1
        if entry["status"] == "verified" and effective != "verified":
            conditional_verified += 1
        for o in entry.get("obligations", []):
            obligations[o["id"]] = o["status"]
    if final_entry is None:
        raise ValueError("certificate slice is missing its final claim")
    ob_counts: Dict[str, int] = {}
    for status in obligations.values():
        ob_counts[status] = ob_counts.get(status, 0) + 1

    # These are deliberately counted only on the certificate's final entry.
    # They are the independently compiled task contract; obligations on all
    # other slice entries describe the solver's supporting narration and are
    # retained above for diagnostics only.
    final_obligations = {
        o["id"]: o["status"] for o in final_entry.get("obligations", [])
    }
    final_ob_counts: Dict[str, int] = {}
    for status in final_obligations.values():
        final_ob_counts[status] = final_ob_counts.get(status, 0) + 1
    # contrary signals on the contract: advisory (should) clauses the
    # candidate violated. They never refute; they are the only thing that
    # separates two candidates inside one verdict band (fewer is better).
    final_contrary = sum(
        1 for o in final_entry.get("obligations", [])
        if o.get("status") == "refuted"
        and o.get("severity", "must") != "must")
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "unscoreable": False,
        "verdict": cert["verdict"],
        # min strength over the final's evidence when verified/supported
        # (None otherwise): supported@bounded vs supported@analytic get
        # separate precision measurements downstream
        "verdict_strength": cert.get("verdict_strength"),
        "certificate_id": cert["certificate_id"],
        "slice_size": len(cert["slice"]),
        # claims counts LOCAL (conditional) statuses; claims_effective counts
        # the read-time premise-chain fold; conditional_verified = locally
        # verified claims whose chain is still open. Diagnostics only — the
        # rank_key never reads them (fully-established mass outranking
        # conditional mass is enforced by scoring only the final's own
        # contract, which the inheritance gate keeps unconditional).
        "claims": claims,
        "claims_effective": claims_effective,
        "conditional_verified": conditional_verified,
        "obligations": ob_counts,
        "obligations_total": len(obligations),
        "final_obligations": final_ob_counts,
        "final_obligations_total": len(final_obligations),
        "final_contrary_signals": final_contrary,
        "extract_rounds": extract_rounds,
        "fingerprint": fingerprint,
    }


def score(state_dir: str, model: Optional[str] = None, kernel_factory=None,
          backend=None, max_rounds: int = 3,
          seats: Optional[str] = None,
          effort: Optional[str] = None,
          critic_rounds: Optional[int] = None,
          service_tier: Optional[str] = None,
          parallel_obligations: Optional[int] = None,
          parallel_claims: Optional[int] = None,
          shared_dir: Optional[str] = None,
          compact_inspection: bool = False,
          judger_prefetch: bool = False,
          restricted_judger_cwd: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract (if not already done) + certify + fold into features.
    Always returns a feature dict (unscoreable on extraction failure)
    and persists it to score.json, stamped with the run's config fields.

    seats="codex" runs every seat (extractor included) through the Codex
    CLI on a ChatGPT-subscription model instead of the Claude Agent SDK;
    `model` then names the codex model (default gpt-5.6-luna) and
    `effort` its reasoning effort (default medium). ``service_tier`` is
    ``standard`` by default; ``fast`` explicitly enables Codex Fast mode.
    ``compact_inspection`` is an opt-in evidence presentation preference,
    not a change to stage order, evidence scope, or independent review.
    ``judger_prefetch`` is an opt-in verifier-owned evidence packet passed
    to the same independent Judger; it does not change its verdict schema.

    Execution strategy (NOT part of the config identity — it changes the
    schedule, never the judgment): ``parallel_obligations`` discharges
    within one claim concurrently, ``parallel_claims`` verifies the
    certificate slice's claims concurrently (the store then uses the lazy
    trigger so replay does not verify claims one by one first), and
    ``shared_dir`` names a verifier-owned directory outside the state
    whose ``toolchain/`` and ``cache/`` are shared across trials. Defaults
    come from GV_PARALLEL_OBLIGATIONS (4), GV_PARALLEL_CLAIMS (4), and
    GV_SHARED_DIR (none).
    """
    def _env_int(name: str, fallback: int) -> int:
        try:
            return max(1, int(os.environ.get(name, fallback)))
        except ValueError:
            return fallback
    if parallel_obligations is None:
        parallel_obligations = _env_int("GV_PARALLEL_OBLIGATIONS", 4)
    if parallel_claims is None:
        parallel_claims = _env_int("GV_PARALLEL_CLAIMS", 4)
    if shared_dir is None:
        shared_dir = os.environ.get("GV_SHARED_DIR") or None
    toolchain_dir = cache_dir = None
    if shared_dir:
        shared_dir = os.path.abspath(os.path.expanduser(shared_dir))
        toolchain_dir = os.path.join(shared_dir, "toolchain")
        cache_dir = os.path.join(shared_dir, "cache")
        for path in (shared_dir, toolchain_dir, cache_dir):
            os.makedirs(path, exist_ok=True)
    cfg = config_fields(model=model, seats=seats, effort=effort,
                        max_rounds=max_rounds, critic_rounds=critic_rounds,
                        service_tier=service_tier,
                        compact_inspection=compact_inspection,
                        judger_prefetch=judger_prefetch)
    if restricted_judger_cwd is not None and (
            seats != "codex"
            or cfg["typed_adapter_judger_packet"] is not True
            or kernel_factory is not None):
        raise ValueError(
            "restricted_judger_cwd requires the built-in typed Codex "
            "Judger pipeline")
    critic_rounds = cfg["critic_rounds"]
    fingerprint = pipeline_fingerprint()
    extract_rounds = 0
    if os.path.lexists(os.path.join(state_dir, DIAGNOSTIC_ONLY_FILE)):
        # Do not parse, clear, or overwrite experimental state. Even a broken
        # marker means this is not an ordinary input boundary. The caller can
        # create a new state from immutable inputs only; old proof artifacts
        # cannot be made safe merely by re-running extraction.
        raise ImmutableInputError(
            "diagnostic-only offline state cannot be scored normally; "
            "create a fresh state from sealed immutable inputs")
    if (backend is not None
            and getattr(backend, "compact_inspection", False) is not compact_inspection):
        raise ValueError(
            "provided backend compact_inspection does not match the "
            "offline scoring config")
    if (backend is not None
            and getattr(backend, "judger_prefetch", False) is not judger_prefetch):
        raise ValueError(
            "provided backend judger_prefetch does not match the "
            "offline scoring config")
    backend_was_injected = backend is not None
    if seats == "codex":
        from .llm.codex_backend import CodexBackend
        if backend is None:
            # The extraction backend never runs a Judger.  Do not give that
            # earlier stage a handle to the operator-bound Judger-only
            # scratch directory; the Kernel backend below binds it when the
            # fresh independent Judger can actually be selected.
            backend = CodexBackend(cwd=state_dir, model=model,
                                   effort=cfg["effort"],
                                   service_tier=cfg["service_tier"],
                                   compact_inspection=cfg["compact_inspection"],
                                   judger_prefetch=cfg["judger_prefetch"],
                                   offline_toolchain_dir=toolchain_dir,
                                   offline_shared_cache_dir=cache_dir,
                                   restricted_judger_cwd=None)
        elif (isinstance(backend, CodexBackend)
              and backend.service_tier != cfg["service_tier"]):
            raise ValueError(
                "provided CodexBackend service_tier does not match the "
                "offline scoring config")
        expected_restricted_cwd = (
            os.path.abspath(os.fspath(restricted_judger_cwd))
            if restricted_judger_cwd is not None else None)
        actual_restricted_cwd = (
            backend._revalidate_restricted_judger_cwd()
            if type(backend) is CodexBackend else None)
        # An injected backend is an operator authority and must still carry
        # the exact declared binding.  An internally-created extraction
        # backend is deliberately unbound; the built-in Kernel factory is
        # the only consumer of the requested Judger scratch.
        expected_backend_cwd = (
            expected_restricted_cwd if backend_was_injected else None)
        if actual_restricted_cwd != expected_backend_cwd:
            raise ValueError(
                "provided backend does not carry the exact restricted "
                "Judger cwd binding")
        if isinstance(backend, CodexBackend):
            backend.validate_offline_tool_scope(
                state_dir, toolchain_dir, cache_dir)

    # A formal adapter publishes a durable anchor before exposing its
    # ``.sanitized`` marker.  Legacy/direct callers do not have such an
    # adapter, so they still receive a same-invocation seal (and the mutation
    # check below) without pretending that it protects interrupted-run
    # recovery.  A sanitized state without its adapter anchor is never
    # silently blessed: that is precisely the resume-contamination case this
    # boundary exists to prevent.
    try:
        anchor_path = os.path.join(
            os.path.abspath(state_dir), IMMUTABLE_INPUT_SEAL_FILE)
        taint_path = os.path.join(
            os.path.abspath(state_dir), IMMUTABLE_INPUT_TAINT_FILE)
        sanitized_marker = os.path.join(
            os.path.abspath(state_dir), ".sanitized")
        if os.path.lexists(taint_path):
            raise ImmutableInputError(
                "offline state is quarantined after immutable-input mutation: "
                f"{taint_path}")
        if os.path.lexists(anchor_path):
            immutable_seal = verify_immutable_input_anchor(state_dir)
            immutable_boundary = "adapter-sealed"
        elif os.path.lexists(sanitized_marker):
            raise ImmutableInputError(
                "sanitized offline state is missing its trusted adapter seal")
        else:
            immutable_seal = immutable_input_seal(state_dir)
            immutable_boundary = "invocation-sealed"
    except ImmutableInputError as err:
        invalid = dict(_unscoreable(
            f"immutable offline input boundary is not trusted: {err}",
            extract_rounds, fingerprint), **cfg)
        invalid["immutable_input_seal_error"] = str(err)
        return _write_score(state_dir, invalid)

    if seats == "codex":
        if kernel_factory is None:
            def kernel_factory(sc):
                from .mcp_server import _default_kernel
                return _default_kernel(
                    sc, model,
                    backend=CodexBackend(cwd=sc.state_dir, model=model,
                                         effort=cfg["effort"],
                                         service_tier=cfg["service_tier"],
                                         compact_inspection=cfg["compact_inspection"],
                                         judger_prefetch=cfg["judger_prefetch"],
                                         offline_toolchain_dir=toolchain_dir,
                                         offline_shared_cache_dir=cache_dir,
                                         restricted_judger_cwd=(
                                             restricted_judger_cwd)),
                    critic_rounds=critic_rounds,
                    parallel_obligations=parallel_obligations,
                    parallel_claims=parallel_claims)

    def run_verification() -> Dict[str, Any]:
        nonlocal extract_rounds
        events_path = os.path.join(state_dir, "events.jsonl")
        extraction_path = os.path.join(state_dir, "extract.json")
        verification_path = os.path.join(state_dir, "verification.json")
        stamp = {"fingerprint": fingerprint, "config": cfg["config"],
                 "immutable_input_seal": immutable_seal}
        meta = json.loads(
            _read(extraction_path) or "{}")
        if os.path.exists(extraction_path) and (
                meta.get("fingerprint") != fingerprint
                or meta.get("config", cfg["config"]) != cfg["config"]
                or meta.get("immutable_input_seal") != immutable_seal):
            # stale OFFLINE state: the cached extraction and every verdict
            # settled on top of it came from a different pipeline version or
            # a different run config or different task inputs. Wipe derived
            # files only; every immutable
            # source remains protected by the adapter seal.
            for name in ("events.jsonl", "extract.json", "claims.sqlite",
                         "score.json"):
                path = os.path.join(state_dir, name)
                if os.path.exists(path):
                    os.remove(path)
            meta = {}
        if not _has_commits(events_path):
            try:
                meta = extract(state_dir, backend=backend, model=model,
                               max_rounds=max_rounds,
                               stamp=stamp)
                extract_rounds = meta["rounds"]
            except ExtractionError as err:
                return dict(
                    _unscoreable(str(err), max_rounds, fingerprint), **cfg)
        else:
            extract_rounds = meta.get("rounds", 0)

        # Bind verification state to the exact declarative graph as well as
        # candidate hooks.  Offline reconstructed claims are deliberately not
        # source evidence, but tampering with them must still invalidate any
        # materialized judgment.
        trace_hash = hashlib.sha256()
        for event in EventLog(events_path).replay():
            if (event["type"] in SOURCE_EVENT_TYPES
                    or event["type"] == "extracted_commit"):
                trace_hash.update(canonical(event).encode("utf-8"))
                trace_hash.update(b"\n")
        verification_stamp = {**stamp, "source_trace": trace_hash.hexdigest()}
        previous = json.loads(_read(verification_path) or "{}")
        if previous != verification_stamp:
            for name in ("claims.sqlite", "score.json"):
                path = os.path.join(state_dir, name)
                if os.path.exists(path):
                    os.remove(path)
        _write_json_atomic(verification_path, verification_stamp)

        from .mcp_server import Service
        from .core.store import TRIGGER_LAZY
        sc = Service(state_dir, kernel_factory=kernel_factory, model=model,
                     trigger=(TRIGGER_LAZY if parallel_claims > 1
                              else TRIGGER_ON_FIRST_DEPENDENT),
                     live_world=False,
                     critic_rounds=critic_rounds,
                     toolchain_dir=toolchain_dir,
                     shared_cache_dir=cache_dir,
                     parallel_obligations=parallel_obligations,
                     parallel_claims=parallel_claims)
        if not sc.builder.finals:
            return dict(_unscoreable(
                "no final claim in trace", extract_rounds, fingerprint),
                **cfg)
        cert = request_certificate(
            sc.store, sc.kernel, sc.log, sc.artifacts,
            sc.builder.finals[-1])
        features = dict(_features(cert, extract_rounds, fingerprint), **cfg)
        features.update(_final_contract_audit(sc.log, cert))
        features["execution"] = {
            "parallel_obligations": parallel_obligations,
            "parallel_claims": parallel_claims,
            "shared_dir": shared_dir or None,
        }
        return features

    features: Optional[Dict[str, Any]] = None
    failure: Optional[BaseException] = None
    try:
        features = run_verification()
    except BaseException as err:  # preserve cancellation after sealing below
        failure = err

    try:
        immutable_seal_after = immutable_input_seal(state_dir)
    except OSError as err:
        immutable_seal_after = f"error:{err}"

    if immutable_seal_after != immutable_seal:
        reason = "immutable offline inputs mutated during verification"
        _quarantine_immutable_inputs(
            state_dir, expected=immutable_seal,
            actual=immutable_seal_after, reason=reason)
        invalid = dict(
            _unscoreable(reason, extract_rounds, fingerprint), **cfg)
        invalid["immutable_input_seal_expected"] = immutable_seal
        invalid["immutable_input_seal_actual"] = immutable_seal_after
        # Backward-compatible diagnostic aliases for the first seal revision.
        invalid["world_seal_before"] = immutable_seal
        invalid["world_seal_after"] = immutable_seal_after
        _write_score(state_dir, invalid)
        features = invalid

    if failure is not None:
        raise failure
    assert features is not None
    if not features.get("unscoreable"):
        features["immutable_input_seal"] = immutable_seal
        features["immutable_input_boundary"] = immutable_boundary
    return _write_score(state_dir, features)


def _write_score(state_dir: str, features: Dict[str, Any]) -> Dict[str, Any]:
    with open(os.path.join(state_dir, "score.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False)
    return features


# ranking
# supported = the strength ladder's read-out (everything established, some
# of it on declared-and-audited bounded/analytic evidence): strictly under
# verified, strictly above partial (where a necessary check never got
# established at all).
VERDICT_RANK = {"verified": 4, "supported": 3, "partial": 2,
                "unverified": 1, "refuted": 0}


def rank_key(features: Optional[Dict[str, Any]]) -> tuple:
    """Lexicographic trace score, higher-is-better: certificate verdict
    dominates absolutely; inside a verdict band, only the number of
    contrary signals on the final claim's task contract (advisory
    clauses the candidate violated) separates candidates — fewer wins.
    Candidates with the same verdict and the same contrary count are a
    TIE, reported as such by selection.argmax_set: the verifier has no
    further information and says so, rather than ranking on how many
    incidental checks happened to be runnable.

    Whole-slice counts and the verified fraction of the contract are
    diagnostics only and never enter selection. Legacy or malformed
    feature records rank exactly like unscoreable records.
    """
    if not features or features.get("unscoreable"):
        return (-1, 0)
    if features.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        return (-1, 0)

    counts = features.get("final_obligations")
    total = features.get("final_obligations_total")
    if not isinstance(counts, dict) or isinstance(total, bool) \
            or not isinstance(total, int) or total < 0:
        return (-1, 0)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in counts.values()) or sum(counts.values()) != total:
        return (-1, 0)
    verdict = features.get("verdict")
    if verdict not in VERDICT_RANK:
        return (-1, 0)
    contrary = features.get("final_contrary_signals", 0)
    if isinstance(contrary, bool) or not isinstance(contrary, int) \
            or contrary < 0:
        return (-1, 0)
    return (VERDICT_RANK[verdict], -contrary)
