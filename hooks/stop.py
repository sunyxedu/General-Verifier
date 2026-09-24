"""Stop hook: the solver may not finish before its delivery is certified.

The claim protocol asks the solver to commit a final claim and request a
certificate. Left to itself, a solver that receives an unverified or
refuted certificate tends to narrate the verdict and stop. This hook
turns the certificate into a goal: when the solver's turn is about to end
and the latest final claim's certificate has not reached the goal verdict
(default: at least ``partial``), the stop is blocked and the solver is
handed the verifier's own findings — the final obligations that did not
hold, with their witnesses — as the reason to keep working.

Only a certificate for the LATEST final counts: one recorded after that
final's commit and naming its claim id. The claim id is not on the commit
event (it is a content hash over the resolved deps), so the ledger's
commits are replayed through the verifier's own Builder with a store that
records nothing; a commit whose references did not resolve is a failed
commit and never the latest final. Without the verifier package the hook
falls back to label resolution and matches by ledger order alone.

The latest final is in one of these states, each answered differently:
  - its certificate is still running: the hook waits for the verdict (up
    to POLL_SECONDS); if none arrives it blocks with "still running" as a
    WAIT, not a block. Waits have their own cap (MAX_WAITS) so a check
    that never ends cannot hold the solver forever;
  - the verifier failed (certificate_failed, or the verifier restarted
    while the check ran): the delivery is not at fault; retry once;
  - the certificate is void (the verifier was restarted with other
    settings, or ran twice): no delivery fix can certify it; the solver is
    told once and then allowed to stop;
  - the verdict is below the goal: the verifier's findings.

Blocks are budgeted per final (GV_GOAL_MAX_BLOCKS, default 3): a new final
is new work and gets a fresh budget, and GLOBAL_BLOCK_FACTOR times that
bounds the run. Whenever the hook lets the solver stop without the goal
met it appends ``goal_released`` to the ledger, so a released run is never
mistaken for a certified one.

Settings (Claude Code settings.json / Codex hooks.json):
  {"hooks": {"Stop": [{"hooks": [{"type": "command",
      "command": "GV_STATE_DIR=/gv-state python3 hooks/stop.py"}]}]}}

Environment: GV_STATE_DIR (or the directory of GV_HOOK_FEED), GV_GOAL
(verified | supported | partial | unverified; default partial),
GV_GOAL_MAX_BLOCKS (default 3). Set GV_GOAL=off to disable.
"""
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

VERDICT_RANK = {"refuted": 0, "unverified": 1, "partial": 2,
                "supported": 3, "verified": 4}
BLOCKS_FILE = "goal_blocks.json"
# the run's block cap is this many per-final budgets: new finals refresh
# the budget, but a solver cycling through finals still ends
GLOBAL_BLOCK_FACTOR = 3
# a running certificate is waited for inside the hook (its timeout is 60 s)
POLL_SECONDS = 45.0
POLL_INTERVAL = 2.0
# a "still running" block is not charged to the block budget; this many of
# them (each after a full poll, so tens of minutes in all) end the waiting
MAX_WAITS = 30
# a hook that raises blocks at most this many times with its diagnostic
MAX_HOOK_ERROR_BLOCKS = 1
# terminal discharge outcomes (verifier.core.model OUTCOME_*); a discharge
# event carrying anything else (a script's "ran") is progress, not a verdict
OUTCOMES = ("holds", "refuted", "failed", "unobtainable")

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _verifier_import(module: str, name: str):
    """A verifier symbol, or None when the package is not importable."""
    try:
        return getattr(__import__(module, fromlist=[name]), name)
    except Exception:  # noqa: BLE001 - the hook runs without it too
        if _HERE not in sys.path:
            sys.path.append(_HERE)
            return _verifier_import(module, name)
        return None


def _events(state_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(state_dir, "events.jsonl")
    out: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def _append_event(state_dir: str, etype: str, payload: Dict[str, Any]) -> bool:
    """Append to the ledger under the verifier's own writer lock; skipped
    (False) when the package or the ledger is missing."""
    path = os.path.join(state_dir, "events.jsonl")
    event_log = _verifier_import("verifier.core.events", "EventLog")
    if event_log is None or not os.path.isfile(path):
        return False
    try:
        event_log(path).append(etype, payload)
        return True
    except Exception:  # noqa: BLE001 - a note must never break the hook
        return False


class _NullStore:
    """The Builder's only store call; the hook needs its labels, not a graph."""

    def add_claim(self, claim) -> None:
        pass


def latest_final(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The newest final commit that resolved: {seq, label, id}. ``id`` is
    the claim id, or None when the verifier package is unavailable."""
    builder_cls = _verifier_import("verifier.stream.builder", "Builder")
    if builder_cls is None:
        return _latest_final_by_labels(events)
    builder = builder_cls(None, _NullStore())
    latest = None
    for e in events:
        if e.get("type") not in ("tool_call", "tool_result", "commit"):
            continue
        try:
            builder.feed(e)
        except Exception:  # noqa: BLE001 - a failed commit resolves nothing
            continue
        payload = e.get("payload") or {}
        if e["type"] == "commit" and payload.get("kind") == "final":
            latest = {"seq": e.get("seq") or 0, "label": payload.get("label"),
                      "id": builder.finals[-1]}
    return latest


def _latest_final_by_labels(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    known = set()
    latest = None
    for e in events:
        etype, payload = e.get("type"), e.get("payload") or {}
        if etype in ("tool_call", "tool_result"):
            known.add(f"effect:{e.get('seq')}")
        elif etype == "commit":
            refs = list(payload.get("deps") or [])
            if payload.get("revises"):
                refs.append(payload["revises"])
            if not all(r in known for r in refs) or "text" not in payload:
                continue
            if payload.get("label"):
                known.add(payload["label"])
            if payload.get("kind") == "final":
                latest = {"seq": e.get("seq") or 0,
                          "label": payload.get("label"), "id": None}
    return latest


def final_state(events: List[Dict[str, Any]], goal: str
                ) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
    """(state, latest final, the event that decided it). States: none,
    unrequested, running, restarted, failed, void, below, met."""
    final = latest_final(events)
    if final is None:
        return "none", None, {}
    fid = final["id"]

    def mine(e: Dict[str, Any], key: str) -> bool:
        return ((e.get("seq") or 0) > final["seq"]
                and (fid is None or (e.get("payload") or {}).get(key) == fid))

    request = terminal = restart = None
    for e in events:
        etype, payload = e.get("type"), e.get("payload") or {}
        if etype == "verify_request" and payload.get("cause") == "certificate" \
                and mine(e, "target"):
            request = e
        elif etype in ("certificate", "certificate_failed") and mine(e, "final"):
            terminal = e
        elif etype == "server_started":
            restart = e
    seq = lambda e: (e or {}).get("seq") or 0  # noqa: E731
    if request is not None and seq(request) > seq(terminal):
        # a server launched after the request never answers it
        return ("restarted" if seq(restart) > seq(request) else "running",
                final, request)
    if terminal is None:
        return "unrequested", final, {}
    payload = terminal.get("payload") or {}
    if terminal["type"] == "certificate_failed":
        return "failed", final, terminal
    if payload.get("void"):
        return "void", final, terminal
    verdict = str(payload.get("verdict") or "unverified")
    goal_rank = VERDICT_RANK.get(goal, VERDICT_RANK["partial"])
    return ("met" if VERDICT_RANK.get(verdict, 1) >= goal_rank else "below",
            final, terminal)


def message(state: str, events: List[Dict[str, Any]],
            final: Optional[Dict[str, Any]], decided: Dict[str, Any],
            goal: str) -> Optional[str]:
    """What the solver is told in ``state``; None when nothing is owed."""
    payload = decided.get("payload") or {}
    if state == "none":
        return ("GOAL NOT MET: no final claim has been committed. Commit a "
                "final claim stating that the task is complete (gv.commit "
                "with kind=final) and request gv.certificate on it before "
                "finishing.")
    if state == "unrequested":
        return ("GOAL NOT MET: the final claim was committed but no "
                "certificate was requested for it. Request gv.certificate "
                "on the final claim; if the verifier reports obligations "
                "that do not hold, fix the delivery and commit a corrected "
                "final.")
    if state == "running":
        return ("GOAL PENDING: the certificate for the final claim is still "
                "being checked (a final's check can take tens of minutes). "
                "Do not stop and do not commit another final for this: wait, "
                "and poll gv.status on the final claim until its check "
                "finishes, then read the certificate.")
    if state in ("restarted", "failed"):
        if state == "restarted":
            cause = "the verifier restarted while the check was running"
        elif payload.get("infrastructure"):
            cause = ("the verifier's own model provider or transport failed"
                     f" ({str(payload.get('error') or '')[:300]})")
        else:
            cause = ("the verifier raised an error "
                     f"({str(payload.get('error') or '')[:300]})")
        return (f"GOAL NOT MET: the certificate produced no verdict because "
                f"{cause}. This is a failure of the verifier, not a defect "
                "of your delivery: do not rewrite the final for it. Request "
                "gv.certificate on the same final once more; if that fails "
                "too, say so and finish.")
    if state == "void":
        return ("GOAL NOT MET: the certificate is void — "
                f"{payload.get('void')}. A void certificate is no verdict of "
                "the verifier the session started with, and no change to the "
                "delivery can certify it. Report the certificate as void; "
                "you may then finish.")
    if state == "below":
        verdict = str(payload.get("verdict") or "unverified")
        lines = [f"GOAL NOT MET: the certificate for the final claim is "
                 f"`{verdict}`; at least `{goal}` is required before "
                 "finishing. The verifier's findings on the final "
                 "obligations:"]
        fid = payload.get("final") or (final or {}).get("id")
        lines += _final_findings(events, fid) or [
            "  (no obligation detail recorded; ask gv.status for the final "
            "claim)"]
        lines.append("Address these in the delivery itself (do not merely "
                     "reword the claim), commit a corrected final claim, and "
                     "request gv.certificate again.")
        return "\n".join(lines)
    return None


def assess(events: List[Dict[str, Any]], goal: str) -> Optional[str]:
    """None when the goal is met; otherwise the reason the solver must
    continue (or wait)."""
    state, final, decided = final_state(events, goal)
    return message(state, events, final, decided, goal)


def _final_findings(events: List[Dict[str, Any]], final_id: Optional[str],
                    limit: int = 8) -> List[str]:
    if not final_id:
        return []
    texts: Dict[str, str] = {}
    for e in events:
        if e.get("type") == "obligations_compiled" \
                and (e.get("payload") or {}).get("claim") == final_id:
            for o in (e["payload"].get("obligations") or []):
                if isinstance(o, dict) and o.get("id"):
                    texts[o["id"]] = str(o.get("text", ""))
    last_outcome: Dict[str, Dict[str, Any]] = {}
    reasons: Dict[str, str] = {}
    for e in events:
        p = e.get("payload") or {}
        oid = p.get("obligation")
        if not oid or oid not in texts:
            continue
        if e.get("type") == "discharge" and p.get("outcome") in OUTCOMES:
            last_outcome[oid] = p
        if e.get("type") in ("judger_verdict", "method_rejected"):
            note = p.get("witness") or p.get("reason") or ""
            if note:
                reasons[oid] = str(note)
    out = []
    for oid, text in texts.items():
        p = last_outcome.get(oid)
        outcome = (p or {}).get("outcome") or "not decided"
        if outcome == "holds":
            continue
        note = (p or {}).get("witness") or reasons.get(oid) or ""
        line = f"  - [{outcome}] {text[:220]}"
        if note:
            line += f" — {str(note)[:260]}"
        out.append(line)
        if len(out) >= limit:
            break
    return out


def _poll_running(state_dir: str, goal: str, deadline: float):
    """Re-read the ledger while the latest final's certificate runs."""
    events = _events(state_dir)
    judged = final_state(events, goal)
    size = -1
    while judged[0] == "running" and time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL)
        try:
            now = os.path.getsize(os.path.join(state_dir, "events.jsonl"))
        except OSError:
            now = -1
        if now != size:
            size = now
            events = _events(state_dir)
            judged = final_state(events, goal)
    return events, judged


def _load_record(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        if isinstance(record, dict):
            return record
    except (OSError, ValueError):
        pass
    return {}


def _save_record(path: str, record: Dict[str, Any]) -> bool:
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def _release(state_dir: str, record: Dict[str, Any], key: str,
             reason: str) -> None:
    """Let the stop through without the goal met, once per final and
    reason, visibly in the ledger."""
    mark = f"{key}|{reason}"
    released = record.setdefault("released", [])
    if mark in released:
        return
    released.append(mark)
    _append_event(state_dir, "goal_released", {"reason": reason, "final": key})


def _block(reason: str) -> None:
    sys.stdout.write(json.dumps({"decision": "block", "reason": reason}))


def run(state_dir: str, goal: str, max_blocks: int) -> None:
    blocks_path = os.path.join(state_dir, BLOCKS_FILE)
    record = _load_record(blocks_path)
    blocks = record.setdefault("blocks", [])
    waits = record.setdefault("waits", [])
    started = time.monotonic()
    events = _events(state_dir)
    state, final, decided = final_state(events, goal)
    if state == "running":
        events, (state, final, decided) = _poll_running(
            state_dir, goal, started + POLL_SECONDS)
    if state == "met":
        return
    key = (final or {}).get("id") or (
        f"seq:{final['seq']}" if final else "no-final")
    reason = message(state, events, final, decided, goal) or ""
    if state == "running":
        if len(waits) >= MAX_WAITS:
            _release(state_dir, record, key,
                     f"certificate still running after {MAX_WAITS} waits")
            _save_record(blocks_path, record)
            return
        waits.append({"final": key})
        _save_record(blocks_path, record)
        _block(reason + f"\n(goal hook: waiting is not counted against the "
                        f"continuations; {MAX_WAITS - len(waits)} waits left)")
        return
    mine = [b for b in blocks if b.get("final") == key]
    if state == "void" and any(b.get("state") == "void" for b in mine):
        _release(state_dir, record, key, "certificate void")
        _save_record(blocks_path, record)
        return
    if state in ("restarted", "failed") and any(
            b.get("state") in ("restarted", "failed")
            and b.get("event") != decided.get("seq") for b in mine):
        # the solver was told to retry once, and the retry failed as well
        _release(state_dir, record, key, "the verifier failed twice")
        _save_record(blocks_path, record)
        return
    if len(mine) >= max_blocks or len(blocks) >= GLOBAL_BLOCK_FACTOR * max_blocks:
        _release(state_dir, record, key,
                 f"block limit reached with the final {state}")
        _save_record(blocks_path, record)
        return
    blocks.append({"final": key, "state": state, "event": decided.get("seq"),
                   "reason": reason[:2000]})
    _save_record(blocks_path, record)
    remaining = max_blocks - len(mine) - 1
    _block(reason + f"\n(goal hook: {remaining} more continuation"
                    f"{'s' if remaining != 1 else ''} available for this "
                    "final)")


def main() -> None:
    goal = (os.environ.get("GV_GOAL") or "partial").strip().lower()
    if goal in ("off", "0", "none"):
        return
    try:
        json.load(sys.stdin)   # the hook payload; the ledger is our source
    except ValueError:
        pass
    state_dir = os.environ.get("GV_STATE_DIR") or os.path.dirname(
        os.environ.get("GV_HOOK_FEED", os.path.join(".gv", "hook_feed.jsonl")))
    if not state_dir or not os.path.isdir(state_dir):
        return
    try:
        max_blocks = int(os.environ.get("GV_GOAL_MAX_BLOCKS", "3"))
    except ValueError:
        max_blocks = 3
    try:
        run(state_dir, goal, max_blocks)
    except Exception as err:  # noqa: BLE001 - a broken hook leaves a trace
        # never a silent pass: the error goes to the ledger, and the solver
        # is stopped once with it (only when that count is on disk, so a
        # persistently broken hook cannot trap the solver)
        error = f"{type(err).__name__}: {err}"[:500]
        _append_event(state_dir, "goal_hook_error", {"error": error})
        blocks_path = os.path.join(state_dir, BLOCKS_FILE)
        record = _load_record(blocks_path)
        errors = int(record.get("hook_errors") or 0)
        if errors >= MAX_HOOK_ERROR_BLOCKS:
            return
        record["hook_errors"] = errors + 1
        if _save_record(blocks_path, record):
            _block("The goal hook could not assess the certificate "
                   f"({error}). Check with gv.status that the final claim's "
                   "certificate reached its verdict before finishing.")


if __name__ == "__main__":
    main()
