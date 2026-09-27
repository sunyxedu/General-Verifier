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
  - the verifier failed (certificate_failed, the verifier restarted while
    the check ran, or its server died and nothing serves the state — the
    state lock is free): the delivery is not at fault; retry once;
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
# the whole hook, reading the ledger included, ends this long after it
# started: a hook the harness kills at 60 s lets the stop through with no
# goal_released on record
HOOK_BUDGET_SECONDS = 40.0
# the state lock a live verifier server holds (verifier.mcp_server
# SERVER_LOCK): takeable means no server is alive to answer a request
SERVER_LOCK = "server.lock"
# states in which the verifier, not the delivery, failed: retried once
VERIFIER_FAILED = ("restarted", "stopped", "failed")
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


class _Ledger:
    """The ledger as read so far. A re-read parses only the complete lines
    appended since the last one: re-parsing a large ledger every poll cost
    more than the hook's whole time on one CPU."""

    def __init__(self, state_dir: str):
        self.path = os.path.join(state_dir, "events.jsonl")
        self.events: List[Dict[str, Any]] = []
        # bumped whenever the file was replaced and read afresh
        self.generation = 0
        self._offset = 0
        self._inode: Optional[int] = None

    def refresh(self) -> bool:
        """Read what was appended; True when the events changed."""
        try:
            with open(self.path, "rb") as f:
                info = os.fstat(f.fileno())
                reset = (info.st_ino != self._inode
                         or info.st_size < self._offset)
                if reset:
                    self.events, self._offset = [], 0
                    self._inode = info.st_ino
                    self.generation += 1
                if info.st_size <= self._offset:
                    return reset
                f.seek(self._offset)
                data = f.read(info.st_size - self._offset)
        except OSError:
            return False
        end = data.rfind(b"\n") + 1     # a torn last line waits for its end
        self._offset += end
        added = False
        for line in data[:end].split(b"\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line.decode("utf-8"))
            except ValueError:
                continue
            if isinstance(event, dict):
                self.events.append(event)
                added = True
        return added or reset


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


def _failed_commits(events: List[Dict[str, Any]]) -> set:
    """Commit seqs the server recorded as having recorded nothing."""
    return {(e.get("payload") or {}).get("commit") for e in events
            if e.get("type") == "commit_failed"}


class _FinalReplay:
    """The ledger's commits replayed, as the server's own replay does:
    through the verifier's Builder with a store that records nothing
    (label resolution without the package), skipping commits a
    commit_failed names. Fed incrementally: only events it has not seen."""

    def __init__(self, failed=()):
        self.failed = set(failed)
        self.seen = 0                   # events consumed
        self.accepted: set = set()      # seqs of the commits that resolved
        self.latest: Optional[Dict[str, Any]] = None
        builder_cls = _verifier_import("verifier.stream.builder", "Builder")
        self.builder = (builder_cls(None, _NullStore())
                        if builder_cls is not None else None)
        self.known: set = set()

    def feed(self, events: List[Dict[str, Any]]) -> None:
        for e in events[self.seen:]:
            self.seen += 1
            self._one(e)

    def _one(self, e: Dict[str, Any]) -> None:
        etype, payload = e.get("type"), e.get("payload") or {}
        if etype not in ("tool_call", "tool_result", "commit"):
            return
        if etype == "commit" and e.get("seq") in self.failed:
            return
        if self.builder is None:
            if etype != "commit":
                self.known.add(f"effect:{e.get('seq')}")
                return
            refs = list(payload.get("deps") or [])
            if payload.get("revises"):
                refs.append(payload["revises"])
            if not all(r in self.known for r in refs) or "text" not in payload:
                return
            if payload.get("label"):
                self.known.add(payload["label"])
            fid = None
        else:
            try:
                self.builder.feed(e)
            except Exception:  # noqa: BLE001 - a failed commit resolves nothing
                return
            if etype != "commit":
                return
            fid = self.builder.finals[-1] if payload.get("kind") == "final" \
                else None
        self.accepted.add(e.get("seq"))
        if payload.get("kind") == "final":
            self.latest = {"seq": e.get("seq") or 0,
                           "label": payload.get("label"), "id": fid}


def latest_final(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The newest final commit that resolved: {seq, label, id}. ``id`` is
    the claim id, or None when the verifier package is unavailable. A
    commit a commit_failed event names is skipped, as the server's replay
    skips it."""
    replay = _FinalReplay(_failed_commits(events))
    replay.feed(events)
    return replay.latest


def server_alive(state_dir: Optional[str]) -> Optional[bool]:
    """Whether a verifier server holds the state's lock (SERVER_LOCK):
    False when the lock can be taken — no server is alive to answer a
    request — and None when that cannot be told (no lock file, no flock)."""
    if not state_dir:
        return None
    try:
        import fcntl
    except ImportError:
        return None
    try:
        fd = os.open(os.path.join(state_dir, SERVER_LOCK), os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError):
            return True
        except OSError:
            return None
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


_UNSET = object()


def final_state(events: List[Dict[str, Any]], goal: str,
                state_dir: Optional[str] = None, final: Any = _UNSET
                ) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
    """(state, latest final, the event that decided it). States: none,
    unrequested, running, restarted, stopped, failed, void, below, met.
    With ``state_dir`` a request no server is alive to answer is
    ``stopped``, not running; ``final`` is the latest final when the
    caller already knows it."""
    if final is _UNSET:
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
        if seq(restart) > seq(request):
            return "restarted", final, request
        # nor does a server that died with nothing restarting it: waiting
        # on it spent the whole wait budget on a check nobody runs
        if server_alive(state_dir) is False:
            return "stopped", final, request
        return "running", final, request
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
    if state in VERIFIER_FAILED:
        if state == "restarted":
            cause = "the verifier restarted while the check was running"
        elif state == "stopped":
            cause = ("the verifier's server exited while the check was "
                     "running and no server is serving the state")
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
    if not texts:
        # a verdict the screener settled: no contract was compiled, the
        # screener's reading of the record is the finding
        for e in reversed(events):
            p = e.get("payload") or {}
            if e.get("type") == "claim_screened" \
                    and p.get("claim") == final_id:
                if p.get("verdict") == "refuted" and not p.get("refused"):
                    note = p.get("witness") or p.get("reason") or ""
                    out.append(f"  - [refuted by the screener on the record] "
                               f"{str(note)[:400]}")
                break
        return out
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


class _Judge:
    """The latest final's state, kept current as the ledger grows: the
    ledger is read incrementally and the Builder replays only commits and
    tool events it has not seen (from scratch only when a commit_failed
    names a commit it accepted, or the file was replaced)."""

    def __init__(self, state_dir: str, goal: str):
        self.state_dir, self.goal = state_dir, goal
        self.ledger = _Ledger(state_dir)
        self.replay: Optional[_FinalReplay] = None
        self._generation = -1

    @property
    def events(self) -> List[Dict[str, Any]]:
        return self.ledger.events

    def judge(self):
        events = self.ledger.events
        if self.replay is not None and self._generation == self.ledger.generation:
            fresh = events[self.replay.seen:]
            if any(e.get("type") == "commit_failed"
                   and (e.get("payload") or {}).get("commit")
                   in self.replay.accepted for e in fresh):
                self.replay = None
        if self.replay is None or self._generation != self.ledger.generation:
            self.replay = _FinalReplay(_failed_commits(events))
            self._generation = self.ledger.generation
        self.replay.failed |= _failed_commits(events[self.replay.seen:])
        self.replay.feed(events)
        return final_state(events, self.goal, state_dir=self.state_dir,
                           final=self.replay.latest)

    def read(self):
        self.ledger.refresh()
        return self.judge()

    def poll(self, judged, deadline: float):
        """Re-read the ledger while the latest final's certificate runs,
        until ``deadline`` less the time a re-read takes."""
        cost = 0.0
        while judged[0] == "running" \
                and time.monotonic() + POLL_INTERVAL + cost < deadline:
            time.sleep(POLL_INTERVAL)
            began = time.monotonic()
            # judged afresh even when nothing was appended: the server may
            # have died meanwhile
            judged = self.read()
            cost = time.monotonic() - began
        return judged


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


def run(state_dir: str, goal: str, max_blocks: int,
        started: Optional[float] = None) -> None:
    """``started``: when the hook process began (monotonic); the whole run,
    reading included, ends within HOOK_BUDGET_SECONDS of it."""
    if started is None:
        started = time.monotonic()
    blocks_path = os.path.join(state_dir, BLOCKS_FILE)
    record = _load_record(blocks_path)
    blocks = record.setdefault("blocks", [])
    waits = record.setdefault("waits", [])
    judge = _Judge(state_dir, goal)
    state, final, decided = judge.read()
    if state == "running":
        state, final, decided = judge.poll(
            (state, final, decided),
            started + min(POLL_SECONDS, HOOK_BUDGET_SECONDS))
    events = judge.events
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
    if state in VERIFIER_FAILED and any(
            b.get("state") in VERIFIER_FAILED
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
    started = time.monotonic()
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
        run(state_dir, goal, max_blocks, started)
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
