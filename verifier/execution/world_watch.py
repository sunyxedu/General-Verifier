"""Which world changes a check made, and which the solver made meanwhile.

Online, the solver keeps working while the verifier checks. A pending
verifier call does not stop the solver's other tools (a client may hand
control back to the model while an MCP call is in flight), and the
final's contract is checked in the background while the solver is still
building the delivery. A tree delta across a check's window is therefore
not by itself the check's doing.

The solver's hooks write every tool call to the feed before it runs and
every result after it finishes, each stamped with the wall clock. Read
as it grew during one check, the feed splits the delta in two:

  charged -- changed while no world-mutating solver tool was running:
             the check's own doing, as far as the record can tell
  moved   -- changed while such a tool was running, deleted, or
             stamped outside the window: attributable to neither side

With no solver activity in the window every change is charged, exactly
as before this split existed. With activity but no timestamps on the
feed (an older hook), every change is moved.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.world import tool_mutates_world

HOOK_FEED = "hook_feed.jsonl"
# hook stamps and inode change times come from the same clock; the slack
# absorbs the gap between a hook firing and its tool starting or ending
SLACK_S = 1.0


@dataclass
class WorldWatch:
    """One check's window on the delivered world: the root and its
    manifest when the check began, the feed's length then, and when."""
    root: Optional[str] = None
    before: Optional[dict] = None
    mark: Optional[int] = None
    started: float = 0.0


def feed_mark(state_dir: Optional[str]) -> Optional[int]:
    """The hook feed's length in bytes (0 when it does not exist yet;
    None when there is no state directory to read it from)."""
    if not state_dir:
        return None
    try:
        return os.path.getsize(os.path.join(state_dir, HOOK_FEED))
    except FileNotFoundError:
        return 0
    except OSError:
        return None


def _rows(chunk: bytes) -> List[dict]:
    rows = []
    for raw in chunk.splitlines():
        try:
            row = json.loads(raw)
        except ValueError:
            continue    # a torn or foreign line: the feed is untrusted input
        if isinstance(row, dict) and isinstance(row.get("payload"), dict):
            rows.append(row)
    return rows


def _uid(row: dict) -> str:
    uid = row["payload"].get("tool_use_id")
    return str(uid) if uid else f"anonymous-{id(row)}"


def _ts(row: Optional[dict]) -> Optional[float]:
    if row is None:
        return None
    value = row["payload"].get("ts")
    return float(value) if isinstance(value, (int, float)) else None


def _mutating(row: dict) -> bool:
    return tool_mutates_world(row["payload"].get("tool"))


def solver_activity(state_dir: Optional[str], mark: Optional[int],
                    until: float) -> Dict[str, Any]:
    """The solver's world-mutating tool runs that overlapped a check window
    that began when the feed was `mark` bytes long and ended at `until`.

    A run overlaps when its call or its result was written after the mark,
    or when it was the last thing its session had started before the mark
    and it had not finished by then. Returns {"acted", "timed",
    "intervals", "tools"}: `intervals` are (start, end) wall-clock spans
    (start -inf when the call predates the feed we read, end `until` for
    a run still going); `timed` is False when a run lacks the stamps an
    interval needs."""
    quiet = {"acted": False, "timed": True, "intervals": [], "tools": [],
             "runs": 0}
    if mark is None or not state_dir:
        return quiet
    try:
        with open(os.path.join(state_dir, HOOK_FEED), "rb") as handle:
            data = handle.read()
    except OSError:
        return quiet
    # a line the mark cut through was still being written when the check
    # began: it belongs to the window
    cut = data.rfind(b"\n", 0, mark) + 1 if mark > 0 else 0
    earlier, during = _rows(data[:cut]), _rows(data[cut:])

    calls: Dict[str, dict] = {}
    for row in earlier + during:
        if row.get("type") == "tool_call":
            calls[_uid(row)] = row
    answered = {_uid(row): row for row in during
                if row.get("type") == "tool_result"}
    active: Dict[str, dict] = {}
    last_by_session: Dict[Any, dict] = {}
    for row in earlier:
        last_by_session[row["payload"].get("session")] = row
    for row in last_by_session.values():
        # in flight when the check began: the session's last act before
        # the mark was starting this tool (a later act would mean the
        # session had moved on, and an unanswered call left behind by an
        # interrupted run is not still running)
        if row.get("type") == "tool_call" and _mutating(row):
            active[_uid(row)] = row
    for row in during:
        if row.get("type") in ("tool_call", "tool_result") and _mutating(row):
            active.setdefault(_uid(row), row)

    intervals: List[Tuple[float, float]] = []
    tools = set()
    timed = True
    for uid, row in active.items():
        tools.add(str(row["payload"].get("tool") or "?"))
        call, result = calls.get(uid), answered.get(uid)
        start = _ts(call)
        end = _ts(result) if result is not None else until
        if (call is not None and start is None) \
                or (result is not None and end is None):
            timed = False
            continue
        intervals.append((start if start is not None else float("-inf"),
                          end))
    return {"acted": bool(active), "timed": timed, "intervals": intervals,
            "tools": sorted(tools), "runs": len(active)}


def _delta(before: dict, after: dict, cap: int) -> dict:
    from ..baseline import manifest_delta
    return manifest_delta(before, after, cap)


def split_changes(root: str, before: dict, after: dict,
                  activity: Dict[str, Any], window: Tuple[float, float],
                  cap: int = 20) -> Tuple[dict, dict]:
    """(charged, moved) tree deltas of the root across one check window.

    A changed path is charged to the check when the solver ran nothing
    world-mutating in the window, or when its inode changed inside the
    window at a moment no such solver run spans. Everything else — a
    deletion (no change time survives it), a change inside a solver run,
    a change time outside the window — is moved."""
    added = set(after) - set(before)
    removed = set(before) - set(after)
    modified = {k for k in before if k in after and before[k] != after[k]}
    changed = added | removed | modified
    if not changed:
        return {}, {}
    if not activity.get("acted"):
        return _delta(before, after, cap), {}
    moved = set()
    if not activity.get("timed"):
        moved = set(changed)
    else:
        low, high = window
        spans = activity.get("intervals") or []
        for rel in changed:
            if rel in removed:
                moved.add(rel)
                continue
            try:
                # the inode change time: set by the kernel on every write
                # and not settable by the writer, unlike the modification
                # time a copy or an archive may carry over
                changed_at = os.lstat(os.path.join(root, rel)).st_ctime
            except OSError:
                moved.add(rel)
                continue
            if not (low - SLACK_S <= changed_at <= high + SLACK_S):
                moved.add(rel)
            elif any(start - SLACK_S <= changed_at <= end + SLACK_S
                     for start, end in spans):
                moved.add(rel)
    charged = changed - moved

    def pick(manifest, keep):
        return {k: v for k, v in manifest.items() if k in keep}

    return (_delta(pick(before, charged), pick(after, charged), cap),
            _delta(pick(before, moved), pick(after, moved), cap))


def watch_world(state_dir: Optional[str], live_world: bool) -> WorldWatch:
    """Open a check's window: the feed mark first, then the clock, then the
    manifest, so a solver act that lands while the manifest is being taken
    is inside the window rather than lost between the two."""
    from ..baseline import delivered_root, tree_manifest
    if not live_world:
        return WorldWatch()
    root = delivered_root(state_dir)
    if not root:
        return WorldWatch()
    mark = feed_mark(state_dir)
    started = time.time()
    try:
        before = tree_manifest(root)
    except OSError:
        return WorldWatch()
    return WorldWatch(root=root, before=before, mark=mark, started=started)


def close_watch(watch: WorldWatch, state_dir: Optional[str]
                ) -> Tuple[dict, dict]:
    """(charged, moved) for a window opened by watch_world. The moved half
    carries the solver runs that made the split, for the audit record."""
    from ..baseline import tree_manifest
    if watch.root is None or watch.before is None:
        return {}, {}
    try:
        after = tree_manifest(watch.root)
    except OSError:
        return {}, {}
    ended = time.time()
    activity = solver_activity(state_dir, watch.mark, ended)
    charged, moved = split_changes(watch.root, watch.before, after,
                                   activity, (watch.started, ended))
    if moved:
        moved = dict(moved, solver_tools=activity["tools"],
                     solver_runs=activity["runs"], timed=activity["timed"])
    return charged, moved
