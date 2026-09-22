"""Stop hook: the solver may not finish before its delivery is certified.

The claim protocol asks the solver to commit a final claim and request a
certificate. Left to itself, a solver that receives an unverified or
refuted certificate tends to narrate the verdict and stop. This hook
turns the certificate into a goal: when the solver's turn is about to end
and the final claim's certificate has not reached the goal verdict
(default: at least ``partial``), the stop is blocked and the solver is
handed the verifier's own findings — the final obligations that did not
hold, with their witnesses — as the reason to keep working. A bounded
number of blocks (default 3) keeps a genuinely stuck run from looping.

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
from typing import Any, Dict, List, Optional

VERDICT_RANK = {"refuted": 0, "unverified": 1, "partial": 2,
                "supported": 3, "verified": 4}
BLOCKS_FILE = "goal_blocks.json"


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


def assess(events: List[Dict[str, Any]], goal: str) -> Optional[str]:
    """None when the goal is met (or no verdict can be expected yet);
    otherwise the reason the solver must continue."""
    finals = [e for e in events if e.get("type") == "commit"
              and (e.get("payload") or {}).get("kind") == "final"]
    certs = [e for e in events if e.get("type") == "certificate"]
    goal_rank = VERDICT_RANK.get(goal, VERDICT_RANK["partial"])
    if not finals:
        return ("GOAL NOT MET: no final claim has been committed. Commit a "
                "final claim stating that the task is complete (gv.commit "
                "with kind=final) and request gv.certificate on it before "
                "finishing.")
    final_id = finals[-1].get("claim") or (finals[-1].get("payload") or {}).get("claim")
    if not certs:
        # A request that produced no verdict (the verifier errored or did
        # not finish) is not the solver's omission: telling it to request a
        # certificate it already requested sends it chasing a step it has
        # taken, so name the state it is actually in.
        last_final_seq = finals[-1].get("seq") or 0
        requested = any(
            e.get("type") == "verify_request"
            and (e.get("seq") or 0) > last_final_seq for e in events)
        if requested:
            return ("GOAL NOT MET: the certificate for the final claim was "
                    "requested but the verifier returned no verdict (its "
                    "reply carried an error or the check did not finish). "
                    "Read that reply; if it names a defect in the delivery "
                    "or the claim, fix it and commit a corrected final, then "
                    "request gv.certificate again.")
        return ("GOAL NOT MET: the final claim was committed but no "
                "certificate was requested. Request gv.certificate on the "
                "final claim; if the verifier reports obligations that do "
                "not hold, fix the delivery and commit a corrected final.")
    last = certs[-1].get("payload") or {}
    verdict = str(last.get("verdict") or "unverified")
    if VERDICT_RANK.get(verdict, 1) >= goal_rank:
        return None
    findings = _final_findings(events, last.get("final") or final_id)
    lines = [f"GOAL NOT MET: the certificate for the final claim is "
             f"`{verdict}`; at least `{goal}` is required before finishing. "
             "The verifier's findings on the final obligations:"]
    lines += findings or ["  (no obligation detail recorded; ask gv.status "
                          "for the final claim)"]
    lines.append("Address these in the delivery itself (do not merely "
                 "reword the claim), commit a corrected final claim, and "
                 "request gv.certificate again.")
    return "\n".join(lines)


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
        if e.get("type") == "discharge":
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
    blocks_path = os.path.join(state_dir, BLOCKS_FILE)
    try:
        with open(blocks_path, encoding="utf-8") as f:
            record = json.load(f)
    except (OSError, ValueError):
        record = {"blocks": []}
    if len(record.get("blocks", [])) >= max_blocks:
        return
    reason = assess(_events(state_dir), goal)
    if reason is None:
        return
    record.setdefault("blocks", []).append({"reason": reason[:2000]})
    try:
        with open(blocks_path, "w", encoding="utf-8") as f:
            json.dump(record, f)
    except OSError:
        pass
    remaining = max_blocks - len(record["blocks"])
    sys.stdout.write(json.dumps({
        "decision": "block",
        "reason": reason + f"\n(goal hook: {remaining} more continuation"
                           f"{'s' if remaining != 1 else ''} available)",
    }))


if __name__ == "__main__":
    main()
