"""Where does the verifier's wall time go? Per-trial breakdown from the
verifier's own ledger (events.jsonl) and the seat receipts
(role_calls.jsonl) that every Agent SDK call leaves next to it.

For each committed claim: when the solver's commit arrived, when the claim
settled (the commit call blocks in between, so that span is solver wall
time spent waiting), how many obligations were compiled, and how the
waiting decomposes into seat calls (compiler, critic, oracle, executor,
judger, screener) — both as a sum of seat seconds and as the critical
path the parallel schedule actually produced.

Usage:
  uv run python experiments/terminal_bench/latency_breakdown.py \
      experiments/terminal_bench/results/jobs/<job> [--json]
  uv run python experiments/terminal_bench/latency_breakdown.py <gv-state dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List


def _read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _state_dirs(root: str) -> List[str]:
    """Every gv-state directory under root (a job dir, a trial dir, or a
    state dir itself)."""
    if os.path.exists(os.path.join(root, "events.jsonl")):
        return [root]
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "events.jsonl" in filenames and "claims.sqlite" in filenames:
            found.append(dirpath)
            dirnames[:] = []
    return sorted(found)


def _trial_reward(state_dir: str):
    """Harbor's reward for the trial that produced this state, if the
    state sits inside a trial directory (artifacts/gv-state)."""
    cur = state_dir
    for _ in range(4):
        cur = os.path.dirname(cur)
        result = os.path.join(cur, "result.json")
        if os.path.exists(result):
            try:
                with open(result, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                return None, None
            vr = data.get("verifier_result") or {}
            rewards = vr.get("rewards") or {}
            reward = rewards.get("reward", vr.get("reward"))
            started = data.get("started_at")
            finished = data.get("finished_at")
            return reward, (started, finished)
    return None, None


def _seat_family(role: str) -> str:
    return {"obligations": "compile", "critic": "critic",
            "oracles": "oracle", "discharge": "executor",
            "judger": "judger", "screener": "screener",
            "script_judger": "script_judger",
            "extract": "extract"}.get(role, role)


def breakdown(state_dir: str) -> dict:
    events = _read_jsonl(os.path.join(state_dir, "events.jsonl"))
    calls = _read_jsonl(os.path.join(state_dir, "role_calls.jsonl"))
    commits = [e for e in events if e["type"] == "commit"]
    by_claim: Dict[str, dict] = {}
    # a claim id is only known after the commit is applied; the next
    # obligations_compiled / status_change for it reveals the id. Pair by
    # label -> id through the certificate-free route: the store's ids are
    # not in the ledger, so use the text prefix the receipts carry.
    order: List[str] = []
    for e in commits:
        p = e["payload"]
        key = p["label"]
        by_claim[key] = {"label": key, "kind": p["kind"],
                         "text": p["text"][:120], "committed": e["ts"],
                         "settled": None, "status": None,
                         "obligations": 0, "critic_rounds": 0,
                         "dedup_hits": 0, "methods_bound": 0,
                         "method_rejected": 0, "redos": 0,
                         "screened": None,
                         "seats": defaultdict(float), "seat_calls": 0}
        order.append(key)
    # events after a commit and before the next commit belong to that
    # commit's verification (commits are serialized: the call blocks)
    cursor = -1
    current = None
    for e in events:
        if e["type"] == "commit":
            cursor += 1
            current = by_claim[order[cursor]]
            continue
        if current is None:
            continue
        t = e["type"]
        p = e.get("payload", {})
        if t == "obligations_compiled":
            current["obligations"] += len(p.get("obligations", []))
        elif t == "decomposition_attacked":
            current["critic_rounds"] += 1
        elif t == "dedup_hit":
            current["dedup_hits"] += 1
        elif t == "method_bound":
            current["methods_bound"] += 1
        elif t == "method_rejected":
            current["method_rejected"] += 1
        elif t == "method_redo":
            current["redos"] += 1
        elif t == "claim_screened":
            current["screened"] = p.get("confidence")
        elif t == "status_change" and "obligation" not in p:
            current["settled"] = e["ts"]
            current["status"] = p.get("status")
    # seat receipts: attribute by time window (commit .. next commit)
    windows = []
    for i, key in enumerate(order):
        start = by_claim[key]["committed"]
        end = (by_claim[order[i + 1]]["committed"] if i + 1 < len(order)
               else float("inf"))
        windows.append((start, end, key))
    precompile = defaultdict(float)
    precompile_calls = 0
    for c in calls:
        fam = _seat_family(c.get("role", "?"))
        started = c.get("started", 0)
        target = None
        for start, end, key in windows:
            if start <= started < end:
                target = by_claim[key]
                break
        if target is None:
            precompile[fam] += c.get("seconds", 0)
            precompile_calls += 1
            continue
        target["seats"][fam] += c.get("seconds", 0)
        target["seat_calls"] += 1
    claims = []
    total_wait = 0.0
    for key in order:
        c = by_claim[key]
        wait = (c["settled"] - c["committed"]) if c["settled"] else None
        if wait is not None:
            total_wait += wait
        claims.append({**c, "seats": dict(c["seats"]), "wait_s": wait})
    reward, span = _trial_reward(state_dir)
    seat_totals = defaultdict(float)
    for c in claims:
        for k, v in c["seats"].items():
            seat_totals[k] += v
    for k, v in precompile.items():
        seat_totals[k] += v
    return {
        "state_dir": state_dir,
        "reward": reward,
        "trial_span": span,
        "claims": claims,
        "total_commit_wait_s": total_wait,
        "seat_seconds_total": dict(seat_totals),
        "seat_calls": len(calls),
        "background_seats": {"calls": precompile_calls,
                             "seconds": dict(precompile)},
        "certificates": [e["payload"] for e in events
                         if e["type"] == "certificate"],
    }


def render(report: dict) -> str:
    lines = [f"== {report['state_dir']}",
             f"reward={report['reward']} "
             f"commit-wait total={report['total_commit_wait_s']:.0f}s "
             f"seat calls={report['seat_calls']} "
             f"seat seconds by family="
             + json.dumps({k: round(v) for k, v in
                           report['seat_seconds_total'].items()})]
    if report["background_seats"]["calls"]:
        lines.append(f"background seats: {report['background_seats']}")
    for c in report["claims"]:
        wait = f"{c['wait_s']:.0f}s" if c["wait_s"] is not None else "n/a"
        seats = ", ".join(f"{k}={v:.0f}" for k, v in
                          sorted(c["seats"].items(), key=lambda kv: -kv[1]))
        lines.append(
            f"  [{c['kind']}] {c['label']}: wait={wait} status={c['status']}"
            f" screened={c['screened']} obligations={c['obligations']}"
            f" critic_rounds={c['critic_rounds']} dedup={c['dedup_hits']}"
            f" bound={c['methods_bound']} rejected={c['method_rejected']}"
            f" redos={c['redos']} calls={c['seat_calls']} | {seats}")
        lines.append(f"      {c['text']}")
    for cert in report["certificates"]:
        lines.append(f"  certificate: {cert.get('verdict')}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    dirs = _state_dirs(args.root)
    if not dirs:
        sys.exit(f"no gv-state under {args.root}")
    reports = [breakdown(d) for d in dirs]
    if args.json:
        print(json.dumps(reports, indent=1, default=str))
        return
    for r in reports:
        print(render(r))
        print()


if __name__ == "__main__":
    main()
