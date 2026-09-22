"""Terminal-Bench — per-trial timing and verdict table for online jobs.

Usage: python experiments/terminal_bench/online_timing.py JOB [JOB ...] [--md OUT]

For every trial of each job: task, reward, agent wall (s), number of
commits, claim statuses, certificate verdicts, method rejections, direct
seat calls and their total seconds (from role_calls.jsonl when the state
was pulled), and whether the /gv-state pull succeeded.
"""
import argparse
import collections
import datetime as dt
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, "results", "jobs")


def trial_row(d):
    task = os.path.basename(d).split("__")[0]
    reward = "pending"
    p = os.path.join(d, "verifier", "reward.txt")
    if os.path.exists(p):
        reward = open(p).read().strip()
    wall = ""
    rp = os.path.join(d, "result.json")
    if os.path.exists(rp):
        a = (json.load(open(rp)).get("agent_execution") or {})
        if a.get("started_at") and a.get("finished_at"):
            s = dt.datetime.fromisoformat(a["started_at"].rstrip("Z"))
            e = dt.datetime.fromisoformat(a["finished_at"].rstrip("Z"))
            wall = round((e - s).total_seconds())
    S = os.path.join(d, "artifacts", "gv-state")
    ev = []
    if os.path.exists(os.path.join(S, "events.jsonl")):
        ev = [json.loads(l) for l in open(os.path.join(S, "events.jsonl")) if l.strip()]
    statuses = dict(collections.Counter(
        x["payload"].get("status") for x in ev if x["type"] == "status_change"))
    cert = [x["payload"].get("verdict") for x in ev if x["type"] == "certificate"]
    commits = [str(x["payload"].get("kind"))[0] for x in ev if x["type"] == "commit"]
    rej = sum(1 for x in ev if x["type"] == "method_rejected")
    rows = []
    if os.path.exists(os.path.join(S, "role_calls.jsonl")):
        rows = [json.loads(l) for l in open(os.path.join(S, "role_calls.jsonl"))]
    direct = [x for x in rows if x.get("backend") == "direct"]
    pull = ""
    mp = os.path.join(d, "artifacts", "manifest.json")
    if os.path.exists(mp):
        pull = ",".join(x.get("status", "") for x in json.load(open(mp))
                        if x.get("source") == "/gv-state") or "-"
    return dict(task=task, reward=reward, wall=wall, commits="".join(commits),
                statuses=statuses, cert=cert, rejections=rej,
                direct_calls=len(direct),
                direct_seconds=round(sum(x["seconds"] for x in direct)),
                pull=pull)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--md", help="append a markdown table to this file")
    args = ap.parse_args()
    lines = []
    for job in args.jobs:
        lines.append(f"\n### {job}  ({dt.datetime.now():%Y-%m-%d %H:%M})\n")
        lines.append("| task | reward | wall s | commits | statuses | cert | rej | direct calls | direct s | pull |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for d in sorted(glob.glob(os.path.join(JOBS, job, "*__*"))):
            r = trial_row(d)
            lines.append(f"| {r['task']} | {r['reward']} | {r['wall']} | {r['commits']} | "
                         f"{r['statuses']} | {r['cert']} | {r['rejections']} | "
                         f"{r['direct_calls']} | {r['direct_seconds']} | {r['pull']} |")
    text = "\n".join(lines)
    print(text)
    if args.md:
        os.makedirs(os.path.dirname(os.path.abspath(args.md)), exist_ok=True)
        with open(args.md, "a", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
