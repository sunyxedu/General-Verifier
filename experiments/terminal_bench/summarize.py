"""Terminal-Bench — the comparison table.

Reads every Harbor job under results/jobs/ (arms: plain / turbo / codex /
gv) plus the gv arm's downloaded /gv-state artifacts for certificate
verdicts.

Usage:  uv run python experiments/terminal_bench/summarize.py
"""
import os
import sqlite3
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import harborjobs                                        # noqa: E402


def gv_verdict(trial_dir):
    """Certificate verdict from the downloaded /gv-state artifact."""
    for root, _dirs, files in os.walk(trial_dir):
        if "claims.sqlite" in files:
            try:
                db = sqlite3.connect(os.path.join(root, "claims.sqlite"))
                row = db.execute("SELECT status FROM claims "
                                 "WHERE kind='final' LIMIT 1").fetchone()
                db.close()
                return row[0] if row else None
            except sqlite3.Error:
                return None
    return None


def main():
    jobs_dir = os.path.join(HERE, "results", "jobs")
    arms = {}
    if os.path.isdir(jobs_dir):
        for job in sorted(os.listdir(jobs_dir)):
            by_task = defaultdict(list)
            for t in harborjobs.trials(os.path.join(jobs_dir, job)):
                by_task[t["task"]].append(t)
            if by_task:
                arms[job] = by_task

    with open(os.path.join(HERE, "tasks.txt")) as f:
        tasks = [ln.strip() for ln in f
                 if ln.strip() and not ln.startswith("#")]

    cols = list(arms)
    print(f"{'task':<28}" + "".join(f"{c:>10}" for c in cols)
          + f"{'GV verdict':>12}")
    stats = Counter()
    for task in tasks:
        row = f"{task:<28}"
        for c in cols:
            rs = [t["reward"] for t in arms[c].get(task, [])
                  if t["reward"] is not None]
            row += f"{('/'.join(str(int(r)) for r in rs) or '?'):>10}"
        verdict = None
        for t in arms.get("gv", {}).get(task, []):
            verdict = gv_verdict(t["trial_dir"]) or verdict
        row += f"{str(verdict):>12}"
        print(row)

        gvr = [t["reward"] for t in arms.get("gv", {}).get(task, [])
               if t["reward"] is not None]
        if gvr and verdict == "verified":
            stats["ver_ok" if max(gvr) > 0 else "ver_BAD"] += 1

    print()
    for c in cols:
        rs = [t["reward"] for ts in arms[c].values() for t in ts
              if t["reward"] is not None]
        if rs:
            print(f"{c:<8} mean reward {sum(rs)/len(rs):.2f} "
                  f"({int(sum(rs))}/{len(rs)} trials)")
    print(f"GV verified&passed: {stats['ver_ok']} · "
          f"verified&FAILED (false certificate): {stats['ver_BAD']}")


if __name__ == "__main__":
    main()
