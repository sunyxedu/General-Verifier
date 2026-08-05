"""Terminal-Bench — the comparison table.

Usage:  uv run python experiments/terminal_bench/summarize.py
"""
import os
import sqlite3
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from common import results as res                        # noqa: E402
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

    sel = {r["task"]: r for r in res.load(
        os.path.join(HERE, "results", "judge", "selection.jsonl"))}

    with open(os.path.join(HERE, "tasks.txt")) as f:
        tasks = [ln.strip() for ln in f
                 if ln.strip() and not ln.startswith("#")]

    cols = list(arms)
    print(f"{'task':<28}" + "".join(f"{c:>14}" for c in cols)
          + f"{'judge pick':>12}{'GV verdict':>12}")
    stats = Counter()
    for task in tasks:
        row = f"{task:<28}"
        for c in cols:
            rs = [t["reward"] for t in arms[c].get(task, [])
                  if t["reward"] is not None]
            row += f"{('/'.join(str(int(r)) for r in rs) or '?'):>14}"
        s = sel.get(task, {})
        row += f"{str(s.get('picked_reward', '?')):>12}"
        verdict = None
        for t in arms.get("gv", {}).get(task, []):
            verdict = gv_verdict(t["trial_dir"]) or verdict
        row += f"{str(verdict):>12}"
        print(row)

        if s.get("picked_reward") is not None:
            stats["judge_num"] += s["picked_reward"]
            stats["judge_den"] += 1
        plain = [t["reward"] for t in arms.get("plain", {}).get(task, [])
                 if t["reward"] is not None]
        if plain:
            stats["p1_num"] += sum(plain)
            stats["p1_den"] += len(plain)
            stats["oracle"] += max(plain)
            stats["oracle_den"] += 1
        gvr = [t["reward"] for t in arms.get("gv", {}).get(task, [])
               if t["reward"] is not None]
        if gvr:
            if verdict == "verified":
                stats["ver_ok" if max(gvr) > 0 else "ver_BAD"] += 1

    print()
    for c in cols:
        rs = [t["reward"] for ts in arms[c].values() for t in ts
              if t["reward"] is not None]
        if rs:
            print(f"{c:<12} mean reward {sum(rs)/len(rs):.2f} "
                  f"({int(sum(rs))}/{len(rs)} trials)")
    if stats["p1_den"]:
        print(f"plain pass@1: {int(stats['p1_num'])}/{stats['p1_den']} · "
              f"oracle pass@N: {int(stats['oracle'])}/{stats['oracle_den']}")
    if stats["judge_den"]:
        print(f"judge-selected reward: {int(stats['judge_num'])}"
              f"/{stats['judge_den']}")
    print(f"GV verified&passed: {stats['ver_ok']} · "
          f"verified&FAILED (false certificate): {stats['ver_BAD']}")


if __name__ == "__main__":
    main()
