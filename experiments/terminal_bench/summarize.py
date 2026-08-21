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

    rec_selection()


def rec_selection():
    """Best-of-k over the recorder arms' offline scores (rewards are
    Harbor's own ground truth): pass@1 / verifier argmax / pass@k oracle.
    Grouped per job — each job is one generator's candidate pool, and
    pooling candidates across generators would let one job's trials be
    selected for another's task."""
    from common import results as res
    from verifier import selection

    scores_path = os.path.join(HERE, "results", "rec_scores.jsonl")
    rows = res.load(scores_path)
    if not rows:
        return
    by_job = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_job[r.get("job", "?")][r["task"]].append(
            {"solved": bool(r.get("reward")), "features": r["features"],
             "trial": r["trial"]})
    for job in sorted(by_job):
        totals = Counter()
        print(f"\n{'best-of-k [' + job + ']':<28}{'k':>4}"
              f"{'picked verdict':>16}{'pass@1':>8}{'argmax':>8}{'oracle':>8}")
        for task in sorted(by_job[job]):
            trials = by_job[job][task]
            chosen = selection.argmax(trials)
            p1 = selection.pass_at_1(trials)
            argmax_ok = bool(chosen and chosen["solved"])
            oracle_ok = selection.oracle(trials)
            totals["pass@1"] += p1
            totals["argmax"] += argmax_ok
            totals["oracle"] += oracle_ok
            totals["n"] += 1
            print(f"{task:<28}{len(trials):>4}"
                  f"{str(chosen['features'].get('verdict')):>16}"
                  f"{p1:>8.2f}{str(argmax_ok):>8}{str(oracle_ok):>8}")
        n = totals["n"]
        if n:
            argmax_cell = "%d/%d" % (totals["argmax"], n)
            oracle_cell = "%d/%d" % (totals["oracle"], n)
            print(f"{'TOTAL':<28}{'':>4}{'':>16}"
                  f"{totals['pass@1'] / n:>8.2f}{argmax_cell:>8}{oracle_cell:>8}")


if __name__ == "__main__":
    main()
