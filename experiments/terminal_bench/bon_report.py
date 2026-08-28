"""Terminal-Bench — BoN correctness report for offline GV scores.

For one (job, config, rep): rank each task's scored trials by
offline.rank_key, mark whether the argmax pick solved the task, and
classify every task where selection could matter. Optionally joins the
LAV baseline results for a head-to-head column.

Usage:
  uv run python experiments/terminal_bench/bon_report.py --job J [--config C]
      [--rep r1] [--lav results/lav_J.jsonl] [--details]

With no --config, the config with the most records for the job is used.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.dirname(HERE), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import harborjobs                       # noqa: E402
from verifier import offline            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--rep", default="r1")
    ap.add_argument("--scores-path", default=os.path.join(
        HERE, "results", "rec_scores.jsonl"))
    ap.add_argument("--lav", default=None,
                    help="lav_<job>.jsonl for the head-to-head column")
    ap.add_argument("--details", action="store_true",
                    help="per-trial rows for every non-clean task")
    ap.add_argument("--merge-configs", action="store_true",
                    help="carry-forward mode: per trial, use the newest GOOD "
                    "record across all configs (config provenance printed); "
                    "prompt iterations then only re-run what failed")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.scores_path)
            if l.strip()]
    recs = [r for r in recs if r.get("job") == args.job
            and r.get("rep") == args.rep]
    if args.merge_configs:
        merged = {}
        for r in recs:   # file order = chronological; later wins, but a
            good = not (r.get("features") or {}).get("unscoreable")
            if r["trial"] not in merged:      # good record is never shadowed
                if good:
                    merged[r["trial"]] = r    # by a later unscoreable one
            elif good:
                merged[r["trial"]] = r
        recs = merged
        args.config = "merged"
    else:
        if args.config is None:
            counts = collections.Counter(r.get("config") for r in recs)
            if not counts:
                sys.exit(f"no records for job={args.job}")
            args.config = counts.most_common(1)[0][0]
        recs = {r["trial"]: r for r in recs
                if r.get("config") == args.config}

    rewards = {}
    for t in harborjobs.trials(os.path.join(HERE, "results", "jobs",
                                            args.job)):
        rewards[t["trial"]] = (t["task"], t["reward"])

    lav = {}
    if args.lav and os.path.exists(args.lav):
        for line in open(args.lav):
            if line.strip():
                r = json.loads(line)
                lav[r["task"]] = r

    by_task = collections.defaultdict(list)
    for trial, (task, reward) in rewards.items():
        by_task[task].append((trial, reward, recs.get(trial)))

    stats = collections.Counter()
    print(f"job={args.job} config={args.config} rep={args.rep} "
          f"scored={len(recs)}")
    header = (f"{'task':38s} {'k':>2s} {'scored':>6s} {'oracle':>6s} "
              f"{'pick':>4s} {'GV':>3s}" + ("  LAV" if lav else ""))
    print(header)
    for task in sorted(by_task):
        rows = by_task[task]
        k = len(rows)
        scored = [(t, rw, r) for t, rw, r in rows if r is not None]
        oracle = any(rw == 1 for _t, rw, _r in rows)
        kind = ("all1" if all(rw == 1 for _t, rw, _r in rows)
                else "all0" if all(rw in (0, None) for _t, rw, _r in rows)
                else "MIXED")
        if len(scored) < k:
            status = "…" if scored else "-"
            print(f"{task:38s} {k:2d} {len(scored):6d} "
                  f"{'y' if oracle else 'n':>6s} {status:>4s}   ?"
                  + (f"    {'✓' if lav.get(task, {}).get('picked_reward') else '✗' if task in lav else '?'}" if lav else ""))
            stats["incomplete"] += 1
            continue
        ranked = sorted(scored, key=lambda x: (
            tuple(-c for c in offline.rank_key(x[2].get("features"))),
            x[0]))
        pick_trial, pick_reward, pick_rec = ranked[0]
        ok = pick_reward == 1
        gv_mark = "✓" if ok else ("✗" if oracle else "·")
        stats[f"{kind}:{gv_mark}"] += 1
        if oracle:
            stats["selectable"] += 1
            stats["selected_ok"] += int(ok)
        lav_col = ""
        if lav:
            lr = lav.get(task)
            lav_col = ("    " + ("✓" if lr and lr.get("picked_reward")
                                 else "✗" if lr else "?"))
        print(f"{task:38s} {k:2d} {len(scored):6d} "
              f"{'y' if oracle else 'n':>6s} "
              f"{pick_trial.split('__')[-1]:>4s}... {gv_mark:>1s}{lav_col}"
              f"   [{kind}]")
        if args.details and (kind == "MIXED" or not ok):
            for t, rw, r in ranked:
                f = r.get("features") or {}
                key = offline.rank_key(f)
                cfg = (" cfg=" + str(r.get("config"))[:8]
                       if args.config == "merged" else "")
                print(f"    {t:44s} reward={rw} "
                      f"verdict={f.get('verdict') or 'unscoreable':10s} "
                      f"frac={key[1]:.3f}{cfg}")
    print()
    sel, ok = stats.get("selectable", 0), stats.get("selected_ok", 0)
    print(f"selectable (oracle=1, fully scored): {sel} · GV picked solved: "
          f"{ok} ({ok/sel:.0%})" if sel else "nothing fully scored yet")
    for key in sorted(stats):
        if ":" in key:
            print(f"  {key}: {stats[key]}")
    if lav:
        both = [t for t in by_task if t in lav]
        lav_ok = sum(1 for t in both if lav[t].get("picked_reward"))
        print(f"LAV on same pool: {lav_ok}/{len(both)} picked solved")


if __name__ == "__main__":
    main()
