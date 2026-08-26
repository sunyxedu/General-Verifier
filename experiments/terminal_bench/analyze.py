"""Best-of-N acceptance analysis over rec_scores.jsonl.

Per task: rank trials by offline.rank_key (the selection order), pick
rank-1, and compare the picked trial's benchmark reward against the
random-pick baseline (mean reward) and the oracle ceiling (max reward).
Cross-tabulate verdict x reward to expose the two failure classes that
matter: false certificates (verified & reward=0 — must be zero) and
false refutations (refuted & reward=1 — poison best-of-N by sinking a
correct sample). supported precision is reported per strength tier.

Usage:
  uv run python experiments/terminal_bench/analyze.py --job rec-codex-k3 \
      --config b901ae1614ba971e [--rep r1 r2] \
      [--scores shard-a.jsonl shard-b.jsonl] [--expected-n 5]
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(EXPERIMENTS)
for path in (REPO_ROOT, EXPERIMENTS, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from verifier import offline                              # noqa: E402
from verifier.selection import scalar                     # noqa: E402


def load(paths, job, config, reps):
    rows = {}
    reps = set(reps)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if (r.get("job") == job and r.get("config") == config
                        and r.get("rep") in reps):
                    trial = r["trial"]
                    prior = rows.get(trial)
                    if prior is not None and prior != r:
                        raise ValueError(
                            f"conflicting score rows for {trial} across shards")
                    rows[trial] = r
    return list(rows.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="rec-codex-k3")
    ap.add_argument("--config", required=True)
    ap.add_argument("--rep", nargs="+", default=["r1"])
    ap.add_argument("--scores", nargs="+", default=[os.path.join(
        HERE, "results", "rec_scores.jsonl")],
        help="one or more JSONL score shards (exact duplicate rows deduplicate)")
    ap.add_argument("--expected-n", type=int, default=None,
        help="include only complete N-candidate tasks in BoN totals")
    args = ap.parse_args()

    rows = load(args.scores, args.job, args.config, args.rep)
    if not rows:
        print("no rows for this (job, config, rep)")
        return

    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    # ---- verdict x reward cross table ------------------------------------
    cross = Counter()
    strength_cross = Counter()
    for r in rows:
        f = r.get("features") or {}
        verdict = f.get("verdict") or "unscoreable"
        reward = r.get("reward")
        cross[(verdict, reward)] += 1
        if verdict in ("verified", "supported"):
            strength_cross[(verdict, f.get("verdict_strength"), reward)] += 1

    verdicts = ["verified", "supported", "partial", "unverified", "refuted",
                "unscoreable"]
    print(f"== verdict x reward  ({len(rows)} trials, "
          f"{len(by_task)} tasks, config {args.config}) ==")
    print(f"{'verdict':<12}{'reward=1':>9}{'reward=0':>9}{'reward=?':>9}")
    for v in verdicts:
        r1 = cross.get((v, 1.0), 0)
        r0 = cross.get((v, 0.0), 0)
        rq = sum(n for (vv, rw), n in cross.items()
                 if vv == v and rw is None)
        if r1 or r0 or rq:
            print(f"{v:<12}{r1:>9}{r0:>9}{rq:>9}")

    false_cert = cross.get(("verified", 0.0), 0)
    false_ref = cross.get(("refuted", 1.0), 0)
    print(f"\nfalse certificates (verified & reward=0): {false_cert}"
          f"{'  <-- HARD GATE VIOLATION' if false_cert else ''}")
    print(f"false refutations  (refuted  & reward=1): {false_ref}"
          f"{'  <-- selection poison' if false_ref else ''}")
    for (v, s, rw), n in sorted(strength_cross.items()):
        print(f"  {v}@{s} reward={rw}: {n}")

    # ---- best-of-N selection --------------------------------------------
    complete_tasks = by_task
    if args.expected_n is not None:
        complete_tasks = {
            task: trials for task, trials in by_task.items()
            if len(trials) == args.expected_n
        }
        incomplete = sorted(
            (task, len(trials)) for task, trials in by_task.items()
            if len(trials) != args.expected_n
        )
        print(f"\ncomplete BoN pools: {len(complete_tasks)}/{len(by_task)} "
              f"observed tasks at N={args.expected_n}")
        if incomplete:
            print("incomplete pools excluded: " + ", ".join(
                f"{task}={count}/{args.expected_n}"
                for task, count in incomplete))

    print("\n== best-of-N by rank_key (per complete task) ==")
    print(f"{'task':<28}{'picked':>7}{'mean':>6}{'max':>5}  picked trial "
          "(verdict, score)")
    picked_sum = mean_sum = orac_sum = n_tasks = 0
    hits = winnable = 0
    losses = []
    for task in sorted(complete_tasks):
        trs = complete_tasks[task]
        rewards = [t["reward"] for t in trs if t["reward"] is not None]
        if not rewards:
            continue
        ranked = sorted(trs, key=lambda t: (
            tuple(-c for c in offline.rank_key(t.get("features"))),
            t["trial"]))
        pick = ranked[0]
        pr = pick["reward"] if pick["reward"] is not None else 0.0
        mean = sum(rewards) / len(rewards)
        best = max(rewards)
        picked_sum += pr
        mean_sum += mean
        orac_sum += best
        n_tasks += 1
        if best > 0:
            winnable += 1
            if pr == best:
                hits += 1
            else:
                losses.append((task, pick))
        f = pick.get("features") or {}
        print(f"{task:<28}{pr:>7.2f}{mean:>6.2f}{best:>5.2f}  "
              f"{pick['trial'].split('__')[-1]} "
              f"({f.get('verdict')}, {scalar(f):.4f})")
    print(f"\n{'TOTAL':<28}{picked_sum:>7.2f}{mean_sum:>6.2f}{orac_sum:>5.2f}"
          f"   ({n_tasks} tasks)")
    print(f"selection accuracy: picked {picked_sum:.0f}/{orac_sum:.0f} "
          f"oracle-winnable reward "
          f"(random baseline {mean_sum:.2f}); rank-1 hit "
          f"{hits}/{winnable} winnable tasks")
    if losses:
        print("\nmissed winnable tasks (picked a losing trial over an "
              "available winner):")
        for task, pick in losses:
            f = pick.get("features") or {}
            others = [(t["trial"].split("__")[-1], t["reward"],
                       (t.get("features") or {}).get("verdict"))
                      for t in by_task[task] if t["reward"]]
            print(f"  {task}: picked {pick['trial'].split('__')[-1]} "
                  f"reward={pick['reward']} ({f.get('verdict')}); "
                  f"winners were {others}")


if __name__ == "__main__":
    main()
