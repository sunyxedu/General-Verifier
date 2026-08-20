"""SWE-bench Pro — offline ORM scoring of the `rec` arm's traces.

Each (instance_id, sample) trace is scored independently against its own
final workdir + ledger. Selection over samples happens at summarize time;
the SELECTED patches then go through the official Docker evaluation
(evaluate.py) under their own tag.

Usage:
  uv run python experiments/swebench_pro/score.py [--iid I] [--model m]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import results as res                        # noqa: E402
from verifier import offline                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="verifier seats' model (default: provider default)")
    ap.add_argument("--iid", default=None, help="score only this instance id")
    ap.add_argument("--max-rounds", type=int, default=3)
    args = ap.parse_args()

    results_path = os.path.join(HERE, "results", "rec", "results.jsonl")
    scores_path = os.path.join(HERE, "results", "rec", "scores.jsonl")
    rows = res.load(results_path)
    done = res.recorded(scores_path, "instance_id", "sample")
    todo = [r for r in rows if (r["instance_id"], r["sample"]) not in done]
    if args.iid:
        todo = [r for r in todo if r["instance_id"] == args.iid]
    print(f"[swebench:score] {len(todo)} traces ({len(done)} scored)",
          flush=True)

    for i, r in enumerate(todo, 1):
        state = os.path.join(r["workdir"], ".gv")
        features = offline.score(state, model=args.model,
                                 max_rounds=args.max_rounds)
        res.append(scores_path, {"instance_id": r["instance_id"],
                                 "sample": r["sample"], "features": features})
        print(f"[swebench:score {i}/{len(todo)}] "
              f"{r['instance_id']}#{r['sample']} "
              f"verdict={features.get('verdict')} "
              f"rank={offline.rank_key(features)}", flush=True)
    print("[swebench:score] done", flush=True)


if __name__ == "__main__":
    main()
