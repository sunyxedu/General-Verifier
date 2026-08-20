"""FrontierMath — offline ORM scoring of the `rec` arm's traces.

Each (pid, sample) trace is scored independently: its own extraction, its
own claim graph, its own certificate — no shared state, no memory. Scores
land in results/rec/scores.jsonl (resume by (pid, sample)); selection is
computed at summarize time from results.jsonl x scores.jsonl.

Usage:
  uv run python experiments/frontiermath/score.py [--pid P] [--model m]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import results as res                        # noqa: E402
from common import scoring                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="verifier seats' model (default: provider default)")
    ap.add_argument("--pid", default=None, help="score only this problem id")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4,
                    help="traces scored concurrently (they share nothing)")
    args = ap.parse_args()

    results_path = os.path.join(HERE, "results", "rec", "results.jsonl")
    scores_path = os.path.join(HERE, "results", "rec", "scores.jsonl")
    rows = res.load(results_path)
    done = res.recorded(scores_path, "pid", "sample")
    todo = [r for r in rows if (r["pid"], r["sample"]) not in done]
    if args.pid:
        todo = [r for r in todo if r["pid"] == args.pid]
    print(f"[fm:score] {len(todo)} traces ({len(done)} scored)", flush=True)

    jobs = [({"pid": r["pid"], "sample": r["sample"]},
             os.path.join(r["workdir"], ".gv")) for r in todo]
    scoring.score_many(jobs, scores_path, workers=args.workers,
                       model=args.model, max_rounds=args.max_rounds,
                       label=lambda r: f"{r['pid']}#{r['sample']}",
                       tag="fm:score")
    print("[fm:score] done", flush=True)


if __name__ == "__main__":
    main()
