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
from common import scoring                                # noqa: E402
from verifier import offline                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="verifier seats' model (default: provider default)")
    ap.add_argument("--iid", default=None, help="score only this instance id")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4,
                    help="traces scored concurrently (they share nothing)")
    args = ap.parse_args()

    results_path = os.path.join(HERE, "results", "rec", "results.jsonl")
    scores_path = os.path.join(HERE, "results", "rec", "scores.jsonl")
    rows = res.load(results_path)
    # resume keyed on the pipeline fingerprint: a record from different
    # prompts/schemas is a different verifier's judgment
    fingerprint = offline.pipeline_fingerprint()
    done = res.recorded(scores_path, "instance_id", "sample", "fingerprint")
    todo = [r for r in rows
            if (r["instance_id"], r["sample"], fingerprint) not in done]
    if args.iid:
        todo = [r for r in todo if r["instance_id"] == args.iid]
    print(f"[swebench:score] {len(todo)} traces ({len(done)} scored) "
          f"fingerprint={fingerprint}", flush=True)

    jobs = [({"instance_id": r["instance_id"], "sample": r["sample"],
              "fingerprint": fingerprint},
             os.path.join(r["workdir"], ".gv")) for r in todo]
    scoring.score_many(jobs, scores_path, workers=args.workers,
                       model=args.model, max_rounds=args.max_rounds,
                       label=lambda r: f"{r['instance_id']}#{r['sample']}",
                       tag="swebench:score")
    print("[swebench:score] done", flush=True)


if __name__ == "__main__":
    main()
