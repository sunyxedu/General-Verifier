"""Terminal-Bench — offline ORM scoring of the `rec` arm's trials.

Walks the Harbor job's trial dirs, finds each pulled /gv-state artifact,
fills in final.txt from the agent log (the solver's last words), and runs
the offline extract+certify pipeline on the host. Scores append to
results/rec_scores.jsonl (resume by (job, trial)).

Caveat: the task container is gone by scoring time, so the final world is
NOT reachable — obligations about the live state discharge as failed or
unobtainable (fail-closed → partial), and the signal rests on ledger
obligations. That is inherent to scoring containerized runs post-hoc.

Usage:
  uv run python experiments/terminal_bench/score.py [--job rec] [--model m]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import harborjobs                                        # noqa: E402
from common import results as res                        # noqa: E402
from verifier import offline                             # noqa: E402


def find_state(trial_dir):
    """The pulled /gv-state artifact inside a trial dir."""
    for root, dirs, files in os.walk(trial_dir):
        if os.path.basename(root) == "gv-state" or "hook_feed.jsonl" in files:
            return root
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="rec",
                    help="job name under results/jobs (default: rec)")
    ap.add_argument("--model", default=None,
                    help="verifier seats' model (default: provider default)")
    ap.add_argument("--max-rounds", type=int, default=3)
    args = ap.parse_args()

    job_dir = os.path.join(HERE, "results", "jobs", args.job)
    scores_path = os.path.join(HERE, "results", "rec_scores.jsonl")
    done = res.recorded(scores_path, "job", "trial")
    todo = [t for t in harborjobs.trials(job_dir)
            if (args.job, t["trial"]) not in done]
    print(f"[tb:score] {len(todo)} trials ({len(done)} scored)", flush=True)

    for i, t in enumerate(todo, 1):
        state = find_state(t["trial_dir"])
        if state is None:
            print(f"[tb:score {i}/{len(todo)}] {t['trial']} no gv-state "
                  "artifact — skipped", flush=True)
            continue
        final_path = os.path.join(state, "final.txt")
        if not os.path.exists(final_path):
            with open(final_path, "w") as f:
                f.write(harborjobs.log_tail(t["agent_log"]))
        features = offline.score(state, model=args.model,
                                 max_rounds=args.max_rounds)
        res.append(scores_path, {"job": args.job, "task": t["task"],
                                 "trial": t["trial"], "reward": t["reward"],
                                 "features": features})
        print(f"[tb:score {i}/{len(todo)}] {t['trial']} "
              f"verdict={features.get('verdict')} reward={t['reward']} "
              f"rank={offline.rank_key(features)}", flush=True)
    print("[tb:score] done", flush=True)


if __name__ == "__main__":
    main()
