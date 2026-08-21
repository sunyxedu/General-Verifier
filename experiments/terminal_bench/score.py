"""Terminal-Bench — offline ORM scoring of a recorder arm's trials.

Two trace sources, one pipeline:
  - `rec` (claude-code): the container's recording hooks wrote the ledger
    directly; harbor pulled /gv-state as a trial artifact.
  - `rec-codex`: codex has no hooks, but harbor copies its own rollout log
    to trial_dir/agent/sessions/. verifier.stream.codex rewrites that into
    the same feed format, so both arms score identically.

Each trial is scored independently: its own extraction, its own claim
graph, its own certificate. Scores append to results/rec_scores.jsonl
(resume by (job, trial)).

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
from common import scoring                                # noqa: E402
from verifier.stream import codex as codex_stream        # noqa: E402


def find_state(trial_dir):
    """The pulled /gv-state artifact inside a trial dir."""
    for root, _dirs, files in os.walk(trial_dir):
        if os.path.basename(root) == "gv-state" or "hook_feed.jsonl" in files:
            return root
    return None


def link_world(trial_dir, state):
    """Point <state>/world at the pulled working tree, if the run captured
    one. Without it the container is gone by scoring time and every
    delivered-state obligation can only fail — the ledger would be the
    sole evidence, which proves reporting accuracy, not task success."""
    world = os.path.join(state, "world")
    if os.path.exists(world):
        return
    app = os.path.join(trial_dir, "artifacts", "app")
    if os.path.isdir(app):
        os.symlink(os.path.abspath(app), world)


def codex_state(trial):
    """Build a state dir from a codex trial's own rollout log. The ledger
    is converted from the rollout; task.md comes from the trial config
    (the harness's authoritative instruction), and there is no baseline —
    a vanilla codex container carries no GV payload to snapshot with."""
    sessions = os.path.join(trial["trial_dir"], "agent", "sessions")
    rollout = codex_stream.newest_rollout(sessions) \
        if os.path.isdir(sessions) else None
    if rollout is None:
        return None
    state = os.path.join(trial["trial_dir"], "gv-offline")
    codex_stream.convert(rollout, state)
    instruction = harborjobs.instruction_of(trial["trial_dir"])
    if instruction:
        with open(os.path.join(state, "task.md"), "w") as f:
            f.write(instruction)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="rec",
                    help="job name under results/jobs (default: rec)")
    ap.add_argument("--model", default=None,
                    help="verifier seats' model (default: provider default)")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4,
                    help="traces scored concurrently (they share nothing)")
    args = ap.parse_args()

    job_dir = os.path.join(HERE, "results", "jobs", args.job)
    scores_path = os.path.join(HERE, "results", "rec_scores.jsonl")
    done = res.recorded(scores_path, "job", "trial")
    todo = [t for t in harborjobs.trials(job_dir)
            if (args.job, t["trial"]) not in done]
    print(f"[tb:score] {len(todo)} trials ({len(done)} scored)", flush=True)

    jobs = []
    for t in todo:
        state = find_state(t["trial_dir"]) or codex_state(t)
        if state is None:
            print(f"[tb:score] {t['trial']} no ledger (neither gv-state nor "
                  "a codex rollout) — skipped", flush=True)
            continue
        link_world(t["trial_dir"], state)
        final_path = os.path.join(state, "final.txt")
        if not os.path.exists(final_path) or not open(final_path).read().strip():
            with open(final_path, "w") as f:
                f.write(harborjobs.log_tail(t["agent_log"]))
        jobs.append(({"job": args.job, "task": t["task"], "trial": t["trial"],
                      "reward": t["reward"]}, state))

    scoring.score_many(jobs, scores_path, workers=args.workers,
                       model=args.model, max_rounds=args.max_rounds,
                       label=lambda r: f"{r['trial']} reward={r['reward']}",
                       tag="tb:score")
    print("[tb:score] done", flush=True)


if __name__ == "__main__":
    main()
