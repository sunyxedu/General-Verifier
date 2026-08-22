"""Terminal-Bench — offline ORM scoring of a recorder arm's trials.

Two trace sources, one pipeline:
  - `rec` (claude-code): the container's recording hooks wrote the ledger
    directly; harbor pulled /gv-state as a trial artifact.
  - `rec-codex`: codex has no hooks, but harbor copies its own rollout log
    to trial_dir/agent/sessions/. verifier.stream.codex rewrites that into
    the same feed format, so both arms score identically.

Each trial is scored independently: its own extraction, its own claim
graph, its own certificate. Scores append to results/rec_scores.jsonl
(resume by (job, trial)). Progress goes one task at a time; as each
task finishes, its trials are ranked by rank_key (best-of-N selection
is per-task) and emitted as TSV rows on stdout — diagnostics go to
stderr, so stdout pipes through `tee` clean. A rerun re-emits cached
rows too, so the TSV is always the complete table.

Caveat: the task container is gone by scoring time, so the final world is
NOT reachable — obligations about the live state discharge as failed or
unobtainable (fail-closed → partial), and the signal rests on ledger
obligations. That is inherent to scoring containerized runs post-hoc.

Usage:
  uv run python experiments/terminal_bench/score.py [--job rec] [--model m]
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import harborjobs                                        # noqa: E402
from common import results as res                        # noqa: E402
from common import scoring                                # noqa: E402
from verifier import offline                             # noqa: E402
from verifier.stream import codex as codex_stream        # noqa: E402


def find_state(trial_dir):
    """The pulled /gv-state artifact inside a trial dir."""
    for root, _dirs, files in os.walk(trial_dir):
        if os.path.basename(root) == "gv-state" or "hook_feed.jsonl" in files:
            return root
    return None


def link_world(trial_dir, state):
    """Build <state>/world as a mirror of the run's filesystem ROOT: the
    pulled /app artifact lands at <state>/world/app, so the run's own
    absolute paths resolve inside the tree exactly as the obligation
    compiler is told they do (/app/x lives at <world>/app/x). Without it
    the container is gone by scoring time and every delivered-state
    obligation can only fail — the ledger would be the sole evidence,
    which proves reporting accuracy, not task success.

    The artifact is COPIED, not symlinked: find/os.walk/glob do not
    follow directory symlinks, so a symlinked mount silently hides the
    whole tree from search-based checks — an executor that 'searched
    the tree' and found nothing would refute honest deliverables."""
    world = os.path.join(state, "world")
    if os.path.islink(world):
        os.remove(world)   # oldest layout: world -> artifacts/app directly
    mount = os.path.join(world, "app")
    if os.path.islink(mount):
        os.remove(mount)   # old layout: world/app symlink — upgrade to copy
    app = os.path.join(trial_dir, "artifacts", "app")
    if os.path.isdir(app) and not os.path.exists(mount):
        os.makedirs(world, exist_ok=True)
        shutil.copytree(app, mount, symlinks=True)


def codex_state(trial):
    """Build a state dir from a codex trial's own rollout log. The ledger
    is converted from the rollout; task.md comes from the trial config
    when the harbor version records it there, else from the rollout's
    first user message (the instruction the run was actually given —
    harbor 0.20 configs carry only a task ref, no instruction text).
    There is no baseline — a vanilla codex container carries no GV
    payload to snapshot with."""
    sessions = os.path.join(trial["trial_dir"], "agent", "sessions")
    rollout = codex_stream.newest_rollout(sessions) \
        if os.path.isdir(sessions) else None
    if rollout is None:
        return None
    state = os.path.join(trial["trial_dir"], "gv-offline")
    codex_stream.convert(rollout, state)
    instruction = harborjobs.instruction_of(trial["trial_dir"]) \
        or codex_stream.task_instruction(rollout)
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
    ap.add_argument("--task", default=None, nargs="+",
                    help="score only these tasks (verification is the "
                    "expensive stage; spend it where selection differs)")
    ap.add_argument("--seats", default=None, choices=["codex"],
                    help="codex: run every verifier seat through the Codex "
                    "CLI on the ChatGPT subscription (default model "
                    "gpt-5.6-luna) instead of the SDK/provider path")
    args = ap.parse_args()

    job_dir = os.path.join(HERE, "results", "jobs", args.job)
    scores_path = os.path.join(HERE, "results", "rec_scores.jsonl")
    done = res.recorded(scores_path, "job", "trial")
    # cached feature records, so already-scored trials still rank in the TSV
    scored = {r["trial"]: r for r in res.load(scores_path)
              if r.get("job") == args.job}

    by_task = {}
    for t in harborjobs.trials(job_dir):
        if not args.task or t["task"] in args.task:
            by_task.setdefault(t["task"], []).append(t)
    total = sum(len(ts) for ts in by_task.values())
    todo = sum(1 for ts in by_task.values()
               for t in ts if (args.job, t["trial"]) not in done)
    print(f"[tb:score] backend={args.seats or 'sdk'} "
          f"model={args.model or 'provider-default'} "
          f"scoring=claim-graph-certificate max_rounds={args.max_rounds} "
          f"workers={args.workers} job={args.job}",
          file=sys.stderr, flush=True)
    print(f"[tb:score] loaded {total} trials in {len(by_task)} tasks; "
          f"{todo} to score ({len(scored)} cached in "
          f"{os.path.relpath(scores_path, HERE)})",
          file=sys.stderr, flush=True)

    # TSV to stdout (tee-able), diagnostics to stderr — one task at a
    # time, its rows emitted as soon as the task finishes. Traces are
    # still scored independently; the grouping is for selection, which
    # is per-task best-of-N.
    print("task\ttrial\ttb_reward\tgv_verdict\t"
          "gv_verified_frac\tgv_verified_obligations\trank", flush=True)
    for task in sorted(by_task):
        trials = sorted(by_task[task], key=lambda t: t["trial"])
        jobs = []
        for t in trials:
            if (args.job, t["trial"]) in done:
                continue
            state = find_state(t["trial_dir"]) or codex_state(t)
            if state is None:
                print(f"[tb:score] {t['trial']} no ledger (neither gv-state "
                      "nor a codex rollout) — skipped",
                      file=sys.stderr, flush=True)
                continue
            link_world(t["trial_dir"], state)
            final_path = os.path.join(state, "final.txt")
            if not os.path.exists(final_path) \
                    or not open(final_path).read().strip():
                with open(final_path, "w") as f:
                    f.write(harborjobs.log_tail(t["agent_log"]))
            jobs.append(({"job": args.job, "task": task, "trial": t["trial"],
                          "reward": t["reward"]}, state))
        if jobs:
            print(f"[tb:score] scoring {task} ({len(jobs)}/{len(trials)} "
                  "traces)", file=sys.stderr, flush=True)
            for rec in scoring.score_many(
                    jobs, scores_path, workers=args.workers,
                    model=args.model, max_rounds=args.max_rounds,
                    seats=args.seats,
                    label=lambda r: f"{r['trial']} reward={r['reward']}",
                    tag="tb:score"):
                scored[rec["trial"]] = rec

        rows = [(t, scored[t["trial"]]) for t in trials
                if t["trial"] in scored]
        ranked = sorted(rows, key=lambda tr: (
            tuple(-c for c in offline.rank_key(tr[1].get("features"))),
            tr[0]["trial"]))
        rank = {t["trial"]: i + 1 for i, (t, _r) in enumerate(ranked)}
        for t, r in rows:
            features = r.get("features") or {}
            key = offline.rank_key(features)
            reward = "?" if t["reward"] is None else f"{t['reward']:.0f}"
            print(f"{task}\t{t['trial']}\t{reward}\t"
                  f"{features.get('verdict') or 'unscoreable'}\t"
                  f"{key[1]:.6f}\t{key[2]}\t{rank[t['trial']]}", flush=True)
    print("[tb:score] done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
