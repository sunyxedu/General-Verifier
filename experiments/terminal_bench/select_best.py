"""Terminal-Bench — LLM-as-a-Verifier arm: best-of-N trial selection.

Terminal-Bench solutions are container states, not portable artifacts, so
the paper-style judge selects among the plain arm's K independent trials by
reading each trial's transcript (text only, K repeated 0-10 scorings).
The picked trial's Harbor reward is the arm's score for the task; pass@1
(mean reward) and oracle pass@N (max reward) come from the same trials.

Usage:  uv run python experiments/terminal_bench/select.py [--k 4]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from common import judge as J                            # noqa: E402
from common import results as res                        # noqa: E402
import harborjobs                                        # noqa: E402

JUDGE = J.make_judge(
    "You are a verifier of terminal task executions. You judge, from an "
    "agent's transcript, whether the agent fully and correctly completed "
    "the task it was given.")

CRITERION = ("based on this transcript, did the agent fully and correctly "
             "complete the task (not merely claim to)?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--job", default="plain")
    args = ap.parse_args()

    job_dir = os.path.join(HERE, "results", "jobs", args.job)
    by_task = {}
    for t in harborjobs.trials(job_dir):
        by_task.setdefault(t["task"], []).append(t)

    sel_path = os.path.join(HERE, "results", "judge", "selection.jsonl")
    done = res.recorded(sel_path, "task")
    backend = J.make_backend(JUDGE, os.path.join(HERE, "results", "judge"),
                             args.model)

    todo = [task for task in sorted(by_task) if (task,) not in done]
    print(f"[tb:select] {len(todo)} tasks, K={args.k} "
          f"({len(done)} recorded)", flush=True)
    for n, task in enumerate(todo, 1):
        ts = by_task[task]
        instruction = harborjobs.instruction_of(ts[0]["trial_dir"]) \
            or f"Terminal-Bench task '{task}'"
        scored = []
        for t in ts:
            tail = harborjobs.log_tail(t["agent_log"])
            if not tail.strip():
                scored.append({"trial": t["trial"], "mean": -1.0,
                               "reward": t["reward"]})
                continue
            mean, scores, failures = J.score(
                JUDGE, backend, instruction, tail, CRITERION, k=args.k)
            scored.append({"trial": t["trial"],
                           "mean": -1.0 if mean is None else mean,
                           "scores": scores, "failures": failures,
                           "reward": t["reward"]})
        pick = max(scored, key=lambda s: s["mean"])
        res.append(sel_path, {"task": task, "pick": pick["trial"],
                              "picked_reward": pick["reward"],
                              "scored": scored, "k": args.k,
                              "model": args.model})
        print(f"[tb:select {n}/{len(todo)}] {task} -> {pick['trial']} "
              f"reward={pick['reward']}", flush=True)

    print("[tb:select] done", flush=True)


if __name__ == "__main__":
    main()
