# /// script
# requires-python = ">=3.11"
# dependencies = ["google-genai", "openai", "tqdm"]
# ///
"""Terminal-Bench — judge arm: the OFFICIAL LLM-as-a-Verifier, as shipped.

Best-of-N trial selection over the plain arm's K Harbor trials with the
authors' pipeline (arXiv 2607.05391), on its NATIVE task shape: the repo's
bundled `terminal_bench` criteria, and traces built by the repo's own
Terminal-Bench trajectory formatter from each trial's ATIF trajectory.json
(which Harbor's claude-code agent emits). The picked trial's Harbor reward
is the arm's score.

Backend: logprob-capable endpoint — VERTEX_API_KEY (gemini-2.5-flash,
default) or OPENAI_BASE_URL/OPENAI_API_KEY (vLLM etc.). Anthropic models
cannot serve this arm (no logprobs; the paper's App. A).

Usage:  uv run experiments/terminal_bench/select_best.py [--model m] [--k 8]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from common import official_verifier                     # noqa: E402
from common import results as res                        # noqa: E402
import harborjobs                                        # noqa: E402


def trial_trace(trial, loaders):
    """(problem, trace) via the official repo's own ATIF formatter."""
    traj_path = os.path.join(trial["trial_dir"], "agent", "trajectory.json")
    if os.path.exists(traj_path):
        try:
            with open(traj_path) as f:
                trajectory = json.load(f)
            steps = trajectory.get("steps", [])
            if steps:
                return (loaders._tb_extract_problem(steps, trial["task"]),
                        loaders._tb_format_trace(trajectory))
        except (json.JSONDecodeError, TypeError):
            pass
    # fallback: raw agent log tail (still their pipeline, degraded trace)
    return (harborjobs.instruction_of(trial["trial_dir"])
            or f"(Task: {trial['task']})",
            harborjobs.log_tail(trial["agent_log"]) or "(no trajectory data)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="verifier model (default: the package's own)")
    ap.add_argument("--k", type=int, default=8, help="n_evaluations")
    ap.add_argument("--pivots", type=int, default=2)
    ap.add_argument("--job", default="plain")
    args = ap.parse_args()

    lv = official_verifier.load()
    from llm_verifier import loaders                     # noqa: E402
    model = args.model or lv.DEFAULT_MODEL

    job_dir = os.path.join(HERE, "results", "jobs", args.job)
    by_task = {}
    for t in harborjobs.trials(job_dir):
        by_task.setdefault(t["task"], []).append(t)

    out_dir = os.path.join(HERE, "results", "judge")
    os.makedirs(out_dir, exist_ok=True)
    sel_path = os.path.join(out_dir, "selection.jsonl")
    done = res.recorded(sel_path, "task")

    todo = [task for task in sorted(by_task) if (task,) not in done]
    print(f"[tb:official] {len(todo)} selections, K={args.k}, "
          f"model={model} ({len(done)} recorded)", flush=True)
    for n, task in enumerate(todo, 1):
        ts = by_task[task]
        pairs = [trial_trace(t, loaders) for t in ts]
        problem = next((p for p, _ in pairs if p), f"(Task: {task})")
        t0 = time.time()
        try:
            r = lv.select(problem, [trace for _, trace in pairs],
                          criteria="terminal_bench", n_evaluations=args.k,
                          pivots=args.pivots, model=model,
                          cache=os.path.join(out_dir, f"cache_{task}.json"),
                          progress=False)
            picked = ts[r.index]
            rec = {"task": task, "pick": picked["trial"],
                   "picked_reward": picked["reward"],
                   "rewards": [t["reward"] for t in ts],
                   "scores": [round(s, 4) for s in r.scores],
                   "ranking": r.ranking, "n_comparisons": r.n_comparisons,
                   "seconds": round(time.time() - t0, 1),
                   "model": model, "k": args.k}
        except Exception as err:                          # noqa: BLE001
            rec = {"task": task, "error": repr(err)[:300],
                   "seconds": round(time.time() - t0, 1), "model": model}
        res.append(sel_path, rec)
        print(f"[tb:official {n}/{len(todo)}] {task} -> "
              f"{rec.get('pick', 'ERROR')} reward={rec.get('picked_reward')}",
              flush=True)

    print("[tb:official] done", flush=True)


if __name__ == "__main__":
    main()
