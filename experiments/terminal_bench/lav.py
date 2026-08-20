"""Baseline: LLM-as-a-Verifier best-of-N on the same traces GV scores.

The comparison this exists for: both verifiers see the SAME candidate
trajectories (the recorder arm's k trials per task) and pick one. GV
reads the ledger, extracts a claim graph and discharges obligations
mechanically; LLM-as-a-Verifier reads a text rendering of that same
ledger and scores it with its Probabilistic Pivot Tournament. Ground
truth is Harbor's own reward.

Model: whatever the OpenAI-compatible endpoint in OPENAI_BASE_URL serves.
For a ChatGPT-subscription model, start experiments/common/codex_endpoint.py
first and point OPENAI_BASE_URL at it — but note that path returns NO
logprobs, so LLM-as-a-Verifier degrades to parsing the score letter out
of the text (its documented fallback, coarser than the logprob
expectation it is designed around). The printed report says which mode
ran; do not compare a degraded run against a logprob run.

Usage (installs the baseline into a throwaway env):
  uv run --no-project --with git+https://github.com/llm-as-a-verifier/llm-as-a-verifier \
      python experiments/terminal_bench/lav.py --job rec-codex-5
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root
sys.path.insert(0, os.path.dirname(HERE))                    # experiments/
sys.path.insert(0, HERE)

import harborjobs                                        # noqa: E402
from common import results as res                        # noqa: E402
from verifier.stream import codex as codex_stream        # noqa: E402

TRACE_CHARS = 60000        # per-candidate cap; tournaments send pairs


def render_trace(state_dir):
    """The ledger as text, in LLM-as-a-Verifier's Terminal-Bench trace
    shape ([Command] / [Output] per step). Same evidence GV verifies,
    rendered for a text-only verifier."""
    feed = os.path.join(state_dir, "hook_feed.jsonl")
    if not os.path.exists(feed):
        return ""
    calls, parts, step = {}, [], 0
    with open(feed, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            p = event["payload"]
            if event["type"] == "tool_call":
                calls[p.get("tool_use_id")] = p.get("input")
            else:
                step += 1
                command = calls.pop(p.get("tool_use_id"), None)
                parts.append(f"--- Agent Step {step} ---")
                if command is not None:
                    text = command if isinstance(command, str) else \
                        (command.get("cmd") or command.get("command")
                         or json.dumps(command, ensure_ascii=False))
                    parts.append(f"[Command] {str(text).rstrip()}")
                parts.append(f"[Output]\n{p.get('excerpt', '')}")
                parts.append("")
    final = os.path.join(state_dir, "final.txt")
    if os.path.exists(final):
        with open(final, encoding="utf-8") as f:
            parts.append("--- Agent Final Message ---\n" + f.read())
    return "\n".join(parts)[-TRACE_CHARS:]


def candidates_for(job_dir):
    """task -> [{trial, reward, trace}] from the recorder arm's trials."""
    import score as score_mod           # reuses the same state-dir logic

    tasks = {}
    for trial in harborjobs.trials(job_dir):
        state = score_mod.find_state(trial["trial_dir"]) \
            or score_mod.codex_state(trial)
        if state is None:
            continue
        trace = render_trace(state)
        if not trace.strip():
            continue
        tasks.setdefault(trial["task"], []).append({
            "trial": trial["trial"], "reward": trial["reward"],
            "trace": trace,
            "problem": harborjobs.instruction_of(trial["trial_dir"]) or
            trial["task"],
        })
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="rec-codex-5")
    ap.add_argument("--pivots", type=int, default=1)
    ap.add_argument("--n-evaluations", type=int, default=2)
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--model", default=None,
                    help="model id to request (default: resolved from the "
                    "endpoint's /v1/models)")
    args = ap.parse_args()

    import llm_verifier

    base_url = os.environ.get("OPENAI_BASE_URL")
    degraded = bool(base_url)      # our codex endpoint serves no logprobs
    print(f"[lav] endpoint={base_url or 'default provider'} "
          f"model={args.model or 'auto'} "
          f"scoring={'TEXT-FALLBACK (no logprobs)' if degraded else 'logprobs'}",
          flush=True)

    job_dir = os.path.join(HERE, "results", "jobs", args.job)
    tasks = candidates_for(job_dir)
    if not tasks:
        sys.exit(f"no usable trials under {job_dir}")

    out_path = os.path.join(HERE, "results", f"lav_{args.job}.jsonl")
    done = res.recorded(out_path, "task")
    print(f"[lav] {len(tasks)} tasks ({len(done)} already done)", flush=True)

    for task, trials in sorted(tasks.items()):
        if (task,) in done:
            continue
        if len(trials) < 2:
            print(f"[lav] {task}: only {len(trials)} candidate — skipped",
                  flush=True)
            continue
        kwargs = dict(criteria="terminal_bench", pivots=args.pivots,
                      n_evaluations=args.n_evaluations,
                      max_workers=args.max_workers)
        if args.model:
            kwargs["model"] = args.model
        result = llm_verifier.select(
            problem=trials[0]["problem"],
            candidates=[t["trace"] for t in trials], **kwargs)
        picked = trials[result.index]
        rec = {"task": task, "job": args.job,
               "picked_index": result.index, "picked_trial": picked["trial"],
               "picked_reward": picked["reward"],
               "scores": list(result.scores),
               "n_comparisons": result.n_comparisons,
               "rewards": [t["reward"] for t in trials],
               "degraded_scoring": degraded}
        res.append(out_path, rec)
        print(f"[lav] {task}: picked #{result.index} "
              f"reward={picked['reward']} of {[t['reward'] for t in trials]} "
              f"({result.n_comparisons} comparisons)", flush=True)

    rows = res.load(out_path)
    if rows:
        ok = sum(1 for r in rows if r["picked_reward"])
        oracle = sum(1 for r in rows if any(r["rewards"]))
        mean = sum(sum(r["rewards"]) / len(r["rewards"]) for r in rows)
        print(f"\n[lav] pass@1 {mean / len(rows):.2f} · "
              f"LAV selection {ok}/{len(rows)} · oracle {oracle}/{len(rows)}"
              + ("  [degraded: no logprobs]" if degraded else ""))


if __name__ == "__main__":
    main()
