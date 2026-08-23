"""FrontierMath — offline ORM scoring of a recorder arm's traces.

Two trace sources, one pipeline (same shape as terminal_bench):
  --arm rec    Claude Code traces recorded by the hooks
  --arm codex  Codex CLI traces whose rollout run.py already converted
               into the same hook-feed layout at run time

Each (pid, sample) trace is scored independently: its own extraction, its
own claim graph, its own certificate — no shared state, no memory. Scores
land in results/<arm>/scores.jsonl (resume by (pid, sample, config));
selection is computed at summarize time from results.jsonl x scores.jsonl.

Usage:
  uv run python experiments/frontiermath/score.py [--arm rec|codex]
      [--pid P] [--model m] [--seats codex] [--effort e]
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
    ap.add_argument("--arm", default="rec", choices=["rec", "codex"],
                    help="which solver arm's traces to score (default rec)")
    ap.add_argument("--model", default=None,
                    help="verifier seats' model (default: provider default)")
    ap.add_argument("--pid", default=None, help="score only this problem id")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--critic-rounds", type=int, default=None,
                    help="decomposition-game rounds before a claim fails "
                    "closed (default: the kernel's MAX_CRITIC_ROUNDS)")
    ap.add_argument("--workers", type=int, default=4,
                    help="traces scored concurrently (they share nothing)")
    ap.add_argument("--seats", default=None, choices=["codex"],
                    help="codex: run every verifier seat through the Codex "
                    "CLI on the ChatGPT subscription (default model "
                    "gpt-5.6-luna) instead of the SDK/provider path")
    ap.add_argument("--effort", default=None,
                    choices=["low", "medium", "high", "xhigh", "max"],
                    help="codex seats' reasoning effort (default: medium)")
    args = ap.parse_args()

    results_path = os.path.join(HERE, "results", args.arm, "results.jsonl")
    scores_path = os.path.join(HERE, "results", args.arm, "scores.jsonl")
    rows = res.load(results_path)
    # resume keyed on the full config identity (verifier code+prompts
    # fingerprint plus model/max_rounds): a record from a different config
    # is a different verifier's judgment
    cfg = offline.config_fields(model=args.model, seats=args.seats,
                                effort=args.effort,
                                max_rounds=args.max_rounds,
                                critic_rounds=args.critic_rounds)
    done = res.recorded(scores_path, "pid", "sample", "config")
    todo = [r for r in rows
            if (r["pid"], r["sample"], cfg["config"]) not in done]
    if args.pid:
        todo = [r for r in todo if r["pid"] == args.pid]
    print(f"[fm:score] arm={args.arm} {len(todo)} traces "
          f"({len(done)} scored) config={cfg['config']}", flush=True)

    jobs = [({"pid": r["pid"], "sample": r["sample"], **cfg},
             os.path.join(r["workdir"], ".gv")) for r in todo]
    scoring.score_many(jobs, scores_path, workers=args.workers,
                       model=args.model, max_rounds=args.max_rounds,
                       seats=args.seats, effort=args.effort,
                       critic_rounds=args.critic_rounds,
                       label=lambda r: f"{r['pid']}#{r['sample']}",
                       tag="fm:score")
    print("[fm:score] done", flush=True)


if __name__ == "__main__":
    main()
