"""Arm B — LLM-as-a-Verifier-style baseline (arXiv 2607.05391), ported to
single-claim verification.

Faithful to the paper where possible, honest about deviations:
- Continuous scores: the paper takes the expectation over scoring-token
  logits. Anthropic models do not expose logits (the paper's own Appendix A
  lists this exclusion), so we use its other scaling axis — repeated
  evaluation — as the Monte-Carlo estimate of the same expectation: K
  independent single-shot judgments, mean score. Granularity G=11 (0..10).
- The judge reads the SAME claim text the General Verifier sees, plus an
  explicit correctness criterion. No tools, no execution — judging text is
  the method under test.
- Same model as the General Verifier's seats, so the comparison is
  evidence-vs-opinion at matched model capability, not model shopping.

Usage:
  uv run python experiments/aime2024/run_judge.py [--model sonnet]
      [--set aime|novel] [--k 8]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from problems import claim_text                          # noqa: E402
from verifier.llm.agents import AgentBackend, AgentRole  # noqa: E402
from verifier.llm.roles import RoleError                 # noqa: E402

JUDGE = AgentRole(
    name="llm_judge",
    system=(
        "You are a mathematical verifier. You judge whether the final answer "
        "stated in a claim about a competition problem is exactly correct. "
        "Reason carefully, then give a calibrated score."
    ),
    template=(
        "Claim to verify: {claim}\n\n"
        "Criterion: is the stated final answer exactly correct for this "
        "problem?\n"
        "Score the claim on an integer scale from 0 (certainly wrong) to 10 "
        "(certainly correct). Output JSON: {{\"score\": s}}"
    ),
    schema={"score": int},
    tools=[],          # the method under test reads; it does not execute
    max_turns=1,
)


def recorded(results_path: str) -> set:
    done = set()
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["pid"], r["tag"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--set", dest="pset", default="aime",
                    choices=["aime", "novel"])
    ap.add_argument("--k", type=int, default=8, help="repeated evaluations")
    args = ap.parse_args()

    if args.pset == "novel":
        from problems_novel import PROBLEMS
    else:
        from problems import PROBLEMS
    suffix = "" if args.pset == "aime" else "-novel"
    results_dir = os.path.join(HERE, "results", "judge" + suffix)
    results = os.path.join(results_dir, "results.jsonl")

    os.makedirs(results_dir, exist_ok=True)
    backend = AgentBackend(roles={"llm_judge": JUDGE}, cwd=results_dir,
                           model=args.model)
    done = recorded(results)
    todo = [(p, tag) for p in PROBLEMS for tag in ("true", "trap")
            if (p["id"], tag) not in done]
    print(f"[judge:{args.pset}] {len(todo)} claims x K={args.k} repeats "
          f"({len(done)} already recorded)", flush=True)

    for i, (p, tag) in enumerate(todo, 1):
        answer = p[tag]
        scores, failures = [], 0
        t0 = time.time()
        for _ in range(args.k):
            try:
                payload = JUDGE.call(backend,
                                     {"claim": claim_text(p, answer)})
                scores.append(max(0, min(10, int(payload["score"]))))
            except RoleError:
                failures += 1
        seconds = round(time.time() - t0, 1)
        mean = round(sum(scores) / len(scores), 3) if scores else None
        rec = {
            "pid": p["id"], "tag": tag, "answer": answer,
            "label": "correct" if tag == "true" else "wrong",
            "scores": scores, "mean": mean, "failures": failures,
            "seconds": seconds, "k": args.k, "model": args.model,
        }
        with open(results, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[judge:{args.pset} {i}/{len(todo)}] {p['id']}_{tag} "
              f"answer={answer} -> mean={mean} scores={scores}  ({seconds}s)",
              flush=True)

    print(f"[judge:{args.pset}] done", flush=True)


if __name__ == "__main__":
    main()
