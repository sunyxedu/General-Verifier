"""Arm C — the OFFICIAL LLM-as-a-Verifier implementation (arXiv 2607.05391),
run as-shipped from the authors' repository, on its native task shape.

Mapping: our (true, trap) pair per problem is exactly a best-of-2 selection —
`select(problem, [claim_true, claim_trap])` with a single correctness
criterion. The official pipeline does the rest: pairwise prompt, analysis,
score tags, expectation over the 20-letter score-token LOGPROB distribution
(the real thing, not our Monte-Carlo stand-in), directed comparisons both
ways so slot bias cancels, K=8 repeated evaluations.

Backend requirement (theirs, not ours): a logprob-capable endpoint.
  export OPENAI_BASE_URL=https://api.openai.com/v1  OPENAI_API_KEY=sk-...
      (or any vLLM/SGLang server)                   -> e.g. --model gpt-4o
  export VERTEX_API_KEY=...                          -> gemini-2.5-flash
Anthropic models cannot run this arm (no logprobs — the paper's App. A).

The vendored clone lives in vendor/llm-as-a-verifier (gitignored):
  git clone https://github.com/llm-as-a-verifier/llm-as-a-verifier \
      experiments/aime2024/vendor/llm-as-a-verifier

Usage:
  uv run python experiments/aime2024/run_official.py [--set aime|novel|aime2026]
      [--model gemini-2.5-flash] [--k 8]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vendor", "llm-as-a-verifier"))

from problems import claim_text                    # noqa: E402
import llm_verifier                                # noqa: E402  (official)

# One correctness criterion, written in the style of the repo's own
# criteria files (criteria/TEMPLATE.md).
CRITERIA = {
    "Final Answer Correctness": (
        "Work the mathematics of the task yourself and determine whether the "
        "final answer stated in each trajectory is exactly correct. Score "
        "HIGH only when the stated answer equals the true answer of the "
        "problem; score LOW when it is wrong, no matter how plausible it "
        "looks. Ignore style, phrasing, and length — only the numerical "
        "correctness of the stated answer matters."
    ),
}
NOTE = ("Each trajectory consists solely of a claimed final answer to the "
        "task. Do not reward confidence; reward mathematical correctness.")


def recorded(results_path: str) -> set:
    done = set()
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["pid"])
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="verifier model (default: gemini-2.5-flash on "
                         "Vertex, or the model your OpenAI endpoint serves)")
    ap.add_argument("--set", dest="pset", default="aime",
                    choices=["aime", "novel", "aime2026"])
    ap.add_argument("--k", type=int, default=8, help="repeated evaluations")
    args = ap.parse_args()

    if args.pset == "novel":
        from problems_novel import PROBLEMS
    elif args.pset == "aime2026":
        from problems_aime2026 import PROBLEMS
    else:
        from problems import PROBLEMS
    suffix = "" if args.pset == "aime" else "-" + args.pset
    results_dir = os.path.join(HERE, "results", "official" + suffix)
    results = os.path.join(results_dir, "results.jsonl")
    os.makedirs(results_dir, exist_ok=True)

    model = args.model or llm_verifier.DEFAULT_MODEL
    done = recorded(results)
    todo = [p for p in PROBLEMS if p["id"] not in done]
    print(f"[official:{args.pset}] {len(todo)} best-of-2 selections, "
          f"K={args.k}, model={model} ({len(done)} already recorded)",
          flush=True)

    for i, p in enumerate(todo, 1):
        cands = [claim_text(p, p["true"]), claim_text(p, p["trap"])]
        t0 = time.time()
        try:
            r = llm_verifier.select(
                p["statement"], cands,
                criteria=CRITERIA, ground_truth_note=NOTE,
                n_evaluations=args.k, model=model,
                cache=os.path.join(results_dir, f"cache_{p['id']}.json"),
                progress=False)
            rec = {
                "pid": p["id"], "picked": r.index,
                "correct": r.index == 0,          # slot 0 holds the truth
                "scores": [round(s, 4) for s in r.scores],
                "n_comparisons": r.n_comparisons,
                "seconds": round(time.time() - t0, 1),
                "model": model, "k": args.k,
            }
        except Exception as err:                   # noqa: BLE001
            rec = {"pid": p["id"], "error": repr(err)[:300],
                   "seconds": round(time.time() - t0, 1), "model": model}
        with open(results, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[official:{args.pset} {i}/{len(todo)}] {p['id']} -> {rec}",
              flush=True)

    with open(results, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    ok = sum(1 for r in recs if r.get("correct"))
    print(f"[official:{args.pset}] selection accuracy {ok}/{len(recs)}",
          flush=True)


if __name__ == "__main__":
    main()
