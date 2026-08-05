"""FrontierMath — LLM-as-a-Verifier arm: best-of-N answer selection.

The paper-style judge (K repeated 0-10 scorings, text only) scores each
plain-arm sample's final answer against the problem; argmax is the pick.
Selection accuracy is then mechanical (official answers).

Usage:  uv run python experiments/frontiermath/select.py [--k 4]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import judge as J                            # noqa: E402
from common import results as res                        # noqa: E402
from check import matches                                # noqa: E402

JUDGE = J.make_judge(
    "You are a mathematical verifier. You judge whether the final answer "
    "stated for a research-level mathematics problem is exactly correct.")

CRITERION = "is the stated final answer exactly correct for this problem?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--model", default="sonnet")
    args = ap.parse_args()

    with open(os.path.join(HERE, "problems.json")) as f:
        problems = {p["id"]: p for p in json.load(f)}
    plain = res.load(os.path.join(HERE, "results", "plain", "results.jsonl"))
    by_pid = {}
    for r in plain:
        by_pid.setdefault(r["pid"], []).append(r)

    sel_path = os.path.join(HERE, "results", "judge", "selection.jsonl")
    done = res.recorded(sel_path, "pid")
    backend = J.make_backend(JUDGE, os.path.join(HERE, "results", "judge"),
                             args.model)

    todo = [pid for pid in by_pid if (pid,) not in done]
    print(f"[fm:select] {len(todo)} problems, K={args.k} "
          f"({len(done)} recorded)", flush=True)
    for n, pid in enumerate(todo, 1):
        p = problems[pid]
        scored = []
        for r in sorted(by_pid[pid], key=lambda r: r["sample"]):
            if r["answer"] is None:
                scored.append({"sample": r["sample"], "answer": None,
                               "mean": -1.0})
                continue
            cand = f"The final answer is: {r['answer']}"
            mean, scores, failures = J.score(
                JUDGE, backend, p["statement"], cand, CRITERION, k=args.k)
            scored.append({"sample": r["sample"], "answer": r["answer"],
                           "mean": -1.0 if mean is None else mean,
                           "scores": scores, "failures": failures})
        best = max(scored, key=lambda s: s["mean"])
        correct = matches(best["answer"], p)
        res.append(sel_path, {"pid": pid, "pick": best["sample"],
                              "answer": best["answer"], "correct": correct,
                              "scored": scored, "k": args.k,
                              "model": args.model})
        print(f"[fm:select {n}/{len(todo)}] {pid} -> sample {best['sample']} "
              f"correct={correct}", flush=True)

    print("[fm:select] done", flush=True)


if __name__ == "__main__":
    main()
