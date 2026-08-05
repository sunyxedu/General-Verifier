"""SWE-bench Pro — LLM-as-a-Verifier arm: best-of-N patch selection.

The paper-style judge (K repeated 0-10 scorings, text only, no execution)
reads each plain-arm candidate patch against the issue and picks the
highest-scoring one per instance. The pick is written to selection.jsonl;
evaluate.py --arm judge evaluates exactly the picked patches, so the
selection layer is measured end-to-end against pass@1 and oracle pass@N.

Usage:
  uv run python experiments/swebench_pro/select.py [--k 4] [--model sonnet]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import judge as J                            # noqa: E402
from common import results as res                        # noqa: E402

PATCH_CHAR_CAP = 30000     # keep the judge prompt bounded; cap is logged

JUDGE = J.make_judge(
    "You are a software patch verifier. You judge whether a candidate "
    "patch (a unified diff) correctly and completely fixes the reported "
    "issue without breaking existing behavior.")

CRITERION = ("does this patch correctly and completely fix the issue "
             "(and satisfy the stated requirements/interface) without "
             "breaking existing tests?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--model", default="sonnet")
    args = ap.parse_args()

    with open(os.path.join(HERE, "instances.json")) as f:
        instances = {i["instance_id"]: i for i in json.load(f)}
    plain = res.load(os.path.join(HERE, "results", "plain", "results.jsonl"))
    by_iid = {}
    for r in plain:
        by_iid.setdefault(r["instance_id"], []).append(r)

    sel_path = os.path.join(HERE, "results", "judge", "selection.jsonl")
    done = res.recorded(sel_path, "instance_id")
    backend = J.make_backend(JUDGE, os.path.join(HERE, "results", "judge"),
                             args.model)

    todo = [iid for iid in by_iid if (iid,) not in done]
    print(f"[swebench:select] {len(todo)} instances, K={args.k} "
          f"({len(done)} recorded)", flush=True)
    for n, iid in enumerate(todo, 1):
        inst = instances[iid]
        task = (f"Repository: {inst['repo']}\n\nIssue:\n"
                f"{inst['problem_statement']}\n\nRequirements:\n"
                f"{inst.get('requirements') or '(none)'}")
        scored = []
        for r in sorted(by_iid[iid], key=lambda r: r["sample"]):
            with open(r["patch_file"]) as f:
                patch = f.read()
            if not patch.strip():
                scored.append({"sample": r["sample"], "mean": -1.0,
                               "scores": [], "empty": True})
                continue
            truncated = len(patch) > PATCH_CHAR_CAP
            mean, scores, failures = J.score(
                JUDGE, backend, task, patch[:PATCH_CHAR_CAP], CRITERION,
                k=args.k)
            scored.append({"sample": r["sample"],
                           "mean": -1.0 if mean is None else mean,
                           "scores": scores, "failures": failures,
                           "truncated": truncated})
        pick = max(scored, key=lambda s: s["mean"])["sample"]
        res.append(sel_path, {"instance_id": iid, "pick": pick,
                              "scored": scored, "k": args.k,
                              "model": args.model})
        print(f"[swebench:select {n}/{len(todo)}] {iid} -> sample {pick} "
              f"({[s['mean'] for s in scored]})", flush=True)

    print("[swebench:select] done", flush=True)


if __name__ == "__main__":
    main()
