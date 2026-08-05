# /// script
# requires-python = ">=3.11"
# dependencies = ["google-genai", "openai", "tqdm"]
# ///
"""SWE-bench Pro — judge arm: the OFFICIAL LLM-as-a-Verifier, as shipped.

Best-of-N selection over the plain arm's candidates with the authors'
pipeline (arXiv 2607.05391): fine-grained logprob reward + Probabilistic
Pivot Tournament, using the repo's own bundled `swe_bench` criteria file.
Each candidate is the solver's transcript plus its final patch — the
trajectory shape the criteria were written for. The pick lands in
selection.jsonl; evaluate.py --arm judge scores it mechanically.

Backend: logprob-capable endpoint — VERTEX_API_KEY (gemini-2.5-flash,
default) or OPENAI_BASE_URL/OPENAI_API_KEY (vLLM etc.). Anthropic models
cannot serve this arm (no logprobs; the paper's App. A).

Usage:  uv run experiments/swebench_pro/select_best.py [--model m] [--k 8]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import official_verifier                     # noqa: E402
from common import results as res                        # noqa: E402

PATCH_CHAR_CAP = 30000
TRANSCRIPT_CHAR_CAP = 30000


def candidate_text(r):
    parts = []
    tf = r.get("transcript_file")
    if tf and os.path.exists(tf):
        with open(tf, errors="replace") as f:
            t = f.read().strip()
        if t:
            parts.append("Agent transcript:\n" + t[-TRANSCRIPT_CHAR_CAP:])
    with open(r["patch_file"]) as f:
        patch = f.read().strip()
    parts.append("Final patch:\n" + (patch[:PATCH_CHAR_CAP]
                                     if patch else "(empty patch)"))
    return "\n\n".join(parts), bool(patch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="verifier model (default: the package's own)")
    ap.add_argument("--k", type=int, default=8, help="n_evaluations")
    ap.add_argument("--pivots", type=int, default=2)
    args = ap.parse_args()

    lv = official_verifier.load()
    model = args.model or lv.DEFAULT_MODEL

    with open(os.path.join(HERE, "instances.json")) as f:
        instances = {i["instance_id"]: i for i in json.load(f)}
    plain = res.load(os.path.join(HERE, "results", "plain", "results.jsonl"))
    by_iid = {}
    for r in plain:
        by_iid.setdefault(r["instance_id"], []).append(r)

    out_dir = os.path.join(HERE, "results", "judge")
    os.makedirs(out_dir, exist_ok=True)
    sel_path = os.path.join(out_dir, "selection.jsonl")
    done = res.recorded(sel_path, "instance_id")

    todo = [iid for iid in by_iid if (iid,) not in done]
    print(f"[swebench:official] {len(todo)} selections, K={args.k}, "
          f"model={model} ({len(done)} recorded)", flush=True)
    for n, iid in enumerate(todo, 1):
        inst = instances[iid]
        problem = (f"Repository: {inst['repo']}\n\nIssue:\n"
                   f"{inst['problem_statement']}\n\nRequirements:\n"
                   f"{inst.get('requirements') or '(none)'}")
        rows = sorted(by_iid[iid], key=lambda r: r["sample"])
        cands = [candidate_text(r) for r in rows]
        t0 = time.time()
        try:
            r = lv.select(problem, [c for c, _ in cands],
                          criteria="swe_bench", n_evaluations=args.k,
                          pivots=args.pivots, model=model,
                          cache=os.path.join(out_dir, f"cache_{iid}.json"),
                          progress=False)
            rec = {"instance_id": iid, "pick": rows[r.index]["sample"],
                   "scores": [round(s, 4) for s in r.scores],
                   "ranking": r.ranking, "n_comparisons": r.n_comparisons,
                   "empty_patch": [not has for _, has in cands],
                   "seconds": round(time.time() - t0, 1),
                   "model": model, "k": args.k}
        except Exception as err:                          # noqa: BLE001
            rec = {"instance_id": iid, "error": repr(err)[:300],
                   "seconds": round(time.time() - t0, 1), "model": model}
        res.append(sel_path, rec)
        print(f"[swebench:official {n}/{len(todo)}] {iid} -> "
              f"{rec.get('pick', 'ERROR')} scores={rec.get('scores')}",
              flush=True)

    print("[swebench:official] done", flush=True)


if __name__ == "__main__":
    main()
