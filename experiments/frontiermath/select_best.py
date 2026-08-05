# /// script
# requires-python = ">=3.11"
# dependencies = ["google-genai", "openai", "tqdm"]
# ///
"""FrontierMath — judge arm: the OFFICIAL LLM-as-a-Verifier, as shipped.

Best-of-N selection over the plain arm's samples with the authors' pipeline
(arXiv 2607.05391) — fine-grained logprob reward + Probabilistic Pivot
Tournament. Criteria live in criteria_frontiermath.md (written in the
repo's own TEMPLATE style, since FrontierMath is not among its bundled
benchmarks). Candidates are the solvers' transcripts; the pick is then
scored mechanically against the official answers.

Backend: logprob-capable endpoint — VERTEX_API_KEY (gemini-2.5-flash,
default) or OPENAI_BASE_URL/OPENAI_API_KEY (vLLM etc.). Anthropic models
cannot serve this arm (no logprobs; the paper's App. A).

Usage:  uv run experiments/frontiermath/select_best.py [--model m] [--k 8]
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
from check import matches                                # noqa: E402

TRANSCRIPT_CHAR_CAP = 30000


def candidate_text(r):
    tf = r.get("transcript_file")
    text = ""
    if tf and os.path.exists(tf):
        with open(tf, errors="replace") as f:
            text = f.read().strip()[-TRANSCRIPT_CHAR_CAP:]
    if not text:
        text = (f"(transcript unavailable)\n"
                f"FINAL ANSWER: {r.get('answer')}")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="verifier model (default: the package's own)")
    ap.add_argument("--k", type=int, default=8, help="n_evaluations")
    ap.add_argument("--pivots", type=int, default=2)
    args = ap.parse_args()

    lv = official_verifier.load()
    model = args.model or lv.DEFAULT_MODEL
    criteria = os.path.join(HERE, "criteria_frontiermath.md")

    with open(os.path.join(HERE, "problems.json")) as f:
        problems = {p["id"]: p for p in json.load(f)}
    plain = res.load(os.path.join(HERE, "results", "plain", "results.jsonl"))
    by_pid = {}
    for r in plain:
        by_pid.setdefault(r["pid"], []).append(r)

    out_dir = os.path.join(HERE, "results", "judge")
    os.makedirs(out_dir, exist_ok=True)
    sel_path = os.path.join(out_dir, "selection.jsonl")
    done = res.recorded(sel_path, "pid")

    todo = [pid for pid in by_pid if (pid,) not in done]
    print(f"[fm:official] {len(todo)} selections, K={args.k}, "
          f"model={model} ({len(done)} recorded)", flush=True)
    for n, pid in enumerate(todo, 1):
        p = problems[pid]
        rows = sorted(by_pid[pid], key=lambda r: r["sample"])
        t0 = time.time()
        try:
            r = lv.select(p["statement"], [candidate_text(x) for x in rows],
                          criteria=criteria, n_evaluations=args.k,
                          pivots=args.pivots, model=model,
                          cache=os.path.join(out_dir, f"cache_{pid}.json"),
                          progress=False)
            picked = rows[r.index]
            rec = {"pid": pid, "pick": picked["sample"],
                   "answer": picked.get("answer"),
                   "correct": matches(picked.get("answer"), p),
                   "scores": [round(s, 4) for s in r.scores],
                   "ranking": r.ranking, "n_comparisons": r.n_comparisons,
                   "seconds": round(time.time() - t0, 1),
                   "model": model, "k": args.k}
        except Exception as err:                          # noqa: BLE001
            rec = {"pid": pid, "error": repr(err)[:300],
                   "seconds": round(time.time() - t0, 1), "model": model}
        res.append(sel_path, rec)
        print(f"[fm:official {n}/{len(todo)}] {pid} -> "
              f"{rec.get('pick', 'ERROR')} correct={rec.get('correct')}",
              flush=True)

    print("[fm:official] done", flush=True)


if __name__ == "__main__":
    main()
