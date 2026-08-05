"""FrontierMath (official public sample, 12 problems) — solver arms.

Per problem the solver gets ONE prompt in a fresh scratch directory (it may
write and run code there). Answers are mechanically checked against the
official published answers (check.py).

Arms:
  --arm gv       Claude Code + full General Verifier mounted (MCP sidecar +
                 hooks + solver protocol) — one prompt, normally used
  --arm plain    stock headless Claude Code  (--n samples/problem)
  --arm codex    stock headless Codex CLI

Usage:
  uv run python experiments/frontiermath/run.py --arm gv
  uv run python experiments/frontiermath/run.py --arm plain --n 3
  uv run python experiments/frontiermath/run.py --arm codex
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import results as res                        # noqa: E402
from common import solver                                # noqa: E402
from check import matches, parse_final                   # noqa: E402

WORKROOT = "/tmp/gv-frontiermath"      # outside the repo (see common.solver)

ANSWER_SPEC = {
    "integer": "The answer is a single (possibly very large) integer.",
    "sympy": ("The answer is an exact closed-form expression; write it as "
              "a sympy-parseable expression (e.g. sqrt(3)/36 + ...)."),
}

BASE_PROMPT = """\
{statement}

{answer_spec} You may write and run code in this directory. End your reply \
with a line of exactly this form: FINAL ANSWER: <answer>
"""

GV_SUFFIX = """
Follow the claim protocol in CLAUDE.md while you work: commit your key \
intermediate results as claims, commit a final claim whose text ends with \
'The answer is <answer>.', and request gv.certificate on it before you \
finish. Report the verdict verbatim.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["gv", "plain", "codex"])
    ap.add_argument("--n", type=int, default=1, help="samples per problem "
                    "(use >1 on plain for best-of-N selection)")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--timeout", type=int, default=5400)
    args = ap.parse_args()

    with open(os.path.join(HERE, "problems.json")) as f:
        problems = json.load(f)
    results_path = os.path.join(HERE, "results", args.arm, "results.jsonl")
    done = res.recorded(results_path, "pid", "sample")
    todo = [(p, s) for p in problems for s in range(args.n)
            if (p["id"], s) not in done]
    print(f"[fm:{args.arm}] {len(todo)} runs ({len(done)} recorded)",
          flush=True)

    for i, (p, s) in enumerate(todo, 1):
        work = os.path.join(WORKROOT, args.arm, p["id"], f"s{s}")
        os.makedirs(work, exist_ok=True)
        prompt = BASE_PROMPT.format(statement=p["statement"],
                                    answer_spec=ANSWER_SPEC[p["answer_type"]])
        if args.arm == "gv":
            prompt += GV_SUFFIX
            solver.mount_gv(work, model=args.model)
            out, seconds = solver.run_claude(work, prompt, model=args.model,
                                             mcp=True, timeout=args.timeout)
        elif args.arm == "plain":
            out, seconds = solver.run_claude(work, prompt, model=args.model,
                                             timeout=args.timeout)
        else:
            out, seconds = solver.run_codex(work, prompt,
                                            timeout=args.timeout)
        transcripts_dir = os.path.join(HERE, "results", args.arm,
                                       "transcripts")
        os.makedirs(transcripts_dir, exist_ok=True)
        transcript_file = os.path.join(transcripts_dir,
                                       f"{p['id']}_s{s}.txt")
        with open(transcript_file, "w") as f:
            f.write(out)
        answer = parse_final(out)
        rec = {"pid": p["id"], "sample": s, "arm": args.arm,
               "model": args.model, "answer": answer,
               "solved": matches(answer, p), "seconds": seconds,
               "workdir": work, "transcript_file": transcript_file,
               "output_tail": out[-300:]}
        if args.arm == "gv":
            gvo = solver.gv_outcome(work)
            # prefer the certified final claim's answer when present
            claim_answer = parse_final(gvo["final_text"] or "")
            if claim_answer is None and gvo["final_text"]:
                import re
                m = re.findall(r"answer is\s*([^.]+)\.?\s*$",
                               gvo["final_text"], re.IGNORECASE)
                claim_answer = m[-1].strip() if m else None
            if claim_answer is not None:
                rec["answer"] = claim_answer
                rec["solved"] = matches(claim_answer, p)
            rec.update({k: v for k, v in gvo.items() if k != "final_text"})
            rec["final_text"] = (gvo["final_text"] or "")[:200]
        res.append(results_path, rec)
        extra = (f" verdict={rec.get('verdict')}"
                 f" refuted={rec.get('refuted_claims')}"
                 if args.arm == "gv" else "")
        print(f"[fm:{args.arm} {i}/{len(todo)}] {p['id']}#{s} "
              f"answer={str(rec['answer'])[:40]} solved={rec['solved']}"
              f"{extra} ({seconds}s)", flush=True)

    print(f"[fm:{args.arm}] done", flush=True)


if __name__ == "__main__":
    main()
