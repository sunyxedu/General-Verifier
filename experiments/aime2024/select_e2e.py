"""Paper-style best-of-N selection over the plain arm's samples, plus the
end-to-end comparison table.

The judge (the ported LLM-as-a-Verifier baseline, K repeats, no tools)
scores each plain sample's answer; argmax is the paper-pipeline's pick.
Majority vote and oracle pass@N are reported alongside. The e2e-gv arm's
row reads straight from its results (answer + certificate verdict).

Usage:  uv run python experiments/aime2024/select_e2e.py [--k 4]
"""
import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from problems_aime2026 import PROBLEMS               # noqa: E402
from run_judge import JUDGE                          # noqa: E402
from verifier.llm.agents import AgentBackend         # noqa: E402
from verifier.llm.roles import RoleError             # noqa: E402


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_score(backend, statement, answer, k):
    claim = (f"AIME problem: {statement} The answer is {answer}.")
    scores = []
    for _ in range(k):
        try:
            payload = JUDGE.call(backend, {"claim": claim})
            scores.append(max(0, min(10, int(payload["score"]))))
        except RoleError:
            pass
    return sum(scores) / len(scores) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    plain = load(os.path.join(HERE, "results", "e2e-plain", "results.jsonl"))
    gv = {r["pid"]: r
          for r in load(os.path.join(HERE, "results", "e2e-gv",
                                     "results.jsonl"))}
    by_pid = {}
    for r in plain:
        by_pid.setdefault(r["pid"], []).append(r)

    sel_path = os.path.join(HERE, "results", "e2e-plain", "selection.jsonl")
    sel_done = {json.loads(line)["pid"] for line in open(sel_path)} \
        if os.path.exists(sel_path) else set()
    backend = AgentBackend(roles={"llm_judge": JUDGE},
                           cwd=os.path.join(HERE, "results"), model="sonnet")

    stats = Counter()
    rows = []
    for p in PROBLEMS:
        pid = p["id"]
        samples = by_pid.get(pid, [])
        if not samples:
            continue
        answers = [s["answer"] for s in samples]
        truth = p["true"]
        stats["pass1_num"] += sum(1 for s in samples if s["solved"])
        stats["pass1_den"] += len(samples)
        stats["oracle"] += any(s["solved"] for s in samples)
        maj = Counter(a for a in answers if a is not None).most_common(1)
        maj_ans = maj[0][0] if maj else None
        stats["majority"] += maj_ans == truth

        if pid not in sel_done:
            scored = [(judge_score(backend, p["statement"], a, args.k)
                       if a is not None else -1.0, a) for a in answers]
            pick = max(scored)[1]
            with open(sel_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"pid": pid, "scored": scored,
                                    "pick": pick}) + "\n")
        picks = {json.loads(line)["pid"]: json.loads(line)
                 for line in open(sel_path)}
        pick = picks[pid]["pick"]
        stats["judge_sel"] += pick == truth
        g = gv.get(pid, {})
        rows.append((pid, answers, maj_ans, pick,
                     g.get("answer"), g.get("verdict"), truth))

    print(f"{'pid':<6}{'plain samples':<22}{'majority':>9}{'judge pick':>11}"
          f"{'GV answer':>10}{'GV verdict':>11}{'truth':>7}")
    for pid, answers, maj_ans, pick, ga, gverd, truth in rows:
        print(f"{pid:<6}{str(answers):<22}{str(maj_ans):>9}{str(pick):>11}"
              f"{str(ga):>10}{str(gverd):>11}{truth:>7}")

    n = len(rows)
    print(f"\nplain pass@1 (mean over samples): "
          f"{stats['pass1_num']}/{stats['pass1_den']}")
    print(f"oracle pass@N: {stats['oracle']}/{n}")
    print(f"majority vote: {stats['majority']}/{n}")
    print(f"paper-style judge selection: {stats['judge_sel']}/{n}")
    gv_solved = sum(1 for r in gv.values() if r.get("solved"))
    gv_cert_ok = sum(1 for r in gv.values()
                     if r.get("verdict") == "verified" and r.get("solved"))
    gv_false_cert = sum(1 for r in gv.values()
                        if r.get("verdict") == "verified"
                        and not r.get("solved"))
    print(f"GV end-to-end solved: {gv_solved}/{len(gv)} · "
          f"verified&correct {gv_cert_ok} · verified&WRONG {gv_false_cert}")


if __name__ == "__main__":
    main()
