"""FrontierMath — the comparison table.

Usage:  uv run python experiments/frontiermath/summarize.py
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import results as res                        # noqa: E402


def main():
    with open(os.path.join(HERE, "problems.json")) as f:
        problems = json.load(f)
    plain = res.load(os.path.join(HERE, "results", "plain", "results.jsonl"))
    codex = {r["pid"]: r for r in res.load(
        os.path.join(HERE, "results", "codex", "results.jsonl"))}
    gv = {r["pid"]: r for r in res.load(
        os.path.join(HERE, "results", "gv", "results.jsonl"))}
    sel = {r["pid"]: r for r in res.load(
        os.path.join(HERE, "results", "judge", "selection.jsonl"))}

    by_pid = {}
    for r in plain:
        by_pid.setdefault(r["pid"], []).append(r)

    stats = Counter()
    print(f"{'problem':<30}{'tier':>5}{'plain':>8}{'judge':>7}{'codex':>7}"
          f"{'GV':>5}{'GV verdict':>12}")
    for p in problems:
        pid = p["id"]
        samples = by_pid.get(pid, [])
        n_ok = sum(1 for s in samples if s["solved"])
        stats["pass1_num"] += n_ok
        stats["pass1_den"] += len(samples)
        stats["oracle"] += any(s["solved"] for s in samples)
        s = sel.get(pid, {})
        stats["judge_sel"] += bool(s.get("correct"))
        c = codex.get(pid, {})
        stats["codex"] += bool(c.get("solved"))
        g = gv.get(pid, {})
        stats["gv"] += bool(g.get("solved"))
        if g.get("verdict") == "verified":
            stats["gv_ver_ok" if g.get("solved") else "gv_ver_BAD"] += 1
        print(f"{pid:<30}{p['tier']:>5}"
              f"{f'{n_ok}/{len(samples)}':>8}"
              f"{str(s.get('correct', '?')):>7}"
              f"{str(c.get('solved', '?')):>7}"
              f"{str(g.get('solved', '?')):>5}"
              f"{str(g.get('verdict')):>12}")

    n = len(problems)
    print(f"\nplain pass@1 (mean over samples): "
          f"{stats['pass1_num']}/{stats['pass1_den']}")
    print(f"plain oracle pass@N: {stats['oracle']}/{n}")
    print(f"judge selection correct: {stats['judge_sel']}/{n}")
    print(f"codex solved: {stats['codex']}/{n}")
    print(f"GV solved: {stats['gv']}/{n} · verified&correct "
          f"{stats['gv_ver_ok']} · verified&WRONG {stats['gv_ver_BAD']}")


if __name__ == "__main__":
    main()
