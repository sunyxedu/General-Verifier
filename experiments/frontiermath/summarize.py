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

ARMS = ["plain", "turbo", "codex", "gv"]


def main():
    with open(os.path.join(HERE, "problems.json")) as f:
        problems = json.load(f)
    by_arm = {}
    for arm in ARMS:
        rows = res.load(os.path.join(HERE, "results", arm, "results.jsonl"))
        per_pid = {}
        for r in rows:
            per_pid.setdefault(r["pid"], []).append(r)
        by_arm[arm] = per_pid

    stats = Counter()
    print(f"{'problem':<30}{'tier':>5}" +
          "".join(f"{a:>8}" for a in ARMS) + f"{'GV verdict':>12}")
    for p in problems:
        pid = p["id"]
        row = f"{pid:<30}{p['tier']:>5}"
        for arm in ARMS:
            samples = by_arm[arm].get(pid, [])
            if not samples:
                row += f"{'?':>8}"
                continue
            n_ok = sum(1 for s in samples if s["solved"])
            row += f"{f'{n_ok}/{len(samples)}':>8}"
            stats[f"{arm}_num"] += n_ok
            stats[f"{arm}_den"] += len(samples)
        g = (by_arm["gv"].get(pid) or [{}])[0]
        row += f"{str(g.get('verdict')):>12}"
        if g.get("verdict") == "verified":
            stats["gv_ver_ok" if g.get("solved") else "gv_ver_BAD"] += 1
        print(row)

    print()
    for arm in ARMS:
        if stats[f"{arm}_den"]:
            print(f"{arm:<7} solved: "
                  f"{stats[f'{arm}_num']}/{stats[f'{arm}_den']}")
    print(f"GV verified&correct {stats['gv_ver_ok']} · "
          f"verified&WRONG (false certificate) {stats['gv_ver_BAD']}")


if __name__ == "__main__":
    main()
