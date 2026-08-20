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
from verifier import selection                           # noqa: E402

ARMS = ["plain", "turbo", "codex", "gv", "rec"]


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

    selection_table(problems, by_arm.get("rec", {}))


def selection_table(problems, rec_by_pid):
    """Best-of-N over the rec arm's offline scores (the ORM story):
    pass@1 / majority vote / verifier argmax / verifier-weighted vote /
    pass@N oracle, plus the verified-precision count."""
    scores = {(r["pid"], r["sample"]): r["features"] for r in
              res.load(os.path.join(HERE, "results", "rec", "scores.jsonl"))}
    if not scores:
        return
    strategies = [("pass@1", None), ("majority", selection.majority_vote),
                  ("argmax", selection.argmax),
                  ("weighted", selection.weighted_vote), ("oracle", None)]
    totals = Counter()
    print(f"\n{'best-of-N (rec)':<30}{'N':>4}" +
          "".join(f"{name:>10}" for name, _ in strategies))
    for p in problems:
        rows = [dict(r, features=scores.get((p['id'], r['sample'])))
                for r in rec_by_pid.get(p["id"], [])]
        rows = [r for r in rows if r["features"] is not None]
        if not rows:
            continue
        line = f"{p['id']:<30}{len(rows):>4}"
        for name, strategy in strategies:
            if name == "pass@1":
                value = selection.pass_at_1(rows)
                line += f"{value:>10.2f}"
                totals["pass@1"] += value
            elif name == "oracle":
                ok = selection.oracle(rows)
                line += f"{str(ok):>10}"
                totals["oracle"] += ok
            else:
                chosen = strategy(rows)
                ok = bool(chosen and chosen["solved"])
                line += f"{str(ok):>10}"
                totals[name] += ok
        totals["problems"] += 1
        print(line)
    n = totals["problems"]
    if n:
        print(f"{'TOTAL':<30}{'':>4}"
              f"{totals['pass@1'] / n:>10.2f}"
              + "".join(f"{totals[k]}/{n}".rjust(10)
                        for k in ("majority", "argmax", "weighted", "oracle")))
    verified = [(pid_s, f) for pid_s, f in scores.items()
                if f and f.get("verdict") == "verified"]
    rec_rows = {(r["pid"], r["sample"]): r
                for rows in rec_by_pid.values() for r in rows}
    ok = sum(1 for k, _ in verified if rec_rows.get(k, {}).get("solved"))
    print(f"rec verified&correct {ok} · "
          f"verified&WRONG (false certificate) {len(verified) - ok}")


if __name__ == "__main__":
    main()
