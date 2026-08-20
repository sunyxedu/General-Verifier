"""SWE-bench Pro — the comparison table.

Reads the official-harness eval results for every evaluated arm plus the
GV arm's certificate verdicts and prints:
  - per-arm resolve rates (gv / turbo / plain / codex)
  - GV verdict cross-table: verified&resolved vs verified&UNRESOLVED
    (false certificate — the one-vote veto number)

Usage:  uv run python experiments/swebench_pro/summarize.py
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import results as res                        # noqa: E402


def eval_results(tag):
    path = os.path.join(HERE, "results", "eval", tag, "eval_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    with open(os.path.join(HERE, "instances.json")) as f:
        instances = json.load(f)
    iids = [i["instance_id"] for i in instances]
    short = {iid: iid.split("__")[1][:28] for iid in iids}

    evals = {}
    eval_dir = os.path.join(HERE, "results", "eval")
    if os.path.isdir(eval_dir):
        for tag in sorted(os.listdir(eval_dir)):
            r = eval_results(tag)
            if r is not None:
                evals[tag] = r

    gv = {r["instance_id"]: r for r in res.load(
        os.path.join(HERE, "results", "gv", "results.jsonl"))}

    tags = list(evals)
    print(f"{'instance':<30}" + "".join(f"{t:>12}" for t in tags)
          + f"{'GV verdict':>12}")
    for iid in iids:
        row = f"{short[iid]:<30}"
        for t in tags:
            v = evals[t].get(iid)
            row += f"{'PASS' if v else '-' if v is not None else '?':>12}"
        row += f"{str(gv.get(iid, {}).get('verdict')):>12}"
        print(row)

    print()
    for t in tags:
        r = evals[t]
        ok = sum(1 for v in r.values() if v)
        print(f"{t:<12} resolved {ok}/{len(r)}")

    plain_tags = [t for t in tags if t.startswith("plain_s")]
    if plain_tags:
        per_iid = defaultdict(list)
        for t in plain_tags:
            for iid, v in evals[t].items():
                per_iid[iid].append(v)
        num = sum(sum(vs) for vs in per_iid.values())
        den = sum(len(vs) for vs in per_iid.values())
        oracle = sum(1 for vs in per_iid.values() if any(vs))
        print(f"\nplain pass@1 (mean over samples): {num}/{den}")
        print(f"plain oracle pass@N: {oracle}/{len(per_iid)}")

    if gv and "gv" in evals:
        ver_ok = sum(1 for iid, r in gv.items()
                     if r.get("verdict") == "verified" and evals["gv"].get(iid))
        ver_bad = sum(1 for iid, r in gv.items()
                      if r.get("verdict") == "verified"
                      and evals["gv"].get(iid) is False)
        print(f"\nGV verified&resolved: {ver_ok} · "
              f"verified&UNRESOLVED (false certificate): {ver_bad}")

    rec_selection(evals)


def rec_selection(evals):
    """Best-of-N over the rec arm's offline scores. Patches are not
    comparable answers, so no vote strategies here — argmax vs pass@1 vs
    the pass@N oracle, using the rec_s<sample> eval tags as ground truth
    (rank-only table when evaluation has not run yet)."""
    from verifier import offline, selection

    recs = res.load(os.path.join(HERE, "results", "rec", "results.jsonl"))
    scores = {(r["instance_id"], r["sample"]): r["features"] for r in
              res.load(os.path.join(HERE, "results", "rec", "scores.jsonl"))}
    if not scores:
        return
    by_iid = defaultdict(list)
    for r in recs:
        features = scores.get((r["instance_id"], r["sample"]))
        if features is None:
            continue
        verdict = evals.get(f"rec_s{r['sample']}", {}).get(r["instance_id"])
        by_iid[r["instance_id"]].append(
            {"sample": r["sample"], "features": features,
             "solved": bool(verdict), "evaluated": verdict is not None})
    evaluated = all(row["evaluated"] for rows in by_iid.values() for row in rows)
    totals = defaultdict(float)
    print(f"\n{'best-of-N (rec)':<30}{'N':>4}{'picked':>8}"
          f"{'verdict':>12}" + (f"{'pass@1':>8}{'argmax':>8}{'oracle':>8}"
                                if evaluated else "  (not evaluated yet)"))
    for iid in sorted(by_iid):
        rows = by_iid[iid]
        chosen = selection.argmax(rows)
        picked = "s%d" % chosen["sample"]
        short_iid = iid.split("__")[-1][:28]
        line = (f"{short_iid:<30}{len(rows):>4}{picked:>8}"
                f"{str(chosen['features'].get('verdict')):>12}")
        if evaluated:
            p1 = selection.pass_at_1(rows)
            argmax_ok = bool(chosen["solved"])
            oracle_ok = selection.oracle(rows)
            line += f"{p1:>8.2f}{str(argmax_ok):>8}{str(oracle_ok):>8}"
            totals["pass@1"] += p1
            totals["argmax"] += argmax_ok
            totals["oracle"] += oracle_ok
        totals["n"] += 1
        print(line)
    if evaluated and totals["n"]:
        n = int(totals["n"])
        argmax_cell = "%d/%d" % (totals["argmax"], n)
        oracle_cell = "%d/%d" % (totals["oracle"], n)
        print(f"{'TOTAL':<30}{'':>4}{'':>8}{'':>12}"
              f"{totals['pass@1'] / n:>8.2f}"
              f"{argmax_cell:>8}{oracle_cell:>8}")


if __name__ == "__main__":
    main()
