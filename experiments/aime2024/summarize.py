"""Join both arms' results into the comparison table.

GV scoring: label=correct -> verified is right, refuted is a FALSE ALARM;
label=wrong -> refuted is right, verified is a FALSE PASS (the one-vote
veto); partial/unverified/error count as abstention either way.

Judge scoring: mean score >= 5.5 predicts correct (midpoint of the 0..10
scale); also reported per-threshold so the cut is not doing hidden work.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[(r["pid"], r["tag"])] = r
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="pset", default="aime",
                    choices=["aime", "novel"])
    args = ap.parse_args()
    suffix = "" if args.pset == "aime" else "-novel"
    gv = load(os.path.join(HERE, "results", "gv" + suffix, "results.jsonl"))
    jd = load(os.path.join(HERE, "results", "judge" + suffix,
                           "results.jsonl"))
    keys = sorted(set(gv) | set(jd))

    print(f"{'claim':<12}{'label':<9}{'GV verdict':<14}{'GV s':>7}"
          f"{'judge mean':>12}  scores")
    stats = {"gv_right": 0, "gv_false_pass": 0, "gv_false_alarm": 0,
             "gv_abstain": 0, "jd_right": 0, "jd_wrong": 0}
    for k in keys:
        g, j = gv.get(k), jd.get(k)
        label = (g or j)["label"]
        gvv = g["verdict"] if g else "—"
        if g:
            if label == "correct":
                if gvv == "verified":
                    stats["gv_right"] += 1
                elif gvv == "refuted":
                    stats["gv_false_alarm"] += 1
                else:
                    stats["gv_abstain"] += 1
            else:
                if gvv == "refuted":
                    stats["gv_right"] += 1
                elif gvv == "verified":
                    stats["gv_false_pass"] += 1
                else:
                    stats["gv_abstain"] += 1
        jmean = j["mean"] if j else None
        if j and jmean is not None:
            pred = "correct" if jmean >= 5.5 else "wrong"
            stats["jd_right" if pred == label else "jd_wrong"] += 1
        print(f"{k[0]+'_'+k[1]:<12}{label:<9}{gvv:<14}"
              f"{(str(g['seconds'])+'s') if g else '—':>7}"
              f"{jmean if jmean is not None else '—':>12}  "
              f"{j['scores'] if j else ''}")

    n_gv, n_jd = len(gv), len(jd)
    print(f"\nGeneral Verifier ({n_gv} claims): "
          f"{stats['gv_right']} right · {stats['gv_false_pass']} FALSE PASS · "
          f"{stats['gv_false_alarm']} false alarm · "
          f"{stats['gv_abstain']} abstained")
    if n_gv:
        total_s = sum(g["seconds"] for g in gv.values())
        print(f"  wall time {round(total_s/60,1)} min total, "
              f"{round(total_s/n_gv,1)} s/claim")
    print(f"LLM judge ({n_jd} claims, threshold 5.5): "
          f"{stats['jd_right']} right · {stats['jd_wrong']} wrong")
    if jd:
        for th in (3.5, 4.5, 5.5, 6.5, 7.5, 8.5):
            right = sum(
                1 for r in jd.values() if r["mean"] is not None and
                (("correct" if r["mean"] >= th else "wrong") == r["label"]))
            print(f"  threshold {th}: {right}/{n_jd}")


if __name__ == "__main__":
    main()
