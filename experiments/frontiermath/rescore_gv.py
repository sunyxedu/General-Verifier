"""Rescore recorded gv-arm rows from what they left behind: the reply
(transcript) and the ledger. Earlier rows were scored against the FIRST
final claim committed instead of the reply; this recomputes answer/solved
with the fixed rule (reply first, the certified or latest final as fallback)
and keeps the old values under `rescored_from` so nothing is lost.

  uv run python experiments/frontiermath/rescore_gv.py [--arm gv-codex] [--write]
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE)), HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from check import matches, parse_final          # noqa: E402
from common import solver                       # noqa: E402
from run import final_claim_answer              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gv-codex")
    ap.add_argument("--write", action="store_true",
                    help="rewrite results.jsonl (a .bak copy is kept)")
    args = ap.parse_args()
    problems = {p["id"]: p for p in json.load(open(os.path.join(HERE, "problems.json")))}
    path = os.path.join(HERE, "results", args.arm, "results.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = []
    for r in rows:
        p = problems[r["pid"]]
        tf = r.get("transcript_file") or os.path.join(
            HERE, "results", args.arm, "transcripts", f"{r['pid']}_s{r.get('sample', 0)}.txt")
        reply = open(tf, encoding="utf-8").read() if os.path.exists(tf) else None
        reply_answer = parse_final(reply) if reply is not None else None
        work = r.get("workdir") or ""
        gvo = solver.gv_outcome(work) if os.path.isdir(os.path.join(work, ".gv")) else None
        claim_answer = final_claim_answer(gvo["final_text"]) if gvo else None
        if reply_answer is not None:
            answer, source = reply_answer, "reply"
        elif claim_answer is not None:
            answer, source = claim_answer, "final claim (reply had none)"
        else:
            answer, source = r.get("answer"), "unchanged (no reply or ledger left)"
        new = dict(r)
        new.update(answer=answer, solved=matches(answer, p), answer_source=source,
                   reply_answer=reply_answer, claim_answer=claim_answer)
        if gvo:
            new.update(certificate=gvo["certificate"], finals=gvo["finals"],
                       final_id=gvo["final_id"])
        if (new["answer"], new["solved"]) != (r.get("answer"), r.get("solved")):
            new["rescored_from"] = {"answer": r.get("answer"), "solved": r.get("solved")}
        print("%-32s solved %-5s -> %-5s  [%s]" % (r["pid"], r.get("solved"), new["solved"], source))
        out.append(new)
    if args.write:
        shutil.copy2(path, path + ".bak")
        with open(path, "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print("written; previous file kept at", path + ".bak")


if __name__ == "__main__":
    main()
