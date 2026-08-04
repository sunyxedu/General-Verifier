"""Arm A — General Verifier: every claim runs the full pipeline.

One fresh state dir per claim (no cross-claim reuse: measurement stays
i.i.d.), agent seats, binding game on. The claim is committed as a single
`final`; requesting the certificate is what triggers verification (the
consumer is the first dependent). Verdict, wall time and event count land
in results.jsonl; a claim already recorded is skipped, so the run is
resumable.

Usage:  uv run python experiments/aime2024/run_gv.py [--model sonnet] [--only p13]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from problems import PROBLEMS, claim_text          # noqa: E402
from verifier.sidecar import Sidecar               # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results", "gv")
RESULTS = os.path.join(RESULTS_DIR, "results.jsonl")
TOOLCHAIN = os.path.expanduser("~/.cache/gv-toolchain")


def recorded() -> set:
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["pid"], r["tag"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--only", default=None, help="comma-separated problem ids")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = recorded()
    todo = [(p, tag) for p in PROBLEMS for tag in ("true", "trap")
            if (only is None or p["id"] in only)
            and (p["id"], tag) not in done]
    print(f"[gv] {len(todo)} claims to run ({len(done)} already recorded)",
          flush=True)

    for i, (p, tag) in enumerate(todo, 1):
        answer = p[tag]
        state_dir = os.path.join(RESULTS_DIR, f"{p['id']}_{tag}")
        t0 = time.time()
        verdict, cert_id, error = "error", None, None
        try:
            sc = Sidecar(state_dir, model=args.model,
                         toolchain_dir=TOOLCHAIN)
            sc.commit("f", "final", claim_text(p, answer), [])
            cert = sc.certificate("f")
            verdict = cert.get("verdict", f"error: {cert.get('error')}")
            cert_id = cert.get("certificate_id")
        except Exception as err:                    # noqa: BLE001
            error = repr(err)[:300]
        seconds = round(time.time() - t0, 1)
        events = 0
        ev_path = os.path.join(state_dir, "events.jsonl")
        if os.path.exists(ev_path):
            with open(ev_path, encoding="utf-8") as f:
                events = sum(1 for line in f if line.strip())
        rec = {
            "pid": p["id"], "tag": tag, "answer": answer,
            "label": "correct" if tag == "true" else "wrong",
            "verdict": verdict, "cert": cert_id, "error": error,
            "seconds": seconds, "events": events, "model": args.model,
        }
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[gv {i}/{len(todo)}] {p['id']}_{tag} answer={answer} "
              f"-> {verdict}  ({seconds}s, {events} events)", flush=True)

    print("[gv] done", flush=True)


if __name__ == "__main__":
    main()
