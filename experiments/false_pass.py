"""成立性实验 (report §16, M1 acceptance): false-pass rate, verifier vs direct judge.

Benchmark pairs live HERE, in the experiment — they are the exam FOR the
system, not knowledge IN the system. The verifier package contains no domain
content; this file may, exactly like a test set may contain labeled answers.

Two arms, same model:
- direct judge: one no-tool structured call, "is this claim true?"
- verifier: the full kernel — obligations agent, binding game (oracle
  enumeration, discrimination exam, entailment audit), agent discharger.

Metrics (per arm): false-pass (false claim called true/verified) — the metric
that matters; false-alarm (true claim called false/refuted); abstentions
(verifier only: unverified is an honest "don't know", never a pass).

Usage:
  .venv/bin/python experiments/false_pass.py --out experiments/results/run1 \
      --model claude-sonnet-5 --concurrency 4 [--limit N] [--dry-run]

Results are written incrementally per claim (idempotent: finished claims are
skipped on rerun).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (claim, is_true) — machine-checkable, false ones subtly so
BENCHMARK = [
    ("对所有 1≤n≤1000，6 整除 n^3-n", True),
    ("对所有 1≤n≤1000，12 整除 n^3-n", False),          # n=2 -> 6
    ("对所有 1≤n≤300，30 整除 n^5-n", True),
    ("对所有 1≤n≤300，60 整除 n^5-n", False),           # n=2 -> 30
    ("1 到 100 的所有整数之和等于 5050", True),
    ("1 到 100 的所有整数之和等于 5051", False),
    ("2^13-1 = 8191 是素数", True),
    ("2^11-1 = 2047 是素数", False),                     # 23 * 89
    ("对所有 1≤n≤200，9 整除 10^n-1", True),
    ("对所有 1≤n≤200，11 整除 10^n-1", False),          # n=1 -> 9
    ("组合数 C(10,3) 等于 120", True),
    ("组合数 C(10,3) 等于 125", False),
]


def claim_slug(text: str) -> str:
    from verifier.model import content_hash
    return content_hash(text)[:12]


# ---- arms --------------------------------------------------------------------

def run_judge(text: str, model: str) -> dict:
    """Direct judge baseline: no tools, pure reasoning, structured verdict."""
    from verifier.agents import AgentBackend, AgentRole

    role = AgentRole(
        name="direct_judge",
        template="命题：{claim_text}\n判断该命题是否为真。",
        schema={"verdict": ("true", "false"), "reason": str},
        tools=[],
        model=model,
        max_turns=4,
    )
    backend = AgentBackend(roles={"direct_judge": role})
    t0 = time.time()
    payload = role.call(backend, {"claim_text": text})
    return {"verdict": payload["verdict"], "reason": payload["reason"],
            "seconds": round(time.time() - t0, 1)}


def run_verifier(text: str, model: str, workdir: str,
                 n_variants: int = 1, max_methods: int = 2) -> dict:
    """Full pipeline on one claim: commit A, verify A via the single rule."""
    from verifier.agents import ALL_AGENT_ROLES, AgentBackend, COMPILE_OBLIGATIONS_AGENT
    from verifier.builder import Builder
    from verifier.discharge import AgentDischarger
    from verifier.events import EventLog
    from verifier.game import BindingGame
    from verifier.instruments import ArtifactStore
    from verifier.kernel import Kernel
    from verifier.model import Claim
    from verifier.store import ClaimStore

    os.makedirs(workdir, exist_ok=True)
    log = EventLog(os.path.join(workdir, "events.jsonl"))
    log.append("commit", {"label": "A", "kind": "assertion", "text": text,
                          "deps": []})
    backend = AgentBackend(roles=ALL_AGENT_ROLES, cwd=workdir, model=model)
    store = ClaimStore()
    artifacts = ArtifactStore(os.path.join(workdir, "artifacts"))
    game = BindingGame(
        backend, artifacts, log,
        AgentDischarger(backend, artifacts,
                        workdir_base=os.path.join(workdir, "work")),
        n_variants=n_variants, max_methods=max_methods,
    )
    kernel = Kernel(store, log, backend, artifacts,
                    obligations_role=COMPILE_OBLIGATIONS_AGENT, binder=game)
    builder = Builder(log, store).replay()
    a = builder.labels["A"]

    t0 = time.time()
    status = kernel.verify(a)
    return {
        "status": status,
        "grade": store.grade(a),
        "evidence": store.evidence_for(a),
        "seconds": round(time.time() - t0, 1),
    }


# ---- scoring (pure; unit-tested) ---------------------------------------------

def classify(is_true: bool, judge_verdict: str = None,
             verifier_status: str = None) -> str:
    """-> ok | false_pass | false_alarm | abstain"""
    if judge_verdict is not None:
        called_true = judge_verdict == "true"
        if called_true and not is_true:
            return "false_pass"
        if not called_true and is_true:
            return "false_alarm"
        return "ok"
    if verifier_status == "verified":
        return "ok" if is_true else "false_pass"
    if verifier_status == "refuted":
        return "false_alarm" if is_true else "ok"
    return "abstain"


def summarize(rows: list) -> dict:
    def count(arm, key):
        return sum(1 for r in rows if r["arm"] == arm and r["outcome"] == key)

    out = {}
    for arm in ("judge", "verifier"):
        n = sum(1 for r in rows if r["arm"] == arm)
        out[arm] = {
            "n": n,
            "false_pass": count(arm, "false_pass"),
            "false_alarm": count(arm, "false_alarm"),
            "abstain": count(arm, "abstain"),
            "ok": count(arm, "ok"),
        }
    return out


# ---- driver ------------------------------------------------------------------

def run_one(item, model, out_dir):
    text, is_true = item
    slug = claim_slug(text)
    path = os.path.join(out_dir, f"{slug}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    record = {"claim": text, "is_true": is_true, "slug": slug}
    try:
        judge = run_judge(text, model)
        record["judge"] = judge
        record["judge_outcome"] = classify(is_true, judge_verdict=judge["verdict"])
    except Exception as err:  # noqa: BLE001 - record and move on
        record["judge"] = {"error": str(err)}
        record["judge_outcome"] = "error"

    try:
        verifier = run_verifier(text, model, os.path.join(out_dir, slug))
        record["verifier"] = verifier
        record["verifier_outcome"] = classify(
            is_true, verifier_status=verifier["status"])
    except Exception as err:  # noqa: BLE001
        record["verifier"] = {"error": str(err)}
        record["verifier_outcome"] = "error"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    print(f"[done] {record['judge_outcome']}/{record['verifier_outcome']} {text}",
          flush=True)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = BENCHMARK[: args.limit * 2] if args.limit else BENCHMARK
    if args.dry_run:
        for text, is_true in items:
            print(("T" if is_true else "F"), text)
        return

    os.makedirs(args.out, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        records = list(pool.map(
            lambda it: run_one(it, args.model, args.out), items))

    rows = []
    for r in records:
        rows.append({"arm": "judge", "outcome": r["judge_outcome"]})
        rows.append({"arm": "verifier", "outcome": r["verifier_outcome"]})
    summary = summarize(rows)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "n_pairs": len(items) // 2,
                   "summary": summary, "records": records},
                  f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
