#!/usr/bin/env python3
"""Summarize a v1 full-offline comparison run root, read-only.

Prints one row per job (cost from the native usage ledger at the pinned
rates, wall from the coordinator's spawn-to-exit measurement, the final
verdict and contract outcomes) and the paired control/treatment ratios.
Cost for an arm is complete only when every attempt has known usage.
When an operator manifest with labels is given, the verdict is also
compared with the official truth label.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

RATES = {"uncached_input_tokens": 0.20, "cached_input_tokens": 0.02,
         "cache_write_input_tokens": 0.25, "output_tokens": 1.20}
POSITIVE = {"verified", "supported"}


def _rows(path: Path):
    out = []
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _cost(rows):
    total = 0.0
    complete = True
    by_role = collections.Counter()
    wall_by_role = collections.Counter()
    calls_by_role = collections.Counter()
    for row in rows:
        if row.get("record_type") not in (None, "codex_cli_attempt"):
            continue
        role = row.get("role")
        calls_by_role[role] += 1
        wall_by_role[role] += row.get("duration_seconds") or 0.0
        if row.get("usage_complete") is False or row.get("input_tokens") is None:
            complete = False
            continue
        cost = sum((row.get(key) or 0) * rate / 1e6
                   for key, rate in RATES.items())
        total += cost
        by_role[role] += cost
    return total, complete, by_role, wall_by_role, calls_by_role


def summarize(run_root: Path, labels: dict | None):
    comparison = run_root / "comparison"
    jobs = []
    for path in sorted(comparison.glob("job-*-result.json")):
        job = json.loads(path.read_text())
        output_root = job.get("worker_output_root")
        if output_root is None:
            plan = run_root / "plans" / f"{job['candidate_id']}-{job['arm']}.json"
            output_root = json.loads(plan.read_text())["output_root"]
        output = Path(output_root)
        cost, complete, by_role, wall_by_role, calls = _cost(
            _rows(output / "native-usage.jsonl"))
        view = job.get("quality_view") or {}
        verdict = view.get("verdict")
        label = (labels or {}).get(job["candidate_id"])
        jobs.append({
            "candidate": job["candidate_id"], "arm": job["arm"],
            "wall": job.get("spawn_to_exit_wall_seconds"),
            "cost": cost, "cost_complete": complete,
            "accounting_complete": job.get("accounting_complete"),
            "worker_complete": job.get("worker_complete"),
            "calls": sum(calls.values()), "verdict": verdict,
            "final_obligations": view.get("final_obligations"),
            "label": label,
            "agrees": (None if label is None or verdict is None
                       else (verdict in POSITIVE) == (label == 1)),
            "by_role": by_role, "wall_by_role": wall_by_role,
            "calls_by_role": calls,
        })
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--labels", help="operator-manifest.json with truth labels")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    labels = None
    if args.labels:
        manifest = json.loads(Path(args.labels).read_text())
        labels = {case["candidate_id"]: case["truth"]["label"]
                  for case in manifest["cases"]}
    jobs = summarize(Path(args.run_root), labels)
    if args.json:
        print(json.dumps(jobs, indent=1, sort_keys=True, default=dict))
        return 0
    print(f"{'candidate':34s} {'arm':9s} {'wall':>7s} {'cost':>7s} {'calls':>5s} "
          f"{'verdict':10s} {'label':>5s} {'ok':>4s} complete")
    pairs = collections.defaultdict(dict)
    for job in jobs:
        pairs[job["candidate"]][job["arm"]] = job
        print(f"{job['candidate']:34s} {job['arm']:9s} "
              f"{(job['wall'] or 0):7.0f} {job['cost']:7.3f} {job['calls']:5d} "
              f"{str(job['verdict']):10s} {str(job['label']):>5s} "
              f"{str(job['agrees']):>4s} "
              f"{'yes' if job['cost_complete'] and job['worker_complete'] else 'NO'}")
    print()
    control_cost = treatment_cost = control_wall = treatment_wall = 0.0
    counted = 0
    for candidate, arms in pairs.items():
        if set(arms) != {"control", "treatment"}:
            continue
        c, t = arms["control"], arms["treatment"]
        if not (c["cost_complete"] and t["cost_complete"] and c["wall"] and t["wall"]):
            continue
        counted += 1
        control_cost += c["cost"]; treatment_cost += t["cost"]
        control_wall += c["wall"]; treatment_wall += t["wall"]
        print(f"{candidate:34s} cost x{c['cost'] / t['cost'] if t['cost'] else float('nan'):5.2f}"
              f"  latency x{c['wall'] / t['wall']:5.2f}"
              f"  verdict {c['verdict']} -> {t['verdict']}")
    if counted:
        print(f"\n{counted} complete pairs: cost ${control_cost:.3f} -> ${treatment_cost:.3f}"
              f" (x{control_cost / treatment_cost:.2f}); latency {control_wall:.0f}s ->"
              f" {treatment_wall:.0f}s (x{control_wall / treatment_wall:.2f})")
    for arm in ("control", "treatment"):
        cost = collections.Counter(); wall = collections.Counter(); calls = collections.Counter()
        for job in jobs:
            if job["arm"] == arm:
                cost.update(job["by_role"]); wall.update(job["wall_by_role"]); calls.update(job["calls_by_role"])
        print(f"\n{arm} by role: " + ", ".join(
            f"{role}: {calls[role]} calls ${cost[role]:.3f} {wall[role]:.0f}s"
            for role in ("extract", "obligations", "critic", "oracles", "discharge", "judger")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
