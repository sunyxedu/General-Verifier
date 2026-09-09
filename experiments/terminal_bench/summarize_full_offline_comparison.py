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


def _event_counts(output):
    counts = collections.Counter()
    for row in _rows(output / "state" / "events.jsonl"):
        counts[row.get("type")] += 1
        if row.get("type") == "sufficiency_audited" \
                and row.get("payload", {}).get("sufficient") == "no":
            counts["sufficiency_no"] += 1
    return counts


def _job_record(candidate, arm, output, wall, verdict, final_obligations,
                accounting_complete, worker_complete, labels):
    cost, complete, by_role, wall_by_role, calls = _cost(
        _rows(output / "native-usage.jsonl"))
    label = (labels or {}).get(candidate)
    events = _event_counts(output)
    return {
        "redos": events.get("method_redo", 0),
        "sufficiency_no": events.get("sufficiency_no", 0),
        "cache_hits": events.get("contract_cache_hit", 0)
        + events.get("method_cache_hit", 0),
        "candidate": candidate, "arm": arm, "wall": wall,
        "cost": cost, "cost_complete": complete,
        "accounting_complete": accounting_complete,
        "worker_complete": worker_complete,
        "calls": sum(calls.values()), "verdict": verdict,
        "final_obligations": final_obligations, "label": label,
        "agrees": (None if label is None or verdict is None
                   else (verdict in POSITIVE) == (label == 1)),
        "by_role": by_role, "wall_by_role": wall_by_role,
        "calls_by_role": calls,
    }


def summarize(run_root: Path, labels: dict | None):
    """Jobs from a coordinator run, or from bare worker outputs.

    A coordinator run has ``comparison/job-*-result.json`` with the
    spawn-to-exit wall.  A run of bare workers (for example a treatment-only
    rerun of a cohort) is read from ``worker-outputs/*/worker-result.json``,
    whose ``worker_wall_seconds`` excludes only interpreter start-up.
    """
    comparison = run_root / "comparison"
    jobs = []
    seen = set()
    for path in sorted(comparison.glob("job-*-result.json")):
        job = json.loads(path.read_text())
        output_root = job.get("worker_output_root")
        if output_root is None:
            plan = run_root / "plans" / f"{job['candidate_id']}-{job['arm']}.json"
            output_root = json.loads(plan.read_text())["output_root"]
        view = job.get("quality_view") or {}
        seen.add(Path(output_root).name)
        jobs.append(_job_record(
            job["candidate_id"], job["arm"], Path(output_root),
            job.get("spawn_to_exit_wall_seconds"), view.get("verdict"),
            view.get("final_obligations"), job.get("accounting_complete"),
            job.get("worker_complete"), labels))
    for path in sorted((run_root / "worker-outputs").glob("*/worker-result.json")):
        output = path.parent
        if output.name in seen:
            continue
        result = json.loads(path.read_text())
        plan = result.get("plan") or {}
        features = result.get("features") or {}
        accounting = result.get("accounting") or {}
        candidate = output.name.rsplit("-", 1)[0]
        jobs.append(_job_record(
            candidate, plan.get("arm", output.name.rsplit("-", 1)[-1]),
            output, result.get("worker_wall_seconds"),
            features.get("verdict"), features.get("final_obligations"),
            accounting.get("accounting_complete") is True,
            result.get("complete") is True, labels))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--labels", help="operator-manifest.json with truth labels")
    parser.add_argument("--control-root", help="pair with another run's control jobs")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    labels = None
    if args.labels:
        manifest = json.loads(Path(args.labels).read_text())
        labels = {case["candidate_id"]: case["truth"]["label"]
                  for case in manifest["cases"]}
    jobs = summarize(Path(args.run_root), labels)
    if args.control_root:
        jobs = [job for job in jobs if job["arm"] == "treatment"] + [
            job for job in summarize(Path(args.control_root), labels)
            if job["arm"] == "control"]
    if args.json:
        print(json.dumps(jobs, indent=1, sort_keys=True, default=dict))
        return 0
    print(f"{'candidate':34s} {'arm':9s} {'wall':>7s} {'cost':>7s} {'calls':>5s} "
          f"{'redo':>4s} {'pre':>3s} {'verdict':10s} {'label':>5s} {'ok':>4s} complete")
    pairs = collections.defaultdict(dict)
    for job in jobs:
        pairs[job["candidate"]][job["arm"]] = job
        print(f"{job['candidate']:34s} {job['arm']:9s} "
              f"{(job['wall'] or 0):7.0f} {job['cost']:7.3f} {job['calls']:5d} "
              f"{job['redos']:4d} {job['sufficiency_no']:3d} "
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
