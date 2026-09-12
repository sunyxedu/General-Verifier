#!/usr/bin/env python3
"""Summarize warm treatment batches, read-only.

Each batch output holds one usage ledger for every candidate of a task
family; attempts carry the candidate's state directory, so per-candidate
cost is exact. Optionally pair each candidate with a cold run's control job
of the same candidate id for ratios, and with truth labels.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

try:
    from . import summarize_full_offline_comparison as cold
except ImportError:  # direct script execution
    import summarize_full_offline_comparison as cold


def summarize(run_root: Path):
    batches = []
    for path in sorted((run_root / "worker-outputs").glob("*/worker-result.json")):
        output = path.parent
        result = json.loads(path.read_text())
        rows = [row for row in cold._rows(output / "native-usage.jsonl")
                if row.get("record_type") in (None, "codex_cli_attempt")]
        by_state = collections.defaultdict(list)
        for row in rows:
            by_state[row.get("state_dir")].append(row)
        candidates = []
        for item in result["candidates"]:
            state = str(output / item["candidate_id"] / "state")
            cost, complete, by_role, wall_by_role, calls = cold._cost(by_state.get(state, []))
            features = item.get("features") or {}
            events = item.get("event_counts") or {}
            candidates.append({
                "candidate": item["candidate_id"], "wall": item.get("wall_seconds"),
                "cost": cost, "cost_complete": complete,
                "calls": sum(calls.values()), "verdict": features.get("verdict"),
                "final_obligations": features.get("final_obligations"),
                "failures": item.get("failures"),
                "contract_cache_hit": events.get("contract_cache_hit", 0),
                "method_cache_hit": events.get("method_cache_hit", 0),
                "redos": events.get("method_redo", 0),
                "by_role": by_role,
            })
        batches.append({
            "task_family": result["plan"]["task_family"],
            "batch_wall": result.get("batch_wall_seconds"),
            "complete": result.get("complete"), "failures": result.get("failures"),
            "accounting_complete": (result.get("accounting") or {}).get("accounting_complete"),
            "candidates": candidates,
        })
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--control-root")
    parser.add_argument("--labels")
    args = parser.parse_args()
    labels = {}
    if args.labels:
        manifest = json.loads(Path(args.labels).read_text())
        labels = {case["candidate_id"]: case["truth"]["label"]
                  for case in manifest["cases"]}
    controls = {}
    if args.control_root:
        for job in cold.summarize(Path(args.control_root), labels):
            if job["arm"] == "control":
                controls[job["candidate"]] = job
    print(f"{'candidate':34s} {'wall':>6s} {'cost':>7s} {'calls':>5s} {'ccache':>6s} "
          f"{'mcache':>6s} {'redo':>4s} {'verdict':10s} {'label':>5s}  control (cost/wall)  ratios")
    total_cost = total_wall = ctrl_cost = ctrl_wall = 0.0
    for batch in summarize(Path(args.run_root)):
        print(f"-- {batch['task_family']}: batch wall {batch['batch_wall']:.0f}s, "
              f"complete={batch['complete']} accounting={batch['accounting_complete']} "
              f"failures={batch['failures']}")
        for item in batch["candidates"]:
            control = controls.get(item["candidate"])
            ratio = ""
            if control and control["cost_complete"] and item["cost_complete"] and item["wall"]:
                ratio = (f"${control['cost']:.3f}/{control['wall']:.0f}s  "
                         f"x{control['cost'] / item['cost']:.2f}/x{control['wall'] / item['wall']:.2f}")
                total_cost += item["cost"]; total_wall += item["wall"]
                ctrl_cost += control["cost"]; ctrl_wall += control["wall"]
            print(f"{item['candidate']:34s} {(item['wall'] or 0):6.0f} {item['cost']:7.3f} "
                  f"{item['calls']:5d} {item['contract_cache_hit']:6d} {item['method_cache_hit']:6d} "
                  f"{item['redos']:4d} {str(item['verdict']):10s} "
                  f"{str(labels.get(item['candidate'])):>5s}  {ratio}")
    if ctrl_cost:
        print(f"\npaired totals: cost ${ctrl_cost:.3f} -> ${total_cost:.3f} (x{ctrl_cost / total_cost:.2f}); "
              f"latency {ctrl_wall:.0f}s -> {total_wall:.0f}s (x{ctrl_wall / total_wall:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
