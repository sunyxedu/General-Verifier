"""Replay official LLM-as-a-Verifier terminal-bench traces into GV-scorable
synthetic Harbor trials.

For each official mini-swe-agent trajectory:
  1. replay the recorded bash commands into a fresh task container
     (per-step output comparison = the replay-fidelity record);
  2. capture the reconstructed /app as the trial's artifacts/ world;
  3. convert the trajectory into the hook-feed ledger dialect, extract
     task.md and final.txt;
  4. write a Harbor-layout trial dir that score.py consumes unchanged.

The reconstructed world is a *derived* input under a declared replay bridge:
scores over these trials cap at supported by provenance discipline, and each
trial dir carries replay_report.json (per-step fidelity) as the bridge's
audit trail.

Usage:
  replay_lav.py --dataset <trajs_root> --job <job_name>
                (--task t1 t2 ... | --group N)
                [--k 5] [--workers 4] [--image-tag 20251031]
  --group N reads group N's task list from official_groups.txt (generated
  splits of the official dataset; job name defaults per group when omitted).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay  # noqa: E402  (step parsing + normalization)


def convert_feed(traj: dict, feed_path: str):
    """Official steps -> hook-feed ledger dialect (the TieredCodex shape the
    extractor already reads). Commands become tools.exec_command calls."""
    session = str(uuid.uuid4())
    events = []
    n = 0
    for step in traj["trajectory"]["steps"]:
        for tc in step.get("tool_calls") or []:
            if tc.get("tool_name") != "bash":
                continue
            cmd = (tc.get("arguments") or {}).get("keystrokes") or ""
            n += 1
            tid = f"call_replay{n:04d}"
            args = json.dumps({"cmd": cmd, "workdir": "/app",
                               "yield_time_ms": 10000})
            events.append({"type": "tool_call", "payload": {
                "agent": None,
                "input": f"const r = await tools.exec_command({args});\n"
                         "text(r.output);\n",
                "session": session, "tool": "exec", "tool_use_id": tid}})
            recorded = ""
            obs = step.get("observation") or {}
            results = obs.get("results") or []
            if results:
                content = results[0].get("content") or ""
                try:
                    recorded = json.loads(content).get("output", "")
                except (json.JSONDecodeError, AttributeError):
                    recorded = content
            excerpt = json.dumps([{"text": recorded, "type": "input_text"}])
            events.append({"type": "tool_result", "payload": {
                "agent": None, "excerpt": excerpt,
                "excerpt_truncated": False,
                "response_chars": len(recorded),
                "response_sha256": hashlib.sha256(
                    recorded.encode()).hexdigest(),
                "session": session, "tool": "exec", "tool_use_id": tid}})
    with open(feed_path, "w") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def extract_problem(steps):
    for step in steps:
        if step.get("source") == "user":
            msg = step.get("message", "")
            if msg and not (msg.startswith("$") and len(msg) < 5):
                return msg
    return ""


def final_message(steps):
    for step in reversed(steps):
        if step.get("source") == "agent" and step.get("message"):
            return step["message"]
    return ""


_ERROR_MARKS = ("Traceback (most recent call last)", "ModuleNotFoundError",
                "command not found", "No such file or directory",
                "E: Unable to locate", "Could not find a version",
                "Connection refused", "Temporary failure in name resolution",
                "Read-only file system", "Permission denied")


def classify_miss(got: str, want: str) -> str:
    """Rough triage of a fidelity miss: environment failure (repair signal)
    vs. benign nondeterminism (timestamps/uuids/durations) vs. unknown."""
    if not got.strip():
        return "got_empty"
    if any(m in got for m in _ERROR_MARKS) \
            and not any(m in want for m in _ERROR_MARKS):
        return "env_error"
    lg, lw = len(got), len(want)
    if lw and 0.5 <= lg / lw <= 2.0:
        gt, wt = set(got.split()), set(want.split())
        if wt and len(gt & wt) / len(wt) >= 0.6:
            return "noise_suspect"
    return "other"


def build_trial(traj_path: str, job_dir: str, image: str) -> dict:
    traj = json.load(open(traj_path))
    trial_name = traj["trial_name"]
    task = traj["task_name"].split("/")[-1] if "/" in str(
        traj.get("task_name")) else traj["task_name"]
    trial_dir = os.path.join(job_dir, trial_name)
    if os.path.exists(os.path.join(trial_dir, "result.json")):
        return {"trial": trial_name, "status": "exists"}
    os.makedirs(os.path.join(trial_dir, "gv-state"), exist_ok=True)
    os.makedirs(os.path.join(trial_dir, "artifacts"), exist_ok=True)

    # 1. replay
    name = f"gv-replay-{uuid.uuid4().hex[:8]}"
    up = replay.docker(["run", "-d", "--name", name, image,
                        "sleep", "infinity"])
    if up.returncode:
        return {"trial": trial_name, "status": "container-failed",
                "error": up.stderr[:200]}
    steps_report = []
    try:
        for kind, payload, recorded in replay.parse_official(traj_path):
            if kind != "exec":
                steps_report.append({"kind": kind, **payload})
                continue
            try:
                # The source harness cut every command at 30s, so no recorded
                # step ever ran longer; 180s leaves headroom for cold caches.
                run = replay.docker(
                    ["exec", "-w", payload.get("workdir") or "/app", name,
                     "bash", "-lc", payload["cmd"]], timeout=180)
            except subprocess.TimeoutExpired:
                # A single overlong step (e.g. a pure-compute simulation)
                # must not kill the trial: record the infidelity and move on.
                steps_report.append({"kind": "exec", "match": False,
                                     "timeout": True,
                                     "cmd": payload["cmd"][:100]})
                continue
            got = replay.norm(run.stdout + run.stderr)
            want = replay.norm(recorded)
            match = (got == want) or (want and want in got) \
                or (got and got in want)
            if not match:
                extra = {"miss_class": classify_miss(got, want),
                         "got_head": (run.stdout + run.stderr)[:300],
                         "want_head": (recorded or "")[:300],
                         "rc": run.returncode}
            else:
                extra = {}
            steps_report.append({"kind": "exec", "match": bool(match),
                                 **extra,
                                 "cmd": payload["cmd"][:100]})
        # 2. world capture
        cp = replay.docker(["cp", f"{name}:/app",
                            os.path.join(trial_dir, "artifacts", "app")],
                           timeout=600)
        world_ok = cp.returncode == 0
    finally:
        replay.docker(["rm", "-f", name])

    execs = [s for s in steps_report if s["kind"] == "exec"]
    fidelity = {"exec_steps": len(execs),
                "exec_matched": sum(s["match"] for s in execs),
                "world_captured": world_ok, "steps": steps_report}
    with open(os.path.join(trial_dir, "replay_report.json"), "w") as f:
        json.dump(fidelity, f, ensure_ascii=False, indent=1)

    # 3. ledger + contract + final
    steps = traj["trajectory"]["steps"]
    convert_feed(traj, os.path.join(trial_dir, "gv-state",
                                    "hook_feed.jsonl"))
    with open(os.path.join(trial_dir, "gv-state", "task.md"), "w") as f:
        f.write(extract_problem(steps))
    with open(os.path.join(trial_dir, "gv-state", "final.txt"), "w") as f:
        f.write(final_message(steps))

    # 4. harbor-layout metadata
    with open(os.path.join(trial_dir, "artifacts", "manifest.json"),
              "w") as f:
        json.dump([{"source": "/app", "destination": "artifacts/app",
                    "type": "directory",
                    "status": "ok" if world_ok else "failed",
                    "service": None}], f)
    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump({"task_name": f"terminal-bench/{task}",
                   "trial_name": trial_name,
                   "source": "llm-as-a-verifier official trace (replayed)",
                   "verifier_result": {"reward": traj.get("reward", 0)}},
                  f)
    return {"trial": trial_name, "status": "ok",
            "fidelity": f"{fidelity['exec_matched']}/{fidelity['exec_steps']}",
            "world": world_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--job", help="synthetic job name "
                    "(default with --group: lav-official-g<N>-k<k>)")
    ap.add_argument("--task", nargs="+")
    ap.add_argument("--group", type=int,
                    help="run group N from official_groups.txt")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--image-tag", default="20251031")
    args = ap.parse_args()

    if bool(args.task) == bool(args.group):
        ap.error("exactly one of --task / --group is required")
    if args.group:
        groups, current = {}, None
        with open(os.path.join(HERE, "official_groups.txt")) as f:
            for line in f:
                line = line.strip()
                if line.startswith("# group"):
                    current = int(line.split()[2])
                    groups[current] = []
                elif line and current is not None:
                    groups[current].append(line)
        if args.group not in groups:
            ap.error(f"no group {args.group} in official_groups.txt "
                     f"(have {sorted(groups)})")
        args.task = groups[args.group]
        if not args.job:
            args.job = f"lav-official-g{args.group}-k{args.k}"
    if not args.job:
        ap.error("--job is required with --task")

    job_dir = os.path.join(HERE, "results", "jobs", args.job)
    os.makedirs(job_dir, exist_ok=True)
    work = []
    for task in args.task:
        trajs = sorted(glob.glob(os.path.join(
            args.dataset, task, "*_trajectory.json")))[:args.k]
        image = f"alexgshaw/{task}:{args.image_tag}"
        for t in trajs:
            work.append((t, image))
    print(f"[replay-lav] {len(work)} traces across {len(args.task)} tasks")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_trial, t, job_dir, img)
                   for t, img in work]
        for fut in futures:
            try:
                r = fut.result()
            except Exception as exc:
                print(f"[replay-lav] trial crashed: {exc!r}"[:300],
                      flush=True)
                continue
            print(f"[replay-lav] {r['trial']}: {r['status']} "
                  f"{r.get('fidelity', '')}", flush=True)


if __name__ == "__main__":
    main()
