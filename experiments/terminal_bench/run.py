"""Terminal-Bench 2.1 — run a solver arm via the official Harbor harness.

Arms (identical task subset from tasks.txt, resumable by job name):
  --arm plain    stock claude-code agent (Harbor built-in)
  --arm codex    stock codex agent (Harbor built-in)
  --arm gv       gv_agent:GVClaudeCode — stock claude-code + the full
                 General Verifier mounted in the container (see gv_agent.py)

Scoring is Harbor's own (task verifiers write reward 0/1). The GV arm also
downloads /gv-state as a trial artifact so certificate verdicts can be
compared with rewards in summarize.py.

Requires: docker running, `uv tool install harbor`, ANTHROPIC_API_KEY (and
codex auth for the codex arm).

Usage:
  uv run python experiments/terminal_bench/run.py --arm plain --k 3
  uv run python experiments/terminal_bench/run.py --arm gv
  uv run python experiments/terminal_bench/run.py --arm codex
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = "terminal-bench/terminal-bench-2-1"
JOBS_DIR = os.path.join(HERE, "results", "jobs")


def read_tasks():
    with open(os.path.join(HERE, "tasks.txt")) as f:
        return [ln.strip() for ln in f
                if ln.strip() and not ln.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["plain", "gv", "codex"])
    ap.add_argument("--k", type=int, default=1, help="attempts per task "
                    "(use >1 on plain for best-of-N selection)")
    ap.add_argument("--model", default=None,
                    help="harbor model id (default: anthropic/sonnet for "
                    "claude arms; codex CLI default for codex)")
    ap.add_argument("--n-concurrent", type=int, default=2)
    ap.add_argument("--job-name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = read_tasks()
    job = args.job_name or f"{args.arm}"
    cmd = ["harbor", "run", "-d", DATASET, "-o", JOBS_DIR,
           "--job-name", job, "-k", str(args.k),
           "-n", str(args.n_concurrent)]
    for t in tasks:
        # registry task names are org-prefixed: terminal-bench/<task>
        cmd += ["-i", f"terminal-bench/{t}"]

    env = dict(os.environ)
    if args.arm == "plain":
        cmd += ["-a", "claude-code", "-m", args.model or "anthropic/sonnet"]
    elif args.arm == "codex":
        cmd += ["-a", "codex"]
        if args.model:
            cmd += ["-m", args.model]
    else:  # gv
        cmd += ["-a", "gv_agent:GVClaudeCode",
                "-m", args.model or "anthropic/sonnet",
                # GV verification is slow by design; give it room
                "--agent-timeout-multiplier", "4",
                "--agent-setup-timeout-multiplier", "4",
                # pull the verifier state out for verdict-vs-reward analysis
                "--artifact", "/gv-state"]
        env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")

    print("[tb] " + " ".join(cmd), flush=True)
    if args.dry_run:
        return
    os.makedirs(JOBS_DIR, exist_ok=True)
    proc = subprocess.run(cmd, env=env)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
