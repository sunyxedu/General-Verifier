"""Terminal-Bench 2.1 — run a solver arm via the official Harbor harness.

Arms (identical task subset from tasks.txt, resumable by job name):
  --arm plain    stock claude-code agent (Harbor built-in)
  --arm codex    stock codex agent (Harbor built-in)
  --arm gv       gv_agent:GVClaudeCode — stock claude-code + the full
                 General Verifier mounted in the container (see gv_agent.py)
  --arm turbo    stock claude-code agent + TurboAgent, the official
                 LLM-as-a-Verifier Claude Code plugin, as shipped: the
                 proxy runs on the host and the in-container Claude Code
                 reaches it via ANTHROPIC_BASE_URL=host.docker.internal
                 (needs VERTEX_API_KEY; generations use the built-in
                 DeepSeek key)
  --arm rec      gv_agent:RecClaudeCode — stock claude-code + recording
                 hooks only (no MCP, no protocol); /gv-state is pulled as
                 an artifact for offline ORM scoring (score.py)
  --arm rec-codex
                 stock codex agent on the user's ChatGPT subscription
                 (CODEX_FORCE_AUTH_JSON uploads the host auth.json), fully
                 vanilla — no GV payload in the container at all. Codex
                 writes its own rollout log, which harbor copies to
                 trial_dir/agent/sessions/; score.py converts it into the
                 ledger format (verifier.stream.codex) and scores it.
                 Default model gpt-5.6-luna.

Scoring is Harbor's own (task verifiers write reward 0/1). The GV arm also
downloads /gv-state as a trial artifact so certificate verdicts can be
compared with rewards in summarize.py.

Requires: docker running, `uv tool install harbor` (and codex auth for the
codex arm). Claude arms run on DeepSeek-V4-Flash via verifier/llm/provider.

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
sys.path.insert(0, os.path.dirname(HERE))

from common import turbo                                 # noqa: E402
from verifier.llm import provider                        # noqa: E402

DATASET = "terminal-bench/terminal-bench-2-1"
JOBS_DIR = os.path.join(HERE, "results", "jobs")


def read_tasks():
    with open(os.path.join(HERE, "tasks.txt")) as f:
        return [ln.strip() for ln in f
                if ln.strip() and not ln.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["plain", "gv", "turbo", "codex", "rec",
                             "rec-codex"])
    ap.add_argument("--k", type=int, default=1, help="attempts per task")
    ap.add_argument("--candidates", type=int, default=3,
                    help="turbo: parallel candidates per model call")
    ap.add_argument("--model", default=None,
                    help="harbor model id (default: anthropic/sonnet for "
                    "claude arms; codex CLI default for codex)")
    ap.add_argument("--n-concurrent", type=int, default=2)
    ap.add_argument("--job-name", default=None)
    ap.add_argument("--task", default=None,
                    help="run only this task from tasks.txt (e.g. fix-git); "
                    "gets its own job name so it never collides with a full "
                    "run's resumable job")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = read_tasks()
    if args.task:
        if args.task not in tasks:
            sys.exit(f"task {args.task!r} not in tasks.txt ({len(tasks)} "
                     "tasks); add it there first")
        tasks = [args.task]
    job = args.job_name or (f"{args.arm}-{args.task}" if args.task
                            else f"{args.arm}")
    cmd = ["harbor", "run", "-d", DATASET, "-o", JOBS_DIR,
           "--job-name", job, "-k", str(args.k),
           "-n", str(args.n_concurrent)]
    for t in tasks:
        # registry task names are org-prefixed: terminal-bench/<task>
        cmd += ["-i", f"terminal-bench/{t}"]

    # DeepSeek env on the host; harbor forwards ANTHROPIC_BASE_URL and the
    # key (as ANTHROPIC_API_KEY — the endpoint accepts x-api-key) into the
    # container, and the endpoint maps claude-* aliases to deepseek-v4-flash
    # server-side, so `-m anthropic/sonnet` lands on DeepSeek too.
    env = provider.claude_env()
    if args.arm == "plain":
        cmd += ["-a", "claude-code", "-m", args.model or "anthropic/sonnet"]
    elif args.arm == "codex":
        cmd += ["-a", "codex"]
        if args.model:
            cmd += ["-m", args.model]
    elif args.arm == "turbo":
        # stock claude-code agent; every model call goes through the
        # host-side TurboAgent proxy (harbor forwards ANTHROPIC_BASE_URL
        # into the container). Model must be un-prefixed so the claude CLI
        # resolves it itself; the proxy substitutes its backend model anyway.
        cmd += ["-a", "claude-code", "-m", args.model or "sonnet",
                "--agent-timeout-multiplier", "3"]
    elif args.arm == "rec-codex":
        # stock codex agent, ChatGPT-subscription auth: harbor uploads the
        # host ~/.codex/auth.json into the container's CODEX_HOME and
        # deletes it afterwards, so host token state is untouched. No GV
        # payload at all — codex's own rollout log IS the ledger, and
        # harbor already copies it to trial_dir/agent/sessions/.
        cmd += ["-a", "codex", "-m", args.model or "gpt-5.6-luna",
                "--agent-timeout-multiplier", "1000",
                "--agent-setup-timeout-multiplier", "100"]
        env["CODEX_FORCE_AUTH_JSON"] = "1"
        env.pop("OPENAI_API_KEY", None)      # subscription auth, not a key
    elif args.arm == "rec":
        cmd += ["-a", "gv_agent:RecClaudeCode",
                "-m", args.model or "anthropic/sonnet",
                "--agent-timeout-multiplier", "1000",
                "--agent-setup-timeout-multiplier", "100",
                # pull the ledger out for offline scoring (score.py)
                "--artifact", "/gv-state"]
        env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
    else:  # gv
        cmd += ["-a", "gv_agent:GVClaudeCode",
                "-m", args.model or "anthropic/sonnet",
                # effectively no agent time limit (user directive): harbor
                # only offers multipliers over the per-task official base,
                # so use one large enough to never bind (900s base -> ~10d)
                "--agent-timeout-multiplier", "1000",
                "--agent-setup-timeout-multiplier", "100",
                # pull the verifier state out for verdict-vs-reward analysis
                "--artifact", "/gv-state"]
        env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")

    print("[tb] " + " ".join(cmd), flush=True)
    if args.dry_run:
        return
    os.makedirs(JOBS_DIR, exist_ok=True)
    if args.arm == "turbo":
        turbo.check_keys()
        turbo_dir = os.path.join(HERE, "results", "turbo")
        turbo.write_config(turbo_dir,
                           solver_model=(args.model or "sonnet").split("/")[-1],
                           num_candidates=args.candidates)
        with turbo.Proxy(turbo_dir) as base_url:
            port = base_url.rsplit(":", 1)[1]
            env["ANTHROPIC_BASE_URL"] = f"http://host.docker.internal:{port}"
            proc = subprocess.run(cmd, env=env)
    else:
        proc = subprocess.run(cmd, env=env)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
