"""Terminal-Bench 2.1 — run a solver arm via the official Harbor harness.

Arms (identical task subset from tasks.txt, resumable by job name):
  --arm plain    stock claude-code agent (Harbor built-in)
  --arm codex    stock Harbor codex behavior through the repository's
                 minimal service-tier subclass
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
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(EXPERIMENTS)
for _path in (REPO_ROOT, EXPERIMENTS, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common import harbor_task_artifacts, turbo          # noqa: E402
from verifier.llm import provider                        # noqa: E402

DATASET = "terminal-bench/terminal-bench-2-1"
# Every Claude Code arm (plain / rec / gv / turbo) runs headless under
# `claude --print`: nobody is there to answer a question or approve a
# plan, and a run that stops to ask ends with the task undone. The stock
# agent does not say so, and the solver has been seen stopping at "Want
# me to proceed?" on a merge conflict. One arm-agnostic system note fixes
# the framing without touching any task's instruction.
UNATTENDED_NOTE = (
    "You are running unattended in a non-interactive session: no one can "
    "answer questions, approve plans, or resolve choices for you. Make the "
    "reasonable decisions yourself, carry the task through to completion, "
    "and only stop when it is done."
)
JOBS_DIR = os.path.join(HERE, "results", "jobs")
CODEX_AGENT = "codex_agent:TieredCodex"
RUNTIME_SNAPSHOT_ROOT = "/gv-runtime-snapshot"


def _login_credentials():
    """The interactive login's OAuth record: the macOS Keychain entry the
    CLI maintains (the file copy in the config dir goes stale once the
    CLI refreshes through the Keychain), else the file."""
    import subprocess
    try:
        raw = subprocess.run(
            ["security", "find-generic-password", "-s",
             "Claude Code-credentials", "-w"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        if raw:
            return json.loads(raw)["claudeAiOauth"]
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        pass
    path = os.path.join(os.environ.get("GV_CLAUDE_CONFIG_DIR")
                        or os.path.expanduser("~/.claude"),
                        ".credentials.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["claudeAiOauth"]


def login_access_token(min_minutes_left: float = 75.0) -> str:
    """The login's current access token, guaranteed to have at least
    ``min_minutes_left`` of life: the CLI rotates access tokens on an
    eight-hour cycle and refreshes only on demand, so when the remaining
    life is too short for a trial this waits for the expiry, triggers the
    refresh (a trivial headless call), and returns the rotated token. A
    trial that outlives its token fails with 401 mid-way, which no
    retry can repair — hence the wait."""
    import subprocess
    import time
    while True:
        creds = _login_credentials()
        left = creds["expiresAt"] / 1000 - time.time()
        if left >= min_minutes_left * 60:
            return creds["accessToken"]
        if left <= 0:
            subprocess.run(
                ["claude", "-p", "Reply with exactly: OK", "--model",
                 "claude-sonnet-5", "--output-format", "json"],
                capture_output=True, text=True, timeout=180, cwd="/tmp")
        else:
            print(f"[tb] login token has {left / 60:.0f} min left; waiting "
                  f"for the CLI to rotate it before starting", flush=True)
            time.sleep(min(60, left + 5))


def read_tasks(path=None):
    path = path or os.path.join(HERE, "tasks.txt")
    with open(path) as f:
        return [ln.strip() for ln in f
                if ln.strip() and not ln.startswith("#")]


def codex_service_agent_kwargs(service_tier):
    """Map the user-facing tier to two explicit Codex CLI controls."""
    if service_tier == "fast":
        return ["--ak", "service_tier=fast", "--ak", "fast_mode=enable"]
    if service_tier == "standard":
        return ["--ak", "service_tier=default", "--ak", "fast_mode=disable"]
    raise ValueError(f"unknown Codex service tier: {service_tier!r}")


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
    ap.add_argument("--effort", default=None,
                    choices=["low", "medium", "high", "xhigh", "max"],
                    help="reasoning effort for codex/rec-codex arms")
    ap.add_argument(
        "--service-tier", default=None, choices=["standard", "fast"],
        help=("Codex service tier for codex/rec-codex arms (default: "
              "standard; fast enables Codex Fast mode)"),
    )
    ap.add_argument("--dataset", default=DATASET,
                    help="Harbor dataset id; may include an @sha256 digest")
    ap.add_argument("--tasks-file", default=os.path.join(HERE, "tasks.txt"),
                    help="newline-delimited task list (default: tasks.txt)")
    ap.add_argument("--n-concurrent", type=int, default=2)
    ap.add_argument("--max-retries", type=int, default=None,
                    help="Harbor retry count for failed trials")
    ap.add_argument("--artifact-root", action="append", default=None,
                    help="final-world root to collect for rec/rec-codex; may "
                         "be repeated (default: /app, plus /gv-state for rec). "
                         "Safe task-referenced paths are always added.")
    ap.add_argument("--job-name", default=None)
    ap.add_argument("--task", default=None, nargs="+",
                    help="run only these tasks from tasks.txt (e.g. fix-git); "
                    "gets its own job name so it never collides with a full "
                    "run's resumable job")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.effort and args.arm not in ("codex", "rec-codex"):
        ap.error("--effort is only valid for codex/rec-codex arms")
    if args.service_tier and args.arm not in ("codex", "rec-codex"):
        ap.error("--service-tier is only valid for codex/rec-codex arms")
    if args.artifact_root and args.arm not in ("rec", "rec-codex"):
        ap.error("--artifact-root is only valid for rec/rec-codex arms")
    tasks = read_tasks(args.tasks_file)
    service_tier = args.service_tier or "standard"
    if args.task:
        missing = [t for t in args.task if t not in tasks]
        if missing:
            sys.exit(f"tasks {missing} not in {args.tasks_file} "
                     f"({len(tasks)} tasks); add them there first")
        tasks = list(args.task)
    job = args.job_name or (f"{args.arm}-{'-'.join(args.task)}" if args.task
                            else f"{args.arm}")

    # Harbor 0.20 exposes artifacts at job scope, not per task.  Resolve the
    # exact selected dataset packages before the job and pass the safe union of
    # paths named by their public instruction/config.  This avoids unreliable
    # broad captures such as /etc while retaining /app as the explicit default.
    artifact_roots = tuple(args.artifact_root or ["/app"])
    if args.arm in ("rec", "rec-codex"):
        try:
            artifact_resolution = (
                harbor_task_artifacts.resolve_harbor_dataset_artifacts(
                    args.dataset,
                    [f"terminal-bench/{task}" for task in tasks],
                    artifact_roots,
                    download=not args.dry_run,
                )
            )
        except harbor_task_artifacts.ArtifactDiscoveryError as exc:
            sys.exit(f"[tb] artifact discovery failed: {exc}")
        args.dataset = artifact_resolution.dataset
        artifact_roots = artifact_resolution.artifacts
        print(f"[tb] resolved dataset: {artifact_resolution.dataset}", flush=True)
        for task, paths in artifact_resolution.referenced_by_task.items():
            if paths:
                print(f"[tb] task artifacts {task}: {', '.join(paths)}", flush=True)
        for task, records in artifact_resolution.audit_by_task.items():
            for record in records:
                if record.decision != "rejected":
                    continue
                print(
                    f"[tb] rejected artifact {task} {record.source}: "
                    f"{record.raw!r} ({record.reason})",
                    flush=True,
                )
        print(f"[tb] collected artifact union: {', '.join(artifact_roots)}", flush=True)

    cmd = ["harbor", "run", "-d", args.dataset, "-o", JOBS_DIR,
           "--job-name", job, "-k", str(args.k),
           "-n", str(args.n_concurrent)]
    if args.max_retries is not None:
        cmd += ["--max-retries", str(args.max_retries)]
    for t in tasks:
        # registry task names are org-prefixed: terminal-bench/<task>
        cmd += ["-i", f"terminal-bench/{t}"]

    # DeepSeek env on the host; harbor forwards ANTHROPIC_BASE_URL and the
    # key (as ANTHROPIC_API_KEY — the endpoint accepts x-api-key) into the
    # container, and the endpoint maps claude-* aliases to deepseek-v4-flash
    # server-side, so `-m anthropic/sonnet` lands on DeepSeek too.
    env = (dict(os.environ) if args.arm in ("codex", "rec-codex")
           else provider.claude_env())
    claude_model = args.model or (f"anthropic/{provider.MODEL}"
                                  if provider.PROVIDER == "anthropic"
                                  else "anthropic/sonnet")
    if args.arm not in ("codex", "rec-codex") and provider.PROVIDER == "anthropic":
        # first-party subscription auth: harbor forwards the OAuth token
        # into the container (CLAUDE_FORCE_OAUTH mirrors codex's auth
        # upload). Without a `claude setup-token` token the interactive
        # login's rotating access token is used and refreshed per trial.
        env["CLAUDE_FORCE_OAUTH"] = "1"
        if not env.get("CLAUDE_CODE_OAUTH_TOKEN"):
            env["CLAUDE_CODE_OAUTH_TOKEN"] = login_access_token()
            env["GV_OAUTH_FROM_LOGIN"] = "1"
            print("[tb] no CLAUDE_CODE_OAUTH_TOKEN: using the Claude Code "
                  "login's access token (rotates; `claude setup-token` "
                  "gives a long-lived one)", flush=True)
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
    if args.arm == "plain":
        cmd += ["-a", "claude-code", "-m", claude_model]
    elif args.arm == "codex":
        cmd += ["-a", CODEX_AGENT]
        if args.model:
            cmd += ["-m", args.model]
        if args.effort:
            cmd += ["--ak", f"reasoning_effort={args.effort}"]
        cmd += codex_service_agent_kwargs(service_tier)
        env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
        # same subscription auth as rec-codex: harbor uploads ~/.codex/auth.json
        # into the container; an absent OPENAI_API_KEY otherwise becomes 401
        env["CODEX_FORCE_AUTH_JSON"] = "1"
        env.pop("OPENAI_API_KEY", None)
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
        cmd += ["-a", CODEX_AGENT,
                "-m", args.model or "gpt-5.6-luna",
                "--agent-timeout-multiplier", "1000",
                "--agent-setup-timeout-multiplier", "100"]
        # Capture declared final-world roots so offline obligations can be
        # checked against delivered state. Harbor records each attempt in its
        # artifact manifest and treats a missing root as a best-effort miss.
        for root in dict.fromkeys((*artifact_roots, RUNTIME_SNAPSHOT_ROOT)):
            cmd += ["--artifact", root]
        if args.effort:
            cmd += ["--ak", f"reasoning_effort={args.effort}"]
        cmd += codex_service_agent_kwargs(service_tier)
        env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
        env["CODEX_FORCE_AUTH_JSON"] = "1"
        env.pop("OPENAI_API_KEY", None)      # subscription auth, not a key
    elif args.arm == "rec":
        cmd += ["-a", "gv_agent:RecClaudeCode",
                "-m", claude_model,
                "--agent-timeout-multiplier", "1000",
                "--agent-setup-timeout-multiplier", "100",
                # The recorder ledger is always required. Additional roots
                # are the captured delivered world.
                "--artifact", "/gv-state"]
        for root in artifact_roots:
            cmd += ["--artifact", root]
        env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
    else:  # gv
        cmd += ["-a", "gv_agent:GVClaudeCode",
                "-m", claude_model,
                # effectively no agent time limit (user directive): harbor
                # only offers multipliers over the per-task official base,
                # so use one large enough to never bind (900s base -> ~10d)
                "--agent-timeout-multiplier", "1000",
                "--agent-setup-timeout-multiplier", "100",
                # pull the verifier state out for verdict-vs-reward analysis
                "--artifact", "/gv-state"]
        env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")

    if args.arm not in ("codex", "rec-codex"):
        cmd += ["--ak", f"append_system_prompt={UNATTENDED_NOTE}"]
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
