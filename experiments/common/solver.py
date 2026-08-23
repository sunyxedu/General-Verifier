"""Shared solver runners for the benchmark experiments.

Three ways to run a solver on ONE prompt in ONE working directory:

  run_claude(...)            stock headless Claude Code
  run_codex(...)             stock headless Codex CLI
  mount_gv(...) + run_claude(..., mcp=True)
                             Claude Code with the full General Verifier
                             mounted — MCP server + Pre/PostToolUse hooks;
                             the solver protocol arrives as the gv server's
                             MCP instructions. The solver is simply given a
                             prompt; everything else is the system being
                             measured.

Hard-won details baked in (learned on earlier live runs):
  - the repo path contains a space — hook commands are shell-parsed, so the
    script path MUST be quoted or every tool call gets blocked by a failing
    PreToolUse hook;
  - headless `claude -p` does not auto-trust a project .mcp.json; it must be
    loaded explicitly with --mcp-config or the harness silently stays
    unmounted;
  - workdirs must live OUTSIDE this repo, otherwise Claude Code adopts the
    repo as the project and the verifier tree leaks into the solver context.
"""
import json
import os
import sqlite3
import subprocess
import time

from verifier.llm import provider   # DeepSeek wiring (single source of truth)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # the General Verifier repo
TOOLCHAIN = os.path.expanduser("~/.cache/gv-toolchain")
# We impose no time limits of our own — only the official benchmark
# harnesses do. Pass an explicit `timeout` to opt in.
DEFAULT_TIMEOUT = None


def mcp_json(model: str | None = None) -> dict:
    args = ["run", "--project", ROOT, "python", "-m",
            "verifier.mcp_server", "--state-dir", ".gv",
            "--toolchain-dir", TOOLCHAIN]
    if model:
        args += ["--model", model]
    return {"mcpServers": {"gv": {"type": "stdio", "command": "uv",
                                  "args": args}}}


SETTINGS_JSON = {
    "hooks": {
        "PreToolUse": [{"matcher": "*", "hooks": [
            {"type": "command",
             "command": f'python3 "{ROOT}/hooks/pre_tool_use.py"'}]}],
        "PostToolUse": [{"matcher": "*", "hooks": [
            {"type": "command",
             "command": f'python3 "{ROOT}/hooks/post_tool_use.py"'}]}],
    }
}


def mount_gv(work: str, model: str | None = None,
             claude_md_extra: str = "") -> None:
    """Write .mcp.json / .claude/settings.json into `work`. The solver
    protocol is NOT copied here: the gv server serves it as MCP server
    instructions, so it arrives with the mount and cannot drift."""
    os.makedirs(os.path.join(work, ".claude"), exist_ok=True)
    with open(os.path.join(work, ".mcp.json"), "w") as f:
        json.dump(mcp_json(model), f)
    with open(os.path.join(work, ".claude", "settings.json"), "w") as f:
        json.dump(SETTINGS_JSON, f)
    if claude_md_extra:
        with open(os.path.join(work, "CLAUDE.md"), "w") as f:
            f.write(claude_md_extra)


def mount_recorder(work: str) -> None:
    """Hooks-only mount for the offline ORM path: the solver stays fully
    vanilla (no MCP server, no protocol in context — its distribution is
    untouched); the recording hooks just write the ledger feed that the
    offline extractor scores after the fact."""
    os.makedirs(os.path.join(work, ".claude"), exist_ok=True)
    with open(os.path.join(work, ".claude", "settings.json"), "w") as f:
        json.dump(SETTINGS_JSON, f)


def record_final(work: str, stdout: str) -> None:
    """Persist the solver's final output where the offline extractor
    expects it (state-dir/final.txt)."""
    state = os.path.join(work, ".gv")
    os.makedirs(state, exist_ok=True)
    with open(os.path.join(state, "final.txt"), "w") as f:
        f.write(stdout)


def run_claude(work: str, prompt: str, model: str = "sonnet",
               mcp: bool = False, record: bool = False,
               timeout: int | None = DEFAULT_TIMEOUT,
               base_url: str | None = None, task: str | None = None,
               web: bool = True):
    """Headless Claude Code on the configured provider (see .env /
    provider.py). Returns (stdout, seconds).

    `base_url` points ANTHROPIC_BASE_URL at a proxy (e.g. TurboAgent)
    instead of the provider's endpoint; the rest of the provider env stays.
    `task` is the clean task statement for the verifier's task.md (defaults
    to `prompt`; pass it when the prompt carries protocol plumbing like the
    GV suffix, which is not task content).
    `record` writes the verifier ground truth (task.md + pre-run baseline)
    without mounting the MCP server — the recorder arm for offline scoring.
    `web=False` drops the WebSearch/WebFetch tools — for benchmarks whose
    answers are published online (Bash-level curl is NOT blocked here).
    """
    cmd = ["claude", "-p", prompt, "--model", model,
           "--dangerously-skip-permissions"]
    if not web:
        cmd += ["--disallowed-tools", "WebSearch,WebFetch"]
    if mcp:
        cmd += ["--mcp-config", ".mcp.json", "--strict-mcp-config"]
    if mcp or record:
        # harness ground truth for the verifier: the authoritative task
        # statement + a pre-task snapshot (obligation compiler reads both;
        # the solver's paraphrase of the task is untrusted)
        from verifier import baseline
        state = os.path.join(work, ".gv")
        os.makedirs(state, exist_ok=True)
        with open(os.path.join(state, "task.md"), "w") as f:
            f.write(task or prompt)
        baseline.snapshot(work, os.path.join(state, "baseline"))
    env = provider.claude_env()
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                              timeout=timeout or None, env=env)
        out = proc.stdout
    except subprocess.TimeoutExpired:
        out = "<timeout>"
    return out, round(time.time() - t0, 1)


def rollout_for(workdir: str, since: float) -> str | None:
    """The newest codex rollout under ~/.codex/sessions recorded in
    `workdir` (session_meta cwd match, realpath'd both sides — /tmp is a
    symlink on macOS) and modified after `since`. cwd-exact pairing keeps
    concurrent solver runs — and any verifier seats sharing the same codex
    install — from cross-capturing each other's sessions."""
    sessions = os.path.expanduser("~/.codex/sessions")
    if not os.path.isdir(sessions):
        return None
    target = os.path.realpath(workdir)
    best, best_mtime = None, -1.0
    for root, _dirs, files in os.walk(sessions):
        for name in files:
            if not (name.startswith("rollout-") and name.endswith(".jsonl")):
                continue
            path = os.path.join(root, name)
            mtime = os.path.getmtime(path)
            if mtime < since or mtime <= best_mtime:
                continue
            if _rollout_cwd(path) == target:
                best, best_mtime = path, mtime
    return best


def _rollout_cwd(path: str) -> str | None:
    """cwd from a rollout's leading session_meta lines (None if absent)."""
    try:
        with open(path, encoding="utf-8") as f:
            for _ in range(8):
                line = f.readline()
                if not line:
                    return None
                try:
                    payload = json.loads(line).get("payload") or {}
                except json.JSONDecodeError:
                    continue
                if payload.get("cwd"):
                    return os.path.realpath(payload["cwd"])
    except OSError:
        pass
    return None


# run_codex(web=False): every channel below verified closed on codex-cli
# 0.147 (the flags are version-sensitive — re-verify after codex updates):
#   shell network   — workspace-write sandbox + network_access=false
#                     (curl: could not resolve host)
#   web search tool — top-level `web_search="disabled"` is the key that
#                     works; tools.web_search=false and web_search_mode
#                     do NOT remove the tool. Provider-side execution, so
#                     the local sandbox alone never blocks it.
#   MCP / apps      — mcp_servers={} + --disable enable_mcp_apps, else
#                     connector search tools (Notion, OpenAI docs) leak
#                     through as an out-of-band lookup channel.
#   plugins         — --ignore-user-config, else the openai-bundled browser
#                     plugin ships a repl__js + browser runtime the model
#                     WILL find and use to reopen a web channel (observed
#                     live: "Look up five-conics monodromy" — each attempt
#                     crash-looped Chrome against the seatbelt sandbox).
NO_WEB_CODEX_FLAGS = [
    "--ignore-user-config",
    "--sandbox", "workspace-write",
    "-c", "sandbox_workspace_write.network_access=false",
    "-c", 'web_search="disabled"',
    "-c", "approval_policy=never",
    "-c", "mcp_servers={}",
    "--disable", "enable_mcp_apps",
]


def run_codex(work: str, prompt: str, model: str | None = None,
              timeout: int | None = DEFAULT_TIMEOUT, web: bool = True):
    """Headless Codex CLI. Returns (stdout, seconds).

    `web=False` cuts every network channel (see NO_WEB_CODEX_FLAGS) — for
    benchmarks whose answers are published online: on an early FrontierMath
    run the solver fetched the benchmark paper and the epoch-research
    solution repo mid-task."""
    access = (NO_WEB_CODEX_FLAGS if not web
              else ["--dangerously-bypass-approvals-and-sandbox"])
    cmd = ["codex", "exec", *access, "--skip-git-repo-check", prompt]
    if model:
        cmd[2:2] = ["-m", model]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                              timeout=timeout or None)
        out = proc.stdout
    except subprocess.TimeoutExpired:
        out = "<timeout>"
    return out, round(time.time() - t0, 1)


def gv_outcome(work: str):
    """Read the GV state a solver left behind in `work`.

    Returns dict: final claim text/status, refuted-claim count, event count.
    All None/0 when the solver never engaged the harness.
    """
    state = os.path.join(work, ".gv")
    out = {"final_text": None, "verdict": None, "refuted_claims": 0,
           "events": 0}
    db_path = os.path.join(state, "claims.sqlite")
    if os.path.exists(db_path):
        db = sqlite3.connect(db_path)
        row = db.execute(
            "SELECT text, status FROM claims WHERE kind='final' LIMIT 1"
        ).fetchone()
        if row:
            out["final_text"], out["verdict"] = row
        (out["refuted_claims"],) = db.execute(
            "SELECT COUNT(*) FROM claims WHERE status='refuted' "
            "AND kind != 'obligation'").fetchone()
        db.close()
    ev = os.path.join(state, "events.jsonl")
    if os.path.exists(ev):
        out["events"] = sum(1 for line in open(ev) if line.strip())
    return out
