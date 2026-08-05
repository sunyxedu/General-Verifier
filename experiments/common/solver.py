"""Shared solver runners for the benchmark experiments.

Three ways to run a solver on ONE prompt in ONE working directory:

  run_claude(...)            stock headless Claude Code
  run_codex(...)             stock headless Codex CLI
  mount_gv(...) + run_claude(..., mcp=True)
                             Claude Code with the full General Verifier
                             mounted — MCP sidecar + Pre/PostToolUse hooks +
                             the solver protocol as CLAUDE.md. The solver is
                             simply given a prompt; everything else is the
                             system being measured.

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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # the General Verifier repo
TOOLCHAIN = os.path.expanduser("~/.cache/gv-toolchain")
DEFAULT_TIMEOUT = 3600


def mcp_json(model: str | None = None) -> dict:
    args = ["run", "--project", ROOT, "python", "-m",
            "verifier.sidecar", "--state-dir", ".gv",
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
    """Write .mcp.json / .claude/settings.json / CLAUDE.md into `work`."""
    os.makedirs(os.path.join(work, ".claude"), exist_ok=True)
    with open(os.path.join(work, ".mcp.json"), "w") as f:
        json.dump(mcp_json(model), f)
    with open(os.path.join(work, ".claude", "settings.json"), "w") as f:
        json.dump(SETTINGS_JSON, f)
    with open(os.path.join(ROOT, "protocol", "solver-protocol.md")) as f:
        protocol = f.read()
    with open(os.path.join(work, "CLAUDE.md"), "w") as f:
        f.write(protocol + ("\n\n" + claude_md_extra if claude_md_extra
                            else ""))


def run_claude(work: str, prompt: str, model: str = "sonnet",
               mcp: bool = False, timeout: int = DEFAULT_TIMEOUT,
               base_url: str | None = None):
    """Headless Claude Code. Returns (stdout, seconds).

    `base_url` points ANTHROPIC_BASE_URL at a proxy (e.g. TurboAgent).
    """
    cmd = ["claude", "-p", prompt, "--model", model,
           "--dangerously-skip-permissions"]
    if mcp:
        cmd += ["--mcp-config", ".mcp.json", "--strict-mcp-config"]
    env = None
    if base_url:
        env = dict(os.environ, ANTHROPIC_BASE_URL=base_url)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                              timeout=timeout, env=env)
        out = proc.stdout
    except subprocess.TimeoutExpired:
        out = "<timeout>"
    return out, round(time.time() - t0, 1)


def run_codex(work: str, prompt: str, model: str | None = None,
              timeout: int = DEFAULT_TIMEOUT):
    """Headless Codex CLI. Returns (stdout, seconds)."""
    cmd = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
           "--skip-git-repo-check", prompt]
    if model:
        cmd[2:2] = ["-m", model]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                              timeout=timeout)
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
