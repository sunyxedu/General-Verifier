"""Launch an interactive Claude Code session with the General Verifier
mounted in the CURRENT directory.

Usage:
  gv [claude args...]                   # via the ~/.local/bin/gv shim
  uv run python gv_tui.py [claude args...]

Mounts the harness for the cwd and launches `claude` there on the provider
configured in .env (see verifier/llm/provider.py) — the same setup the
experiment arms use, but interactive. Extra arguments are forwarded to
claude verbatim (e.g. `gv -c` to continue the previous session).

Unlike the experiment mount, nothing is written into the directory itself
except the `.gv/` state dir: the MCP config and hook settings live under
`.gv/` and are passed to claude by absolute path, and the solver protocol
is injected with --append-system-prompt instead of overwriting a CLAUDE.md
the project may already have. The project's own CLAUDE.md / settings (if
any) still load normally alongside the harness. No pre-task baseline is
recorded (unlike the experiment mount): completion claims are compiled
from the claim text alone.

Must be run OUTSIDE this repo (otherwise Claude Code adopts the repo as
the project and the verifier tree leaks into the solver context). First
launch on a fresh CLAUDE_CONFIG_DIR shows the one-time onboarding (theme
pick); that's expected.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import solver  # noqa: E402


def main():
    work = os.getcwd()
    if os.path.commonpath([os.path.realpath(work), ROOT]) == ROOT:
        sys.exit("gv must be run outside the General Verifier repo "
                 "(claude would adopt the repo as its project)")
    if os.path.realpath(work) in (os.path.realpath(os.path.expanduser("~")), os.sep):
        sys.exit("gv mounts the CURRENT directory as the project — claude "
                 "would adopt your entire home directory as its workspace. "
                 "cd into a project directory first.")

    state = os.path.join(work, ".gv")
    os.makedirs(state, exist_ok=True)
    mcp_path = os.path.join(state, "mcp.json")
    settings_path = os.path.join(state, "settings.json")
    with open(mcp_path, "w") as f:
        json.dump(solver.mcp_json(), f)
    with open(settings_path, "w") as f:
        json.dump(solver.SETTINGS_JSON, f)
    with open(os.path.join(ROOT, "protocol", "solver-protocol.md")) as f:
        protocol = f.read()

    env = solver.provider.claude_env()
    env.pop("CLAUDECODE", None)   # allow launching from inside a Claude session
    print(f"[gv] harness mounted in {work} — starting claude "
          f"({solver.provider.PROVIDER}: {solver.provider.MODEL})", flush=True)
    os.execvpe("claude", [
        "claude",
        "--mcp-config", mcp_path, "--strict-mcp-config",
        "--settings", settings_path,
        "--append-system-prompt", protocol,
        *sys.argv[1:],
    ], env)


if __name__ == "__main__":
    main()
