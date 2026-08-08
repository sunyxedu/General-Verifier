"""Open an interactive Claude Code TUI with the General Verifier mounted.

Usage:
  uv run python gv_tui.py [workdir]     # default: ~/gv-workspace

Mounts the harness into `workdir` (gv MCP server + Pre/PostToolUse hooks +
the solver protocol as CLAUDE.md) and launches `claude` there on the
provider configured in .env (see verifier/llm/provider.py) — the same
setup the experiment arms use, but interactive.

The workdir must live OUTSIDE this repo (otherwise Claude Code adopts the
repo as the project and the verifier tree leaks into the solver context).
First launch on a fresh CLAUDE_CONFIG_DIR shows the one-time onboarding
(theme pick); that's expected.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import solver  # noqa: E402


def main():
    work = os.path.abspath(os.path.expanduser(
        sys.argv[1] if len(sys.argv) > 1 else "~/gv-workspace"))
    if work == ROOT or work.startswith(ROOT + os.sep):
        sys.exit("workdir must be outside the General Verifier repo "
                 "(claude would adopt the repo as its project)")
    os.makedirs(work, exist_ok=True)
    solver.mount_gv(work)

    env = solver.provider.claude_env()
    env.pop("CLAUDECODE", None)          # allow launching from inside a session
    os.chdir(work)
    print(f"[gv] harness mounted in {work} — starting claude "
          f"({solver.provider.PROVIDER}: {solver.provider.MODEL})", flush=True)
    # explicit --mcp-config skips the project-trust dialog; strict keeps the
    # session hermetic (just the gv server)
    os.execvpe("claude", ["claude", "--mcp-config", ".mcp.json",
                          "--strict-mcp-config"], env)


if __name__ == "__main__":
    main()
