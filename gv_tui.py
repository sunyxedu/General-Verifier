"""Launch an interactive Claude Code session with the General Verifier
mounted in the CURRENT directory.

Usage:
  gv [--cyber] [claude args...]         # via the ~/.local/bin/gv shim
  uv run python gv_tui.py [claude args...]

Mounts the harness for the cwd and launches `claude` there on the provider
configured in .env (see verifier/llm/provider.py) — the same setup the
experiment arms use, but interactive. Extra arguments are forwarded to
claude verbatim (e.g. `gv -c` to continue the previous session).

Unlike the experiment mount, nothing is written into the directory itself
except the `.gv/` state dir: the MCP config and hook settings live under
`.gv/` and are passed to claude by absolute path, and the solver protocol
arrives as the gv server's MCP instructions — nothing to inject, and no
CLAUDE.md the project may already have gets touched. The project's own
CLAUDE.md / settings (if any) still load normally alongside the harness. No pre-task baseline is
recorded (unlike the experiment mount): completion claims are compiled
from the claim text alone.

Must be run OUTSIDE this repo (otherwise Claude Code adopts the repo as
the project and the verifier tree leaks into the solver context). First
launch on a fresh CLAUDE_CONFIG_DIR shows the one-time onboarding (theme
pick); that's expected.
"""
import json
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import solver  # noqa: E402


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cyber", action="store_true")
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--verifier-seats", choices=("sdk", "codex"), default="sdk")
    parser.add_argument("--verifier-effort", default=None)
    parser.add_argument("--verifier-service-tier", choices=("standard", "fast"), default=None)
    parser.add_argument("--cyber-final-destination", default=None)
    parser.add_argument("--cyber-evaluator-command-json", default=None)
    parser.add_argument("--cyber-evaluator-pin", action="append", default=None)
    args, claude_args = parser.parse_known_args()
    try:
        evaluator_command = (json.loads(args.cyber_evaluator_command_json)
                             if args.cyber_evaluator_command_json is not None else None)
        config = solver.mcp_json(args.verifier_model,
            profile="cyber" if args.cyber else "default", seats=args.verifier_seats,
            effort=args.verifier_effort, service_tier=args.verifier_service_tier,
            final_destination=args.cyber_final_destination,
            evaluator_command=evaluator_command, evaluator_pins=args.cyber_evaluator_pin)
    except ValueError as err:
        parser.error(str(err))
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
        json.dump(config, f)
    with open(settings_path, "w") as f:
        json.dump(solver.SETTINGS_JSON, f)

    env = solver.provider.claude_env()
    env.pop("CLAUDECODE", None)   # allow launching from inside a Claude session
    print(f"[gv] harness mounted in {work} — starting claude "
          f"({solver.provider.PROVIDER}: {solver.provider.MODEL})", flush=True)
    os.execvpe("claude", [
        "claude",
        "--mcp-config", mcp_path, "--strict-mcp-config",
        "--settings", settings_path,
        *claude_args,
    ], env)


if __name__ == "__main__":
    main()
