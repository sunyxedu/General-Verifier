"""Harbor custom agent: Codex CLI with the full General Verifier mounted.

The Codex counterpart of gv_agent.GVClaudeCode. Runs inside the task
container exactly like the stock codex agent (same install, same headless
`codex exec`), plus:

  - the General Verifier codebase shipped to /opt/general-verifier and
    uv-synced (gv_agent.install_payload)
  - the gv MCP server registered in $CODEX_HOME/config.toml through
    Harbor's own mechanism; its verifier seats run through the Codex CLI
    on the same ChatGPT subscription (mcp_server --seats codex), from a
    seats-only CODEX_HOME so a seat never loads the solver's MCP servers
    or hooks
  - Pre/PostToolUse hooks in $CODEX_HOME/hooks.json feeding
    /gv-state/hook_feed.jsonl: Codex's hook payload carries the same
    fields as Claude Code's, so the same hook scripts record the ledger
  - the solver protocol installed as $CODEX_HOME/AGENTS.md, and a short
    reminder appended to the task instruction

Used via:
  PYTHONPATH=experiments/terminal_bench harbor run \
      -a gv_codex_agent:GVCodex -m gpt-5.6-luna ...
"""
import json
import os
import shlex
from typing import override

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from harbor.models.trial.paths import EnvironmentPaths

from harbor.agents.installed.codex import Codex

from codex_agent import TieredCodex
from gv_agent import (GV_DEST, GV_ROOT, STATE_DIR, TOOLCHAIN, MCP_SERVER,
                      WRAPPER_ENV_SKIP, credential_exports, install_payload,
                      release_state_for_collection, stage_credentials)

SEATS_HOME = "/tmp/codex-seats-home"
# where Harbor stages the uploaded subscription auth for the run; the
# wrapper reads it by this fixed path because Codex starts MCP servers
# with a minimal environment (no CODEX_HOME)
AUTH_JSON = (Codex._REMOTE_CODEX_SECRETS_DIR / "auth.json").as_posix()

GV_SUFFIX = (
    "\n\nFollow the claim protocol in your AGENTS.md while you work: commit "
    "your key intermediate conclusions as claims, commit a final claim "
    "stating that the task above is complete, and request gv.certificate "
    "on it before you finish."
)


class GVCodex(TieredCodex):
    @staticmethod
    @override
    def name() -> str:
        return "gv-codex"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Harbor writes stdio servers as `command = "<command> <args>"`;
        # Codex reads `command` as the executable alone, so the wrapper
        # takes no arguments and the trigger rides in as an env knob.
        self.mcp_servers.append(MCPServerConfig(
            name="gv", transport="stdio", command=MCP_SERVER, args=[]))

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        await install_payload(self, environment)
        model = (self.model_name or "gpt-5.6-luna").split("/")[-1]
        # the verifier's own seats are infrastructure, not the subject of
        # the comparison: their effort is set once for the system
        # (GV_SEAT_EFFORT), never inherited from the solver under test,
        # whose per-call latency the verification loop cannot absorb
        effort = os.environ.get("GV_SEAT_EFFORT") or "high"
        tier = os.environ.get("GV_SEAT_SERVICE_TIER") or "standard"
        knobs = "".join(
            f"export {k}={shlex.quote(v)}\n"
            for k, v in sorted(os.environ.items())
            if k.startswith("GV_") and k not in WRAPPER_ENV_SKIP)
        trigger = os.environ.get("GV_TRIGGER")
        trigger_arg = f" --trigger {shlex.quote(trigger)}" if trigger else ""
        wrapper = (
            "#!/bin/sh\n"
            'if [ -s "$HOME/.nvm/nvm.sh" ]; then . "$HOME/.nvm/nvm.sh"; fi\n'
            'export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"\n'
            f"{knobs}"
            f"{credential_exports()}"
            # the seats' own CODEX_HOME: the solver's auth, none of its
            # MCP servers or hooks (a seat loading the gv server would
            # recurse into the verifier it serves)
            f"mkdir -p {SEATS_HOME}\n"
            f'if [ -e {AUTH_JSON} ]; then ln -sf {AUTH_JSON} '
            f"{SEATS_HOME}/auth.json; "
            f'elif [ -n "$CODEX_HOME" ] && [ -e "$CODEX_HOME/auth.json" ]; '
            f'then ln -sf "$(readlink -f "$CODEX_HOME/auth.json")" '
            f"{SEATS_HOME}/auth.json; fi\n"
            f'[ -e {SEATS_HOME}/config.toml ] || printf %s "" > {SEATS_HOME}/config.toml\n'
            f'export HOME="${{HOME:-/root}}"\n'
            f"export CODEX_HOME={SEATS_HOME}\n"
            # the server's stderr is otherwise lost with the per-run
            # CODEX_HOME; keep it beside the ledger for diagnosis
            f"exec uv run --project {GV_DEST} python -m verifier.mcp_server "
            f"--state-dir {STATE_DIR} --toolchain-dir {TOOLCHAIN} "
            f"--seats codex --model {shlex.quote(model)} "
            f"--effort {shlex.quote(effort)} "
            f"--service-tier {shlex.quote(tier)}{trigger_arg} "
            f"2>>{STATE_DIR}/mcp_server.stderr\n")
        await self.exec_as_root(environment, command=(
            f"printf %s {shlex.quote(wrapper)} > {MCP_SERVER} && "
            f"chmod 755 {MCP_SERVER}"))

    # -- per-run config ----------------------------------------------------
    def _hooks_json(self) -> str:
        hook = lambda script: {"type": "command", "timeout": 60,  # noqa: E731
                               "command": (
            f"GV_HOOK_FEED={STATE_DIR}/hook_feed.jsonl "
            f"{GV_DEST}/.venv/bin/python {GV_DEST}/hooks/{script}")}
        return json.dumps({"hooks": {
            "PreToolUse": [{"hooks": [hook("pre_tool_use.py")]}],
            "PostToolUse": [{"hooks": [hook("post_tool_use.py")]}],
            # the certificate is a goal: the solver may not finish before
            # the final claim is certified (hooks/stop.py)
            "Stop": [{"hooks": [{"type": "command", "timeout": 60,
                                 "command": (
                f"GV_STATE_DIR={STATE_DIR} "
                f"{GV_DEST}/.venv/bin/python {GV_DEST}/hooks/stop.py")}]}],
        }})

    @override
    def _build_register_mcp_servers_command(self) -> str | None:
        """Harbor's setup step, extended: the hooks feature and file, and
        the protocol as the global AGENTS.md, all inside the per-run
        CODEX_HOME Harbor creates and removes."""
        base = super()._build_register_mcp_servers_command() or ""
        protocol = (GV_ROOT / "protocol" / "solver-protocol.md").read_text()
        # a commit returns only once the claim is verified, which takes
        # minutes; Codex's default tool call timeout (300s) would turn every
        # verification into an error the solver never sees the result of.
        # These keys extend Harbor's [mcp_servers.gv] table, so they must
        # follow it directly.
        timeouts = "tool_timeout_sec = 14400\nstartup_timeout_sec = 300\n"
        features = "\n[features]\nhooks = true\n"
        return "\n".join(part for part in (
            base,
            f'printf %s {shlex.quote(timeouts)} >> "$CODEX_HOME/config.toml"',
            f'printf %s {shlex.quote(features)} >> "$CODEX_HOME/config.toml"',
            f'printf %s {shlex.quote(self._hooks_json())} > "$CODEX_HOME/hooks.json"',
            f'printf %s {shlex.quote(protocol)} > "$CODEX_HOME/AGENTS.md"',
        ) if part)

    @override
    def build_cli_flags(self) -> str:
        # the hooks are ours, installed this run: no interactive trust
        flags = super().build_cli_flags()
        return (flags + " " if flags else "") + "--dangerously-bypass-hook-trust"

    @override
    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        await stage_credentials(self, environment)
        # harness ground truth for the verifier: verbatim task statement +
        # pre-task snapshot of /app, both written before the solver acts
        await self.exec_as_agent(environment, command=(
            f"mkdir -p {STATE_DIR}; "
            f"printf %s {shlex.quote(instruction)} > {STATE_DIR}/task.md; "
            'export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"; '
            f"uv run --project {GV_DEST} python -m verifier.baseline "
            f"/app {STATE_DIR}/baseline || true"))
        try:
            await super().run(instruction + GV_SUFFIX, environment, context)
        finally:
            await release_state_for_collection(self, environment)
