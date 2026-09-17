"""Harbor custom agent: Claude Code with the full General Verifier mounted.

Runs inside the Terminal-Bench task container exactly like the stock
`claude-code` agent (same install, same headless invocation), plus:

  - the General Verifier codebase is shipped into the container
    (/opt/general-verifier) and `uv sync`ed with a uv-managed Python
  - the gv MCP server is registered user-scope (Harbor's own mechanism), so
    the solver gets the gv.* verbs with no trust dialog
  - Pre/PostToolUse hooks are wired via the user-level settings.json in
    $CLAUDE_CONFIG_DIR, feeding /gv-state/hook_feed.jsonl
  - the solver protocol is installed as user-level CLAUDE.md, and a short
    reminder is appended to the task instruction

The solver still receives ONE prompt: the task instruction. Everything else
is the system under measurement.

Used via:
  PYTHONPATH=experiments/terminal_bench harbor run \
      -a gv_agent:GVClaudeCode -m anthropic/sonnet ...
"""
import importlib.util
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import override

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from harbor.models.trial.paths import EnvironmentPaths

GV_ROOT = Path(__file__).resolve().parents[2]      # the General Verifier repo

# DeepSeek wiring — file-loaded because this module runs under Harbor's own
# Python, where the `verifier` package is not installed
_spec = importlib.util.spec_from_file_location(
    "gv_provider", GV_ROOT / "verifier" / "llm" / "provider.py")
provider = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(provider)
GV_DEST = "/opt/general-verifier"
STATE_DIR = "/gv-state"
TOOLCHAIN = "/opt/gv-toolchain"
MCP_SERVER = "/usr/local/bin/gv-mcp-server"

# only what the in-container verifier needs: the package, the hooks, the
# solver protocol, and the project files uv syncs from. Tarring the whole
# checkout would ship experiment archives and scratch trees by the gigabyte.
TAR_MEMBERS = ["verifier", "hooks", "protocol", "pyproject.toml", "uv.lock",
               "README.md", "LICENSE", ".env"]
TAR_EXCLUDES = ["__pycache__", ".pytest_cache", "*.pyc"]

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


def login_access_token(min_minutes_left: float = 60.0) -> str:
    """The login's current access token, guaranteed to have at least
    ``min_minutes_left`` of life: the CLI rotates access tokens on a
    short cycle and refreshes only on demand, so when the remaining life
    is too short for a trial this triggers a refresh (a trivial headless
    call once the token has expired) and waits for the new one."""
    import subprocess
    import time
    deadline = time.time() + 30 * 60
    while True:
        creds = _login_credentials()
        left = creds["expiresAt"] / 1000 - time.time()
        if left >= min_minutes_left * 60 or time.time() > deadline:
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


def refresh_login_token() -> None:
    if os.environ.get("GV_OAUTH_FROM_LOGIN") == "1":
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = login_access_token()


TOKEN_FILE = ".gv-oauth-token"
# host GV_* variables are forwarded into the in-container MCP server's
# environment (verifier knobs such as GV_SCREENER, GV_PARALLEL_OBLIGATIONS,
# GV_MAX_CONCURRENT_SEATS); these name host-side paths or launcher state
WRAPPER_ENV_SKIP = {"GV_OAUTH_FROM_LOGIN", "GV_CLAUDE_CONFIG_DIR",
                    "GV_HOOK_FEED", "GV_PROVIDER", "GV_MODEL", "GV_TRIGGER"}


GV_SUFFIX = (
    "\n\nFollow the claim protocol in your CLAUDE.md while you work: commit "
    "your key intermediate conclusions as claims, commit a final claim "
    "stating that the task above is complete, and request gv.certificate "
    "on it before you finish."
)


class GVClaudeCode(ClaudeCode):
    @staticmethod
    @override
    def name() -> str:
        return "gv-claude-code"

    def __init__(self, logs_dir: Path, *args, **kwargs):
        super().__init__(logs_dir, *args, **kwargs)
        # user-scoped MCP registration; ClaudeCode.run() writes it into
        # $CLAUDE_CONFIG_DIR/.claude.json before invoking claude
        # GV_TRIGGER=on-first-dependent runs the lazy control arm; unset means
        # the default (verify at commit)
        args = (["--trigger", os.environ["GV_TRIGGER"]]
                if os.environ.get("GV_TRIGGER") else [])
        self.mcp_servers.append(MCPServerConfig(
            name="gv", transport="stdio", command=MCP_SERVER, args=args))

    # -- payload -----------------------------------------------------------
    def _make_tarball(self) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".tgz", delete=False)
        tmp.close()
        cmd = ["tar", "czf", tmp.name]
        for pat in TAR_EXCLUDES:
            cmd += ["--exclude", pat]
        cmd += ["-C", str(GV_ROOT)]
        cmd += [m for m in TAR_MEMBERS if (GV_ROOT / m).exists()]
        subprocess.run(cmd, check=True)
        return tmp.name

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)

        tarball = self._make_tarball()
        await environment.upload_file(tarball, "/tmp/gv-src.tgz")
        await self.exec_as_root(environment, command=(
            "set -e; "
            f"rm -rf {GV_DEST} && mkdir -p {GV_DEST} && "
            f"tar xzf /tmp/gv-src.tgz -C {GV_DEST} && rm /tmp/gv-src.tgz && "
            f"mkdir -p {STATE_DIR} {TOOLCHAIN} && "
            f"chmod -R 777 {GV_DEST} {STATE_DIR} {TOOLCHAIN}"))

        # uv + a managed Python + the verifier's deps (mcp, claude-agent-sdk,
        # openai). One-time per task container.
        await self.exec_as_agent(environment, command=(
            'set -e; export PATH="$HOME/.local/bin:$PATH"; '
            "command -v uv >/dev/null 2>&1 || "
            "  (curl -LsSf https://astral.sh/uv/install.sh | sh); "
            f"cd {GV_DEST} && uv sync"))

        # stable entry point for the MCP registration (claude spawns it with
        # the agent user's env, so the credential env / PATH are inherited).
        # The provider selection is pinned explicitly: the shipped .env may
        # name a different default than the host run was configured with.
        model = (self.model_name or "sonnet").split("/")[-1]
        knobs = "".join(
            f"export {k}={shlex.quote(v)}\n"
            for k, v in sorted(os.environ.items())
            if k.startswith("GV_") and k not in WRAPPER_ENV_SKIP)
        wrapper = (
            "#!/bin/sh\n"
            'export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"\n'
            f"export GV_PROVIDER={shlex.quote(provider.PROVIDER)}\n"
            f"export GV_MODEL={shlex.quote(str(provider.MODEL))}\n"
            f"{knobs}"
            # the CLI does not hand its own credential env to MCP servers;
            # the trial stages the token in a file the wrapper reads
            f'if [ -f "$HOME/{TOKEN_FILE}" ]; then '
            f'export CLAUDE_CODE_OAUTH_TOKEN="$(cat "$HOME/{TOKEN_FILE}")"; fi\n'
            f"exec uv run --project {GV_DEST} python -m verifier.mcp_server "
            f"--state-dir {STATE_DIR} --toolchain-dir {TOOLCHAIN} "
            f'--model {model} "$@"\n')
        await self.exec_as_root(environment, command=(
            f"printf %s {shlex.quote(wrapper)} > {MCP_SERVER} && "
            f"chmod 755 {MCP_SERVER}"))

    async def _stage_token(self, environment: BaseEnvironment) -> None:
        """Stage the verifier seats' credential where the MCP wrapper reads
        it (outside /gv-state, which is pulled as an artifact)."""
        token = (os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or "").strip()
        if not token:
            return
        await self.exec_as_agent(environment, command=(
            f'umask 077; printf %s "$GV_TOKEN" > "$HOME/{TOKEN_FILE}"'),
            env={"GV_TOKEN": token})

    # -- per-run config ----------------------------------------------------
    def _settings_json(self) -> str:
        hook = lambda script: {"type": "command", "command": (  # noqa: E731
            f"GV_HOOK_FEED={STATE_DIR}/hook_feed.jsonl "
            f"{GV_DEST}/.venv/bin/python {GV_DEST}/hooks/{script}")}
        return json.dumps({
            # pin the in-container claude (and the MCP server it spawns) to
            # DeepSeek — harbor only forwards BASE_URL + key, not the
            # DEFAULT_* model pins
            "env": {k: v for k, v in provider.CLAUDE_ENV.items()
                    if k != "CLAUDE_CODE_OAUTH_TOKEN"},
            "hooks": {
                "PreToolUse": [{"matcher": "*",
                                "hooks": [hook("pre_tool_use.py")]}],
                "PostToolUse": [{"matcher": "*",
                                 "hooks": [hook("post_tool_use.py")]}],
            }})

    @override
    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        refresh_login_token()
        await self._stage_token(environment)
        cfg = (EnvironmentPaths.agent_dir / "sessions").as_posix()
        protocol = (GV_ROOT / "protocol" / "solver-protocol.md").read_text()
        await self.exec_as_agent(environment, command=(
            f"mkdir -p {cfg} && "
            f"printf %s {shlex.quote(self._settings_json())} "
            f"> {cfg}/settings.json && "
            f"printf %s {shlex.quote(protocol)} > {cfg}/CLAUDE.md"))
        # harness ground truth for the verifier: verbatim task statement +
        # pre-task snapshot of /app (the T-Bench working-tree convention),
        # both written before the solver gets to act
        await self.exec_as_agent(environment, command=(
            f"printf %s {shlex.quote(instruction)} > {STATE_DIR}/task.md; "
            'export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"; '
            f"uv run --project {GV_DEST} python -m verifier.baseline "
            f"/app {STATE_DIR}/baseline || true"))
        await super().run(instruction + GV_SUFFIX, environment, context)


class RecClaudeCode(GVClaudeCode):
    """Recording-only arm for the offline ORM path: same container payload
    as GVClaudeCode (its venv python runs the hooks; uv runs the baseline
    snapshot), but no MCP server registration, no protocol CLAUDE.md, and
    no prompt suffix — the solver is stock claude-code with its
    distribution untouched. The ledger + task.md + baseline land in
    /gv-state for the --artifact pull; scoring happens on the host
    (score.py) after the fact."""

    @staticmethod
    @override
    def name() -> str:
        return "rec-claude-code"

    def __init__(self, logs_dir: Path, *args, **kwargs):
        super().__init__(logs_dir, *args, **kwargs)
        # GVClaudeCode.__init__ registered the gv MCP server; the recorder
        # must not — its absence is the whole point of the arm
        self.mcp_servers = [s for s in self.mcp_servers if s.name != "gv"]

    @override
    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        refresh_login_token()
        cfg = (EnvironmentPaths.agent_dir / "sessions").as_posix()
        await self.exec_as_agent(environment, command=(
            f"mkdir -p {cfg} && "
            f"printf %s {shlex.quote(self._settings_json())} "
            f"> {cfg}/settings.json"))
        await self.exec_as_agent(environment, command=(
            f"printf %s {shlex.quote(instruction)} > {STATE_DIR}/task.md; "
            'export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"; '
            f"uv run --project {GV_DEST} python -m verifier.baseline "
            f"/app {STATE_DIR}/baseline || true"))
        # grandparent run: plain instruction, no protocol, no suffix
        await ClaudeCode.run(self, instruction, environment, context)
