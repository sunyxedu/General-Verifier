"""
Provider wiring: every Claude Code invocation in this repo runs on the provider configured in `.env` at the repo root.

This part doesn't include OpenAI backend. (Codex SDK)

One source of truth. 
Agents ll take their provider env from here:
- CLI spawns (experiments/common/solver.py), 
- Agent SDK seats (verifier/llm/agents.py), 
- TurboAgent proxy backend (experiments/common/turbo.py)
- Harbor container agents (experiments/terminal_bench/)

Experiments import it through the installed `verifier` package; gv_agent.py file-loads it from GV_ROOT.

Configuration (.env, or real environment variables. The real environment wins; see .env.example for the full reference):
  GV_PROVIDER = deepseek (default) | anthropic | openrouter | custom
  GV_MODEL    = model id on that provider (defaults below)
  DEEPSEEK_API_KEY / OPENROUTER_API_KEY = the key for the chosen provider.
  GV_ANTHROPIC_BASE_URL / GV_API_KEY / GV_LITELLM_MODEL = everything explicit, for GV_PROVIDER=custom.
  CLAUDE_CODE_OAUTH_TOKEN (or ANTHROPIC_API_KEY, or the interactive login) for GV_PROVIDER=anthropic.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_dotenv(path):
    """Minimal .env parser: KEY=VALUE lines, #-comments, optional quotes."""
    conf = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip("'\"")
                conf[k.strip()] = v
    except OSError:
        pass
    return conf


_DOTENV = _load_dotenv(os.path.join(ROOT, ".env"))


def cfg(name, default=None):
    """Real environment > .env file > baked-in default."""
    v = os.environ.get(name)
    if v is None:
        v = _DOTENV.get(name)
    return v if v not in (None, "") else default


PROVIDER = (cfg("GV_PROVIDER", "deepseek") or "deepseek").lower()

if PROVIDER == "anthropic":
    # First-party Claude: the CLI's own login. Credentials come from, in
    # order, CLAUDE_CODE_OAUTH_TOKEN (a `claude setup-token` token, also the
    # form Harbor forwards into task containers under CLAUDE_FORCE_OAUTH),
    # ANTHROPIC_API_KEY / GV_API_KEY, or the interactive login's
    # .credentials.json in the config dir. No base-URL override.
    MODEL = cfg("GV_MODEL", "claude-sonnet-5")
    ANTHROPIC_BASE_URL = None
    OAUTH_TOKEN = cfg("CLAUDE_CODE_OAUTH_TOKEN")
    API_KEY = cfg("GV_API_KEY") or cfg("ANTHROPIC_API_KEY")
    KEY_ENV = "ANTHROPIC_API_KEY"
    LITELLM_MODEL = f"anthropic/{MODEL}"
    _X_API_KEY = API_KEY
elif PROVIDER == "deepseek":
    MODEL = cfg("GV_MODEL", "deepseek-v4-flash")
    ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
    API_KEY = cfg("DEEPSEEK_API_KEY")
    KEY_ENV = "DEEPSEEK_API_KEY"
    LITELLM_MODEL = f"deepseek/{MODEL}"
    _X_API_KEY = API_KEY
elif PROVIDER == "openrouter":
    MODEL = cfg("GV_MODEL", "deepseek/deepseek-v4-flash")
    ANTHROPIC_BASE_URL = "https://openrouter.ai/api"
    API_KEY = cfg("OPENROUTER_API_KEY")
    KEY_ENV = "OPENROUTER_API_KEY"
    LITELLM_MODEL = f"openrouter/{MODEL}"
    _X_API_KEY = ""    # MUST be explicitly blank: Bearer-only auth, and a
                       # non-empty x-api-key makes claude fall back to it
elif PROVIDER == "custom":
    MODEL = cfg("GV_MODEL")
    ANTHROPIC_BASE_URL = cfg("GV_ANTHROPIC_BASE_URL")
    API_KEY = cfg("GV_API_KEY")
    KEY_ENV = "GV_API_KEY"
    LITELLM_MODEL = cfg("GV_LITELLM_MODEL", MODEL)
    _X_API_KEY = API_KEY
else:
    raise RuntimeError(f"unknown GV_PROVIDER: {PROVIDER!r} "
                       "(expected anthropic | deepseek | openrouter | custom)")

if PROVIDER != "anthropic":
    OAUTH_TOKEN = None

# Kept for backward reference; prefer API_KEY.
DEEPSEEK_API_KEY = API_KEY if PROVIDER == "deepseek" else \
    cfg("DEEPSEEK_API_KEY")

if PROVIDER == "anthropic" and not (OAUTH_TOKEN or API_KEY):
    # credentials live in the interactive login's config dir; an explicit
    # override lets a harness point at a staged copy
    CONFIG_DIR = (cfg("GV_CLAUDE_CONFIG_DIR")
                  or os.environ.get("CLAUDE_CONFIG_DIR")
                  or os.path.expanduser("~/.claude"))
else:
    # token/key auth needs no login state: an isolated, hook-free config
    # dir keeps verifier seats out of any solver-side settings
    CONFIG_DIR = os.path.expanduser(f"~/.claude-gv-{PROVIDER}")

# Text-only provider: the model cannot accept image content blocks.
# Set GV_TEXT_ONLY=1 (env or .env) for models like deepseek-v4-flash;
# SDK seats then refuse to Read raster images into context (agents.py).
TEXT_ONLY = (cfg("GV_TEXT_ONLY", "") == "1")

CLAUDE_ENV = {
    "ANTHROPIC_MODEL": MODEL,
    "ANTHROPIC_SMALL_FAST_MODEL": MODEL,
    "ANTHROPIC_DEFAULT_OPUS_MODEL": MODEL,
    "ANTHROPIC_DEFAULT_SONNET_MODEL": MODEL,
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": MODEL,
    "CLAUDE_CODE_SUBAGENT_MODEL": MODEL,
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "API_TIMEOUT_MS": "3600000",
    "MCP_TIMEOUT": "600000",
    "MCP_TOOL_TIMEOUT": "14400000",
}
if PROVIDER == "anthropic":
    # first-party: no base URL; whichever credential is configured rides
    # along (a token and a key together would make the CLI prefer the key)
    if OAUTH_TOKEN:
        CLAUDE_ENV["CLAUDE_CODE_OAUTH_TOKEN"] = OAUTH_TOKEN
    elif API_KEY:
        CLAUDE_ENV["ANTHROPIC_API_KEY"] = API_KEY
else:
    CLAUDE_ENV["ANTHROPIC_BASE_URL"] = ANTHROPIC_BASE_URL
    CLAUDE_ENV["ANTHROPIC_AUTH_TOKEN"] = API_KEY
    CLAUDE_ENV["ANTHROPIC_API_KEY"] = _X_API_KEY
    CLAUDE_ENV[KEY_ENV] = API_KEY


def _keychain_login():
    """Whether the interactive login lives in the macOS Keychain: the CLI
    keeps its OAuth record there and no longer writes the file copy, so
    a login without .credentials.json is still a login."""
    if sys.platform != "darwin":
        return False
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s",
             "Claude Code-credentials", "-w"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out)


def validate_configuration():
    """Require credentials only when starting a provider-backed process."""
    if PROVIDER == "anthropic":
        if OAUTH_TOKEN or API_KEY:
            return
        credentials = os.path.join(CONFIG_DIR, ".credentials.json")
        if os.path.isfile(credentials) or _keychain_login():
            return
        raise RuntimeError(
            "GV_PROVIDER=anthropic needs CLAUDE_CODE_OAUTH_TOKEN "
            "(`claude setup-token`), ANTHROPIC_API_KEY, or a Claude "
            f"Code login (macOS Keychain or {credentials})")
    if PROVIDER == "custom" and not (MODEL and ANTHROPIC_BASE_URL and API_KEY):
        raise RuntimeError("GV_PROVIDER=custom needs GV_MODEL, "
                           "GV_ANTHROPIC_BASE_URL and GV_API_KEY")
    if not API_KEY:
        raise RuntimeError(f"GV_PROVIDER={PROVIDER} needs {KEY_ENV} "
                           "(in .env or the environment)")


def without_verifier_venv(env):
    """The environment with the verifier's own virtualenv taken out of
    the way. The MCP server runs under `uv run`, which puts the
    verifier's .venv first on PATH and sets VIRTUAL_ENV; a check that
    then runs `python3` gets the verifier's interpreter — without the
    task's packages — instead of the task's own. Checks, and the seats
    that run them, must see the world's interpreters, not ours."""
    env = dict(env)
    venv = os.path.join(ROOT, ".venv")
    parts = [p for p in env.get("PATH", "").split(os.pathsep)
             if p and not os.path.abspath(p).startswith(venv + os.sep)
             and os.path.abspath(p) != venv]
    env["PATH"] = os.pathsep.join(parts)
    if os.path.abspath(env.get("VIRTUAL_ENV", "") or "/") == venv:
        env.pop("VIRTUAL_ENV", None)
    return env


def claude_env(base=None):
    """
    Full process env for spawning `claude` (or anything that spawns it) against the configured provider.    
    The current environment overlaid with CLAUDE_ENV, plus a login-free CLAUDE_CONFIG_DIR.
    """
    validate_configuration()
    env = without_verifier_venv(os.environ if base is None else base)
    env.update(CLAUDE_ENV)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    env["CLAUDE_CONFIG_DIR"] = CONFIG_DIR
    return env
