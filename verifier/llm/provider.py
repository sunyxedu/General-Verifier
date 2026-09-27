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

Configuration (.env, or real environment variables. The real environment wins; see .env.example for the full reference;
parsed by verifier.config.ProviderSettings, defaults from the preset table llm/endpoints/presets.py):
  GV_PROVIDER = deepseek (default) | anthropic | openrouter | custom
  GV_MODEL    = model id on that provider (default: the provider preset)
  DEEPSEEK_API_KEY / OPENROUTER_API_KEY = the key for the chosen provider.
  GV_ANTHROPIC_BASE_URL / GV_API_KEY / GV_LITELLM_MODEL = everything explicit, for GV_PROVIDER=custom.
  CLAUDE_CODE_OAUTH_TOKEN (or ANTHROPIC_API_KEY, or the interactive login) for GV_PROVIDER=anthropic.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .. import config as _config
    from .endpoints import presets as _presets
except ImportError:
    # File-loaded under a harness's own Python (terminal_bench/gv_agent.py),
    # where the verifier package is not installed: its two stdlib-only
    # dependencies are loaded by path the same way.
    import importlib.util as _importlib_util

    def _load_sibling(name, *path):
        spec = _importlib_util.spec_from_file_location(
            name, os.path.join(ROOT, "verifier", *path))
        module = _importlib_util.module_from_spec(spec)
        sys.modules[name] = module   # dataclasses resolve their module
        spec.loader.exec_module(module)
        return module

    _config = _load_sibling("gv_config", "config.py")
    _presets = _load_sibling("gv_endpoint_presets",
                             "llm", "endpoints", "presets.py")

SETTINGS = _config.ProviderSettings.from_env()
# unknown provider names fail here, at import, as they always did
AGENT = _presets.resolve_agent_provider(SETTINGS)

PROVIDER = AGENT.name
# First-party Claude (anthropic) uses the CLI's own login. Credentials come
# from, in order, CLAUDE_CODE_OAUTH_TOKEN (a `claude setup-token` token, also
# the form Harbor forwards into task containers under CLAUDE_FORCE_OAUTH),
# ANTHROPIC_API_KEY / GV_API_KEY, or the interactive login's
# .credentials.json in the config dir. No base-URL override.
MODEL = AGENT.model
ANTHROPIC_BASE_URL = AGENT.base_url
OAUTH_TOKEN = AGENT.oauth_token
API_KEY = AGENT.api_key
KEY_ENV = AGENT.key_env
LITELLM_MODEL = AGENT.litellm_model
# MUST be explicitly blank on a Bearer-only provider (openrouter): a
# non-empty x-api-key makes claude fall back to it
_X_API_KEY = AGENT.x_api_key

# Kept for backward reference; prefer API_KEY.
DEEPSEEK_API_KEY = API_KEY if PROVIDER == "deepseek" else \
    SETTINGS.value("DEEPSEEK_API_KEY")

if PROVIDER == "anthropic" and not (OAUTH_TOKEN or API_KEY):
    # credentials live in the interactive login's config dir; an explicit
    # override lets a harness point at a staged copy
    CONFIG_DIR = (SETTINGS.claude_config_dir
                  or SETTINGS.inherited_config_dir
                  or os.path.expanduser("~/.claude"))
else:
    # token/key auth needs no login state: an isolated, hook-free config
    # dir keeps verifier seats out of any solver-side settings
    CONFIG_DIR = os.path.expanduser(f"~/.claude-gv-{PROVIDER}")

# Text-only provider: the model cannot accept image content blocks.
# Set GV_TEXT_ONLY=1 (env or .env) for models like deepseek-v4-flash;
# SDK seats then refuse to Read raster images into context (agents.py).
TEXT_ONLY = SETTINGS.text_only

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


def agent_provider():
    """The Claude Code seats' provider as this module currently holds it
    (a harness or a test may rebind the module's values after import)."""
    return _presets.AgentProvider(
        name=PROVIDER, model=MODEL, base_url=ANTHROPIC_BASE_URL,
        api_key=API_KEY, key_env=KEY_ENV, litellm_model=LITELLM_MODEL,
        x_api_key=_X_API_KEY, oauth_token=OAUTH_TOKEN)


def _keychain_login():
    """The interactive login's OAuth record from the macOS Keychain, or
    None: the CLI keeps it there and no longer writes the file copy, so
    a login without .credentials.json is still a login."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s",
             "Claude Code-credentials", "-w"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return json.loads(out)["claudeAiOauth"] if out else None
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return None


def login_access_token():
    """The login's current access token for a spawned `claude`, or None
    when there is no login. A child process gets only the environment
    we hand it, so it cannot consult the Keychain itself; the token is
    read at spawn time (it rotates every few hours) and an expired one
    is refreshed first through a trivial headless call of the CLI."""
    import time
    creds = _keychain_login()
    if creds is None:
        return None
    if creds.get("expiresAt", 0) / 1000 <= time.time():
        try:
            subprocess.run(
                ["claude", "-p", "Reply with exactly: OK", "--model", MODEL,
                 "--output-format", "json"],
                capture_output=True, text=True, timeout=180, cwd="/tmp")
        except (OSError, subprocess.SubprocessError):
            pass
        creds = _keychain_login() or creds
    return creds.get("accessToken")


def validate_configuration():
    """Require credentials only when starting a provider-backed process."""
    if PROVIDER == "anthropic":
        if OAUTH_TOKEN or API_KEY:
            return
        credentials = os.path.join(CONFIG_DIR, ".credentials.json")
        if os.path.isfile(credentials) or _keychain_login() is not None:
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
    if PROVIDER == "anthropic" and not (OAUTH_TOKEN or API_KEY):
        # login mode: the child cannot reach the Keychain, so hand it the
        # login's token explicitly (HOME too — the CLI resolves its own
        # state through it)
        env.setdefault("HOME", os.path.expanduser("~"))
        token = login_access_token()
        if token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    os.makedirs(CONFIG_DIR, exist_ok=True)
    env["CLAUDE_CONFIG_DIR"] = CONFIG_DIR
    return env
