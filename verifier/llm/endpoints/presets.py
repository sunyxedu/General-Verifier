"""The provider presets: the one table of where each model is reached.

Three consumers used to decide provider and model each on their own: the
Claude Code environment (``llm/provider.py``), the direct seats' transport
choice, and the Codex runtime's default model. Every one of them now reads
this table: a preset names the wire protocol, base URL, credential
variables and default model of a provider, for the surface an agent
runtime talks to and, where it differs, for a direct model call.

Standard library only and free of verifier imports: ``llm/provider.py`` is
loaded by path under a harness's own Python, and loads this module the same
way. The settings themselves are parsed by ``verifier.config``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

ANTHROPIC_MESSAGES = "anthropic-messages"
OPENAI_RESPONSES = "openai-responses"
OPENAI_CHAT = "openai-chat"
CODEX_CLI = "codex-cli"

AUTH_X_API_KEY = "x-api-key"
AUTH_BEARER = "authorization"


@dataclass(frozen=True)
class EndpointPreset:
    """One surface of a provider. ``base_url`` None means the protocol's
    own first-party address; ``base_url_env`` names a setting that supplies
    (or overrides) it. The first set variable of ``key_envs`` is the key."""
    protocol: str
    base_url: Optional[str] = None
    base_url_env: Optional[str] = None
    key_envs: Tuple[str, ...] = ()
    default_model: Optional[str] = None
    auth_header: str = AUTH_X_API_KEY
    extra_headers: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ProviderPreset:
    """A provider: ``agent`` is the surface an agent runtime talks to (the
    Anthropic Messages surface Claude Code speaks, or the Codex CLI);
    ``direct`` the raw surface a direct seat calls when this provider is
    chosen for it (None: the agent surface itself). ``key_env`` is the
    variable the agent key is exported under for spawned processes."""
    name: str
    agent: Optional[EndpointPreset] = None
    direct: Optional[EndpointPreset] = None
    key_env: Optional[str] = None
    litellm_prefix: Optional[str] = None
    litellm_env: Optional[str] = None
    # the first-party login (an OAuth token or the interactive login) works
    login: bool = False
    # Bearer-only auth: a non-empty x-api-key makes claude fall back to it
    blank_x_api_key: bool = False


PROVIDERS = {
    "deepseek": ProviderPreset(
        "deepseek",
        agent=EndpointPreset(
            ANTHROPIC_MESSAGES, base_url="https://api.deepseek.com/anthropic",
            key_envs=("DEEPSEEK_API_KEY",), default_model="deepseek-v4-flash"),
        key_env="DEEPSEEK_API_KEY", litellm_prefix="deepseek/"),
    "anthropic": ProviderPreset(
        "anthropic",
        # first-party Claude: no base-URL override
        agent=EndpointPreset(
            ANTHROPIC_MESSAGES, key_envs=("GV_API_KEY", "ANTHROPIC_API_KEY"),
            default_model="claude-sonnet-5"),
        key_env="ANTHROPIC_API_KEY", litellm_prefix="anthropic/", login=True),
    "openrouter": ProviderPreset(
        "openrouter",
        agent=EndpointPreset(
            ANTHROPIC_MESSAGES, base_url="https://openrouter.ai/api",
            key_envs=("OPENROUTER_API_KEY",),
            default_model="deepseek/deepseek-v4-flash",
            auth_header=AUTH_BEARER),
        # direct seats use the OpenAI-compatible surface OpenRouter and most
        # gateways expose
        direct=EndpointPreset(
            OPENAI_CHAT, base_url="https://openrouter.ai/api",
            base_url_env="OPENROUTER_BASE_URL",
            key_envs=("OPENROUTER_API_KEY",),
            default_model="openai/gpt-5.6-luna", auth_header=AUTH_BEARER,
            extra_headers=(
                ("HTTP-Referer", "https://github.com/hemlix/general-verifier"),
                ("X-Title", "General Verifier"))),
        key_env="OPENROUTER_API_KEY", litellm_prefix="openrouter/",
        blank_x_api_key=True),
    "custom": ProviderPreset(
        "custom",
        # everything explicit
        agent=EndpointPreset(
            ANTHROPIC_MESSAGES, base_url_env="GV_ANTHROPIC_BASE_URL",
            key_envs=("GV_API_KEY",)),
        key_env="GV_API_KEY", litellm_env="GV_LITELLM_MODEL"),
    "openai": ProviderPreset(
        "openai",
        direct=EndpointPreset(
            OPENAI_RESPONSES, base_url="https://api.openai.com",
            base_url_env="OPENAI_BASE_URL", key_envs=("OPENAI_API_KEY",),
            default_model="gpt-5.6-luna", auth_header=AUTH_BEARER)),
    # the ChatGPT subscription through the Codex CLI: its own login, no key
    "codex": ProviderPreset(
        "codex", agent=EndpointPreset(CODEX_CLI, default_model="gpt-5.6-luna")),
}

AGENT_PROVIDERS = tuple(
    name for name, preset in PROVIDERS.items()
    if preset.agent is not None and preset.agent.protocol == ANTHROPIC_MESSAGES)


def default_model(provider: str) -> Optional[str]:
    """The model a provider's agent surface uses when none is named."""
    return PROVIDERS[provider].agent.default_model


@dataclass(frozen=True)
class AgentProvider:
    """The resolved provider of the Claude Code seats."""
    name: str
    model: Optional[str]
    base_url: Optional[str]
    api_key: Optional[str]
    key_env: Optional[str]
    litellm_model: Optional[str]
    x_api_key: Optional[str]
    oauth_token: Optional[str]


def _first(settings: Any, names: Tuple[str, ...]) -> Optional[str]:
    for name in names:
        value = settings.value(name)
        if value:
            return value
    return None


def resolve_agent_provider(settings: Any) -> AgentProvider:
    """``settings`` is a ``verifier.config.ProviderSettings``."""
    preset = PROVIDERS.get(settings.provider)
    if preset is None or settings.provider not in AGENT_PROVIDERS:
        raise RuntimeError(f"unknown GV_PROVIDER: {settings.provider!r} "
                           "(expected anthropic | deepseek | openrouter | custom)")
    surface = preset.agent
    model = settings.model or surface.default_model
    api_key = _first(settings, surface.key_envs)
    base_url = (settings.value(surface.base_url_env) if surface.base_url_env
                else surface.base_url)
    if preset.litellm_prefix is not None:
        litellm_model = f"{preset.litellm_prefix}{model}"
    else:
        litellm_model = settings.value(preset.litellm_env) or model
    return AgentProvider(
        name=preset.name, model=model, base_url=base_url, api_key=api_key,
        key_env=preset.key_env, litellm_model=litellm_model,
        x_api_key="" if preset.blank_x_api_key else api_key,
        oauth_token=(settings.value("CLAUDE_CODE_OAUTH_TOKEN")
                     if preset.login else None))


@dataclass(frozen=True)
class EndpointSpec:
    """A resolved model endpoint: everything a wire implementation needs."""
    protocol: str
    api_key: str
    model: str
    base_url: Optional[str] = None
    auth_header: str = AUTH_X_API_KEY
    reasoning_effort: Optional[str] = None
    extra_headers: Tuple[Tuple[str, str], ...] = ()


def _direct_spec(preset: ProviderPreset, key: str, direct: Any) -> EndpointSpec:
    surface = preset.direct
    return EndpointSpec(
        protocol=surface.protocol, api_key=key,
        model=direct.model or surface.default_model,
        base_url=direct.value(surface.base_url_env),
        auth_header=surface.auth_header,
        reasoning_effort=direct.effort or None,
        extra_headers=surface.extra_headers)


def resolve_direct_endpoint(direct: Any, agent: AgentProvider,
                            ) -> Optional[EndpointSpec]:
    """The endpoint the direct seats call, or None when the only
    credential is a subscription login (which no raw API accepts).
    ``direct`` is a ``verifier.config.DirectSettings``; ``agent`` the Claude
    Code seats' provider, whose key serves a direct seat on the same
    provider."""
    choice = direct.provider
    if choice == "openrouter" or (not choice and agent.name == "openrouter"):
        preset = PROVIDERS["openrouter"]
        key = _first(direct, preset.direct.key_envs) or (
            agent.api_key if agent.name == "openrouter" else None)
        if not key:
            return None
        return _direct_spec(preset, key, direct)
    if choice == "openai" or (not choice and agent.name not in
                              ("anthropic", "deepseek", "custom")
                              and direct.value("OPENAI_API_KEY")):
        preset = PROVIDERS["openai"]
        key = _first(direct, preset.direct.key_envs)
        if not key:
            return None
        return _direct_spec(preset, key, direct)
    if agent.name in AGENT_PROVIDERS:
        if not agent.api_key:
            return None   # login-only: no raw API behind an OAuth token
        surface = PROVIDERS[agent.name].agent
        return EndpointSpec(
            protocol=surface.protocol, api_key=agent.api_key,
            model=direct.model or str(agent.model), base_url=agent.base_url,
            auth_header=surface.auth_header)
    return None
