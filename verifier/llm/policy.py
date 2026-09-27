"""Tool policies: what a seat's session may use, declared by the seat and
translated by the runtime that runs it.

A ``ToolPolicy`` is runtime-neutral: a tool palette (none, read, read and
web, execute), network access, image inputs, and for the strictest
sessions two requirements a runtime must be able to meet. Each runtime
translates it into its own mechanism: the Claude Code runtime into the
SDK's tool palette, the direct runtime needs nothing (a single model call
has no tools), the Codex runtime into a root-deny permission profile with
every feature gate closed, plus an audit of its JSON event stream.

``NoToolsSession`` is one fresh session under such a policy: the empty
directory it runs in, and how the runtime's own event stream is checked
to prove the policy held. The candidate-independent contract compiler and
critic (``authority_backend``) and the typed-packet Judger
(``typed_judger_backend``) each run the ordinary runtime under one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

TOOLS_NONE = "none"
TOOLS_READ = "read"
TOOLS_READ_WEB = "read_web"
TOOLS_EXECUTE = "execute"
TOOL_PALETTES = (TOOLS_NONE, TOOLS_READ, TOOLS_READ_WEB, TOOLS_EXECUTE)


@dataclass(frozen=True)
class ToolPolicy:
    """What a session may use.

    ``denied_integrations`` names further runtime integrations to close
    beyond the policy's shared profile (in the runtime's own vocabulary).
    ``stream_audit`` requires a runtime that reports every event of the
    session, so the absence of tool activity is checked, not assumed; a
    palette restriction alone does not meet it.
    """
    tools: str = TOOLS_EXECUTE
    network: bool = True
    images: bool = True
    denied_integrations: Tuple[str, ...] = ()
    stream_audit: bool = False

    def __post_init__(self):
        if self.tools not in TOOL_PALETTES:
            raise ValueError(f"unknown tool palette: {self.tools!r}")

    @property
    def no_tools(self) -> bool:
        """Nothing to use: no tool, no network, no image input."""
        return (self.tools == TOOLS_NONE and not self.network
                and not self.images)

    @property
    def web_search(self) -> bool:
        return self.tools in (TOOLS_READ_WEB, TOOLS_EXECUTE)


# a single turn over the inlined prompt, nothing else
NO_TOOLS = ToolPolicy(tools=TOOLS_NONE, network=False, images=False)


@dataclass(frozen=True)
class NoToolsSession:
    """One fresh session under a no-tools ``policy``.

    ``scratch`` returns the empty directory the session's process runs in
    (asked on every launch, so a bound directory is re-validated each time).
    For a runtime that reports its event stream: ``validate_stream`` raises
    on any tool, unknown event or incomplete turn; ``bound_response``, when
    set, returns the final message the answer is bound to; and
    ``recover_response`` reads the completed answer from a stream whose
    usage record is unusable. ``instruction_identity`` folds the session's
    boundary into the runtime's instruction identity. ``evidence_prefetch``
    False drops the runtime's own Judger evidence-prefetch channel.
    """
    policy: ToolPolicy
    scratch: Callable[[], str]
    validate_stream: Callable[[str], Any]
    recover_response: Callable[[str], Optional[str]]
    bound_response: Optional[Callable[[str], str]] = None
    instruction_identity: Optional[Callable[[str], str]] = None
    evidence_prefetch: bool = True

    def __post_init__(self):
        if not self.policy.no_tools:
            raise ValueError("a no-tools session needs a no-tools policy")
