"""Cybersecurity agents with the original GV output and tool contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifier.llm.agents import AgentRole


def build_roles() -> dict[str, AgentRole]:
    from verifier.llm.agents import ALL_AGENT_ROLES
    from .prompts import SYSTEMS

    context = (
        "Controller profile context (scope and immutable identity only, "
        "never a candidate verdict): {profile_context}\n\n"
    )
    return {
        name: replace(
            ALL_AGENT_ROLES[name],
            system=system,
            template=context + ALL_AGENT_ROLES[name].template,
            tools=[
                tool for tool in ALL_AGENT_ROLES[name].tools
                if tool not in {"WebSearch", "WebFetch"}
            ],
        )
        for name, system in SYSTEMS.items()
    }
