"""Complete verifier role configurations sharing the original GV kernel.

Importing this module does not import a backend or the role registry. This
keeps the profile boundary usable while those modules initialize.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from verifier.llm.agents import AgentRole


def _role_configuration(role: AgentRole) -> dict[str, Any]:
    from verifier.llm.agents import to_json_schema

    return {
        "name": role.name,
        "system": role.system,
        "template": role.template,
        "schema": to_json_schema(role.schema),
        "tools": list(role.tools),
        "model": role.model,
        "max_turns": role.max_turns,
    }


@dataclass(frozen=True)
class VerificationProfile:
    """A complete, content-addressed suite of stateless verifier agents.

    Explicit role overrides are full AgentRole objects, including output
    schemas and templates. The default profile preserves the supplied
    registry's exact objects. A custom suite may extend that registry.
    """

    name: str = "default"
    version: int = 1
    solver_instructions: str = ""
    disable_external_search: bool = False
    role_overrides: Mapping[str, AgentRole] | None = field(
        default=None, repr=False, compare=False
    )

    def _overrides(self) -> Mapping[str, AgentRole]:
        if self.role_overrides is not None:
            return self.role_overrides
        if self.name == "cyber":
            from .cyber import build_roles

            return build_roles()
        return {}

    def apply_roles(
        self, base_roles: Mapping[str, AgentRole]
    ) -> dict[str, AgentRole]:
        roles = dict(base_roles)
        for name, role in self._overrides().items():
            if name != role.name:
                raise ValueError(f"role registry key {name!r} differs from role name")
            roles[name] = role
        return roles

    def identity_for_roles(
        self, roles: Mapping[str, AgentRole]
    ) -> dict[str, Any]:
        """Include the effective registry, e.g. a harness-configured default."""
        return {
            "name": self.name,
            "version": self.version,
            "solver_instructions": self.solver_instructions,
            "disable_external_search": self.disable_external_search,
            "roles": {
                name: _role_configuration(role)
                for name, role in sorted(roles.items())
            },
        }

    @property
    def identity(self) -> dict[str, Any]:
        from verifier.llm.agents import ALL_AGENT_ROLES

        return self.identity_for_roles(self.apply_roles(ALL_AGENT_ROLES))

    @property
    def id(self) -> str:
        encoded = json.dumps(
            self.identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def get_profile(name: str | VerificationProfile | None = None) -> VerificationProfile:
    if isinstance(name, VerificationProfile):
        if name.name == "default" and (
            name.version != 1
            or name.solver_instructions
            or name.disable_external_search
            or name.role_overrides is not None
        ):
            raise ValueError("the reserved default profile cannot have custom configuration")
        return name
    if name in (None, "default", "plain"):
        return VerificationProfile()
    if name == "cyber":
        from .cyber.prompts import SOLVER_INSTRUCTIONS

        return VerificationProfile(
            name="cyber",
            solver_instructions=SOLVER_INSTRUCTIONS,
            disable_external_search=True,
        )
    raise ValueError(f"unknown verification profile: {name!r}")
