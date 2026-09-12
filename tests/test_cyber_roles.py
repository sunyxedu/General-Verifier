"""The cyber suite specializes roles without changing GV's contracts."""

from __future__ import annotations

import json
import re
import string
import subprocess
import sys
from dataclasses import replace

import pytest

from verifier.llm.agents import ALL_AGENT_ROLES, AgentRole, to_json_schema
from verifier.llm.prompts import AXIOM_INDEPENDENCE, AXIOM_NECESSITY, AXIOM_WITNESS
from verifier.profiles import VerificationProfile, get_profile
from verifier.profiles.cyber.prompts import ANSWER_BLIND, SCOPE, SECURITY, SYSTEMS


def test_default_preserves_exact_supplied_registry():
    original = dict(ALL_AGENT_ROLES)
    original["obligations"] = replace(
        original["obligations"], system=original["obligations"].system + " harness rule"
    )
    applied = get_profile().apply_roles(original)
    assert applied is not original
    assert applied.keys() == original.keys()
    assert all(applied[name] is role for name, role in original.items())
    assert not get_profile().solver_instructions
    assert not get_profile().disable_external_search


def test_all_cyber_roles_are_complete_replacements_with_original_contracts():
    roles = get_profile("cyber").apply_roles(ALL_AGENT_ROLES)
    assert roles.keys() == ALL_AGENT_ROLES.keys() == SYSTEMS.keys()
    for name, original in ALL_AGENT_ROLES.items():
        role = roles[name]
        assert isinstance(role, AgentRole)
        assert role is not original
        assert role.name == name
        assert role.system == SYSTEMS[name]
        assert role.system != original.system
        assert not role.system.startswith(original.system)
        assert role.schema == original.schema
        assert role.model == original.model
        assert role.max_turns == original.max_turns
        assert role.tools == [
            tool for tool in original.tools if tool not in {"WebSearch", "WebFetch"}
        ]
        assert "{profile_context}" in role.template
    assert get_profile("cyber").disable_external_search


@pytest.mark.parametrize("name", ALL_AGENT_ROLES)
def test_every_cyber_template_renders_original_protocol_and_trusted_context(name):
    role = get_profile("cyber").apply_roles(ALL_AGENT_ROLES)[name]
    fields = {
        field_name for _, field_name, _, _ in string.Formatter().parse(role.template)
        if field_name is not None
    }
    variables = dict.fromkeys(fields, "(fixture)")
    context = {"certificate_scope": "candidate", "subject": {"digest": "bound"}}
    variables["profile_context"] = json.dumps(context, sort_keys=True)
    rendered = role.render(variables)
    assert json.dumps(context, sort_keys=True) in rendered
    assert ALL_AGENT_ROLES[name].render(variables) in rendered


@pytest.mark.parametrize(
    "change",
    [
        {"system": "independent complete replacement"},
        {"template": "A different complete template: {profile_context}"},
        {"schema": {"answer": str, "present?": bool}},
        {"tools": ["Read"]},
        {"model": "fixture-model"},
        {"max_turns": 3},
    ],
)
def test_profile_identity_covers_every_role_execution_field(change):
    original = ALL_AGENT_ROLES["judger"]
    profile = VerificationProfile(name="custom", role_overrides={"judger": original})
    modified = replace(
        profile, role_overrides={"judger": replace(original, **change)}
    )
    assert modified.id != profile.id
    json.dumps(modified.identity)
    assert modified.identity["roles"]["judger"]["schema"] == to_json_schema(
        modified.apply_roles(ALL_AGENT_ROLES)["judger"].schema
    )


def test_identity_is_stable_and_covers_profile_policies_and_effective_registry():
    cyber = get_profile("cyber")
    assert cyber.id == get_profile("cyber").id
    assert re.fullmatch(r"[a-f0-9]{64}", cyber.id)
    assert cyber.id != get_profile().id
    assert cyber.id != replace(cyber, solver_instructions="changed").id
    assert cyber.id != replace(cyber, disable_external_search=False).id
    assert cyber.id != replace(cyber, version=2).id
    configured = dict(ALL_AGENT_ROLES)
    configured["critic"] = replace(configured["critic"], max_turns=1)
    assert get_profile().identity_for_roles(configured) != get_profile().identity


def test_complete_override_supports_new_template_schema_and_palette():
    original = ALL_AGENT_ROLES["oracles"]
    replacement = AgentRole(
        name="oracles", system="complete system", template="{query}",
        schema={"answer": str}, tools=["Read"], model="fixture", max_turns=2
    )
    profile = VerificationProfile(name="custom", role_overrides={"oracles": replacement})
    assert profile.apply_roles(ALL_AGENT_ROLES)["oracles"] is replacement
    assert ALL_AGENT_ROLES["oracles"] is original
    assert original.tools == ["Read", "Grep", "Glob", "WebSearch", "WebFetch"]
    with pytest.raises(ValueError, match="registry key"):
        VerificationProfile(role_overrides={"wrong": replacement}).apply_roles({})


def test_compiler_and_critic_are_answer_blind_and_preserve_security_contract():
    for name in ("obligations", "critic"):
        system = SYSTEMS[name]
        assert ANSWER_BLIND in system
        assert AXIOM_NECESSITY in system
        assert AXIOM_INDEPENDENCE in system
        assert "do not inspect solver hypotheses, candidate contents, evaluator outcomes" in system
        assert "do not residualize it against solver-derived premises" in system
        assert "every task-defined artifact content and security requirement" in system
        assert "Without trusted candidate scope, completion covers the full actual task contract" in system


def test_candidate_scope_is_controller_only_and_preserves_all_security_requirements():
    for system in SYSTEMS.values():
        assert SCOPE in system
        assert "cannot select or narrow certification scope" in system
        assert "pending until authorization" in system
        assert "not pre-certificate proof requirements" in system
    for name in ("oracles", "discharge", "judger"):
        assert SECURITY in SYSTEMS[name]
        assert AXIOM_WITNESS in SYSTEMS[name]
    assert "distinguish" in get_profile("cyber").solver_instructions.lower()


def test_cyber_prompts_are_general_and_have_no_extra_candidate_decision_policy():
    corpus = "\n".join(SYSTEMS.values()) + get_profile("cyber").solver_instructions
    for forbidden in (
        "cybergym", "arvo", "oss-fuzz", "sanitycheck", "fixed evaluator",
        "raw bytes", "raw-byte", "deadline", "ranking", "readiness",
        "minimum controls", "resume three", "fresh thread",
    ):
        assert forbidden not in corpus.lower()
    assert not re.search(r"CVE-\d|/task/|/evaluator/", corpus, re.IGNORECASE)
    assert all(len(system.split()) < 1_500 for system in SYSTEMS.values())
    assert "ordinary GV claim graph" in get_profile("cyber").solver_instructions


def test_profile_import_does_not_load_agents_or_backend():
    script = (
        "import sys; import verifier.profiles; "
        "assert 'verifier.llm.agents' not in sys.modules; "
        "assert 'verifier.llm.codex_backend' not in sys.modules; "
        "assert 'claude_agent_sdk' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


def test_profile_selection_rejects_unknown_profiles():
    assert get_profile("plain").name == "default"
    profile = get_profile("cyber")
    assert get_profile(profile) is profile
    with pytest.raises(ValueError, match="unknown verification profile"):
        get_profile("unknown")


@pytest.mark.parametrize("change", [
    {"version": 2},
    {"solver_instructions": "custom solver policy"},
    {"disable_external_search": True},
    {"role_overrides": {}},
    {"role_overrides": {"judger": ALL_AGENT_ROLES["judger"]}},
])
def test_reserved_default_profile_rejects_custom_configuration_at_selection(change):
    default = VerificationProfile()
    assert get_profile(default) is default
    custom = replace(default, **change)
    with pytest.raises(ValueError, match="reserved default"):
        get_profile(custom)
    # Low-level registry validation remains available independently.
    assert custom.apply_roles(ALL_AGENT_ROLES).keys() == ALL_AGENT_ROLES.keys()


def test_offline_config_rejects_reserved_default_role_override():
    from verifier.offline import config_fields
    custom = VerificationProfile(role_overrides={"judger": ALL_AGENT_ROLES["judger"]})
    with pytest.raises(ValueError, match="reserved default"):
        config_fields(seats="codex", model="fixture", profile=custom)
