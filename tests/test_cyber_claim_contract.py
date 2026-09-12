"""Cyber specializes claim/method semantics, not GV proof or wire contracts.

No models, benchmark tasks, evaluator submissions, or runtime credentials.
"""
import hashlib
import inspect
from dataclasses import replace

import pytest

from verifier.llm.agents import ALL_AGENT_ROLES, to_json_schema
from verifier.llm.oracle_execution import ORACLE_EXECUTION_WIRE_SCHEMA, lower_oracle_execution_payload
from verifier.profiles import VerificationProfile, get_profile
from verifier.profiles.cyber import build_roles
from verifier.profiles.cyber.claims import CLAIM_DEFINITION
from verifier.profiles.cyber.methods import METHOD_PRACTICES
from verifier.profiles.cyber.prompts import SECURITY, SOLVER_INSTRUCTIONS, SYSTEMS


def _lower_first(payload):
    return {"methods": payload["methods"]}


def _lower_second(payload):
    return {"methods": list(payload["methods"])}


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_every_cyber_role_has_one_explicit_security_claim_contract(name):
    roles = build_roles()
    assert set(roles) == set(SYSTEMS)
    assert roles[name].system.count(CLAIM_DEFINITION) == 1
    assert "not a new claim kind" in roles[name].system


def test_solver_claim_guidance_preserves_scope_quantifier_and_authority():
    assert SOLVER_INSTRUCTIONS.count(CLAIM_DEFINITION) == 1
    assert "only material dimensions" in CLAIM_DEFINITION
    assert "unspecified facts remain unknown" in CLAIM_DEFINITION
    assert "controlling authority" in CLAIM_DEFINITION
    assert "One admissible witness may establish an existential claim" in CLAIM_DEFINITION
    assert "not merely successful finite tests" in CLAIM_DEFINITION
    assert "never a solver-narrowed scope" in CLAIM_DEFINITION
    assert "only when the task or the claimed proposition requires them" in SECURITY


@pytest.mark.parametrize("name", ["oracles", "discharge"])
def test_oracle_and_executor_share_general_practices_without_new_requirements(name):
    system = build_roles()[name].system
    assert system.count(METHOD_PRACTICES) == 1
    assert "guidance, not extra requirements" in METHOD_PRACTICES
    assert "exact operations" in METHOD_PRACTICES
    assert "Independent Judger" in METHOD_PRACTICES
    assert "Neither may read hidden reference answers, invoke a private evaluator" in METHOD_PRACTICES


def test_compilers_compile_conditions_while_executor_preserves_bound_method():
    roles = build_roles()
    assert "State what must be true, not how to prove it" in roles["obligations"].system
    assert "Compile, do not execute or predict results" in roles["oracles"].system
    assert "Availability selects an operation, never acceptance predicates" in roles["oracles"].system
    assert "Execute this exact bound method" in roles["discharge"].system
    assert "A substituted path, interface, build, or instrument is a different method" in roles["discharge"].system
    assert "never authorizes a candidate, substitutes for the Judger, or issues evidence" in roles["sufficiency"].system


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_cyber_roles_inherit_full_original_output_wire_tools_and_turn_contracts(name):
    base = ALL_AGENT_ROLES[name]
    role = build_roles()[name]
    assert role.schema == base.schema
    assert role.wire_schema is base.wire_schema
    assert role.wire_lower is base.wire_lower
    assert role.max_turns == base.max_turns
    assert role.model == base.model
    assert role.template.endswith(base.template)
    assert "{profile_context}" in role.template
    assert role.tools == [tool for tool in base.tools if tool not in {"WebSearch", "WebFetch"}]


def test_oracle_keeps_atomic_adapter_or_custom_execution_wire():
    role = build_roles()["oracles"]
    assert role.wire_schema is ORACLE_EXECUTION_WIRE_SCHEMA
    assert role.wire_lower is lower_oracle_execution_payload
    execution = to_json_schema(role.wire_schema)["properties"]["methods"]["items"]["properties"]["execution"]
    assert execution["additionalProperties"] is False
    assert execution["properties"]["kind"]["enum"] == ["adapter", "custom"]
    assert set(execution["required"]) == set(execution["properties"])


def test_profile_identity_includes_actual_wire_and_lowering_implementation():
    configuration = get_profile("cyber").identity["roles"]["oracles"]
    assert configuration["wire_schema"] == to_json_schema(ORACLE_EXECUTION_WIRE_SCHEMA)
    lowering = configuration["wire_lowering"]
    assert lowering["module"] == lower_oracle_execution_payload.__module__
    assert lowering["name"] == lower_oracle_execution_payload.__qualname__
    expected = hashlib.sha256(inspect.getsource(lower_oracle_execution_payload).encode("utf-8")).hexdigest()
    assert lowering["implementation_sha256"] == expected
    module_source = inspect.getsource(inspect.getmodule(lower_oracle_execution_payload))
    assert lowering["module_sha256"] == hashlib.sha256(module_source.encode("utf-8")).hexdigest()


def test_different_static_lowering_contracts_have_different_profile_ids():
    base = build_roles()["oracles"]
    first = VerificationProfile(name="contract-fixture", role_overrides={"oracles": replace(base, wire_lower=_lower_first)})
    second = VerificationProfile(name="contract-fixture", role_overrides={"oracles": replace(base, wire_lower=_lower_second)})
    assert first.id != second.id
    assert first.id == first.id and second.id == second.id


@pytest.mark.parametrize("key", ["methods", "different"])
def test_unversioned_closure_lowering_cannot_enter_profile_identity(key):
    def lower(payload):
        return {key: payload["methods"]}

    base = build_roles()["oracles"]
    profile = VerificationProfile(name="closure-fixture", role_overrides={"oracles": replace(base, wire_lower=lower)})
    assert lower.__closure__
    with pytest.raises(ValueError, match="unversioned closure state"):
        _ = profile.id


def test_different_model_facing_wire_schema_has_different_profile_id():
    base = build_roles()["oracles"]
    first = VerificationProfile(name="contract-fixture", role_overrides={"oracles": base})
    second = VerificationProfile(name="contract-fixture", role_overrides={"oracles": replace(base, wire_schema={"different": bool})})
    assert first.id != second.id


def test_plain_role_objects_are_not_replaced_by_cyber_specialization():
    roles = get_profile("plain").apply_roles(ALL_AGENT_ROLES)
    assert all(roles[name] is role for name, role in ALL_AGENT_ROLES.items())
