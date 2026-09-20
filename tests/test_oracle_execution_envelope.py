"""Contract tests for the Oracle's atomic execution binding."""
from __future__ import annotations

import copy
import re

import pytest

from verifier.llm.agents import ENUMERATE_ORACLES_AGENT, to_json_schema
from verifier.llm.oracle_execution import lower_oracle_execution_payload
from verifier.llm.schema import FixtureBackend, RoleError


_EXECUTION_FIELDS = {
    "kind", "custom_gap_reason", "adapter", "adapter_version",
    "adapter_input_binding", "adapter_arguments", "adapter_inputs",
}


def _base_method() -> dict:
    return {
        "procedure": "Apply the declared method to its declared inputs.",
        "artifact": "A typed execution receipt.",
        "level": "obligation",
        "basis": "record",
        "strength": "decisive",
        "bridges": [],
    }


def _adapter_execution(binding: str = "explicit-v1") -> dict:
    return {
        "kind": "adapter", "custom_gap_reason": "",
        "adapter": "generic.adapter", "adapter_version": 7,
        "adapter_input_binding": binding,
        "adapter_arguments": '{"predicate":"declared-by-oracle"}',
        "adapter_inputs": [{
            "ref": "ledger://declared/input",
            "provenance": "ledger", "role": "input",
        }],
    }


def _custom_execution() -> dict:
    return {
        "kind": "custom",
        "custom_gap_reason": "No registered adapter exposes this operation.",
        "adapter": "", "adapter_version": 0,
        "adapter_input_binding": "none", "adapter_arguments": "",
        "adapter_inputs": [],
    }


def _wire_payload(execution: dict) -> dict:
    return {"methods": [{**_base_method(), "execution": execution}]}


def _oracle_variables() -> dict:
    names = set(re.findall(r"{(\w+)}", ENUMERATE_ORACLES_AGENT.template))
    return {name: "" for name in names}


def test_wire_schema_requires_the_atomic_execution_envelope():
    wire_schema = ENUMERATE_ORACLES_AGENT.wire_schema
    assert wire_schema is not None
    json_schema = to_json_schema(wire_schema)
    method_schema = json_schema["properties"]["methods"]["items"]
    assert "execution" in method_schema["required"]
    assert method_schema["additionalProperties"] is False
    execution_schema = method_schema["properties"]["execution"]
    assert set(execution_schema["properties"]) == _EXECUTION_FIELDS
    assert set(execution_schema["required"]) == _EXECUTION_FIELDS
    assert execution_schema["additionalProperties"] is False


@pytest.mark.parametrize(("execution", "expected_binding"), [
    (_adapter_execution("explicit-v1"), None),
    (_adapter_execution("fixed-v1"), "fixed-v1"),
])
def test_lowering_preserves_a_complete_adapter_choice(
    execution: dict, expected_binding: str | None,
):
    payload = _wire_payload(execution)
    before = copy.deepcopy(payload)
    lowered = lower_oracle_execution_payload(payload)
    assert payload == before
    method = lowered["methods"][0]
    assert "execution" not in method
    assert method["adapter"] == execution["adapter"]
    assert method["adapter_version"] == execution["adapter_version"]
    assert method["adapter_arguments"] == execution["adapter_arguments"]
    assert method["adapter_inputs"] == execution["adapter_inputs"]
    if expected_binding is None:
        assert "adapter_input_binding" not in method
    else:
        assert method["adapter_input_binding"] == expected_binding
    assert {key: method[key] for key in _base_method()} == _base_method()


def test_lowering_preserves_custom_method_and_strips_diagnostic_reason():
    payload = _wire_payload(_custom_execution())
    before = copy.deepcopy(payload)
    lowered = lower_oracle_execution_payload(payload)
    assert payload == before
    assert lowered == {"methods": [_base_method()]}


def test_lowering_carries_a_custom_method_executable_form():
    """The oracle may hand a custom method its EXECUTABLE FORM: the wire
    contract carries `script` as an optional method field and lowering
    keeps it, so the executor runs it verbatim instead of seating an
    agent."""
    payload = _wire_payload(_custom_execution())
    payload["methods"][0]["script"] = "#!/bin/bash\ngit -C /app status --short"
    lowered = lower_oracle_execution_payload(payload)
    assert lowered["methods"][0]["script"] == \
        "#!/bin/bash\ngit -C /app status --short"
    assert {k: v for k, v in lowered["methods"][0].items() if k != "script"} \
        == _base_method()


def test_lowering_drops_an_empty_script_and_rejects_one_on_an_adapter():
    payload = _wire_payload(_custom_execution())
    payload["methods"][0]["script"] = ""
    assert lower_oracle_execution_payload(payload) == {"methods": [_base_method()]}

    bound = _wire_payload(_adapter_execution())
    bound["methods"][0]["script"] = "echo bound"
    with pytest.raises(RoleError, match="executable form is for custom"):
        lower_oracle_execution_payload(bound)


def test_wire_schema_offers_the_executable_form():
    json_schema = to_json_schema(ENUMERATE_ORACLES_AGENT.wire_schema)
    method_schema = json_schema["properties"]["methods"]["items"]
    assert "script" in method_schema["properties"]
    assert "script" not in method_schema["required"]


@pytest.mark.parametrize("missing", sorted(_EXECUTION_FIELDS))
def test_lowering_rejects_every_incomplete_execution_envelope(missing: str):
    execution = _adapter_execution()
    del execution[missing]
    with pytest.raises(RoleError):
        lower_oracle_execution_payload(_wire_payload(execution))


def test_lowering_rejects_unknown_fields_at_every_wire_level():
    payloads = []
    top = _wire_payload(_adapter_execution())
    top["unexpected"] = True
    payloads.append(top)
    method = _wire_payload(_adapter_execution())
    method["methods"][0]["unexpected"] = True
    payloads.append(method)
    execution = _wire_payload(_adapter_execution())
    execution["methods"][0]["execution"]["unexpected"] = True
    payloads.append(execution)
    adapter_input = _wire_payload(_adapter_execution())
    adapter_input["methods"][0]["execution"]["adapter_inputs"][0][
        "unexpected"
    ] = True
    payloads.append(adapter_input)
    for payload in payloads:
        with pytest.raises(RoleError):
            lower_oracle_execution_payload(payload)


def test_lowering_rejects_boolean_adapter_version():
    execution = _adapter_execution()
    execution["adapter_version"] = True
    with pytest.raises(RoleError):
        lower_oracle_execution_payload(_wire_payload(execution))


@pytest.mark.parametrize(
    "arguments", ["", "[]", "null", '{"x":NaN}', '{"x":1,"x":2}'],
)
def test_lowering_rejects_non_object_or_non_strict_adapter_arguments(
    arguments: str,
):
    execution = _adapter_execution()
    execution["adapter_arguments"] = arguments
    with pytest.raises(RoleError):
        lower_oracle_execution_payload(_wire_payload(execution))


@pytest.mark.parametrize("execution", [
    {**_adapter_execution(), "custom_gap_reason": "claimed gap"},
    {**_adapter_execution(), "adapter": ""},
    {**_adapter_execution(), "adapter_version": 0},
    {**_adapter_execution(), "adapter_input_binding": "none"},
    {**_custom_execution(), "custom_gap_reason": ""},
    {**_custom_execution(), "adapter": "generic.adapter"},
    {**_custom_execution(), "adapter_version": 1},
    {**_custom_execution(), "adapter_input_binding": "explicit-v1"},
    {**_custom_execution(), "adapter_arguments": "{}"},
    {**_custom_execution(), "adapter_inputs": [{
        "ref": "ledger://declared/input", "provenance": "ledger",
        "role": "input",
    }]},
])
def test_lowering_rejects_mixed_adapter_and_custom_sentinels(execution: dict):
    with pytest.raises(RoleError):
        lower_oracle_execution_payload(_wire_payload(execution))


def test_default_fixture_backend_remains_legacy_compatible():
    legacy = {"methods": [{
        **_base_method(), "adapter": "generic.adapter",
        "adapter_version": 7, "adapter_arguments": "{}",
        "adapter_inputs": [],
    }]}
    backend = FixtureBackend({"oracles": copy.deepcopy(legacy)})
    result = ENUMERATE_ORACLES_AGENT.call(backend, _oracle_variables())
    assert result == legacy
    assert "execution" not in result["methods"][0]


def test_wire_fixture_is_lowered_before_reaching_legacy_consumers():
    wire = _wire_payload(_adapter_execution("fixed-v1"))
    before = copy.deepcopy(wire)
    backend = FixtureBackend({"oracles": wire}, wire_output=True)
    result = ENUMERATE_ORACLES_AGENT.call(backend, _oracle_variables())
    assert wire == before
    assert "execution" not in result["methods"][0]
    assert result["methods"][0]["adapter"] == "generic.adapter"
    assert result["methods"][0]["adapter_input_binding"] == "fixed-v1"


def test_raise_is_the_exact_inverse_of_lowering_for_fixtures():
    from verifier.llm.oracle_execution import raise_oracle_execution_payload

    adapter_flat = {**_base_method(), "adapter": "generic.adapter",
                    "adapter_version": 7,
                    "adapter_arguments": '{"predicate":"declared-by-oracle"}',
                    "adapter_inputs": [{"ref": "ledger://declared/input",
                                        "provenance": "ledger",
                                        "role": "input"}]}
    fixed_flat = {**adapter_flat, "adapter_input_binding": "fixed-v1"}
    custom_flat = _base_method()
    flat = {"methods": [adapter_flat, fixed_flat, custom_flat]}
    lifted = raise_oracle_execution_payload(flat)
    assert [m["execution"]["kind"] for m in lifted["methods"]] \
        == ["adapter", "adapter", "custom"]
    assert lifted["methods"][1]["execution"]["adapter_input_binding"] \
        == "fixed-v1"
    assert lower_oracle_execution_payload(lifted) == flat


def test_raise_refuses_a_partial_adapter_binding():
    from verifier.llm.oracle_execution import raise_oracle_execution_payload

    partial = {**_base_method(), "adapter": "generic.adapter"}
    with pytest.raises(ValueError, match="partial adapter binding"):
        raise_oracle_execution_payload({"methods": [partial]})
