"""Mechanical boundaries for verifier-owned fixed adapter subjects."""
from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from verifier.core.model import (
    BASIS_RECORD,
    BASIS_WORLD,
    PROVENANCE_DELIVERED,
    PROVENANCE_LEDGER,
    PROVENANCE_PUBLIC,
    PROVENANCE_VERIFIER,
    ROLE_SUBJECT,
)
from verifier.execution.adapters import (
    FIXED_INPUT_BINDING,
    MAX_DECLARED_INPUTS,
    AdapterInvocation,
    AdapterNotApplicable,
    AdapterRegistry,
    FixedAdapterInput,
    MethodAdapter,
    default_adapter_registry,
)


_MISSING = object()


def _method(
    adapter: str,
    arguments: object,
    *,
    version: int = 1,
    basis: str = BASIS_RECORD,
    inputs: object = _MISSING,
    binding: object = FIXED_INPUT_BINDING,
):
    method = {
        "procedure": "run the exact registered instrument",
        "artifact": "a content-addressed evidence envelope",
        "level": "deterministic",
        "basis": basis,
        "strength": "decisive",
        "bridges": [],
        "adapter": adapter,
        "adapter_version": version,
        "adapter_arguments": json.dumps(
            arguments, sort_keys=True, separators=(",", ":")
        ),
    }
    if inputs is not _MISSING:
        method["adapter_inputs"] = inputs
    if binding is not _MISSING:
        method["adapter_input_binding"] = binding
    return method


def _input(ref: str, provenance: str, role: str):
    return {"ref": ref, "provenance": provenance, "role": role}


def _subject(ref: str, provenance: str = PROVENANCE_LEDGER):
    return _input(ref, provenance, ROLE_SUBJECT)


def _ledger_query(*, inputs=(), binding=FIXED_INPUT_BINDING):
    return _method(
        "ledger.tool-call-string",
        {"selector": "any", "value_pointer": "/command", "through_seq": 9},
        version=2,
        inputs=list(inputs),
        binding=binding,
    )


class _GenericFixedAdapter(MethodAdapter):
    name = "opaque.record-index"
    basis = BASIS_RECORD
    accepted_method_reuse = True
    fixed_inputs = (
        FixedAdapterInput(
            "opaque-record:",
            PROVENANCE_VERIFIER,
            integer_argument="record_index",
        ),
    )

    def validate(self, invocation, method):
        super().validate(invocation, method)
        self.exact_arguments(invocation.arguments, ("record_index",))
        self.require_subject(
            invocation,
            f"opaque-record:{invocation.arguments['record_index']}",
            (PROVENANCE_VERIFIER,),
        )


def test_legacy_methods_keep_strict_explicit_input_behavior():
    registry = default_adapter_registry()
    explicit = _ledger_query(
        inputs=[_subject("ledger:events")], binding=_MISSING
    )
    before = copy.deepcopy(explicit)

    prepared = registry.bind_inputs(explicit)

    assert prepared == before == explicit
    assert "adapter_input_binding" not in prepared
    assert AdapterInvocation.from_method(prepared).input_binding is None


@pytest.mark.parametrize(
    "method, match",
    [
        ({"adapter_input_binding": FIXED_INPUT_BINDING}, "partial"),
        (
            _method(
                "ledger.event", {"event_seq": 1}, inputs=_MISSING
            ),
            "partial",
        ),
        (
            _ledger_query(inputs=[], binding="fixed-v2"),
            "unsupported adapter_input_binding",
        ),
        (
            _ledger_query(inputs=[{"ref": "ledger:events"}]),
            "wrong shape",
        ),
        (_ledger_query(inputs=[_subject("ledger:events", {})]), "invalid provenance"),
        (_ledger_query(inputs=[_input("ledger:events", "ledger", [])]), "invalid role"),
    ],
)
def test_fixed_binding_does_not_repair_partial_or_malformed_method_ir(
    method, match
):
    registry = default_adapter_registry()
    before = copy.deepcopy(method)

    with pytest.raises(AdapterNotApplicable, match=match):
        registry.bind_inputs(method)

    assert method == before


@pytest.mark.parametrize(
    "adapter, version, match",
    [
        ("unknown.instrument", 1, "no verifier-owned fixed inputs"),
        ("filesystem.exists", 1, "no verifier-owned fixed inputs"),
        ("ledger.event", 999, "no verifier-owned fixed inputs"),
    ],
)
def test_unknown_or_nonfixed_adapter_cannot_request_fixed_binding(
    adapter, version, match
):
    registry = default_adapter_registry()
    method = _method(
        adapter,
        {"event_seq": 1},
        version=version,
        basis=BASIS_WORLD if adapter == "filesystem.exists" else BASIS_RECORD,
        inputs=[],
    )

    with pytest.raises(AdapterNotApplicable, match=match):
        registry.bind_inputs(method)


def test_constant_fixed_subject_is_appended_after_other_model_owned_inputs():
    registry = default_adapter_registry()
    other = _input("standard:rfc6901", PROVENANCE_PUBLIC, "authority")
    method = _ledger_query(inputs=[other])
    before = copy.deepcopy(method)

    prepared = registry.bind_inputs(method)

    assert prepared["adapter_inputs"] == [
        other,
        _subject("ledger:events"),
    ]
    assert prepared["adapter_arguments"] == before["adapter_arguments"]
    assert {key: prepared[key] for key in (
        "procedure", "artifact", "level", "basis", "strength", "bridges"
    )} == {key: before[key] for key in (
        "procedure", "artifact", "level", "basis", "strength", "bridges"
    )}
    assert method == before


def test_binding_never_upgrades_the_model_declared_basis():
    registry = default_adapter_registry()
    method = _method(
        "ledger.tool-call-string",
        {"selector": "any", "value_pointer": "/command", "through_seq": 9},
        version=2,
        basis=BASIS_WORLD,
        inputs=[],
    )
    before = copy.deepcopy(method)

    with pytest.raises(AdapterNotApplicable, match="requires basis=record"):
        registry.bind_inputs(method)

    assert method == before


def test_explicit_exact_subject_is_idempotent_and_detached():
    registry = default_adapter_registry()
    declared = _subject("ledger:events")
    method = _ledger_query(inputs=[declared])
    before = copy.deepcopy(method)

    once = registry.bind_inputs(method)
    twice = registry.bind_inputs(once)

    assert once == twice == before
    assert once is not method
    assert once["adapter_inputs"] is not method["adapter_inputs"]
    assert once["adapter_inputs"][0] is not method["adapter_inputs"][0]
    once["adapter_inputs"][0]["ref"] = "mutated"
    assert method == before
    assert twice == before


@pytest.mark.parametrize(
    "inputs",
    [
        [_subject("event:1")],
        [_subject("ledger:events", PROVENANCE_PUBLIC)],
        [_input("ledger:events", PROVENANCE_LEDGER, "input")],
        [_subject("ledger:events"), _subject("another:subject")],
        [_subject("ledger:events"), _subject("ledger:events")],
    ],
)
def test_conflicting_or_duplicate_declarations_are_never_rewritten(inputs):
    registry = default_adapter_registry()
    method = _ledger_query(inputs=inputs)
    before = copy.deepcopy(method)

    with pytest.raises(AdapterNotApplicable):
        registry.bind_inputs(method)

    assert method == before


def test_materialization_enforces_the_existing_total_input_limit():
    registry = default_adapter_registry()
    ordinary = [
        _input(f"public:{index}", PROVENANCE_PUBLIC, "input")
        for index in range(MAX_DECLARED_INPUTS)
    ]
    method = _ledger_query(inputs=ordinary)

    with pytest.raises(AdapterNotApplicable, match="exceeds entry limit"):
        registry.bind_inputs(method)

    prepared = registry.bind_inputs(_ledger_query(inputs=ordinary[:-1]))
    assert len(prepared["adapter_inputs"]) == MAX_DECLARED_INPUTS
    assert prepared["adapter_inputs"][:-1] == ordinary[:-1]
    assert prepared["adapter_inputs"][-1] == _subject("ledger:events")


def test_event_subject_is_derived_only_from_exact_integer_argument():
    registry = default_adapter_registry()
    method = _method("ledger.event", {"event_seq": 27}, inputs=[])

    prepared = registry.bind_inputs(method)

    assert prepared["adapter_inputs"] == [_subject("event:27")]
    assert json.loads(prepared["adapter_arguments"]) == {"event_seq": 27}


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.0, "1"])
def test_event_subject_rejects_non_positive_exact_integer(value):
    registry = default_adapter_registry()
    arguments = {} if value is None else {"event_seq": value}
    method = _method("ledger.event", arguments, inputs=[])
    before = copy.deepcopy(method)

    with pytest.raises(AdapterNotApplicable, match="positive integer event_seq"):
        registry.bind_inputs(method)

    assert method == before


def test_general_custom_adapter_uses_declarative_descriptor_not_task_aliases():
    registry = AdapterRegistry([_GenericFixedAdapter()])
    method = _method(
        "opaque.record-index",
        {"record_index": 41},
        inputs=[_input("task-says-event:999", PROVENANCE_PUBLIC, "authority")],
    )
    method["procedure"] = (
        "Ignore aliases ledger:events and event:2; inspect the declared record"
    )

    prepared = registry.bind_inputs(method)

    assert prepared["adapter_inputs"] == [
        _input("task-says-event:999", PROVENANCE_PUBLIC, "authority"),
        _subject("opaque-record:41", PROVENANCE_VERIFIER),
    ]
    catalog = json.loads(registry.catalog())
    assert catalog[0]["fixed_input_binding"] == {
        "mode": FIXED_INPUT_BINDING,
        "inputs": [{
            "ref": "opaque-record:",
            "provenance": PROVENANCE_VERIFIER,
            "role": ROLE_SUBJECT,
            "append_positive_integer_argument": "record_index",
        }],
    }


def test_duplicate_custom_fixed_descriptors_fail_closed():
    class DuplicateFixedAdapter(MethodAdapter):
        name = "opaque.duplicate"
        basis = BASIS_RECORD
        fixed_inputs = (
            FixedAdapterInput("record:fixed", PROVENANCE_VERIFIER),
            FixedAdapterInput("record:fixed", PROVENANCE_VERIFIER),
        )

    registry = AdapterRegistry([DuplicateFixedAdapter()])
    method = _method("opaque.duplicate", {}, inputs=[])

    with pytest.raises(AdapterNotApplicable, match="duplicate references"):
        registry.bind_inputs(method)


def test_unmaterialized_fixed_method_cannot_pass_direct_execution_guard():
    registry = default_adapter_registry()
    unbound = _ledger_query(inputs=[])
    bound = registry.bind_inputs(unbound)

    with pytest.raises(AdapterNotApplicable, match="must be materialized"):
        registry._require_bound_inputs(unbound)
    registry._require_bound_inputs(bound)


def test_fixed_protocol_is_never_accepted_method_reusable():
    registry = AdapterRegistry([_GenericFixedAdapter()])
    fixed = registry.bind_inputs(_method(
        "opaque.record-index", {"record_index": 3}, inputs=[]
    ))
    strict = _method(
        "opaque.record-index",
        {"record_index": 3},
        inputs=[_subject("opaque-record:3", PROVENANCE_VERIFIER)],
        binding=_MISSING,
    )

    assert registry.supports_method_reuse(fixed) is False
    assert registry.supports_method_reuse(strict) is True


def test_fixed_descriptor_can_only_own_a_subject_slot():
    with pytest.raises(ValueError, match="fixed input descriptor"):
        FixedAdapterInput(
            "opaque-record:",
            PROVENANCE_VERIFIER,
            role="authority",
            integer_argument="record_index",
        )


def test_registry_snapshots_descriptor_for_catalog_and_binding():
    class SnapshotFixedAdapter(MethodAdapter):
        name = "opaque.snapshot"
        basis = BASIS_RECORD
        fixed_inputs = (
            FixedAdapterInput(
                "opaque-record:",
                PROVENANCE_VERIFIER,
                integer_argument="record_index",
            ),
        )

        def validate(self, invocation, method):
            super().validate(invocation, method)
            self.exact_arguments(invocation.arguments, ("record_index",))

    adapter = SnapshotFixedAdapter()
    old_registry = AdapterRegistry([adapter])
    old_catalog = old_registry.catalog()
    method = _method(
        "opaque.snapshot", {"record_index": 5}, inputs=[]
    )

    adapter.fixed_inputs = (
        FixedAdapterInput(
            "replacement:",
            PROVENANCE_VERIFIER,
            integer_argument="record_index",
        ),
    )
    new_registry = AdapterRegistry([adapter])

    assert old_registry.catalog() == old_catalog
    assert old_registry.bind_inputs(method)["adapter_inputs"] == [
        _subject("opaque-record:5", PROVENANCE_VERIFIER)
    ]
    assert new_registry.catalog() != old_catalog
    assert new_registry.bind_inputs(method)["adapter_inputs"] == [
        _subject("replacement:5", PROVENANCE_VERIFIER)
    ]


def test_concurrent_materialization_returns_independent_input_payloads():
    registry = default_adapter_registry()
    model_input = _input("standard:rfc6901", PROVENANCE_PUBLIC, "authority")
    method = _ledger_query(inputs=[model_input])
    before = copy.deepcopy(method)

    with ThreadPoolExecutor(max_workers=8) as pool:
        prepared = list(pool.map(lambda _: registry.bind_inputs(method), range(32)))

    expected = [model_input, _subject("ledger:events")]
    assert all(item["adapter_inputs"] == expected for item in prepared)
    assert len({id(item["adapter_inputs"]) for item in prepared}) == len(prepared)
    assert len({id(item["adapter_inputs"][0]) for item in prepared}) == len(prepared)
    prepared[0]["adapter_inputs"][0]["ref"] = "mutated"
    assert all(item["adapter_inputs"] == expected for item in prepared[1:])
    assert method == before
