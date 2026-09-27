"""The discharger interface (execution.discharger): what the binding game
may ask an executor, the defaults of one that declares nothing, and a
duck-typed discharger read with the same defaults."""
from __future__ import annotations

from verifier.execution.adapters import AdapterRoutingDischarger
from verifier.execution.discharge import AgentDischarger
from verifier.execution.discharger import Discharger, discharger_of


class OnlyDischarges:
    def discharge(self, row, obligation, method=None, tag=""):
        return ("discharged", tag)


def test_a_bare_discharger_declares_nothing_beyond_discharge():
    bare = discharger_of(OnlyDischarges())
    assert bare.discharge_with_repair({}, None, tag="redo") == (
        "discharged", "redo")
    assert not bare.runs_scripts and not bare.reuses_adapters
    assert bare.adapter_catalog() == "[]"
    assert bare.oracle_adapter_catalog("[1]", "world") == "[1]"
    assert bare.prepare_method({"a": 1}) == {"a": 1}
    assert bare.supports_method_reuse({}) is False
    assert bare.discharge_adapter_only({}, None, {}) == (
        None, "discharger has no authenticated adapter-only path")
    assert bare.registry is None and bare.workspace_dir == ""


def test_a_duck_typed_discharger_is_read_through_its_members():
    class Scripted(OnlyDischarges):
        registry = "registry"

        def run_script(self, row, obligation, method, tag=""):
            return "ran"

        def discharge_with_repair(self, row, obligation, method=None,
                                  tag="", *, previous_artifact=None):
            return ("repaired", previous_artifact)

    class Wrapper:
        def __init__(self, inner):
            self.inner = inner

    inner = OnlyDischarges()
    inner.workspace_dir = "/work"
    duck = discharger_of(Scripted())
    assert duck.runs_scripts and duck.run_script({}, None, {}) == "ran"
    assert duck.discharge_with_repair({}, None, previous_artifact="a") == (
        "repaired", "a")
    assert duck.registry == "registry"
    assert discharger_of(Wrapper(inner)).workspace_dir == "/work"


def test_the_shipped_dischargers_declare_their_capabilities():
    assert issubclass(AgentDischarger, Discharger)
    assert issubclass(AdapterRoutingDischarger, Discharger)
    assert AgentDischarger.runs_scripts and not AgentDischarger.reuses_adapters
    assert AdapterRoutingDischarger.runs_scripts
    assert AdapterRoutingDischarger.reuses_adapters


def test_the_router_reports_its_executors_workspace(tmp_path):
    from verifier.core.events import EventLog
    from verifier.execution.adapters import default_adapter_registry
    from verifier.execution.instruments import ArtifactStore

    executor = OnlyDischarges()
    executor.workspace_dir = str(tmp_path / "workspace")
    router = AdapterRoutingDischarger(
        executor, default_adapter_registry(),
        ArtifactStore(str(tmp_path / "art")),
        EventLog(str(tmp_path / "events.jsonl")), str(tmp_path))
    assert router.workspace_dir == executor.workspace_dir
