"""What the binding game may ask of the thing that carries a method out.

``discharge`` is the one required member. Every other capability — typed
adapters, scripts run verbatim, a repair-aware redo, accepted-method reuse,
the durable workspace — is declared here with the answer a discharger that
lacks it gives, so the game asks through these members instead of probing
for attributes. ``discharger_of`` reads a duck-typed discharger (a test
double, an experiment's stub) with the same defaults.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class Discharger:
    """A discharger's declared capabilities, with their defaults."""

    # the typed-adapter registry whose evidence validator authenticates a
    # reused method's fresh envelope; None: no adapters
    registry = None

    def discharge(self, row: Dict, obligation, method=None, tag: str = ""):
        raise NotImplementedError

    def discharge_with_repair(self, row: Dict, obligation, method=None,
                              tag: str = "", *, previous_artifact=None):
        """A redo of a bound method (the previous attempt's artifact is
        repair context for an executor that can use it)."""
        return self.discharge(row, obligation, method=method, tag=tag)

    # whether a method's executable form can run verbatim here
    runs_scripts = False
    # whether a reused method's typed adapter can run alone here
    reuses_adapters = False
    # the executor's durable workspace ("" when it has none)
    workspace_dir = ""

    def run_script(self, row: Dict, obligation, method, tag: str = ""):
        raise NotImplementedError

    def adapter_catalog(self) -> str:
        """The typed adapters the oracle may bind, as a JSON catalog."""
        return "[]"

    def oracle_adapter_catalog(self, full_catalog: str,
                               required_basis: str) -> str:
        """The catalog as the oracle is shown it for this evidence floor."""
        return full_catalog

    def prepare_method(self, method: Dict[str, Any]) -> Dict[str, Any]:
        """An enumerated method made ready to execute (fixed adapter
        inputs bound)."""
        return method

    def supports_method_reuse(self, method: Dict[str, Any]) -> bool:
        """Whether an accepted method may be reused on another candidate."""
        return False

    def discharge_adapter_only(self, row: Dict, obligation,
                               method) -> Tuple[Optional[Any], str]:
        """Run a reused method's typed adapter alone: (result, "") or
        (None, why not)."""
        return None, "discharger has no authenticated adapter-only path"


class _DuckDischarger(Discharger):
    """A discharger that does not derive from Discharger, read with the
    defaults: members it has are used, the rest take the default."""

    def __init__(self, discharger: Any):
        self._discharger = discharger

    def _member(self, name: str):
        member = getattr(self._discharger, name, None)
        return member if callable(member) else None

    @property
    def registry(self):
        return getattr(self._discharger, "registry", None)

    def discharge(self, row, obligation, method=None, tag=""):
        return self._discharger.discharge(row, obligation, method=method,
                                          tag=tag)

    def discharge_with_repair(self, row, obligation, method=None, tag="", *,
                              previous_artifact=None):
        member = self._member("discharge_with_repair")
        if member is None:
            return self._discharger.discharge(row, obligation, method=method,
                                              tag=tag)
        return member(row, obligation, method=method, tag=tag,
                      previous_artifact=previous_artifact)

    @property
    def runs_scripts(self) -> bool:
        return hasattr(self._discharger, "run_script")

    @property
    def reuses_adapters(self) -> bool:
        return self._member("discharge_adapter_only") is not None

    def run_script(self, row, obligation, method, tag=""):
        return self._discharger.run_script(row, obligation, method=method,
                                           tag=tag)

    def adapter_catalog(self) -> str:
        member = self._member("adapter_catalog")
        return member() if member else "[]"

    def oracle_adapter_catalog(self, full_catalog, required_basis):
        member = self._member("oracle_adapter_catalog")
        return (member(full_catalog, required_basis) if member
                else full_catalog)

    def prepare_method(self, method):
        member = self._member("prepare_method")
        return member(method) if member else method

    def supports_method_reuse(self, method) -> bool:
        member = self._member("supports_method_reuse")
        return member(method) if member else False

    def discharge_adapter_only(self, row, obligation, method):
        member = self._member("discharge_adapter_only")
        if member is None:
            return Discharger.discharge_adapter_only(self, row, obligation,
                                                     method)
        return member(row, obligation, method)

    @property
    def workspace_dir(self) -> str:
        return _workspace_through_wrappers(self._discharger)


def _workspace_through_wrappers(discharger) -> str:
    """The workspace of a duck-typed discharger, through any wrapper that
    names its wrapped executor ``inner`` or ``agent``."""
    seen = set()
    node = discharger
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        value = getattr(node, "workspace_dir", None)
        if isinstance(value, str) and value:
            return value
        node = getattr(node, "inner", None) or getattr(node, "agent", None)
    return ""


def discharger_of(discharger: Any) -> Discharger:
    """The declared capabilities of ``discharger``."""
    if isinstance(discharger, Discharger):
        return discharger
    return _DuckDischarger(discharger)
