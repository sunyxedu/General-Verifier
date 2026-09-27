"""The oracle seat's side of the binding game: the sufficient methods it
enumerates for an obligation, the batches enumerated ahead of time (the
final-contract precompile), and the boundary notes siblings share."""
from __future__ import annotations

import json
import os
import threading
from typing import Dict, List

from ...core.model import Obligation
from ...execution.instruments import world_evidence_context, world_map
from ...execution.instruments import authoritative_task_context
from ...execution.record_shape import record_shape_json
from ...llm.model_spans import model_scope
from ...llm.schema import RoleError
from ..source_evidence import bind_source_evidence_variables
from .methods import features, method_strength


class MethodSource:
    """Method batches for the obligations of one binding game (the
    ``host``, whose seats and settings it uses)."""

    BOUNDARY_NOTES = 12

    def __init__(self, host):
        self.host = host
        # method batches enumerated ahead of time (the final-contract
        # precompile), keyed by obligation node id: the first-pass
        # enumeration of a matching obligation is served from here
        self.batch_cache: Dict[str, List[Dict[str, str]]] = {}
        # the fixed evidence boundary is one boundary for every check of
        # a claim: an input a sibling check found absent is absent for
        # this one too, so the oracle is told before it lists a method
        # that needs it
        self.boundary: Dict[str, List[str]] = {}
        self.reached: Dict[str, List[str]] = {}
        # prewarms in flight: a commit that needs a batch being enumerated
        # right now waits for that one enumeration instead of starting a
        # second, and never waits for the others
        self.prewarm_events: Dict[str, threading.Event] = {}
        self.lock = threading.Lock()

    def prewarm(self, row: Dict, obligation: Obligation) -> None:
        """Enumerate an obligation's first method batch now and keep it
        for the commit that will need it (and for a revised commit of
        the same contract: the batch is served, never consumed)."""
        with self.lock:
            if obligation.id in self.batch_cache \
                    or obligation.id in self.prewarm_events:
                return
            event = self.prewarm_events[obligation.id] = threading.Event()
        try:
            try:
                methods = self.enumerate(row, obligation, prewarm=True)
            except RoleError as err:
                self.host.log.append(
                    "methods_enumerated",
                    {"claim": row["id"], "obligation": obligation.id,
                     "prewarm": True, "error": str(err)})
                return
            with self.lock:
                self.batch_cache[obligation.id] = methods
        finally:
            with self.lock:
                self.prewarm_events.pop(obligation.id, None)
            event.set()

    def served_batch(self, row: Dict, obligation: Obligation):
        with self.lock:
            cached = self.batch_cache.get(obligation.id)
            pending = self.prewarm_events.get(obligation.id)
        if cached is None and pending is not None:
            pending.wait()
            with self.lock:
                cached = self.batch_cache.get(obligation.id)
        if cached is None:
            return None
        self.host.log.append(
            "methods_enumerated",
            {"claim": row["id"], "obligation": obligation.id,
             "repair": False, "cached": True,
             "methods": [{"procedure": m["procedure"][:80],
                          "level": m.get("level", ""),
                          "basis": m.get("basis", ""),
                          "strength": method_strength(m)}
                         for m in cached]})
        return [dict(m) for m in cached]

    def note_boundary(self, row: Dict, obligation: Obligation,
                       detail: str) -> None:
        with self.lock:
            notes = self.boundary.setdefault(row["id"], [])
            notes.append(f"[{obligation.text[:80]}] {str(detail)[:300]}")
            del notes[:-self.BOUNDARY_NOTES]

    def note_reached(self, row: Dict, result) -> None:
        """The inputs a sibling check actually consumed. A boundary report
        carried only absences, so one check could call an address
        unobtainable while another had just read it — the second finding is
        the one that settles whether the address is reachable."""
        refs = []
        for item in getattr(result, "inputs", None) or ():
            if not isinstance(item, dict):
                continue
            ref = str(item.get("ref") or "").strip()
            if ref and len(ref) <= 200:
                refs.append(ref)
        if not refs:
            return
        with self.lock:
            reached = self.reached.setdefault(row["id"], [])
            for ref in refs:
                if ref not in reached:
                    reached.append(ref)
            del reached[:-self.BOUNDARY_NOTES]

    def sibling_boundary(self, row: Dict) -> str:
        with self.lock:
            notes = list(self.boundary.get(row["id"], ()))
            reached = list(self.reached.get(row["id"], ()))
        if not notes and not reached:
            return "[]"
        return json.dumps({"unavailable": notes, "already_read": reached},
                          ensure_ascii=False)

    def enumerate(
        self,
        row: Dict,
        obligation: Obligation,
        method_feedback: str = "(none)",
        oracle_round: int = 0,
        prewarm: bool = False,
    ) -> List[Dict[str, str]]:
        if method_feedback == "(none)" and not prewarm:
            served = self.served_batch(row, obligation)
            if served is not None:
                return served
        registry = (self.host.memory.registry_summary()
                    if self.host.memory is not None else self.host.registry)
        catalog = self.host.executor.adapter_catalog()
        oracle_catalog = catalog
        try:
            candidate = self.host.executor.oracle_adapter_catalog(
                catalog, obligation.required_basis)
            if isinstance(candidate, str):
                oracle_catalog = candidate
        except Exception:
            oracle_catalog = catalog
        state = os.path.dirname(os.path.abspath(self.host.log.path))
        world_mode, world, manifest = world_evidence_context(
            state, live_world=self.host.live_world)
        with model_scope(
                stage=("oracle_repair" if oracle_round else
                       "oracle_enumerate"),
                round=oracle_round):
            variables = bind_source_evidence_variables(
                self.host.log,
                {
                "claim_text": row["text"],
                "obligation_text": obligation.text,
                "required_basis": obligation.required_basis,
                "severity": obligation.severity,
                "clause": obligation.clause,
                "refutable_by_absence": obligation.refutable_by_absence,
                "task_context": authoritative_task_context(state),
                "features": features(row),
                "registry": registry,
                "adapter_catalog": oracle_catalog,
                "method_feedback": method_feedback,
                "sibling_boundary": self.sibling_boundary(row),
                "ledger_path": self.host.log.record_path_for(row["id"]),
                "record_shape": record_shape_json(self.host.log),
                "artifact_root": os.path.abspath(self.host.artifacts.root),
                "world_mode": world_mode,
                "world_path": world,
                "world_manifest": manifest,
                "world_map": world_map(state, live_world=self.host.live_world),
                },
                enabled=self.host.run_options.compact_inspection,
            )
            payload = self.host.oracles_role.call(self.host.backend, variables)
        methods = [self.host.executor.prepare_method(method)
                   for method in payload["methods"]]
        event = {
            "claim": row["id"],
            "obligation": obligation.id,
            "repair": method_feedback != "(none)",
            "prewarm": prewarm,
            "methods": [{"procedure": m["procedure"][:80],
                         "level": m.get("level", ""),
                         "basis": m.get("basis", ""),
                         "strength": method_strength(m),
                         **({"adapter": m["adapter"],
                             "adapter_version": m.get("adapter_version")}
                            if "adapter" in m else {}),
                         **({"adapter_input_binding": m["adapter_input_binding"]}
                            if "adapter_input_binding" in m else {})}
                        for m in methods],
        }
        self.host.log.append("methods_enumerated", event)
        return methods
