"""The screening stage: a claim's verdict read off the retained record.

Before any instrument is spent, the screener seat reads the claim against
the record the solver's hooks retained. A high-confidence verified,
supported, or refuted becomes the claim's status and the formal checks are
skipped; anything else leaves the claim to them.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

from ..core.model import (
    BASIS_WORLD, CLAIM_EFFECT, CLAIM_FINAL, OUTCOME_HOLDS, OUTCOME_REFUTED,
    STATUS_REFUTED, STATUS_VERIFIED, STRENGTH_ANALYTIC, STRENGTH_DECISIVE,
    WITNESS_NONE,
)
from ..execution.instruments import authoritative_task_context
from .excerpts import bounded_context_value
from .gates import admissible_refutation


class Screener:
    """The screener seat over a claim graph and its ledger. ``report_status``
    logs a refused settlement as the kernel does; ``current_epoch`` and
    ``candidate_stamp`` stamp the evidence a settled verdict leaves."""

    def __init__(self, role, backend, store, log, *,
                 report_status: Callable, current_epoch: Callable[[], int],
                 candidate_stamp: Callable[[Dict], Dict]):
        self.role = role
        self.backend = backend
        self.store = store
        self.log = log
        self.report_status = report_status
        self.current_epoch = current_epoch
        self.candidate_stamp = candidate_stamp

    # a high-confidence screener verdict -> (status, strength): supported
    # is verified below decisive strength (the certificate reads it out
    # as supported); a screener refutation is decisive or it is nothing —
    # nothing below decisive may refute
    SCREEN_SETTLES = {"verified": (STATUS_VERIFIED, STRENGTH_DECISIVE),
                      "supported": (STATUS_VERIFIED, STRENGTH_ANALYTIC),
                      "refuted": (STATUS_REFUTED, STRENGTH_DECISIVE)}

    def screen(self, row: Dict, generation: Optional[int]) -> bool:
        """Read the claim's verdict off the retained record before any
        instrument is spent. A high-confidence verified, supported, or
        refuted becomes the claim's status and the formal checks are
        skipped. Returns True when the screener settled the claim (or the
        claim moved under it) and verification must stop here."""
        state = os.path.dirname(os.path.abspath(self.log.path))
        premises = []
        for pid in self.store.parents(row["id"]):
            prow = self.store.get(pid)
            eff = self.store.effective(pid)
            premises.append({
                "id": pid[:12],
                "text": ("[a tool-execution record]"
                         if prow["kind"] == CLAIM_EFFECT else prow["text"]),
                "status": eff["effective_status"],
                "screened": bool(prow.get("confident")),
            })
        cited, index, recent = self.record_excerpt(row)
        variables = {
            "task_context": authoritative_task_context(state),
            "claim_text": row["text"],
            "claim_kind": row["kind"],
            "premises": json.dumps(premises, ensure_ascii=False),
            "cited_events": json.dumps(cited, ensure_ascii=False),
            "call_index": json.dumps(index, ensure_ascii=False),
            "recent_events": json.dumps(recent, ensure_ascii=False),
            "graph": json.dumps(self.store.graph_snapshot(exclude=row["id"]),
                                ensure_ascii=False),
        }
        try:
            verdict = self.role.call(self.backend, variables)
        except Exception as err:  # noqa: BLE001 - screening is advisory
            # a screener failure never blocks: the claim takes the full
            # pipeline exactly as if no screener were seated
            self.log.append("claim_screened",
                            {"claim": row["id"], "confidence": "error",
                             "error": str(err)[:300]})
            return False
        witness = str(verdict.get("witness") or "")
        witness_kind = verdict.get("witness_kind")
        event = {"claim": row["id"], "verdict": verdict["verdict"],
                 "confidence": verdict["confidence"],
                 "basis": verdict["basis"],
                 "grounds": verdict["grounds"][:1000],
                 "reason": verdict["reason"][:500]}
        if witness:
            event["witness"] = witness[:1000]
            event["witness_kind"] = witness_kind
        settles = self.SCREEN_SETTLES.get(verdict["verdict"])
        refused = None
        if verdict["confidence"] != "high" or settles is None:
            settles = None
        elif verdict["verdict"] == "refuted" and not admissible_refutation(
                witness, witness_kind):
            # WITNESS AXIOM, as for any refutation: only a positive,
            # retained observation refutes; the full checks decide instead
            refused = ("a refutation needs a positive witness; reported "
                       f"kind {witness_kind or WITNESS_NONE!r}")
        elif (row["kind"] == CLAIM_FINAL
              and settles[0] == STATUS_VERIFIED
              and verdict["basis"] != BASIS_WORLD):
            # the completion-claim abstention the formal fold applies: a
            # final established on the record alone is not established
            refused = "no world-basis evidence for a completion claim"
        if refused:
            event["refused"] = refused
            settles = None
        self.log.append("claim_screened", event)
        if settles is None:
            return False
        status, strength = settles
        basis = verdict["basis"]
        reason = f"{verdict['reason']} — grounds: {verdict['grounds']}"
        if status == STATUS_REFUTED:
            reason = f"{witness} — {reason}"
        if not self.store.set_screened(
                row["id"], status, reason[:1000], basis=basis,
                strength=strength,
                generation=generation):
            # revised or re-staled while the screener read it: the reading
            # was about a graph that no longer stands. Nothing is settled
            # and nothing is checked — a check begun on that graph could
            # not settle the claim either; it stays stale for a recheck
            self.report_status(row["id"], False, status,
                                reason="screener verdict refused")
            return True
        self.store.record_evidence(
            row["id"],
            {"outcome": (OUTCOME_REFUTED if status == STATUS_REFUTED
                         else OUTCOME_HOLDS),
             "instrument": "screener", "verdict": verdict["verdict"],
             "confidence": "high", "basis": basis, "strength": strength,
             "grounds": verdict["grounds"], "reason": verdict["reason"],
             "witness": witness or None,
             "witness_kind": witness_kind if witness else None,
             "world_epoch": self.current_epoch(),
             **self.candidate_stamp(row)})
        self.log.append("status_change",
                        {"claim": row["id"], "status": status,
                         "strength": strength, "basis": basis,
                         "confident": True,
                         "reason": "settled by the screener on the record"})
        return True

    # the retained record, sliced for the screener: each cited event in
    # full (bounded), a compact index of every recorded call, and the most
    # recent events bounded per entry — a table of contents plus the
    # decisive pages, never the whole ledger
    SCREEN_CITED_CHARS = 8_000
    SCREEN_INDEX_ENTRIES = 150
    SCREEN_RECENT_EVENTS = 24
    SCREEN_RECENT_CHARS = 2_500

    def record_excerpt(self, row: Dict):
        cited_seqs = set()
        for pid in self.store.parents(row["id"]):
            prow = self.store.get(pid)
            if prow["kind"] != CLAIM_EFFECT:
                continue
            try:
                cited_seqs.add(int(json.loads(prow["text"])["event_seq"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        calls: List[Dict] = []
        results: List[Dict] = []
        for event in self.log.replay():
            if event["type"] == "tool_call":
                calls.append(event)
            elif event["type"] == "tool_result":
                results.append(event)
        call_by_id = {}
        for c in calls:
            cid = c["payload"].get("tool_use_id")
            if cid:
                call_by_id[cid] = c

        def render_call(c, cap):
            p = c["payload"]
            return {"seq": c["seq"], "tool": p.get("tool"),
                    "input": bounded_context_value(p.get("input"), cap)}

        def render_result(r, cap):
            p = r["payload"]
            entry = {"seq": r["seq"], "tool": p.get("tool"),
                     "excerpt": bounded_context_value(
                         p.get("excerpt"), cap),
                     "response_chars": p.get("response_chars"),
                     "excerpt_truncated": p.get("excerpt_truncated")}
            call = call_by_id.get(p.get("tool_use_id"))
            if call is not None:
                entry["call"] = render_call(call, cap)
            return entry

        cited = [render_result(r, self.SCREEN_CITED_CHARS)
                 for r in results if r["seq"] in cited_seqs]
        index = [{"seq": c["seq"], "tool": c["payload"].get("tool"),
                  "input": json.dumps(c["payload"].get("input"),
                                      ensure_ascii=False,
                                      default=str)[:120]}
                 for c in calls[-self.SCREEN_INDEX_ENTRIES:]]
        recent = [render_result(r, self.SCREEN_RECENT_CHARS)
                  for r in results[-self.SCREEN_RECENT_EVENTS:]]
        return cited, index, recent
