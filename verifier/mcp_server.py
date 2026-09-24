"""MCP server: the verifier as a separate process.

The solver stays a vanilla Claude Code session; it mounts this server over
stdio and gets four verbs — commit, verify, certificate, status. Every
call is an event in the server's own append-only log, verification runs
inside the server (through the same kernel as offline replay, zero changes),
and the certificate returns to the solver over MCP; only CI bypasses it via
`python -m verifier.cli`.

`commit` is where verification happens (`--trigger on-commit`, the default):
the call blocks until the claim it just recorded has been checked, and returns
the failing obligation with its counterexample. The solver therefore cannot
write a second claim on top of an unchecked first one.

Hook events (tool_call/tool_result written by the solver's PreToolUse /
PostToolUse hooks) arrive on a separate raw feed file: hooks are concurrent
short-lived processes and cannot share seq discipline, so the server — the
canonical log's single writer — ingests the feed in arrival order and assigns
seq itself. The feed is drained before every MCP verb so effect claims are
referenceable (`effect:<seq>`) at commit time.

Run:
  uv run python -m verifier.mcp_server --state-dir .gv [--model m]
      [--memory-dir d] [--toolchain-dir d]
      [--trigger on-commit|on-first-dependent]

Claude Code config (.mcp.json):
  {"mcpServers": {"gv": {"type": "stdio", "command": "uv",
                          "args": ["run", "python", "-m", "verifier.mcp_server", "--state-dir", ".gv"]}}}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import deque
from typing import Any, Dict, List, Optional

from verifier.core.model import CLAIM_KINDS
from .stream.builder import Builder, TraceError, UnresolvedCommit
from .engine.certificate import request_certificate
from .core.events import EventLog
from .execution.instruments import ArtifactStore
from .engine.kernel import Kernel
from .llm.schema import RoleError
from .core.store import (
    ClaimStore, UnknownDependency, TRIGGER_ON_COMMIT, TRIGGERS,
)
from .stream.tail import Tail

VERB_ERRORS = (TraceError, UnknownDependency, RoleError, ValueError)


def _witness(obligation: dict) -> str:
    """The counterexample recorded on an obligation node, if any."""
    for ev in reversed(obligation.get("evidence") or []):
        if ev.get("witness"):
            return str(ev["witness"])[:300]
    return ""


def _obligation_summary(store, claim_id: str) -> list:
    """Obligation-level detail straight from the graph nodes."""
    out = []
    for o in store.obligation_children(claim_id):
        if o["status"] == "verified":
            continue
        entry = {"text": o["text"], "status": o["status"],
                 "required_by_task": o.get("task_bearing", "unclear")}
        reason = o.get("reason")
        if not reason:
            ev = store.evidence_for(o["id"])
            reason = (str(ev[-1].get("witness"))[:300]
                      if ev and ev[-1].get("witness") else None)
        if reason:
            entry["reason"] = reason
        out.append(entry)
    return out


def _holding(store, claim_id: str) -> list:
    """The claim's obligations that DID hold: a fix must keep them true.
    Returned beside the residue so a repair aimed at one failing condition
    is not made by breaking another the same contract demands."""
    return [o["text"] for o in store.obligation_children(claim_id)
            if o["status"] == "verified"]


def _residue(store, claim_id: str) -> Dict[str, Any]:
    """What a non-verified claim owes the solver: the obligations that did not
    hold with their counterexamples — or, when no obligation was ever compiled,
    the fail-closed reason (a refuted premise, say), which is otherwise
    invisible to the solver."""
    obligations = _obligation_summary(store, claim_id)
    if obligations:
        out = {"obligations": obligations}
        holding = _holding(store, claim_id)
        if holding:
            out["still_holding"] = holding
        out["note"] = (
            "required_by_task says whether the AUTHORITATIVE TASK requires "
            "the failing condition (required: the deliverable must satisfy "
            "it), or the claim asserted it beyond the task (beyond_task: "
            "narrow or drop the claim; the deliverable need not change), or "
            "the verifier could not tell (unclear: read the task). "
            "still_holding lists the conditions of the same contract that "
            "were established; a fix for the failing ones must keep every "
            "one of them true")
        return out
    for ev in reversed(store.evidence_for(claim_id)):
        if ev.get("error"):
            out = {"error": ev["error"]}
            if ev.get("refuted_deps"):
                out["refuted_deps"] = [d[:12] for d in ev["refuted_deps"]]
            return out
    return {"obligations": []}


def _refutation_brief(store, claim_id: str, cap: int = 320) -> str:
    """The failing obligation of a refuted claim and its counterexample,
    bounded. A client may abandon the pending call that carried the full
    residue (codex hands control back to the model while an MCP call is in
    flight), so the status map says why a claim was refuted, not only that
    it was; gv.status(label) still returns the whole residue."""
    for o in store.obligation_children(claim_id):
        if o["status"] != "refuted":
            continue
        ev = store.evidence_for(o["id"])
        witness = (str(ev[-1].get("witness"))
                   if ev and ev[-1].get("witness") else "")
        text = o["text"][:160]
        return (f"{text} -- counterexample: {witness}" if witness
                else text)[:cap]
    residue = _residue(store, claim_id)
    return str(residue.get("error") or "ask gv.status for this label")[:cap]


HOOK_FEED = "hook_feed.jsonl"
# how far the hook feed has been ingested into the ledger: a restarted
# server resumes there instead of re-reading the feed from its start (a
# restart used to append every earlier tool call to the ledger again —
# 8164 rows for 1169 calls after eleven restarts)
FEED_OFFSET = "hook_feed.offset"
# the live server's pid, for information (liveness is the lock below)
SERVER_PIDFILE = "server.pid"
# held (flock) by the one server serving a state for its whole lifetime: a
# second process on the same state exits before touching it. A pid probe
# could not tell a live server from a zombie or a reused pid, and voided
# the certificates of a legitimate restart.
SERVER_LOCK = "server.lock"
HOOK_EVENT_TYPES = ("tool_call", "tool_result")


class Service:
    def __init__(self, state_dir: str, kernel_factory=None,
                 model: Optional[str] = None,
                 memory_dir: Optional[str] = None,
                 toolchain_dir: Optional[str] = None,
                 trigger: str = TRIGGER_ON_COMMIT,
                 live_world: bool = True,
                 critic_rounds: Optional[int] = None,
                 parallel_obligations: Optional[int] = None,
                 parallel_claims: Optional[int] = None,
                 shared_cache_dir: Optional[str] = None,
                 server_lock=None):
        # the held state lock (hold_server_lock), kept for the process's
        # lifetime; None when this Service did not take one
        self.server_lock = server_lock
        self.toolchain_dir = toolchain_dir
        # a verifier-owned cache root shared across trials (install caches
        # for the executor's instruments); None keeps every cache
        # state-local as before
        self.shared_cache_dir = shared_cache_dir
        self.parallel_obligations = parallel_obligations
        self.parallel_claims = parallel_claims
        self.live_world = live_world
        os.makedirs(state_dir, exist_ok=True)
        self.state_dir = state_dir
        self.memory = None
        if memory_dir:
            from .memory import Memory
            self.memory = Memory(memory_dir)
        self.log = EventLog(os.path.join(state_dir, "events.jsonl"))
        self.store = ClaimStore(os.path.join(state_dir, "claims.sqlite"),
                                trigger=trigger)
        self.artifacts = ArtifactStore(os.path.join(state_dir, "artifacts"))
        self.builder = Builder(self.log, self.store)
        # which refutation of each claim the solver was told of. A claim is
        # keyed by the ledger seq of the status_change that refuted it, so
        # a claim refuted again after a recheck is news again, and the
        # delivered keys are rebuilt from the ledger after a restart
        self._refutations: Dict[str, List[int]] = {}
        self._last_change: Dict[str, tuple] = {}
        self._reported: Dict[str, int] = {}
        # told before its status_change landed: claim -> the seq of the
        # claim's last status_change at that time (0 for none)
        self._pending_reported: Dict[str, int] = {}
        self.log.subscribe(self._observe)

        if kernel_factory is None:
            kernel_factory = lambda sc: _default_kernel(  # noqa: E731
                sc, model=model, critic_rounds=critic_rounds,
                parallel_obligations=parallel_obligations,
                parallel_claims=parallel_claims)
        self.kernel = kernel_factory(self)  # factory receives the service

        # verbs run concurrently (build_server), and the feed's tail keeps
        # one read offset: two drains at once would ingest the same lines
        # twice. Only the drain is serialized; verification is not.
        self._drain_lock = threading.Lock()
        # online, an intermediate claim's commit returns once the claim is
        # recorded and its checks run in the background, so the solver keeps
        # working instead of waiting on every check; a refutation reaches it
        # in the `news` of its next call (and in gv.status). A final's commit
        # still waits for its checks, and the certificate waits for every
        # check in its slice. GV_BACKGROUND_CHECKS=0 restores blocking commits.
        self.background_checks = (
            live_world and os.environ.get("GV_BACKGROUND_CHECKS", "1") != "0")
        self._news_lock = threading.Lock()
        # the solver cannot see the ledger's sequence numbers, and cited the
        # wrong effects when it guessed them (or counted lines of the hook
        # feed): every gv response lists the tool results recorded since the
        # previous one, each under the id a claim cites it by
        self._effect_index: deque = deque(maxlen=RECENT_EFFECTS)
        self._call_inputs: Dict[str, str] = {}
        self._effects_listed = 0        # seq of the newest effect listed
        self._gv_calls: Dict[str, int] = {}
        # a restarted server rebuilds its view from its own ledger
        feed_offset = self._replay()
        self._feed = Tail(os.path.join(state_dir, HOOK_FEED),
                          self._ingest_hook_event, raw=True,
                          start_offset=max(feed_offset,
                                           self._saved_feed_offset()),
                          on_reject=self._reject_feed_line)

    def _replay(self) -> int:
        """Rebuild the graph, the news already delivered, and the effect
        index from the ledger; returns the hook-feed offset the ledger has
        ingested through. An event that no longer resolves (a commit whose
        reference failed, recorded before its failure was) is skipped and
        noted; it never keeps the server from starting."""
        events = list(self.log.replay())
        failed = {e["payload"].get("commit") for e in events
                  if e["type"] == "commit_failed"}
        unresolved = []
        feed_offset = 0
        delivered_through = 0
        for event in events:
            self._observe(event)
            payload = event.get("payload") or {}
            if isinstance(payload.get("feed_offset"), int) and \
                    event["type"] in HOOK_EVENT_TYPES + ("hook_feed_rejected",):
                feed_offset = max(feed_offset, payload["feed_offset"])
            if event["type"] in ("commit", "extracted_commit") \
                    and event["seq"] in failed:
                continue
            try:
                self.builder.feed(event)
            except VERB_ERRORS as err:
                if event["type"] == "commit" \
                        and isinstance(err, UnresolvedCommit):
                    unresolved.append((event["seq"], str(err)))
                print(f"gv: replay skipped event {event['seq']} "
                      f"({event['type']}): {err}", file=sys.stderr)
                continue
            self._index_effect(event)
            delivered_through = max(delivered_through,
                                    self._gv_response_boundary(event))
        # the effects the solver was last shown: those recorded before the
        # last gv call that returned, so a restart does not list them again
        self._effects_listed = delivered_through
        for seq, error in unresolved:
            # recorded now, so the next replay (and every other reader of
            # the ledger) knows this commit recorded nothing
            self.log.append("commit_failed", {"commit": seq,
                                              "error": error[:500]})
        return feed_offset

    def _gv_response_boundary(self, event: Dict[str, Any]) -> int:
        """The seq of the call a delivered gv response answered (0 when
        the event is not such a response): every effect before that call
        was listed in the response."""
        payload = event.get("payload") or {}
        if not str(payload.get("tool") or "").startswith("mcp__gv__"):
            return 0
        call_id = payload.get("tool_use_id")
        if event["type"] == "tool_call":
            if call_id:
                self._gv_calls[call_id] = event["seq"]
            return 0
        if event["type"] == "tool_result" and call_id:
            return self._gv_calls.pop(call_id, 0)
        return 0

    def _observe(self, event: Dict[str, Any]) -> None:
        """Every ledger event, replayed or appended: the claims' status
        changes and the refutations already delivered. Touches only its own
        maps (it runs on whichever thread appended), never a lock."""
        etype = event.get("type")
        payload = event.get("payload") or {}
        claim = payload.get("claim")
        if not isinstance(claim, str):
            return
        if etype == "status_change" and "obligation" not in payload:
            status = payload.get("status")
            self._last_change[claim] = (event["seq"], status)
            if status == "refuted":
                self._refutations.setdefault(claim, []).append(event["seq"])
        elif etype == "news_delivered":
            if isinstance(payload.get("change_seq"), int):
                self._reported[claim] = max(self._reported.get(claim, 0),
                                            payload["change_seq"])
                self._pending_reported.pop(claim, None)
            else:
                self._pending_reported[claim] = int(payload.get("after") or 0)

    def start_background(self) -> bool:
        """Online: compile the final's task contract and its method
        batches while the solver works (GV_FINAL_PRECOMPILE=0 disables).
        Called by the entry point AFTER the server's own imports are
        done: the seat thread imports the Agent SDK, whose import chain
        shares modules with the MCP server's, and two threads importing
        the same package concurrently deadlock the import lock — which
        killed the server at startup on some hosts."""
        if not self.live_world \
                or os.environ.get("GV_FINAL_PRECOMPILE", "1") == "0":
            return False
        start = getattr(self.kernel, "start_final_precompile", None)
        return bool(start is not None and start())

    # hook feed
    def _ingest_hook_event(self, raw_event: Dict[str, Any]) -> None:
        etype = raw_event.get("type")
        if etype not in HOOK_EVENT_TYPES:
            return  # unknown feed line: ignored, the feed is untrusted input
        payload = raw_event.get("payload", {})
        end = self._feed.position
        if not isinstance(payload, dict):
            self._reject_feed_line(None, end, raw_event,
                                   "payload is not an object")
            return
        # the feed offset just past this line: a restart resumes after the
        # last line the ledger holds, even when it died before saving the
        # offset file (it used to ingest the batch again, duplicating
        # tool_use_ids)
        payload = dict(payload, feed_offset=end)
        try:
            event = self.log.append(etype, payload)
        except ValueError as err:   # a payload the ledger cannot encode
            self._reject_feed_line(None, end, raw_event, str(err))
            return
        self.builder.feed(event)
        self._index_effect(event)

    def _reject_feed_line(self, start: Optional[int], end: int, line: Any,
                          error: str) -> None:
        """A feed line the server cannot ingest: recorded with where it
        was, then skipped; the rest of the feed is still read."""
        if isinstance(line, bytes):
            excerpt = line[:200].decode("utf-8", "replace")
        else:
            excerpt = json.dumps(line, ensure_ascii=True, default=str)[:200]
        payload = {"feed_offset": end, "error": str(error)[:200],
                   "excerpt": excerpt}
        if start is not None:
            payload["start"] = start
        self.log.append("hook_feed_rejected", payload)

    def _index_effect(self, event: Dict[str, Any]) -> None:
        etype = event.get("type")
        if etype not in ("tool_call", "tool_result"):
            return
        payload = event.get("payload") or {}
        tool = str(payload.get("tool") or "")
        if tool.startswith("mcp__gv__"):
            return
        call_id = payload.get("tool_use_id")
        if etype == "tool_call":
            if call_id:
                self._call_inputs[call_id] = _excerpt(payload.get("input"))
            return
        label = f"effect:{event['seq']}"
        if label not in self.builder.labels:
            return
        entry = {"ref": label, "tool": tool or None}
        command = self._call_inputs.pop(call_id, None) if call_id else None
        if command:
            entry["input"] = command
        entry["output"] = _excerpt(payload.get("excerpt"))
        with self._news_lock:
            self._effect_index.append((event["seq"], entry))

    def drain_hooks(self) -> int:
        with self._drain_lock:
            try:
                return self._feed.poll()
            finally:
                self._save_feed_offset()

    def _drain_for_verb(self) -> None:
        """A verb's drain: an error while ingesting the feed is the
        verifier's own problem and never fails the solver's call."""
        try:
            self.drain_hooks()
        except Exception as err:  # noqa: BLE001
            print(f"gv: hook feed drain failed: {err!r}", file=sys.stderr)

    def _saved_feed_offset(self) -> int:
        try:
            with open(os.path.join(self.state_dir, FEED_OFFSET),
                      encoding="ascii") as f:
                return max(0, int(f.read().strip() or 0))
        except (OSError, ValueError):
            return 0

    def _save_feed_offset(self) -> None:
        path = os.path.join(self.state_dir, FEED_OFFSET)
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="ascii") as f:
                f.write(str(self._feed.consumed))
            os.replace(tmp, path)
        except OSError:
            pass

    def record_start(self, launch: Dict[str, Any]) -> None:
        """Record this server's launch in the ledger. The solver shares the
        verifier's machine and can stop it and start its own with other
        settings (observed: seats moved off their provider, their effort cut
        to low); a launch whose settings or code differ from the first is
        recorded and voids every later certificate (see
        verifier_integrity). A second server on the same state never gets
        here while the first holds the state lock; only where no lock could
        be taken does the pidfile probe stand in, and a live pid there is
        recorded as a concurrent start."""
        fingerprint = _digest(launch)
        first = next((e["payload"] for e in self.log.replay()
                      if e["type"] == "server_started"), None)
        pidfile = os.path.join(self.state_dir, SERVER_PIDFILE)
        other = (None if self.server_lock is not None
                 else _live_server_pid(pidfile))
        self.log.append("server_started", {
            "pid": os.getpid(), "fingerprint": fingerprint,
            "launch": launch})
        if other is not None:
            self.log.append("server_concurrent_start",
                            {"pid": os.getpid(), "other_pid": other})
        if first is not None and first.get("fingerprint") != fingerprint:
            before = first.get("launch") or {}
            self.log.append("server_config_changed", {
                "from": first.get("fingerprint"), "to": fingerprint,
                "changed": _changed_keys(before, launch)})
        try:
            with open(pidfile, "w", encoding="ascii") as f:
                f.write(str(os.getpid()))
        except OSError:
            pass

    # MCP verbs
    def commit(self, label: str, kind: str, text: str,
               deps: List[str],
               revises: Optional[str] = None) -> Dict[str, Any]:
        self._drain_for_verb()
        payload = {"label": label, "kind": kind, "text": text, "deps": deps}
        if revises:
            payload["revises"] = revises
        failure = None
        background = self.background_checks and kind not in ("final", "effect")
        deferred: List[str] = []
        split_deps = self.background_checks and kind == "final"
        event = None
        try:
            event = self.log.append("commit", payload)
            if background or split_deps:
                # recorded and wired now (later commits may cite its label at
                # once); the checks its commit triggers are queued, not run
                with self.store.deferred_triggers() as deferred:
                    self.builder.feed(event)
            else:
                # under the on-commit rule this is where verification
                # happens: the call returns only once the claim is checked
                self.builder.feed(event)
        except VERB_ERRORS as err:
            failure = str(err)
            if event is not None and isinstance(err, UnresolvedCommit):
                # the commit event stays for audit; this one says it
                # recorded nothing, so replay skips it instead of failing
                # on it (a restart used to die on such a commit, over and
                # over) and readers of the ledger do not count it
                self._record_commit_failed(event["seq"], failure)
        if split_deps and failure is None and label in self.builder.labels:
            # a final's contract is compiled and checked independently of the
            # claims it cites: their checks (first triggered by this commit,
            # or still running from their own) proceed in the background, and
            # the final's own check starts now instead of after theirs
            final_id = self.builder.labels[label]
            others = [cid for cid in deferred if cid != final_id]
            if others:
                self._check_in_background(final_id, others)
            if final_id in deferred:
                try:
                    self.store.fire([final_id])
                except VERB_ERRORS as err:
                    failure = str(err)
        if label not in self.builder.labels:
            result = {"error": failure or "commit failed"}
            if failure and "unknown claim reference" in failure:
                # the solver cannot see the ledger's sequence numbers; an
                # unresolved reference gets the record's live index so the
                # retry needs no status round trip
                # a snapshot: concurrent verbs add labels meanwhile
                labels = list(self.builder.labels)
                effects = [l for l in labels if l.startswith("effect:")]
                result["known_references"] = (
                    [l for l in labels if not l.startswith("effect:")]
                    + effects[-EFFECTS_LISTED:])
                result["note"] = ("effect:<seq> names a recorded TOOL RESULT "
                                  "by its ledger sequence number; the most "
                                  "recent results are listed last")
            return result
        claim_id = self.builder.labels[label]
        if background and deferred and failure is None:
            self._check_in_background(claim_id, deferred)
            row = self.store.get(claim_id)
            return self._with_news({
                "claim_id": claim_id, "status": row["status"], "checking": True,
                "note": ("recorded; its check runs in the background while you "
                         "keep working. gv.status(label) shows the verdict, and "
                         "a refutation arrives in the `news` of your next gv "
                         "call. Build on it only as far as you would build on "
                         "an unchecked conclusion.")})
        row = self.store.get(claim_id)
        status = row["status"]
        result = {"claim_id": claim_id, "status": status}
        if failure is not None:
            # the claim is recorded; its check did not finish — fail-closed, so
            # the status stands as unverified and the solver is told why
            result["error"] = failure
        if row.get("confident"):
            # admitted on the record: assumed by every consumer, no check
            # run, nothing owed — the solver may build on it (and may ask
            # for the full check with `verify` if it wants one)
            result["confident"] = True
            result["confidence_reason"] = row.get("confidence_reason")
            return self._with_news(result)
        if kind != "effect" and status != "verified":
            # the verdict lands on the commit call now, so its residue must too
            result.update(_residue(self.store, claim_id))
            self._mark_reported(claim_id, status)
        return self._with_news(result)

    def _record_commit_failed(self, seq: int, error: str) -> None:
        try:
            self.log.append("commit_failed", {"commit": seq,
                                              "error": error[:500]})
        except Exception as err:  # noqa: BLE001 - the verb still answers
            print(f"gv: could not record commit_failed: {err!r}",
                  file=sys.stderr)

    def _check_in_background(self, claim_id: str, deferred: List[str]) -> None:
        """Run the deferred checks off the verb's thread: the claim's own
        check and those of the claims it cites side by side (a claim's local
        verdict is conditional on its premises and never waits for theirs)."""
        def run(batch):
            try:
                self.store.fire(batch)
            except Exception as err:  # noqa: BLE001 - never takes the server down
                try:
                    self.log.append("background_check_failed",
                                    {"claim": claim_id, "error": str(err)[:500]})
                except Exception:  # noqa: BLE001
                    pass
        own = [cid for cid in deferred if cid == claim_id]
        cited = [cid for cid in deferred if cid != claim_id]
        for batch in (cited, own):
            if batch:
                threading.Thread(target=run, args=(batch,),
                                 name=f"gv-check-{claim_id[:8]}",
                                 daemon=True).start()

    def _mark_reported(self, claim_id: str, status: str) -> None:
        if status == "refuted":
            with self._news_lock:
                self._deliver(claim_id)

    def _refuted_at(self, claim_id: str) -> Optional[int]:
        """The seq of the status_change that refuted the claim as it stands
        now; None when that event has not landed (the store changes first)
        or the claim was refuted without one."""
        last = self._last_change.get(claim_id)
        return last[0] if last and last[1] == "refuted" else None

    def _delivered(self, claim_id: str) -> bool:
        """Whether the solver was told of the claim's current refutation.
        Called under the news lock."""
        after = self._pending_reported.get(claim_id)
        if after is not None:
            # told before the refuting event landed: the first refutation
            # after that point is the one it was told of
            landed = [r for r in self._refutations.get(claim_id, ())
                      if r > after]
            if not landed:
                return True
            del self._pending_reported[claim_id]
            self._reported[claim_id] = max(self._reported.get(claim_id, 0),
                                           landed[0])
        key = self._refuted_at(claim_id)
        if key is None:
            return claim_id in self._reported
        return self._reported.get(claim_id, -1) >= key

    def _deliver(self, claim_id: str) -> None:
        """Note (in memory and in the ledger) that the solver was told of
        the claim's current refutation. Called under the news lock."""
        if self._delivered(claim_id):
            return
        key = self._refuted_at(claim_id)
        if key is not None:
            marker = {"claim": claim_id, "change_seq": key}
        else:
            last = self._last_change.get(claim_id)
            marker = {"claim": claim_id, "change_seq": None,
                      "after": last[0] if last else 0}
        try:
            # _observe applies the marker as it is appended
            self.log.append("news_delivered", marker)
        except Exception:  # noqa: BLE001 - the in-memory note still holds
            self._observe({"type": "news_delivered", "seq": 0,
                           "payload": marker})

    def _news(self) -> list:
        """Refutations the solver has not been told of yet: a check that
        finished in the background after the commit call returned."""
        out = []
        with self._news_lock:
            for label, cid in list(self.builder.labels.items()):
                if label.startswith("effect:") or self._delivered(cid):
                    continue
                row = self.store.get(cid)
                if row["kind"] == "effect" or row["status"] != "refuted":
                    continue
                self._deliver(cid)
                out.append({"label": label, "status": "refuted",
                            "why": _refutation_brief(self.store, cid)})
        return out

    def _with_news(self, result: Dict[str, Any]) -> Dict[str, Any]:
        news = self._news()
        out = dict(result, news=news) if news else dict(result)
        with self._news_lock:
            fresh = [entry for seq, entry in self._effect_index
                     if seq > self._effects_listed]
            if self._effect_index:
                self._effects_listed = max(self._effects_listed,
                                           self._effect_index[-1][0])
        if fresh:
            out["recent_effects"] = fresh
        return out

    def verify(self, label: str) -> Dict[str, Any]:
        self._drain_for_verb()
        try:
            claim_id = self.builder.resolve(label)
            self.log.append("verify_request", {"target": claim_id, "cause": "mcp"})
            # an explicit request is never answered by the screener: a
            # confident claim gets its full check here
            status = self.kernel.verify(claim_id, force=True)
        except VERB_ERRORS as err:
            return {"error": str(err)}
        result = {"claim_id": claim_id, "status": status}
        if status != "verified":
            # every non-verified status owes the solver its residue: which
            # obligation failed and the counterexample witness. Withholding
            # it on `refuted` left the solver with nothing to act on but the
            # claim's wording — the mechanism behind claim-shopping.
            result.update(_residue(self.store, claim_id))
            self._mark_reported(claim_id, status)
        return self._with_news(result)

    def certificate(self, final_label: str) -> Dict[str, Any]:
        self._drain_for_verb()
        try:
            final_id = self.builder.resolve(final_label)
            cert = request_certificate(self.store, self.kernel, self.log,
                                       self.artifacts, final_id)
        except VERB_ERRORS as err:
            return {"error": str(err)}
        slice_summary = []
        not_rechecked = getattr(self.kernel, "not_rechecked", ())
        for e in cert["slice"]:
            entry = {"text": e["text"], "status": e["status"]}
            if e["status"] == "unverified" and e["id"] in not_rechecked:
                entry["not_rechecked"] = (
                    "its last check ran on the world as it stands and is "
                    "not repeated; change what it found missing, or "
                    "gv.verify it after a change outside the files")
            if e.get("confident"):
                entry["confident"] = True
                slice_summary.append(entry)
                continue
            if e["status"] != "verified":
                obligations = [
                    {"text": o["text"], "status": o["status"],
                     **({"reason": o["reason"] or _witness(o)}
                        if (o["reason"] or _witness(o)) else {})}
                    for o in e.get("obligations", [])
                    if o["status"] != "verified"
                ]
                if obligations:
                    entry["obligations"] = obligations
                else:
                    for ev in reversed(self.store.evidence_for(e["id"])):
                        if ev.get("error"):
                            entry["error"] = ev["error"]
                            break
            slice_summary.append(entry)
            if e["status"] == "refuted":
                self._mark_reported(e["id"], "refuted")
        result = {
            "verdict": cert["verdict"],
            "certificate_id": cert["certificate_id"],
            "slice": slice_summary,
        }
        if cert.get("void"):
            result["void"] = cert["void"]
        return self._with_news(result)

    def status(self, label: Optional[str] = None) -> Dict[str, Any]:
        self._drain_for_verb()
        if label is not None:
            try:
                claim_id = self.builder.resolve(label)
            except TraceError as err:
                return {"error": str(err)}
            row = self.store.get(claim_id)
            result = {k: row[k] for k in ("id", "kind", "text", "status")}
            if self._checking(claim_id):
                result["checking"] = True
            if row.get("confident"):
                result["confident"] = True
                result["confidence_reason"] = row.get("confidence_reason")
            elif row["status"] != "verified":
                result.update(_residue(self.store, claim_id))
                self._mark_reported(claim_id, row["status"])
            return self._with_news(result)
        out = {}
        effects = []
        # a snapshot: concurrent verbs add labels meanwhile
        for label, cid in list(self.builder.labels.items()):
            if label.startswith("effect:"):
                # every recorded tool result has a label; listing each one
                # grew the map past 20K characters (and every response is
                # recorded back into the ledger), so effects are counted
                effects.append(label)
                continue
            row = self.store.get(cid)
            if row.get("confident"):
                out[label] = f"{row['status']} (confident)"
            elif row["status"] == "refuted" and row["kind"] != "effect":
                out[label] = "refuted: " + _refutation_brief(self.store, cid)
                self._mark_reported(cid, "refuted")
            elif self._checking(cid):
                out[label] = f"{row['status']} (checking)"
            else:
                out[label] = row["status"]
        if effects:
            out["effects" if "effects" not in out else "effect:*"] = {
                "recorded": len(effects),
                "latest": effects[-EFFECTS_LISTED:],
                "note": "effect:<seq> names a recorded tool result; "
                        "gv.status(<ref>) shows one"}
        return self._with_news(out)

    def _checking(self, claim_id: str) -> bool:
        checking = getattr(self.kernel, "checking", None)
        return bool(checking is not None and checking(claim_id))

SECRET_MARKS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")


def launch_identity(args: argparse.Namespace,
                    environ=os.environ) -> Dict[str, Any]:
    """What decides how this server verifies: its arguments, the verifier's
    own settings in the environment (never a credential's value), and a
    digest of its code and shipped configuration."""
    env = {k: v for k, v in environ.items()
           if (k.startswith("GV_") or k == "CODEX_HOME")
           and not any(m in k.upper() for m in SECRET_MARKS)}
    return {"args": {k: v for k, v in sorted(vars(args).items())},
            "env": dict(sorted(env.items())),
            "code": _code_digest()}


def _code_digest() -> str:
    import hashlib
    root = os.path.dirname(os.path.abspath(__file__))
    top = os.path.dirname(root)
    digest = hashlib.sha256()
    paths = []
    for base in (root, os.path.join(top, "protocol")):
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            paths += [os.path.join(dirpath, n) for n in sorted(filenames)
                      if n.endswith((".py", ".md", ".json", ".toml"))]
    paths.append(os.path.join(top, ".env"))
    for path in paths:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            continue
        digest.update(os.path.relpath(path, top).encode() + b"\0")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def _digest(value: Any) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str)
                          .encode("utf-8")).hexdigest()[:16]


def _changed_keys(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    changed = []
    for section in sorted(set(before) | set(after)):
        a, b = before.get(section), after.get(section)
        if isinstance(a, dict) and isinstance(b, dict):
            changed += [f"{section}.{k}" for k in sorted(set(a) | set(b))
                        if a.get(k) != b.get(k)]
        elif a != b:
            changed.append(section)
    return changed


def _live_server_pid(pidfile: str) -> Optional[int]:
    """The pid in the pidfile when that process is alive and not us."""
    try:
        with open(pidfile, encoding="ascii") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return None
    if pid == os.getpid():
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid
    return pid


class StateBusy(RuntimeError):
    """Another live process holds the state's server lock."""


def hold_server_lock(state_dir: str):
    """Take the state's server lock for this process's lifetime: an
    exclusive, non-blocking flock on <state>/server.lock. Returns the open
    lock file (keep it referenced), or None where advisory locks are
    unavailable; raises StateBusy when another process holds it. The
    kernel drops the lock when its holder exits, however it exits, so a
    dead or zombie server never blocks its restart."""
    try:
        import fcntl
    except ImportError:
        return None
    os.makedirs(state_dir, exist_ok=True)
    handle = open(os.path.join(state_dir, SERVER_LOCK), "a+", encoding="ascii")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, PermissionError):
        handle.close()
        raise StateBusy(
            f"another gv server holds {os.path.join(state_dir, SERVER_LOCK)}; "
            "this process exits without touching the state") from None
    except OSError:
        handle.close()
        return None   # the filesystem offers no flock: the pidfile probe
    return handle


RECENT_EFFECTS = 8           # tool results a gv response lists at most
# recorded tool results named in an unresolved commit's known_references
# and in the all-labels status map
EFFECTS_LISTED = 40
EFFECT_EXCERPT_CHARS = 100   # of each listed call's input and output


def _excerpt(value: Any, cap: int = EFFECT_EXCERPT_CHARS) -> str:
    """A one-line glimpse of a tool call's input or output: enough to tell
    which recorded effect is which, never the evidence itself."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = value
        value = decoded
    if isinstance(value, dict):
        value = " ".join(f"{k}={v}" if isinstance(v, str)
                         else f"{k}={json.dumps(v, ensure_ascii=False, default=str)}"
                         for k, v in value.items())
    elif not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(value.split())
    if len(text) <= cap:
        return text
    half = (cap - 3) // 2
    return text[:half] + " … " + text[-half:]


def _parallel_default(name: str, fallback: int) -> int:
    """Per-trial parallelism defaults: GV_PARALLEL_OBLIGATIONS (discharges
    concurrent within one claim) and GV_PARALLEL_CLAIMS (claims of a
    certificate slice verified concurrently; offline only)."""
    try:
        return max(1, int(os.environ.get(name, fallback)))
    except ValueError:
        return fallback


def harness_roles(state_dir: str) -> Dict[str, Any]:
    """The role registry a state's kernel seats: the mounting harness's
    ground truth (task.md / baseline/) is the toggle; when present
    (experiment mounts write it; the interactive mount does not), the
    obligation compiler is told to compile against it."""
    from dataclasses import replace

    from .llm import prompts
    from .llm.agents import ALL_AGENT_ROLES, COMPILE_OBLIGATIONS_AGENT

    obligations = COMPILE_OBLIGATIONS_AGENT
    if os.path.exists(os.path.join(state_dir, "task.md")) or \
            os.path.isdir(os.path.join(state_dir, "baseline")):
        obligations = replace(
            obligations, system=obligations.system + "\n"
            + prompts.OBLIGATIONS_HARNESS_ADDENDUM)
    return dict(ALL_AGENT_ROLES, obligations=obligations)


def _default_kernel(sc: Service, model, backend=None,
                    critic_rounds: Optional[int] = None,
                    parallel_obligations: Optional[int] = None,
                    parallel_claims: Optional[int] = None) -> Kernel:
    from .llm.agents import AgentBackend, SCREENER_AGENT
    from .execution.config import (
        contract_cache_enabled,
        method_cache_enabled,
        sufficiency_preaudit_enabled,
        typed_adapters_enabled,
        typed_judger_packet_enabled,
    )
    from .execution.discharge import AgentDischarger
    from .engine.game import BindingGame

    # The backend resolves roles by name, so the harness-swapped role must
    # go into its registry too, not just into the Kernel.
    roles = harness_roles(sc.state_dir)
    obligations = roles["obligations"]
    if backend is None:
        # the agent backend, with the tool-free roles seated on a direct
        # model call when the configuration carries an API key
        from .llm.direct_backend import build_backend
        backend = build_backend(roles, cwd=sc.state_dir, model=model)
    else:
        # an injected backend (e.g. CodexBackend) still needs the
        # harness-swapped role registry — same double-registration rule
        backend.roles = roles
    validate_tool_scope = getattr(
        backend, "validate_offline_tool_scope", None)
    if not sc.live_world and validate_tool_scope is not None:
        validate_tool_scope(
            sc.state_dir, sc.toolchain_dir,
            getattr(sc, "shared_cache_dir", None))
    agent_discharger = AgentDischarger(
        backend, sc.artifacts,
        workdir_base=os.path.join(sc.state_dir, "work"),
        cleanup=True,   # snapshots are the evidence; workdirs are scratch
        toolchain_dir=sc.toolchain_dir,
        shared_cache_dir=getattr(sc, "shared_cache_dir", None),
        state_dir=sc.state_dir,
        live_world=sc.live_world,
        store=sc.store,   # evidence-workspace index over the claim graph
        log=sc.log)
    # Process-local, explicit opt-in reuse across the candidates of one task
    # scored in this process.  Off by default: every candidate then follows
    # the fresh Compiler/Critic/Oracle/executor/Judger sequence.
    method_cache = None
    if method_cache_enabled():
        from .engine.method_cache import default_accepted_method_cache
        method_cache = default_accepted_method_cache()
    contract_cache = None
    if contract_cache_enabled():
        from .engine.contract_cache import (
            default_compiled_final_contract_cache,
        )
        contract_cache = default_compiled_final_contract_cache()
    typed_adapters = typed_adapters_enabled()
    typed_judger_packet = typed_judger_packet_enabled()
    if typed_judger_packet and not typed_adapters:
        raise ValueError("typed-adapter Judger packet requires typed adapters")
    if typed_adapters:
        from .execution.adapters import (
            AdapterRoutingDischarger, default_adapter_registry,
        )

        discharger = AdapterRoutingDischarger(
            agent_discharger, default_adapter_registry(), sc.artifacts, sc.log,
            sc.state_dir, live_world=sc.live_world)
    else:
        discharger = agent_discharger
    binder = BindingGame(backend, sc.artifacts, sc.log, discharger,
                         memory=sc.memory, live_world=sc.live_world,
                         method_cache=method_cache,
                         typed_adapter_judger_packet=typed_judger_packet,
                         sufficiency_role=(
                             roles["sufficiency"]
                             if sufficiency_preaudit_enabled() else None),
                         packet_method_reuse=method_cache is not None)
    if parallel_obligations is None:
        parallel_obligations = getattr(sc, "parallel_obligations", None)
    if parallel_claims is None:
        parallel_claims = getattr(sc, "parallel_claims", None)
    if parallel_obligations is None:
        # a compound claim can compile to dozens of obligations; with the
        # executable-form path most of them are a short script plus one
        # no-tools reading, so eight in flight is well inside the seat gate
        parallel_obligations = _parallel_default("GV_PARALLEL_OBLIGATIONS", 8)
    if parallel_claims is None:
        # online (live world): one claim at a time as commits arrive
        parallel_claims = (1 if sc.live_world
                           else _parallel_default("GV_PARALLEL_CLAIMS", 4))
    # the admission screener is an ONLINE economy: with a live world the
    # solver waits on every commit, and a claim the record already settles
    # is assumed rather than checked. Offline scoring seats none — every
    # claim in a scored slice gets its check. GV_SCREENER=0 disables it.
    screener = (SCREENER_AGENT
                if sc.live_world and os.environ.get("GV_SCREENER", "1") != "0"
                else None)
    # online, the task contract lives at the final, which keeps the full
    # adversarial game; a non-final decomposition is the compiler's own by
    # default (GV_CRITIC_ROUNDS_NONFINAL=0). A positive budget seats an
    # ADVISORY review: that many rounds, the last repair standing. Offline
    # scoring keeps the full game everywhere. (An advisory repair was
    # observed to replace a concrete obligation with a vaguer, costlier
    # one; the solver's intermediate feedback is better served by the
    # compiler's decomposition, gated by the same evidence axioms.)
    nonfinal_rounds = None
    if sc.live_world:
        try:
            nonfinal_rounds = max(
                0, int(os.environ.get("GV_CRITIC_ROUNDS_NONFINAL", "0")))
        except ValueError:
            nonfinal_rounds = 0
    return Kernel(sc.store, sc.log, backend, sc.artifacts,
                  obligations_role=obligations,
                  discharger=discharger, binder=binder,
                  memory=sc.memory, live_world=sc.live_world,
                  record_boundary=sc.builder.commit_seq.get,
                  critic_rounds=critic_rounds,
                  contract_cache=contract_cache,
                  parallel_obligations=parallel_obligations,
                  parallel_claims=parallel_claims,
                  screener_role=screener,
                  critic_rounds_nonfinal=nonfinal_rounds,
                  # online: overlap the critic's review with the checks
                  # (GV_SPECULATIVE_DISCHARGE=0 disables)
                  speculative_discharge=(
                      sc.live_world
                      and os.environ.get("GV_SPECULATIVE_DISCHARGE", "1")
                      != "0"))


# MCP wiring

_PROTOCOL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "protocol", "solver-protocol.md")


def _solver_protocol() -> str:
    """The solver-side protocol, served as MCP server instructions: every
    client that mounts gv gets it injected into the solver's context, so
    there is no CLAUDE.md copy to keep in sync. Advisory only — soundness
    does not depend on the solver following it; ignoring it forfeits the
    certificate, never corrupts one."""
    with open(_PROTOCOL_PATH, encoding="utf-8") as f:
        return f.read()


def build_server(service: Service):
    import anyio
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("gv", instructions=_solver_protocol())

    async def off_loop(verb, *args) -> str:
        # Every verb runs in a worker thread, never on the server's event
        # loop. FastMCP runs a synchronous tool on the loop itself, so a
        # commit whose checks took minutes froze the whole server: a status
        # query, the final's own commit, and the certificate request all
        # waited behind it (a final queued up to five hours behind earlier
        # claims' checks). Each commit still returns only once ITS checks
        # are done; different claims' checks now proceed side by side (the
        # store and kernel already verify claims concurrently).
        return await anyio.to_thread.run_sync(
            lambda: json.dumps(verb(*args), ensure_ascii=False))

    @server.tool()
    async def commit(label: str, kind: CLAIM_KINDS, text: str,
                     deps: List[str], revises: str = "") -> str:
        """Commit a claim (kind: assertion|effect|final). deps is a list of
        labels of already-committed claims (including effect:<seq>). An
        assertion's commit returns once the claim is recorded (`checking`:
        its checks run in the background while you work; gv.status(label)
        shows the verdict, and a refutation arrives in the `news` of your
        next gv call). A final is VERIFIED BEFORE ITS COMMIT RETURNS: the
        call blocks for as long as its checks take, and the result carries
        the status and, on anything but `verified`, the obligation that
        failed plus its counterexample."""
        return await off_loop(service.commit, label, kind, text, deps,
                              revises or None)

    @server.tool()
    async def verify(label: str) -> str:
        """Run the full independent check on a committed claim now (also
        for a claim that commit admitted as `confident`); returns its
        status."""
        return await off_loop(service.verify, label)

    @server.tool()
    async def certificate(final_label: str) -> str:
        """Certify a final claim: settle the dependency closure, freeze the
        certificate, and return the verdict plus a slice summary."""
        return await off_loop(service.certificate, final_label)

    @server.tool()
    async def status(label: str = "") -> str:
        """Query the status of one label or of all labels."""
        return await off_loop(service.status, label or None)

    return server


def _instruments_read_any_repository(environ=os.environ) -> None:
    """Git refuses to read a repository owned by another OS user, and the
    verifier's seats and scripts run as whichever user the harness gives
    them, so every inspection of a task repository owned by the task's own
    service user failed with 'dubious ownership'. The override lives in the
    verifier's own process environment, which its instruments inherit and
    the solver's processes do not."""
    if environ.get("GIT_CONFIG_COUNT"):
        return
    environ["GIT_CONFIG_COUNT"] = "1"
    environ["GIT_CONFIG_KEY_0"] = "safe.directory"
    environ["GIT_CONFIG_VALUE_0"] = "*"


def main() -> None:
    _instruments_read_any_repository()
    from .execution.resource_share import yield_server
    yield_server()
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--memory-dir", default=None)
    ap.add_argument("--toolchain-dir", default=None)
    ap.add_argument("--trigger", default=TRIGGER_ON_COMMIT, choices=list(TRIGGERS),
                    help="when a claim is verified (default: at commit)")
    ap.add_argument("--seats", default="sdk", choices=["sdk", "codex"],
                    help="verifier seats: the Agent SDK on the configured "
                    "provider (default) or the Codex CLI")
    ap.add_argument("--effort", default="medium",
                    help="codex seats' reasoning effort (default: medium)")
    ap.add_argument("--service-tier", default="standard",
                    choices=["standard", "fast"],
                    help="codex seats' service tier (default: standard)")
    ap.add_argument("--seat-search", default="on", choices=["on", "off"],
                    help="the seats' web_search knowledge channel; turn it "
                         "off for a benchmark whose answers are published")
    args = ap.parse_args()
    # before anything reads or writes the state: a second process on a
    # served state (a stray `gv-mcp-server </dev/null`, a solver's own
    # launch) leaves no event, no precompile, no pidfile
    try:
        server_lock = hold_server_lock(args.state_dir)
    except StateBusy as busy:
        print(f"gv-mcp-server: {busy}", file=sys.stderr)
        sys.exit(3)
    kernel_factory = None
    if args.seats == "codex":
        from .llm.codex_backend import CodexBackend
        from .llm.direct_backend import build_backend
        codex = CodexBackend(cwd=args.state_dir, model=args.model,
                             effort=args.effort,
                             service_tier=args.service_tier,
                             seat_search=args.seat_search == "on")
        backend = build_backend(harness_roles(args.state_dir),
                                cwd=args.state_dir, model=args.model,
                                agent=codex)

        def kernel_factory(sc):
            return _default_kernel(sc, args.model, backend=backend)
    service = Service(args.state_dir, kernel_factory=kernel_factory,
                      model=args.model,
                      memory_dir=args.memory_dir,
                      toolchain_dir=args.toolchain_dir,
                      trigger=args.trigger,
                      server_lock=server_lock)
    server = build_server(service)   # imports the MCP server modules
    # every module the seats import must be importable by the main thread
    # first: a seat thread and the server importing the same package at
    # once deadlock the import lock (observed at startup)
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        pass
    service.record_start(launch_identity(args))
    service.start_background()
    server.run()   # stdio


if __name__ == "__main__":
    main()
