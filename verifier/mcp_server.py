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
import asyncio
import json
import os
import hashlib
import inspect
import stat
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from verifier.core.model import CLAIM_KINDS, content_hash, canonical
from .profiles import get_profile
from .stream.builder import Builder, TraceError
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
        entry = {"text": o["text"], "status": o["status"]}
        reason = o.get("reason")
        if not reason:
            ev = store.evidence_for(o["id"])
            reason = (str(ev[-1].get("witness"))[:300]
                      if ev and ev[-1].get("witness") else None)
        if reason:
            entry["reason"] = reason
        out.append(entry)
    return out


def _residue(store, claim_id: str) -> Dict[str, Any]:
    """What a non-verified claim owes the solver: the obligations that did not
    hold with their counterexamples — or, when no obligation was ever compiled,
    the fail-closed reason (a refuted premise, say), which is otherwise
    invisible to the solver."""
    obligations = _obligation_summary(store, claim_id)
    if obligations:
        return {"obligations": obligations}
    for ev in reversed(store.evidence_for(claim_id)):
        if ev.get("error"):
            out = {"error": ev["error"]}
            if ev.get("refuted_deps"):
                out["refuted_deps"] = [d[:12] for d in ev["refuted_deps"]]
            return out
    return {"obligations": []}


HOOK_FEED = "hook_feed.jsonl"
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
                 profile: str = "default",
                 project_root: Optional[str] = None,
                 evaluator_spec=None,
                 final_destination: Optional[str] = None,
                 seats: str = "sdk",
                 effort: Optional[str] = None,
                 service_tier: Optional[str] = None,
                 seat_capabilities=None):
        self.toolchain_dir = toolchain_dir
        # a verifier-owned cache root shared across trials (install caches
        # for the executor's instruments); None keeps every cache
        # state-local as before
        self.shared_cache_dir = shared_cache_dir
        self.parallel_obligations = parallel_obligations
        self.parallel_claims = parallel_claims
        self.live_world = live_world
        self.profile = get_profile(profile)
        self.project_root = os.path.realpath(project_root or os.getcwd())
        self.final_destination = (os.path.abspath(os.path.join(self.project_root, final_destination))
                                  if final_destination is not None else None)
        self.seats, self.effort, self.service_tier = seats, effort, service_tier
        self._event_feed_lock = threading.RLock()
        self._cyber_lock = threading.RLock()
        self._cyber_bindings = {}
        self._cyber_adjudications = {}
        self._seat_capabilities = seat_capabilities
        self.candidates = None
        self.evaluator = None
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
        if self.profile.name == "cyber":
            from .profiles.cyber.candidates import CandidateStore
            from .profiles.cyber.evaluator import Evaluator
            self.candidates = CandidateStore(state_dir, self.project_root,
                                             self.artifacts, self.log)
            if evaluator_spec is not None:
                self.evaluator = Evaluator(evaluator_spec, self.artifacts)
        elif evaluator_spec is not None:
            raise ValueError("evaluator configuration requires --cyber")
        self._pin_profile(model)

        if kernel_factory is None:
            kernel_factory = lambda sc: _default_kernel(  # noqa: E731
                sc, model=model, critic_rounds=critic_rounds,
                parallel_obligations=parallel_obligations,
                parallel_claims=parallel_claims)
        self.kernel = kernel_factory(self)  # factory receives the service

        self._feed = Tail(os.path.join(state_dir, HOOK_FEED),
                          self._ingest_hook_event, raw=True)
        # a restarted server rebuilds its view from its own ledger
        for event in self.log.replay():
            if event["type"] == "commit" and "cybergv_binding" in event["payload"]:
                self._restore_binding(event["payload"])
            self.builder.feed(event)
        self._jobs = None
        if self.profile.name == "cyber":
            from .service_jobs import VerificationJobs
            self._jobs = VerificationJobs(self.state_dir)

    # hook feed
    def _ingest_hook_event(self, raw_event: Dict[str, Any]) -> None:
        if (not isinstance(raw_event, dict)
                or not isinstance(raw_event.get("payload", {}), dict)):
            return
        etype = raw_event.get("type")
        if etype not in HOOK_EVENT_TYPES:
            return  # unknown feed line: ignored, the feed is untrusted input
        with self._event_feed_lock:
            payload = raw_event.get("payload", {})
            if self.profile.name == "cyber":
                payload = {**payload, "origin": "solver-origin-untrusted",
                           "authority_scope": "untrusted_solver_observation"}
            event = self.log.append(etype, payload)
            self.builder.feed(event)

    def drain_hooks(self) -> int:
        with self._event_feed_lock:
            return self._feed.poll()

    # MCP verbs
    def commit(self, label: str, kind: str, text: str,
               deps: List[str],
               revises: Optional[str] = None) -> Dict[str, Any]:
        try:
            self._check_task()
        except (OSError, ValueError) as err:
            return {"error": str(err)}
        if label.startswith(("cyber-final:", "effect:")):
            return {"error": "label is reserved for host-owned records"}
        return self._commit(label, kind, text, deps, revises)

    def _commit(self, label, kind, text, deps, revises=None, binding=None):
        self.drain_hooks()
        payload = {"label": label, "kind": kind, "text": text, "deps": deps}
        if revises:
            payload["revises"] = revises
        failure = None
        try:
            with self._event_feed_lock:
                claim = self.builder.claim_from_commit(payload)
                self.builder.require_available_label(label, claim.id)
                if binding is None and claim.id in self._cyber_bindings:
                    raise ValueError("public claims cannot reuse a host candidate final identity")
                if binding is not None:
                    try:
                        self.store.get(claim.id)
                    except UnknownDependency:
                        pass
                    else:
                        raise ValueError("host scope cannot be attached to an existing ordinary claim")
                    payload["cybergv_binding"] = binding
                event = self.log.append("commit", payload)
                if binding is not None:
                    self._cyber_bindings[claim.id] = binding
                self.builder.feed(event)
        except VERB_ERRORS as err:
            failure = str(err)
        if label not in self.builder.labels:
            return {"error": failure or "commit failed"}
        claim_id = self.builder.labels[label]
        status = self.store.status(claim_id)
        result = {"claim_id": claim_id, "status": status}
        if failure is not None:
            # the claim is recorded; its check did not finish — fail-closed, so
            # the status stands as unverified and the solver is told why
            result["error"] = failure
        if kind != "effect" and status != "verified":
            # the verdict lands on the commit call now, so its residue must too
            result.update(_residue(self.store, claim_id))
        return result

    def verify(self, label: str) -> Dict[str, Any]:
        self.drain_hooks()
        try:
            self._check_task()
            claim_id = self.builder.resolve(label)
            self.log.append("verify_request", {"target": claim_id, "cause": "mcp"})
            status = self.kernel.verify(claim_id)
        except VERB_ERRORS + (OSError,) as err:
            return {"error": str(err)}
        result = {"claim_id": claim_id, "status": status}
        if status != "verified":
            # every non-verified status owes the solver its residue: which
            # obligation failed and the counterexample witness. Withholding
            # it on `refuted` left the solver with nothing to act on but the
            # claim's wording — the mechanism behind claim-shopping.
            result.update(_residue(self.store, claim_id))
        return result

    def certificate(self, final_label: str) -> Dict[str, Any]:
        self.drain_hooks()
        try:
            self._check_task()
            final_id = self.builder.resolve(final_label)
            if final_id in self._cyber_bindings:
                self._validate_binding(self._cyber_bindings[final_id])
            cert = request_certificate(self.store, self.kernel, self.log,
                                       self.artifacts, final_id,
                                       profile_metadata=self._certificate_metadata(final_id))
        except VERB_ERRORS + (OSError,) as err:
            return {"error": str(err)}
        slice_summary = []
        for e in cert["slice"]:
            entry = {"text": e["text"], "status": e["status"]}
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
        result = {
            "verdict": cert["verdict"],
            "certificate_id": cert["certificate_id"],
            "slice": slice_summary,
        }
        if "profile_metadata" in cert:
            result["profile_metadata"] = cert["profile_metadata"]
        return result

    def status(self, label: Optional[str] = None) -> Dict[str, Any]:
        if self._jobs is not None:
            if not self._event_feed_lock.acquire(blocking=False):
                return {"pending": True, "verified": False,
                        "status_snapshot": "unavailable_while_verifying",
                        "jobs": self._jobs.progress()["jobs"]}
            try:
                self._feed.poll()
                if label is None:
                    labels = tuple(self.builder.labels.items())
                else:
                    claim_id = self.builder.resolve(label)
            except TraceError as err:
                return {"error": str(err)}
            finally:
                self._event_feed_lock.release()
            if label is None:
                return {name: self.store.status(cid) for name, cid in labels}
            row = self.store.get(claim_id)
            result = {key: row[key] for key in ("id", "kind", "text", "status")}
            if row["status"] != "verified":
                result.update(_residue(self.store, claim_id))
            return result
        self.drain_hooks()
        if label is not None:
            try:
                claim_id = self.builder.resolve(label)
            except TraceError as err:
                return {"error": str(err)}
            row = self.store.get(claim_id)
            result = {k: row[k] for k in ("id", "kind", "text", "status")}
            if row["status"] != "verified":
                result.update(_residue(self.store, claim_id))
            return result
        # New commits and hook Effects may add labels on another worker.
        # Snapshot only the mapping under the existing feed lock, then release
        # it before querying the store; status does not hold a verification lock.
        with self._event_feed_lock:
            labels = tuple(self.builder.labels.items())
        return {
            label: self.store.status(cid)
            for label, cid in labels
        }

    def _pin_profile(self, model):
        pins = [e["payload"] for e in self.log.replay()
                if e["type"] == "verification_profile"]
        identity = {"profile": self.profile.name, "profile_id": self.profile.id,
                    "project_root": self.project_root,
                    "final_destination": self.final_destination,
                    "evaluator_id": self.evaluator.spec.id if self.evaluator else None,
                    "runtime": {"model": model, "seats": self.seats,
                                "effort": self.effort, "service_tier": self.service_tier}}
        if self.profile.name != "default":
            from .execution.config import (
                typed_adapters_enabled, typed_judger_packet_enabled,
                contract_cache_enabled, method_cache_enabled,
                sufficiency_preaudit_enabled, append_only_deltas_enabled,
            )
            identity["execution_policy"] = {
                "typed_adapters": typed_adapters_enabled(),
                "typed_judger_packet": typed_judger_packet_enabled(),
                "contract_cache": contract_cache_enabled(),
                "method_cache": method_cache_enabled(),
                "sufficiency_preaudit": sufficiency_preaudit_enabled(),
                "append_only_deltas": append_only_deltas_enabled(),
                "adapter_boundary": "public-evidence-frozen-record-v1",
            }
            # Pin effective defaults too: an omitted --model must not let a
            # later provider/default-model change reuse earlier judgments.
            if self.seats == "codex":
                from .llm.codex_backend import DEFAULT_MODEL
                from .llm.codex_service import normalize_service_tier
                effective = {"model": model or DEFAULT_MODEL,
                    "effort": self.effort or "medium",
                    "service_tier": normalize_service_tier(self.service_tier)}
            else:
                from .llm import provider
                effective = {"model": model or provider.MODEL,
                    "provider": provider.PROVIDER,
                    "endpoint_fingerprint": content_hash(provider.ANTHROPIC_BASE_URL),
                    "text_only": provider.TEXT_ONLY}
            identity["runtime"]["effective"] = effective
            identity["task_sha256"] = (self._task_digest()
                if os.path.isfile(os.path.join(self.state_dir, "task.md")) else None)
        self._profile_pin = identity
        if pins:
            if any(pin != identity for pin in pins):
                raise ValueError("state belongs to a different verification profile/runtime")
        elif self.profile.name != "default":
            if any(e["type"] in ("commit", "obligations_compiled") for e in self.log.replay()):
                raise ValueError("use a fresh state directory for the cyber profile")
            self.log.append("verification_profile", identity)

    def _task_digest(self):
        with open(os.path.join(self.state_dir, "task.md"), "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    def _check_task(self):
        if self.profile.name == "default":
            return
        current = (self._task_digest()
            if os.path.isfile(os.path.join(self.state_dir, "task.md")) else None)
        if current != self._profile_pin["task_sha256"]:
            raise ValueError("authoritative task changed; use a fresh verification state")

    def _auth_artifact(self, digest):
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid artifact identity")
        artifact = self.artifacts.get(digest)
        if content_hash(artifact) != digest:
            raise ValueError("artifact content no longer matches its identity")
        return artifact

    def _observations(self, candidate_id):
        return [e["payload"] for e in self.log.replay()
                if e["type"] == "cyber_observation_recorded"
                and e["payload"].get("candidate_id") == candidate_id]

    def _validate_binding(self, binding):
        if self.candidates is None or binding.get("certificate_scope") != "candidate":
            raise ValueError("candidate binding requires host cyber scope")
        if binding.get("profile_id") != self.profile.id:
            raise ValueError("candidate binding belongs to a different profile")
        if binding.get("task_sha256") != self._task_digest():
            raise ValueError("authoritative task changed after candidate certification")
        # History remains evidence, but a new binding authorizes delivery only
        # within the host capability scope under which it was certified.
        # Legacy bindings without this field retain their existing rules.
        if ("capability_revision" in binding
                and binding["capability_revision"] != self._evidence_manifest(
                    "obligations")["capability_revision"]):
            raise ValueError("candidate host capability scope changed")
        # Artifact metadata alone cannot authenticate the current blob bytes.
        with self.candidates.lease(binding["subject"]["candidate_id"]) as lease:
            candidate = lease.candidate
        if any(binding["subject"].get(key) != candidate.get(key)
               for key in ("candidate_id", "blob_path", "sha256", "size")):
            raise ValueError("candidate binding identity changed")
        if self.final_destination and binding["subject"].get("address") != self.final_destination:
            raise ValueError("candidate binding has a different canonical destination")
        expected_id = content_hash({k: v for k, v in binding.items() if k != "context_id"})
        if binding.get("context_id") != expected_id:
            raise ValueError("candidate verification context changed")
        for observation in binding.get("observations", []):
            if observation not in self._observations(candidate["candidate_id"]):
                raise ValueError("candidate observation is not host-retained evidence")
            artifact = self._auth_artifact(observation["artifact_id"])
            if (artifact.get("candidate_id") != candidate["candidate_id"]
                    or artifact.get("candidate_sha256") != candidate["sha256"]
                    or artifact.get("candidate_size") != candidate["size"]
                    or any(artifact.get(key) != observation.get(key)
                           for key in ("evaluator_id", "environment", "authority_scope"))
                    or self.evaluator is None or observation["evaluator_id"] != self.evaluator.spec.id):
                raise ValueError("evaluator observation has a different subject or authority context")
            event = self.log.get(observation["event_seq"])
            if (event is None or event["type"] != "tool_result"
                    or EventLog.event_digest(event) != observation["event_digest"]
                    or event["payload"].get("output", {}).get("artifact_id") != observation["artifact_id"]
                    or observation["effect_ref"] != f"effect:{event['seq']}"
                    or self.builder.resolve(observation["effect_ref"]) != observation["effect_claim_id"]):
                raise ValueError("candidate observation effect changed")
        return candidate

    def _restore_binding(self, payload):
        binding = payload["cybergv_binding"]
        self._validate_binding(binding)
        if payload["kind"] != "final" or not payload.get("label", "").startswith("cyber-final:"):
            raise ValueError("candidate scope is not a host final commit")
        text = json.loads(payload["text"])
        if text != {"schema": "gv-cyber-candidate-v1", "binding": binding}:
            raise ValueError("candidate scope does not match its final statement")
        claim = self.builder.claim_from_commit(payload)
        if claim.id in self.builder.labels.values() and claim.id not in self._cyber_bindings:
            raise ValueError("host scope cannot promote an earlier ordinary claim during replay")
        self._cyber_bindings[claim.id] = binding
        self._cyber_adjudications[self._adjudication_key(binding)] = {
            "claim_id": claim.id, "label": payload["label"], "binding": binding}

    @staticmethod
    def _adjudication_key(binding):
        return content_hash({key: value for key, value in binding.items()
                             if key not in ("context_id", "verification_epoch")})

    def _evidence_manifest(self, role, claim_id=None, binding=None):
        from .profiles.cyber.evidence import SeatCapabilities, build_evidence_manifest
        capabilities = self._seat_capabilities or SeatCapabilities()
        observations = []
        events = {}
        calls = {}
        if role not in ("obligations", "critic", "oracles", "sufficiency"):
            snapshot = list(self.log.replay())
            events = {event["seq"]: event for event in snapshot}
            if binding is not None:
                observations = binding["observations"]
            elif claim_id is not None:
                closure = set(self.store.backward_slice(claim_id))
                observations = [event["payload"] for event in snapshot
                    if event["type"] == "cyber_observation_recorded"
                    and event["payload"].get("effect_claim_id") in closure]
            needed = {events[observation["event_seq"]]["payload"].get("tool_use_id")
                      for observation in observations}
            for event in snapshot:
                if (event["type"] == "tool_call"
                        and event["payload"].get("tool") == "cyber_evaluator"
                        and event["payload"].get("tool_use_id") in needed
                        and event["payload"].get("authority_scope") != "untrusted_solver_observation"):
                    call_id = event["payload"].get("tool_use_id")
                    if call_id in calls:
                        raise ValueError("duplicate host evaluator execution identity")
                    calls[call_id] = event

        def subject(candidate_id):
            with self.candidates.lease(candidate_id) as lease:
                candidate = dict(lease.candidate)
                self.candidates.validate(lease)
            return candidate

        return build_evidence_manifest(
            evaluator_spec=self.evaluator.spec if self.evaluator else None,
            capabilities=capabilities, observations=observations,
            candidate_provider=subject, artifact_provider=self._auth_artifact,
            artifact_root=self.artifacts.root,
            event_provider=events.get, tool_call_provider=calls.get,
            effect_resolver=self.builder.resolve,
            answer_blind=role in ("obligations", "critic", "oracles", "sufficiency"))

    def profile_context(self, role, claim_id):
        context = {"profile": self.profile.name, "profile_id": self.profile.id,
                   "certificate_scope": "task"}
        binding = self._cyber_bindings.get(claim_id)
        if binding:
            context.update({k: v for k, v in binding.items() if k != "observations"})
            if role not in ("obligations", "critic", "oracles", "sufficiency"):
                context["observations"] = binding["observations"]
        if self.profile.name == "cyber":
            context["execution_capabilities"] = self._evidence_manifest(role, claim_id, binding)
        return canonical(context)

    def _job_request(self, tool, args):
        self._check_task()
        snapshot = list(self.log.replay())
        revisions = []
        for event in snapshot:
            payload = event["payload"]
            if event["type"] != "commit" or not payload.get("revises"):
                continue
            # A revision's own output does not change that request's identity.
            same_request = (tool == "commit" and len(args) == 5
                and [payload.get("label"), payload.get("kind"), payload.get("text"),
                     payload.get("deps", []), payload.get("revises")] == args)
            if not same_request:
                revisions.append(EventLog.event_digest(event))
        scope = {"profile": self._profile_pin,
                 "graph_revision": content_hash(revisions),
                 "capability_revision": self._evidence_manifest("obligations")["capability_revision"]}
        if tool in ("cyber_finalize_candidate", "cyber_evaluate_candidate"):
            with self.candidates.lease(args[0]) as lease:
                scope["subject"] = {key: lease.candidate[key]
                                    for key in ("candidate_id", "sha256", "size")}
                self.candidates.validate(lease)
            if self.evaluator is not None:
                self.evaluator.spec.validate()
            scope["observations"] = self._observations(args[0])
            if tool == "cyber_finalize_candidate":
                address = os.path.abspath(os.path.join(self.project_root,
                    args[1] or self.final_destination or ""))
                if not args[1] and self.final_destination is None:
                    raise ValueError("a canonical destination must be specified")
                if self.final_destination and address != self.final_destination:
                    raise ValueError("destination differs from host-required canonical path")
                prior = next((event["payload"] for event in reversed(list(self.log.replay()))
                    if event["type"] == "cyber_materialized"
                    and event["payload"].get("candidate_id") == args[0]
                    and event["payload"].get("receipt", {}).get("destination") == address), None)
                if prior is None:
                    self.candidates.preflight_destination(address)
                else:
                    self._authenticate_candidate_certificate(prior["certificate_id"],
                        prior["final_claim_id"], prior.get("full_certificate_id"))
                    self.candidates.verify_delivery(prior["receipt"])
                # Aliases of the same canonical address are one request.
                args = [args[0], address]
        return {"tool": tool, "args": args, "scope": scope}

    def _job_operation(self, request):
        tool, args = request["tool"], request["args"]

        def operation(retry):
            # Request identity is checked again when the serialized worker gets
            # to it. Evidence may have changed while another job was running.
            if self._job_request(tool, args) != request:
                return {"error": "verification inputs changed while queued; resubmit",
                        "verified": False, "materialized": False, "delivery_pending": True}
            if tool == "cyber_finalize_candidate":
                return self.cyber_finalize_candidate(*args, retry=retry)
            if tool == "commit" and self.store.trigger == TRIGGER_ON_COMMIT:
                # On server recovery a recorded claim is already in SQLite.
                # add_claim is a no-op for that identity, so it cannot fire the
                # on-commit trigger a second time. Resume only that unresolved
                # recorded claim, without recompiling settled obligation nodes.
                known = args[0] in self.builder.labels
                result = self.commit(*args)
                if known and not result.get("error") and result.get("status") in ("unverified", "stale"):
                    return self.verify(args[0])
                return result
            return getattr(self, tool)(*args)
        return operation

    def verification_job(self, job_id: str):
        if self._jobs is None:
            return {"error": "recoverable jobs require the cyber profile"}
        return self._jobs.status(job_id)

    def verification_jobs(self):
        return self._jobs.progress() if self._jobs is not None else {"jobs": []}

    def verification_retry(self, job_id: str):
        if self._jobs is None:
            return {"error": "recoverable jobs require the cyber profile"}
        try:
            request = self._jobs.request(job_id)
            if self._job_request(request["tool"], request["args"]) != request:
                raise ValueError("verification inputs changed; resubmit the original operation")
            return self._jobs.retry(job_id, self._job_operation(request))
        except (OSError, ValueError) as err:
            return {"error": str(err), "verified": False}

    def profile_context_id(self, claim_id):
        if self.profile.name == "default":
            return ""
        binding = self._cyber_bindings.get(claim_id)
        return (binding["context_id"] if binding
                else content_hash({"profile_id": self.profile.id, "scope": "task",
                                   "task_sha256": self._profile_pin["task_sha256"]}))

    def _certificate_metadata(self, claim_id):
        if self.profile.name == "default":
            return None
        binding = self._cyber_bindings.get(claim_id)
        return {"profile": self.profile.name, "profile_id": self.profile.id,
                "certificate_scope": "candidate" if binding else "task",
                "task_sha256": self._profile_pin["task_sha256"],
                "context_id": self.profile_context_id(claim_id),
                **({"binding": binding, "delivery_pending": True} if binding else {})}

    def cyber_checkpoint(self, working_theory: str, next_actions: Optional[List[str]] = None):
        if self.candidates is None:
            return {"error": "cyber profile is not active"}
        event = self.log.append("cyber_checkpoint", {
            "working_theory": working_theory, "next_actions": next_actions or []})
        return {"event_seq": event["seq"], "verified": False}

    def cyber_register_candidate(self, path: str, label: str = ""):
        if self.candidates is None:
            return {"error": "cyber profile is not active"}
        try:
            with self._cyber_lock:
                return self.candidates.register(path, label)
        except (OSError, ValueError) as err:
            return {"error": str(err)}

    def cyber_candidates(self):
        return {"candidates": self.candidates.candidates() if self.candidates else [],
                "verified": False}

    def cyber_evaluate_candidate(self, candidate_id: str):
        if self.candidates is None or self.evaluator is None:
            return {"error": "no task-authorized evaluator is configured"}
        try:
            with self._cyber_lock:
                candidate = self.candidates.get(candidate_id)
                call_id = uuid.uuid4().hex
                with self._event_feed_lock:
                    self.builder.feed(self.log.append("tool_call", {
                        "tool": "cyber_evaluator", "tool_use_id": call_id,
                        "origin": "verifier_controller", "authority_scope": "host_evaluator_execution",
                        "input": {"candidate_id": candidate_id,
                                  "evaluator_id": self.evaluator.spec.id}}))
                try:
                    result = self.evaluator.run(candidate)
                except (OSError, ValueError) as err:
                    result = {"error": str(err), "candidate_id": candidate_id}
                with self._event_feed_lock:
                    event = self.log.append("tool_result", {
                        "tool": "cyber_evaluator", "tool_use_id": call_id,
                        "origin": "verifier_controller", "authority_scope": "host_evaluator_execution",
                        "output": result})
                    self.builder.feed(event)
                    effect_ref = f"effect:{event['seq']}"
                if "artifact_id" in result:
                    observation = {"candidate_id": candidate_id,
                        "artifact_id": result["artifact_id"],
                        "artifact_path": result["artifact_path"],
                        "effect_ref": effect_ref,
                        "effect_claim_id": self.builder.resolve(effect_ref),
                        "event_seq": event["seq"], "event_digest": EventLog.event_digest(event),
                        "evaluator_id": result["evaluator_id"],
                        "environment": result["environment"],
                        "authority_scope": result["authority_scope"]}
                    self.log.append("cyber_observation_recorded", observation)
                return {**result, "effect_ref": effect_ref,
                        "effect_claim_id": self.builder.resolve(effect_ref), "verified": False}
        except (OSError, ValueError) as err:
            return {"error": str(err)}

    def _certificate_matches_store(self, certificate):
        """Check a retained snapshot without scheduling any verification work."""
        final_id = certificate.get("final")
        entries = certificate.get("slice")
        if not isinstance(entries, list) or [entry.get("id") for entry in entries] \
                != self.store.backward_slice(final_id):
            raise ValueError("certificate dependency slice changed")
        memo = {}
        for entry in entries:
            row = self.store.get(entry["id"])
            effective = self.store.effective(entry["id"], memo)
            expected = {key: row[key] for key in ("kind", "text", "status", "reason", "basis", "strength")}
            expected.update({key: effective[key] for key in (
                "effective_status", "effective_basis", "effective_strength", "open_premises", "premise_refuted")})
            if any(entry.get(key) != value for key, value in expected.items()) \
                    or entry.get("evidence") != self.store.evidence_for(entry["id"]):
                raise ValueError("certificate no longer matches the settled graph")
            nodes = self.store.obligation_children(entry["id"])
            exported = entry.get("obligations")
            if not isinstance(exported, list) or len(nodes) != len(exported):
                raise ValueError("certificate obligation slice changed")
            for node, retained in zip(nodes, exported):
                expected_node = {key: node[key] for key in ("id", "text", "status", "reason", "basis", "strength")}
                expected_node.update(severity=node.get("severity", "must"),
                    clause=node.get("clause", ""),
                    refutable_by_absence=bool(node.get("refutable_by_absence", False)),
                    evidence=self.store.evidence_for(node["id"]))
                if retained != expected_node:
                    raise ValueError("certificate obligation evidence changed")

    def _authenticate_candidate_certificate(self, certificate_id, final_id, full_certificate_id=None):
        """Authenticate the full GV verdict and, when present, its narrow scope.

        A MUST acceptance certificate cannot overwrite the full GV verdict. It
        links that artifact and is recomputed from its currently settled graph.
        No positive evaluator observation can enter this authorization path.
        """
        artifact = self._auth_artifact(certificate_id)
        if self.evaluator is not None:
            self.evaluator.spec.validate()
        binding = self._cyber_bindings.get(final_id)
        if binding is None or artifact.get("final") != final_id \
                or self.store.status(final_id) != "verified":
            raise ValueError("certificate has no matching current candidate final")
        self._validate_binding(binding)
        if binding["observations"] != self._observations(binding["subject"]["candidate_id"]):
            raise ValueError("new candidate observations require fresh adjudication")
        full = artifact
        if artifact.get("schema") == "gv-cyber-candidate-acceptance-v1":
            from .profiles.cyber.acceptance import build_candidate_acceptance
            if full_certificate_id is None or artifact.get("full_certificate_id") != full_certificate_id:
                raise ValueError("acceptance certificate has no designated full certificate")
            full = self._auth_artifact(full_certificate_id)
            if build_candidate_acceptance(full, binding) != artifact:
                raise ValueError("acceptance certificate does not match the full GV audit")
        elif full_certificate_id is not None or artifact.get("verdict") != "verified":
            raise ValueError("full candidate certificate is not verified")
        if (full.get("final") != final_id
                or full.get("profile_metadata") != self._certificate_metadata(final_id)):
            raise ValueError("certificate does not authorize the exact candidate binding")
        self._certificate_matches_store(full)
        # Evidence identities are not enough if their designated executor
        # snapshot has disappeared or changed. Follow only schema-defined
        # winning links, never arbitrary paths or digest-looking solver text.
        for entry in full["slice"]:
            if entry["id"] != final_id:
                continue
            for node in entry["obligations"]:
                if node.get("status") != "verified" or not node.get("evidence"):
                    continue
                digest = node["evidence"][-1].get("artifact")
                if not digest:
                    continue  # Legacy direct/graph evidence keeps its semantics.
                evidence = self._auth_artifact(digest)
                if evidence.get("instrument") == "binding-game" and evidence.get("bound") is not None:
                    bound = evidence["bound"]
                    if not isinstance(bound, dict) or not bound.get("discharge_artifact"):
                        raise ValueError("winning evidence has no executor snapshot")
                    self._auth_artifact(bound["discharge_artifact"])
        return artifact, full

    def _retained_candidate_certificate(self, final_id):
        for event in reversed(list(self.log.replay())):
            if event["type"] == "certificate" and event["payload"].get("final") == final_id:
                digest = event["payload"]["digest"]
                artifact = self._auth_artifact(digest)
                if artifact.get("final") != final_id \
                        or artifact.get("profile_metadata") != self._certificate_metadata(final_id):
                    raise ValueError("retained candidate certificate has different scope")
                self._certificate_matches_store(artifact)
                return {"verdict": artifact["verdict"], "certificate_id": digest,
                        "profile_metadata": artifact["profile_metadata"],
                        "slice": [{key: entry[key] for key in ("text", "status")}
                                  for entry in artifact["slice"]]}
        return None

    def cyber_finalize_candidate(self, candidate_id: str, destination: str = "", *, retry: bool = False):
        if self.candidates is None:
            return {"error": "cyber profile is not active"}
        receipt = None
        cert = {}
        try:
            with self._cyber_lock:
                self._check_task()
                address = os.path.abspath(os.path.join(self.project_root,
                    destination or self.final_destination or ""))
                if not destination and self.final_destination is None:
                    raise ValueError("a canonical destination must be specified")
                if self.final_destination and address != self.final_destination:
                    raise ValueError("destination differs from host-required canonical path")
                for event in reversed(list(self.log.replay())):
                    if (event["type"] == "cyber_materialized"
                            and event["payload"].get("candidate_id") == candidate_id
                            and event["payload"]["receipt"].get("destination") == address):
                        prior = event["payload"]
                        final_id = prior["final_claim_id"]
                        binding = self._cyber_bindings.get(final_id)
                        artifact, _ = self._authenticate_candidate_certificate(prior["certificate_id"],
                            final_id, prior.get("full_certificate_id"))
                        with self.candidates.lease(candidate_id) as lease:
                            self._validate_binding(binding)
                            if binding["observations"] != self._observations(candidate_id):
                                raise ValueError("new candidate observations require fresh adjudication")
                            self.candidates.verify_delivery(prior["receipt"])
                            self.candidates.validate(lease)
                        return {"verdict": "verified", "certificate_id": prior["certificate_id"],
                                **({"full_certificate_id": prior["full_certificate_id"],
                                    "full_verdict": artifact["full_verdict"]} if "full_certificate_id" in prior else {}),
                                "final_claim_id": final_id, "profile_metadata": artifact["profile_metadata"],
                                "receipt": prior["receipt"], "receipt_event_seq": event["seq"],
                                "materialized": True, "delivery_pending": False, "official_score": None}
                # Reject impossible paths before spending any verifier calls.
                address = self.candidates.preflight_destination(address)
                with self.candidates.lease(candidate_id) as lease:
                    candidate = lease.candidate
                    binding = {"certificate_scope": "candidate", "profile_id": self.profile.id,
                        "task_sha256": self._task_digest(), "delivery_pending": True,
                        "capability_revision": self._evidence_manifest("obligations")["capability_revision"],
                        # A live world is not an immutable capture. Separate
                        # attempts cannot blindly inherit world conclusions.
                        "verification_epoch": uuid.uuid4().hex,
                        "subject": {k: candidate[k] for k in ("candidate_id", "blob_path", "sha256", "size")},
                        "observations": self._observations(candidate_id)}
                    binding["subject"]["address"] = address
                    binding["subject"]["address_source"] = ("host_required" if self.final_destination else "candidate_proposal")
                    key = self._adjudication_key(binding)
                    prior = self._cyber_adjudications.get(key)
                    cert = None
                    if prior is not None and retry:
                        retained = self._retained_candidate_certificate(prior["claim_id"])
                        if retained is not None and retained.get("verdict") == "verified":
                            retry = False  # Retry delivery, not an already complete proof.
                        elif retained is not None and retained.get("verdict") == "supported":
                            from .profiles.cyber.acceptance import AcceptanceError, build_candidate_acceptance
                            try:
                                build_candidate_acceptance(self._auth_artifact(retained["certificate_id"]), prior["binding"])
                                retry = False
                            except AcceptanceError:
                                pass
                    if prior is not None and not retry:
                        binding, label = prior["binding"], prior["label"]
                        self._validate_binding(binding)
                        commit = {"claim_id": prior["claim_id"]}
                        cert = self._retained_candidate_certificate(commit["claim_id"])
                    else:
                        binding["context_id"] = content_hash(binding)
                        label = "cyber-final:" + uuid.uuid4().hex
                        # Record the stable identity before entering the kernel.
                        # A restarted service restores it from the commit event.
                        commit = self._commit(label, "final",
                            canonical({"schema": "gv-cyber-candidate-v1", "binding": binding}),
                            [o["effect_ref"] for o in binding["observations"]], binding=binding)
                        if "claim_id" in commit:
                            self._cyber_adjudications[key] = {"binding": binding,
                                "label": label, "claim_id": commit["claim_id"]}
                        if "claim_id" not in commit or commit.get("error"):
                            return {**commit, "delivery_pending": True, "materialized": False}
                    if cert is None:
                        cert = self.certificate(label)
                    cert["final_claim_id"] = commit["claim_id"]
                    if cert.get("verdict") == "supported":
                        from .profiles.cyber.acceptance import AcceptanceError, build_candidate_acceptance
                        try:
                            full_id = cert["certificate_id"]
                            full = self._auth_artifact(full_id)
                            self._certificate_matches_store(full)
                            acceptance = build_candidate_acceptance(full, binding)
                            digest = self.artifacts.put(acceptance)
                            self.log.append("cyber_acceptance_certified", {
                                "final": commit["claim_id"], "digest": digest,
                                "full_certificate_id": full_id, "context_id": binding["context_id"]})
                            cert.update(verdict="verified", certificate_id=digest,
                                full_certificate_id=full_id, full_verdict=full["verdict"],
                                profile_metadata=acceptance["profile_metadata"])
                        except AcceptanceError as err:
                            cert["acceptance_rejection"] = err.reason_code
                    if cert.get("verdict") != "verified":
                        return {**cert, "delivery_pending": True, "materialized": False}
                    self._authenticate_candidate_certificate(cert["certificate_id"],
                        commit["claim_id"], cert.get("full_certificate_id"))
                    self._validate_binding(binding)
                    if binding["observations"] != self._observations(candidate_id):
                        raise ValueError("new candidate observations invalidate the prepared certificate")
                    self.candidates.validate(lease)
                    receipt = self.candidates.materialize(lease, address)
                # Lease exit validates once more before authenticating delivery.
                self.candidates.verify_delivery(receipt)
                with self._event_feed_lock:
                    # Revalidate after byte writing/lease exit as well: a
                    # concurrent graph revision must not leave an authorized
                    # receipt that the next host poll would immediately reject.
                    self._authenticate_candidate_certificate(cert["certificate_id"],
                        commit["claim_id"], cert.get("full_certificate_id"))
                    self.candidates.verify_delivery(receipt)
                    event = self.log.append("cyber_materialized", {
                        "receipt": receipt, "candidate_id": candidate_id,
                        **({"full_certificate_id": cert["full_certificate_id"]} if "full_certificate_id" in cert else {}),
                        "certificate_id": cert["certificate_id"], "final_claim_id": commit["claim_id"]})
                return {**cert, "materialized": True, "delivery_pending": False,
                        "receipt": receipt, "receipt_event_seq": event["seq"],
                        "official_score": None}
        except (OSError, ValueError, KeyError) as err:
            if receipt is not None:
                self.candidates.rollback(receipt)
            return {**(cert or {}), "error": str(err), "delivery_pending": True, "materialized": False}

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
    from dataclasses import replace

    from .llm import prompts
    from .llm.agents import (
        ALL_AGENT_ROLES, AgentBackend,
    )
    from .execution.config import (
        contract_cache_enabled,
        method_cache_enabled,
        sufficiency_preaudit_enabled,
        typed_adapters_enabled,
        typed_judger_packet_enabled,
    )
    from .execution.discharge import AgentDischarger
    from .engine.game import BindingGame

    # The mounting harness's ground truth (task.md / baseline/) is the
    # toggle: when present (experiment mounts write it; the interactive
    # mount does not), the obligation compiler is told to compile
    # against it. The backend resolves roles by name, so the swapped
    # role must go into its registry too, not just into the Kernel.
    profile = getattr(sc, "profile", get_profile())
    roles = profile.apply_roles(ALL_AGENT_ROLES)
    obligations = roles["obligations"]
    if profile.name == "default" and (os.path.exists(os.path.join(sc.state_dir, "task.md")) or \
            os.path.isdir(os.path.join(sc.state_dir, "baseline"))):
        obligations = replace(
            obligations, system=obligations.system + "\n"
            + prompts.OBLIGATIONS_HARNESS_ADDENDUM)
    roles["obligations"] = obligations
    if backend is None:
        if getattr(sc, "seats", "sdk") == "codex":
            from .llm.codex_backend import CodexBackend
            backend_type = CodexBackend
            backend_options = {}
            if profile.name == "cyber":
                from .profiles.cyber.backend import CyberCodexBackend
                backend_type = CyberCodexBackend
                backend_options = {"timeout": None,
                    "project_root": getattr(sc, "project_root", None)}
            backend = backend_type(roles=roles, cwd=sc.state_dir, model=model,
                effort=sc.effort or "medium", service_tier=sc.service_tier or "standard",
                **backend_options)
        else:
            backend = AgentBackend(roles=roles, cwd=sc.state_dir, model=model)
    else:
        # an injected backend (e.g. CodexBackend) still needs the
        # harness-swapped role registry — same double-registration rule
        backend.roles = roles
    if profile.disable_external_search and hasattr(backend, "search_roles"):
        backend.search_roles = frozenset()
    context_provider = getattr(sc, "profile_context", None)
    validate_tool_scope = getattr(
        backend, "validate_offline_tool_scope", None)
    if not sc.live_world and validate_tool_scope is not None:
        validate_tool_scope(
            sc.state_dir, sc.toolchain_dir,
            getattr(sc, "shared_cache_dir", None))
    agent_discharger = AgentDischarger(
        backend, sc.artifacts,
        role=roles["discharge"],
        workdir_base=os.path.join(sc.state_dir, "work"),
        cleanup=True,   # snapshots are the evidence; workdirs are scratch
        toolchain_dir=sc.toolchain_dir,
        shared_cache_dir=getattr(sc, "shared_cache_dir", None),
        state_dir=sc.state_dir,
        live_world=sc.live_world,
        store=sc.store,   # evidence-workspace index over the claim graph
        log=sc.log, context_provider=context_provider)
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
        from .profiles.cyber.adapters import CyberAdapterContext

        discharger = AdapterRoutingDischarger(
            agent_discharger, default_adapter_registry(), sc.artifacts, sc.log,
            sc.state_dir, live_world=sc.live_world,
            context_factory=(
                (lambda row: CyberAdapterContext(sc.state_dir, sc.live_world,
                    sc.log, row["id"], getattr(sc, "project_root", None)))
                if profile.name == "cyber" else None))
    else:
        discharger = agent_discharger
    oracle_feedback_filter = None
    if profile.name == "cyber":
        from .profiles.cyber.feedback import filter_oracle_feedback
        oracle_feedback_filter = filter_oracle_feedback
    binder = BindingGame(backend, sc.artifacts, sc.log, discharger,
                         oracles_role=roles["oracles"], judger_role=roles["judger"],
                         memory=sc.memory, live_world=sc.live_world,
                         context_provider=context_provider,
                         oracle_feedback_filter=oracle_feedback_filter,
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
        parallel_obligations = _parallel_default("GV_PARALLEL_OBLIGATIONS", 8)
    if parallel_claims is None:
        # online (live world): one claim at a time as commits arrive
        parallel_claims = (1 if sc.live_world
                           else _parallel_default("GV_PARALLEL_CLAIMS", 4))
    return Kernel(sc.store, sc.log, backend, sc.artifacts,
                  obligations_role=obligations,
                  critic_role=roles["critic"],
                  discharger=discharger, binder=binder,
                  memory=sc.memory, live_world=sc.live_world,
                  record_boundary=sc.builder.commit_seq.get,
                  critic_rounds=critic_rounds,
                  contract_cache=contract_cache,
                  parallel_obligations=parallel_obligations,
                  parallel_claims=parallel_claims,
                  context_provider=context_provider,
                  context_id_provider=getattr(sc, "profile_context_id", None))


# MCP wiring

_JOB_VERBS = frozenset({"commit", "verify", "certificate", "cyber_register_candidate",
                        "cyber_evaluate_candidate", "cyber_finalize_candidate"})
_JOB_CONTRACT = (
    "Cyber transport overrides blocking-return semantics: pending=true with job_id "
    "means adjudication is still running, not success or failure. Query "
    "verification_job(job_id) to recover the same result; identical requests "
    "reconnect without starting another adjudication. Disconnecting a waiter "
    "does not cancel the worker.")

class _ToolDispatcher:
    """Keep synchronous verifier work off the MCP transport's event loop.

    Cyber transport diagnostics are metadata only, outside the evidence log.
    Cancelling a request does not stop its worker or manufacture a verdict.
    """

    def __init__(self, service):
        self.service = service
        self._diagnostic_lock = threading.Lock()
        self._diagnostic_path = None
        if service.profile.name == "cyber":
            state_dir = getattr(service, "state_dir", None)
            if isinstance(state_dir, (str, os.PathLike)):
                self._diagnostic_path = os.path.join(state_dir, "transport_calls.jsonl")

    def _record(self, call_id, tool, phase, started, error_type=None):
        if self._diagnostic_path is None:
            return
        record = {"schema": 1, "call_id": call_id, "tool": tool,
                  "phase": phase, "ts": time.time(),
                  "elapsed_seconds": max(0.0, time.monotonic() - started)}
        if error_type is not None:
            record["error_type"] = error_type
        # Never follow a solver-created symlink or block on a special file.
        # Failure to retain diagnostics cannot affect verification.
        flags = (os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NONBLOCK
                 | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
        try:
            with self._diagnostic_lock:
                fd = os.open(self._diagnostic_path, flags, 0o600)
                try:
                    info = os.fstat(fd)
                    if (not stat.S_ISREG(info.st_mode)
                            or info.st_uid != os.geteuid() or info.st_nlink != 1):
                        return
                    encoded = (json.dumps(record, ensure_ascii=True) + "\n").encode("utf-8")
                    while encoded:
                        written = os.write(fd, encoded)
                        if written == 0:
                            break
                        encoded = encoded[written:]
                finally:
                    os.close(fd)
        except OSError:
            pass

    async def call(self, tool, *args):
        call_id, started = uuid.uuid4().hex, time.monotonic()
        # Telemetry is filesystem I/O too; it must not block transport health.
        if self._diagnostic_path is not None:
            await asyncio.to_thread(self._record, call_id, tool, "received", started)

        jobs = self.service.__dict__.get("_jobs")
        if jobs is not None and tool in _JOB_VERBS:
            # Retain before returning a handle. The background adjudication is
            # independent of this request Task and any solver code-mode cell.
            frozen_args = json.loads(canonical(list(args)))

            def submit():
                try:
                    request = self.service._job_request(tool, frozen_args)
                    operation = self.service._job_operation(request)

                    def invoke_job(retry):
                        self._record(call_id, tool, "invoked", started)
                        try:
                            result = operation(retry)
                        except BaseException as exc:
                            self._record(call_id, tool, "failed", started, type(exc).__name__)
                            raise
                        self._record(call_id, tool, "finished", started)
                        return result

                    handle = jobs.submit(request, invoke_job)
                    retained = jobs.wait(handle["job_id"], 0.1)
                    self._record(call_id, tool, "responded", started)
                    if retained.get("state") == "completed":
                        return {**retained["result"], "job_id": retained["job_id"],
                                "job_state": "completed", "pending": False,
                                "result_is_retained": True}
                    return {**retained, "job_state": retained.get("state")}
                except VERB_ERRORS + (OSError, KeyError) as exc:
                    self._record(call_id, tool, "failed", started, type(exc).__name__)
                    return {"error": str(exc), "verified": False,
                            "materialized": False, "delivery_pending": True}

            return await asyncio.to_thread(submit)

        def invoke():
            self._record(call_id, tool, "invoked", started)
            try:
                result = getattr(self.service, tool)(*args)
            except BaseException as exc:
                self._record(call_id, tool, "failed", started, type(exc).__name__)
                raise
            self._record(call_id, tool, "finished", started)
            return result

        return await asyncio.to_thread(invoke)


_PROTOCOL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "protocol", "solver-protocol.md")


def _solver_protocol(profile=None) -> str:
    """The solver-side protocol, served as MCP server instructions: every
    client that mounts gv gets it injected into the solver's context, so
    there is no CLAUDE.md copy to keep in sync. Advisory only — soundness
    does not depend on the solver following it; ignoring it forfeits the
    certificate, never corrupts one."""
    with open(_PROTOCOL_PATH, encoding="utf-8") as f:
        protocol = f.read()
    if profile is not None and profile.solver_instructions:
        protocol += "\n\n" + profile.solver_instructions
    if profile is not None and profile.name == "cyber":
        # Operational protocol only, not a task-solving prompt or a soundness
        # assumption. This overrides synchronous-transport wording above.
        protocol += (
            "\n\nCyber transport: commit, verify, certificate, candidate registration, "
            "evaluation and finalization may return pending=true with a stable job_id. "
            "That response is not a verdict. Use verification_job(job_id) to recover "
            "the existing result; verification_jobs lists outstanding work. A lost "
            "waiter does not cancel the worker. Repeated identical requests reconnect. "
            "Only verification_retry explicitly retries terminal non-verified work. "
            "Retained job results describe completion time, not current delivery or "
            "an official score. The controller authenticates current delivery separately.")
    return protocol


def build_server(service: Service):
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("gv", instructions=_solver_protocol(service.profile))
    dispatcher = _ToolDispatcher(service)

    def recoverable_contract(function):
        if service.profile.name == "cyber":
            function.__doc__ = inspect.cleandoc(function.__doc__ or "") + "\n\n" + _JOB_CONTRACT
        return function

    @server.tool()
    @recoverable_contract
    async def commit(label: str, kind: CLAIM_KINDS, text: str, deps: List[str],
                     revises: str = "") -> str:
        """Commit a claim (kind: assertion|effect|final). deps is a list of
        labels of already-committed claims (including effect:<seq>). The claim
        is VERIFIED BEFORE THIS RETURNS, so the call blocks for as long as its
        checks take; the result carries the resulting status, and on anything
        but `verified` the obligation that failed plus its counterexample."""
        return json.dumps(
            await dispatcher.call("commit", label, kind, text, deps,
                                  revises or None),
            ensure_ascii=False)

    @server.tool()
    @recoverable_contract
    async def verify(label: str) -> str:
        """Verify a committed claim now; returns its status."""
        return json.dumps(await dispatcher.call("verify", label), ensure_ascii=False)

    @server.tool()
    @recoverable_contract
    async def certificate(final_label: str) -> str:
        """Certify a final claim: settle the dependency closure, freeze the
        certificate, and return the verdict plus a slice summary."""
        return json.dumps(await dispatcher.call("certificate", final_label), ensure_ascii=False)

    @server.tool()
    async def status(label: str = "") -> str:
        """Query the status of one label or of all labels."""
        return json.dumps(await dispatcher.call("status", label or None), ensure_ascii=False)

    if service.profile.name == "cyber":
        @server.tool()
        async def cyber_checkpoint(working_theory: str, next_actions: List[str]) -> str:
            """Retain exploration notes in the ledger. These are not verified claims."""
            return json.dumps(await dispatcher.call("cyber_checkpoint", working_theory, next_actions), ensure_ascii=False)

        @server.tool()
        @recoverable_contract
        async def cyber_register_candidate(path: str, label: str = "") -> str:
            """Snapshot exact bytes from a staging file. Records identity, not correctness."""
            return json.dumps(await dispatcher.call("cyber_register_candidate", path, label), ensure_ascii=False)

        @server.tool()
        async def cyber_candidates() -> str:
            """List immutable candidate identities; this tool makes no ranking or verdict."""
            return json.dumps(await dispatcher.call("cyber_candidates"), ensure_ascii=False)

        if service.evaluator is not None:
            @server.tool()
            @recoverable_contract
            async def cyber_evaluate_candidate(candidate_id: str) -> str:
                """Run the configured task-authorized evaluator on the exact candidate.
                Returns retained observations and an ordinary Effect reference, not a certificate."""
                return json.dumps(await dispatcher.call("cyber_evaluate_candidate", candidate_id), ensure_ascii=False)

        @server.tool()
        @recoverable_contract
        async def cyber_finalize_candidate(candidate_id: str, destination: str = "") -> str:
            """Verify the candidate through the normal GV roles, then copy certified bytes.
            A verified candidate certificate is scoped to content; materialization has its own
            receipt. Neither the receipt nor a candidate signal asserts an official task score."""
            return json.dumps(await dispatcher.call("cyber_finalize_candidate", candidate_id, destination), ensure_ascii=False)

        @server.tool()
        async def verification_job(job_id: str) -> str:
            """Query an existing verification job without restarting it.
            A completed result is retained history; current delivery is authenticated separately."""
            return json.dumps(await dispatcher.call("verification_job", job_id), ensure_ascii=False)

        @server.tool()
        async def verification_jobs() -> str:
            """List recoverable verification jobs and their progress; starts no verification."""
            return json.dumps(await dispatcher.call("verification_jobs"), ensure_ascii=False)

        @server.tool()
        async def verification_retry(job_id: str) -> str:
            """Explicitly retry a terminal, non-verified job; pending jobs are only reconnected.
            Changed inputs require resubmission. Never retry merely to recover a disconnected waiter."""
            return json.dumps(await dispatcher.call("verification_retry", job_id), ensure_ascii=False)

    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--cyber", action="store_true", help="select the cybersecurity AgentRole suite")
    ap.add_argument("--seats", choices=("sdk", "codex"), default="sdk")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--service-tier", choices=("standard", "fast"), default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--cyber-final-destination", default=None)
    ap.add_argument("--cyber-evaluator-command-json", default=None,
                    help="host-configured argv JSON with exactly one {candidate} operand")
    ap.add_argument("--cyber-evaluator-environment", default="host")
    ap.add_argument("--cyber-evaluator-authority-scope", default="candidate_signal",
                    choices=("candidate_signal", "task_outcome"))
    ap.add_argument("--cyber-evaluator-timeout", type=float, default=None)
    ap.add_argument("--cyber-evaluator-pin", action="append", default=[],
                    help="explicit host program/dependency file to pin (repeatable)")
    ap.add_argument("--memory-dir", default=None)
    ap.add_argument("--toolchain-dir", default=None)
    ap.add_argument("--trigger", default=TRIGGER_ON_COMMIT, choices=list(TRIGGERS),
                    help="when a claim is verified (default: at commit)")
    args = ap.parse_args()
    if args.seats != "codex" and (args.effort is not None or args.service_tier is not None):
        ap.error("--effort and --service-tier require --seats codex")
    if not args.cyber and (args.cyber_evaluator_command_json is not None
                          or args.cyber_final_destination is not None):
        ap.error("cyber evaluator/destination configuration requires --cyber")
    evaluator_spec = None
    if args.cyber_evaluator_command_json is not None:
        from .profiles.cyber.evaluator import EvaluatorSpec
        try:
            argv = json.loads(args.cyber_evaluator_command_json)
            if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
                raise ValueError("evaluator command must be a JSON argv array")
            evaluator_spec = EvaluatorSpec(tuple(argv), args.cyber_evaluator_environment,
                args.cyber_evaluator_authority_scope, args.cyber_evaluator_timeout,
                tuple(args.cyber_evaluator_pin))
        except (OSError, ValueError) as err:
            ap.error(str(err))
    service = Service(args.state_dir, model=args.model,
                      memory_dir=args.memory_dir,
                      toolchain_dir=args.toolchain_dir,
                      trigger=args.trigger, profile="cyber" if args.cyber else "default",
                      project_root=args.project_root, evaluator_spec=evaluator_spec,
                      final_destination=args.cyber_final_destination,
                      seats=args.seats, effort=args.effort, service_tier=args.service_tier)
    build_server(service).run()   # stdio


if __name__ == "__main__":
    main()
