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
import hashlib
import threading
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
                 service_tier: Optional[str] = None):
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

    # hook feed
    def _ingest_hook_event(self, raw_event: Dict[str, Any]) -> None:
        etype = raw_event.get("type")
        if etype not in HOOK_EVENT_TYPES:
            return  # unknown feed line: ignored, the feed is untrusted input
        with self._event_feed_lock:
            event = self.log.append(etype, raw_event.get("payload", {}))
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
        return {
            label: self.store.status(cid)
            for label, cid in self.builder.labels.items()
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

    def profile_context(self, role, claim_id):
        context = {"profile": self.profile.name, "profile_id": self.profile.id,
                   "certificate_scope": "task"}
        binding = self._cyber_bindings.get(claim_id)
        if binding:
            context.update({k: v for k, v in binding.items() if k != "observations"})
            if role not in ("obligations", "critic"):
                context["observations"] = binding["observations"]
        return canonical(context)

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
                        "input": {"candidate_id": candidate_id,
                                  "evaluator_id": self.evaluator.spec.id}}))
                try:
                    result = self.evaluator.run(candidate)
                except (OSError, ValueError) as err:
                    result = {"error": str(err), "candidate_id": candidate_id}
                with self._event_feed_lock:
                    event = self.log.append("tool_result", {
                        "tool": "cyber_evaluator", "tool_use_id": call_id,
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

    def cyber_finalize_candidate(self, candidate_id: str, destination: str = ""):
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
                        artifact = self._auth_artifact(prior["certificate_id"])
                        final_id = prior["final_claim_id"]
                        binding = self._cyber_bindings.get(final_id)
                        if (binding is None or artifact.get("final") != final_id
                                or artifact.get("verdict") != "verified"
                                or self.store.status(final_id) != "verified"
                                or artifact.get("profile_metadata") != self._certificate_metadata(final_id)):
                            raise ValueError("prior receipt has no matching verified candidate certificate")
                        with self.candidates.lease(candidate_id) as lease:
                            self._validate_binding(binding)
                            if binding["observations"] != self._observations(candidate_id):
                                raise ValueError("new candidate observations require fresh adjudication")
                            self.candidates.verify_delivery(prior["receipt"])
                            self.candidates.validate(lease)
                        return {"verdict": "verified", "certificate_id": prior["certificate_id"],
                                "final_claim_id": final_id, "profile_metadata": artifact["profile_metadata"],
                                "receipt": prior["receipt"], "receipt_event_seq": event["seq"],
                                "materialized": True, "delivery_pending": False, "official_score": None}
                # Reject impossible paths before spending any verifier calls.
                address = self.candidates.preflight_destination(address)
                with self.candidates.lease(candidate_id) as lease:
                    candidate = lease.candidate
                    binding = {"certificate_scope": "candidate", "profile_id": self.profile.id,
                        "task_sha256": self._task_digest(), "delivery_pending": True,
                        # A live world is not an immutable capture. Separate
                        # attempts cannot blindly inherit world conclusions.
                        "verification_epoch": uuid.uuid4().hex,
                        "subject": {k: candidate[k] for k in ("candidate_id", "blob_path", "sha256", "size")},
                        "observations": self._observations(candidate_id)}
                    binding["subject"]["address"] = address
                    binding["subject"]["address_source"] = ("host_required" if self.final_destination else "candidate_proposal")
                    binding["context_id"] = content_hash(binding)
                    label = "cyber-final:" + uuid.uuid4().hex
                    commit = self._commit(label, "final",
                        canonical({"schema": "gv-cyber-candidate-v1", "binding": binding}),
                        [o["effect_ref"] for o in binding["observations"]], binding=binding)
                    if "claim_id" not in commit or commit.get("error"):
                        return {**commit, "delivery_pending": True, "materialized": False}
                    cert = self.certificate(label)
                    cert["final_claim_id"] = commit["claim_id"]
                    if cert.get("verdict") != "verified":
                        return {**cert, "delivery_pending": True, "materialized": False}
                    artifact = self._auth_artifact(cert["certificate_id"])
                    if (self.store.status(commit["claim_id"]) != "verified"
                            or artifact.get("final") != commit["claim_id"] or artifact.get("verdict") != "verified"
                            or artifact.get("profile_metadata") != self._certificate_metadata(commit["claim_id"])):
                        raise ValueError("certificate does not authorize the exact candidate binding")
                    self._validate_binding(binding)
                    if binding["observations"] != self._observations(candidate_id):
                        raise ValueError("new candidate observations invalidate the prepared certificate")
                    self.candidates.validate(lease)
                    receipt = self.candidates.materialize(lease, address)
                # Lease exit validates once more before authenticating delivery.
                self.candidates.verify_delivery(receipt)
                event = self.log.append("cyber_materialized", {
                    "receipt": receipt, "candidate_id": candidate_id,
                    "certificate_id": cert["certificate_id"], "final_claim_id": commit["claim_id"]})
                return {**cert, "materialized": True, "delivery_pending": False,
                        "receipt": receipt, "receipt_event_seq": event["seq"],
                        "official_score": None}
        except (OSError, ValueError, KeyError) as err:
            if receipt is not None:
                self.candidates.rollback(receipt)
            return {**cert, "error": str(err), "delivery_pending": True, "materialized": False}

def _parallel_default(name: str, fallback: int) -> int:
    """Per-trial parallelism defaults: GV_PARALLEL_OBLIGATIONS (discharges
    concurrent within one claim) and GV_PARALLEL_CLAIMS (claims of a
    certificate slice verified concurrently; offline only)."""
    try:
        return max(1, int(os.environ.get(name, fallback)))
    except ValueError:
        return fallback


def _default_kernel(sc: Service, model, backend=None,
                    critic_rounds: Optional[int] = None,
                    parallel_obligations: Optional[int] = None,
                    parallel_claims: Optional[int] = None) -> Kernel:
    from dataclasses import replace

    from .llm import prompts
    from .llm.agents import (
        ALL_AGENT_ROLES, AgentBackend,
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
            backend = CodexBackend(roles=roles, cwd=sc.state_dir, model=model,
                effort=sc.effort or "medium", service_tier=sc.service_tier or "standard",
                **({"timeout": None} if profile.name == "cyber" else {}))
        else:
            backend = AgentBackend(roles=roles, cwd=sc.state_dir, model=model)
    else:
        # an injected backend (e.g. CodexBackend) still needs the
        # harness-swapped role registry — same double-registration rule
        backend.roles = roles
    if profile.disable_external_search and hasattr(backend, "search_roles"):
        backend.search_roles = frozenset()
    context_provider = getattr(sc, "profile_context", None)
    discharger = AgentDischarger(
        backend, sc.artifacts,
        role=roles["discharge"],
        workdir_base=os.path.join(sc.state_dir, "work"),
        cleanup=True,   # snapshots are the evidence; workdirs are scratch
        toolchain_dir=sc.toolchain_dir,
        shared_cache_dir=getattr(sc, "shared_cache_dir", None),
        state_dir=sc.state_dir,
        live_world=sc.live_world,
        store=sc.store, context_provider=context_provider)
    binder = BindingGame(backend, sc.artifacts, sc.log, discharger,
                         oracles_role=roles["oracles"], judger_role=roles["judger"],
                         memory=sc.memory, live_world=sc.live_world,
                         context_provider=context_provider)
    if parallel_obligations is None:
        parallel_obligations = getattr(sc, "parallel_obligations", None)
    if parallel_claims is None:
        parallel_claims = getattr(sc, "parallel_claims", None)
    if parallel_obligations is None:
        parallel_obligations = _parallel_default("GV_PARALLEL_OBLIGATIONS", 4)
    if parallel_claims is None:
        # online (live world): one claim at a time as commits arrive
        parallel_claims = (1 if sc.live_world
                           else _parallel_default("GV_PARALLEL_CLAIMS", 4))
    return Kernel(sc.store, sc.log, backend, sc.artifacts,
                  obligations_role=obligations,
                  critic_role=roles["critic"],
                  discharger=discharger, binder=binder,
                  memory=sc.memory, live_world=sc.live_world,
                  critic_rounds=critic_rounds,
                  parallel_obligations=parallel_obligations,
                  parallel_claims=parallel_claims,
                  context_provider=context_provider,
                  context_id_provider=getattr(sc, "profile_context_id", None))


# MCP wiring

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
        return protocol + "\n\n" + profile.solver_instructions
    return protocol


def build_server(service: Service):
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("gv", instructions=_solver_protocol(service.profile))

    @server.tool()
    def commit(label: str, kind: CLAIM_KINDS, text: str, deps: List[str],
               revises: str = "") -> str:
        """Commit a claim (kind: assertion|effect|final). deps is a list of
        labels of already-committed claims (including effect:<seq>). The claim
        is VERIFIED BEFORE THIS RETURNS, so the call blocks for as long as its
        checks take; the result carries the resulting status, and on anything
        but `verified` the obligation that failed plus its counterexample."""
        return json.dumps(
            service.commit(label, kind, text, deps,
                           revises or None),
            ensure_ascii=False)

    @server.tool()
    def verify(label: str) -> str:
        """Verify a committed claim now; returns its status."""
        return json.dumps(service.verify(label), ensure_ascii=False)

    @server.tool()
    def certificate(final_label: str) -> str:
        """Certify a final claim: settle the dependency closure, freeze the
        certificate, and return the verdict plus a slice summary."""
        return json.dumps(service.certificate(final_label), ensure_ascii=False)

    @server.tool()
    def status(label: str = "") -> str:
        """Query the status of one label or of all labels."""
        return json.dumps(service.status(label or None), ensure_ascii=False)

    if service.profile.name == "cyber":
        @server.tool()
        def cyber_checkpoint(working_theory: str, next_actions: List[str]) -> str:
            """Retain exploration notes in the ledger. These are not verified claims."""
            return json.dumps(service.cyber_checkpoint(working_theory, next_actions), ensure_ascii=False)

        @server.tool()
        def cyber_register_candidate(path: str, label: str = "") -> str:
            """Snapshot exact bytes from a staging file. Records identity, not correctness."""
            return json.dumps(service.cyber_register_candidate(path, label), ensure_ascii=False)

        @server.tool()
        def cyber_candidates() -> str:
            """List immutable candidate identities; this tool makes no ranking or verdict."""
            return json.dumps(service.cyber_candidates(), ensure_ascii=False)

        if service.evaluator is not None:
            @server.tool()
            def cyber_evaluate_candidate(candidate_id: str) -> str:
                """Run the configured task-authorized evaluator on the exact candidate.
                Returns retained observations and an ordinary Effect reference, not a certificate."""
                return json.dumps(service.cyber_evaluate_candidate(candidate_id), ensure_ascii=False)

        @server.tool()
        def cyber_finalize_candidate(candidate_id: str, destination: str = "") -> str:
            """Verify the candidate through the normal GV roles, then copy certified bytes.
            A verified candidate certificate is scoped to content; materialization has its own
            receipt. Neither the receipt nor a candidate signal asserts an official task score."""
            return json.dumps(service.cyber_finalize_candidate(candidate_id, destination), ensure_ascii=False)

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
