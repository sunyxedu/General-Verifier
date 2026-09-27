"""Run one cold, current-source, full-offline verifier arm.

Both arms execute the selected frozen source's own ``offline.score`` path.
The only arm differences are the source's general typed-adapter and Judger
transport feature flags.  Expected answers are not part of this worker's
schema and cannot enter any model call.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time
import uuid


SCHEMA = "gv-full-offline-worker-plan-v2"
RESULT_SCHEMA = "gv-full-offline-worker-result-v2"
RUNTIME = {
    "model": "gpt-5.6-luna", "effort": "medium",
    "service_tier": "standard",
}
INPUT_PATHS = (
    "world", "world_manifest.json", "task.md", "baseline",
    "hook_feed.jsonl", "final.txt",
)
SUPPORT_FILES = (
    "run_full_offline_worker_v2.py", "observe_codex_attempts.py",
    "analyze_codex_observer.py", "eval_offline_policy.py",
    "run_typed_adapter_eval.py",
)
_PLAN_KEYS = {
    "schema", "source_root", "source_commit", "source_fingerprint",
    "input_source", "input_seal", "output_root", "arm",
    "runtime_read_roots", "support_sha256",
}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_ATTEMPT = re.compile(r"[0-9a-f]{32}\Z")
_ROLES = frozenset({
    "extract", "obligations", "critic", "oracles", "discharge", "judger",
})
_COUNTERS = (
    "input_tokens", "cached_input_tokens", "uncached_input_tokens",
    "cache_write_input_tokens", "output_tokens", "reasoning_tokens",
)


def _sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _content_digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _inside(parent: Path, child: Path) -> bool:
    try:
        return os.path.commonpath((str(parent), str(child))) == str(parent)
    except ValueError:
        return False


def _path(value, *, existing=False) -> Path:
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ValueError("worker paths must be canonical absolute strings")
    path = Path(value)
    if str(path.resolve(strict=existing)) != value:
        raise ValueError("worker paths must contain no aliases")
    return path


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError("non-finite JSON constant")


def _read_json(path, cap=64 * 1024 * 1024):
    with Path(path).open("rb") as handle:
        raw = handle.read(cap + 1)
    if len(raw) > cap:
        raise ValueError("worker JSON exceeds size limit")
    return (json.loads(raw.decode("utf-8", errors="strict"),
                       object_pairs_hook=_strict_object,
                       parse_constant=_nonfinite),
            hashlib.sha256(raw).hexdigest())


def validate_plan(plan):
    if type(plan) is not dict or set(plan) != _PLAN_KEYS \
            or plan.get("schema") != SCHEMA:
        raise ValueError("wrong offline v2 worker plan schema or fields")
    if plan.get("arm") not in {"control", "treatment"}:
        raise ValueError("worker arm must be control or treatment")
    if not isinstance(plan.get("source_commit"), str) \
            or not _COMMIT.fullmatch(plan["source_commit"]):
        raise ValueError("source commit must be exact")
    for key in ("source_fingerprint", "input_seal"):
        if not isinstance(plan.get(key), str) or not _HASH.fullmatch(plan[key]):
            raise ValueError("source and input identities must be pinned")
    source = _path(plan["source_root"], existing=True)
    inputs = _path(plan["input_source"], existing=True)
    output = _path(plan["output_root"])
    if not source.is_dir() or not inputs.is_dir() or (source / ".git").exists():
        raise ValueError("source/input must be real dirs and source must be frozen")
    if output.exists() or output.is_symlink():
        raise ValueError("worker output must be fresh")
    for left, right in ((source, inputs), (source, output), (inputs, output)):
        if _inside(left, right) or _inside(right, left):
            raise ValueError("source, input, and output roots must be disjoint")
    for temporary in (Path("/private/tmp"), Path("/private/var/folders")):
        if _inside(temporary, output):
            raise ValueError("paid output must not use a host temporary tree")
    roots = plan.get("runtime_read_roots")
    if type(roots) is not list or len(roots) > 16:
        raise ValueError("runtime roots must be an explicit bounded list")
    seen = set()
    for value in roots:
        root = _path(value, existing=True)
        if not root.is_dir() or value in seen or value == "/":
            raise ValueError("runtime roots must be distinct real directories")
        seen.add(value)
        for private in (Path.home().resolve(), source, inputs, output):
            if _inside(root, private) or _inside(private, root):
                raise ValueError("runtime root overlaps a private boundary")
    pins = plan.get("support_sha256")
    if type(pins) is not dict or set(pins) != set(SUPPORT_FILES):
        raise ValueError("every offline v2 support file must be pinned")
    for name, digest in pins.items():
        if not isinstance(digest, str) or not _HASH.fullmatch(digest) \
                or _sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("offline v2 support source changed")
    return plan


def _support(name):
    path = Path(__file__).with_name(name + ".py")
    identity = _sha(path)
    spec = importlib.util.spec_from_file_location(
        f"_gv_offline_v2_{name}_{identity[:16]}", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned offline v2 support")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _environment(plan):
    treatment = plan["arm"] == "treatment"
    for name in list(os.environ):
        if name.startswith("GV_") or name in {
                "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}:
            del os.environ[name]
    os.environ.update({
        "GV_TYPED_ADAPTERS": "1" if treatment else "0",
        "GV_TYPED_JUDGER_PACKET": "1" if treatment else "0",
        "GV_CONTRACT_CACHE": "0", "GV_APPEND_ONLY_DELTAS": "0",
        "GV_MAX_CONCURRENT_SEATS": "12",
        "GV_OFFLINE_RUNTIME_READ_ROOTS": json.dumps(
            plan["runtime_read_roots"], separators=(",", ":")),
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "/private/tmp", "PYTHONDONTWRITEBYTECODE": "1",
        "TZ": "Europe/London",
    })
    os.environ.pop("GV_SHARED_DIR", None)
    sys.dont_write_bytecode = True



def _frozen_module(source: Path, current: str, legacy: str) -> str:
    """The module path of a runtime part in the frozen tree: its path
    before the refactor when only that file exists there, else its
    current path."""
    def present(name):
        return source.joinpath(*name.split(".")).with_suffix(".py").is_file()
    return legacy if present(legacy) and not present(current) else current

def _load_runtime(source):
    if any(name == "verifier" or name.startswith("verifier.")
           for name in sys.modules):
        raise RuntimeError("worker requires a fresh verifier import process")
    source = Path(source)
    if any((source / "verifier").rglob("*.pyc")) \
            or any(path.is_symlink()
                   for path in (source / "verifier").rglob("*.py")):
        raise ValueError("frozen source contains bytecode or aliased Python")
    sys.path.insert(0, str(source))
    modules = {
        "offline": importlib.import_module("verifier.offline"),
        "backend": importlib.import_module(_frozen_module(
            source, "verifier.llm.codex.runtime",
            "verifier.llm.codex_backend")),
        "typed_backend": importlib.import_module(
            "verifier.llm.typed_judger_backend"),
        "typed_packet": importlib.import_module(_frozen_module(
            source, "verifier.execution.judger_packets.typed",
            "verifier.engine.typed_judger_packet")),
    }
    for name, module in list(sys.modules.items()):
        if name == "verifier" or name.startswith("verifier."):
            location = getattr(module, "__file__", None)
            if not location or not _inside(source, Path(location).resolve()):
                raise RuntimeError("selected verifier source was shadowed")
    if tuple(modules["offline"].IMMUTABLE_INPUT_PATHS) != INPUT_PATHS:
        raise ValueError("source has an unsupported immutable input boundary")
    return modules


def preflight(plan):
    validate_plan(plan)
    _environment(plan)
    source = Path(plan["source_root"])
    source_tools = _support("run_typed_adapter_eval")
    if source_tools._source_fingerprint(source) != plan["source_fingerprint"]:
        raise ValueError("frozen source fingerprint mismatch")
    observer = _support("observe_codex_attempts")
    del observer
    _support("analyze_codex_observer")
    policy = _support("eval_offline_policy")
    modules = _load_runtime(source)
    backend = modules["backend"]
    offline = modules["offline"]
    if not hasattr(backend, "_run_codex_process"):
        raise ValueError("current source lacks the common observer seam")
    if offline.verify_immutable_input_anchor(plan["input_source"]) \
            != plan["input_seal"]:
        raise ValueError("operator input seal mismatch")
    policy_plan = policy.plan_eval_offline_policy(
        backend, bindings=(policy.OfflinePolicyBinding(
            state_dir=plan["input_source"],
            runtime_read_roots=tuple(plan["runtime_read_roots"])),))
    return {
        "schema": SCHEMA, "preflight_complete": True, "model_calls": 0,
        "source_fingerprint": plan["source_fingerprint"],
        "input_seal": plan["input_seal"], "runtime": RUNTIME,
        "cold_stage_cache_disabled": True,
        "warm_cache": {
            "supported": False,
            "reason": ("source has no complete Extractor-through-Judger "
                       "cross-run exact signed artifact receipt protocol"),
        },
        "online_policy_identity": policy_plan.identity,
    }


def _clone(offline, source: Path, destination: Path, expected: str):
    if offline.verify_immutable_input_anchor(str(source)) != expected:
        raise ValueError("source input seal mismatch")
    destination.mkdir(exist_ok=False)
    for name in INPUT_PATHS:
        original, target = source / name, destination / name
        if not original.exists() and not original.is_symlink():
            continue
        mode = original.lstat().st_mode
        if stat.S_ISLNK(mode):
            target.symlink_to(os.readlink(original))
        elif stat.S_ISDIR(mode):
            shutil.copytree(original, target, symlinks=True)
        elif stat.S_ISREG(mode):
            shutil.copy2(original, target)
        else:
            raise ValueError("unsupported immutable input type")
    if (offline.verify_immutable_input_anchor(str(source)) != expected
            or offline.immutable_input_seal(str(destination)) != expected
            or offline.write_immutable_input_anchor(str(destination)) != expected):
        raise ValueError("input clone does not match its immutable seal")


def _write_new(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path, cap=64 * 1024 * 1024):
    with Path(path).open("rb") as handle:
        raw = handle.read(cap + 1)
    if len(raw) > cap or not raw or not raw.endswith(b"\n"):
        raise ValueError("audit ledger is oversized or incomplete")
    rows = [json.loads(line, object_pairs_hook=_strict_object,
                       parse_constant=_nonfinite)
            for line in raw.decode("utf-8", errors="strict").split("\n")[:-1]]
    if any(type(row) is not dict for row in rows):
        raise ValueError("audit ledger must contain objects")
    return rows, hashlib.sha256(raw).hexdigest()


def _artifact(state: Path, digest: str) -> dict:
    if not isinstance(digest, str) or not _HASH.fullmatch(digest):
        raise ValueError("invalid artifact address")
    value, _raw = _read_json(state / "artifacts" / (digest + ".json"),
                             cap=32 * 1024 * 1024)
    if type(value) is not dict or _content_digest(value) != digest:
        raise ValueError("artifact content address mismatch")
    return value


def _certificate(state: Path, features: dict) -> dict:
    if type(features) is not dict or features.get("unscoreable") is not False:
        raise ValueError("full offline score is unscoreable")
    digest = features.get("certificate_id")
    certificate = _artifact(state, digest)
    if (certificate.get("verdict") != features.get("verdict")
            or certificate.get("verdict_strength")
            != features.get("verdict_strength")):
        raise ValueError("certificate and score disagree")
    return {"artifact": digest, "certificate": certificate}


def _truth_eligible(certificate) -> bool:
    return bool(type(certificate) is dict
                and type(certificate.get("certificate")) is dict
                and certificate["certificate"].get("verdict")
                in {"verified", "supported", "refuted"})


def _committed_claim_identities(events) -> dict[int, str]:
    """Replay durable labels exactly as production does and return commit IDs.

    Commit payload dependencies are labels, not content-addressed claim IDs;
    effect labels are also created by tool-result events.  Hashing the raw
    ``deps`` list therefore cannot authenticate a certificate.  A triggerless
    Builder replay applies the frozen source's real label resolution and Claim
    identity rules without invoking any verifier stage or model.
    """
    from verifier.core.store import ClaimStore
    from verifier.stream.builder import Builder, TraceError

    builder = Builder(None, ClaimStore())
    identities = {}
    try:
        for event in events:
            builder.feed(event)
            if event["type"] not in ("commit", "extracted_commit"):
                continue
            payload = event["payload"]
            label = payload.get("label")
            if not isinstance(label, str) or label not in builder.labels:
                raise ValueError("commit label is malformed")
            identities[event["seq"]] = builder.labels[label]
    except (KeyError, TypeError, ValueError, TraceError) as error:
        raise ValueError("commit graph replay failed") from error
    return identities


def _attempt_in_binding(binding, method, discharge, audit) -> bool:
    for attempt in binding.get("attempts", []):
        if type(attempt) is not dict or attempt.get("method") != method:
            continue
        if (attempt.get("discharge_artifact") == discharge
                and attempt.get("audit") == audit):
            return True
        if any(type(redo) is dict and redo.get("artifact") == discharge
               and redo.get("audit") == audit
               for redo in attempt.get("redos", [])):
            return True
    return False


def _audit_architecture(state: Path, certificate: dict,
                        certificate_id: str) -> dict:
    events, events_sha = _read_jsonl(state / "events.jsonl")
    for seq, event in enumerate(events, 1):
        if event.get("seq") != seq or not isinstance(event.get("type"), str) \
                or type(event.get("payload")) is not dict:
            raise ValueError("verification event sequence is malformed")
    final = certificate.get("final")
    entries = certificate.get("slice")
    finals = [row for row in entries if type(row) is dict
              and row.get("id") == final] if type(entries) is list else []
    if len(finals) != 1 or type(finals[0].get("obligations")) is not list \
            or not finals[0]["obligations"]:
        raise ValueError("certificate has no unique nonempty final contract")
    commit_identities = _committed_claim_identities(events)
    commits = [event for event in events
               if event["type"] == "extracted_commit"
               and commit_identities.get(event["seq"]) == final]
    if (len(commits) != 1 or commits[0]["payload"].get("kind") != "final"
            or finals[0].get("kind") != "final"
            or finals[0].get("text") != commits[0]["payload"].get("text")):
        raise ValueError("Extractor final commit is not certificate-bound")
    cert_events = [event for event in events if event["type"] == "certificate"]
    if (len(cert_events) != 1 or cert_events[0]["seq"] != len(events)
            or cert_events[0]["payload"].get("final") != final
            or cert_events[0]["payload"].get("digest")
            != certificate_id):
        raise ValueError("certificate is not the terminal bound event")
    compiled = [event for event in events
                if event["type"] == "obligations_compiled"
                and event["payload"].get("claim") == final]
    if not compiled:
        raise ValueError("final contract was not compiled")
    compiled_event = compiled[-1]
    critics = [event for event in events
               if event["type"] == "decomposition_attacked"
               and commits[0]["seq"] < event["seq"]
               and event["seq"] + 1 == compiled_event["seq"]
               and event["payload"].get("claim") == final
               and event["payload"].get("verdict") == "pass"]
    if len(critics) != 1:
        raise ValueError("final contract lacks an adjacent durable Critic pass")
    compiled_values = compiled_event["payload"].get("obligations")
    if type(compiled_values) is not list:
        raise ValueError("compiled final contract is malformed")
    compiled_by_id = {item.get("id"): item for item in compiled_values
                      if type(item) is dict and isinstance(item.get("id"), str)}
    final_values = finals[0]["obligations"]
    if len(compiled_by_id) != len(compiled_values) \
            or set(compiled_by_id) != {item.get("id") for item in final_values}:
        raise ValueError("compiled and certified obligations disagree")
    for item in compiled_values:
        expected = _content_digest({
            "schema": "gv-obligation-v2", "text": item.get("text"),
            "required": item.get("required_basis"),
            "refutable_by_absence": item.get("refutable_by_absence"),
            "premise_scope": item.get("premise_scope"),
        })
        if item.get("id") != expected:
            raise ValueError("compiled obligation identity is not payload-bound")
    lineages = []
    for obligation in final_values:
        oid = obligation.get("id")
        compiled_item = compiled_by_id[oid]
        if any(compiled_item.get(key) != obligation.get(key) for key in (
                "text", "severity", "clause", "refutable_by_absence")):
            raise ValueError("compiled obligation changed in the certificate")
        enumerations = [event for event in events
                        if event["type"] == "methods_enumerated"
                        and compiled_event["seq"] < event["seq"]
                        < cert_events[0]["seq"]
                        and event["payload"].get("claim") == final
                        and event["payload"].get("obligation") == oid]
        if not enumerations:
            raise ValueError("final obligation has no Oracle enumeration")
        settled = None
        for evidence in obligation.get("evidence", []):
            if type(evidence) is not dict:
                continue
            binding_digest = evidence.get("artifact")
            binding = _artifact(state, binding_digest)
            bound = binding.get("bound") if binding.get("instrument") \
                == "binding-game" else None
            if type(bound) is not dict:
                continue
            method, audit = bound.get("method"), bound.get("audit")
            discharge_digest = bound.get("discharge_artifact")
            if (type(method) is not dict or type(audit) is not dict
                    or audit.get("sufficient") != "yes"
                    or audit.get("faithful") != "yes"
                    or binding.get("obligation")
                    != compiled_by_id[oid].get("text")
                    or not _attempt_in_binding(
                        binding, method, discharge_digest, audit)):
                continue
            procedure = method.get("procedure", "")[:80]
            oracle = next((event for event in reversed(enumerations)
                           if any(type(item) is dict
                                  and item.get("procedure") == procedure
                                  for item in event["payload"].get("methods", []))),
                          None)
            if oracle is None:
                continue
            discharge = _artifact(state, discharge_digest)
            instrument = discharge.get("instrument")
            adapter = (instrument.removeprefix("adapter:")
                       if isinstance(instrument, str)
                       and instrument.startswith("adapter:") else None)
            if instrument != "agent:discharge" and adapter is None:
                continue
            executions = ([event for event in events
                           if event["type"] == "adapter_discharge"
                           and oracle["seq"] < event["seq"]
                           and event["payload"].get("artifact")
                           == discharge_digest]
                          if adapter is not None else [None])
            judges = [event for event in events
                      if event["type"] == "judger_verdict"
                      and oracle["seq"] < event["seq"]
                      and event["payload"].get("obligation") == oid
                      and event["payload"].get("procedure") == procedure
                      and all(event["payload"].get(key) == audit.get(key)
                              for key in ("sufficient", "faithful",
                                          "degradation", "reason"))]
            if not executions or not judges:
                continue
            judge = judges[-1]
            if executions[-1] is not None and executions[-1]["seq"] >= judge["seq"]:
                continue
            bounds = [event for event in events
                      if event["type"] == "method_bound"
                      and judge["seq"] < event["seq"]
                      and event["payload"].get("obligation") == oid
                      and event["payload"].get("procedure") == procedure]
            discharges = [event for event in events
                          if event["type"] == "discharge"
                          and bounds and bounds[-1]["seq"] < event["seq"]
                          < cert_events[0]["seq"]
                          and event["payload"].get("obligation") == oid
                          and event["payload"].get("artifact") == binding_digest
                          and event["payload"].get("evidence_id")
                          == evidence.get("evidence_id")]
            if bounds and discharges:
                settled = {
                    "obligation": oid, "oracle_seq": oracle["seq"],
                    "execution_kind": "adapter" if adapter else "agent",
                    "execution_artifact": discharge_digest,
                    "judger_seq": judge["seq"],
                    "method_bound_seq": bounds[-1]["seq"],
                    "kernel_discharge_seq": discharges[-1]["seq"],
                }
                break
        if obligation.get("status") in {"verified", "refuted"}:
            if settled is None:
                raise ValueError("settled obligation lacks complete fresh lineage")
            lineages.append(settled)
        else:
            lineages.append({
                "obligation": oid, "oracle_seqs": [row["seq"]
                                                     for row in enumerations],
                "unsettled_status": obligation.get("status"),
                "stages_skipped_reason": "no method survived fresh audit",
            })
    cache_hits = {kind: sum(event["type"] == kind for event in events)
                  for kind in ("method_cache_hit", "contract_cache_hit")}
    if any(cache_hits.values()):
        raise ValueError("cold worker reused a stage cache")
    return {
        "events_sha256": events_sha, "events_rows": len(events),
        "critic_pass_seq": critics[0]["seq"], "compiled_seq": compiled_event["seq"],
        "lineages": lineages, "cache_hits": cache_hits,
    }


def _transport_profile(modules, backend, arm):
    ordinary = getattr(backend, "runtime_instruction_identity", None)
    if not isinstance(ordinary, str) or not _HASH.fullmatch(ordinary):
        raise ValueError("ordinary runtime identity is unavailable")
    profile = {
        "ordinary": ordinary, "restricted": None, "typed_protocol": None,
        "raw_protocol": (getattr(backend, "_judger_prefetch_protocol", None)
                         if arm == "treatment" else None),
    }
    if arm == "treatment":
        with modules["typed_backend"].restricted_typed_judger_backend(
                backend) as restricted:
            profile["restricted"] = getattr(
                restricted, "runtime_instruction_identity", None)
        profile["typed_protocol"] = modules[
            "typed_packet"].TYPED_JUDGER_PACKET_PROTOCOL
        if (not isinstance(profile["restricted"], str)
                or not _HASH.fullmatch(profile["restricted"])
                or profile["restricted"] == ordinary
                or not isinstance(profile["raw_protocol"], str)):
            raise ValueError("treatment transport profile is incomplete")
    return profile


def _validated_attempt_rows(rows, label):
    """Validate one ledger's identities without equating its ID namespace."""
    result = list(rows)
    attempt_ids = set()
    logical_attempts = set()
    threads = set()
    for row in result:
        attempt_id = row.get("attempt_id")
        call_id = row.get("call_id")
        attempt_index = row.get("attempt_index")
        thread = row.get("thread_id")
        logical_attempt = (call_id, attempt_index)
        if (not isinstance(attempt_id, str)
                or not _ATTEMPT.fullmatch(attempt_id)
                or attempt_id in attempt_ids
                or not isinstance(call_id, str)
                or not _ATTEMPT.fullmatch(call_id)
                or type(attempt_index) is not int or attempt_index < 1
                or logical_attempt in logical_attempts
                or type(row.get("usage_complete")) is not bool
                or row.get("usage_status") not in {"complete", "partial"}
                or row["usage_complete"] is not (
                    row["usage_status"] == "complete")):
            raise ValueError(
                f"{label} attempt identities are invalid or duplicate")
        attempt_ids.add(attempt_id)
        logical_attempts.add(logical_attempt)
        if thread is None:
            if row["usage_complete"] is not False:
                raise ValueError(
                    f"{label} threadless attempt is not incomplete")
        elif (not isinstance(thread, str) or not thread or thread in threads):
            raise ValueError(
                f"{label} attempt thread identities are invalid or duplicate")
        else:
            threads.add(thread)
    return result


def _attempt_threads_agree(native, observed):
    """Bind same-position records, including a process killed after thread start."""
    native_thread = native.get("thread_id")
    observed_thread = observed.get("thread_id")
    if native_thread == observed_thread and isinstance(native_thread, str) \
            and native_thread:
        return True
    if native_thread is not None and observed_thread is not None:
        return False
    if (native.get("usage_complete") is not False
            or observed.get("usage_complete") is not False
            or native.get("usage_status") != "partial"
            or observed.get("usage_status") != "partial"
            or native.get("attempt_status") != "timeout"
            or observed.get("terminal_status") != "timeout"):
        return False
    # The native parser can observe ``thread.started`` even when the outer
    # observer kills the process before receiving a return object.  These two
    # ledgers use independent IDs, so bounded timing plus preserved order is
    # the common fail-closed identity for that interrupted attempt.
    for key in ("started_at_unix_ns", "finished_at_unix_ns", "duration_ns"):
        left, right = native.get(key), observed.get(key)
        if (type(left) is not int or type(right) is not int
                or abs(left - right) > 50_000_000):
            return False
    return True


def _audit_attempts(output: Path, state: Path, arm: str,
                    profile: dict) -> dict:
    observer, observer_sha = _read_jsonl(output / "observer.jsonl")
    roles, role_order, attempts = set(), [], []
    for row in observer:
        role = row.get("role")
        if not isinstance(role, str) or role not in _ROLES:
            raise ValueError("observer recorded an unknown offline role")
        if row.get("record_type") == "codex_logical_call_started":
            roles.add(role)
            role_order.append(role)
        elif row.get("record_type") == "codex_cli_attempt_finished":
            attempts.append(row)
    for left, right in zip(("extract", "obligations", "critic"),
                           ("obligations", "critic", "oracles")):
        if left not in roles or right not in roles \
                or role_order.index(left) >= role_order.index(right):
            raise ValueError("offline roles began outside architecture order")
    native, native_sha = _read_jsonl(output / "native-usage.jsonl")
    restricted_attempts = raw_packets = 0
    for row in native:
        required = {
            "record_type", "backend", "model", "effort", "service_tier",
            "compact_inspection", "judger_prefetch",
            "judger_prefetch_protocol", "judger_prefetch_attached",
            "judger_prefetch_packet_sha256", "runtime_instruction_identity",
            "typed_judger_transport", "typed_judger_protocol",
            "typed_judger_packet_sha256", "attempt_id", "call_id",
            "attempt_index", "thread_id", "role", *_COUNTERS,
        }
        if not required <= set(row) or row["record_type"] != "codex_cli_attempt" \
                or row["backend"] != "codex-cli" \
                or any(row[key] != value for key, value in RUNTIME.items()) \
                or row["compact_inspection"] is not (arm == "treatment"):
            raise ValueError("native attempt runtime profile changed")
        typed = row["typed_judger_transport"]
        if type(typed) is not bool:
            raise ValueError("native typed transport flag is malformed")
        if typed:
            digest = row["typed_judger_packet_sha256"]
            if (arm != "treatment" or row["role"] != "judger"
                    or row["typed_judger_protocol"] != profile["typed_protocol"]
                    or not isinstance(digest, str) or not _HASH.fullmatch(digest)
                    or row["runtime_instruction_identity"] != profile["restricted"]
                    or row["judger_prefetch"] is not False
                    or row["judger_prefetch_protocol"] is not None
                    or row["judger_prefetch_attached"] is not False
                    or row["judger_prefetch_packet_sha256"] is not None):
                raise ValueError("restricted Judger telemetry is inconsistent")
            restricted_attempts += 1
        else:
            expected_prefetch = arm == "treatment"
            if (row["typed_judger_protocol"] is not None
                    or row["typed_judger_packet_sha256"] is not None
                    or row["runtime_instruction_identity"] != profile["ordinary"]
                    or row["judger_prefetch"] is not expected_prefetch
                    or row["judger_prefetch_protocol"]
                    != profile["raw_protocol"]
                    or type(row["judger_prefetch_attached"]) is not bool
                    or (row["judger_prefetch_attached"]
                        and (row["role"] != "judger"
                             or not isinstance(
                                 row["judger_prefetch_packet_sha256"], str)
                             or not _HASH.fullmatch(
                                 row["judger_prefetch_packet_sha256"])))
                    or (not row["judger_prefetch_attached"]
                        and row["judger_prefetch_packet_sha256"] is not None)):
                raise ValueError("ordinary attempt transport profile changed")
            raw_packets += int(row["judger_prefetch_attached"])
    native_rows = _validated_attempt_rows(native, "native")
    observed_rows = _validated_attempt_rows(attempts, "observer")
    if len(observed_rows) != len(native_rows):
        raise ValueError("observer/native attempt inventories disagree")
    for row, observed in zip(native_rows, observed_rows):
        fields = ("role", "model", "effort", "service_tier",
                  "attempt_index", "usage_complete", "usage_status",
                  "runtime_instruction_identity", "typed_judger_transport",
                  "typed_judger_protocol", "typed_judger_packet_sha256",
                  *_COUNTERS)
        expected_cwd = (
            str(output / "typed-judger-scratch")
            if row["typed_judger_transport"] else str(state))
        if any(type(row.get(key)) is not type(observed.get(key))
               or row.get(key) != observed.get(key) for key in fields) \
                or not _attempt_threads_agree(row, observed) \
                or observed.get("cwd") != expected_cwd \
                or any(observed.get("command_" + key) != row.get(key)
                       for key in ("model", "effort", "service_tier")):
            raise ValueError("observer/native attempt accounting disagrees")
    return {
        "roles_observed": sorted(roles), "observer_sha256": observer_sha,
        "native_sha256": native_sha, "attempts": len(native),
        "restricted_attempts": restricted_attempts,
        "raw_prefetch_packets": raw_packets,
    }


def run_worker(plan):
    validate_plan(plan)
    _environment(plan)
    source, inputs, output = (Path(plan[key]) for key in (
        "source_root", "input_source", "output_root"))
    source_tools = _support("run_typed_adapter_eval")
    if source_tools._source_fingerprint(source) != plan["source_fingerprint"]:
        raise ValueError("frozen source fingerprint mismatch")
    observer = _support("observe_codex_attempts")
    analyzer = _support("analyze_codex_observer")
    policy = _support("eval_offline_policy")
    modules = _load_runtime(source)
    offline, backend_module = modules["offline"], modules["backend"]
    if not hasattr(backend_module, "_run_codex_process"):
        raise ValueError("current source lacks the common observer seam")
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    state = output / "state"
    restricted_judger_cwd = None
    if plan["arm"] == "treatment":
        scratch = output / "typed-judger-scratch"
        os.mkdir(scratch, 0o500)
        restricted_judger_cwd = str(scratch)
    worker_id = uuid.uuid4().hex
    started_unix, started_mono = time.time_ns(), time.monotonic_ns()
    context = {str(state): {"arm": plan["arm"], "worker_id": worker_id,
                            "scope": "full-offline-v2-cold"}}
    start = {
        "schema": RESULT_SCHEMA, "status": "started", "plan": plan,
        "runtime": RUNTIME, "worker_id": worker_id,
        "started_at_unix_ns": started_unix,
        "scope": "current-source-full-offline-cold",
        "automatic_outer_retries": 0, "stage_cache": "disabled",
        "warm_cache": {"supported": False,
                       "reason": "no complete exact signed stage receipt protocol"},
    }
    _write_new(output / "worker-start.json", start)
    observation = overlay = accounting = features = certificate = None
    transport = None
    architecture = attempt_audit = None
    failures, caught = [], None
    try:
        _clone(offline, inputs, state, plan["input_seal"])
        backend = policy.codex_class(backend_module)(
            cwd=str(state), **RUNTIME,
            compact_inspection=plan["arm"] == "treatment",
            judger_prefetch=plan["arm"] == "treatment",
            # This instance only derives immutable transport identities.  It
            # never executes a Judger, so the live Kernel remains the sole
            # owner of the operator-bound Judger scratch.
            restricted_judger_cwd=None)
        transport = _transport_profile(modules, backend, plan["arm"])
        binding = policy.OfflinePolicyBinding(
            state_dir=str(state),
            runtime_read_roots=tuple(plan["runtime_read_roots"]))
        with policy.eval_offline_policy(
                backend_module, bindings=(binding,)) as overlay:
            with observer.observe_codex_attempts(
                    backend_module, output / "observer.jsonl",
                    seam=observer.RUN_CODEX_PROCESS_SEAM,
                    context_by_cwd=context) as observation:
                with backend_module.codex_usage_ledger(
                        str(output / "native-usage.jsonl"),
                        context_by_cwd=context):
                    features = offline.score(
                        str(state), **RUNTIME, seats="codex", max_rounds=3,
                        parallel_claims=4, parallel_obligations=4,
                        shared_dir=None,
                        compact_inspection=plan["arm"] == "treatment",
                        judger_prefetch=plan["arm"] == "treatment",
                        restricted_judger_cwd=restricted_judger_cwd)
        certificate = _certificate(state, features)
        architecture = _audit_architecture(
            state, certificate["certificate"], certificate["artifact"])
        attempt_audit = _audit_attempts(
            output, state, plan["arm"], transport)
        observed_roles = set(attempt_audit["roles_observed"])
        if (any("judger_seq" in row for row in architecture["lineages"])
                and "judger" not in observed_roles) \
                or (any(row.get("execution_kind") == "agent"
                        for row in architecture["lineages"])
                    and "discharge" not in observed_roles):
            raise ValueError("observer omitted an executed verifier stage")
    except BaseException as error:
        caught = error
        failures.append(type(error).__name__)
    finally:
        try:
            if observation is None:
                failures.append("ObserverUnavailable")
            else:
                accounting = analyzer.analyze_file(
                    output / "observer.jsonl", context_by_cwd=context,
                    **RUNTIME, session_summary=observation.summary(),
                    execution_cwds_by_state={str(state): (
                        (str(state), restricted_judger_cwd)
                        if restricted_judger_cwd is not None
                        else (str(state),))},
                    authority_scratch_prefix=os.path.join(
                        os.path.realpath(tempfile.gettempdir()),
                        "gv-authority-"))
                if accounting.get("accounting_complete") is not True:
                    failures.append("AccountingIncomplete")
            if (offline.verify_immutable_input_anchor(str(inputs))
                    != plan["input_seal"]
                    or offline.verify_immutable_input_anchor(str(state))
                    != plan["input_seal"]):
                failures.append("InputSealChanged")
            if source_tools._source_fingerprint(source) \
                    != plan["source_fingerprint"]:
                failures.append("SourceChanged")
            if any(_sha(Path(__file__).with_name(name)) != digest
                   for name, digest in plan["support_sha256"].items()):
                failures.append("EvaluationSupportChanged")
        except BaseException as error:
            failures.append("Postflight" + type(error).__name__)
        truth_eligible = _truth_eligible(certificate)
        result = {
            **start, "status": "terminal", "complete": not failures,
            "failures": list(dict.fromkeys(failures)), "features": features,
            "certificate": certificate, "architecture": architecture,
            "attempt_audit": attempt_audit, "accounting": accounting,
            "observer_summary": (observation.summary()
                                 if observation is not None else None),
            "isolation_overlay_identity": (overlay.identity
                                           if overlay is not None else None),
            "truth_eligible": truth_eligible,
            "finished_at_unix_ns": time.time_ns(),
            "worker_wall_seconds": (time.monotonic_ns() - started_mono) / 1e9,
            "quality_acceptance": None, "ten_x_acceptance": None,
        }
        _write_new(output / "worker-result.json", result)
    if caught is not None:
        raise caught
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirm-paid")
    args = parser.parse_args(argv)
    plan, digest = _read_json(args.plan)
    if digest != args.plan_sha256:
        parser.error("operator worker-plan digest mismatch")
    validate_plan(plan)
    if args.preflight:
        if args.execute:
            parser.error("preflight and execute are mutually exclusive")
        print(json.dumps(preflight(plan), sort_keys=True))
        return 0
    if not args.execute:
        print(json.dumps({"plan": plan, "model_calls": 0}, sort_keys=True))
        return 0
    if args.confirm_paid != "GPT-5.6-Luna":
        parser.error("execution requires --confirm-paid GPT-5.6-Luna")
    result = run_worker(plan)
    print(json.dumps({"complete": result["complete"],
                      "worker_wall_seconds": result["worker_wall_seconds"]}))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
