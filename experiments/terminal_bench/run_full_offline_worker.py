"""One fresh full-offline score invocation from an explicitly pinned source.

No contract replay, precompiled obligations or task-specific routing. This
evaluation worker imports the selected source's own offline.score, including
its extractor and full certificate machinery. Common observation/isolation
overlays are separate, operator-pinned files, not replacement backends.
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


SCHEMA = "gv-full-offline-worker-plan-v1"
RESULT_SCHEMA = "gv-full-offline-worker-result-v1"
RUNTIME = {"model": "gpt-5.6-luna", "effort": "medium", "service_tier": "standard"}
BASELINE = "be3581a9a5c61973fb2d98ed6c88625d95cda8dc"
SUPPORT_FILES = (
    "run_full_offline_worker.py", "observe_codex_attempts.py",
    "analyze_codex_observer.py", "eval_offline_policy.py",
    "run_typed_adapter_eval.py",
)
INPUT_PATHS = (
    "world", "world_manifest.json", "task.md", "baseline",
    "hook_feed.jsonl", "final.txt",
)
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_PLAN_KEYS = {
    "schema", "source_root", "source_commit", "source_fingerprint",
    "input_source", "input_seal", "output_root", "arm", "judger_prefetch",
    "runtime_read_roots", "support_sha256",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inside(parent, child):
    return os.path.commonpath((str(parent), str(child))) == str(parent)


def _path(value, *, existing=False):
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ValueError("worker paths must be explicit absolute strings")
    path = Path(value)
    if str(path.resolve(strict=existing)) != value:
        raise ValueError("worker paths must be canonical, without aliases")
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


def _read_json(path, cap=64 * 1024):
    with Path(path).open("rb") as handle:
        raw = handle.read(cap + 1)
    if len(raw) > cap:
        raise ValueError("JSON input exceeds worker size limit")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object,
                       parse_constant=_nonfinite)
    return value, hashlib.sha256(raw).hexdigest()


def validate_plan(plan):
    """No output creation or model work; reject unsupported evaluation scope."""
    if type(plan) is not dict or set(plan) != _PLAN_KEYS or plan["schema"] != SCHEMA:
        raise ValueError("wrong worker plan schema or fields")
    if plan["arm"] not in ("control", "treatment") or type(plan["judger_prefetch"]) is not bool:
        raise ValueError("invalid arm or prefetch profile")
    if (not isinstance(plan["source_commit"], str)
            or not _COMMIT.fullmatch(plan["source_commit"])):
        raise ValueError("source commit must be pinned")
    for key in ("source_fingerprint", "input_seal"):
        if not isinstance(plan[key], str) or not _HASH.fullmatch(plan[key]):
            raise ValueError("source fingerprint and input seal must be pinned")
    if plan["arm"] == "control" and (
            plan["source_commit"] != BASELINE or plan["judger_prefetch"]):
        raise ValueError("control must be the pinned original-main profile")
    if plan["arm"] == "treatment" and not plan["judger_prefetch"]:
        raise ValueError("this fresh-review treatment requires prefetch enabled")
    source = _path(plan["source_root"], existing=True)
    inputs = _path(plan["input_source"], existing=True)
    output = _path(plan["output_root"])
    if not source.is_dir() or not inputs.is_dir():
        raise ValueError("source and input roots must be directories")
    if (source / ".git").exists():
        raise ValueError("use a frozen source archive, not an active repository")
    if output.exists() or output.is_symlink():
        raise ValueError("worker output must be new; no resume or overwrite")
    for left, right in ((source, inputs), (source, output), (inputs, output)):
        if _inside(left, right) or _inside(right, left):
            raise ValueError("source, input and output roots must be disjoint")
    for root in (Path("/private/tmp"), Path("/private/var/folders")):
        if _inside(root, output):
            raise ValueError("paid worker output must not be in host temporary trees")
    roots = plan["runtime_read_roots"]
    if not isinstance(roots, list) or len(roots) > 16:
        raise ValueError("runtime roots must be an explicit bounded list")
    seen = set()
    for value in roots:
        root = _path(value, existing=True)
        if not root.is_dir() or value in seen or value == "/":
            raise ValueError("runtime roots must be distinct real directories")
        seen.add(value)
        for private in (Path.home().resolve(), source, inputs, output):
            if _inside(root, private) or _inside(private, root):
                raise ValueError("runtime roots overlap a private evaluation boundary")
    pins = plan["support_sha256"]
    if type(pins) is not dict or set(pins) != set(SUPPORT_FILES):
        raise ValueError("every common support file must be pinned")
    for name, digest in pins.items():
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise ValueError("invalid support source digest")
        if _sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("common evaluation support source changed")
    return plan


def _support(name):
    path = Path(__file__).with_name(name + ".py")
    module_name = "_gv_eval_support_" + name + "_" + _sha(path)[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _environment(plan):
    """Fix the evaluation schedule/profile, never backend retry semantics."""
    for name in list(os.environ):
        if name.startswith("GV_") or name in {
                "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}:
            del os.environ[name]
    treatment = plan["arm"] == "treatment"
    os.environ.update({
        "GV_TYPED_ADAPTERS": "1" if treatment else "0",
        # The treatment's fresh independent Judger may receive a verifier
        # built typed-adapter packet in a no-tools session; the control
        # source predates the flag and ignores it.
        "GV_TYPED_JUDGER_PACKET": "1" if treatment else "0",
        # Audit an agent method's sufficiency before executing it.
        "GV_SUFFICIENCY_PREAUDIT": "1" if treatment else "0",
        # Compile the candidate-blind final contract beside extraction.
        "GV_FINAL_CONTRACT_PRECOMPILE": "1" if treatment else "0",
        "GV_CONTRACT_CACHE": "0", "GV_APPEND_ONLY_DELTAS": "0",
        "GV_MAX_CONCURRENT_SEATS": "12",
        "GV_OFFLINE_RUNTIME_READ_ROOTS": json.dumps(plan["runtime_read_roots"]),
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "/private/tmp", "PYTHONDONTWRITEBYTECODE": "1",
        "TZ": "Europe/London",
    })
    if treatment:
        # The verdict is the final claim's own status; narration claims are
        # exported open rather than audited.  Part of the scoring identity.
        os.environ["GV_CERTIFICATE_SCOPE"] = "final"
    sys.dont_write_bytecode = True


def _load_runtime(source):
    if any(name == "verifier" or name.startswith("verifier.") for name in sys.modules):
        raise RuntimeError("worker requires a fresh process with no imported verifier")
    source = Path(source)
    if any((source / "verifier").rglob("*.pyc")):
        raise ValueError("frozen verifier source contains bytecode")
    if any(path.is_symlink() for path in (source / "verifier").rglob("*.py")):
        raise ValueError("frozen verifier source contains aliased Python files")
    sys.path.insert(0, str(source))
    offline = importlib.import_module("verifier.offline")
    backend = importlib.import_module("verifier.llm.codex_backend")
    for name, module in list(sys.modules.items()):
        if name == "verifier" or name.startswith("verifier."):
            location = getattr(module, "__file__", None)
            if not location or not _inside(source, Path(location).resolve()):
                raise RuntimeError("selected verifier source was shadowed by another version")
    if tuple(offline.IMMUTABLE_INPUT_PATHS) != INPUT_PATHS:
        raise ValueError("unsupported immutable input boundary")
    return offline, backend


def preflight(plan):
    """Fresh-process read-only source/import/input validation, without scoring."""
    validate_plan(plan)
    _environment(plan)
    source = Path(plan["source_root"])
    source_tools = _support("run_typed_adapter_eval")
    if source_tools._source_fingerprint(source) != plan["source_fingerprint"]:
        raise ValueError("frozen verifier fingerprint mismatch")
    # All support modules must remain independent of the selected verifier.
    for name in ("observe_codex_attempts", "analyze_codex_observer"):
        _support(name)
    policy = _support("eval_offline_policy")
    offline, backend = _load_runtime(source)
    if (hasattr(backend, "_run_codex_process") != (plan["arm"] == "treatment")):
        raise ValueError("source and observer seam do not match")
    if offline.verify_immutable_input_anchor(plan["input_source"]) != plan["input_seal"]:
        raise ValueError("operator input seal mismatch")
    policy_plan = policy.plan_eval_offline_policy(backend, bindings=(
        policy.OfflinePolicyBinding(
            state_dir=plan["input_source"],
            runtime_read_roots=tuple(plan["runtime_read_roots"])),))
    if not callable(getattr(policy.codex_class(backend), "_is_offline_execution", None)):
        raise ValueError("selected backend lacks the required offline classification hook")
    return {"schema": SCHEMA, "preflight_complete": True, "model_calls": 0,
            "source_fingerprint": plan["source_fingerprint"],
            "input_seal": plan["input_seal"], "runtime": RUNTIME,
            "read_only_policy_preflight_identity": policy_plan.identity}


def _clone(offline, source, destination, expected):
    """Retain only the existing immutable offline inputs; no previous judgments."""
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
            or offline.immutable_input_seal(str(destination)) != expected):
        raise ValueError("inputs changed while cloning")
    if offline.write_immutable_input_anchor(str(destination)) != expected:
        raise ValueError("clone input anchor mismatch")


def _write_new(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _certificate(state, features):
    if not isinstance(features, dict) or features.get("unscoreable") is not False:
        raise ValueError("full scoring did not produce a scoreable result")
    digest = features.get("certificate_id")
    if not isinstance(digest, str) or not _HASH.fullmatch(digest):
        raise ValueError("full scoring did not publish a certificate address")
    path = state / "artifacts" / (digest + ".json")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("certificate artifact is not a private regular file")
    cert, raw_hash = _read_json(path, cap=32 * 1024 * 1024)
    encoded = json.dumps(cert, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError("certificate content address mismatch")
    if (type(cert) is not dict or cert.get("verdict") != features.get("verdict")
            or cert.get("verdict_strength") != features.get("verdict_strength")
            or not isinstance(cert.get("slice"), list) or not cert["slice"]):
        raise ValueError("certificate and score disagree")
    finals = [row for row in cert["slice"] if isinstance(row, dict)
              and row.get("id") == cert.get("final")]
    if len(finals) != 1 or not finals[0].get("obligations"):
        raise ValueError("certificate lacks a nonempty final obligation set")
    return {"artifact": digest, "raw_sha256": raw_hash, "certificate": cert}


def _read_jsonl(path, cap=64 * 1024 * 1024):
    with Path(path).open("rb") as handle:
        raw = handle.read(cap + 1)
    if len(raw) > cap or not raw.endswith(b"\n"):
        raise ValueError("incomplete or oversized worker audit ledger")
    rows = [json.loads(line, object_pairs_hook=_strict_object,
                       parse_constant=_nonfinite)
            for line in raw.decode("utf-8").split("\n")[:-1]]
    if any(type(row) is not dict for row in rows):
        raise ValueError("worker audit ledger contains non-object rows")
    return rows


def _audit_role_and_cache_records(output, arm, expected_prefetch_protocol=None):
    roles = set()
    native_attempts = 0
    attached_attempts = 0
    typed_attempts = 0
    observed_attempts = []
    for row in _read_jsonl(output / "observer.jsonl"):
        if row.get("record_type") == "codex_logical_call_started":
            role = row.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError("malformed observed verifier role")
            roles.add(role)
        elif row.get("record_type") == "codex_cli_attempt_finished":
            observed_attempts.append(row)
    if not {"extract", "obligations", "critic", "oracles", "judger"} <= roles:
        raise ValueError("full fresh verifier role coverage was not observed")
    if arm == "treatment":
        if not isinstance(expected_prefetch_protocol, str) or not expected_prefetch_protocol:
            raise ValueError("treatment prefetch protocol was not pinned by selected source")
        native_rows = _read_jsonl(output / "native-usage.jsonl")
        for row in native_rows:
            if row.get("record_type") != "codex_cli_attempt":
                raise ValueError("treatment native usage has an unknown row type")
            native_attempts += 1
            if row.get("typed_judger_transport") is True:
                # A restricted no-tools Judger seat: the packet replaces the
                # prefetch channel, and only a Judger may use it.
                typed_digest = row.get("typed_judger_packet_sha256")
                if (row.get("role") != "judger"
                        or row.get("compact_inspection") is not False
                        or not isinstance(row.get("typed_judger_protocol"), str)
                        or not row["typed_judger_protocol"]
                        or not isinstance(typed_digest, str)
                        or not _HASH.fullmatch(typed_digest)
                        or row.get("judger_prefetch") is not False
                        or row.get("judger_prefetch_attached") is not False
                        or row.get("judger_prefetch_packet_sha256") is not None):
                    raise ValueError("treatment typed Judger profile mismatch")
                typed_attempts += 1
                continue
            if (row.get("judger_prefetch") is not True
                    or row.get("compact_inspection") is not False
                    or row.get("judger_prefetch_protocol") != expected_prefetch_protocol
                    or type(row.get("judger_prefetch_attached")) is not bool
                    or "judger_prefetch_packet_sha256" not in row):
                raise ValueError("treatment actual inspection profile mismatch")
            attached = row["judger_prefetch_attached"]
            digest = row["judger_prefetch_packet_sha256"]
            if attached:
                if (row.get("role") != "judger" or not isinstance(digest, str)
                        or not _HASH.fullmatch(digest)):
                    raise ValueError("invalid treatment prefetch packet identity")
                attached_attempts += 1
            elif digest is not None:
                raise ValueError("unattached treatment packet has a digest")
        if not native_attempts:
            raise ValueError("treatment actual profile was not recorded")
        native_threads = [row.get("thread_id") for row in native_rows]
        observed_threads = [row.get("thread_id") for row in observed_attempts]
        if (len(native_threads) != len(observed_threads)
                or any(not isinstance(item, str) or not item for item in native_threads)
                or any(not isinstance(item, str) or not item for item in observed_threads)
                or len(set(native_threads)) != len(native_threads)
                or set(native_threads) != set(observed_threads)):
            raise ValueError("treatment native profile and common attempt ledger disagree")
    event_counts = {}
    for event in _read_jsonl(output / "state" / "events.jsonl"):
        kind = event.get("type")
        if not isinstance(kind, str):
            raise ValueError("malformed retained verification event")
        event_counts[kind] = event_counts.get(kind, 0) + 1
    if any(event_counts.get(kind) for kind in (
            "method_cache_hit", "method_cache_direction_audit", "contract_cache_hit")):
        raise ValueError("fresh-review evaluation unexpectedly reused a signed method or contract")
    return {"roles_observed": sorted(roles), "event_counts": event_counts,
            "treatment_native_attempts": native_attempts if arm == "treatment" else None,
            "prefetch_attached_attempts": attached_attempts,
            "prefetch_exercised": attached_attempts > 0,
            "typed_judger_attempts": typed_attempts,
            "typed_judger_exercised": typed_attempts > 0,
            "expected_prefetch_protocol": expected_prefetch_protocol}


def run_worker(plan):
    """Execute exactly once. The caller owns explicit paid approval and isolation."""
    validate_plan(plan)
    started_unix = time.time_ns()
    started_mono = time.monotonic_ns()
    _environment(plan)
    source, inputs, output = (Path(plan[key]) for key in (
        "source_root", "input_source", "output_root"))
    source_tools = _support("run_typed_adapter_eval")
    if source_tools._source_fingerprint(source) != plan["source_fingerprint"]:
        raise ValueError("frozen verifier fingerprint mismatch")
    observer = _support("observe_codex_attempts")
    analyzer = _support("analyze_codex_observer")
    policy = _support("eval_offline_policy")
    offline, backend = _load_runtime(source)
    expected_prefetch_protocol = (
        importlib.import_module("verifier.engine.judger_prefetch").PREFETCH_PROTOCOL
        if plan["arm"] == "treatment" else None)
    seam = ("legacy-subprocess-run" if plan["arm"] == "control"
            else "run-codex-process")
    if (hasattr(backend, "_run_codex_process")
            != (seam == "run-codex-process")):
        raise ValueError("source and observer seam do not match")
    if offline.verify_immutable_input_anchor(str(inputs)) != plan["input_seal"]:
        raise ValueError("operator input seal mismatch before scoring")
    output.mkdir(parents=True, exist_ok=False)
    state = output / "state"
    worker_id = uuid.uuid4().hex
    contexts = {str(state): {"arm": plan["arm"], "worker_id": worker_id}}
    start = {
        "schema": RESULT_SCHEMA, "status": "started", "plan": plan,
        "runtime": RUNTIME, "worker_id": worker_id,
        "started_at_unix_ns": started_unix,
        "scope": "full-offline-score-fresh-cold-candidate",
        "certificate_scope": "final" if plan["arm"] == "treatment" else "full",
        "common_isolation_overlay": True, "pristine_baseline_execution": False,
        "contract_cache": False, "accepted_method_reuse": False,
        "automatic_outer_retries": 0,
    }
    _write_new(output / "worker-start.json", start)
    observation = overlay = features = certificate = accounting = role_audit = None
    failures = []
    caught = None
    try:
        _clone(offline, inputs, state, plan["input_seal"])
        binding = policy.OfflinePolicyBinding(
            state_dir=str(state), runtime_read_roots=tuple(plan["runtime_read_roots"]))
        with policy.eval_offline_policy(backend, bindings=(binding,)) as overlay:
            with observer.observe_codex_attempts(
                    backend, output / "observer.jsonl", seam=seam,
                    context_by_cwd=contexts) as observation:
                with backend.codex_usage_ledger(
                        str(output / "native-usage.jsonl"), context_by_cwd=contexts):
                    kwargs = dict(
                        **RUNTIME, seats="codex", max_rounds=3,
                        parallel_claims=4, parallel_obligations=4, shared_dir=None)
                    if plan["arm"] == "treatment":
                        # the treatment source's shipped schedule: eight
                        # obligation slots under the same 12-seat ceiling
                        kwargs.update(compact_inspection=False,
                                      judger_prefetch=plan["judger_prefetch"],
                                      parallel_obligations=8)
                    features = offline.score(str(state), **kwargs)
        certificate = _certificate(state, features)
        role_audit = _audit_role_and_cache_records(
            output, plan["arm"], expected_prefetch_protocol)
    except BaseException as error:
        caught = error
        failures.append(type(error).__name__)
    finally:
        try:
            if observation is not None:
                accounting = analyzer.analyze_file(
                    output / "observer.jsonl", context_by_cwd=contexts,
                    **RUNTIME, session_summary=observation.summary(),
                    # the candidate-blind final contract compiles in a
                    # verifier-owned temporary scratch, not the state
                    authority_scratch_prefix=os.path.join(
                        os.path.realpath(tempfile.gettempdir()),
                        "gv-authority-"),
                    typed_judger_scratch_prefix=os.path.join(
                        os.path.realpath(tempfile.gettempdir()),
                        "gv-typed-judger-"))
                if not accounting["accounting_complete"]:
                    failures.append("AccountingIncomplete")
            else:
                failures.append("ObserverUnavailable")
            if (offline.verify_immutable_input_anchor(str(inputs)) != plan["input_seal"]
                    or offline.verify_immutable_input_anchor(str(state)) != plan["input_seal"]):
                failures.append("InputSealChanged")
            if source_tools._source_fingerprint(source) != plan["source_fingerprint"]:
                failures.append("SourceChanged")
            if any(_sha(Path(__file__).with_name(name)) != digest
                   for name, digest in plan["support_sha256"].items()):
                failures.append("EvaluationSupportChanged")
        except BaseException as error:
            failures.append("Postflight" + type(error).__name__)
        result = {
            **start, "status": "terminal", "complete": not failures,
            "failures": failures, "features": features, "certificate": certificate,
            "accounting": accounting,
            "role_and_cache_audit": role_audit,
            "observer_summary": observation.summary() if observation is not None else None,
            "isolation_overlay_identity": overlay.identity if overlay is not None else None,
            "finished_at_unix_ns": time.time_ns(),
            "worker_wall_seconds": (time.monotonic_ns() - started_mono) / 1e9,
            "quality_acceptance": None, "original_project_10x_acceptance": False,
        }
        _write_new(output / "worker-result.json", result)
    if caught is not None:
        raise caught
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirm-paid")
    args = parser.parse_args()
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
