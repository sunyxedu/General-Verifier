"""Sealed v2 coordinator for full-online capability A/B measurements.

Expected truth is retained only here.  A worker receives only its v2 plan;
failed attempts remain in cost/latency totals and unknown usage never becomes
zero.  This module is evaluation-only and does not alter verifier stages.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid


SCHEMA = "gv-full-online-comparison-plan-v2"
RESULT_SCHEMA = "gv-full-online-comparison-result-v2"
TRUTH_SCHEMA = "gv-full-online-expected-truth-v2"
WORKER_SCHEMA = "gv-full-online-worker-plan-v2"
WORKER_RESULT_SCHEMA = "gv-full-online-worker-result-v2"
SEMANTIC_SCHEMA = "gv-online-semantic-evidence-v1"
TYPED_PACKET_PROTOCOL = (
    "gv-typed-adapter-judger-packet-v3;evidence-cap=16384;"
    "render-cap=131072;fresh-replay=before+after;transport=no-tools-v1"
)
TYPED_BOUNDARY = "gv-typed-judger-no-tools-v1"
RUNTIME = {"model": "gpt-5.6-luna", "effort": "medium",
           "service_tier": "standard"}
COST_RATES_USD_PER_MILLION = {
    "uncached_input_tokens": .20, "cached_input_tokens": .02,
    "cache_write_input_tokens": .25, "output_tokens": 1.20,
}
LONG_CONTEXT_INPUT_THRESHOLD = 272_000
LONG_CONTEXT_INPUT_MULTIPLIER = 2.0
LONG_CONTEXT_OUTPUT_MULTIPLIER = 1.5
_TOKENS = ("input_tokens", "uncached_input_tokens", "cached_input_tokens",
           "cache_write_input_tokens", "output_tokens", "reasoning_tokens")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PLAN_KEYS = {"schema", "comparison_id", "output_root", "worker_path",
              "worker_sha256", "coordinator_sha256", "source_identity",
              "runtime_identity", "pairs"}
_PAIR_KEYS = {"candidate_id", "truth_path", "truth_sha256", "jobs"}
_JOB_KEYS = {"arm", "plan_path", "plan_sha256"}
_TRUTH_KEYS = {"schema", "candidate_id", "input_seal", "task_family",
               "polarity", "typed_adapter_eligibility",
               "typed_packet_eligibility", "browser_required",
               "expected_final_status", "acceptable_verdicts",
               "minimum_evidence_count", "required_evidence",
               "acceptable_semantic_evidence"}
_SOURCE_KEYS = {"source_commit", "source_fingerprint"}
_RUNTIME_KEYS = {"codex_cli_path", "codex_cli_realpath",
                 "codex_cli_sha256", "codex_cli_version",
                 "runtime_read_root_fingerprints"}
_INTEGRITY_FAILURES = {"InputBytesChanged", "SourceChanged",
                       "EvaluationSupportChanged", "QualityProjectionChanged"}
_WORKER_PATH = "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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
        raise ValueError("JSON exceeds size limit")
    value = json.loads(raw.decode("utf-8", errors="strict"),
                       object_pairs_hook=_strict_object,
                       parse_constant=_nonfinite)
    return value, hashlib.sha256(raw).hexdigest()


def _canonical_json_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _canonical(value, *, existing=False):
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ValueError("paths must be canonical absolute strings")
    path = Path(value)
    if str(path.resolve(strict=existing)) != value:
        raise ValueError("paths must contain no aliases")
    return path


def _inside(parent, child):
    try:
        return os.path.commonpath((str(parent), str(child))) == str(parent)
    except ValueError:
        return False


def _load_worker(path, digest):
    path = _canonical(path, existing=True)
    if not path.is_file() or not _HASH.fullmatch(digest or "") \
            or _sha(path) != digest:
        raise ValueError("worker source changed")
    name = "_gv_online_worker_v2_" + digest[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    helpers = ("validate_plan", "input_snapshot", "_certificate_semantics",
               "_score_semantics", "_content_digest")
    if (getattr(module, "SCHEMA", None) != WORKER_SCHEMA
            or getattr(module, "RESULT_SCHEMA", None) != WORKER_RESULT_SCHEMA
            or getattr(module, "RUNTIME", None) != RUNTIME
            or any(not callable(getattr(module, item, None)) for item in helpers)):
        raise ValueError("worker does not provide the strict v2 contract")
    return module


def _bind_worker_before_import(plan, raw_pairs):
    """Bind executable coordinator code to the frozen source as data first."""
    identity = plan.get("source_identity")
    if (type(identity) is not dict or set(identity) != _SOURCE_KEYS
            or not _COMMIT.fullmatch(identity.get("source_commit", ""))
            or not _HASH.fullmatch(identity.get("source_fingerprint", ""))):
        raise ValueError("source identity is incomplete")
    try:
        first_job = raw_pairs[0]["jobs"][0]
        first_plan_path = _canonical(first_job["plan_path"], existing=True)
        worker_plan, worker_plan_sha = _read_json(first_plan_path, 256 * 1024)
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError("cannot bind worker before import") from error
    if (type(first_job) is not dict or set(first_job) != _JOB_KEYS
            or worker_plan_sha != first_job.get("plan_sha256")
            or type(worker_plan) is not dict):
        raise ValueError("first worker plan identity is invalid")
    source = _canonical(worker_plan.get("source_root"), existing=True)
    if not source.is_dir() or (source / ".git").exists():
        raise ValueError("first worker source is not a frozen directory")
    if (worker_plan.get("source_commit") != identity["source_commit"]
            or worker_plan.get("source_fingerprint")
            != identity["source_fingerprint"]):
        raise ValueError("first worker source identity differs")
    if _source_fingerprint(source) != identity["source_fingerprint"]:
        raise ValueError("source archive fingerprint changed")
    expected = (source / "experiments" / "terminal_bench"
                / "run_full_online_worker.py").resolve(strict=True)
    selected = _canonical(plan.get("worker_path"), existing=True)
    if selected != expected:
        raise ValueError("worker must be the canonical frozen-source worker")
    return selected


def _source_fingerprint(source_root):
    root = Path(source_root)
    paths = [root / "pyproject.toml", root / "uv.lock"]
    paths += sorted((root / "experiments" / "terminal_bench").glob("*.py"))
    paths += sorted((root / "experiments" / "common").rglob("*.py"))
    paths += sorted((root / "verifier").rglob("*.py"))
    paths += sorted((root / "protocol").rglob("*"))
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            rel, raw = path.relative_to(root).as_posix().encode(), path.read_bytes()
            digest.update(len(rel).to_bytes(8, "big") + rel)
            digest.update(len(raw).to_bytes(8, "big") + raw)
    return digest.hexdigest()


def _runtime_root_fingerprint(root, *, max_entries=100_000,
                              max_bytes=2 * 1024 * 1024 * 1024):
    root = Path(root)
    if not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise ValueError("runtime root must be canonical")
    digest, count, total = hashlib.sha256(b"gv-online-runtime-root-v1\0"), 0, 0
    for directory, names, files in os.walk(root, followlinks=False):
        names.sort(); files.sort()
        for name in [*names, *files]:
            path = Path(directory) / name
            info = path.lstat(); count += 1
            if count > max_entries:
                raise ValueError("runtime root has too many entries")
            rel = path.relative_to(root).as_posix().encode()
            digest.update(len(rel).to_bytes(8, "big") + rel)
            digest.update(stat.S_IFMT(info.st_mode).to_bytes(8, "big"))
            digest.update(stat.S_IMODE(info.st_mode).to_bytes(8, "big"))
            if stat.S_ISREG(info.st_mode):
                raw = path.read_bytes(); total += len(raw)
                if total > max_bytes or len(raw) != info.st_size:
                    raise ValueError("runtime root is too large or changed")
                digest.update(len(raw).to_bytes(8, "big") + raw)
            elif stat.S_ISLNK(info.st_mode):
                raw = os.readlink(path).encode()
                digest.update(len(raw).to_bytes(8, "big") + raw)
            elif not stat.S_ISDIR(info.st_mode):
                raise ValueError("unsupported runtime-root entry")
    return digest.hexdigest()


def _verify_source(plan, pairs):
    identity = plan.get("source_identity")
    if (type(identity) is not dict or set(identity) != _SOURCE_KEYS
            or not _COMMIT.fullmatch(identity.get("source_commit", ""))
            or not _HASH.fullmatch(identity.get("source_fingerprint", ""))):
        raise ValueError("source identity is incomplete")
    root = pairs[0]["jobs"][0]["worker_plan"]["source_root"]
    if _source_fingerprint(root) != identity["source_fingerprint"]:
        raise ValueError("source archive fingerprint changed")


def _verify_runtime(plan, pairs, *, version=False):
    value = plan.get("runtime_identity")
    if type(value) is not dict or set(value) != _RUNTIME_KEYS:
        raise ValueError("runtime identity is incomplete")
    invoked = Path(value["codex_cli_path"]); real = Path(value["codex_cli_realpath"])
    if (not invoked.exists() or invoked.resolve() != real or real.resolve() != real
            or not real.is_file() or _sha(real) != value["codex_cli_sha256"]
            or not isinstance(value["codex_cli_version"], str)
            or not value["codex_cli_version"]):
        raise ValueError("CLI executable identity changed")
    roots = value["runtime_read_root_fingerprints"]
    expected = pairs[0]["jobs"][0]["worker_plan"]["runtime_read_roots"]
    if type(roots) is not dict or set(roots) != set(expected):
        raise ValueError("runtime roots are incomplete")
    if any(_runtime_root_fingerprint(root) != digest
           for root, digest in roots.items()):
        raise ValueError("runtime root changed")
    if not any(_inside(Path(root), real) for root in roots):
        raise ValueError("CLI is outside attested runtime roots")
    if version:
        resolved = shutil.which("codex", path=_WORKER_PATH)
        if resolved is None or Path(resolved) != invoked:
            raise ValueError("PATH resolves another CLI")
        run = subprocess.run([str(invoked), "--version"], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, timeout=30, check=False)
        if (run.returncode or run.stdout.decode(errors="strict").strip()
                != value["codex_cli_version"]):
            raise ValueError("CLI version changed")


def _validate_truth(value, candidate, seal):
    if type(value) is not dict or set(value) != _TRUTH_KEYS \
            or value.get("schema") != TRUTH_SCHEMA:
        raise ValueError("wrong truth schema")
    if value["candidate_id"] != candidate or value["input_seal"] != seal:
        raise ValueError("truth does not bind candidate input")
    if not _ID.fullmatch(value.get("task_family", "")):
        raise ValueError("invalid task family")
    if value.get("polarity") not in {"positive", "negative"}:
        raise ValueError("invalid polarity")
    adapter, packet = (value.get("typed_adapter_eligibility"),
                       value.get("typed_packet_eligibility"))
    if adapter not in {"eligible", "ineligible"} \
            or packet not in {"eligible", "ineligible"} \
            or (packet == "eligible" and adapter != "eligible"):
        raise ValueError("invalid typed eligibility")
    if type(value.get("browser_required")) is not bool:
        raise ValueError("browser requirement must be explicit")
    if value["browser_required"]:
        raise ValueError("browser pins are unsupported")
    verdicts = value.get("acceptable_verdicts")
    if type(verdicts) is not list or not verdicts or len(verdicts) != len(set(verdicts)):
        raise ValueError("invalid acceptable verdicts")
    if value["polarity"] == "positive":
        if (value.get("expected_final_status") != "verified"
                or not set(verdicts) <= {"verified", "supported"}):
            raise ValueError("positive truth must be verified")
    elif value.get("expected_final_status") != "refuted" \
            or verdicts != ["refuted"]:
        raise ValueError("negative truth must be refuted")
    required = value.get("required_evidence")
    minimum = value.get("minimum_evidence_count")
    if (type(required) is not list or len(required) > 64
            or any(type(item) is not dict or not item for item in required)
            or type(minimum) is not int or not 1 <= minimum <= 512):
        raise ValueError("invalid evidence truth floor")
    acceptable_semantics = value.get("acceptable_semantic_evidence")
    if (type(acceptable_semantics) is not list
            or not 1 <= len(acceptable_semantics) <= 16):
        raise ValueError("complete acceptable semantic truth is required")
    identities = set()
    for semantic in acceptable_semantics:
        if (type(semantic) is not dict
                or set(semantic) != {"schema", "input_tree_seal", "verdict",
                                     "verdict_strength", "truth_eligible",
                                     "obligations"}
                or semantic.get("schema") != SEMANTIC_SCHEMA
                or semantic.get("input_tree_seal") != seal
                or semantic.get("verdict") not in verdicts
                or semantic.get("truth_eligible") is not True
                or type(semantic.get("obligations")) is not list):
            raise ValueError("acceptable semantic truth is malformed")
        digest = _canonical_json_sha(semantic)
        if digest in identities:
            raise ValueError("acceptable semantic truth is duplicated")
        identities.add(digest)
    _canonical_json_sha(required)
    return value


def _stratum(truth):
    row = {key: truth[key] for key in ("task_family", "polarity",
           "typed_adapter_eligibility", "typed_packet_eligibility")}
    return {"id": "|".join(row.values()), **row}


def validate_plan(plan, *, require_fresh_outputs=True):
    if type(plan) is not dict or set(plan) != _PLAN_KEYS \
            or plan.get("schema") != SCHEMA:
        raise ValueError("wrong comparison v2 plan")
    if not _ID.fullmatch(plan.get("comparison_id", "")):
        raise ValueError("invalid comparison id")
    output = _canonical(plan["output_root"])
    if require_fresh_outputs and (output.exists() or output.is_symlink()):
        raise ValueError("comparison output is not fresh")
    if any(_inside(root, output) for root in map(Path, (
            "/tmp", "/var/tmp", "/private/tmp", "/private/var/folders"))):
        raise ValueError("paid output must be persistent")
    if (_sha(__file__) != plan.get("coordinator_sha256")
            or not _HASH.fullmatch(plan.get("coordinator_sha256", ""))):
        raise ValueError("coordinator source changed")
    raw_pairs = plan.get("pairs")
    if type(raw_pairs) is not list or not 6 <= len(raw_pairs) <= 64 \
            or len(raw_pairs) % 2:
        raise ValueError("need 6-64 even candidate pairs")
    _bind_worker_before_import(plan, raw_pairs)
    worker = _load_worker(plan["worker_path"], plan["worker_sha256"])
    candidates, seals, outputs, truths = set(), set(), set(), set()
    first_arms, pairs, frozen = [], [], None
    for raw_pair in raw_pairs:
        if type(raw_pair) is not dict or set(raw_pair) != _PAIR_KEYS:
            raise ValueError("wrong pair schema")
        candidate = raw_pair.get("candidate_id")
        if not _ID.fullmatch(candidate or "") or candidate in candidates:
            raise ValueError("candidate ids must be unique")
        candidates.add(candidate)
        truth_path = _canonical(raw_pair["truth_path"], existing=True)
        if (not truth_path.is_file() or truth_path.is_symlink()
                or str(truth_path) in truths):
            raise ValueError("truth files must be unique regular files")
        truths.add(str(truth_path))
        truth, truth_digest = _read_json(truth_path, 256 * 1024)
        if (truth_digest != raw_pair.get("truth_sha256")
                or not _HASH.fullmatch(raw_pair.get("truth_sha256", ""))):
            raise ValueError("sealed truth digest mismatch")
        raw_jobs = raw_pair.get("jobs")
        if type(raw_jobs) is not list or len(raw_jobs) != 2:
            raise ValueError("each pair requires two arms")
        jobs, arms, identity, input_seal = [], [], None, None
        for raw_job in raw_jobs:
            if type(raw_job) is not dict or set(raw_job) != _JOB_KEYS:
                raise ValueError("wrong job schema")
            arm = raw_job.get("arm")
            if arm not in {"control", "treatment"} or arm in arms:
                raise ValueError("pair needs distinct control/treatment")
            path = _canonical(raw_job["plan_path"], existing=True)
            worker_plan, digest = _read_json(path, 256 * 1024)
            if digest != raw_job.get("plan_sha256"):
                raise ValueError("worker plan digest mismatch")
            worker.validate_plan(worker_plan)
            if worker_plan.get("schema") != WORKER_SCHEMA \
                    or worker_plan.get("arm") != arm:
                raise ValueError("worker plan is not v2 or has wrong arm")
            pair_identity = {key: value for key, value in worker_plan.items()
                             if key not in {"arm", "output_root"}}
            if identity is None:
                identity = pair_identity
            elif identity != pair_identity:
                raise ValueError("paired arms may differ only by flags/output")
            snapshot = worker.input_snapshot(worker_plan["input_source"])
            if snapshot.get("seal") != worker_plan.get("input_seal"):
                raise ValueError("input seal mismatch")
            if input_seal is None:
                input_seal = worker_plan["input_seal"]
            elif input_seal != worker_plan["input_seal"]:
                raise ValueError("paired arms use different inputs")
            current = {key: worker_plan[key] for key in (
                "source_root", "source_commit", "source_fingerprint",
                "runtime_read_roots", "support_sha256")}
            if frozen is None:
                frozen = current
            elif current != frozen:
                raise ValueError("jobs do not share current frozen source")
            worker_output = _canonical(worker_plan["output_root"])
            if str(worker_output) in outputs:
                raise ValueError("worker outputs must be unique")
            outputs.add(str(worker_output))
            if require_fresh_outputs and worker_output.exists():
                raise ValueError("worker output is not fresh")
            if _inside(output, worker_output) or _inside(worker_output, output):
                raise ValueError("coordinator and worker outputs overlap")
            visible = [Path(worker_plan[key]) for key in
                       ("source_root", "input_source", "output_root")]
            visible += list(map(Path, worker_plan["runtime_read_roots"]))
            if any(_inside(root, truth_path) or _inside(truth_path, root)
                   for root in visible):
                raise ValueError("truth overlaps a worker-visible root")
            arms.append(arm)
            jobs.append({**raw_job, "worker_plan": worker_plan})
        if input_seal in seals:
            raise ValueError("candidate input seals must be unique")
        seals.add(input_seal)
        truth = _validate_truth(truth, candidate, input_seal)
        first_arms.append(arms[0])
        pairs.append({"candidate_id": candidate, "truth_path": str(truth_path),
                      "truth_sha256": truth_digest, "truth": truth,
                      "stratum": _stratum(truth), "jobs": jobs})
    if first_arms.count("control") != first_arms.count("treatment"):
        raise ValueError("arm order must be counterbalanced")
    families = {pair["truth"]["task_family"] for pair in pairs}
    polarities = {pair["truth"]["polarity"] for pair in pairs}
    cells = {(pair["truth"]["typed_adapter_eligibility"],
              pair["truth"]["typed_packet_eligibility"]) for pair in pairs}
    required_cells = {("eligible", "eligible"), ("eligible", "ineligible"),
                      ("ineligible", "ineligible")}
    if len(families) < 3 or polarities != {"positive", "negative"} \
            or not required_cells <= cells:
        raise ValueError("required strata are incomplete")
    _verify_source(plan, pairs)
    expected_frozen = {"source_root": frozen["source_root"],
                       **plan["source_identity"],
                       "runtime_read_roots": frozen["runtime_read_roots"],
                       "support_sha256": frozen["support_sha256"]}
    if frozen != expected_frozen:
        raise ValueError("worker/coordinator source identity differs")
    _verify_runtime(plan, pairs)
    return worker, pairs


def _write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())


def _private_output(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                     "wb")


def _subprocess_env():
    env = dict(os.environ); env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(name, None)
    return env


def preflight(plan):
    worker, pairs = validate_plan(plan)
    _verify_runtime(plan, pairs, version=True)
    reports = []
    for pair in pairs:
        for job in pair["jobs"]:
            run = subprocess.run([
                sys.executable, plan["worker_path"], "--plan", job["plan_path"],
                "--plan-sha256", job["plan_sha256"], "--preflight"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
                timeout=300, env=_subprocess_env())
            if run.returncode:
                raise ValueError("worker preflight failed")
            report = json.loads(run.stdout.decode(errors="strict"),
                                object_pairs_hook=_strict_object,
                                parse_constant=_nonfinite)
            browser = report.get("browser_runtime_pins", {})
            wp = job["worker_plan"]
            if (report.get("schema") != WORKER_SCHEMA
                    or report.get("preflight_complete") is not True
                    or report.get("model_calls") != 0
                    or report.get("runtime") != RUNTIME
                    or report.get("source_fingerprint") != wp["source_fingerprint"]
                    or report.get("input_seal") != wp["input_seal"]
                    or browser.get("supported") is not False
                    or Path(wp["output_root"]).exists()):
                raise ValueError("worker preflight is not sealed/read-only")
            reports.append({"candidate_id": pair["candidate_id"],
                            "arm": job["arm"],
                            "truth_sha256": pair["truth_sha256"],
                            "stratum_id": pair["stratum"]["id"],
                            "report": report})
    validate_plan(plan)
    return {"schema": SCHEMA, "preflight_complete": True, "model_calls": 0,
            "worker_schema": WORKER_SCHEMA, "runtime": RUNTIME,
            "truth_transport": "coordinator-only-sealed-v2", "jobs": reports}


def _quality_view(worker, result):
    chain = result.get("chain", {}); quality = chain.get("quality_view", {})
    audit = result.get("execution_audit", {})
    if (set(quality) != {"projection", "ordered_contract", "ordered_outcomes",
                        "semantic_evidence", "certificate", "score"}
            or quality["certificate"] != worker._certificate_semantics(
                chain.get("certificate"))
            or quality["score"].get("features") != worker._score_semantics(
                chain.get("features"))):
        raise ValueError("quality identity differs")
    projection = quality["projection"]
    if (projection.get("schema") != "gv-full-offline-result-projection-v1"
            or not _HASH.fullmatch(projection.get("events_sha256", ""))
            or projection.get("events_sha256") != audit.get("events_sha256")
            or projection.get("events_rows") != audit.get("events_rows")):
        raise ValueError("projection identity differs")
    semantic = quality["semantic_evidence"]
    document = semantic.get("document", {})
    if (set(document) != {"schema", "input_tree_seal", "verdict",
                          "verdict_strength", "truth_eligible", "obligations"}
            or document.get("schema") != SEMANTIC_SCHEMA
            or type(document.get("truth_eligible")) is not bool
            or type(document.get("obligations")) is not list
            or semantic.get("content_sha256") != worker._content_digest(document)
            or result.get("truth_eligible") is not document["truth_eligible"]):
        raise ValueError("semantic evidence identity differs")
    certificate = chain.get("certificate", {})
    matches = [row for row in certificate.get("slice", [])
               if type(row) is dict and row.get("id") == certificate.get("final")]
    if len(matches) != 1:
        raise ValueError("final certificate is not unique")
    eligible = document.get("verdict") in {
        "verified", "supported", "refuted",
    }
    if document["truth_eligible"] is not eligible:
        raise ValueError("truth eligibility is not the terminal allowlist")
    _canonical_json_sha(document)
    return document, projection, matches[0].get("status")


def _lossless_contains(actual, required):
    """No-weaker JSON relation; no business key is stripped or renamed."""
    if type(required) is dict:
        return type(actual) is dict and all(
            key in actual and _lossless_contains(actual[key], value)
            for key, value in required.items())
    if type(required) is list:
        if type(actual) is not list or len(required) > len(actual):
            return False
        unused = set(range(len(actual)))
        for expected in required:
            found = next((i for i in unused
                          if _lossless_contains(actual[i], expected)), None)
            if found is None:
                return False
            unused.remove(found)
        return True
    return type(actual) is type(required) and actual == required


def _truth_evaluation(truth, semantic, final_status):
    findings = []
    if semantic.get("truth_eligible") is not True:
        findings.append("not_truth_eligible")
    if semantic.get("verdict") not in {"verified", "supported", "refuted"}:
        findings.append("nonterminal_or_unknown_verdict")
    if semantic.get("input_tree_seal") != truth["input_seal"]:
        findings.append("input_seal")
    if semantic.get("verdict") not in truth["acceptable_verdicts"]:
        findings.append("verdict")
    if final_status != truth["expected_final_status"]:
        findings.append("final_status")
    evidence = []
    for obligation in semantic.get("obligations", []):
        if type(obligation) is not dict or type(obligation.get("evidence")) is not list:
            findings.append("malformed_obligation")
        else:
            evidence += obligation["evidence"]
    shortfall = max(0, truth["minimum_evidence_count"] - len(evidence))
    missing = sum(not any(_lossless_contains(row, required) for row in evidence)
                  for required in truth["required_evidence"])
    if shortfall:
        findings.append("evidence_count")
    if missing:
        findings.append("business_evidence")
    semantic_identity = _canonical_json_sha(semantic)
    acceptable_identities = {
        _canonical_json_sha(item)
        for item in truth["acceptable_semantic_evidence"]
    }
    if semantic_identity not in acceptable_identities:
        findings.append("complete_semantic_evidence")
    # Distance is reported separately from correctness. Equal nonzero distance
    # can establish no-regression, never correctness.
    distance = len(findings) + shortfall + missing
    return {"absolute_truth_correct": not findings, "truth_distance": distance,
            "findings": findings, "evidence_count": len(evidence),
            "required_evidence_missing": missing,
            "lossless_business_keys": True,
            "semantic_evidence_sha256": _canonical_json_sha(semantic)}


def compare_pair(control, treatment, truth):
    left, right = control.get("truth_evaluation"), treatment.get("truth_evaluation")
    identities = all(type(job.get("semantic_evidence")) is dict
                     and job.get("semantic_evidence_sha256") ==
                     _canonical_json_sha(job["semantic_evidence"])
                     for job in (control, treatment))
    transport = treatment.get("typed_transport", {})
    adapter_required = truth["typed_adapter_eligibility"] == "eligible"
    packet_required = truth["typed_packet_eligibility"] == "eligible"
    adapter_ok = not adapter_required or treatment.get("typed_adapter_exercised") is True
    packet_hit = (type(transport.get("packet_attached_attempts")) is int
                  and transport["packet_attached_attempts"] > 0
                  and transport.get("restricted_attempted") is True
                  and transport.get("restricted_fallback_attempted") is False
                  and transport.get("packet_unavailable_fallback") is False)
    packet_ok = not packet_required or packet_hit
    complete = (control.get("worker_complete") is True
                and treatment.get("worker_complete") is True
                and control.get("accounting_complete") is True
                and treatment.get("accounting_complete") is True
                and control.get("transport_audit_complete") is True
                and treatment.get("transport_audit_complete") is True
                and type(left) is dict and type(right) is dict
                and identities and adapter_ok and packet_ok)
    left_correct = bool(complete and left["absolute_truth_correct"])
    right_correct = bool(complete and right["absolute_truth_correct"])
    no_regression = bool(complete
                         and treatment.get("truth_eligible") is True
                         and right["truth_distance"] <= left["truth_distance"])
    return {"complete": complete, "sealed_truth_no_regression": no_regression,
            "control_absolute_truth_correct": left_correct,
            "treatment_absolute_truth_correct": right_correct,
            "both_arms_absolute_truth_correct": left_correct and right_correct,
            "equal_error_is_not_correctness": True,
            "control_truth": left, "treatment_truth": right,
            "stochastic_contract_text_equality_required": False,
            "treatment_typed_adapter_required": adapter_required,
            "treatment_typed_adapter_exercised":
            treatment.get("typed_adapter_exercised") is True,
            "treatment_typed_packet_required": packet_required,
            "treatment_typed_packet_exercised": packet_hit}


def _observer_rows(job, cap=64 * 1024 * 1024):
    path = Path(job["worker_output_root"]) / "observer.jsonl"
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size > cap):
            raise ValueError("observer ledger is unsafe")
        raw = b""
        while len(raw) <= cap:
            block = os.read(descriptor, min(1024 * 1024, cap + 1 - len(raw)))
            if not block:
                break
            raw += block
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (len(raw) > cap
            or (before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError("observer ledger changed while it was read")
    complete_tail = bool(raw) and raw.endswith(b"\n")
    prefix_end = len(raw) if complete_tail else raw.rfind(b"\n") + 1
    complete_prefix = raw[:prefix_end]
    rows = [json.loads(line, object_pairs_hook=_strict_object,
                       parse_constant=_nonfinite)
            for line in complete_prefix.decode(errors="strict").splitlines()]
    if any(type(row) is not dict for row in rows):
        raise ValueError("observer ledger has non-object rows")
    return raw, rows, complete_tail


def _uint(value):
    return type(value) is int and value >= 0


def _attempt_price(row):
    long_context = row["input_tokens"] > LONG_CONTEXT_INPUT_THRESHOLD
    im = LONG_CONTEXT_INPUT_MULTIPLIER if long_context else 1.0
    om = LONG_CONTEXT_OUTPUT_MULTIPLIER if long_context else 1.0
    ordinary = row["uncached_input_tokens"] - row["cache_write_input_tokens"]
    value = im * (
        ordinary * COST_RATES_USD_PER_MILLION["uncached_input_tokens"]
        + row["cached_input_tokens"]
        * COST_RATES_USD_PER_MILLION["cached_input_tokens"]
        + row["cache_write_input_tokens"]
        * COST_RATES_USD_PER_MILLION["cache_write_input_tokens"])
    value += om * row["output_tokens"] \
        * COST_RATES_USD_PER_MILLION["output_tokens"]
    return value / 1_000_000, long_context


def _accounting_summary(jobs, *, expected_jobs=None):
    buckets = {name: {arm: {"known_sum": 0, "unknown_attempts": 0}
                      for arm in ("control", "treatment")}
               for name in _TOKENS}
    known_price = {"control": 0.0, "treatment": 0.0}
    long_context = {"control": 0, "treatment": 0}
    priced = {"control": 0, "treatment": 0}
    findings, complete = set(), bool(jobs)
    if expected_jobs is not None and len(jobs) != expected_jobs:
        findings.add("planned_job_count_incomplete"); complete = False
    if sum(job.get("arm") == "control" for job in jobs) != \
            sum(job.get("arm") == "treatment" for job in jobs):
        findings.add("arm_count_unbalanced"); complete = False
    for job in jobs:
        arm, accounting = job.get("arm"), job.get("accounting")
        if arm not in known_price:
            findings.add("job_arm_invalid"); complete = False
            continue
        if type(accounting) is not dict:
            findings.add("job_accounting_unavailable"); complete = False
            accounting = {}
        try:
            raw, rows, complete_tail = _observer_rows(job)
        except BaseException:
            findings.add("observer_ledger_unavailable"); complete = False
            continue
        if not complete_tail:
            findings.add("observer_ledger_torn_or_empty"); complete = False
        if (accounting.get("ledger_sha256") != hashlib.sha256(raw).hexdigest()
                or accounting.get("ledger_bytes") != len(raw)
                or accounting.get("expected_profile") != RUNTIME):
            findings.add("observer_accounting_identity_mismatch"); complete = False
        starts, finishes = {}, {}
        job_buckets = {name: {"known_sum": 0, "unknown_attempts": 0}
                       for name in _TOKENS}
        counts = {name: 0 for name in (
            "logical_calls_started", "logical_calls_finished",
            "attempts_started", "attempts_finished")}
        mapping = {
            "codex_logical_call_started": "logical_calls_started",
            "codex_logical_call_finished": "logical_calls_finished",
            "codex_cli_attempt_started": "attempts_started",
            "codex_cli_attempt_finished": "attempts_finished",
        }
        for row in rows:
            kind = row.get("record_type")
            if row.get("schema") != "gv-codex-attempt-observer-v1" \
                    or kind not in mapping:
                findings.add("observer_record_schema_mismatch"); complete = False
                continue
            counts[mapping[kind]] += 1
            if any(row.get(key) != value for key, value in RUNTIME.items()):
                findings.add("observer_profile_mismatch"); complete = False
            if kind.startswith("codex_cli_attempt"):
                attempt = row.get("attempt_id")
                target = starts if kind.endswith("_started") else finishes
                if not isinstance(attempt, str) or not attempt or attempt in target:
                    findings.add("observer_attempt_identity_invalid"); complete = False
                else:
                    target[attempt] = row
        missing = set(starts) - set(finishes)
        extra = set(finishes) - set(starts)
        if missing or extra or accounting.get("attempt_count") != len(starts):
            findings.add("observer_attempt_lifecycle_incomplete"); complete = False
        for name in _TOKENS:
            job_buckets[name]["unknown_attempts"] += len(missing) + len(extra)
        for attempt, row in finishes.items():
            start = starts.get(attempt)
            valid_values = (all(_uint(row.get(name)) for name in _TOKENS)
                and row.get("cached_input_tokens", 0) <= row.get("input_tokens", -1)
                and row.get("uncached_input_tokens") == row.get("input_tokens", -1)
                    - row.get("cached_input_tokens", 0)
                and row.get("cache_write_input_tokens", 0)
                    <= row.get("uncached_input_tokens", -1)
                and row.get("reasoning_tokens", 0) <= row.get("output_tokens", -1))
            if start is None or any(start.get(key) != row.get(key) for key in (
                    "call_id", "attempt_index", "command_sha256", "cwd")):
                findings.add("observer_attempt_identity_mismatch"); complete = False
            if valid_values:
                for name in _TOKENS:
                    job_buckets[name]["known_sum"] += row[name]
                price, long = _attempt_price(row)
                known_price[arm] += price; priced[arm] += 1
                long_context[arm] += int(long)
            else:
                for name in _TOKENS:
                    job_buckets[name]["unknown_attempts"] += 1
                findings.add("observer_attempt_usage_invalid"); complete = False
            if (row.get("usage_complete") is not True
                    or row.get("usage_status") != "complete"
                    or row.get("terminal_status") != "returned"
                    or row.get("process_started") is not True):
                findings.add("observer_attempt_usage_incomplete"); complete = False
        if accounting.get("accounting_complete") is not True \
                or job.get("accounting_complete", True) is not True:
            findings.add("job_accounting_incomplete"); complete = False
        if accounting.get("observed_event_counts") != counts:
            findings.add("observer_event_counts_mismatch"); complete = False
        for name in _TOKENS:
            buckets[name][arm]["known_sum"] += job_buckets[name]["known_sum"]
            buckets[name][arm]["unknown_attempts"] += \
                job_buckets[name]["unknown_attempts"]
        reported = accounting.get("tokens", {})
        if type(reported) is not dict:
            findings.add("observer_token_totals_missing"); complete = False
        elif not missing and not extra:
            for name in _TOKENS:
                value = reported.get(name)
                if (type(value) is not dict
                        or value.get("known_sum") != job_buckets[name]["known_sum"]
                        or value.get("unknown_attempts") !=
                        job_buckets[name]["unknown_attempts"]):
                    findings.add("observer_token_totals_mismatch"); complete = False
    ratio = (known_price["control"] / known_price["treatment"]
             if complete and known_price["treatment"] > 0 else None)
    return {
        "complete": complete, "findings": sorted(findings), "tokens": buckets,
        "price_equivalent_usd": known_price if complete else None,
        "known_minimum_price_equivalent_usd": known_price,
        "control_over_treatment": ratio,
        "priced_observer_attempts": priced,
        "long_context_attempts": long_context,
        "rates_usd_per_million": COST_RATES_USD_PER_MILLION,
        "cache_write_rate_replaces_uncached_rate": True,
        "invoice_claim": False,
    }


def _read_worker_result(job):
    result, digest = _read_json(
        Path(job["worker_plan"]["output_root"]) / "worker-result.json")
    if (type(result) is not dict or result.get("schema") != WORKER_RESULT_SCHEMA
            or result.get("status") != "terminal"
            or result.get("plan") != job["worker_plan"]):
        raise ValueError("worker result does not match its plan")
    return result, digest


def _postflight(plan, pair, job):
    if (_sha(__file__) != plan["coordinator_sha256"]
            or _sha(plan["worker_path"]) != plan["worker_sha256"]
            or _sha(job["plan_path"]) != job["plan_sha256"]
            or _sha(pair["truth_path"]) != pair["truth_sha256"]):
        raise ValueError("coordinator, worker, plan, or truth changed")


def _transport_audit(audit, arm):
    transport = audit.get("typed_judger_transport")
    attempts = audit.get("native_attempts")
    observed = audit.get("observer_finished_attempts")
    keys = {"enabled", "protocol", "boundary",
            "packet_eligible_bound_methods", "packet_attached_attempts",
            "restricted_attempted", "ordinary_judger_attempts",
            "raw_prefetch_attached_attempts", "restricted_fallback_attempted",
            "packet_unavailable_fallback"}
    counters = ("packet_eligible_bound_methods", "packet_attached_attempts",
                "ordinary_judger_attempts", "raw_prefetch_attached_attempts")
    if (type(transport) is not dict or set(transport) != keys
            or type(attempts) is not int or attempts <= 0
            or observed != attempts
            or any(type(transport.get(key)) is not int
                   or transport[key] < 0 for key in counters)
            or any(type(transport.get(key)) is not bool for key in (
                "enabled", "restricted_attempted",
                "restricted_fallback_attempted", "packet_unavailable_fallback"))
            or transport["packet_attached_attempts"] > attempts
            or transport["ordinary_judger_attempts"] > attempts
            or transport["restricted_attempted"] is not
               (transport["packet_attached_attempts"] > 0)):
        return False
    if arm == "control":
        return (transport["enabled"] is False
                and transport["protocol"] is None
                and transport["boundary"] is None
                and transport["packet_eligible_bound_methods"] == 0
                and transport["packet_attached_attempts"] == 0
                and transport["restricted_attempted"] is False
                and transport["raw_prefetch_attached_attempts"] == 0
                and transport["restricted_fallback_attempted"] is False
                and transport["packet_unavailable_fallback"] is False)
    return (arm == "treatment" and transport["enabled"] is True
            and transport["protocol"] == TYPED_PACKET_PROTOCOL
            and transport["boundary"] == TYPED_BOUNDARY
            and transport["restricted_fallback_attempted"] is False
            and transport["packet_unavailable_fallback"] is False)


def _run_job(plan, worker, output, pair, job, ordinal):
    stdout_path, stderr_path = (output / f"job-{ordinal:03d}.{kind}"
                                for kind in ("stdout", "stderr"))
    command = [sys.executable, plan["worker_path"], "--plan", job["plan_path"],
               "--plan-sha256", job["plan_sha256"], "--execute",
               "--confirm-paid", "GPT-5.6-Luna"]
    with _private_output(stdout_path) as stdout, _private_output(stderr_path) as stderr:
        started = time.monotonic_ns()
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   env=_subprocess_env(), start_new_session=True)
        try:
            returncode = process.wait(); finished = time.monotonic_ns()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=10)
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise
    base = {
        "candidate_id": pair["candidate_id"], "arm": job["arm"],
        "worker_output_root": job["worker_plan"]["output_root"],
        "returncode": returncode,
        "spawn_to_exit_wall_seconds": (finished - started) / 1e9,
        "stdout_sha256": _sha(stdout_path), "stderr_sha256": _sha(stderr_path),
    }
    try:
        result, digest = _read_worker_result(job); _postflight(plan, pair, job)
        semantic, projection, final_status = _quality_view(worker, result)
        truth_eval = _truth_evaluation(pair["truth"], semantic, final_status)
        accounting = result.get("accounting")
        audit = result.get("execution_audit", {})
        transport = audit.get("typed_judger_transport", {})
        transport_ok = _transport_audit(audit, job["arm"])
    except BaseException as error:
        # A verifier failure is still an observed trial. Preserve its complete
        # subprocess latency and any readable accounting ledger; never turn an
        # unknown attempt into zero or omit the precommitted candidate.
        accounting = None
        try:
            raw_result, _raw_digest = _read_json(
                Path(job["worker_plan"]["output_root"]) / "worker-result.json")
            if type(raw_result) is dict and raw_result.get("plan") == job["worker_plan"]:
                accounting = raw_result.get("accounting")
        except BaseException:
            pass
        summary = {**base, "worker_complete": False,
                   "accounting": accounting, "accounting_complete": False,
                   "transport_audit_complete": False,
                   "supervisor_failure": type(error).__name__}
        _write_new(output / f"job-{ordinal:03d}-result.json", summary)
        return summary
    summary = {
        **base,
        "worker_result_sha256": digest,
        "worker_complete": result.get("complete") is True and returncode == 0,
        "worker_failures": result.get("failures"), "accounting": accounting,
        "accounting_complete": type(accounting) is dict
            and accounting.get("accounting_complete") is True,
        "semantic_evidence": semantic,
        "semantic_evidence_sha256": _canonical_json_sha(semantic),
        "quality_view": {key: result["chain"]["quality_view"][key] for key in (
            "ordered_contract", "ordered_outcomes", "certificate", "score")},
        "quality_sha256": _canonical_json_sha({key: result["chain"]["quality_view"][key]
            for key in ("ordered_contract", "ordered_outcomes", "certificate", "score")}),
        "projection": projection, "truth_evaluation": truth_eval,
        "truth_eligible": semantic.get("truth_eligible") is True,
        "typed_adapter_exercised": audit.get("typed_adapter_exercised") is True,
        "typed_transport": transport,
        "transport_audit_complete": transport_ok,
    }
    _write_new(output / f"job-{ordinal:03d}-result.json", summary)
    return summary


def _latency(jobs, expected):
    totals = {"control": 0.0, "treatment": 0.0}
    complete = len(jobs) == expected
    for job in jobs:
        value, arm = job.get("spawn_to_exit_wall_seconds"), job.get("arm")
        if arm not in totals or type(value) is not float or value < 0:
            complete = False
        else:
            totals[arm] += value
    ratio = totals["control"] / totals["treatment"] \
        if complete and totals["treatment"] > 0 else None
    return {"complete": complete, "control_wall_seconds": totals["control"],
            "treatment_wall_seconds": totals["treatment"],
            "control_over_treatment": ratio,
            "boundary": "subprocess-spawn-to-exit"}


def _ten_x(no_loss, cost, latency):
    if no_loss is not True:
        return False
    if cost.get("complete") is not True or latency.get("complete") is not True:
        return None
    cr, lr = cost.get("control_over_treatment"), latency.get("control_over_treatment")
    return None if cr is None or lr is None else cr >= 10 and lr >= 10


def run_comparison(plan):
    preflight_report = preflight(plan)
    worker, pairs = validate_plan(plan)
    output = Path(plan["output_root"]); output.mkdir(parents=True, exist_ok=False)
    started, run_id = time.monotonic_ns(), uuid.uuid4().hex
    _write_new(output / "comparison-start.json", {
        "schema": RESULT_SCHEMA, "status": "started", "run_id": run_id,
        "plan": plan, "preflight": preflight_report, "runtime": RUNTIME,
        "automatic_outer_retries": 0})
    jobs, reports, ordinal = [], [], 0
    for pair in pairs:
        current = []
        for job in pair["jobs"]:
            ordinal += 1
            try:
                summary = _run_job(plan, worker, output, pair, job, ordinal)
            except BaseException as error:
                summary = {"candidate_id": pair["candidate_id"], "arm": job["arm"],
                           "worker_output_root": job["worker_plan"]["output_root"],
                           "worker_complete": False, "accounting_complete": False,
                           "supervisor_failure": type(error).__name__}
            jobs.append(summary); current.append(summary)
        by_arm = {row["arm"]: row for row in current}
        reports.append({"candidate_id": pair["candidate_id"],
                        "stratum": pair.get("stratum"),
                        **compare_pair(by_arm.get("control", {}),
                                       by_arm.get("treatment", {}), pair["truth"])})
    cost = _accounting_summary(jobs, expected_jobs=2 * len(pairs))
    latency = _latency(jobs, 2 * len(pairs))
    no_loss = all(row["sealed_truth_no_regression"] for row in reports) \
        if len(reports) == len(pairs) else False
    absolute = all(row["both_arms_absolute_truth_correct"] for row in reports) \
        if len(reports) == len(pairs) else False
    strata = []
    strata_ids = sorted({row.get("stratum", {}).get("id") for row in reports})
    for key in strata_ids:
        selected_pairs = [row for row in reports if row.get("stratum", {}).get("id") == key]
        ids = {row["candidate_id"] for row in selected_pairs}
        selected_jobs = [row for row in jobs if row.get("candidate_id") in ids]
        scost, slatency = (_accounting_summary(selected_jobs,
            expected_jobs=2 * len(ids)), _latency(selected_jobs, 2 * len(ids)))
        snoloss = all(row["sealed_truth_no_regression"] for row in selected_pairs)
        sabsolute = all(row["both_arms_absolute_truth_correct"] for row in selected_pairs)
        strata.append({"stratum_id": key, "candidate_ids": sorted(ids),
                       "sealed_truth_no_regression": snoloss,
                       "all_truth_correct": sabsolute, "cost": scost,
                       "latency": slatency,
                       "ten_x_no_regression": _ten_x(snoloss, scost, slatency),
                       "ten_x_all_truth_correct": _ten_x(sabsolute, scost, slatency)})
    acceptance = _ten_x(no_loss, cost, latency)
    strict = _ten_x(absolute, cost, latency)
    if acceptance is True and any(row["ten_x_no_regression"] is not True for row in strata):
        acceptance = False
    if strict is True and any(row["ten_x_all_truth_correct"] is not True for row in strata):
        strict = False
    result = {
        "schema": RESULT_SCHEMA, "status": "terminal", "run_id": run_id,
        "jobs": jobs, "pairs": reports, "strata": strata, "cost": cost,
        "latency": {**latency, "batch_makespan_seconds":
                    (time.monotonic_ns() - started) / 1e9},
        "sealed_truth_no_regression": no_loss,
        "sampled_exact_semantic_quality_no_loss": no_loss,
        "sampled_all_truth_correct": absolute,
        "sampled_candidate_10x_acceptance": acceptance,
        "sampled_all_truth_correct_10x_acceptance": strict,
        "selection_policy": "all-precommitted-candidates-and-strata",
        "general_no_loss_claim": False, "general_10x_claim": False,
    }
    _write_new(output / "comparison-result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid")
    args = parser.parse_args(argv)
    plan, digest = _read_json(args.plan)
    if digest != args.plan_sha256:
        parser.error("operator comparison-plan digest mismatch")
    if args.preflight and args.execute:
        parser.error("preflight and execute are mutually exclusive")
    if args.preflight:
        print(json.dumps(preflight(plan), sort_keys=True)); return 0
    if not args.execute:
        validate_plan(plan)
        print(json.dumps({"plan": plan, "model_calls": 0}, sort_keys=True)); return 0
    if args.confirm_paid != "GPT-5.6-Luna":
        parser.error("execution requires --confirm-paid GPT-5.6-Luna")
    result = run_comparison(plan)
    print(json.dumps({"sampled_candidate_10x_acceptance":
                      result["sampled_candidate_10x_acceptance"]}, sort_keys=True))
    return 0 if result["sampled_candidate_10x_acceptance"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
