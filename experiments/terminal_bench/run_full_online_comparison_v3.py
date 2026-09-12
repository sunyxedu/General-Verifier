"""Scalar-truth, fixed-cohort coordinator for full-online capability A/B.

The worker remains the v2 full-online worker: it executes the production live
Service pipeline in a fresh process, with both stage caches disabled and a
fresh independent Judger.  This coordinator changes only evaluation policy:
it consumes a precommitted general input package and its retained Harbor
scalar truth, never task-specific adapter/evidence annotations.

Capability observations are grouped only after every precommitted candidate
has run.  They cannot select candidates or change the quality denominator;
coverage of eligible/ineligible mechanisms is a separate prerequisite for the
10x acceptance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

try:
    from . import run_full_online_comparison as v2
except ImportError:  # direct script execution
    import run_full_online_comparison as v2


SCHEMA = "gv-full-online-comparison-plan-v3"
RESULT_SCHEMA = "gv-full-online-comparison-result-v3"
MODEL_INPUT_SCHEMA = "gv-general-offline-model-inputs-v1"
SCALAR_TRUTH_SCHEMA = "gv-general-offline-coordinator-truth-v1"
TRUTH_IDENTITY_SCHEMA = "gv-general-offline-official-truth-v1"
WORKER_SCHEMA = v2.WORKER_SCHEMA
WORKER_RESULT_SCHEMA = v2.WORKER_RESULT_SCHEMA
RUNTIME = v2.RUNTIME
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PLAN_KEYS = {
    "schema", "comparison_id", "output_root", "worker_path",
    "worker_sha256", "coordinator_sha256", "comparison_support_sha256",
    "source_identity", "runtime_identity", "cohort", "pairs",
}
_COHORT_KEYS = {
    "package_id", "model_input_manifest", "model_input_manifest_sha256",
    "scalar_truth_manifest", "scalar_truth_manifest_sha256",
    "operator_manifest", "operator_manifest_sha256",
}
_PAIR_KEYS = {"case_id", "candidate_id", "jobs"}
_JOB_KEYS = {"arm", "plan_path", "plan_sha256"}
_MODEL_KEYS = {"schema", "package_id", "cases"}
_MODEL_CASE_KEYS = {
    "case_id", "task_family", "candidate_id", "state", "input_seal",
}
_TRUTH_KEYS = {
    "schema", "package_id", "operator_manifest_sha256",
    "model_input_manifest_sha256", "model_input_root", "cases",
}
_TRUTH_CASE_KEYS = {
    "case_id", "task_family", "candidate_id", "input_seal", "label",
    "identity", "provenance",
}
_OPERATOR_KEYS = {
    "schema", "package_id", "model_input_root", "truth_root", "cases",
}
_OPERATOR_CASE_KEYS = {
    "case_id", "task_family", "candidate_id", "source_state", "input_seal",
    "truth",
}
_OPERATOR_TRUTH_KEYS = {"label", "identity", "provenance"}
_CACHE_KEYS = {
    "method_cache_hit", "method_cache_direction_audit",
    "contract_cache_hit", "contract_cache_published",
}


def _canonical_sha(value) -> str:
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def _truth_identity(label: float, provenance: dict) -> str:
    return _canonical_sha({"schema": TRUTH_IDENTITY_SCHEMA,
                           "label": label, "provenance": provenance})


def _json_path(value, path):
    if type(path) is not list or not path or len(path) > 32:
        raise ValueError("Harbor truth json_path is invalid")
    for item in path:
        if isinstance(item, str) and type(value) is dict and item in value:
            value = value[item]
        elif type(item) is int and item >= 0 and type(value) is list \
                and item < len(value):
            value = value[item]
        else:
            raise ValueError("Harbor truth json_path does not resolve")
    return value


def _official_truth(item):
    if type(item) is not dict or set(item) != _TRUTH_CASE_KEYS:
        raise ValueError("scalar truth case has the wrong fields")
    label = item.get("label")
    if isinstance(label, bool) or not isinstance(label, (int, float)) \
            or not math.isfinite(float(label)) or float(label) not in {0.0, 1.0}:
        raise ValueError("official Harbor truth must be binary scalar truth")
    label = float(label)
    provenance = item.get("provenance")
    if type(provenance) is not dict or provenance.get("kind") not in {
            "harbor_result_json", "harbor_reward_txt"}:
        raise ValueError("official Harbor provenance is unsupported")
    expected = ({"kind", "path", "sha256", "json_path"}
                if provenance["kind"] == "harbor_result_json"
                else {"kind", "path", "sha256"})
    if set(provenance) != expected or not _HASH.fullmatch(
            provenance.get("sha256", "")):
        raise ValueError("official Harbor provenance is incomplete")
    source = v2._canonical(provenance["path"], existing=True)
    if (not source.is_file() or source.is_symlink() or source.stat().st_nlink != 1
            or v2._sha(source) != provenance["sha256"]):
        raise ValueError("official Harbor provenance changed")
    if provenance["kind"] == "harbor_result_json":
        payload, _digest = v2._read_json(source)
        observed = _json_path(payload, provenance["json_path"])
    else:
        raw = source.read_bytes()
        if len(raw) > 256:
            raise ValueError("Harbor reward text is too large")
        try:
            observed = float(raw.decode("ascii").strip())
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("Harbor reward text is invalid") from error
    if isinstance(observed, bool) or not isinstance(observed, (int, float)) \
            or float(observed) != label:
        raise ValueError("Harbor provenance and scalar truth disagree")
    if v2._sha(source) != provenance["sha256"]:
        raise ValueError("Harbor provenance changed while validating truth")
    if item.get("identity") != _truth_identity(label, provenance):
        raise ValueError("scalar truth identity mismatch")
    return {**item, "label": label, "provenance_path": source}


def _load_cohort(plan):
    cohort = plan.get("cohort")
    if type(cohort) is not dict or set(cohort) != _COHORT_KEYS \
            or not _ID.fullmatch(cohort.get("package_id", "")):
        raise ValueError("comparison cohort is malformed")
    model_path = v2._canonical(cohort["model_input_manifest"], existing=True)
    truth_path = v2._canonical(cohort["scalar_truth_manifest"], existing=True)
    operator_path = v2._canonical(cohort["operator_manifest"], existing=True)
    if (not model_path.is_file() or model_path.stat().st_nlink != 1
            or not truth_path.is_file() or truth_path.stat().st_nlink != 1
            or not operator_path.is_file() or operator_path.stat().st_nlink != 1
            or v2._inside(model_path.parent, truth_path)
            or v2._inside(truth_path.parent, model_path)
            or any(v2._inside(root, operator_path)
                   for root in (model_path.parent, truth_path.parent))):
        raise ValueError(
            "model, scalar truth, and operator manifests must be private and disjoint")
    model, model_sha = v2._read_json(model_path)
    truth, truth_sha = v2._read_json(truth_path)
    operator, operator_sha = v2._read_json(operator_path, 4 * 1024 * 1024)
    if (model_sha != cohort.get("model_input_manifest_sha256")
            or truth_sha != cohort.get("scalar_truth_manifest_sha256")
            or operator_sha != cohort.get("operator_manifest_sha256")
            or not all(_HASH.fullmatch(value) for value in (
                model_sha, truth_sha, operator_sha))):
        raise ValueError("cohort manifest digest mismatch")
    if (type(model) is not dict or set(model) != _MODEL_KEYS
            or model.get("schema") != MODEL_INPUT_SCHEMA
            or model.get("package_id") != cohort["package_id"]):
        raise ValueError("wrong model input cohort")
    if (type(truth) is not dict or set(truth) != _TRUTH_KEYS
            or truth.get("schema") != SCALAR_TRUTH_SCHEMA
            or truth.get("package_id") != cohort["package_id"]
            or truth.get("operator_manifest_sha256") != operator_sha
            or truth.get("model_input_manifest_sha256") != model_sha
            or truth.get("model_input_root") != str(model_path.parent)):
        raise ValueError("wrong or unbound scalar truth cohort")
    if (type(operator) is not dict or set(operator) != _OPERATOR_KEYS
            or operator.get("schema") != "gv-general-offline-eval-operator-v1"
            or operator.get("package_id") != cohort["package_id"]
            or operator.get("model_input_root") != str(model_path.parent)
            or operator.get("truth_root") != str(truth_path.parent)
            or type(operator.get("cases")) is not list):
        raise ValueError("retained operator authority does not bind the cohort")
    models, truths = model.get("cases"), truth.get("cases")
    if (type(models) is not list or type(truths) is not list
            or not 6 <= len(models) <= 64 or len(models) % 2
            or len(models) != len(truths)):
        raise ValueError("cohort must contain 6-64 even paired candidates")
    truth_by_case = {}
    for raw in truths:
        item = _official_truth(raw)
        case_id = item.get("case_id")
        if not _ID.fullmatch(case_id or "") or case_id in truth_by_case:
            raise ValueError("scalar truth case ids must be unique")
        truth_by_case[case_id] = item
        if (v2._inside(model_path.parent, item["provenance_path"])
                or v2._inside(item["provenance_path"], model_path.parent)):
            raise ValueError("official truth provenance overlaps model inputs")
    rows, candidates, seals, states = [], set(), set(), set()
    for raw in models:
        if type(raw) is not dict or set(raw) != _MODEL_CASE_KEYS:
            raise ValueError("model input case has the wrong fields")
        case_id, candidate = raw.get("case_id"), raw.get("candidate_id")
        family, seal = raw.get("task_family"), raw.get("input_seal")
        if (not _ID.fullmatch(case_id or "") or not _ID.fullmatch(candidate or "")
                or not _ID.fullmatch(family or "") or not _HASH.fullmatch(seal or "")
                or candidate in candidates or seal in seals):
            raise ValueError("model cohort identities are invalid or duplicate")
        relative = raw.get("state")
        if (not isinstance(relative, str) or os.path.isabs(relative)
                or ".." in Path(relative).parts):
            raise ValueError("model input state is not a safe relative path")
        unresolved_state = model_path.parent / relative
        state = unresolved_state.resolve(strict=True)
        if not state.is_dir() or not v2._inside(model_path.parent, state) \
                or str(unresolved_state) != str(state) or str(state) in states:
            raise ValueError("model input state escapes or aliases the cohort")
        truth_item = truth_by_case.get(case_id)
        if truth_item is None or any(truth_item.get(key) != raw.get(key)
                                     for key in ("task_family", "candidate_id",
                                                 "input_seal")):
            raise ValueError("model and scalar truth case identities disagree")
        anchor, _anchor_sha = v2._read_json(state / "immutable_input_seal.json")
        if (type(anchor) is not dict or anchor.get("seal") != seal
                or anchor.get("schema") != "gv-immutable-input-seal-v1"
                or anchor.get("paths") != [
                    "world", "world_manifest.json", "task.md", "baseline",
                    "hook_feed.jsonl", "final.txt"]):
            raise ValueError("offline input anchor does not bind the cohort case")
        candidates.add(candidate); seals.add(seal); states.add(str(state))
        rows.append({"case_id": case_id, "candidate_id": candidate,
                     "task_family": family, "offline_input_seal": seal,
                     "state": state, "truth": truth_item})
    if set(truth_by_case) != {row["case_id"] for row in rows}:
        raise ValueError("scalar truth has extra cases")
    operator_cases = operator["cases"]
    if len(operator_cases) != len(rows):
        raise ValueError("retained operator cohort membership changed")
    for raw, row in zip(operator_cases, rows):
        if (type(raw) is not dict or set(raw) != _OPERATOR_CASE_KEYS
                or type(raw.get("source_state")) is not str
                or not os.path.isabs(raw["source_state"])
                or type(raw.get("truth")) is not dict
                or set(raw["truth"]) != _OPERATOR_TRUTH_KEYS
                or any(raw.get(key) != row.get(key) for key in (
                    "case_id", "task_family", "candidate_id"))
                or raw.get("input_seal") != row.get("offline_input_seal")
                or any(raw["truth"].get(key) != row["truth"].get(key)
                       for key in _OPERATOR_TRUTH_KEYS)):
            raise ValueError("retained operator cohort membership changed")
    family_labels = {}
    for row in rows:
        family_labels.setdefault(row["task_family"], set()).add(
            row["truth"]["label"])
    if (len(family_labels) < 3
            or any(labels != {0.0, 1.0}
                   for labels in family_labels.values())):
        raise ValueError("fixed cohort lacks general family/polarity coverage")
    return {"model_path": model_path, "truth_path": truth_path,
            "operator_path": operator_path,
            "model_sha256": model_sha, "truth_sha256": truth_sha,
            "operator_sha256": operator_sha,
            "cases": rows}


def validate_plan(plan, *, require_fresh_outputs=True):
    if type(plan) is not dict or set(plan) != _PLAN_KEYS \
            or plan.get("schema") != SCHEMA \
            or not _ID.fullmatch(plan.get("comparison_id", "")):
        raise ValueError("wrong scalar full-online comparison plan")
    output = v2._canonical(plan["output_root"])
    if require_fresh_outputs and (output.exists() or output.is_symlink()):
        raise ValueError("comparison output must be fresh")
    if any(v2._inside(root, output) for root in map(Path, (
            "/tmp", "/var/tmp", "/private/tmp", "/private/var/folders"))):
        raise ValueError("paid comparison output must be persistent")
    if (v2._sha(__file__) != plan.get("coordinator_sha256")
            or v2._sha(Path(v2.__file__)) != plan.get("comparison_support_sha256")):
        raise ValueError("comparison coordinator/support changed")
    cohort = _load_cohort(plan)
    raw_pairs = plan.get("pairs")
    if type(raw_pairs) is not list or len(raw_pairs) != len(cohort["cases"]):
        raise ValueError("plan must retain the complete precommitted cohort")
    selected_worker = v2._bind_worker_before_import(plan, raw_pairs)
    if str(selected_worker) != plan["worker_path"]:
        raise ValueError("comparison worker binding changed")
    worker = v2._load_worker(plan["worker_path"], plan["worker_sha256"])
    first_arms, outputs, frozen, pairs = [], set(), None, []
    for raw_pair, case in zip(raw_pairs, cohort["cases"]):
        if (type(raw_pair) is not dict or set(raw_pair) != _PAIR_KEYS
                or raw_pair.get("case_id") != case["case_id"]
                or raw_pair.get("candidate_id") != case["candidate_id"]):
            raise ValueError("pair order/identity differs from the fixed cohort")
        raw_jobs = raw_pair.get("jobs")
        if type(raw_jobs) is not list or len(raw_jobs) != 2:
            raise ValueError("every candidate requires two arms")
        jobs, arms, paired_identity, online_seal = [], [], None, None
        for raw_job in raw_jobs:
            if type(raw_job) is not dict or set(raw_job) != _JOB_KEYS:
                raise ValueError("wrong worker job schema")
            arm = raw_job.get("arm")
            if arm not in {"control", "treatment"} or arm in arms:
                raise ValueError("candidate needs distinct control/treatment arms")
            path = v2._canonical(raw_job["plan_path"], existing=True)
            worker_plan, digest = v2._read_json(path, 256 * 1024)
            if digest != raw_job.get("plan_sha256"):
                raise ValueError("worker plan digest mismatch")
            worker.validate_plan(worker_plan)
            if worker_plan.get("schema") != WORKER_SCHEMA \
                    or worker_plan.get("arm") != arm \
                    or worker_plan.get("input_source") != str(case["state"]):
                raise ValueError("worker plan does not bind its cohort input")
            snapshot = worker.input_snapshot(worker_plan["input_source"])
            if snapshot.get("seal") != worker_plan.get("input_seal"):
                raise ValueError("worker online input seal mismatch")
            if online_seal is None:
                online_seal = snapshot["seal"]
            elif online_seal != snapshot["seal"]:
                raise ValueError("paired arms use different input bytes")
            identity = {key: value for key, value in worker_plan.items()
                        if key not in {"arm", "output_root"}}
            if paired_identity is None:
                paired_identity = identity
            elif identity != paired_identity:
                raise ValueError("paired arms may differ only by arm/output")
            current = {key: worker_plan[key] for key in (
                "source_root", "source_commit", "source_fingerprint",
                "runtime_read_roots", "support_sha256")}
            if frozen is None:
                frozen = current
            elif frozen != current:
                raise ValueError("jobs do not share one frozen source/runtime")
            worker_output = v2._canonical(worker_plan["output_root"])
            if str(worker_output) in outputs or v2._inside(output, worker_output) \
                    or v2._inside(worker_output, output):
                raise ValueError("worker outputs alias or overlap coordinator output")
            outputs.add(str(worker_output))
            if require_fresh_outputs and worker_output.exists():
                raise ValueError("worker output is not fresh")
            visible = [Path(worker_plan[key]) for key in
                       ("source_root", "input_source", "output_root")]
            visible += list(map(Path, worker_plan["runtime_read_roots"]))
            private_truth = [cohort["truth_path"], cohort["operator_path"],
                             case["truth"]["provenance_path"]]
            if any(v2._inside(root, secret) or v2._inside(secret, root)
                   for root in visible for secret in private_truth):
                raise ValueError("scalar truth overlaps a worker-visible root")
            arms.append(arm)
            jobs.append({**raw_job, "worker_plan": worker_plan})
        first_arms.append(arms[0])
        pairs.append({**case, "online_input_seal": online_seal, "jobs": jobs})
    if first_arms.count("control") != first_arms.count("treatment"):
        raise ValueError("paired arm order must be counterbalanced")
    v2._verify_source(plan, pairs)
    expected_frozen = {"source_root": frozen["source_root"],
                       **plan["source_identity"],
                       "runtime_read_roots": frozen["runtime_read_roots"],
                       "support_sha256": frozen["support_sha256"]}
    if frozen != expected_frozen:
        raise ValueError("worker and coordinator source identities differ")
    v2._verify_runtime(plan, pairs)
    return worker, cohort, pairs


def _transport_integrity(audit, arm):
    transport = audit.get("typed_judger_transport") if type(audit) is dict else None
    keys = {"enabled", "protocol", "boundary",
            "packet_eligible_bound_methods", "packet_attached_attempts",
            "restricted_attempted", "ordinary_judger_attempts",
            "raw_prefetch_attached_attempts", "restricted_fallback_attempted",
            "packet_unavailable_fallback"}
    counters = ("packet_eligible_bound_methods", "packet_attached_attempts",
                "ordinary_judger_attempts", "raw_prefetch_attached_attempts")
    if (type(transport) is not dict or set(transport) != keys
            or type(audit.get("native_attempts")) is not int
            or audit.get("native_attempts", 0) <= 0
            or audit.get("observer_finished_attempts") != audit["native_attempts"]
            or any(type(transport.get(key)) is not int or transport[key] < 0
                   for key in counters)
            or any(type(transport.get(key)) is not bool for key in (
                "enabled", "restricted_attempted", "restricted_fallback_attempted",
                "packet_unavailable_fallback"))
            or transport["restricted_attempted"] is not
               (transport["packet_attached_attempts"] > 0)
            or transport["packet_attached_attempts"] > audit["native_attempts"]
            or transport["ordinary_judger_attempts"] > audit["native_attempts"]):
        return False
    if arm == "control":
        return (transport["enabled"] is False
                and transport["protocol"] is None
                and transport["boundary"] is None
                and transport["packet_eligible_bound_methods"] == 0
                and transport["packet_attached_attempts"] == 0
                and transport["raw_prefetch_attached_attempts"] == 0
                and transport["restricted_attempted"] is False
                and transport["restricted_fallback_attempted"] is False
                and transport["packet_unavailable_fallback"] is False)
    if (arm != "treatment" or transport["enabled"] is not True
            or transport["protocol"] != v2.TYPED_PACKET_PROTOCOL
            or transport["boundary"] != v2.TYPED_BOUNDARY):
        return False
    if transport["restricted_fallback_attempted"] \
            and (not transport["ordinary_judger_attempts"] or not
                 transport["packet_attached_attempts"]):
        return False
    if transport["packet_unavailable_fallback"] \
            and (not transport["packet_eligible_bound_methods"]
                 or not transport["ordinary_judger_attempts"]
                 or transport["packet_attached_attempts"]
                 or transport["restricted_attempted"]):
        return False
    if (transport["packet_attached_attempts"]
            and not transport["packet_eligible_bound_methods"]):
        return False
    return True


def scalar_truth_evaluation(truth, online_seal, semantic, final_status):
    findings = []
    if semantic.get("input_tree_seal") != online_seal:
        findings.append("input_seal")
    verdict = semantic.get("verdict")
    if semantic.get("truth_eligible") is not True \
            or verdict not in {"verified", "supported", "refuted"}:
        findings.append("nonterminal")
    if truth["label"] == 1.0:
        if verdict not in {"verified", "supported"}:
            findings.append("positive_verdict")
        if final_status != "verified":
            findings.append("positive_final_status")
    else:
        if verdict != "refuted":
            findings.append("negative_verdict")
        if final_status != "refuted":
            findings.append("negative_final_status")
    return {"official_label": truth["label"],
            "official_truth_identity": truth["identity"],
            "correct": not findings, "findings": findings,
            "nonterminal_is_failure": True}


def _basis_rank(value):
    return {None: -1, "record": 0, "world": 1}.get(value, -2)


def _strength_rank(value):
    return {None: 2, "analytic": 0, "bounded": 1, "decisive": 2}.get(value, -1)


def _basis_no_weaker(treatment, control):
    if treatment not in {None, "record", "world"} \
            or control not in {None, "record", "world"}:
        return False
    if treatment is None or control is None:
        return treatment is control
    return _basis_rank(treatment) >= _basis_rank(control)


def _strength_no_weaker(treatment, control):
    allowed = {None, "analytic", "bounded", "decisive"}
    return (treatment in allowed and control in allowed
            and _strength_rank(treatment) >= _strength_rank(control))


def _canon(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _multiset(values):
    result = {}
    for value in values:
        key = _canon(value)
        result[key] = result.get(key, 0) + 1
    return result


_ORDERED_CONTRACT_FIELDS = (
    "id", "text", "premise_scope", "required_basis", "severity", "clause",
    "refutable_by_absence",
)
_SEMANTIC_CONTRACT_FIELDS = (
    "text", "severity", "clause", "refutable_by_absence",
)


def _contract_key(obligation):
    """Return the complete compiler contract identity, or fail closed."""
    if (type(obligation) is not dict
            or set(obligation) != {"ordinal", *_ORDERED_CONTRACT_FIELDS}
            or type(obligation["ordinal"]) is not int
            or obligation["ordinal"] < 0
            or not isinstance(obligation["id"], str)
            or not obligation["id"]
            or not isinstance(obligation["text"], str)
            or type(obligation["premise_scope"]) is not list
            or any(not isinstance(item, str)
                   for item in obligation["premise_scope"])
            or obligation["required_basis"] not in {"record", "world"}
            or obligation["severity"] not in {"must", "should"}
            or not isinstance(obligation["clause"], str)
            or type(obligation["refutable_by_absence"]) is not bool):
        return None
    return {"ordinal": obligation["ordinal"],
            **{key: obligation[key] for key in _ORDERED_CONTRACT_FIELDS}}


def _semantic_contract_key(obligation):
    if (type(obligation) is not dict
            or any(key not in obligation for key in _SEMANTIC_CONTRACT_FIELDS)):
        return None
    return {key: obligation[key] for key in _SEMANTIC_CONTRACT_FIELDS}


def _certificate_contract_key(obligation):
    semantic = _semantic_contract_key(obligation)
    if semantic is None or not isinstance(obligation.get("id"), str) \
            or not obligation["id"]:
        return None
    return {"id": obligation["id"], **semantic}


def _valid_mechanism(value):
    if type(value) is not dict:
        return False
    if value.get("kind") == "agent":
        if set(value) not in ({"kind", "snapshot"},
                              {"kind", "snapshot", "oracle_adapter_request"}):
            return False
        snapshot = value.get("snapshot")
        if (type(snapshot) is not list
                or any(type(item) is not dict
                       or set(item) != {"logical_ref", "content_sha256"}
                       or not isinstance(item["logical_ref"], str)
                       or not _HASH.fullmatch(item.get("content_sha256", ""))
                       for item in snapshot)):
            return False
        request = value.get("oracle_adapter_request")
        return request is None or (
            type(request) is dict
            and set(request) == {"identity", "arguments"}
            and isinstance(request["identity"], str)
            and type(request["arguments"]) is dict)
    if value.get("kind") == "typed_adapter":
        return (set(value) == {
                    "kind", "adapter_identity", "arguments", "consumed",
                    "observation",
                }
                and isinstance(value["adapter_identity"], str)
                and type(value["arguments"]) is dict
                and type(value["consumed"]) is list
                and type(value["observation"]) is dict)
    return False


def _mechanism_equivalent(treatment, control):
    if not _valid_mechanism(treatment) or not _valid_mechanism(control):
        return False
    if treatment == control:
        return True
    # This is the sole cross-mechanism equivalence: the control's ordinary
    # discharge retained the Oracle's exact adapter request, and treatment
    # executed that same identity and arguments through the typed boundary.
    request = control.get("oracle_adapter_request")
    return (control.get("kind") == "agent"
            and treatment.get("kind") == "typed_adapter"
            and type(request) is dict
            and request.get("identity") == treatment.get("adapter_identity")
            and request.get("arguments") == treatment.get("arguments"))


def _evidence_dominates(treatment, control):
    fields = {"outcome", "basis", "strength", "witness_kind",
              "assumed_premises", "inputs", "bridges", "judger", "mechanism"}
    if type(treatment) is not dict or type(control) is not dict \
            or set(treatment) != fields or set(control) != fields:
        return False
    exact = ("outcome", "witness_kind", "judger")
    if any(treatment.get(key) != control.get(key) for key in exact):
        return False
    if (type(treatment.get("judger")) is not dict
            or type(control.get("judger")) is not dict
            or set(treatment["judger"]) != {"sufficient", "faithful", "degradation"}
            or set(control["judger"]) != {"sufficient", "faithful", "degradation"}
            or treatment["judger"].get("sufficient") != "yes"
            or treatment.get("judger", {}).get("faithful") != "yes"
            or control.get("judger", {}).get("sufficient") != "yes"
            or control.get("judger", {}).get("faithful") != "yes"
            or not _basis_no_weaker(
                treatment.get("basis"), control.get("basis"))
            or not _strength_no_weaker(
                treatment["strength"], control["strength"])
            or not _mechanism_equivalent(
                treatment["mechanism"], control["mechanism"])):
        return False
    if (type(treatment["inputs"]) is not list or type(control["inputs"]) is not list
            or type(treatment["assumed_premises"]) is not list
            or type(control["assumed_premises"]) is not list
            or type(treatment["bridges"]) is not list
            or type(control["bridges"]) is not list
            or _multiset(treatment["inputs"]) != _multiset(control["inputs"])):
        return False
    for field in ("assumed_premises", "bridges"):
        left = {_canon(item) for item in treatment.get(field) or []}
        right = {_canon(item) for item in control.get(field) or []}
        if not left <= right:
            return False
    return True


def _slice_no_weaker(control, treatment, reasons):
    left, right = control.get("slice"), treatment.get("slice")
    if type(left) is not list or type(right) is not list or len(left) != len(right):
        reasons.append("complete_claim_slice_not_exactly_alignable")
        return False
    entry_fields = {
        "id", "kind", "text", "status", "effective_status",
        "effective_basis", "effective_strength", "open_premises",
        "premise_refuted", "reason", "basis", "strength", "obligations",
    }
    obligation_fields = {
        "id", "text", "status", "reason", "basis", "strength", "severity",
        "clause", "refutable_by_absence",
    }
    if any(type(item) is not dict or set(item) != entry_fields
           for item in [*left, *right]):
        reasons.append("certificate_claim_shape_is_unknown")
        return False
    right_by_id = {item.get("id"): item for item in right if type(item) is dict}
    if len(right_by_id) != len(right):
        reasons.append("treatment_claim_slice_identity_invalid")
        return False
    for row in left:
        other = right_by_id.get(row.get("id")) if type(row) is dict else None
        if type(other) is not dict or any(row.get(key) != other.get(key) for key in (
                "kind", "text", "status", "effective_status", "open_premises",
                "premise_refuted")):
            reasons.append("claim_semantics_not_exactly_alignable")
            return False
        for basis, strength in (("basis", "strength"),
                                ("effective_basis", "effective_strength")):
            if (not _basis_no_weaker(other.get(basis), row.get(basis))
                    or not _strength_no_weaker(
                        other.get(strength), row.get(strength))):
                reasons.append("treatment_claim_assurance_is_weaker")
                return False
        left_obs, right_obs = row.get("obligations"), other.get("obligations")
        if type(left_obs) is not list or type(right_obs) is not list \
                or any(type(item) is not dict or set(item) != obligation_fields
                       for item in [*left_obs, *right_obs]) \
                or any(_certificate_contract_key(item) is None
                       for item in [*left_obs, *right_obs]) \
                or _multiset(_certificate_contract_key(item) for item in left_obs) \
                != _multiset(_certificate_contract_key(item) for item in right_obs):
            reasons.append("claim_contract_not_exactly_alignable")
            return False
        if len({_canon(_certificate_contract_key(item))
                for item in left_obs}) != len(left_obs):
            reasons.append("duplicate_contract_is_not_safely_alignable")
            return False
        buckets = {_canon(_certificate_contract_key(item)): item
                   for item in right_obs}
        for item in left_obs:
            target = buckets[_canon(_certificate_contract_key(item))]
            if item.get("status") != target.get("status"):
                reasons.append("obligation_status_changed")
                return False
            if (not _basis_no_weaker(target.get("basis"), item.get("basis"))
                    or not _strength_no_weaker(
                        target.get("strength"), item.get("strength"))):
                reasons.append("obligation_assurance_is_weaker")
                return False
    return True


_SETTLED_LINEAGE_KEYS = {
    "obligation", "status", "commit_seq", "critic_pass_seq", "compiled_seq",
    "oracle_seq", "execution_kind", "adapter_identity", "execution_seq",
    "execution_artifact", "judger_seq", "method_bound_seq", "discharge_seq",
    "binding_artifact", "certificate_seq", "stages_skipped_reason",
}


def _settled_lineage(value):
    if (type(value) is not dict or set(value) != _SETTLED_LINEAGE_KEYS
            or value.get("status") not in {"verified", "refuted"}
            or value.get("stages_skipped_reason") is not None
            or any(type(value.get(key)) is not int for key in (
                "commit_seq", "critic_pass_seq", "compiled_seq", "oracle_seq",
                "judger_seq", "method_bound_seq", "discharge_seq",
                "certificate_seq"))
            or not all(isinstance(value.get(key), str)
                       and _HASH.fullmatch(value[key]) for key in (
                           "obligation", "execution_artifact",
                           "binding_artifact"))):
        return False
    ordered = [value[key] for key in (
        "commit_seq", "critic_pass_seq", "compiled_seq", "oracle_seq")]
    execution = value.get("execution_seq")
    if value.get("execution_kind") == "adapter":
        if type(execution) is not int \
                or not isinstance(value.get("adapter_identity"), str):
            return False
        ordered.append(execution)
    elif value.get("execution_kind") == "agent":
        if execution is not None or value.get("adapter_identity") is not None:
            return False
    else:
        return False
    ordered += [value[key] for key in (
        "judger_seq", "method_bound_seq", "discharge_seq", "certificate_seq")]
    return all(left < right for left, right in zip(ordered, ordered[1:]))


def semantic_no_weaker(control_job, treatment_job):
    reasons = []
    def complete_boundary(job):
        architecture = job.get("architecture")
        caches = architecture.get("cache_events") if type(architecture) is dict else None
        raw_roles = architecture.get("roles_observed") \
            if type(architecture) is dict else None
        roles = set(raw_roles) if type(raw_roles) is list \
            and all(isinstance(item, str) for item in raw_roles) else set()
        lineages = architecture.get("final_obligation_lineage") \
            if type(architecture) is dict else None
        return (job.get("worker_complete") is True
                and job.get("accounting_complete") is True
                and job.get("transport_audit_complete") is True
                and job.get("scalar_truth", {}).get("correct") is True
                and type(caches) is dict and set(caches) == _CACHE_KEYS
                and all(type(value) is int and value == 0
                        for value in caches.values())
                and {"obligations", "critic", "oracles", "judger"} <= roles
                and roles <= {"obligations", "critic", "oracles",
                              "discharge", "judger"}
                and type(lineages) is list and bool(lineages)
                and all(_settled_lineage(item) for item in lineages)
                and (not any(item["execution_kind"] == "agent"
                             for item in lineages) or "discharge" in roles))
    if not all(complete_boundary(job) for job in (control_job, treatment_job)):
        return {"proven": False, "status": "unproven",
                "reasons": ["worker_or_scalar_truth_incomplete"]}
    control_quality, treatment_quality = (job.get("quality_view", {})
                                          for job in (control_job, treatment_job))
    control_sem, treatment_sem = (job.get("semantic_evidence", {})
                                  for job in (control_job, treatment_job))
    full_control = {"semantic_evidence": control_sem, **control_quality}
    full_treatment = {"semantic_evidence": treatment_sem, **treatment_quality}
    full_equal = _canon(full_control) == _canon(full_treatment)
    verdict_rank = {"refuted": 0, "supported": 1, "verified": 2}
    if verdict_rank.get(treatment_sem.get("verdict"), -1) \
            < verdict_rank.get(control_sem.get("verdict"), -1):
        reasons.append("treatment_verdict_is_weaker")
    control_cert, treatment_cert = (quality.get("certificate", {})
                                    for quality in (control_quality,
                                                    treatment_quality))
    if control_cert.get("final") != treatment_cert.get("final"):
        reasons.append("final_claim_identity_changed")
    control_ordered = control_quality.get("ordered_contract")
    treatment_ordered = treatment_quality.get("ordered_contract")
    if (type(control_ordered) is not list or type(treatment_ordered) is not list
            or any(_contract_key(item) is None
                   for item in [*control_ordered, *treatment_ordered])
            or _multiset(_contract_key(item) for item in control_ordered)
            != _multiset(_contract_key(item) for item in treatment_ordered)):
        reasons.append("ordered_compiler_contract_not_exactly_alignable")
        control_ordered = treatment_ordered = []
    slice_ok = _slice_no_weaker(control_cert, treatment_cert, reasons)
    left_obs, right_obs = control_sem.get("obligations"), treatment_sem.get("obligations")
    if (type(left_obs) is not list or type(right_obs) is not list
            or any(type(item) is not dict
                   or _semantic_contract_key(item.get("contract")) is None
                   for item in [*left_obs, *right_obs])
            or _multiset(_semantic_contract_key(item["contract"]) for item in left_obs)
            != _multiset(_semantic_contract_key(item["contract"])
                         for item in right_obs)):
        reasons.append("final_evidence_contract_not_exactly_alignable")
    else:
        ordered_semantics = [_semantic_contract_key(item)
                             for item in control_ordered]
        if (_multiset(ordered_semantics) != _multiset(
                _semantic_contract_key(item["contract"]) for item in left_obs)
                or len({_canon(item) for item in ordered_semantics})
                != len(ordered_semantics)):
            reasons.append("semantic_evidence_is_not_bound_to_ordered_contract")
            return {"proven": False, "status": "unproven",
                    "reasons": sorted(set(reasons)),
                    "full_semantic_equal": full_equal,
                    "unknown_is_never_no_weaker": True}
        if len({_canon(_semantic_contract_key(item["contract"]))
                for item in left_obs}) != len(left_obs):
            reasons.append("duplicate_final_contract_is_not_safely_alignable")
            return {"proven": False, "status": "unproven",
                    "reasons": sorted(set(reasons)),
                    "full_semantic_equal": full_equal,
                    "unknown_is_never_no_weaker": True}
        right_by_contract = {
            _canon(_semantic_contract_key(item["contract"])): item
            for item in right_obs}
        for obligation in left_obs:
            target = right_by_contract[_canon(
                _semantic_contract_key(obligation["contract"]))]
            if obligation.get("status") != target.get("status"):
                reasons.append("final_obligation_status_changed")
                continue
            left_evidence, right_evidence = (obligation.get("evidence"),
                                             target.get("evidence"))
            if type(left_evidence) is not list or type(right_evidence) is not list:
                reasons.append("final_evidence_is_malformed")
                continue
            unused = set(range(len(right_evidence)))
            for evidence in left_evidence:
                match = next((index for index in unused
                              if _evidence_dominates(
                                  right_evidence[index], evidence)), None)
                if match is None:
                    reasons.append("control_evidence_has_no_dominating_treatment_evidence")
                    break
                unused.remove(match)
    proven = slice_ok and not reasons
    return {"proven": proven, "status": "proven" if proven else "unproven",
            "reasons": sorted(set(reasons)),
            "full_semantic_equal": full_equal,
            "unknown_is_never_no_weaker": True}


def capability_observation(treatment):
    if treatment.get("worker_complete") is not True \
            or treatment.get("transport_audit_complete") is not True:
        return {"cell": "unknown_or_failed", "descriptive_only": True,
                "used_for_selection": False}
    transport = treatment.get("typed_transport", {})
    adapter_exercised = treatment.get("typed_adapter_exercised") is True
    adapter_bound = treatment.get("typed_adapter_bound") is True
    attached = transport.get("packet_attached_attempts", 0)
    if not adapter_exercised:
        cell = "adapter_not_exercised"
    elif not adapter_bound:
        cell = "adapter_exercised_not_bound"
    elif attached and transport.get("restricted_fallback_attempted"):
        cell = "packet_attempted_fallback"
    elif attached:
        cell = "packet_attempted_no_fallback"
    elif transport.get("packet_unavailable_fallback"):
        cell = "packet_unavailable_fallback"
    else:
        cell = "adapter_bound_no_packet_attempt"
    return {"cell": cell, "descriptive_only": True,
            "used_for_selection": False,
            "typed_adapter_exercised": adapter_exercised,
            "typed_adapter_bound": adapter_bound,
            "typed_transport": transport}


def compare_pair(control, treatment):
    complete = all(job.get("worker_complete") is True
                   and job.get("accounting_complete") is True
                   and job.get("transport_audit_complete") is True
                   for job in (control, treatment))
    both_correct = bool(complete
                        and control.get("scalar_truth", {}).get("correct") is True
                        and treatment.get("scalar_truth", {}).get("correct") is True)
    no_weaker = semantic_no_weaker(control, treatment)
    return {"complete": complete,
            "control_official_label_correct":
                control.get("scalar_truth", {}).get("correct") is True,
            "treatment_official_label_correct":
                treatment.get("scalar_truth", {}).get("correct") is True,
            "both_arms_official_label_correct": both_correct,
            "semantic_no_weaker": no_weaker,
            "strict_quality_pass": both_correct and no_weaker["proven"],
            "capability_observation": capability_observation(treatment)}


def _postflight(plan, pair, job):
    cohort = plan["cohort"]
    if (v2._sha(__file__) != plan["coordinator_sha256"]
            or v2._sha(Path(v2.__file__)) != plan["comparison_support_sha256"]
            or v2._sha(plan["worker_path"]) != plan["worker_sha256"]
            or v2._sha(job["plan_path"]) != job["plan_sha256"]
            or v2._sha(cohort["model_input_manifest"])
            != cohort["model_input_manifest_sha256"]
            or v2._sha(cohort["scalar_truth_manifest"])
            != cohort["scalar_truth_manifest_sha256"]
            or v2._sha(cohort["operator_manifest"])
            != cohort["operator_manifest_sha256"]
            or v2._sha(pair["truth"]["provenance_path"])
            != pair["truth"]["provenance"]["sha256"]):
        raise ValueError("coordinator, worker, cohort, or Harbor truth changed")


def _run_job(plan, worker, output, pair, job, ordinal):
    stdout_path, stderr_path = (output / f"job-{ordinal:03d}.{kind}"
                                for kind in ("stdout", "stderr"))
    command = [sys.executable, plan["worker_path"], "--plan", job["plan_path"],
               "--plan-sha256", job["plan_sha256"], "--execute",
               "--confirm-paid", "GPT-5.6-Luna"]
    with v2._private_output(stdout_path) as stdout, \
            v2._private_output(stderr_path) as stderr:
        started = time.monotonic_ns()
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   env=v2._subprocess_env(), start_new_session=True)
        wait_error = None
        try:
            returncode = process.wait()
            finished = time.monotonic_ns()
        except BaseException as error:
            wait_error = error
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            finished = time.monotonic_ns()
            returncode = process.returncode
    base = {"candidate_id": pair["candidate_id"], "case_id": pair["case_id"],
            "arm": job["arm"],
            "worker_output_root": job["worker_plan"]["output_root"],
            "returncode": returncode,
            "spawn_to_exit_wall_seconds": (finished - started) / 1e9,
            "stdout_sha256": v2._sha(stdout_path),
            "stderr_sha256": v2._sha(stderr_path)}
    if wait_error is not None:
        accounting = None
        try:
            raw_result, _digest = v2._read_json(
                Path(job["worker_plan"]["output_root"]) / "worker-result.json")
            if type(raw_result) is dict and raw_result.get("plan") == job["worker_plan"]:
                accounting = raw_result.get("accounting")
        except BaseException:
            pass
        summary = {**base, "worker_complete": False, "accounting": accounting,
                   "accounting_complete": False,
                   "transport_audit_complete": False,
                   "supervisor_failure": type(wait_error).__name__,
                   "failure_wall_retained": True}
        v2._write_new(output / f"job-{ordinal:03d}-result.json", summary)
        return summary
    try:
        result, digest = v2._read_worker_result(job)
        _postflight(plan, pair, job)
        semantic, projection, final_status = v2._quality_view(worker, result)
        truth_eval = scalar_truth_evaluation(
            pair["truth"], pair["online_input_seal"], semantic, final_status)
        accounting = result.get("accounting")
        audit = result.get("execution_audit", {})
        transport_ok = _transport_integrity(audit, job["arm"])
    except BaseException as error:
        accounting = None
        try:
            raw_result, _digest = v2._read_json(
                Path(job["worker_plan"]["output_root"]) / "worker-result.json")
            if type(raw_result) is dict and raw_result.get("plan") == job["worker_plan"]:
                accounting = raw_result.get("accounting")
        except BaseException:
            pass
        summary = {**base, "worker_complete": False, "accounting": accounting,
                   "accounting_complete": False,
                   "transport_audit_complete": False,
                   "supervisor_failure": type(error).__name__}
        v2._write_new(output / f"job-{ordinal:03d}-result.json", summary)
        return summary
    quality = {key: result["chain"]["quality_view"][key] for key in (
        "ordered_contract", "ordered_outcomes", "certificate", "score")}
    summary = {
        **base, "worker_result_sha256": digest,
        "worker_complete": result.get("complete") is True and returncode == 0,
        "worker_failures": result.get("failures"), "accounting": accounting,
        "accounting_complete": type(accounting) is dict
            and accounting.get("accounting_complete") is True,
        "semantic_evidence": semantic,
        "semantic_evidence_sha256": v2._canonical_json_sha(semantic),
        "quality_view": quality, "quality_sha256": v2._canonical_json_sha(quality),
        "projection": projection, "scalar_truth": truth_eval,
        "truth_eligible": semantic.get("truth_eligible") is True,
        "typed_adapter_exercised": audit.get("typed_adapter_exercised") is True,
        "typed_adapter_bound": audit.get("typed_adapter_bound") is True,
        "typed_transport": audit.get("typed_judger_transport", {}),
        "transport_audit_complete": transport_ok,
        "cache_events": audit.get("cache_events"),
        "architecture": {key: audit.get(key) for key in (
            "roles_observed", "cache_events", "final_obligation_lineage")},
    }
    v2._write_new(output / f"job-{ordinal:03d}-result.json", summary)
    return summary


def preflight(plan):
    worker, cohort, pairs = validate_plan(plan)
    v2._verify_runtime(plan, pairs, version=True)
    reports = []
    for pair in pairs:
        for job in pair["jobs"]:
            run = subprocess.run([
                sys.executable, plan["worker_path"], "--plan", job["plan_path"],
                "--plan-sha256", job["plan_sha256"], "--preflight"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
                timeout=300, env=v2._subprocess_env())
            if run.returncode:
                raise ValueError("worker preflight failed")
            report = json.loads(run.stdout.decode(errors="strict"),
                                object_pairs_hook=v2._strict_object,
                                parse_constant=v2._nonfinite)
            wp = job["worker_plan"]
            if (report.get("schema") != WORKER_SCHEMA
                    or report.get("preflight_complete") is not True
                    or report.get("model_calls") != 0
                    or report.get("runtime") != RUNTIME
                    or report.get("input_seal") != wp["input_seal"]
                    or Path(wp["output_root"]).exists()):
                raise ValueError("worker preflight is not sealed/read-only")
            reports.append({"case_id": pair["case_id"],
                            "candidate_id": pair["candidate_id"],
                            "arm": job["arm"], "report": report})
    validate_plan(plan)
    return {"schema": SCHEMA, "preflight_complete": True, "model_calls": 0,
            "worker_schema": WORKER_SCHEMA, "runtime": RUNTIME,
            "cohort_size": len(cohort["cases"]),
            "truth_transport": "coordinator-only-Harbor-scalar-v1",
            "capability_strata": "post-hoc-all-candidate-coverage-gated",
            "jobs": reports}


def _stratum_report(name, value, selected, jobs):
    candidate_ids = [row["candidate_id"] for row in selected]
    wanted = set(candidate_ids)
    selected_jobs = [job for job in jobs if job.get("candidate_id") in wanted]
    expected = 2 * len(selected)
    complete = bool(selected) and all(row.get("complete") is True for row in selected)
    return {
        "stratum": name, "value": value, "count": len(selected),
        "candidate_ids": candidate_ids, "all_candidates_retained":
            len(selected_jobs) == expected,
        "complete": complete,
        "all_both_arms_official_label_correct": bool(
            complete and all(row.get("both_arms_official_label_correct") is True
                             for row in selected)),
        "all_semantic_no_weaker": bool(
            complete and all(row.get("semantic_no_weaker", {}).get("proven") is True
                             for row in selected)),
        "cost": v2._accounting_summary(selected_jobs, expected_jobs=expected),
        "latency": v2._latency(selected_jobs, expected),
    }


def _typed_mechanism_coverage(reports):
    observations = [row.get("capability_observation", {}) for row in reports]
    usable = [item for item in observations
              if item.get("cell") != "unknown_or_failed"]
    adapter = [item.get("typed_adapter_exercised") is True for item in usable]
    packet_eligible, restricted = [], []
    for item in usable:
        transport = item.get("typed_transport", {})
        eligible = transport.get("packet_eligible_bound_methods")
        packet_eligible.append(type(eligible) is int and eligible > 0)
        restricted.append(transport.get("restricted_attempted") is True)
    result = {
        "complete_observations": len(usable) == len(reports),
        "adapter_eligible_observed": any(adapter),
        "adapter_ineligible_observed": any(not value for value in adapter),
        "packet_eligible_observed": any(packet_eligible),
        "packet_ineligible_observed": any(not value for value in packet_eligible),
        "restricted_packet_attempt_observed": any(restricted),
        "posthoc_gate_does_not_select_candidates": True,
    }
    result["acceptance_coverage_complete"] = bool(reports) and all(result.values())
    return result


def run_comparison(plan):
    preflight_report = preflight(plan)
    worker, cohort, pairs = validate_plan(plan)
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    started, run_id = time.monotonic_ns(), uuid.uuid4().hex
    v2._write_new(output / "comparison-start.json", {
        "schema": RESULT_SCHEMA, "status": "started", "run_id": run_id,
        "plan": plan, "preflight": preflight_report, "runtime": RUNTIME,
        "automatic_outer_retries": 0, "cohort_selection": "precommitted-all"})
    jobs, reports, ordinal = [], [], 0
    for pair in pairs:
        current = []
        for job in pair["jobs"]:
            ordinal += 1
            try:
                summary = _run_job(plan, worker, output, pair, job, ordinal)
            except BaseException as error:
                summary = {"case_id": pair["case_id"],
                           "candidate_id": pair["candidate_id"], "arm": job["arm"],
                           "worker_output_root": job["worker_plan"]["output_root"],
                           "worker_complete": False, "accounting_complete": False,
                           "transport_audit_complete": False,
                           "supervisor_failure": type(error).__name__}
            jobs.append(summary); current.append(summary)
        by_arm = {row["arm"]: row for row in current}
        reports.append({"case_id": pair["case_id"],
                        "candidate_id": pair["candidate_id"],
                        "task_family": pair["task_family"],
                        "official_label": pair["truth"]["label"],
                        **compare_pair(by_arm.get("control", {}),
                                       by_arm.get("treatment", {}))})
    cost = v2._accounting_summary(jobs, expected_jobs=2 * len(pairs))
    latency = v2._latency(jobs, 2 * len(pairs))
    complete = len(reports) == len(pairs) and all(row["complete"] for row in reports)
    all_correct = complete and all(
        row["both_arms_official_label_correct"] for row in reports)
    strict_quality = all_correct and all(row["strict_quality_pass"] for row in reports)
    cells = []
    for cell in sorted({row["capability_observation"]["cell"] for row in reports}):
        selected = [row for row in reports
                    if row["capability_observation"]["cell"] == cell]
        cells.append({"cell": cell,
                      "candidate_ids": [row["candidate_id"] for row in selected],
                      "count": len(selected), "descriptive_only": True,
                      "used_for_selection": False})
    coverage = _typed_mechanism_coverage(reports)
    acceptance_quality = strict_quality and coverage["acceptance_coverage_complete"]
    family_breakdown = [
        _stratum_report("task_family", family,
                        [row for row in reports if row["task_family"] == family], jobs)
        for family in sorted({row["task_family"] for row in reports})
    ]
    capability_breakdown = [
        _stratum_report(
            "capability_cell", cell,
            [row for row in reports
             if row["capability_observation"]["cell"] == cell], jobs)
        for cell in sorted({row["capability_observation"]["cell"]
                            for row in reports})
    ]
    result = {
        "schema": RESULT_SCHEMA, "status": "terminal", "run_id": run_id,
        "complete": complete, "precommitted_candidate_count": len(pairs),
        "reported_candidate_count": len(reports),
        "all_candidates_retained": len(reports) == len(pairs),
        "all_both_arms_official_label_correct": all_correct,
        "all_semantic_no_weaker": strict_quality,
        "strict_quality_acceptance": strict_quality,
        "cost": cost, "latency": latency,
        "typed_mechanism_coverage": coverage,
        "ten_x_strict_quality_acceptance": v2._ten_x(
            acceptance_quality, cost, latency),
        "pairs": reports, "capability_strata": cells,
        "per_family": family_breakdown,
        "per_capability_stratum": capability_breakdown,
        "capability_strata_descriptive_only": False,
        "capability_strata_used_for_selection": False,
        "capability_strata_used_for_ten_x_coverage_gate": True,
        "selection_policy": "all-precommitted-candidates-v1",
        "wall_seconds": (time.monotonic_ns() - started) / 1e9,
    }
    v2._write_new(output / "comparison-result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid", choices=("GPT-5.6-Luna",))
    args = parser.parse_args(argv)
    plan, digest = v2._read_json(args.plan)
    if digest != args.plan_sha256:
        parser.error("operator comparison-plan digest mismatch")
    if args.preflight and args.execute:
        parser.error("preflight and execute are mutually exclusive")
    if args.preflight:
        print(json.dumps(preflight(plan), sort_keys=True))
        return 0
    if not args.execute:
        validate_plan(plan)
        print(json.dumps({"valid": True, "model_calls": 0,
                          "schema": SCHEMA}, sort_keys=True))
        return 0
    if args.confirm_paid != "GPT-5.6-Luna":
        parser.error("paid comparison requires --confirm-paid GPT-5.6-Luna")
    result = run_comparison(plan)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ten_x_strict_quality_acceptance"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
