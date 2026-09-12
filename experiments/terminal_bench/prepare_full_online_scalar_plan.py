"""Materialize task-general full-online v3 plans from a sealed scalar cohort.

No model is invoked.  The builder computes online arbitrary-tree seals and
source/support identities, creates both worker arms for every manifest case,
and preserves the manifest order as the immutable comparison cohort.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

try:
    from . import run_full_online_comparison as v2
    from . import run_full_online_comparison_v3 as v3
except ImportError:  # direct script execution
    import run_full_online_comparison as v2
    import run_full_online_comparison_v3 as v3


SCHEMA = "gv-full-online-scalar-plan-builder-v2"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_KEYS = {
    "schema", "comparison_id", "model_input_manifest",
    "model_input_manifest_sha256", "scalar_truth_manifest",
    "scalar_truth_manifest_sha256", "cohort_operator_manifest",
    "cohort_operator_manifest_sha256", "plan_output_root",
    "comparison_output_root", "worker_outputs_root", "source_root",
    "source_commit", "source_fingerprint", "runtime_read_roots",
    "runtime_identity",
    "worker_path", "coordinator_path",
}


def _fresh(value, field):
    path = v2._canonical(value)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise ValueError(f"{field} must be fresh with an existing parent")
    return path


def _overlap(left, right):
    return v2._inside(left, right) or v2._inside(right, left)


def prepare(operator):
    if type(operator) is not dict or set(operator) != _KEYS \
            or operator.get("schema") != SCHEMA \
            or not v3._ID.fullmatch(operator.get("comparison_id", "")):
        raise ValueError("wrong scalar online plan-builder manifest")
    model_path = v2._canonical(operator["model_input_manifest"], existing=True)
    truth_path = v2._canonical(operator["scalar_truth_manifest"], existing=True)
    cohort_operator_path = v2._canonical(
        operator["cohort_operator_manifest"], existing=True)
    source = v2._canonical(operator["source_root"], existing=True)
    worker_path = v2._canonical(operator["worker_path"], existing=True)
    coordinator_path = v2._canonical(operator["coordinator_path"], existing=True)
    plan_root = _fresh(operator["plan_output_root"], "plan_output_root")
    comparison_output = _fresh(
        operator["comparison_output_root"], "comparison_output_root")
    worker_outputs = _fresh(operator["worker_outputs_root"], "worker_outputs_root")
    boundaries = [model_path.parent, truth_path.parent, cohort_operator_path,
                  source, plan_root, comparison_output, worker_outputs]
    if any(_overlap(left, right) for index, left in enumerate(boundaries)
           for right in boundaries[index + 1:]):
        raise ValueError("plan, source, inputs, truth, and outputs must be disjoint")
    if (not source.is_dir() or (source / ".git").exists()
            or not _COMMIT.fullmatch(operator.get("source_commit", ""))):
        raise ValueError("source must be a frozen archive with an exact commit pin")
    expected_worker = (source / "experiments" / "terminal_bench"
                       / "run_full_online_worker.py").resolve(strict=True)
    if coordinator_path != Path(v3.__file__).resolve() \
            or worker_path != expected_worker:
        raise ValueError("builder must target the current v3 coordinator/v2 worker")
    for path, key in ((model_path, "model_input_manifest_sha256"),
                      (truth_path, "scalar_truth_manifest_sha256"),
                      (cohort_operator_path,
                       "cohort_operator_manifest_sha256")):
        if not _HASH.fullmatch(operator.get(key, "")) \
                or v2._sha(path) != operator[key]:
            raise ValueError("cohort manifest digest mismatch")
    roots = operator["runtime_read_roots"]
    if type(roots) is not list or len(roots) > 16 \
            or any(str(v2._canonical(root, existing=True)) != root for root in roots):
        raise ValueError("runtime roots must be explicit canonical directories")
    runtime_identity = operator["runtime_identity"]
    if type(runtime_identity) is not dict \
            or set(runtime_identity) != v2._RUNTIME_KEYS \
            or set(runtime_identity.get("runtime_read_root_fingerprints", {})) \
            != set(roots):
        raise ValueError("runtime identity does not cover the runtime roots")

    model, _model_sha = v2._read_json(model_path)
    truth, _truth_sha = v2._read_json(truth_path)
    if (type(model) is not dict or model.get("schema") != v3.MODEL_INPUT_SCHEMA
            or type(model.get("cases")) is not list
            or type(truth) is not dict
            or truth.get("schema") != v3.SCALAR_TRUTH_SCHEMA
            or truth.get("operator_manifest_sha256")
            != operator["cohort_operator_manifest_sha256"]):
        raise ValueError("wrong model input manifest")
    # Bind the canonical archive as inert bytes before executing its worker.
    # The post-import recheck also detects import-time source mutation.
    source_fingerprint = v2._source_fingerprint(source)
    if (not _HASH.fullmatch(operator.get("source_fingerprint", ""))
            or source_fingerprint != operator["source_fingerprint"]):
        raise ValueError("source archive differs from the operator authority")
    worker_sha256 = v2._sha(worker_path)
    if v2._source_fingerprint(source) != source_fingerprint:
        raise ValueError("source archive changed before worker import")
    worker = v2._load_worker(str(worker_path), worker_sha256)
    if v2._source_fingerprint(source) != source_fingerprint:
        raise ValueError("source archive changed during worker import")
    support = {name: v2._sha(worker_path.with_name(name))
               for name in worker.SUPPORT_FILES}
    source_identity = {"source_commit": operator["source_commit"],
                       "source_fingerprint": operator["source_fingerprint"]}
    plan_root.mkdir(mode=0o700)
    workers_dir = plan_root / "workers"
    workers_dir.mkdir(mode=0o700)
    pairs = []
    for index, case in enumerate(model["cases"]):
        if type(case) is not dict or set(case) != v3._MODEL_CASE_KEYS:
            raise ValueError("model input case has wrong fields")
        case_id, candidate = case["case_id"], case["candidate_id"]
        if not v3._ID.fullmatch(case_id) or not v3._ID.fullmatch(candidate):
            raise ValueError("case identities are not safe")
        state = (model_path.parent / case["state"]).resolve(strict=True)
        if not state.is_dir() or not v2._inside(model_path.parent, state):
            raise ValueError("candidate state escapes the package")
        online_seal = worker.input_snapshot(state)["seal"]
        case_dir = workers_dir / case_id
        case_dir.mkdir(mode=0o700)
        order = ("control", "treatment") if index % 2 == 0 \
            else ("treatment", "control")
        jobs = []
        for arm in order:
            worker_plan = {
                "schema": worker.SCHEMA, "source_root": str(source),
                "source_commit": operator["source_commit"],
                "source_fingerprint": source_fingerprint,
                "input_source": str(state), "input_seal": online_seal,
                "output_root": str(worker_outputs / case_id / arm),
                "arm": arm, "runtime_read_roots": list(roots),
                "support_sha256": support,
            }
            path = case_dir / f"{arm}.json"
            v2._write_new(path, worker_plan)
            jobs.append({"arm": arm, "plan_path": str(path),
                         "plan_sha256": v2._sha(path)})
        pairs.append({"case_id": case_id, "candidate_id": candidate,
                      "jobs": jobs})
    plan = {
        "schema": v3.SCHEMA, "comparison_id": operator["comparison_id"],
        "output_root": str(comparison_output), "worker_path": str(worker_path),
        "worker_sha256": worker_sha256,
        "coordinator_sha256": v2._sha(coordinator_path),
        "comparison_support_sha256": v2._sha(Path(v2.__file__)),
        "source_identity": source_identity, "runtime_identity": runtime_identity,
        "cohort": {
            "package_id": model["package_id"],
            "model_input_manifest": str(model_path),
            "model_input_manifest_sha256": operator["model_input_manifest_sha256"],
            "scalar_truth_manifest": str(truth_path),
            "scalar_truth_manifest_sha256": operator["scalar_truth_manifest_sha256"],
            "operator_manifest": str(cohort_operator_path),
            "operator_manifest_sha256":
                operator["cohort_operator_manifest_sha256"],
        },
        "pairs": pairs,
    }
    comparison_path = plan_root / "comparison-plan.json"
    v2._write_new(comparison_path, plan)
    v3.validate_plan(plan)
    result = {
        "schema": SCHEMA, "comparison_id": operator["comparison_id"],
        "cases": len(pairs), "jobs": 2 * len(pairs),
        "comparison_plan": str(comparison_path),
        "comparison_plan_sha256": v2._sha(comparison_path),
        "source_fingerprint": source_fingerprint,
        "model_calls": 0, "task_specific_truth_fields": 0,
    }
    v2._write_new(plan_root / "materialization-result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator_manifest")
    args = parser.parse_args(argv)
    operator, _digest = v2._read_json(args.operator_manifest)
    print(json.dumps(prepare(operator), sort_keys=True))


if __name__ == "__main__":
    main()
