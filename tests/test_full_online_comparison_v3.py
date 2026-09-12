"""No-model tests for fixed-cohort scalar full-online comparison v3."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.terminal_bench import prepare_full_online_scalar_plan as builder
from experiments.terminal_bench import run_full_online_comparison as v2
from experiments.terminal_bench import run_full_online_comparison_v3 as comparison


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cohort(tmp_path):
    model_root = tmp_path / "model"
    truth_root = tmp_path / "truth"
    model_root.mkdir(parents=True); truth_root.mkdir(parents=True)
    model_cases, truth_cases = [], []
    families = ("family-a", "family-b", "family-c")
    for index in range(6):
        case_id, candidate = f"case-{index}", f"candidate-{index}"
        state = model_root / "cases" / case_id / "gv"
        state.mkdir(parents=True)
        (state / "task.md").write_text("Inspect the sealed candidate.\n")
        offline_seal = hashlib.sha256(f"offline-{index}".encode()).hexdigest()
        _write(state / "immutable_input_seal.json", {
            "schema": "gv-immutable-input-seal-v1", "seal": offline_seal,
            "paths": ["world", "world_manifest.json", "task.md", "baseline",
                      "hook_feed.jsonl", "final.txt"],
        })
        provenance_path = truth_root / f"harbor-{index}.json"
        label = float(index % 2 == 0)
        provenance_sha = _write(provenance_path, {
            "verifier_result": {"reward": label}})
        provenance = {"kind": "harbor_result_json",
                      "path": str(provenance_path.resolve()),
                      "sha256": provenance_sha,
                      "json_path": ["verifier_result", "reward"]}
        family = families[index % 3]
        model_cases.append({
            "case_id": case_id, "candidate_id": candidate,
            "task_family": family, "state": f"cases/{case_id}/gv",
            "input_seal": offline_seal,
        })
        truth_cases.append({
            "case_id": case_id, "candidate_id": candidate,
            "task_family": family, "input_seal": offline_seal,
            "label": label, "provenance": provenance,
            "identity": comparison._truth_identity(label, provenance),
        })
    model = {"schema": comparison.MODEL_INPUT_SCHEMA,
             "package_id": "arbitrary-cohort", "cases": model_cases}
    model_path = model_root / "input-manifest.json"
    model_sha = _write(model_path, model)
    package_operator = {
        "schema": "gv-general-offline-eval-operator-v1",
        "package_id": "arbitrary-cohort",
        "model_input_root": str(model_root.resolve()),
        "truth_root": str(truth_root.resolve()),
        "cases": [{
            "case_id": model_case["case_id"],
            "task_family": model_case["task_family"],
            "candidate_id": model_case["candidate_id"],
            "source_state": str(
                (model_root / model_case["state"]).resolve()),
            "input_seal": model_case["input_seal"],
            "truth": {key: truth_case[key] for key in (
                "label", "identity", "provenance")},
        } for model_case, truth_case in zip(model_cases, truth_cases)],
    }
    package_operator_path = tmp_path / "package-operator.json"
    package_operator_sha = _write(package_operator_path, package_operator)
    truth = {
        "schema": comparison.SCALAR_TRUTH_SCHEMA,
        "package_id": "arbitrary-cohort",
        "operator_manifest_sha256": package_operator_sha,
        "model_input_manifest_sha256": model_sha,
        "model_input_root": str(model_root.resolve()), "cases": truth_cases,
    }
    truth_path = truth_root / "truth-manifest.json"
    truth_sha = _write(truth_path, truth)
    return (model_path.resolve(), model_sha, truth_path.resolve(), truth_sha,
            package_operator_path.resolve(), package_operator_sha)


def _fixture_plan(tmp_path, monkeypatch):
    (model_path, model_sha, truth_path, truth_sha,
     operator_path, operator_sha) = _cohort(tmp_path)
    source = tmp_path / "frozen-source"; source.mkdir()
    worker_file = tmp_path / "fixture-worker.py"; worker_file.write_text("# fixture\n")
    runtime = tmp_path / "runtime"; runtime.mkdir()
    cli = runtime / "codex"; cli.write_text("fixture\n"); cli.chmod(0o700)

    def snapshot(root):
        values = []
        for path in sorted(Path(root).rglob("*")):
            if path.is_file():
                values.append((str(path.relative_to(root)),
                               hashlib.sha256(path.read_bytes()).hexdigest()))
        return {"seal": hashlib.sha256(json.dumps(values).encode()).hexdigest()}

    fake_worker = SimpleNamespace(
        SCHEMA=comparison.WORKER_SCHEMA,
        RESULT_SCHEMA=comparison.WORKER_RESULT_SCHEMA,
        RUNTIME=comparison.RUNTIME, validate_plan=lambda value: value,
        input_snapshot=snapshot, _certificate_semantics=lambda value: value,
        _score_semantics=lambda value: value, _content_digest=v2._canonical_json_sha,
    )
    monkeypatch.setattr(v2, "_load_worker", lambda *_args: fake_worker)
    monkeypatch.setattr(v2, "_bind_worker_before_import",
                        lambda plan, _pairs: Path(plan["worker_path"]))
    monkeypatch.setattr(v2, "_verify_source", lambda *_args: None)
    monkeypatch.setattr(v2, "_verify_runtime", lambda *_args, **_kwargs: None)
    source_identity = {"source_commit": "b" * 40,
                       "source_fingerprint": "c" * 64}
    runtime_identity = {
        "codex_cli_path": str(cli.resolve()), "codex_cli_realpath": str(cli.resolve()),
        "codex_cli_sha256": v2._sha(cli), "codex_cli_version": "fixture",
        "runtime_read_root_fingerprints": {str(runtime.resolve()): "d" * 64},
    }
    model = json.loads(model_path.read_text())
    pairs = []
    for index, case in enumerate(model["cases"]):
        state = (model_path.parent / case["state"]).resolve()
        online_seal = snapshot(state)["seal"]
        order = ("control", "treatment") if index % 2 == 0 \
            else ("treatment", "control")
        jobs = []
        for arm in order:
            wp = {"schema": comparison.WORKER_SCHEMA,
                  "source_root": str(source.resolve()),
                  "source_commit": source_identity["source_commit"],
                  "source_fingerprint": source_identity["source_fingerprint"],
                  "input_source": str(state), "input_seal": online_seal,
                  "output_root": str((tmp_path / "outputs" / case["case_id"] / arm).resolve()),
                  "arm": arm, "runtime_read_roots": [str(runtime.resolve())],
                  "support_sha256": {"fixture": "e" * 64}}
            path = tmp_path / "plans" / case["case_id"] / f"{arm}.json"
            jobs.append({"arm": arm, "plan_path": str(path.resolve()),
                         "plan_sha256": _write(path, wp)})
        pairs.append({"case_id": case["case_id"],
                      "candidate_id": case["candidate_id"], "jobs": jobs})
    return {
        "schema": comparison.SCHEMA, "comparison_id": "fixture-v3",
        "output_root": f"/Users/fixture-online-v3-{tmp_path.name}",
        "worker_path": str(worker_file.resolve()),
        "worker_sha256": v2._sha(worker_file),
        "coordinator_sha256": v2._sha(comparison.__file__),
        "comparison_support_sha256": v2._sha(v2.__file__),
        "source_identity": source_identity, "runtime_identity": runtime_identity,
        "cohort": {"package_id": "arbitrary-cohort",
                   "model_input_manifest": str(model_path),
                   "model_input_manifest_sha256": model_sha,
                   "scalar_truth_manifest": str(truth_path),
                   "scalar_truth_manifest_sha256": truth_sha,
                   "operator_manifest": str(operator_path),
                   "operator_manifest_sha256": operator_sha},
        "pairs": pairs,
    }


def test_plan_keeps_complete_scalar_cohort_and_truth_out_of_worker(tmp_path, monkeypatch):
    plan = _fixture_plan(tmp_path, monkeypatch)
    _worker, cohort, pairs = comparison.validate_plan(plan)
    assert len(cohort["cases"]) == len(pairs) == 6
    assert [pair["case_id"] for pair in pairs] == [
        f"case-{index}" for index in range(6)]
    worker_text = "\n".join(Path(job["plan_path"]).read_text()
                              for pair in plan["pairs"] for job in pair["jobs"])
    assert "official_label" not in worker_text
    assert "harbor" not in worker_text
    assert "typed_adapter_eligibility" not in json.dumps(plan)


def test_plan_rejects_post_selection_and_harbor_tamper(tmp_path, monkeypatch):
    plan = _fixture_plan(tmp_path / "selection", monkeypatch)
    plan["pairs"].pop()
    with pytest.raises(ValueError, match="complete precommitted cohort"):
        comparison.validate_plan(plan)

    plan = _fixture_plan(tmp_path / "tamper", monkeypatch)
    truth = json.loads(Path(plan["cohort"]["scalar_truth_manifest"]).read_text())
    provenance = Path(truth["cases"][0]["provenance"]["path"])
    _write(provenance, {"verifier_result": {"reward": 0.0}})
    with pytest.raises(ValueError, match="provenance changed"):
        comparison.validate_plan(plan)


def _semantic(verdict="verified", strength="decisive", contract_text="requirement"):
    contract = {"text": contract_text, "severity": "must", "clause": "clause",
                "refutable_by_absence": False}
    evidence = {"outcome": "holds" if verdict != "refuted" else "refuted",
                "basis": "world", "strength": strength,
                "witness_kind": "value", "assumed_premises": [],
                "inputs": [{"logical_ref": "world:item", "provenance": "delivered",
                            "role": "subject", "content_sha256": "f" * 64}],
                "bridges": [],
                "judger": {"sufficient": "yes", "faithful": "yes",
                            "degradation": "n/a"},
                "mechanism": {
                    "kind": "agent",
                    "snapshot": [{"logical_ref": "proof.json",
                                  "content_sha256": "e" * 64}],
                    "oracle_adapter_request": {
                        "identity": "arbitrary@1", "arguments": {}},
                }}
    document = {"schema": v2.SEMANTIC_SCHEMA, "input_tree_seal": "a" * 64,
                "verdict": verdict, "verdict_strength": strength,
                "truth_eligible": True,
                "obligations": [{"contract": contract, "status": verdict,
                                 "evidence": [evidence]}]}
    obligation = {"id": hashlib.sha256(contract_text.encode()).hexdigest(),
                  **contract, "status": verdict, "reason": "", "basis": "world",
                  "strength": strength}
    row = {"id": "final", "kind": "final", "text": "fixed", "status": verdict,
           "effective_status": verdict, "effective_basis": "world",
           "effective_strength": strength, "open_premises": [],
           "premise_refuted": None, "reason": "", "basis": "world",
           "strength": strength, "obligations": [obligation]}
    ordered_contract = [{"ordinal": 0, "id": obligation["id"], **contract,
                         "premise_scope": [], "required_basis": "world"}]
    quality = {"ordered_contract": ordered_contract, "ordered_outcomes": [],
               "certificate": {"final": "final", "verdict": verdict,
                               "verdict_strength": strength, "slice": [row]},
               "score": {"rank_key": [4, 0]}}
    return document, quality


def _job(verdict="verified", strength="decisive", contract_text="requirement"):
    semantic, quality = _semantic(verdict, strength, contract_text)
    return {"worker_complete": True, "accounting_complete": True,
            "transport_audit_complete": True,
            "scalar_truth": {"correct": True}, "semantic_evidence": semantic,
            "quality_view": quality,
            "architecture": {"roles_observed": [
                "obligations", "critic", "oracles", "discharge", "judger"],
                "cache_events": {
                    "method_cache_hit": 0,
                    "method_cache_direction_audit": 0,
                    "contract_cache_hit": 0,
                    "contract_cache_published": 0,
                },
                "final_obligation_lineage": [{
                    "obligation": "1" * 64, "status": verdict,
                    "commit_seq": 1, "critic_pass_seq": 2,
                    "compiled_seq": 3, "oracle_seq": 4,
                    "execution_kind": "agent", "adapter_identity": None,
                    "execution_seq": None, "execution_artifact": "2" * 64,
                    "judger_seq": 5, "method_bound_seq": 6,
                    "discharge_seq": 7, "binding_artifact": "3" * 64,
                    "certificate_seq": 8, "stages_skipped_reason": None,
                }]}}


def test_scalar_truth_requires_both_terminal_absolute_results():
    positive = {"label": 1.0, "identity": "i"}
    semantic, _quality = _semantic("verified")
    assert comparison.scalar_truth_evaluation(
        positive, "a" * 64, semantic, "verified")["correct"] is True
    semantic["verdict"] = "unverified"; semantic["truth_eligible"] = False
    failed = comparison.scalar_truth_evaluation(
        positive, "a" * 64, semantic, "unverified")
    assert failed["correct"] is False
    assert "nonterminal" in failed["findings"]

    negative = {"label": 0.0, "identity": "n"}
    semantic, _quality = _semantic("refuted")
    assert comparison.scalar_truth_evaluation(
        negative, "a" * 64, semantic, "refuted")["correct"] is True


def test_semantic_no_weaker_is_conservative_and_cache_free():
    same = comparison.semantic_no_weaker(_job(), _job())
    assert same["proven"] is True
    assert same["full_semantic_equal"] is True

    changed_contract = comparison.semantic_no_weaker(
        _job(), _job(contract_text="different requirement"))
    assert changed_contract["proven"] is False
    assert "not_exactly_alignable" in " ".join(changed_contract["reasons"])

    weaker = comparison.semantic_no_weaker(
        _job(strength="decisive"), _job(strength="bounded"))
    assert weaker["proven"] is False
    assert any("weaker" in reason for reason in weaker["reasons"])

    typed = _job()
    typed["semantic_evidence"]["obligations"][0]["evidence"][0]["mechanism"] \
        = {"kind": "typed_adapter", "adapter_identity": "arbitrary@1",
           "arguments": {}, "consumed": [], "observation": {}}
    changed_mechanism = comparison.semantic_no_weaker(_job(), typed)
    assert changed_mechanism["proven"] is True
    assert changed_mechanism["full_semantic_equal"] is False

    cached = _job()
    cached["architecture"]["cache_events"]["method_cache_hit"] = 1
    assert comparison.semantic_no_weaker(_job(), cached) == {
        "proven": False, "status": "unproven",
        "reasons": ["worker_or_scalar_truth_incomplete"]}


def test_capability_is_posthoc_and_cannot_rescue_quality():
    control, treatment = _job(), _job()
    treatment.update({"typed_adapter_exercised": False,
                      "typed_adapter_bound": False, "typed_transport": {}})
    result = comparison.compare_pair(control, treatment)
    assert result["strict_quality_pass"] is True
    assert result["capability_observation"] == {
        "cell": "adapter_not_exercised", "descriptive_only": True,
        "used_for_selection": False, "typed_adapter_exercised": False,
        "typed_adapter_bound": False, "typed_transport": {}}

    wrong_control = _job()
    wrong_control["scalar_truth"] = {"correct": False}
    treatment.update({"typed_adapter_exercised": True,
                      "typed_adapter_bound": True,
                      "typed_transport": {"packet_attached_attempts": 1}})
    failed = comparison.compare_pair(wrong_control, treatment)
    assert failed["both_arms_official_label_correct"] is False
    assert failed["strict_quality_pass"] is False


def test_transport_fallback_is_an_observation_not_selection_failure():
    transport = {
        "enabled": True, "protocol": v2.TYPED_PACKET_PROTOCOL,
        "boundary": v2.TYPED_BOUNDARY,
        "packet_eligible_bound_methods": 1, "packet_attached_attempts": 1,
        "restricted_attempted": True, "ordinary_judger_attempts": 1,
        "raw_prefetch_attached_attempts": 0,
        "restricted_fallback_attempted": True,
        "packet_unavailable_fallback": False,
    }
    audit = {"typed_judger_transport": transport, "native_attempts": 2,
             "observer_finished_attempts": 2}
    assert comparison._transport_integrity(audit, "treatment") is True
    job = _job(); job.update({"typed_adapter_exercised": True,
                              "typed_adapter_bound": True,
                              "typed_transport": transport})
    assert comparison.capability_observation(job)["cell"] \
        == "packet_attempted_fallback"


def test_general_builder_emits_all_arms_without_task_specific_truth(
        tmp_path, monkeypatch):
    (model_path, model_sha, truth_path, truth_sha,
     package_operator, package_operator_sha) = _cohort(tmp_path)
    source = tmp_path / "source"; source.mkdir()
    source_worker = source / "experiments" / "terminal_bench" \
        / "run_full_online_worker.py"
    source_worker.parent.mkdir(parents=True)
    source_worker.write_text("# pinned fixture worker\n")
    runtime = tmp_path / "runtime"; runtime.mkdir()
    cli = runtime / "codex"; cli.write_text("fixture"); cli.chmod(0o700)
    monkeypatch.setattr(builder.v3, "validate_plan", lambda value: value)
    source_binding_order = []
    monkeypatch.setattr(
        builder.v2, "_source_fingerprint",
        lambda _root: source_binding_order.append("fingerprint") or "c" * 64)
    fake_worker = SimpleNamespace(
        SCHEMA=comparison.WORKER_SCHEMA, SUPPORT_FILES=(),
        input_snapshot=lambda root: {"seal": hashlib.sha256(
            str(root).encode()).hexdigest()},
    )
    monkeypatch.setattr(
        builder.v2, "_load_worker",
        lambda *_args: source_binding_order.append("import") or fake_worker)
    operator = {
        "schema": builder.SCHEMA, "comparison_id": "builder-fixture",
        "model_input_manifest": str(model_path),
        "model_input_manifest_sha256": model_sha,
        "scalar_truth_manifest": str(truth_path),
        "scalar_truth_manifest_sha256": truth_sha,
        "cohort_operator_manifest": str(package_operator),
        "cohort_operator_manifest_sha256": package_operator_sha,
        "plan_output_root": str((tmp_path / "built-plans").resolve()),
        "comparison_output_root": str((tmp_path / "comparison-out").resolve()),
        "worker_outputs_root": str((tmp_path / "worker-outs").resolve()),
        "source_root": str(source.resolve()), "source_commit": "a" * 40,
        "source_fingerprint": "c" * 64,
        "runtime_read_roots": [str(runtime.resolve())],
        "runtime_identity": {
            "codex_cli_path": str(cli.resolve()),
            "codex_cli_realpath": str(cli.resolve()),
            "codex_cli_sha256": v2._sha(cli), "codex_cli_version": "fixture",
            "runtime_read_root_fingerprints": {str(runtime.resolve()): "b" * 64}},
        "worker_path": str(source_worker.resolve()),
        "coordinator_path": str(Path(comparison.__file__).resolve()),
    }
    result = builder.prepare(operator)
    plan = json.loads(Path(result["comparison_plan"]).read_text())
    assert result["model_calls"] == 0
    assert result["task_specific_truth_fields"] == 0
    assert len(plan["pairs"]) == 6
    assert {job["arm"] for pair in plan["pairs"] for job in pair["jobs"]} \
        == {"control", "treatment"}
    encoded = json.dumps(plan)
    assert "typed_adapter_eligibility" not in encoded
    assert "required_evidence" not in encoded
    assert plan["cohort"]["operator_manifest"] == str(package_operator)
    assert plan["cohort"]["operator_manifest_sha256"] == package_operator_sha
    assert source_binding_order[:2] == ["fingerprint", "fingerprint"]
    assert source_binding_order.index("import") > source_binding_order.index(
        "fingerprint")


def test_retained_operator_authority_and_crossed_family_polarity_are_required(
        tmp_path, monkeypatch):
    plan = _fixture_plan(tmp_path / "authority", monkeypatch)
    truth_path = Path(plan["cohort"]["scalar_truth_manifest"])
    truth = json.loads(truth_path.read_text())
    truth["operator_manifest_sha256"] = "f" * 64
    plan["cohort"]["scalar_truth_manifest_sha256"] = _write(truth_path, truth)
    with pytest.raises(ValueError, match="unbound scalar truth"):
        comparison.validate_plan(plan)

    plan = _fixture_plan(tmp_path / "polarity", monkeypatch)
    truth_path = Path(plan["cohort"]["scalar_truth_manifest"])
    operator_path = Path(plan["cohort"]["operator_manifest"])
    truth = json.loads(truth_path.read_text())
    operator = json.loads(operator_path.read_text())
    operator_by_case = {row["case_id"]: row for row in operator["cases"]}
    for row in truth["cases"]:
        label = 0.0 if row["task_family"] == "family-c" else 1.0
        provenance = row["provenance"]
        provenance["sha256"] = _write(
            Path(provenance["path"]), {"verifier_result": {"reward": label}})
        row["label"] = label
        row["identity"] = comparison._truth_identity(label, provenance)
        operator_by_case[row["case_id"]]["truth"] = {
            key: row[key] for key in ("label", "identity", "provenance")}
    operator_sha = _write(operator_path, operator)
    truth["operator_manifest_sha256"] = operator_sha
    plan["cohort"]["operator_manifest_sha256"] = operator_sha
    plan["cohort"]["scalar_truth_manifest_sha256"] = _write(truth_path, truth)
    with pytest.raises(ValueError, match="family/polarity"):
        comparison.validate_plan(plan)


def test_every_retained_family_requires_both_polarities(tmp_path, monkeypatch):
    plan = _fixture_plan(tmp_path, monkeypatch)
    model_path = Path(plan["cohort"]["model_input_manifest"])
    truth_path = Path(plan["cohort"]["scalar_truth_manifest"])
    operator_path = Path(plan["cohort"]["operator_manifest"])
    model = json.loads(model_path.read_text())
    truth = json.loads(truth_path.read_text())
    operator = json.loads(operator_path.read_text())
    index = 0
    model["cases"][index]["task_family"] = "family-d"
    truth["cases"][index]["task_family"] = "family-d"
    operator["cases"][index]["task_family"] = "family-d"
    model_path.write_text(json.dumps(model))
    operator_path.write_text(json.dumps(operator))
    truth["model_input_manifest_sha256"] = v2._sha(model_path)
    truth["operator_manifest_sha256"] = v2._sha(operator_path)
    truth_path.write_text(json.dumps(truth))
    plan["cohort"].update({
        "model_input_manifest_sha256": v2._sha(model_path),
        "operator_manifest_sha256": v2._sha(operator_path),
        "scalar_truth_manifest_sha256": v2._sha(truth_path),
    })

    with pytest.raises(ValueError, match="family/polarity"):
        comparison.validate_plan(plan)


def test_no_weaker_binds_complete_contract_and_matches_evidence_injectively():
    control, treatment = _job(), _job()
    treatment["quality_view"]["ordered_contract"][0]["required_basis"] = "record"
    assert comparison.semantic_no_weaker(control, treatment)["proven"] is False

    control, treatment = _job(), _job()
    second = json.loads(json.dumps(
        control["semantic_evidence"]["obligations"][0]["evidence"][0]))
    second["mechanism"]["snapshot"][0]["logical_ref"] = "other-proof.json"
    control["semantic_evidence"]["obligations"][0]["evidence"].append(second)
    assert comparison.semantic_no_weaker(control, treatment)["proven"] is False

    treatment = _job()
    treatment["semantic_evidence"]["obligations"][0]["evidence"][0][
        "mechanism"] = {"kind": "future_unknown"}
    assert comparison.semantic_no_weaker(_job(), treatment)["proven"] is False

    treatment = _job()
    del treatment["semantic_evidence"]["obligations"][0]["evidence"][0]["strength"]
    assert comparison.semantic_no_weaker(_job(), treatment)["proven"] is False


@pytest.mark.parametrize("false_zero", [None, False, "", [], {}])
def test_cache_audit_requires_exact_integer_zero(false_zero):
    treatment = _job()
    treatment["architecture"]["cache_events"]["method_cache_hit"] = false_zero
    assert comparison.semantic_no_weaker(_job(), treatment)["proven"] is False


@pytest.mark.parametrize("arm", ["control", "treatment"])
@pytest.mark.parametrize("damage", ["critic", "oracle", "judger", "executor"])
def test_each_arm_requires_complete_online_architecture(damage, arm):
    control, treatment = _job(), _job()
    target = control if arm == "control" else treatment
    if damage == "executor":
        target["architecture"]["final_obligation_lineage"][0][
            "execution_artifact"] = None
    else:
        role = {"critic": "critic", "oracle": "oracles", "judger": "judger"}[
            damage]
        target["architecture"]["roles_observed"].remove(role)
    result = comparison.semantic_no_weaker(control, treatment)
    assert result["proven"] is False
    assert result["reasons"] == ["worker_or_scalar_truth_incomplete"]


def test_typed_coverage_gates_ten_x_without_selecting_candidates():
    reports = [{"capability_observation": comparison.capability_observation(_job())}
               for _ in range(2)]
    assert comparison._typed_mechanism_coverage(reports)[
        "acceptance_coverage_complete"] is False

    eligible = _job()
    eligible.update({
        "typed_adapter_exercised": True, "typed_adapter_bound": True,
        "typed_transport": {"packet_eligible_bound_methods": 1,
                            "packet_attached_attempts": 1,
                            "restricted_attempted": True},
    })
    ineligible = _job()
    ineligible.update({"typed_adapter_exercised": False,
                       "typed_adapter_bound": False,
                       "typed_transport": {
                           "packet_eligible_bound_methods": 0,
                           "packet_attached_attempts": 0,
                           "restricted_attempted": False}})
    reports = [
        {"capability_observation": comparison.capability_observation(eligible)},
        {"capability_observation": comparison.capability_observation(ineligible)},
    ]
    coverage = comparison._typed_mechanism_coverage(reports)
    assert coverage["acceptance_coverage_complete"] is True
    assert coverage["posthoc_gate_does_not_select_candidates"] is True


def test_run_reports_stratum_ratios_and_coverage_blocks_ten_x(
        tmp_path, monkeypatch):
    pairs = []
    for index in range(2):
        pairs.append({
            "case_id": f"case-{index}", "candidate_id": f"candidate-{index}",
            "task_family": f"family-{index}",
            "truth": {"label": float(index)},
            "jobs": [{"arm": "control"}, {"arm": "treatment"}],
        })
    monkeypatch.setattr(comparison, "preflight", lambda _plan: {"model_calls": 0})
    monkeypatch.setattr(
        comparison, "validate_plan", lambda _plan: (None, {"cases": pairs}, pairs))

    def run(_plan, _worker, _output, pair, job, _ordinal):
        result = _job()
        result.update({"case_id": pair["case_id"],
                       "candidate_id": pair["candidate_id"],
                       "arm": job["arm"],
                       "typed_adapter_exercised": False,
                       "typed_adapter_bound": False,
                       "typed_transport": {
                           "packet_eligible_bound_methods": 0,
                           "packet_attached_attempts": 0,
                           "restricted_attempted": False}})
        return result

    monkeypatch.setattr(comparison, "_run_job", run)
    monkeypatch.setattr(
        comparison.v2, "_accounting_summary",
        lambda jobs, expected_jobs=None: {
            "complete": len(jobs) == expected_jobs,
            "control_over_treatment": 20.0})
    monkeypatch.setattr(
        comparison.v2, "_latency",
        lambda jobs, expected: {"complete": len(jobs) == expected,
                                "control_over_treatment": 20.0})
    result = comparison.run_comparison({"output_root": str(tmp_path / "out")})
    assert result["strict_quality_acceptance"] is True
    assert result["typed_mechanism_coverage"][
        "acceptance_coverage_complete"] is False
    assert result["ten_x_strict_quality_acceptance"] is False
    assert result["capability_strata_used_for_selection"] is False
    assert result["capability_strata_used_for_ten_x_coverage_gate"] is True
    assert len(result["per_family"]) == 2
    assert len(result["per_capability_stratum"]) == 1
    assert all(row["cost"]["control_over_treatment"] == 20.0
               and row["latency"]["control_over_treatment"] == 20.0
               for row in [*result["per_family"],
                           *result["per_capability_stratum"]])


def test_main_rejects_ambiguous_mode_and_exits_two_on_failed_acceptance(
        monkeypatch, capsys):
    monkeypatch.setattr(comparison.v2, "_read_json", lambda *_args: ({}, "d"))
    with pytest.raises(SystemExit):
        comparison.main(["--plan", "p", "--plan-sha256", "d",
                         "--preflight", "--execute"])
    monkeypatch.setattr(
        comparison, "run_comparison",
        lambda _plan: {"ten_x_strict_quality_acceptance": False})
    assert comparison.main([
        "--plan", "p", "--plan-sha256", "d", "--execute",
        "--confirm-paid", "GPT-5.6-Luna",
    ]) == 2
    assert "false" in capsys.readouterr().out


def test_interrupted_worker_retains_known_spawn_wall(tmp_path, monkeypatch):
    class Process:
        pid = 123
        returncode = None
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt()
            self.returncode = -15
            return self.returncode

    monkeypatch.setattr(comparison.subprocess, "Popen", lambda *_a, **_k: Process())
    monkeypatch.setattr(comparison.os, "killpg", lambda *_a: None)
    plan_path = tmp_path / "worker-plan.json"
    plan_path.write_text("{}\n")
    job = {"arm": "control", "plan_path": str(plan_path),
           "plan_sha256": v2._sha(plan_path),
           "worker_plan": {"output_root": str(tmp_path / "worker-output")}}
    result = comparison._run_job(
        {"worker_path": str(tmp_path / "worker.py")}, None,
        tmp_path, {"case_id": "case", "candidate_id": "candidate"}, job, 1)
    assert result["failure_wall_retained"] is True
    assert type(result["spawn_to_exit_wall_seconds"]) is float
    assert result["spawn_to_exit_wall_seconds"] >= 0
