"""No-model tests for the task-agnostic offline eval package builder."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from experiments.terminal_bench import prepare_general_eval_inputs as builder
from verifier import offline


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path, name: str, content: str) -> Path:
    state = tmp_path / "sources" / name / "gv"
    (state / "world" / "app").mkdir(parents=True)
    (state / "world" / "app" / "answer.txt").write_text(content)
    (state / "world_manifest.json").write_text(json.dumps({
        "captured_roots": ["/app"],
    }))
    (state / "task.md").write_text(f"Inspect opaque candidate {content}.")
    (state / "hook_feed.jsonl").write_text("")
    (state / "final.txt").write_text(content)
    offline.write_immutable_input_anchor(str(state))
    return state


def _case(tmp_path: Path, index: int, family: str, label: float,
          *, content: str | None = None) -> dict:
    state = _source(tmp_path, f"source-{index}", content or f"value-{index}")
    truth_path = tmp_path / "gold" / f"result-{index}.json"
    truth_path.parent.mkdir(exist_ok=True)
    truth_path.write_text(json.dumps({
        "verifier_result": {"rewards": {"reward": label}},
        "unrelated": f"coordinator-secret-{index}",
    }))
    provenance = {
        "kind": "harbor_result_json",
        "path": str(truth_path.resolve()),
        "sha256": _sha(truth_path),
        "json_path": ["verifier_result", "rewards", "reward"],
    }
    return {
        "case_id": f"case-{index}",
        "task_family": family,
        "candidate_id": f"opaque-candidate-{index}",
        "source_state": str(state.resolve()),
        "input_seal": offline.verify_immutable_input_anchor(str(state)),
        "truth": {
            "label": label,
            "identity": builder.truth_identity(float(label), provenance),
            "provenance": provenance,
        },
    }


def _manifest(tmp_path: Path, cases: list[dict]) -> tuple[Path, dict]:
    value = {
        "schema": builder.OPERATOR_SCHEMA,
        "package_id": "opaque-general-eval",
        "model_input_root": str((tmp_path / "model-inputs").resolve()),
        "truth_root": str((tmp_path / "coordinator-truth").resolve()),
        "cases": cases,
    }
    path = tmp_path / "operator.json"
    path.write_text(json.dumps(value, sort_keys=True))
    return path, value


def _read(path: Path):
    return json.loads(path.read_text())


def test_builds_three_arbitrary_families_with_separate_positive_negative_truth(tmp_path):
    cases = [
        _case(tmp_path, 1, "arbitrary-family-alpha", 1),
        _case(tmp_path, 2, "arbitrary-family-alpha", 0),
        _case(tmp_path, 3, "opaque.family.beta", 1),
        _case(tmp_path, 4, "third family name", 0),
    ]
    manifest_path, manifest = _manifest(tmp_path, cases)
    summary = builder.prepare(manifest_path.resolve())

    model_root = Path(manifest["model_input_root"])
    truth_root = Path(manifest["truth_root"])
    model_manifest = _read(model_root / "input-manifest.json")
    truth_manifest = _read(truth_root / "truth-manifest.json")
    assert summary["cases"] == 4
    assert {row["task_family"] for row in model_manifest["cases"]} == {
        "arbitrary-family-alpha", "opaque.family.beta", "third family name",
    }
    assert [row["label"] for row in truth_manifest["cases"]] == [1, 0, 1, 0]
    assert "label" not in model_manifest["cases"][0]
    assert "provenance" not in model_manifest["cases"][0]
    assert stat.S_IMODE(truth_root.stat().st_mode) == 0o700
    assert stat.S_IMODE((truth_root / "truth-manifest.json").stat().st_mode) == 0o600

    allowed = {*offline.IMMUTABLE_INPUT_PATHS, offline.IMMUTABLE_INPUT_SEAL_FILE}
    for source_case, published in zip(cases, model_manifest["cases"]):
        destination = model_root / published["state"]
        assert {path.name for path in destination.iterdir()} <= allowed
        assert offline.verify_immutable_input_anchor(str(destination)) == \
            source_case["input_seal"]
        assert not (destination / "score.json").exists()
        assert not (destination / "events.jsonl").exists()

    visible = b"".join(
        path.read_bytes() for path in sorted(model_root.rglob("*")) if path.is_file())
    for case in cases:
        assert case["truth"]["identity"].encode() not in visible
        assert case["truth"]["provenance"]["path"].encode() not in visible
    assert b"harbor_result_json" not in visible
    assert b'"label"' not in visible
    assert hashlib.sha256((model_root / "input-manifest.json").read_bytes()).hexdigest() \
        == truth_manifest["model_input_manifest_sha256"]


@pytest.mark.parametrize("duplicate", ["candidate", "seal"])
def test_rejects_duplicate_candidate_or_input_seal_before_writes(tmp_path, duplicate):
    first = _case(tmp_path, 1, "family-one", 1, content="same-content")
    second = _case(tmp_path, 2, "family-two", 0, content="different-content")
    if duplicate == "candidate":
        second["candidate_id"] = first["candidate_id"]
    else:
        duplicate_state = _source(tmp_path, "same-seal", "same-content")
        second["source_state"] = str(duplicate_state.resolve())
        second["input_seal"] = offline.verify_immutable_input_anchor(
            str(duplicate_state))
        assert second["input_seal"] == first["input_seal"]
    path, value = _manifest(tmp_path, [first, second])
    with pytest.raises(builder.PackageError, match=f"duplicate .*{duplicate}"):
        builder.prepare(path.resolve())
    assert not Path(value["model_input_root"]).exists()
    assert not Path(value["truth_root"]).exists()


@pytest.mark.parametrize("target", ["source", "truth"])
def test_rejects_tampered_source_or_external_truth_before_writes(tmp_path, target):
    case = _case(tmp_path, 1, "unrecognized-family", 1)
    path, value = _manifest(tmp_path, [case])
    if target == "source":
        Path(case["source_state"], "task.md").write_text("tampered")
    else:
        Path(case["truth"]["provenance"]["path"]).write_text(
            '{"verifier_result":{"rewards":{"reward":0}}}')
    with pytest.raises((builder.PackageError, offline.ImmutableInputError)):
        builder.prepare(path.resolve())
    assert not Path(value["model_input_root"]).exists()
    assert not Path(value["truth_root"]).exists()


def test_rejects_source_destination_overlap_and_symlink_alias(tmp_path):
    case = _case(tmp_path, 1, "anything", 1)
    path, value = _manifest(tmp_path, [case])
    value["model_input_root"] = str(Path(case["source_state"], "new-package"))
    path.write_text(json.dumps(value))
    with pytest.raises(builder.PackageError, match="disjoint"):
        builder.prepare(path.resolve())

    path, value = _manifest(tmp_path, [case])
    alias = tmp_path / "source-alias"
    alias.symlink_to(case["source_state"], target_is_directory=True)
    value["cases"][0]["source_state"] = str(alias.absolute())
    path.write_text(json.dumps(value))
    with pytest.raises(builder.PackageError, match="symlinks or aliases"):
        builder.prepare(path.resolve())


@pytest.mark.parametrize("fault", ["missing", "wrong_digest", "wrong_label", "wrong_identity"])
def test_rejects_missing_or_invalid_official_truth_provenance(tmp_path, fault):
    case = _case(tmp_path, 1, "family", 1)
    provenance = case["truth"]["provenance"]
    if fault == "missing":
        case["truth"]["provenance"] = {}
    elif fault == "wrong_digest":
        provenance["sha256"] = "0" * 64
    elif fault == "wrong_label":
        case["truth"]["label"] = 0
        case["truth"]["identity"] = builder.truth_identity(0.0, provenance)
    else:
        case["truth"]["identity"] = "f" * 64
    path, value = _manifest(tmp_path, [case])
    with pytest.raises(builder.PackageError):
        builder.prepare(path.resolve())
    assert not Path(value["model_input_root"]).exists()
    assert not Path(value["truth_root"]).exists()


def test_harbor_reward_text_is_supported_without_generalizing_provenance(tmp_path):
    case = _case(tmp_path, 1, "family", 0)
    source = tmp_path / "gold" / "reward.txt"
    source.write_text("0\n")
    provenance = {
        "kind": "harbor_reward_txt",
        "path": str(source.resolve()),
        "sha256": _sha(source),
    }
    case["truth"] = {
        "label": 0,
        "identity": builder.truth_identity(0.0, provenance),
        "provenance": provenance,
    }
    path, _value = _manifest(tmp_path, [case])
    assert builder.prepare(path.resolve())["cases"] == 1
