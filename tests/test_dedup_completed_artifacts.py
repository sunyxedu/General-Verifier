from __future__ import annotations

import os
from pathlib import Path

import pytest

from experiments.common import candidate_seal
from experiments.common import dedup_completed_artifacts as dedup


def _job(tmp_path: Path) -> Path:
    job = tmp_path / "one-harbor-job"
    job.mkdir()
    (job / "config.json").write_text("{}\n", encoding="utf-8")
    (job / "lock.json").write_text("{}\n", encoding="utf-8")
    return job


def _artifact(
    job: Path, trial: str, content: bytes, *, completed: bool = True,
    mtime_ns: int = 1_700_000_000_000_000_000,
) -> Path:
    trial_dir = job / trial
    artifacts = trial_dir / "artifacts" / "app"
    artifacts.mkdir(parents=True)
    if completed:
        (trial_dir / "result.json").write_text("{}\n", encoding="utf-8")
    path = artifacts / "payload.bin"
    path.write_bytes(content)
    path.chmod(0o640)
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def test_apply_preserves_world_digest_and_hardlinks_identical_files(
    tmp_path,
):
    job = _job(tmp_path)
    payload = b"same immutable payload" * 128
    first = _artifact(job, "task__one", payload)
    second = _artifact(job, "task__two", payload)
    before = {
        first.parent: candidate_seal.tree_digest(first.parent),
        second.parent: candidate_seal.tree_digest(second.parent),
    }

    summary = dedup.deduplicate(job, min_bytes=1, apply=True)

    assert summary.completed_trials == 2
    assert summary.duplicate_files == 1
    assert summary.links_created == 1
    assert summary.logical_bytes_deduplicated == len(payload)
    assert os.stat(first).st_ino == os.stat(second).st_ino
    assert first.read_bytes() == payload == second.read_bytes()
    assert {
        root: candidate_seal.tree_digest(root) for root in before
    } == before


def test_dry_run_reports_duplicate_without_changing_inode(tmp_path):
    job = _job(tmp_path)
    first = _artifact(job, "task__one", b"duplicate")
    second = _artifact(job, "task__two", b"duplicate")
    inodes = (os.stat(first).st_ino, os.stat(second).st_ino)

    summary = dedup.deduplicate(job, min_bytes=1)

    assert summary.dry_run is True
    assert summary.duplicate_files == 1
    assert summary.links_created == 0
    assert (os.stat(first).st_ino, os.stat(second).st_ino) == inodes


def test_incomplete_symlink_and_metadata_mismatch_are_not_linked(
    tmp_path,
):
    job = _job(tmp_path)
    first = _artifact(job, "task__one", b"same")
    different_mtime = _artifact(
        job, "task__two", b"same", mtime_ns=1_700_000_000_000_000_001)
    incomplete = _artifact(job, "task__three", b"same", completed=False)
    symlink = first.parent / "alias.bin"
    symlink.symlink_to(first.name)

    summary = dedup.deduplicate(job, min_bytes=1, apply=True)

    assert summary.completed_trials == 2
    assert summary.duplicate_files == 0
    assert summary.links_created == 0
    assert os.stat(first).st_ino != os.stat(different_mtime).st_ino
    assert os.stat(first).st_ino != os.stat(incomplete).st_ino
    assert symlink.is_symlink()


def test_rejects_non_harbor_and_broad_targets(tmp_path):
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    with pytest.raises(dedup.DedupError, match="not a Harbor job"):
        dedup.deduplicate(ordinary)
    with pytest.raises(dedup.DedupError, match="broad target"):
        dedup.deduplicate(Path.home())
