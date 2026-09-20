from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

from verifier.core.model import content_hash
from verifier.profiles.cyber import evaluator


PYTHON = os.path.realpath(sys.executable)


def details(error):
    return json.loads(str(error).split(": ", 1)[1])


@pytest.fixture
def pinned(tmp_path):
    dependency = tmp_path / "private-config-token-sentinel.json"
    dependency.write_bytes(b"private-config-content-sentinel")
    spec = evaluator.EvaluatorSpec((PYTHON, "{candidate}"), pinned_files=(str(dependency),))
    return spec, dependency


def test_unchanged_pin_still_validates_and_fingerprint_is_unaltered(pinned):
    spec, _dependency = pinned
    original_identity = spec.identity
    original_id = spec.id
    spec.validate()
    spec.validate()
    assert spec.identity == original_identity
    assert spec.id == original_id == content_hash(original_identity)


@pytest.mark.parametrize("field_index, field", list(enumerate(evaluator._PIN_IDENTITY_FIELDS)))
def test_unchanged_content_identity_drift_remains_rejected_and_identified(pinned, monkeypatch,
                                                                       field_index, field):
    spec, dependency = pinned
    original_id = spec.id
    actual_pin = evaluator._file_pin

    def changed(path, executable=False):
        digest, identity = actual_pin(path, executable=executable)
        if path == str(dependency):
            changed_identity = list(identity)
            changed_identity[field_index] += 1
            return digest, tuple(changed_identity)
        return digest, identity

    monkeypatch.setattr(evaluator, "_file_pin", changed)
    with pytest.raises(evaluator.EvaluatorError, match="dependency pin changed") as failure:
        spec.validate()
    assert details(failure.value) == {"dependency_index": 0, "sha256_changed": False,
                                      "identity_changed": [field]}
    assert spec.id == original_id
    assert dependency.read_bytes() == b"private-config-content-sentinel"


def test_real_mtime_drift_reports_unchanged_content_but_remains_fail_closed(pinned):
    spec, dependency = pinned
    before = dependency.stat()
    os.utime(dependency, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    with pytest.raises(evaluator.EvaluatorError, match="dependency pin changed") as failure:
        spec.validate()
    diagnostic = details(failure.value)
    assert diagnostic["sha256_changed"] is False
    assert diagnostic["dependency_index"] == 0
    assert "mtime_ns" in diagnostic["identity_changed"]
    assert dependency.read_bytes() == b"private-config-content-sentinel"


def test_same_content_replacement_inode_is_not_silently_accepted(pinned, tmp_path):
    spec, dependency = pinned
    replacement = tmp_path / "replacement"
    replacement.write_bytes(dependency.read_bytes())
    os.replace(replacement, dependency)
    with pytest.raises(evaluator.EvaluatorError, match="dependency pin changed") as failure:
        spec.validate()
    assert details(failure.value)["sha256_changed"] is False
    assert "inode" in details(failure.value)["identity_changed"]


def test_real_content_change_reports_hash_change_and_is_still_rejected(pinned):
    spec, dependency = pinned
    dependency.write_bytes(b"different-private-secret-token")
    with pytest.raises(evaluator.EvaluatorError, match="dependency pin changed") as failure:
        spec.validate()
    diagnostic = details(failure.value)
    assert diagnostic["sha256_changed"] is True
    assert diagnostic["dependency_index"] == 0
    rendered = str(failure.value)
    for secret in (str(dependency), dependency.name, "private-config-content-sentinel",
                   "different-private-secret-token", hashlib.sha256(dependency.read_bytes()).hexdigest()):
        assert secret not in rendered


def test_dependency_index_identifies_second_file_without_names_or_host_paths(tmp_path):
    first, second = tmp_path / "private-first", tmp_path / "private-second"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    spec = evaluator.EvaluatorSpec((PYTHON, "{candidate}"),
                                   pinned_files=(str(first), str(second)))
    second.write_bytes(b"three")
    with pytest.raises(evaluator.EvaluatorError) as failure:
        spec.validate()
    assert details(failure.value)["dependency_index"] == 1
    assert first.name not in str(failure.value) and second.name not in str(failure.value)
    assert str(tmp_path) not in str(failure.value)


def test_executable_identity_drift_remains_rejected_with_safe_fields(pinned, monkeypatch):
    spec, _dependency = pinned
    actual_pin = evaluator._file_pin

    def changed(path, executable=False):
        digest, identity = actual_pin(path, executable=executable)
        if executable:
            changed_identity = list(identity)
            changed_identity[-1] += 1
            return digest, tuple(changed_identity)
        return digest, identity

    monkeypatch.setattr(evaluator, "_file_pin", changed)
    with pytest.raises(evaluator.EvaluatorError, match="executable pin changed") as failure:
        spec.validate()
    assert details(failure.value) == {"sha256_changed": False, "identity_changed": ["ctime_ns"]}
    assert PYTHON not in str(failure.value)


def test_real_executable_content_change_remains_rejected_without_secret_details(tmp_path):
    program = tmp_path / "private-executable-token-sentinel"
    program.write_bytes(b"#!/bin/sh\nexit 0\n")
    program.chmod(0o700)
    spec = evaluator.EvaluatorSpec((str(program), "{candidate}"))
    program.write_bytes(b"#!/bin/sh\nexit 1\n")
    with pytest.raises(evaluator.EvaluatorError, match="executable pin changed") as failure:
        spec.validate()
    assert details(failure.value)["sha256_changed"] is True
    assert str(program) not in str(failure.value) and program.name not in str(failure.value)
