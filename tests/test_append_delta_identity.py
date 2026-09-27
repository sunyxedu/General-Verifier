"""Repair protocol identity follows the actual Kernel/BindingGame snapshot."""
import pytest

from test_append_only_delta import INITIAL, PASS, _machine
from test_method_cache_directions import _game, _row, _obligation
from verifier.contract_replay import compilation_config_sha256, ReplaySetupError
from verifier.core.store import ClaimStore, TRIGGER_LAZY
from verifier.engine.kernel import Kernel
from verifier.engine.binding.method_cache import AcceptedMethodCache
from verifier.llm.schema import FixtureBackend


def test_method_keys_separate_protocols_and_ignore_later_environment_mutation(tmp_path, monkeypatch):
    backend = FixtureBackend({})
    monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "0")
    off, *_ = _game(tmp_path, "off", backend, AcceptedMethodCache(), lambda _world: None)
    off_key = off.method_cache_key(_row("same"), _obligation("same"))
    monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "1")
    on, *_ = _game(tmp_path, "on", backend, AcceptedMethodCache(), lambda _world: None)
    on_key = on.method_cache_key(_row("same"), _obligation("same"))
    assert off_key is not None and on_key is not None
    assert off_key != on_key
    assert off.method_cache_key(_row("same"), _obligation("same")) == off_key
    monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "0")
    assert on.method_cache_key(_row("same"), _obligation("same")) == on_key


def test_kernel_refuses_binder_with_a_different_frozen_protocol(tmp_path, monkeypatch):
    backend = FixtureBackend({})
    monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "0")
    game, artifacts, log, _ = _game(tmp_path, "state", backend, AcceptedMethodCache(), lambda _world: None)
    monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "1")
    with pytest.raises(ValueError, match="different append-repair"):
        Kernel(ClaimStore(trigger=TRIGGER_LAZY), log, backend, artifacts, binder=game)


@pytest.mark.parametrize("config", [
    {"append_only_deltas": False, "append_only_delta_protocol": None},
    {"append_only_deltas": True, "append_only_delta_protocol": "different-version"},
])
def test_replay_refuses_flag_or_protocol_mismatch_before_backend_admission(tmp_path, monkeypatch, config):
    machine, _, _ = _machine(tmp_path, monkeypatch, [INITIAL], [PASS], enabled=True)
    with pytest.raises(ReplaySetupError, match="append-repair protocol"):
        compilation_config_sha256(machine, config)
