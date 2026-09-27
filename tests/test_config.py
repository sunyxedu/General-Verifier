"""The process settings are parsed in one place, with the semantics each
setting always had."""
from __future__ import annotations

import pytest

from verifier.config import (
    ExtensionConfig,
    OnlineConfig,
    ParallelismConfig,
    VerifierConfig,
    launch_settings,
    max_concurrent_seats,
    scratch_cap_gb,
    script_timeout_seconds,
    seat_room_mb,
    server_lock_wait_seconds,
)


def test_extension_defaults_keep_only_typed_adapters_on():
    assert ExtensionConfig.from_env({}) == ExtensionConfig(
        typed_adapters=True, sufficiency_preaudit=False,
        typed_judger_packet=False, contract_cache=False, method_cache=False,
        append_only_deltas=False)


@pytest.mark.parametrize("name", [
    "GV_TYPED_ADAPTERS", "GV_TYPED_JUDGER_PACKET", "GV_CONTRACT_CACHE",
    "GV_METHOD_CACHE", "GV_APPEND_ONLY_DELTAS", "GV_SUFFICIENCY_PREAUDIT"])
def test_every_extension_switch_is_strict(name):
    with pytest.raises(ValueError, match=name):
        ExtensionConfig.from_env({name: "true"})


def test_typed_packet_without_typed_adapters_is_inconsistent():
    config = ExtensionConfig.from_env(
        {"GV_TYPED_ADAPTERS": "0", "GV_TYPED_JUDGER_PACKET": "1"})
    with pytest.raises(ValueError, match="requires typed adapters"):
        config.require_consistent()
    ExtensionConfig.from_env({"GV_TYPED_JUDGER_PACKET": "1"}) \
        .require_consistent()


def test_online_economies_are_on_unless_zero():
    assert OnlineConfig.from_env({}) == OnlineConfig(
        background_checks=True, final_precompile=True, screener=True,
        critic_rounds_nonfinal=0, unify_final=True,
        speculative_discharge=True)
    off = OnlineConfig.from_env({
        "GV_BACKGROUND_CHECKS": "0", "GV_FINAL_PRECOMPILE": "0",
        "GV_SCREENER": "0", "GV_UNIFY_FINAL": "0",
        "GV_SPECULATIVE_DISCHARGE": "0", "GV_CRITIC_ROUNDS_NONFINAL": "2"})
    assert not (off.background_checks or off.final_precompile or off.screener
                or off.unify_final or off.speculative_discharge)
    assert off.critic_rounds_nonfinal == 2
    assert OnlineConfig.from_env(
        {"GV_SCREENER": "no", "GV_CRITIC_ROUNDS_NONFINAL": "x"}) \
        == OnlineConfig.from_env({})
    assert OnlineConfig.from_env(
        {"GV_CRITIC_ROUNDS_NONFINAL": "-3"}).critic_rounds_nonfinal == 0


def test_parallelism_falls_back_and_floors_at_one():
    assert ParallelismConfig.from_env({}) == ParallelismConfig(
        parallel_obligations=8, parallel_claims=4, shared_dir=None,
        certificate_scope=None)
    parsed = ParallelismConfig.from_env({
        "GV_PARALLEL_OBLIGATIONS": "0", "GV_PARALLEL_CLAIMS": "many",
        "GV_SHARED_DIR": "", "GV_CERTIFICATE_SCOPE": "final"})
    assert parsed.parallel_obligations == 1
    assert parsed.parallel_claims == 4
    assert parsed.shared_dir is None
    assert parsed.certificate_scope == "final"


def test_verifier_config_groups_one_snapshot():
    environ = {"GV_METHOD_CACHE": "1", "GV_SCREENER": "0",
               "GV_PARALLEL_CLAIMS": "2"}
    config = VerifierConfig.from_env(environ)
    assert config.extensions.method_cache is True
    assert config.online.screener is False
    assert config.parallelism.parallel_claims == 2


def test_launch_settings_never_carry_credentials():
    assert launch_settings({
        "GV_SEAT_EFFORT": "max", "GV_API_KEY": "secret",
        "GV_AUTH_TOKEN": "secret", "CODEX_HOME": "/codex", "PATH": "/bin",
    }) == {"CODEX_HOME": "/codex", "GV_SEAT_EFFORT": "max"}


def test_call_time_limits_are_read_on_each_call(monkeypatch):
    for name in ("GV_MAX_CONCURRENT_SEATS", "GV_SEAT_ROOM_MB",
                 "GV_SCRATCH_CAP_GB", "GV_SCRIPT_TIMEOUT",
                 "GV_SERVER_LOCK_WAIT"):
        monkeypatch.delenv(name, raising=False)
    assert (max_concurrent_seats(), seat_room_mb(), scratch_cap_gb(),
            script_timeout_seconds(), server_lock_wait_seconds()) \
        == (12, 512, 6.0, 300.0, 30.0)
    monkeypatch.setenv("GV_MAX_CONCURRENT_SEATS", "3")
    monkeypatch.setenv("GV_SEAT_ROOM_MB", "bad")
    monkeypatch.setenv("GV_SCRATCH_CAP_GB", "0.5")
    monkeypatch.setenv("GV_SCRIPT_TIMEOUT", "bad")
    assert (max_concurrent_seats(), seat_room_mb(), scratch_cap_gb(),
            script_timeout_seconds()) == (3, 512, 0.5, 300.0)
    monkeypatch.setenv("GV_SERVER_LOCK_WAIT", "bad")
    with pytest.raises(ValueError):
        server_lock_wait_seconds()
