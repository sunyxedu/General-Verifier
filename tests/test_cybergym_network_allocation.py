"""Deterministic network allocation fixtures: no Docker or model calls."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ipaddress
import json
from types import SimpleNamespace
import threading

import pytest

from experiments.cybergym import run


class FakeDocker:
    def __init__(self, subnets=(), *, create_errors=()):
        self.subnets = list(subnets)
        self.create_errors = iter(create_errors)
        self.calls = []
        self.concurrent_creates = 0
        self.max_concurrent_creates = 0
        self.lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        assert args[0] == "network", "allocation must never alter containers, images, or volumes"
        if args[1] == "ls":
            return SimpleNamespace(returncode=0, stdout="".join(
                f"{index + 1:012x}\n" for index in range(len(self.subnets))), stderr="")
        if args[1] == "inspect":
            return SimpleNamespace(returncode=0, stdout=json.dumps([
                {"IPAM": {"Config": [{"Subnet": self.subnets[int(identifier, 16) - 1]}]}}
                for identifier in args[2:]]), stderr="")
        assert args[1] == "create"
        assert "--internal" in args and "--subnet" in args
        assert "--ipv6" not in args
        subnet = args[args.index("--subnet") + 1]
        with self.lock:
            self.concurrent_creates += 1
            self.max_concurrent_creates = max(self.max_concurrent_creates, self.concurrent_creates)
        try:
            error = next(self.create_errors, None)
            if callable(error):
                error = error(self, subnet)
            if error:
                return SimpleNamespace(returncode=1, stdout="", stderr=error)
            assert not any(ipaddress.ip_network(subnet).overlaps(ipaddress.ip_network(existing))
                           for existing in self.subnets if ":" not in existing)
            self.subnets.append(subnet)
            return SimpleNamespace(returncode=0, stdout="fixture-network-id\n", stderr="")
        finally:
            with self.lock:
                self.concurrent_creates -= 1


@pytest.fixture
def allocator(tmp_path, monkeypatch):
    # Real thread/flock synchronization, but a private temporary lock path.
    monkeypatch.setattr(run, "PROFILE", tmp_path / "formal")
    controller = object.__new__(run.Evaluation)
    controller.stop = threading.Event()
    return controller


def test_explicit_internal_subnet_bypasses_exhausted_default_pool(allocator, monkeypatch):
    docker = FakeDocker([f"172.{number}.0.0/16" for number in range(17, 32)])
    monkeypatch.setattr(run, "docker", docker)
    labels = ["--label", "gv.experiment=fixture", "--label", "gv.attempt_id=attempt"]
    assert allocator.create_isolated_network("fixture-owned-network", labels) == "10.240.0.0/24"
    create = next(args for args, _ in docker.calls if args[1] == "create")
    assert create == ("network", "create", "--internal", "--subnet", "10.240.0.0/24",
                      *labels, "fixture-owned-network")
    assert all(kwargs["check"] is False for _args, kwargs in docker.calls)


@pytest.mark.parametrize("occupied,expected", [
    (["10.240.0.0/16"], "10.241.0.0/24"),
    (["10.240.0.128/25"], "10.240.1.0/24"),
    (["10.240.0.0/25", "10.240.0.128/25", "10.240.1.0/24"], "10.240.2.0/24"),
    (["10.0.0.0/8", "172.16.0.0/16"], "172.17.0.0/24"),
    (["10.240.1.0/24"], "10.240.0.0/24"),
])
def test_allocator_skips_all_overlapping_prefix_lengths(occupied, expected):
    selected = run._available_private_subnet([ipaddress.IPv4Network(item) for item in occupied])
    assert str(selected) == expected
    assert selected.prefixlen == 24 and selected.is_private
    assert not any(selected.overlaps(ipaddress.IPv4Network(item)) for item in occupied)


def test_inventory_ignores_ipv6_but_retains_unlabelled_ipv4(allocator, monkeypatch):
    docker = FakeDocker(["fd00::/64", "10.240.0.0/23"])
    monkeypatch.setattr(run, "docker", docker)
    assert allocator.create_isolated_network("fixture", []) == "10.240.2.0/24"


def test_primary_pool_supports_1507_retained_networks_without_quadratic_scanning():
    pool = ipaddress.IPv4Network(run.ISOLATED_NETWORK_POOLS[0])
    assert pool.num_addresses // 256 >= 4096
    occupied = list(pool.subnets(new_prefix=24))[:1507]
    assert run._available_private_subnet(occupied) == list(pool.subnets(new_prefix=24))[1507]


def test_foreign_overlap_reinspects_and_jumps_new_foreign_allocation(allocator, monkeypatch):
    def external_allocation(docker, _subnet):
        docker.subnets.append("10.240.0.0/16")
        return "Error response from daemon: Pool overlaps with other one on this address space"
    docker = FakeDocker(create_errors=[external_allocation])
    monkeypatch.setattr(run, "docker", docker)
    assert allocator.create_isolated_network("fixture", []) == "10.241.0.0/24"
    assert len([args for args, _ in docker.calls if args[1] == "create"]) == 2


@pytest.mark.parametrize("error", [
    "all predefined address pools have been fully subnetted private-secret-marker",
    "permission denied private-secret-marker",
    "Cannot connect to Docker daemon private-secret-marker",
])
def test_non_overlap_daemon_failures_are_static_without_retry(allocator, monkeypatch, error):
    docker = FakeDocker(create_errors=[error])
    monkeypatch.setattr(run, "docker", docker)
    with pytest.raises(run.NetworkAllocationError, match="^isolated_network_create_failed$") as failure:
        allocator.create_isolated_network("fixture", [])
    assert "private-secret-marker" not in str(failure.value)
    assert len([args for args, _ in docker.calls if args[1] == "create"]) == 1


def test_persistent_foreign_races_stop_with_static_startup_failure(allocator, monkeypatch):
    docker = FakeDocker(create_errors=["Pool overlaps with other one on this address space"]
                        * run.MAX_NETWORK_ALLOCATION_RACES)
    monkeypatch.setattr(run, "docker", docker)
    with pytest.raises(run.NetworkAllocationError, match="^isolated_network_allocation_race_exhausted$"):
        allocator.create_isolated_network("fixture", [])
    assert len([args for args, _ in docker.calls if args[1] == "create"]) == run.MAX_NETWORK_ALLOCATION_RACES


def test_exhausted_private_capacity_is_static_and_never_creates_or_deletes(allocator, monkeypatch):
    docker = FakeDocker(["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"])
    monkeypatch.setattr(run, "docker", docker)
    with pytest.raises(run.NetworkAllocationError, match="^isolated_network_capacity_exhausted$"):
        allocator.create_isolated_network("fixture", [])
    assert not any(args[1] in ("create", "rm", "prune") for args, _ in docker.calls)


@pytest.mark.parametrize("inventory", [
    "private-secret-marker", "{}", '[{"IPAM":{"Config":{}}}]',
    '[{"IPAM":{"Config":[{"Subnet":"private-secret-marker"}]}}]',
    '[{"IPAM":{"Config":[{"Subnet":42}]}}]',
    '[{"IPAM":{"Config":[null]}}]',
])
def test_malformed_inventory_fails_closed_before_network_creation(allocator, monkeypatch, inventory):
    calls = []
    def docker(*args, **_kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="012345abcdef\n" if args[1] == "ls" else inventory,
                               stderr="")
    monkeypatch.setattr(run, "docker", docker)
    with pytest.raises(run.NetworkAllocationError, match="^isolated_network_inventory_invalid$") as failure:
        allocator.create_isolated_network("fixture", [])
    assert "private-secret-marker" not in str(failure.value)
    assert not any(args[1] == "create" for args in calls)


def test_stopped_allocator_never_inspects_or_creates(allocator, monkeypatch):
    allocator.stop.set()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("stopped startup must not call Docker")
    monkeypatch.setattr(run, "docker", forbidden)
    with pytest.raises(run.NetworkAllocationError, match="^isolated_network_startup_stopped$"):
        allocator.create_isolated_network("fixture", [])


def test_three_parallel_tasks_and_sibling_controllers_share_allocation_lock(allocator, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    started = [threading.Event() for _ in range(3)]
    def hold_first_create(_docker, _subnet):
        entered.set()
        assert release.wait(2), "fixture create gate must be released"
    docker = FakeDocker(create_errors=[hold_first_create])
    monkeypatch.setattr(run, "docker", docker)
    sibling = object.__new__(run.Evaluation)
    sibling.stop = threading.Event()
    flock_calls = []
    original_flock = run.fcntl.flock
    def flock(fd, mode):
        flock_calls.append(mode)
        return original_flock(fd, mode)
    monkeypatch.setattr(run.fcntl, "flock", flock)
    def allocate(index, controller):
        started[index].set()
        return controller.create_isolated_network(f"fixture-{index}", [])
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(allocate, index, controller)
                 for index, controller in enumerate([allocator, sibling, allocator])]
        try:
            assert all(event.wait(2) for event in started)
            assert entered.wait(2)
            # All callers have entered, but later callers cannot even inspect
            # while the first allocation transaction holds thread/flock locks.
            assert len([args for args, _ in docker.calls if args[1] == "ls"]) == 1
        finally:
            release.set()
        subnets = [job.result() for job in tasks]
    assert len(set(subnets)) == 3
    assert set(subnets) == {"10.240.0.0/24", "10.240.1.0/24", "10.240.2.0/24"}
    assert docker.max_concurrent_creates == 1
    assert flock_calls == [run.fcntl.LOCK_EX] * 3


def test_network_capacity_failure_prevents_volume_and_solver_startup(allocator, tmp_path, monkeypatch):
    allocator.experiment = "fixture-experiment"
    docker = FakeDocker(["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"])
    monkeypatch.setattr(run, "docker", docker)
    runtime = {"network": "fixture-owned-network", "sidecar": "fixture-owned-sidecar",
               "proxy": "fixture-owned-proxy", "volume": "fixture-owned-state", "token": "fixture-token"}
    with pytest.raises(run.NetworkAllocationError, match="^isolated_network_capacity_exhausted$"):
        allocator._create_runtime(tmp_path, tmp_path / "workspace", tmp_path / "private", runtime)
    assert all(args[:2] in (("network", "ls"), ("network", "inspect")) for args, _ in docker.calls)


@pytest.mark.parametrize("failed_operation", ["ls", "inspect"])
def test_inventory_daemon_error_is_static_and_never_creates(allocator, monkeypatch, failed_operation):
    calls = []
    def docker(*args, **_kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=int(args[1] == failed_operation),
                               stdout="012345abcdef\n", stderr="private-secret-marker")
    monkeypatch.setattr(run, "docker", docker)
    with pytest.raises(run.NetworkAllocationError, match="^isolated_network_inventory_unavailable$"):
        allocator.create_isolated_network("fixture", [])
    assert not any(args[1] == "create" for args in calls)


def test_task_capacity_failure_keeps_official_score_unknown_and_partial_cleanup_scoped(
        allocator, tmp_path, monkeypatch):
    allocator.root, allocator.results = tmp_path, {}
    allocator.lock = threading.RLock()
    allocator.frozen_commit = "fixture-pinned-commit"
    allocator.experiment = "fixture-experiment"
    docker = FakeDocker(["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"])
    monkeypatch.setattr(run, "docker", docker)
    monkeypatch.setattr(allocator, "_wait_for_disk", lambda *_args: None)
    monkeypatch.setattr(allocator, "pull", lambda *_args: ("fixture-cached-image", False))
    def prepare(_task, directory):
        private, workspace = directory / "private", directory / "workspace"
        workspace.mkdir()
        private.mkdir(mode=0o700)
        (private / "state").mkdir(mode=0o700)
        (private / "state/cybergym-observations").mkdir(mode=0o700)
        return workspace, private, "fixture task"
    monkeypatch.setattr(allocator, "prepare", prepare)
    cleanup = []
    monkeypatch.setattr(allocator, "cleanup_runtime", lambda *args: cleanup.append(args))
    def forbidden(*_args, **_kwargs):
        raise AssertionError("infrastructure capacity failure cannot start or score a solver")
    monkeypatch.setattr(allocator, "run_solver", forbidden)
    monkeypatch.setattr(allocator, "score", forbidden)
    state = allocator.run_task("arvo:fixture")
    assert state["official_pass"] is None
    assert state["outcome"] == "infrastructure_error"
    assert state["reason"] == "NetworkAllocationError: isolated_network_capacity_exhausted"
    assert len(cleanup) == 1
    runtime, directory, private = cleanup[0]
    assert runtime["network"].startswith("gv-roles-net-")
    assert directory.name == state["attempt_id"] and private == directory / "private"
    assert all(args[:2] in (("network", "ls"), ("network", "inspect")) for args, _ in docker.calls)
