"""Real, offline MCP requests must not share a verifier's blocking thread.

Test watchdogs only release fixture thread Events. They are not task timeouts,
and these tests invoke no model, evaluator, Docker, or network service.
"""
from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import os
import threading
from types import SimpleNamespace

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent

from verifier.engine.kernel import Kernel
from verifier.core.store import TRIGGER_LAZY
from verifier.core.model import content_hash
from verifier.execution.discharge import DischargeResult
from verifier.llm.schema import FixtureBackend
from verifier.mcp_server import Service, _ToolDispatcher, build_server, _JOB_CONTRACT, _JOB_VERBS
from verifier.profiles import get_profile


CONTEXT = contextvars.ContextVar("mcp_dispatch_test_context", default=None)
CASES = [
    ("commit", {"label": "a", "kind": "assertion", "text": "fact", "deps": []},
     ["a", "assertion", "fact", [], None]),
    ("verify", {"label": "a"}, ["a"]),
    ("certificate", {"final_label": "f"}, ["f"]),
    ("status", {}, [None]),
    ("cyber_checkpoint", {"working_theory": "hypothesis", "next_actions": ["test"]},
     ["hypothesis", ["test"]]),
    ("cyber_register_candidate", {"path": "staged/input"}, ["staged/input", ""]),
    ("cyber_candidates", {}, []),
    ("cyber_evaluate_candidate", {"candidate_id": "subject"}, ["subject"]),
    ("cyber_finalize_candidate", {"candidate_id": "subject"}, ["subject", ""]),
    ("verification_job", {"job_id": "job"}, ["job"]),
    ("verification_jobs", {}, []),
    ("verification_retry", {"job_id": "job"}, ["job"]),
]
CONTRACT = {
    "commit": (("label", "kind", "text", "deps"), ("revises",)),
    "verify": (("label",), ()),
    "certificate": (("final_label",), ()),
    "status": ((), ("label",)),
    "cyber_checkpoint": (("working_theory", "next_actions"), ()),
    "cyber_register_candidate": (("path",), ("label",)),
    "cyber_candidates": ((), ()),
    "cyber_evaluate_candidate": (("candidate_id",), ()),
    "cyber_finalize_candidate": (("candidate_id",), ("destination",)),
    "verification_job": (("job_id",), ()),
    "verification_jobs": ((), ()),
    "verification_retry": (("job_id",), ()),
}
DESCRIPTIONS = {
    "commit": "Commit a claim (kind: assertion|effect|final). deps is a list of\n"
              "        labels of already-committed claims (including effect:<seq>). The claim\n"
              "        is VERIFIED BEFORE THIS RETURNS, so the call blocks for as long as its\n"
              "        checks take; the result carries the resulting status, and on anything\n"
              "        but `verified` the obligation that failed plus its counterexample.",
    "verify": "Verify a committed claim now; returns its status.",
    "certificate": "Certify a final claim: settle the dependency closure, freeze the\n"
                   "        certificate, and return the verdict plus a slice summary.",
    "status": "Query the status of one label or of all labels.",
    "cyber_checkpoint": "Retain exploration notes in the ledger. These are not verified claims.",
    "cyber_register_candidate": "Snapshot exact bytes from a staging file. Records identity, not correctness.",
    "cyber_candidates": "List immutable candidate identities; this tool makes no ranking or verdict.",
    "cyber_evaluate_candidate": "Run the configured task-authorized evaluator on the exact candidate.\n"
                                "                Returns retained observations and an ordinary Effect reference, not a certificate.",
    "cyber_finalize_candidate": "Authorize the candidate through GV, copy exact bytes, then verify the full task.\n"
                                "            Only a verified whole-task final and current exact delivery complete this operation.\n"
                                "            Candidate authorization alone is not completion; neither certificate asserts a score.",
    "verification_job": "Query an existing verification job without restarting it.\n"
                        "            A completed result is retained history; current delivery is authenticated separately.",
    "verification_jobs": "List recoverable verification jobs and their progress; starts no verification.",
    "verification_retry": "Explicitly retry a terminal, non-verified job; pending jobs are only reconnected.\n"
                          "            Changed inputs require resubmission. Never retry merely to recover a disconnected waiter.",
}


class FixtureService:
    def __init__(self, profile="default", *, state_dir=None, blocking=None,
                 failure=None, evaluator=True):
        self.profile = get_profile(profile)
        self.state_dir = state_dir
        self.evaluator = object() if evaluator else None
        self.blocking, self.failure = blocking, failure
        self.entered, self.release = threading.Event(), threading.Event()
        self.calls = []

    def invoke(self, name, *args):
        self.calls.append((name, args, threading.get_ident(), CONTEXT.get()))
        if name == self.blocking:
            self.entered.set()
            self.release.wait()
        if name == self.failure:
            raise ValueError("fixture-sensitive-exception-text")
        return {"method": name, "args": args, "unicode": "安全"}

    def __getattr__(self, name):
        return functools.partial(self.invoke, name)


def decode(result):
    assert not result.isError, result
    return json.loads(result.content[0].text)


async def completed_call(client, name, arguments):
    """These legacy tests assert settled semantics, not blocking transport.

    New job tests separately require that an in-flight request returns promptly.
    Polling here is fixture-only: never a timeout or retry of an adjudication.
    """
    reply = await client.call_tool(name, arguments)
    data = decode(reply)
    if data.get("pending") and data.get("job_id"):
        while data.get("pending"):
            await asyncio.sleep(0.005)
            data = decode(await client.call_tool("verification_job", {"job_id": data["job_id"]}))
        assert data.get("state") == "completed", data
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(data["result"]))])
    if name == "status" and data.get("status_snapshot") == "unavailable_while_verifying":
        while data.get("pending"):
            await asyncio.sleep(0.005)
            reply = await client.call_tool(name, arguments)
            data = decode(reply)
    return reply


def read_diagnostics(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("name,arguments,expected", CASES)
def test_real_mcp_dispatch_preserves_arguments_and_worker_thread(name, arguments, expected, tmp_path):
    service = FixtureService("cyber", state_dir=tmp_path)

    async def scenario():
        async with create_connected_server_and_client_session(build_server(service)) as client:
            result = decode(await completed_call(client, name, arguments))
            assert result == {"method": name, "args": expected, "unicode": "安全"}
        assert service.calls[0][2] != threading.get_ident()

    asyncio.run(scenario())
    events = read_diagnostics(tmp_path / "transport_calls.jsonl")
    assert [event["phase"] for event in events] == ["received", "invoked", "finished"]
    assert len({event["call_id"] for event in events}) == 1
    assert {event["tool"] for event in events} == {name}


@pytest.mark.parametrize("profile,name,arguments", [
    ("default", "commit", {"label": "a", "kind": "assertion", "text": "fact", "deps": []}),
    ("cyber", "cyber_finalize_candidate", {"candidate_id": "subject"}),
])
def test_blocking_verifier_keeps_real_mcp_ping_and_status_responsive(profile, name, arguments, tmp_path):
    service = FixtureService(profile, state_dir=tmp_path, blocking=name)
    # A broken synchronous implementation must fail without hanging pytest.
    watchdog = threading.Timer(3.0, service.release.set)

    async def scenario():
        async with create_connected_server_and_client_session(build_server(service)) as client:
            pending = asyncio.create_task(completed_call(client, name, arguments))
            watchdog.start()
            try:
                assert await asyncio.wait_for(asyncio.to_thread(service.entered.wait, 4.0), 4.5)
                assert not pending.done()
                await asyncio.wait_for(client.send_ping(), 1.0)
                result = decode(await asyncio.wait_for(completed_call(client, "status", {}), 1.0))
                assert result["method"] == "status"
                assert not service.release.is_set(), "transport waited for the blocked verifier"
                assert not pending.done()
                if profile == "cyber":
                    events = read_diagnostics(tmp_path / "transport_calls.jsonl")
                    assert [e["phase"] for e in events if e["tool"] == name] == ["received", "invoked"]
            finally:
                service.release.set()
                watchdog.cancel()
                await asyncio.wait_for(pending, 2.0)

    asyncio.run(scenario())


@pytest.mark.parametrize("profile,evaluator", [("default", True), ("cyber", False), ("cyber", True)])
def test_exact_tool_contract_metadata_and_profile_selection(profile, evaluator):
    service = FixtureService(profile, evaluator=evaluator)

    async def scenario():
        async with create_connected_server_and_client_session(build_server(service)) as client:
            tools = (await client.list_tools()).tools
        names = list(CONTRACT)
        if profile == "default":
            names = names[:4]
        elif not evaluator:
            names.remove("cyber_evaluate_candidate")
        assert [tool.name for tool in tools] == names
        for tool in tools:
            required, optional = CONTRACT[tool.name]
            properties = {}
            for field in (*required, *optional):
                spec = {"title": field.replace("_", " ").title(), "type": "string"}
                if field in ("deps", "next_actions"):
                    spec.update(type="array", items={"type": "string"})
                if field == "kind":
                    spec["enum"] = ["assertion", "effect", "final"]
                if field in optional:
                    spec["default"] = ""
                properties[field] = spec
            schema = {"properties": properties, "title": tool.name + "Arguments", "type": "object"}
            if required:
                schema["required"] = list(required)
            assert tool.inputSchema == schema
            assert tool.outputSchema == {
                "properties": {"result": {"title": "Result", "type": "string"}},
                "required": ["result"], "title": tool.name + "Output", "type": "object",
            }
            expected_description = DESCRIPTIONS[tool.name]
            if profile == "cyber" and tool.name in _JOB_VERBS:
                import inspect
                expected_description = inspect.cleandoc(expected_description) + "\n\n" + _JOB_CONTRACT
            assert tool.description == expected_description
            assert tool.title is None and tool.annotations is None and tool.meta is None

    asyncio.run(scenario())


def test_worker_preserves_contextvars_for_usage_accounting():
    service = FixtureService()

    async def scenario():
        token = CONTEXT.set("usage-context")
        try:
            await _ToolDispatcher(service).call("status", None)
        finally:
            CONTEXT.reset(token)

    asyncio.run(scenario())
    assert service.calls[0][3] == "usage-context"


def test_request_cancellation_does_not_stop_worker_or_invent_a_failed_verdict(tmp_path):
    completed = threading.Event()

    class FinishingService(FixtureService):
        def invoke(self, name, *args):
            try:
                return super().invoke(name, *args)
            finally:
                completed.set()

    service = FinishingService("cyber", state_dir=tmp_path, blocking="verify")

    async def scenario():
        pending = asyncio.create_task(_ToolDispatcher(service).call("verify", "subject"))
        try:
            assert await asyncio.wait_for(asyncio.to_thread(service.entered.wait, 2.0), 2.5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert not completed.is_set()
            assert [e["phase"] for e in read_diagnostics(tmp_path / "transport_calls.jsonl")] == [
                "received", "invoked",
            ]
        finally:
            service.release.set()
            assert await asyncio.wait_for(asyncio.to_thread(completed.wait, 2.0), 2.5)

    asyncio.run(scenario())
    assert [e["phase"] for e in read_diagnostics(tmp_path / "transport_calls.jsonl")] == [
        "received", "invoked", "finished",
    ]


@pytest.mark.parametrize("profile", ["default", "cyber"])
def test_actual_service_kernel_claim_and_certificate_through_mcp_worker(profile, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text("Establish the fixture security condition.")
    discharging_threads = []

    class Discharger:
        def discharge(self, row, obligation, method=None, tag=""):
            discharging_threads.append(threading.get_ident())
            return DischargeResult(
                outcome="holds", witness_kind="execution", witness="fixture observation",
                artifact="", instrument="fixture", basis="world",
            )

    def factory(service):
        backend = FixtureBackend({"obligations": lambda variables: {"obligations": [{
            "text": "the fixture security condition holds", "required_basis": "world",
        }]}})
        return Kernel(service.store, service.log, backend, service.artifacts,
                      critic_role=None, discharger=Discharger())

    service = Service(str(state), kernel_factory=factory, profile=profile,
                      project_root=str(tmp_path))

    async def scenario():
        async with create_connected_server_and_client_session(build_server(service)) as client:
            committed = decode(await completed_call(client, "commit", {
                "label": "final", "kind": "final", "text": "fixture condition holds", "deps": [],
            }))
            assert committed["status"] == "verified"
            assert decode(await completed_call(client, "verify", {"label": "final"}))["status"] == "verified"
            assert decode(await completed_call(client, "certificate", {"final_label": "final"}))["verdict"] == "verified"
        assert discharging_threads and all(t != threading.get_ident() for t in discharging_threads)

    asyncio.run(scenario())
    types = {event["type"] for event in service.log.replay()}
    assert "commit" in types and "obligations_compiled" in types
    assert "discharge" in types and "status_change" in types
    assert not any("transport" in event_type for event_type in types)


@pytest.mark.parametrize("profile", ["default", "cyber"])
def test_real_mcp_status_snapshots_labels_before_concurrent_commit(profile, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text("Establish the fixture security condition.")
    service = Service(str(state), profile=profile, trigger=TRIGGER_LAZY,
                      project_root=str(tmp_path), kernel_factory=lambda _: SimpleNamespace())
    service.commit("seed", "assertion", "seed condition", [])
    original_status, original_store_status = service.status, service.store.status
    entered, release = threading.Event(), threading.Event()
    reader_threads = set()

    def status(label=None):
        reader_threads.add(threading.get_ident())
        return original_status(label)

    def store_status(claim_id):
        if threading.get_ident() in reader_threads and not entered.is_set():
            entered.set()
            release.wait()
        return original_store_status(claim_id)

    service.status, service.store.status = status, store_status
    watchdog = threading.Timer(3.0, release.set)

    async def scenario():
        async with create_connected_server_and_client_session(build_server(service)) as client:
            pending = asyncio.create_task(completed_call(client, "status", {}))
            watchdog.start()
            try:
                assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 2.0), 2.5)
                # This must complete while status is querying its snapshot;
                # holding the feed lock over store queries would block it.
                committed = decode(await asyncio.wait_for(completed_call(client, "commit", {
                    "label": "new", "kind": "assertion", "text": "new condition", "deps": [],
                }), 1.0))
                assert committed["status"] == "unverified"
                assert not release.is_set()
                release.set()
                assert decode(await asyncio.wait_for(pending, 1.0)) == {"seed": "unverified"}
                assert decode(await completed_call(client, "status", {})) == {
                    "seed": "unverified", "new": "unverified",
                }
            finally:
                release.set()
                watchdog.cancel()
                await asyncio.wait_for(pending, 2.0)

    asyncio.run(scenario())
    events = list(service.log.replay())
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))


@pytest.mark.parametrize("profile", ["default", "cyber"])
def test_concurrent_actual_mcp_claims_certificates_keep_ledger_and_verdict_boundaries(profile, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text("Establish each fixture security condition independently.")

    class Discharger:
        def discharge(self, row, obligation, method=None, tag=""):
            return DischargeResult(
                outcome="refuted" if "failing" in row["text"] else "holds",
                witness_kind="execution", witness="fixture observation", artifact="",
                instrument="fixture", basis="world", strength="decisive",
            )

    def factory(service):
        backend = FixtureBackend({"obligations": lambda variables: {"obligations": [{
            "text": "independent condition: " + variables["claim_text"],
            "required_basis": "world",
        }]}})
        return Kernel(service.store, service.log, backend, service.artifacts,
                      critic_role=None, discharger=Discharger())

    service = Service(str(state), profile=profile, project_root=str(tmp_path), kernel_factory=factory)
    labels = ["final-" + str(i) for i in range(16)]

    async def scenario():
        async with create_connected_server_and_client_session(build_server(service)) as client:
            commits = await asyncio.gather(*(completed_call(client, "commit", {
                "label": label, "kind": "final", "text": ("failing " if i % 2 else "passing ") + label,
                "deps": [],
            }) for i, label in enumerate(labels)))
            assert [decode(reply)["status"] for reply in commits] == [
                "refuted" if i % 2 else "verified" for i in range(16)
            ]
            requests = ([completed_call(client, "certificate", {"final_label": label}) for label in labels]
                        + [completed_call(client, "verify", {"label": label}) for label in labels]
                        + [completed_call(client, "status", {}) for _ in range(4)])
            results = await asyncio.gather(*requests)
            for i, reply in enumerate(results[:16]):
                result = decode(reply)
                assert result["verdict"] == ("refuted" if i % 2 else "verified")
                certificate = service.artifacts.get(result["certificate_id"])
                assert content_hash(certificate) == result["certificate_id"]
                assert certificate["final"] == service.builder.resolve(labels[i])
            assert [decode(reply)["status"] for reply in results[16:32]] == [
                "refuted" if i % 2 else "verified" for i in range(16)
            ]
            expected = {label: "refuted" if i % 2 else "verified" for i, label in enumerate(labels)}
            assert all(decode(reply) == expected for reply in results[32:])
            decode(await completed_call(client, "commit", {
                "label": "ordinary", "kind": "assertion", "text": "passing ordinary assertion", "deps": [],
            }))
            invalid = decode(await completed_call(client, "certificate", {"final_label": "ordinary"}))
            assert "error" in invalid and "final claim" in invalid["error"]
            assert "certificate_id" not in invalid
            reserved = decode(await completed_call(client, "commit", {
                "label": "cyber-final:unowned", "kind": "final", "text": "passing", "deps": [],
            }))
            assert "error" in reserved and "claim_id" not in reserved

    asyncio.run(scenario())
    events = list(service.log.replay())
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert len([e for e in events if e["type"] == "certificate"]) == 16
    if profile == "cyber":
        calls = {}
        for event in read_diagnostics(state / "transport_calls.jsonl"):
            calls.setdefault(event["call_id"], []).append(event["phase"])
        assert len(calls) >= 55
        for phases in calls.values():
            assert phases[0] == "received"
            assert set(phases) <= {"received", "invoked", "finished", "responded"}
            assert phases.count("invoked") == phases.count("finished")
            if "invoked" in phases:
                assert phases.index("invoked") < phases.index("finished")


@pytest.mark.parametrize("profile", ["default", "cyber"])
def test_concurrent_verify_and_certificate_share_one_actual_kernel_discharge(profile, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text("Establish the fixture security condition.")
    entered, release = threading.Event(), threading.Event()
    discharges = []

    class Discharger:
        def discharge(self, row, obligation, method=None, tag=""):
            discharges.append(row["id"])
            entered.set()
            release.wait()
            return DischargeResult(
                outcome="holds", witness_kind="execution", witness="fixture observation",
                artifact="", instrument="fixture", basis="world", strength="decisive",
            )

    def factory(service):
        backend = FixtureBackend({"obligations": lambda variables: {"obligations": [{
            "text": "one independent fixture condition", "required_basis": "world",
        }]}})
        return Kernel(service.store, service.log, backend, service.artifacts,
                      critic_role=None, discharger=Discharger())

    service = Service(str(state), profile=profile, project_root=str(tmp_path), kernel_factory=factory,
                      trigger=TRIGGER_LAZY)
    watchdog = threading.Timer(3.0, release.set)

    async def scenario():
        async with create_connected_server_and_client_session(build_server(service)) as client:
            assert decode(await completed_call(client, "commit", {
                "label": "final", "kind": "final", "text": "fixture condition holds", "deps": [],
            }))["status"] == "unverified"
            requests = ([completed_call(client, "verify", {"label": "final"}) for _ in range(6)]
                        + [completed_call(client, "certificate", {"final_label": "final"}) for _ in range(6)])
            pending = asyncio.gather(*requests)
            watchdog.start()
            try:
                assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 2.0), 2.5)
                await asyncio.wait_for(client.send_ping(), 1.0)
                assert not pending.done() and not release.is_set()
            finally:
                release.set()
                watchdog.cancel()
                results = await asyncio.wait_for(pending, 2.0)
            assert all(decode(reply)["status"] == "verified" for reply in results[:6])
            assert all(decode(reply)["verdict"] == "verified" for reply in results[6:])

    asyncio.run(scenario())
    assert discharges == [service.builder.resolve("final")]
    events = list(service.log.replay())
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))


@pytest.mark.parametrize("profile", ["default", "cyber"])
def test_concurrent_label_collision_cannot_rebind_certificate_subject(profile, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text("Establish the fixed fixture condition.")

    class Discharger:
        def discharge(self, row, obligation, method=None, tag=""):
            return DischargeResult(
                outcome="refuted" if "failing" in row["text"] else "holds",
                witness_kind="execution", witness="fixture observation", artifact="",
                instrument="fixture", basis="world", strength="decisive",
            )

    def factory(service):
        backend = FixtureBackend({"obligations": lambda variables: {"obligations": [{
            "text": "independent condition: " + variables["claim_text"], "required_basis": "world",
        }]}})
        return Kernel(service.store, service.log, backend, service.artifacts,
                      critic_role=None, discharger=Discharger())

    service = Service(str(state), profile=profile, project_root=str(tmp_path), kernel_factory=factory)

    async def scenario():
        async with create_connected_server_and_client_session(build_server(service)) as client:
            replies = await asyncio.gather(*(completed_call(client, "commit", {
                "label": "same", "kind": "final", "text": text, "deps": [],
            }) for text in ("passing subject", "failing different subject")))
            results = [decode(reply) for reply in replies]
            assert sum("error" in result for result in results) == 1
            accepted = next(result for result in results if "error" not in result)
            subject = service.builder.resolve("same")
            assert subject == accepted["claim_id"]
            certificates = await asyncio.gather(*(completed_call(client, "certificate", {"final_label": "same"})
                                                  for _ in range(8)))
            for reply in certificates:
                certificate = decode(reply)
                assert certificate["verdict"] == accepted["status"]
                frozen = service.artifacts.get(certificate["certificate_id"])
                assert frozen["final"] == subject
                assert content_hash(frozen) == certificate["certificate_id"]

    asyncio.run(scenario())
    commits = [event for event in service.log.replay() if event["type"] == "commit"]
    assert len(commits) == 1


def test_diagnostic_failure_metadata_never_contains_payload_or_exception_text(tmp_path):
    service = FixtureService("cyber", state_dir=tmp_path, failure="verify")
    with pytest.raises(ValueError):
        asyncio.run(_ToolDispatcher(service).call("verify", "fixture-sensitive-argument"))
    path = tmp_path / "transport_calls.jsonl"
    events = read_diagnostics(path)
    assert [e["phase"] for e in events] == ["received", "invoked", "failed"]
    assert events[-1]["error_type"] == "ValueError"
    allowed = {"schema", "call_id", "tool", "phase", "ts", "elapsed_seconds", "error_type"}
    assert all(set(e) <= allowed for e in events)
    assert "fixture-sensitive" not in path.read_text()
    assert not (tmp_path / "events.jsonl").exists()


def test_default_profile_does_not_add_diagnostics(tmp_path):
    service = FixtureService(state_dir=tmp_path)
    result = asyncio.run(_ToolDispatcher(service).call("status", None))
    assert result["method"] == "status"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "fifo", "missing_directory"])
def test_unwritable_or_unsafe_diagnostic_target_does_not_change_service_result(tmp_path, unsafe):
    state = tmp_path / "state"
    state.mkdir()
    target = state / "transport_calls.jsonl"
    retained = tmp_path / "retained.txt"
    retained.write_text("unchanged")
    if unsafe == "symlink":
        target.symlink_to(retained)
    elif unsafe == "hardlink":
        os.link(retained, target)
    elif unsafe == "fifo":
        os.mkfifo(target)
    else:
        state = tmp_path / "not_created"
    service = FixtureService("cyber", state_dir=state)
    assert asyncio.run(_ToolDispatcher(service).call("status", None))["method"] == "status"
    assert retained.read_text() == "unchanged"
