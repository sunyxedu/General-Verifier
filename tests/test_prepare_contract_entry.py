"""Exercise the real Compiler/Critic path with a local Codex process stub.

Token counters here are synthetic protocol fixtures, never performance data.
No model, network service, or ordinary scoring scheduler is invoked.
"""
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.terminal_bench import prepare_contract_entry as producer
from test_contract_replay import source_state, service, snapshot
from verifier.contract_replay import ReplaySetupError, compile_input_digests, replay_identity
from verifier.core.model import content_hash
from verifier.llm.codex.runtime import CodexRuntime
from verifier.llm.schema import FixtureBackend


PAYLOAD = {"obligations": [{"text": "Every record reports the required answer",
                            "required_basis": "world"}]}
PASS = {"verdict": "pass", "counterexample_world": "n/a", "missing_condition": "n/a"}
ATTACK = {"verdict": "insufficient", "counterexample_world": "A record differs",
          "missing_condition": "Check every record"}
USAGE = {"input_tokens": 100, "cached_input_tokens": 30,
         "output_tokens": 20, "reasoning_output_tokens": 10}


def harness(tmp_path, monkeypatch, responses=None, *, usage=None, fixture=False):
    source = source_state(tmp_path)
    response_queue = copy.deepcopy(responses if responses is not None else [PAYLOAD, PASS])
    calls = []

    def version(cmd, **kwargs):
        assert cmd == ["codex", "--version"]
        return SimpleNamespace(stdout="codex-cli 0.153.4", returncode=0)

    monkeypatch.setattr("verifier.llm.codex.runtime.subprocess.run", version)

    def run(cmd, **kwargs):
        prompt = kwargs["stdin"].read()
        calls.append({"cwd": kwargs["cwd"], "prompt": prompt.decode() if isinstance(prompt, bytes) else prompt})
        assert response_queue, "unexpected extra model call"
        response = response_queue.pop(0)
        text = json.dumps(response)
        Path(cmd[cmd.index("-o") + 1]).write_text(text)
        events = [
            {"type": "thread.started", "thread_id": f"stub-{len(calls)}"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "answer", "type": "agent_message", "text": text}},
            {"type": "turn.completed", "usage": USAGE if usage is None else usage},
        ]
        return SimpleNamespace(returncode=0, stdout="\n".join(map(json.dumps, events)), stderr="")

    monkeypatch.setattr("verifier.llm.codex.runtime._run_codex_process", run)
    backend = FixtureBackend({"obligations": lambda _: copy.deepcopy(PAYLOAD),
                              "critic": lambda _: copy.deepcopy(PASS)}) if fixture else None
    services = []

    def factory(state):
        sc = service(state, backend or CodexRuntime(cwd=str(state), model="gpt-5.6-luna"))
        services.append(sc)
        return sc

    original = factory(source)
    graph = snapshot(original)
    config = {"model": "gpt-5.6-luna", "effort": "medium", "service_tier": "standard"}
    identity = replay_identity(original.kernel, graph, config)
    inputs = compile_input_digests(original.kernel, graph)

    def prepare(kind="support", name="prepare", **overrides):
        cid = original.builder.labels["A"] if kind == "support" else original.builder.finals[-1]
        kwargs = dict(setup_id="local-test", claim_id=cid, service_factory=factory, config=config,
                      expected_identity=identity, expected_compile_input_sha256=inputs[cid])
        kwargs.update(overrides)
        return producer.prepare_contract_entry(source, tmp_path / name, **kwargs)

    return SimpleNamespace(source=source, calls=calls, queue=response_queue,
                           services=services, identity=identity, prepare=prepare)


@pytest.mark.parametrize("kind", ["support", "final"])
def test_direct_producer_uses_real_roles_and_does_not_run_evidence(tmp_path, monkeypatch, kind):
    h = harness(tmp_path, monkeypatch)
    result = h.prepare(kind)
    receipt = result["receipt"]
    assert receipt["findings"] == []
    assert receipt["status"] == "passed"
    assert len(h.calls) == 2
    assert [call["role"] for call in receipt["calls"]] == ["obligations", "critic"]
    assert receipt["terminal_critic"]["successful_call_id"] == receipt["calls"][-1]["call_ids"][-1]
    assert result["entry"]["critic_receipt_sha256"] == content_hash(receipt)
    assert receipt["binding_mode"] == "trusted-kernel-terminal-pass-control-flow"
    assert receipt["accounting_scope"] == "one_claim_excludes_extraction_and_replay"
    assert (tmp_path / "prepare" / "prepare-receipt.json").exists()
    assert not (tmp_path / "prepare" / "state" / "prepare-receipt.json").exists()
    assert {e["type"] for e in h.services[0].log.replay()} == {
        "tool_call", "tool_result", "message", "extracted_commit"}
    kernel = h.services[-1].kernel
    assert kernel.store.db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0
    assert kernel.store.db.execute("SELECT COUNT(*) FROM obligation_edges").fetchone()[0] == 0
    assert not any(e["type"] in {"obligations_compiled", "judger_verdict", "discharge"}
                   for e in kernel.log.replay())


def test_critic_repair_is_charged_not_resampled(tmp_path, monkeypatch):
    h = harness(tmp_path, monkeypatch, [PAYLOAD, ATTACK, PAYLOAD, PASS])
    result = h.prepare()
    assert result["receipt"]["findings"] == []
    assert len(h.calls) == 4
    assert [(call["role"], call["round"]) for call in result["receipt"]["calls"]] == [
        ("obligations", 0), ("critic", 0), ("obligations", 1), ("critic", 1)]
    assert sum(len(call["attempts"]) for call in result["receipt"]["calls"]) == 4


def test_schema_retry_retains_both_bills_and_binds_successful_call(tmp_path, monkeypatch):
    h = harness(tmp_path, monkeypatch, [PAYLOAD, {"bad_schema": True}, PASS])
    result = h.prepare()
    assert result["receipt"]["findings"] == []
    critic = result["receipt"]["calls"][-1]
    assert len(critic["call_ids"]) == 2
    assert [row["attempt_status"] for row in critic["attempts"]] == ["schema_error", "succeeded"]
    assert critic["successful_call_id"] == critic["call_ids"][1]
    assert len(h.calls) == 3


def test_unknown_counters_never_yield_a_replay_entry(tmp_path, monkeypatch):
    h = harness(tmp_path, monkeypatch, usage={})
    result = h.prepare()
    assert len(h.calls) == 2  # Valid answers retained; accounting cannot pass.
    assert result["entry"] is None
    assert result["receipt"]["usage_complete"] is False
    assert "incomplete_usage" in result["receipt"]["findings"]


def test_fixture_pass_is_not_real_usage(tmp_path, monkeypatch):
    h = harness(tmp_path, monkeypatch, fixture=True)
    result = h.prepare()
    assert result["entry"] is None
    assert "incomplete_usage" in result["receipt"]["findings"]
    assert h.calls == []


@pytest.mark.parametrize("change", ["claim_pin", "source_pin", "existing_root"])
def test_preflight_refusal_never_calls_a_model_or_deletes_inputs(tmp_path, monkeypatch, change):
    h = harness(tmp_path, monkeypatch)
    overrides = {}
    if change == "claim_pin":
        overrides["expected_compile_input_sha256"] = "f" * 64
    elif change == "source_pin":
        overrides["expected_identity"] = replace(h.identity, source_sha256="f" * 64)
    else:
        (tmp_path / "prepare").mkdir()
        (tmp_path / "prepare" / "keep").write_text("keep")
    with pytest.raises(ReplaySetupError):
        h.prepare(**overrides)
    assert h.calls == []
    assert (h.source / "task.md").exists()
    if change == "existing_root":
        assert (tmp_path / "prepare" / "keep").read_text() == "keep"


def test_failed_compiler_is_not_retried_by_producer(tmp_path, monkeypatch):
    h = harness(tmp_path, monkeypatch, [{"wrong": 1}, {"wrong": 2}])
    result = h.prepare()
    assert len(h.calls) == 2  # Unchanged AgentRole schema retry budget only.
    assert result["entry"] is None
    assert result["receipt"]["error_class"] == "RoleError"
    assert result["receipt"]["usage_complete"] is True
    assert "compilation_raised" in result["receipt"]["findings"]


def test_new_preparations_never_read_previous_compiler_logs(tmp_path, monkeypatch):
    h = harness(tmp_path, monkeypatch, [PAYLOAD, PASS, PAYLOAD, PASS])
    first, second = h.prepare(name="first"), h.prepare(name="second")
    assert first["receipt"]["status"] == second["receipt"]["status"] == "passed"
    assert first["receipt"]["event_prefix_sha256"] != second["receipt"]["event_prefix_sha256"]  # timestamps differ
    assert first["receipt"]["compile_input_sha256"] == second["receipt"]["compile_input_sha256"]
    assert "first/state" not in h.calls[2]["prompt"]
    assert "decomposition_attacked" not in h.calls[2]["prompt"]


@pytest.mark.parametrize("target", ["usage", "spans", "events", "source", "graph", "evidence"])
def test_late_audit_mutations_never_publish_candidate(tmp_path, monkeypatch, target):
    h = harness(tmp_path, monkeypatch)
    original = producer._receipt_calls

    def changed(usage_path, spans_path, *args, **kwargs):
        answer = original(usage_path, spans_path, *args, **kwargs)
        if target in {"usage", "spans", "events"}:
            path = {"usage": usage_path, "spans": spans_path,
                    "events": tmp_path / "prepare" / "state" / "events.jsonl"}[target]
            with path.open("ab") as handle:
                handle.write(b"\n")
        elif target == "source":
            (h.source / "task.md").write_text("Changed authoritative task")
        elif target == "graph":
            h.services[-1].store.db.execute("UPDATE claims SET text='changed' WHERE kind='assertion'")
        else:
            cid = h.services[-1].builder.labels["A"]
            h.services[-1].store.record_evidence(cid, {"outcome": "holds"})
        return answer

    monkeypatch.setattr(producer, "_receipt_calls", changed)
    result = h.prepare()
    assert result["entry"] is None
    assert result["receipt"]["status"] == "inconclusive"
    assert result["receipt"]["findings"]
    assert len(h.calls) == 2


@pytest.mark.parametrize("kind", ["support", "final"])
def test_empty_compilation_is_inconclusive_even_after_critic_pass(tmp_path, monkeypatch, kind):
    h = harness(tmp_path, monkeypatch, [{"obligations": []}, PASS])
    result = h.prepare(kind)
    assert result["entry"] is None
    assert "empty_or_failed_contract" in result["receipt"]["findings"]
    assert result["receipt"]["usage_complete"] is True
    assert len(h.calls) == 2


def test_critic_exhaustion_keeps_cost_and_does_not_resample(tmp_path, monkeypatch):
    # Kernel rounds are part of the identity, not reduced for this test.
    h = harness(tmp_path, monkeypatch, [PAYLOAD, ATTACK] * 20)
    rounds = h.services[0].kernel.critic_rounds
    result = h.prepare()
    assert result["entry"] is None
    assert result["receipt"]["usage_complete"] is True
    assert len(h.calls) == rounds * 2
    assert len(result["receipt"]["calls"]) == rounds * 2
    assert "empty_or_failed_contract" in result["receipt"]["findings"]
