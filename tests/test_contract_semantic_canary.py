"""No-model tests for the three-nonce Compiler/Critic semantic canary."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.terminal_bench import probe_contract_semantic_canary as canary
from experiments.terminal_bench.prepare_shared_contracts import codex_service
from experiments.terminal_bench.run_typed_adapter_eval import (
    _source_fingerprint,
)
from verifier import offline
from verifier.core.events import EventLog


NONCES = {
    "candidate": "0123456789abcdef0123456789abcdef0123456789abcdef",
    "provenance": "123456789abcdef0123456789abcdef0123456789abcdef0",
    "runner": "23456789abcdef0123456789abcdef0123456789abcdef01",
}
PASS = {
    "verdict": "pass",
    "counterexample_world": "n/a",
    "missing_condition": "n/a",
}
USAGE = {
    "input_tokens": 100,
    "cached_input_tokens": 60,
    "output_tokens": 20,
    "reasoning_output_tokens": 10,
}


def _obligation(text, basis, *, clause="synthetic clause", severity="must"):
    return {
        "id": "0" * 64,
        "claim_id": "1" * 64,
        "text": text,
        "required_basis": basis,
        "severity": severity,
        "clause": clause,
        "refutable_by_absence": False,
        "premise_scope": [],
    }


def _good_obligations():
    return [
        _obligation(
            "The current candidate final response contains "
            + NONCES["candidate"],
            "world",
        ),
        _obligation(
            "The candidate-producing solver emitted "
            + NONCES["provenance"] + " by invoking origin::"
            + NONCES["provenance"],
            "record",
        ),
    ]


def _preparation(obligations=None, *, critic="pass"):
    obligations = copy.deepcopy(
        _good_obligations() if obligations is None else obligations)
    return {
        "receipt": {
            "status": "passed",
            "findings": [],
            "terminal_critic": ({"verdict": "pass"}
                                if critic == "pass" else None),
        },
        "entry": {"status": "passed", "obligations": obligations},
    }


def _compiler_payload():
    return {"obligations": [
        {
            "text": "The current candidate final response contains "
                    + NONCES["candidate"],
            "required_basis": "world",
            "severity": "must",
            "clause": "CANDIDATE-CONTENT requires "
                      + NONCES["candidate"],
            "refutable_by_absence": True,
        },
        {
            "text": "The candidate-producing solver emitted "
                    + NONCES["provenance"] + " by invoking origin::"
                    + NONCES["provenance"],
            "required_basis": "record",
            "severity": "must",
            "clause": "PROVENANCE-HISTORY requires origin::"
                      + NONCES["provenance"],
            "refutable_by_absence": False,
        },
    ]}


def test_semantic_audit_accepts_exact_boundary_classes():
    findings, obligations = canary.audit_semantics(
        _preparation(), NONCES)

    assert findings == []
    assert obligations == _good_obligations()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("candidate_record", "candidate_content_floor_or_modality_changed"),
        ("candidate_should", "candidate_content_floor_or_modality_changed"),
        ("provenance_world", "provenance_floor_or_modality_changed"),
        ("provenance_should", "provenance_floor_or_modality_changed"),
        ("provenance_origin_missing",
         "provenance_process_requirement_missing"),
        ("runner", "runner_protocol_compiled"),
        ("extra", "unclassified_extra_obligation"),
        ("merged", "independent_requirements_merged"),
        ("critic", "terminal_critic_not_passed"),
    ],
)
def test_semantic_audit_fails_closed_on_each_boundary_drift(
        mutation, expected):
    obligations = _good_obligations()
    critic = "pass"
    if mutation == "candidate_record":
        obligations[0]["required_basis"] = "record"
    elif mutation == "candidate_should":
        obligations[0]["severity"] = "should"
    elif mutation == "provenance_world":
        obligations[1]["required_basis"] = "world"
    elif mutation == "provenance_should":
        obligations[1]["severity"] = "should"
    elif mutation == "provenance_origin_missing":
        obligations[1]["text"] = "The solver emitted " + NONCES["provenance"]
    elif mutation == "runner":
        obligations.append(_obligation(
            "The runner records " + NONCES["runner"], "record"))
    elif mutation == "extra":
        obligations.append(_obligation("An invented extra condition", "world"))
    elif mutation == "merged":
        obligations[0]["text"] += " and " + NONCES["provenance"]
    else:
        critic = "missing"

    findings, _ = canary.audit_semantics(
        _preparation(obligations, critic=critic), NONCES)

    assert expected in findings


def test_semantic_audit_requires_both_positive_controls():
    findings, _ = canary.audit_semantics(
        _preparation(_good_obligations()[:1]), NONCES)
    assert "provenance_requirement_missing" in findings

    findings, _ = canary.audit_semantics(
        _preparation(_good_obligations()[1:]), NONCES)
    assert "candidate_content_requirement_missing" in findings


def test_model_free_source_has_three_opaque_independent_clauses(tmp_path):
    source = tmp_path / "source"
    seal = canary._prepare_source(source, NONCES)

    assert offline.verify_immutable_input_anchor(str(source)) == seal
    task = (source / "task.md").read_text(encoding="utf-8")
    assert all(value in task for value in NONCES.values())
    assert "byte-for-byte identical" in task
    assert "CRITICAL OPERATING RULE" in task
    assert "every response MUST include at least one" in task
    assert "runner uses this response handshake" in task
    assert "Opaque-literal contract" in task
    assert "create no candidate obligation" not in task
    assert "origin::" + NONCES["provenance"] in task
    assert "copy::" + NONCES["provenance"] in task
    assert (source / "final.txt").read_text(encoding="utf-8").count(
        NONCES["candidate"]) == 1
    assert NONCES["runner"] not in (source / "final.txt").read_text()
    events = list(EventLog(str(source / "events.jsonl")).replay())
    assert [event["type"] for event in events] == [
        "message", "extracted_commit"]
    assert (source / offline.DIAGNOSTIC_ONLY_FILE).exists()


def test_run_canary_uses_real_direct_roles_with_local_process_stub(
        tmp_path, monkeypatch):
    responses = [_compiler_payload(), PASS]
    prompts = []

    def version(command, **_kwargs):
        assert command == ["codex", "--version"]
        return SimpleNamespace(
            stdout="codex-cli 0.153.4", returncode=0)

    def run(command, **kwargs):
        assert kwargs["stdin"].read() == ""
        prompts.append(command[-1])
        response = responses.pop(0)
        text = json.dumps(response)
        Path(command[command.index("-o") + 1]).write_text(
            text, encoding="utf-8")
        events = [
            {"type": "thread.started", "thread_id": f"stub-{len(prompts)}"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {
                "id": "answer", "type": "agent_message", "text": text,
            }},
            {"type": "turn.completed", "usage": USAGE},
        ]
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(json.dumps(event) for event in events),
            stderr="",
        )

    monkeypatch.setattr(
        "verifier.llm.codex_backend.subprocess.run", version)
    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", run)
    config = canary.runtime_config()
    result = canary.run_canary(
        tmp_path / "canary",
        expected_source_fingerprint=_source_fingerprint(canary.SOURCE_ROOT),
        nonces=dict(NONCES),
        service_factory=lambda state: codex_service(state, config),
    )

    assert result["status"] == "complete", result["findings"]
    assert result["findings"] == []
    assert result["test_only"] is True
    assert result["preparation"]["status"] == "passed"
    assert result["preparation"]["terminal_critic"]["verdict"] == "pass"
    assert [call["role"] for call in result["preparation"]["calls"]] == [
        "obligations", "critic"]
    assert result["preparation"]["usage_complete"] is True
    assert result["preparation"]["attribution_complete"] is True
    assert len(prompts) == 2
    assert responses == []
    missing = [
        [key for key, value in NONCES.items() if value not in prompt]
        for prompt in prompts
    ]
    assert missing == [[], []]
    assert json.loads(
        (tmp_path / "canary" / "canary-result.json").read_text()) == result


def test_cli_dry_run_is_free_and_execution_needs_exact_confirmation(
        tmp_path, capsys):
    output = tmp_path / "canary"
    argv = [
        "--source-fingerprint", "0" * 64,
        "--output-root", str(output),
    ]

    assert canary.main(argv) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["paid_scope"] == "Compiler/Critic direct preparation only"
    assert preview["runtime"] == canary.RUNTIME
    assert not output.exists()

    with pytest.raises(SystemExit):
        canary.main([*argv, "--execute", "--confirm-paid", "wrong-model"])
    assert not output.exists()


@pytest.mark.parametrize("invalid", [
    {"candidate": "same-sufficiently-long", "provenance": "same-sufficiently-long",
     "runner": "different-sufficiently-long"},
    {"candidate": "short", "provenance": NONCES["provenance"],
     "runner": NONCES["runner"]},
    {"candidate": NONCES["candidate"], "provenance": NONCES["provenance"]},
])
def test_invalid_nonce_sets_fail_before_creating_output(tmp_path, invalid):
    output = tmp_path / "canary"
    with pytest.raises(ValueError, match="nonce"):
        canary.run_canary(
            output,
            expected_source_fingerprint=_source_fingerprint(
                canary.SOURCE_ROOT),
            nonces=invalid,
            service_factory=lambda _state: None,
        )
    assert not output.exists()
