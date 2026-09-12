"""Frozen transport must not reveal later observations, including fallback."""
from pathlib import Path

from verifier.core.events import EventLog
from verifier.engine.source_evidence import (
    bind_source_evidence_variables, refresh_source_evidence_variables,
)
from verifier.llm.agents import COMPILE_OBLIGATIONS_AGENT


def frozen_state(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("message", {"text": "early-public-evidence"})
    prefix = log.freeze_record(1)
    log.bind_record("fixture-claim", prefix)
    log.append("tool_result", {"excerpt": "late-result-must-not-enter-the-prefix"})
    return log, prefix


def test_frozen_source_view_and_mixed_fallback_exclude_later_results(tmp_path):
    log, prefix = frozen_state(tmp_path)
    bound = bind_source_evidence_variables(log, {"ledger_path": prefix}, enabled=True)
    assert bound["_mixed_ledger_path"] == prefix
    view = bound["_source_evidence_view"]
    assert view._log.path == prefix
    combined = b"\n".join(raw for _, raw in view._files)
    assert b"early-public-evidence" in combined
    assert b"late-result-must-not-enter-the-prefix" not in combined
    log.append("tool_result", {"excerpt": "even-later-result"})
    refreshed = refresh_source_evidence_variables(bound)
    assert refreshed["_source_evidence_view"].view_id == view.view_id
    assert refreshed["_mixed_ledger_path"] == prefix


def test_backend_opt_out_restores_frozen_not_live_locator(tmp_path):
    log, prefix = frozen_state(tmp_path)
    seen = []

    class OfflineFixtureBackend:
        def allows_compact_source_evidence(self, role_name, variables):
            return False

        def complete(self, role_name, prompt, variables):
            seen.append(dict(variables))
            return {"obligations": [{"text": "the declared fact holds",
                                      "required_basis": "record"}]}

    # Fill the ordinary inherited template without any API or CLI call.
    variables = {key: "(fixture)" for key in (
        "claim_text", "claim_kind", "contract_locus", "task_context", "world_mode",
        "world_path", "world_map", "verified_deps", "pending_deps",
        "dependency_referent_context", "graph", "memory_hits", "critic_feedback")}
    variables["ledger_path"] = prefix
    bound = bind_source_evidence_variables(log, variables, enabled=True)
    COMPILE_OBLIGATIONS_AGENT.call(OfflineFixtureBackend(), bound)
    assert seen[0]["ledger_path"] == prefix
    assert "_source_evidence_view" not in seen[0]
    assert "late-result-must-not-enter-the-prefix" not in Path(prefix).read_text()


def test_disabled_view_preserves_original_frozen_variables(tmp_path):
    log, prefix = frozen_state(tmp_path)
    original = {"ledger_path": prefix, "inert": "fixture"}
    assert bind_source_evidence_variables(log, original, enabled=False) == original
