"""The offline ORM path: feed ingestion, outcome-blind extraction with
structural retry, certify-based scoring, and best-of-N selection.

Fixture-backed and instrument-free, like the rest of the suite: the
extractor is a FixtureBackend role, the kernel seats are fixtures/stubs.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from verifier import offline, selection
from verifier.core.events import EventLog
from verifier.core.store import ClaimStore
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.llm.schema import FixtureBackend
from verifier.llm.agents import CRITIC_AGENT
from verifier.stream.builder import Builder

COMPILE = lambda v: {"obligations": [{"text": f"O of: {v['claim_text']}", "required_basis": "record"}]}  # noqa: E731

FEED = [
    {"type": "tool_call", "payload": {
        "tool": "Bash", "tool_use_id": "t1", "input": {"command": "echo 42"}}},
    {"type": "tool_result", "payload": {
        "tool": "Bash", "tool_use_id": "t1",
        "response_sha256": "abc", "excerpt": "42"}},
    {"type": "not_a_hook_event", "payload": {"x": 1}},   # untrusted line: ignored
]

GOOD_COMMITS = {"commits": [
    {"label": "C1", "kind": "assertion",
     "text": "as of effect:2, echo printed 42", "deps": ["effect:2"],
     "revises": ""},
    {"label": "F", "kind": "final", "text": "the answer is 42",
     "deps": ["C1"], "revises": ""},
]}


class StubDischarger:
    def __init__(self, outcome="holds"):
        self.outcome = outcome
        self.seen = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.seen.append(obligation.text)
        return DischargeResult(witness_kind="execution", outcome=self.outcome, witness=None,
                               artifact="", instrument="stub",
                               basis="world")


def state_dir(tmp_path, feed=FEED):
    state = str(tmp_path / "gv")
    os.makedirs(state, exist_ok=True)
    with open(os.path.join(state, "hook_feed.jsonl"), "w") as f:
        for line in feed:
            f.write(json.dumps(line) + "\n")
    with open(os.path.join(state, "task.md"), "w") as f:
        f.write("print the answer")
    with open(os.path.join(state, "final.txt"), "w") as f:
        f.write("FINAL ANSWER: 42")
    return state


def kernel_factory(sc):
    return Kernel(sc.store, sc.log, FixtureBackend({"obligations": COMPILE}),
                  sc.artifacts, critic_role=None, discharger=StubDischarger())


# ---- ingestion ---------------------------------------------------------------

def test_feed_lines_become_referenceable_effects(tmp_path):
    state = state_dir(tmp_path)
    log = EventLog(str(tmp_path / "events.jsonl"))

    n = offline.ingest_feed(os.path.join(state, "hook_feed.jsonl"), log)

    assert n == 2                                # the unknown line was dropped
    builder = Builder(log, ClaimStore())
    for event in log.replay():
        builder.feed(event)
    assert "effect:2" in builder.labels          # tool_result -> effect claim


# ---- extraction --------------------------------------------------------------

def test_extract_writes_a_replayable_graph(tmp_path):
    state = state_dir(tmp_path)
    backend = FixtureBackend({"extract": lambda v: GOOD_COMMITS})

    meta = offline.extract(state, backend=backend)

    assert meta["rounds"] == 1 and meta["commits"] == 2
    assert meta["fingerprint"] == offline.pipeline_fingerprint()
    log = EventLog(os.path.join(state, "events.jsonl"))
    builder = Builder(log, ClaimStore())
    for event in log.replay():
        builder.feed(event)
    assert len(builder.finals) == 1
    assert builder.labels["F"] == builder.finals[0]
    # the extractor saw the trace, not the verdicts
    variables = backend.calls[0]["variables"]
    assert variables["task"] == "print the answer"
    assert "echo 42" in variables["events_index"]
    assert variables["feedback"] == ""


def test_structural_error_feeds_back_and_retries(tmp_path):
    state = state_dir(tmp_path)
    bad = {"commits": [{"label": "F", "kind": "final", "text": "done",
                        "deps": ["nope"], "revises": ""}]}
    responses = iter([bad, GOOD_COMMITS])
    backend = FixtureBackend({"extract": lambda v: next(responses)})

    meta = offline.extract(state, backend=backend)

    assert meta["rounds"] == 2
    feedback = backend.calls[1]["variables"]["feedback"]
    assert "'F'" in feedback and "nope" in feedback


def test_not_exactly_one_final_is_a_structural_error(tmp_path):
    state = state_dir(tmp_path)
    no_final = {"commits": [{"label": "C1", "kind": "assertion", "text": "x",
                             "deps": [], "revises": ""}]}
    backend = FixtureBackend({"extract": lambda v: no_final})

    with pytest.raises(offline.ExtractionError, match="exactly one final"):
        offline.extract(state, backend=backend, max_rounds=2)
    # every failed round rebuilt the candidate from scratch
    assert not os.path.exists(os.path.join(state, "events.jsonl"))


def test_extraction_failure_scores_unscoreable_and_ranks_last(tmp_path):
    state = state_dir(tmp_path)
    backend = FixtureBackend({})    # no extract fixture -> RoleError every round

    features = offline.score(state, backend=backend,
                             kernel_factory=kernel_factory, max_rounds=2)

    assert features["unscoreable"] is True
    rankable_refuted = {
        "feature_schema_version": offline.FEATURE_SCHEMA_VERSION,
        "unscoreable": False,
        "verdict": "refuted",
        "final_obligations": {"refuted": 1},
        "final_obligations_total": 1,
    }
    assert offline.rank_key(features) < offline.rank_key(rankable_refuted)


# ---- scoring -----------------------------------------------------------------

def test_score_certifies_the_extracted_graph(tmp_path):
    state = state_dir(tmp_path)
    backend = FixtureBackend({"extract": lambda v: GOOD_COMMITS})

    features = offline.score(state, backend=backend,
                             kernel_factory=kernel_factory)

    assert features["verdict"] == "verified"
    assert features["slice_size"] == 3           # effect + C1 + F
    assert features["claims"]["verified"] == 3
    assert features["obligations"]["verified"] == features["obligations_total"]
    assert features["feature_schema_version"] == offline.FEATURE_SCHEMA_VERSION
    assert features["final_obligations"] == {"verified": 1}
    assert features["final_obligations_total"] == 1
    assert features["contract_audit_status"] == "available"
    assert features["final_obligation_contract"] == [{
        "ordinal": 0,
        "id": features["final_obligation_outcomes_ordered"][0][
            "obligation_id"],
        "text": "O of: the answer is 42",
        "premise_scope": [],
        "required_basis": "record",
        "severity": "must",
        "clause": "",
        "refutable_by_absence": False,
    }]
    assert features["final_obligation_outcomes_ordered"] == [{
        "ordinal": 0,
        "obligation_id": features["final_obligation_contract"][0]["id"],
        "evaluated": True,
        "status": "verified",
        "reason": None,
        "basis": "world",
        "strength": None,
    }]
    assert features["extract_rounds"] == 1
    # persisted for the experiment drivers
    on_disk = json.load(open(os.path.join(state, "score.json")))
    assert on_disk == features


def test_captured_world_seal_detects_content_and_topology_changes(tmp_path):
    state = state_dir(tmp_path)
    app = tmp_path / "gv" / "world" / "app"
    app.mkdir(parents=True)
    result = app / "result.txt"
    result.write_text("original")
    (tmp_path / "gv" / "world_manifest.json").write_text(
        json.dumps({"captured_roots": ["/app"]}))
    original = offline.captured_world_seal(state)

    result.write_text("mutated")
    assert offline.captured_world_seal(state) != original
    result.write_text("original")
    assert offline.captured_world_seal(state) == original

    (app / "result.txt-wal").write_text("")
    assert offline.captured_world_seal(state) != original


def test_trusted_adapter_anchor_rejects_later_input_tampering(tmp_path):
    state = state_dir(tmp_path)
    seal = offline.write_immutable_input_anchor(state)

    assert offline.verify_immutable_input_anchor(state) == seal
    with open(tmp_path / "gv" / "hook_feed.jsonl", "a") as handle:
        handle.write('{"type":"tool_result","payload":{}}\n')

    with pytest.raises(offline.ImmutableInputError, match="seal mismatch"):
        offline.verify_immutable_input_anchor(state)


def test_sanitized_state_without_trusted_anchor_fails_closed(tmp_path):
    state = state_dir(tmp_path)
    (tmp_path / "gv" / ".sanitized").write_text("")
    backend = FixtureBackend({"extract": lambda v: GOOD_COMMITS})

    features = offline.score(
        state, backend=backend, kernel_factory=kernel_factory)

    assert features["unscoreable"] is True
    assert "missing its trusted adapter seal" in features["error"]
    assert backend.calls == []


def test_score_fails_closed_when_verifier_mutates_captured_world(tmp_path):
    state = state_dir(tmp_path)
    app = tmp_path / "gv" / "world" / "app"
    app.mkdir(parents=True)
    (app / "main.db").write_bytes(b"frozen-db")
    (tmp_path / "gv" / "world_manifest.json").write_text(
        json.dumps({"captured_roots": ["/app"]}))

    class MutatingDischarger(StubDischarger):
        def discharge(self, row, obligation, method=None, tag=""):
            (app / "main.db-wal").write_bytes(b"")
            return super().discharge(row, obligation, method, tag)

    def mutating_kernel_factory(sc):
        return Kernel(
            sc.store, sc.log,
            FixtureBackend({"obligations": COMPILE}),
            sc.artifacts, critic_role=None,
            discharger=MutatingDischarger(),
        )

    features = offline.score(
        state,
        backend=FixtureBackend({"extract": lambda v: GOOD_COMMITS}),
        kernel_factory=mutating_kernel_factory,
    )

    assert features["unscoreable"] is True
    assert features["verdict"] is None
    assert "immutable offline inputs mutated" in features["error"]
    assert features["world_seal_before"] != features["world_seal_after"]
    assert (tmp_path / "gv" / offline.IMMUTABLE_INPUT_TAINT_FILE).is_file()


def test_feature_fold_keeps_final_contract_separate_from_slice_diagnostics():
    cert = {
        "final": "F",
        "verdict": "partial",
        "verdict_strength": None,
        "certificate_id": "cert",
        "slice": [
            {
                "id": "A", "status": "verified",
                "obligations": [
                    {"id": "OA1", "status": "verified"},
                    {"id": "OA2", "status": "verified"},
                ],
            },
            {
                "id": "F", "status": "partial",
                "obligations": [
                    {"id": "OF1", "status": "verified"},
                    {"id": "OF2", "status": "partial"},
                ],
            },
        ],
    }

    features = offline._features(cert, extract_rounds=1)

    assert features["obligations"] == {"verified": 3, "partial": 1}
    assert features["obligations_total"] == 4
    assert features["final_obligations"] == {"verified": 1, "partial": 1}
    assert features["final_obligations_total"] == 2
    assert features["final_contrary_signals"] == 0
    assert offline.rank_key(features) == (2, 0)


def test_ordered_final_outcomes_retain_unevaluated_contract_items():
    cert = {
        "final": "F",
        "slice": [{
            "id": "F",
            "obligations": [{
                "id": "O1", "status": "refuted", "reason": "witness",
                "basis": "world", "strength": "decisive",
            }],
        }],
    }
    contract = [
        {"ordinal": 0, "id": "O1"},
        {"ordinal": 1, "id": "O2"},
    ]

    outcomes = offline._ordered_final_outcomes(cert, contract)

    assert outcomes[0] == {
        "ordinal": 0, "obligation_id": "O1", "evaluated": True,
        "status": "refuted", "reason": "witness", "basis": "world",
        "strength": "decisive",
    }
    assert outcomes[1] == {
        "ordinal": 1, "obligation_id": "O2", "evaluated": False,
        "status": None, "reason": None, "basis": None, "strength": None,
    }


@pytest.mark.parametrize("audit_case", ["critic_failed", "empty"])
def test_contract_audit_unavailable_does_not_change_unverified_score(
        tmp_path, audit_case):
    state = state_dir(tmp_path)
    final_only = {"commits": [{
        "label": "F", "kind": "final", "text": "the answer is 42",
        "deps": [], "revises": "",
    }]}
    attack = {
        "verdict": "insufficient",
        "counterexample_world": "the condition can fail",
        "missing_condition": "add a complete condition",
    }

    def factory(sc):
        roles = ({"obligations": COMPILE, "critic": attack}
                 if audit_case == "critic_failed"
                 else {"obligations": {"obligations": []}})
        return Kernel(
            sc.store, sc.log, FixtureBackend(roles), sc.artifacts,
            critic_role=(CRITIC_AGENT if audit_case == "critic_failed" else None),
            critic_rounds=1, discharger=StubDischarger(),
        )

    features = offline.score(
        state, backend=FixtureBackend({"extract": final_only}),
        kernel_factory=factory,
    )

    assert features["unscoreable"] is False
    assert features["verdict"] == "unverified"
    assert features["final_obligations_total"] == 0
    assert features["contract_audit_status"] == (
        "missing" if audit_case == "critic_failed" else "empty")
    assert features["final_obligation_contract"] == []
    assert features["final_obligation_outcomes_ordered"] == []


def test_score_resumes_from_an_extracted_graph(tmp_path):
    state = state_dir(tmp_path)
    backend = FixtureBackend({"extract": lambda v: GOOD_COMMITS})
    offline.extract(state, backend=backend)

    # no extract fixture available: score must not need one on resume
    features = offline.score(state, backend=FixtureBackend({}),
                             kernel_factory=kernel_factory)

    assert features["verdict"] == "verified"


def test_pipeline_fingerprint_is_deterministic():
    assert offline.pipeline_fingerprint() == offline.pipeline_fingerprint()


def test_config_identity_tracks_pinned_browser_observer_without_host_path(
        tmp_path, monkeypatch):
    monkeypatch.delenv("GV_BROWSER_OBSERVER", raising=False)
    monkeypatch.delenv("GV_BROWSER_OBSERVER_SHA256", raising=False)
    monkeypatch.delenv("GV_BROWSER_EXECUTABLE", raising=False)
    monkeypatch.delenv("GV_BROWSER_EXECUTABLE_SHA256", raising=False)
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1")
    baseline = offline.config_fields(model="same", seats="codex")
    runner = tmp_path / "observer"
    runner.write_text("#!/bin/sh\nprintf '{\"ok\":true}'\n")
    runner.chmod(0o700)
    digest = hashlib.sha256(runner.read_bytes()).hexdigest()
    monkeypatch.setenv("GV_BROWSER_OBSERVER", os.fspath(runner))
    monkeypatch.setenv("GV_BROWSER_OBSERVER_SHA256", digest)
    monkeypatch.setenv("GV_BROWSER_EXECUTABLE", os.fspath(runner))
    monkeypatch.setenv("GV_BROWSER_EXECUTABLE_SHA256", digest)

    configured = offline.config_fields(model="same", seats="codex")

    assert configured["config"] != baseline["config"]
    assert configured["browser_observer"]["observer_sha256"] == digest
    assert configured["browser_observer"]["browser_sha256"] == digest
    assert os.fspath(runner) not in json.dumps(configured)


def test_config_separates_adapter_control_and_treatment(monkeypatch):
    for variable in ("GV_BROWSER_OBSERVER", "GV_BROWSER_OBSERVER_SHA256",
                     "GV_BROWSER_EXECUTABLE", "GV_BROWSER_EXECUTABLE_SHA256"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "0")
    control = offline.config_fields(model="same", seats="codex")
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1")
    treatment = offline.config_fields(model="same", seats="codex")

    assert control["typed_adapters"] is False
    assert treatment["typed_adapters"] is True
    assert control["config"] != treatment["config"]


def test_codex_config_identity_stamps_canonical_runtime_roots(
        tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv(
        "GV_OFFLINE_RUNTIME_READ_ROOTS", json.dumps([os.fspath(runtime)]))

    configured = offline.config_fields(model="same", seats="codex")

    assert configured["offline_runtime_read_roots"] == [os.fspath(runtime)]
    assert offline.config_fields()["offline_runtime_read_roots"] == []


def test_config_identity_separates_models_and_seats():
    luna = offline.config_fields(model="gpt-5.6-luna", seats="codex")
    terra = offline.config_fields(model="gpt-5.6-terra", seats="codex")
    assert luna["config"] != terra["config"]
    # codex default model resolves so implicit and explicit Luna coincide
    assert offline.config_fields(seats="codex")["config"] == luna["config"]
    assert offline.config_fields()["config"] != luna["config"]
    assert offline.config_fields(effort="high", seats="codex")["config"] \
        != luna["config"]


@pytest.mark.parametrize("field,value", [
    ("MODEL", "a-different-model"),
    ("PROVIDER", "a-different-provider"),
    ("ANTHROPIC_BASE_URL", "https://another.example/anthropic"),
    ("TEXT_ONLY", True),
])
def test_sdk_config_tracks_the_resolved_provider(monkeypatch, field, value):
    from verifier.llm import provider

    monkeypatch.setattr(provider, "TEXT_ONLY", False)
    original = offline.config_fields()
    monkeypatch.setattr(provider, field, value)
    assert offline.config_fields()["config"] != original["config"]


def test_sdk_config_resolves_defaults_without_recording_credentials(monkeypatch):
    from verifier.llm import provider

    monkeypatch.setattr(provider, "ANTHROPIC_BASE_URL",
                        "https://user:secret@example.test/api?token=secret")
    original = offline.config_fields()
    assert original == offline.config_fields(model=provider.MODEL)
    assert "secret" not in json.dumps(original)
    monkeypatch.setattr(provider, "API_KEY", "rotated-key")
    assert offline.config_fields() == original


def test_config_identity_separates_codex_service_tiers():
    implicit = offline.config_fields(model="gpt-5.6-terra", seats="codex")
    standard = offline.config_fields(
        model="gpt-5.6-terra", seats="codex", service_tier="standard")
    fast = offline.config_fields(
        model="gpt-5.6-terra", seats="codex", service_tier="fast")

    assert implicit == standard
    assert standard["service_tier"] == "standard"
    assert fast["service_tier"] == "fast"
    assert standard["config"] != fast["config"]
    assert offline.config_id(
        model="gpt-5.6-terra", seats="codex", service_tier="fast",
    ) == fast["config"]
    assert offline.config_fields()["service_tier"] == "n/a"
    with pytest.raises(ValueError, match="only valid for Codex"):
        offline.config_fields(service_tier="fast")


def test_score_rejects_a_backend_tier_that_disagrees_with_its_stamp(tmp_path):
    from verifier.llm.codex_backend import CodexBackend

    with pytest.raises(ValueError, match="does not match"):
        offline.score(
            str(tmp_path),
            seats="codex",
            service_tier="standard",
            backend=CodexBackend(cwd=str(tmp_path), service_tier="fast"),
        )


def test_config_identity_includes_the_critic_round_budget():
    from verifier.engine.kernel import MAX_CRITIC_ROUNDS

    default = offline.config_fields()
    assert default["critic_rounds"] == MAX_CRITIC_ROUNDS
    deeper = offline.config_fields(critic_rounds=MAX_CRITIC_ROUNDS + 1)
    assert deeper["critic_rounds"] == MAX_CRITIC_ROUNDS + 1
    # a different game budget is a different verifier's judgment
    assert deeper["config"] != default["config"]


def test_score_stamps_the_config_fields(tmp_path):
    state = state_dir(tmp_path)
    backend = FixtureBackend({"extract": lambda v: GOOD_COMMITS})

    features = offline.score(state, backend=backend,
                             kernel_factory=kernel_factory)

    cfg = offline.config_fields()
    assert features["config"] == cfg["config"]
    assert features["model"] == cfg["model"]
    assert features["seats"] == "sdk"
    assert features["service_tier"] == "n/a"


def test_stale_config_invalidates_the_derived_state(tmp_path):
    """Same pipeline, different run config (another model's judgment):
    the derived state is wiped and the trace re-extracted."""
    state = state_dir(tmp_path)
    backend = FixtureBackend({"extract": lambda v: GOOD_COMMITS})
    offline.score(state, backend=backend, kernel_factory=kernel_factory)
    meta = json.load(open(os.path.join(state, "extract.json")))
    meta["config"] = "another-models-config"
    json.dump(meta, open(os.path.join(state, "extract.json"), "w"))

    backend = FixtureBackend({"extract": lambda v: GOOD_COMMITS})
    features = offline.score(state, backend=backend,
                             kernel_factory=kernel_factory)

    assert any(c["role"] == "extract" for c in backend.calls)
    assert features["verdict"] == "verified"
    on_disk = json.load(open(os.path.join(state, "extract.json")))
    assert on_disk["config"] == offline.config_fields()["config"]


def test_stale_fingerprint_invalidates_the_cached_extraction(tmp_path):
    """An extraction cached under different prompts/schemas is a different
    verifier's work: score() wipes the derived state and re-extracts."""
    state = state_dir(tmp_path)
    offline.extract(state, backend=FixtureBackend(
        {"extract": lambda v: GOOD_COMMITS}))
    meta = json.load(open(os.path.join(state, "extract.json")))
    meta["fingerprint"] = "stale-pipeline"
    json.dump(meta, open(os.path.join(state, "extract.json"), "w"))

    backend = FixtureBackend({"extract": lambda v: GOOD_COMMITS})
    features = offline.score(state, backend=backend,
                             kernel_factory=kernel_factory)

    assert any(c["role"] == "extract" for c in backend.calls)   # re-extracted
    assert features["verdict"] == "verified"
    assert features["fingerprint"] == offline.pipeline_fingerprint()
    on_disk = json.load(open(os.path.join(state, "extract.json")))
    assert on_disk["fingerprint"] == offline.pipeline_fingerprint()


def test_live_trace_commits_are_never_invalidated(tmp_path):
    """Commits made by the solver itself (no extract.json) are the run's
    own protocol — a fingerprint change must not wipe them."""
    state = state_dir(tmp_path)
    offline.extract(state, backend=FixtureBackend(
        {"extract": lambda v: GOOD_COMMITS}))
    os.remove(os.path.join(state, "extract.json"))   # looks like a live trace

    # no extract fixture: score would fail if it tried to re-extract
    features = offline.score(state, backend=FixtureBackend({}),
                             kernel_factory=kernel_factory)

    assert features["verdict"] == "verified"
    assert features["extract_rounds"] == 0


# ---- ranking & selection -----------------------------------------------------

def F(verdict, final_verified=0, final_total=1, *, slice_verified=0,
      slice_size=1, obligations=0, contrary=0):
    return {
        "final_contrary_signals": contrary,
        "feature_schema_version": offline.FEATURE_SCHEMA_VERSION,
        "unscoreable": False,
        "verdict": verdict,
        # Diagnostic-only whole-slice fields.
        "slice_size": slice_size,
        "claims": {"verified": slice_verified},
        "obligations": {"verified": obligations},
        "obligations_total": obligations,
        # The only within-verdict ranking signal.
        "final_obligations": {"verified": final_verified,
                              "partial": final_total - final_verified},
        "final_obligations_total": final_total,
    }


def test_rank_is_lexicographic():
    # verdict dominates absolutely, whatever the lower components say
    assert offline.rank_key(F("verified", final_verified=0, final_total=9)) \
        > offline.rank_key(F("partial", final_verified=9, final_total=9))
    # same verdict: only contrary signals on the FINAL contract separate
    # candidates (fewer wins); the verified fraction is a diagnostic and
    # never ranks — two candidates differing only in it are a TIE
    assert offline.rank_key(F("partial", contrary=0)) \
        > offline.rank_key(F("partial", contrary=2))
    assert offline.rank_key(F("partial", final_verified=8, final_total=9)) \
        == offline.rank_key(F("partial", final_verified=7, final_total=9))
    assert offline.rank_key(None) == (-1, 0)


def test_rank_ignores_narration_decomposition_diagnostics():
    compact = F("partial", final_verified=1, final_total=2,
                slice_verified=1, slice_size=2, obligations=3)
    padded = F("partial", final_verified=1, final_total=2,
               slice_verified=900, slice_size=901, obligations=10_000)

    assert offline.rank_key(compact) == offline.rank_key(padded) == (2, 0)
    assert selection.scalar(compact) == selection.scalar(padded)


def test_legacy_and_malformed_feature_schemas_fail_closed():
    legacy = {
        "unscoreable": False, "verdict": "verified", "slice_size": 1,
        "claims": {"verified": 1}, "obligations": {"verified": 99},
    }
    malformed = F("verified", final_verified=1, final_total=1)
    malformed["final_obligations_total"] = 2

    assert offline.rank_key(legacy) == (-1, 0.0)
    assert offline.rank_key(malformed) == (-1, 0.0)
    assert selection.scalar(legacy) == 0.0


def test_selection_strategies():
    rows = [
        {"answer": "7", "solved": False, "features": F("unverified")},
        {"answer": "7", "solved": False, "features": F("unverified")},
        {"answer": "42", "solved": True,
         "features": F("verified", final_verified=1, final_total=1,
                       obligations=4)},
    ]
    assert selection.argmax(rows)["answer"] == "42"
    # one trace one vote: the wrong majority wins — the baseline's failure mode
    assert selection.majority_vote(rows)["answer"] == "7"
    # banded verifier weights: one verified outvotes two unverified
    assert selection.weighted_vote(rows)["answer"] == "42"
    assert selection.oracle(rows) is True
    assert selection.pass_at_1(rows) == pytest.approx(1 / 3)


def test_voting_falls_back_to_argmax_without_answers():
    rows = [{"answer": None, "solved": False, "features": F("partial")},
            {"answer": None, "solved": False,
             "features": F("verified", final_verified=1)}]
    assert selection.majority_vote(rows)["features"]["verdict"] == "verified"


@pytest.mark.parametrize("relative, changed", [
    ("task.md", "a different task"),
    ("final.txt", "a different answer"),
    ("hook_feed.jsonl", "".join(json.dumps(e) + "\n" for e in [*FEED, FEED[0]])),
    ("world/app/result.txt", "a different delivered artifact"),
    ("world_manifest.json", '{"captured_roots": []}'),
    ("baseline/files.json", '{}'),
])
def test_changed_inputs_rebuild_extraction_and_judgments(tmp_path, relative, changed):
    state = state_dir(tmp_path)
    offline.score(state, backend=FixtureBackend({"extract": GOOD_COMMITS}),
                  kernel_factory=kernel_factory)
    path = tmp_path / "gv" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(changed)
    fresh = FixtureBackend({"extract": GOOD_COMMITS})
    checks = StubDischarger("failed")

    def factory(sc):
        return Kernel(sc.store, sc.log, FixtureBackend({"obligations": COMPILE}),
                      sc.artifacts, critic_role=None, discharger=checks)

    features = offline.score(state, backend=fresh, kernel_factory=factory)

    assert any(c["role"] == "extract" for c in fresh.calls)
    assert checks.seen
    assert features["verdict"] == "unverified"
    assert path.read_text() == changed
    meta = json.loads((tmp_path / "gv" / "extract.json").read_text())
    assert meta["immutable_input_seal"] == offline.immutable_input_seal(state)


@pytest.mark.parametrize("change", ["missing_stamp", "changed_trace"])
def test_claim_store_is_rebuilt_without_replacing_live_commits(tmp_path, change):
    state = state_dir(tmp_path)
    offline.score(state, backend=FixtureBackend({"extract": GOOD_COMMITS}),
                  kernel_factory=kernel_factory)
    (tmp_path / "gv" / "extract.json").unlink()
    log = EventLog(os.path.join(state, "events.jsonl"))
    if change == "missing_stamp":
        (tmp_path / "gv" / "verification.json").unlink(missing_ok=True)
    else:
        events = list(log.replay())
        next(e for e in events if e["type"] == "tool_result")[
            "payload"]["excerpt"] = "43"
        (tmp_path / "gv" / "events.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events))
    commits = [e for e in log.replay() if e["type"] == "commit"]
    checks = StubDischarger("failed")

    def factory(sc):
        return Kernel(sc.store, sc.log, FixtureBackend({"obligations": COMPILE}),
                      sc.artifacts, critic_role=None, discharger=checks)

    features = offline.score(state, backend=FixtureBackend({}),
                             kernel_factory=factory)

    assert features["verdict"] == "unverified"
    assert checks.seen
    assert [e for e in log.replay() if e["type"] == "commit"] == commits
