"""The final-contract precompile: with a task statement and a live world,
the final's contract (compiler + critic) and each obligation's first
method batch are prepared in the background while the solver works; the
final's commit then spends neither, and a changed task statement is never
served a stale contract.

Fixture-backed and instrument-free.
"""
from __future__ import annotations

import os

from verifier.core.events import EventLog
from verifier.core.model import Claim, CLAIM_ASSERTION, CLAIM_FINAL
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.engine.game import BindingGame
from verifier.engine.contract_compiler import ContractCompiler
from verifier.engine.kernel import Kernel
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.agents import CRITIC_AGENT
from verifier.llm.schema import FixtureBackend


CONTRACT = {"obligations": [
    {"text": "the deliverable exists at the task's address",
     "required_basis": "world", "refutable_by_absence": True},
    {"text": "the deliverable's content is what the task requires",
     "required_basis": "world"},
]}
PASS = {"verdict": "pass", "counterexample_world": "n/a",
        "missing_condition": "n/a"}
ORACLE = {"methods": [{"procedure": "inspect the address", "artifact": "log",
                       "level": "direct", "basis": "world",
                       "strength": "decisive", "bridges": []}]}
JUDGE_YES = {"sufficient": "yes", "faithful": "yes", "degradation": "n/a",
             "reason": "ok"}


class StubDischarger:
    def discharge(self, row, obligation, method=None, tag=""):
        return DischargeResult(witness_kind="execution", outcome="holds",
                               witness=None, artifact="d", instrument="stub",
                               basis="world")


def build(tmp_path, roles, task="Deliver the thing at /app/out.txt"):
    if task is not None:
        (tmp_path / "task.md").write_text(task)
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend(roles)
    artifacts = ArtifactStore(os.path.join(tmp_path, "art"))
    binder = BindingGame(backend, artifacts, log, StubDischarger())
    kernel = Kernel(store, log, backend, artifacts, critic_role=CRITIC_AGENT,
                    discharger=StubDischarger(), binder=binder,
                    parallel_obligations=2)
    return store, log, backend, kernel


def events(log, etype):
    return [e["payload"] for e in log.replay() if e["type"] == etype]


def test_precompiled_contract_serves_the_final(tmp_path):
    store, log, backend, kernel = build(
        tmp_path, {"obligations": CONTRACT, "critic": PASS,
                   "oracles": ORACLE, "judger": JUDGE_YES})
    assert kernel.start_final_precompile() is True
    kernel.contracts.precompile_thread.join()

    pre = events(log, "contract_precompiled")
    assert pre and pre[0]["obligations"] == [
        o["text"] for o in CONTRACT["obligations"]]
    roles = [c["role"] for c in backend.calls]
    assert roles.count("obligations") == 1 and roles.count("critic") == 1
    assert roles.count("oracles") == 2          # one batch per obligation
    compiled_with = backend.calls[0]["variables"]
    assert compiled_with["claim_kind"] == "final"

    final = store.add_claim(Claim(CLAIM_FINAL, "the task is done"))

    assert store.status(final) == "verified"
    assert events(log, "contract_cache_hit")[0]["claim"] == final
    roles = [c["role"] for c in backend.calls]
    # neither the compiler nor the critic nor the oracle ran again
    assert roles.count("obligations") == 1 and roles.count("critic") == 1
    assert roles.count("oracles") == 2 and roles.count("judger") == 2
    served = [e for e in events(log, "methods_enumerated")
              if e.get("cached")]
    assert len(served) == 2 and all(e["claim"] == final for e in served)
    texts = {o["text"] for o in store.obligation_children(final)}
    assert texts == {o["text"] for o in CONTRACT["obligations"]}


def test_a_changed_task_statement_is_compiled_fresh(tmp_path):
    store, log, backend, kernel = build(
        tmp_path, {"obligations": CONTRACT, "critic": PASS,
                   "oracles": ORACLE, "judger": JUDGE_YES})
    kernel.start_final_precompile()
    kernel.contracts.precompile_thread.join()
    (tmp_path / "task.md").write_text("A different task entirely")

    final = store.add_claim(Claim(CLAIM_FINAL, "the task is done"))

    assert store.status(final) == "verified"
    assert events(log, "contract_cache_hit") == []
    roles = [c["role"] for c in backend.calls]
    assert roles.count("obligations") == 2 and roles.count("critic") == 2


def test_non_final_claims_never_use_the_precompiled_contract(tmp_path):
    store, log, backend, kernel = build(
        tmp_path, {"obligations": CONTRACT, "critic": PASS,
                   "oracles": ORACLE, "judger": JUDGE_YES})
    kernel.start_final_precompile()
    kernel.contracts.precompile_thread.join()

    cid = store.add_claim(Claim(CLAIM_ASSERTION, "an intermediate fact"))

    assert store.status(cid) == "verified"
    assert events(log, "contract_cache_hit") == []


def test_no_task_statement_means_nothing_to_precompile(tmp_path):
    store, log, backend, kernel = build(
        tmp_path, {"obligations": CONTRACT, "critic": PASS,
                   "oracles": ORACLE, "judger": JUDGE_YES}, task=None)
    assert kernel.start_final_precompile() is False
    assert backend.calls == []


def test_a_precompile_the_critic_rejects_leaves_the_final_untouched(tmp_path):
    attack = {"verdict": "insufficient",
              "counterexample_world": "SUFFICIENCY GAP: x",
              "missing_condition": "ADD: y"}
    store, log, backend, kernel = build(
        tmp_path, {"obligations": CONTRACT, "critic": attack,
                   "oracles": ORACLE, "judger": JUDGE_YES})
    kernel.critic_rounds = 1
    kernel.start_final_precompile()
    kernel.contracts.precompile_thread.join()

    pre = events(log, "contract_precompiled")
    assert pre and "critic" in pre[0]["error"]
    # nothing was written to the store for any real claim
    assert store.db.execute(
        "SELECT COUNT(*) FROM evidence WHERE claim_id != ?",
        (ContractCompiler.PRECOMPILE_ID,)).fetchone()[0] == 0
    final = store.add_claim(Claim(CLAIM_FINAL, "the task is done"))
    # the real final compiles fresh and fails closed on its own account
    assert store.status(final) == "unverified"
    assert events(log, "contract_cache_hit") == []
