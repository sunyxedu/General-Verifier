"""A task-start identity an obligation quotes is restored to the recorded
value when the compiler's copy is inexact; nothing else is touched."""
from __future__ import annotations

import hashlib
import json
import os

from verifier.core.events import EventLog
from verifier.core.model import Claim, CLAIM_ASSERTION
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.engine.kernel import Kernel
from verifier.engine.contract.quoted_identities import recorded_identities, restore
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend

DIGEST = hashlib.sha256(b"the task's input").hexdigest()
OTHER = hashlib.sha256(b"another file").hexdigest()


def baseline(state, files):
    os.makedirs(os.path.join(state, "baseline"), exist_ok=True)
    with open(os.path.join(state, "baseline", "manifest.json"), "w") as f:
        json.dump({"root": "/nowhere", "files": files}, f)


def test_a_copy_that_lost_a_character_is_restored():
    text = f"input.dat has SHA-256 digest `{DIGEST[:-1]}`."
    fixed, restored = restore(text, {DIGEST, OTHER})
    assert fixed == f"input.dat has SHA-256 digest `{DIGEST}`."
    assert restored == [(DIGEST[:-1], DIGEST)]


def test_a_copy_with_an_altered_character_is_restored():
    wrong = DIGEST[:40] + ("0" if DIGEST[40] != "0" else "1") + DIGEST[41:]
    fixed, restored = restore(f"digest {wrong}", {DIGEST})
    assert fixed == f"digest {DIGEST}" and len(restored) == 1


def test_exact_and_unrelated_tokens_are_left_alone():
    text = f"{DIGEST} and {'ab' * 32} and deadbeef"
    assert restore(text, {DIGEST}) == (text, [])


def test_an_abbreviated_identity_is_not_expanded():
    text = f"commit {DIGEST[:16]}"
    assert restore(text, {DIGEST}) == (text, [])


def test_an_ambiguous_near_copy_is_left_alone():
    twin = DIGEST[:20] + OTHER[20:]
    text = f"digest {DIGEST[:-1]}"
    # two recorded identities agree with the copy equally far: no guess
    assert restore(text, {DIGEST[:-1] + "0", DIGEST[:-1] + "1"}) == (text, [])
    # one agrees much further than the other: that is the one copied
    assert restore(text, {DIGEST, twin})[0] == f"digest {DIGEST}"


def test_recorded_identities_come_from_the_task_start_record(tmp_path):
    baseline(str(tmp_path), {"input.dat": DIGEST, "big.bin": "size:99",
                             "link": "link:/x"})
    with open(tmp_path / "baseline" / "git.json", "w") as f:
        json.dump({"head": "a" * 40, "note": "not an id: " + "b" * 20}, f)
    assert recorded_identities(str(tmp_path)) == {DIGEST, "a" * 40}


def test_the_kernel_materializes_the_recorded_value(tmp_path):
    baseline(str(tmp_path), {"input.dat": DIGEST})
    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    seen = []

    class Discharger:
        def discharge(self, row, obligation, method=None, tag=""):
            seen.append(obligation.text)
            return DischargeResult(witness_kind="execution", outcome="holds",
                                   witness=None, artifact="", instrument="stub",
                                   basis="world")

    def compile_role(_v):
        return {"obligations": [{
            "text": f"input.dat is unchanged: its SHA-256 is {DIGEST[:-1]}",
            "required_basis": "world"}]}

    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    Kernel(store, log, FixtureBackend({"obligations": compile_role}),
           ArtifactStore(os.path.join(tmp_path, "art")), critic_role=None,
           discharger=Discharger(), live_world=True)
    cid = store.add_claim(Claim(CLAIM_ASSERTION, "the input is untouched"))

    assert store.status(cid) == "verified"
    assert seen == [f"input.dat is unchanged: its SHA-256 is {DIGEST}"]
    restored = [e["payload"] for e in log.replay()
                if e["type"] == "obligation_identity_restored"]
    assert restored and restored[0]["repairs"][0]["restored"] == [
        {"copied": DIGEST[:-1], "recorded": DIGEST}]
