# Concurrent final-contract compilation

## Problem

Offline scoring ran `extract → compile final contract → critic → …` on one
critical path. The final contract is compiled through the candidate-blind
authority boundary (`docs/contract-cache-proposal.md`, "authority-only
compilation"): its Compiler and Critic read the task authority and nothing
about the candidate, so nothing in that compilation depends on extraction.
Serializing the two stages cost the extraction time (20–60 s per candidate)
for no verification benefit.

## Mechanism

`GV_FINAL_CONTRACT_PRECOMPILE=1` makes `offline.score` start the unchanged
authority-only compile-and-critique in a thread before extraction
(`verifier/engine/precompile.py`). The thread uses the kernel's own backend
instance and role registry, records its model spans under a pseudo claim id,
and buffers the events and fail-closed evidence the kernel's critique loop
would have written. After extraction the kernel adopts the result only when
its own contract key (authority view, withheld compile variables, backend
and role identities, critic budget, repair protocol) is identical; the
buffered events are then replayed under the real final claim id and a
`contract_precompiled` event records the adoption. A bypass, a key mismatch
or a failed thread falls back to the unchanged inline compile; the thread's
work is never reused partially.

The flag is an execution schedule, not part of the scoring identity: the
adopted contract is exactly the contract the inline path would have
compiled with the same seats. It is off by default and enabled by the v1
full-offline treatment profile.

## What it does not change

The Compiler and Critic prompts, schemas, tools, budgets and the Critic's
fail-closed rule are unchanged; the Oracle, executor and Judger never see
the difference.
