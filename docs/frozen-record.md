# Frozen records: one immutable ledger prefix per verified claim

## Problem

`events.jsonl` is both the candidate's retained record and the verifier's own
event bus. While the executor scans it for a record obligation, the kernel
appends `judger_verdict`, `discharge`, `status_change` and other audit events
to the same file. The executor reports a complete scan of N records; the
independent Judger recounts N+k and rejects the execution as "stale". The
retained gcode pair showed this as three consecutive `faithful=no` redos of a
trivial "at least one bash call" obligation (eight paid seats, about 830 s on
the critical path) and as single redos in three treatment obligations.

## Mechanism

`Kernel.verify` now publishes, before any seat runs, the exact ledger prefix
through the claim's own commit event as
`state/record/events-through-<seq>-<sha256[:16]>.jsonl` (read-only, verbatim
lines, content-addressed, idempotent). The boundary comes from the Builder's
first-seen commit seq; a bare kernel without a Builder freezes at the ledger's
end. Every seat verifying that claim receives the frozen path as its
`ledger_path`: Obligation Compiler, Critic, Oracle, executor, Judger, and the
Judger prefetch packet. Typed adapters keep reading the live ledger's source
projection, which offline is identical and online is the adapter's own
declared scope. A publication failure keeps the live ledger and records the
error in a `record_frozen` event.

The record directory is derived state: offline cleanup, runtime-state seals
and the official-scoring state audit classify it with `artifacts/` and
`work/`.

## What it does not change

No role, prompt contract, schema, verdict rule, cache or retry budget
changes. The executor and Judger templates only name the record as an
immutable prefix whose count and last seq are final. The verdict semantics
for record obligations are unchanged: the record is the claim-time record.
