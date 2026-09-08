# Canary `fad9290`: first video control diagnostic

This is an in-progress diagnostic snapshot, not a final evaluation result. It
covers source revision `fad9290` and the first control run ending in
`1d8e1857`. No historical gold data was consulted. The figures below must not
be used to claim a cost or latency ratio.

## Runtime snapshot

At roughly 9 minutes 33 seconds, 36 model calls had completed successfully and
all had complete usage records. They accounted for 1,809 model-seconds,
3.871M input tokens (833k uncached and 3.038M cached), and 75.4k output tokens.
The run did not yet have a terminal result.

Discharge and Judger dominated this snapshot: six Discharge calls consumed
773 seconds and five Judger calls consumed 504 seconds. Together they were
about 71% of accumulated model time and 79% of input tokens. Compiler and
Oracle calls were not the observed long pole. Parallel obligations reduced
wall-clock time, but serial `Discharge -> Judger -> redo/repair` chains created
the tail.

All completed calls exited successfully with complete usage. The captured
artifacts contained no certificate, TLS, package-install, fetch, permission, or
backend error signal. This is evidence only for this snapshot; it is not a
general statement about every network path in the runtime.

## Three-step method binding chain

One file-presence obligation required an initial execution and two redos. The
three evidence artifacts remained present and readable, contained only small
text scripts and outputs, and were not image-dependent. Each audit found the
reported positive lookup sufficient; the failures concerned method identity or
input provenance rather than missing, truncated, or inaccessible evidence.

1. Initial execution: the agent used a relative-path shell lookup rather than
   the exact verifier captured-world lookup promised by the method. It also
   labelled the world manifest as `task_given`. The audit returned
   `sufficient=yes, faithful=no`.
2. Redo 1: the observed procedure now matched the promised lookup, and the
   manifest was correctly labelled `verifier`. The delivered subject file was
   still incorrectly labelled `task_given`, so the audit again returned
   `sufficient=yes, faithful=no`.
3. Redo 2: the subject was labelled `delivered`, while the manifest remained
   `verifier` and the task specification `task_given`. The audit returned
   `sufficient=yes, faithful=yes`, and the method bound successfully.

The repeated `faithful=no` label therefore represented two different defects.
The first redo made real progress; identical top-level verdict labels alone are
not a safe basis for stopping a redo chain.

## Treatment comparison to make

The immediate comparison is whether the existing verifier-owned
`filesystem.exists` adapter naturally removes this agent execution and Judger
redo chain while preserving the outcome. A deterministic adapter can emit its
own method identity and provenance when those facts are known from the sealed
world mapping.

The harness must not infer that every path in a captured tree is `delivered`.
A captured tree may also contain genuine task-given inputs, particularly when
no baseline exists. Any future mechanical provenance filling or validation
must use explicit source evidence rather than path membership alone.

## Completed control, before treatment completion

The two-candidate control process subsequently completed with exit code 0,
no result-audit errors, and complete usage accounting. End-to-end wall time was
**3,274.664 seconds (54.6 minutes)**, including both candidates and the cold
toolchain fill. There were 55 recorded CLI model attempts, all with successful
provider returns. This does not mean their contracts or evidence passed audit:

| Role | Attempts | Accumulated model seconds |
| --- | ---: | ---: |
| Extract | 3 | 129.90 |
| Obligation Compiler | 10 | 133.96 |
| Critic | 10 | 227.60 |
| Oracle Compiler | 8 | 195.33 |
| Discharge | 13 | 3,189.36 |
| Judger | 11 | 1,033.51 |

Usage totals were 1,701,698 uncached input tokens, 13,385,984 cached input
tokens, and 202,604 output tokens. These are token counts, not a dollar price.
Discharge and Judger together represented approximately 86% of accumulated
model time. Parallel accumulated time is not wall time or a causal critical
path. Treatment is still running; **there is no completed A/B ratio yet**.

Before this control finished, one trace's final task-contract verdict had
already settled while auxiliary support-claim verification continued for at
least another 24 minutes. The frozen source lacks a per-call claim/attempt
join, so individual Compiler/Critic token totals cannot safely be attributed
by temporal ordering under parallel execution. This motivates operator-only
model spans; it does not authorize skipping the support audit.

## Follow-up selection: coverage, not adapter count

Read-only inspection of 40 complete Oracle method emissions from the older
two-task A/B and five emissions from this canary found little incremental
coverage for another small expression interpreter. Two canary emissions were
only truncated projections and remain unknown. A bounded JSON/integer/string
expression IR would add 4 of the 45 emissions, all repaired variants of one
file-length requirement; line-splitting would add one more. No IR or Wasm
runtime was installed or implemented.

A closer review of 24 record-like older method emissions found ten faithfully
expressible with sufficient retained evidence. Eight already fit existing
ledger adapters. A new call/result join would cover only two additional
methods, and only one would produce a decisive result in those records. Four
other methods require missing response boundaries; ten require semantics
outside a closed record algebra. Treating these as supported would change the
verification standard. A broader ledger interpreter is therefore not the
current priority.

The concrete repair-context gap is narrower: production execution snapshots
scratch into the artifact store and then removes scratch; the redo receives
the rejection reasons but no direct locator for its previous snapshot. The
completed trace's scratch and workspace were empty, while four executions of
one method retained different rebuilt scripts. Supplying the latest snapshot
as **untrusted repair material** can help locate defects and reuse scaffolding.
It cannot supply an absent authoritative grammar or make an unfaithful method
faithful. Fresh execution and independent auditing remain required; savings
from this follow-up are not yet measured.

## First paired candidate: contract drift, not adapter direction drift

The first video candidate (`extract-moves-from-video__KFN5iFo`) has now
finished in both arms. Control returned `unverified`; treatment returned
`refuted`. The whole treatment process is still running, so there is still
no completed process-level A/B cost or wall-time ratio. Its elapsed time has
already exceeded one tenth of the completed control's 54.6 minutes; this
particular pair cannot deliver 10x latency reduction.

The candidate task, final answer, and world-manifest bytes match across arms.
The control's final decomposition failed Critic after repair oscillation and
never materialized final obligations. Treatment instead completed a cold
authority-contract build with six final MUST obligations. One required the
last solver action to be a standalone echo command with no later action.
The retained record's last solver call was a file-writing command instead.
After repairing an incorrect ledger terminal watermark, a fresh Judger
accepted the negative audit. The same solver-record facts existed in control,
but control never executed that final obligation.

For this candidate, treatment had a contract-cache miss and publish, no hit;
neither arm had method-cache events, and treatment had no typed adapter
execution or fallback. The refutation therefore did not come from reused
evidence or direct adapter execution. This is differing contract coverage,
not opposite outcomes on a shared exact obligation. The authority-only
compilation path and stochastic claim/decomposition differences remain
confounded in a single full-pipeline pair. Do not call this a proven quality
regression, a proven improvement, or a successful zero-loss comparison.
