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
path. This was the first completed arm; the completed pair is reported below.

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
`refuted`. The whole treatment process subsequently finished; its process-level
cost and wall-time totals appear below. This pair did not deliver 10x latency
reduction.

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

## Completed two-candidate video pair

Both process arms exited 0 with no result-audit errors and complete provider
usage accounting. The source is still the frozen `fad9290` revision, not the
later repair-context, span-accounting, or diagnostic replay changes.

| Metric | Control | Treatment | Treatment / control |
| --- | ---: | ---: | ---: |
| End-to-end wall seconds | 3,274.664 | 1,862.019 | 0.569 |
| Model attempts | 55 | 78 | 1.418 |
| Uncached input tokens | 1,701,698 | 2,051,455 | 1.206 |
| Cached input tokens | 13,385,984 | 11,325,440 | 0.846 |
| Output tokens | 202,604 | 260,203 | 1.284 |

Latency improved **1.76x**, from 54.6 to 31.0 minutes, including both
candidates and the cold seed. Treatment includes one schema-error attempt
with fully retained usage. Reasoning tokens are already a subset of output
tokens and are not charged a second time. No unverified Luna dollar price is
assumed. Under any fixed nonnegative linear prices for the three token bins,
their weighted cost ratio cannot be below 0.846: this pair cannot establish
a 10x token-cost reduction regardless of those prices.

The second candidate was `refuted` in both arms; the first candidate differs
as detailed above. One treatment contract-cache hit and one typed adapter
execution occurred across the pair, both on the second candidate. There were
no method-cache hits. Full result equivalence is **not established** because
the exact contracts and coverage differ. The gcode and compressor process
pairs remain separate pending measurements, not zero-cost completed work.

The next diagnostic fixes the entire post-extraction, pre-verification graph
and independently Critic-approved obligation contracts. It must count shared
preparation in full against each arm, and still run fresh effects, Oracle,
execution, and Judger. This is an explicit offline adapter-isolation test,
not a change to production scheduling and not a substitute for full-e2e
acceptance. Compiler/Critic source logs must remain unadjudicated; a completed
arm's verdicts or certificates cannot be used to manufacture a shared setup.

## Gcode control: completed, cost incomplete

A read-only snapshot before completion of the first gcode control candidate found no
certificate or completed result row yet. The runner and Codex processes were
alive, and the discharge script/output files were still changing. The active
semantic check had progressed from a method that only checked the output file,
through a rejected hard-coded answer, to an actual rasterization whose
independent decoder was still insufficient. Existing work artifacts remained
present; this snapshot did not show a cleanup-loss problem.

One discharge CLI return took 445.47 seconds and exited 0 but produced invalid
usage telemetry, leaving its token counts unknown. It must remain an
incomplete paid attempt, not a zero-token success. The frozen run lacks the
new causal sidecar; cumulative role times and temporal proximity cannot
establish its dependency critical path or allocate concurrent costs by claim.
The control process subsequently completed both candidates in **3,686.011
seconds (61.4 minutes)**, exit 0, with no result-audit errors. It recorded 70
started model attempts: 27 discharge, 20 Oracle, 13 Judger, 4 Compiler, 4
Critic and 2 Extractor calls. One discharge attempt still has unknown usage.
The outer driver therefore exited 1 at this accounting gate and did not run
the gcode treatment or either compressor arm. Its session is terminal, not
merely quiet; it must not be described as continuing in the background.

This is not a completed gcode A/B comparison. Do not extrapolate a cost or
latency ratio from it. The new
[counter-only response recovery](usage-response-recovery.md) cannot establish
what happened in that historical stream or retroactively repair its costs.
