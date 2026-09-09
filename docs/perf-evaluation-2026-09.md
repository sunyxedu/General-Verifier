# Cost and latency optimization: evaluation log (September 2026)

Goal: task-agnostic online/offline optimizations that keep the
`Extractor → Obligation Compiler → Critic → Oracle Compiler → executor →
independent Judger` architecture, evaluated against original main
(`be3581a`) with GPT-5.6-Luna (medium, standard) on sealed Terminal-Bench
candidates, targeting at least 5× on complete cost and end-to-end latency
without result loss.

## Measurement

- Cold pairs: `build_full_offline_comparison_plan.py` + the v1 coordinator
  (`run_full_offline_comparison.py`); one fresh process per candidate and arm,
  spawn-to-exit wall, every attempt billed, cost only when every attempt's
  usage is known. Rates: uncached input $0.20/M, cached $0.02/M, output
  $1.20/M.
- Warm batches: `run_full_offline_warm_worker.py`; every candidate of one
  task family in one treatment process with process-local contract and
  accepted-method reuse. Compared against the same cold controls.
- Summaries: `summarize_full_offline_comparison.py <run> [--labels
  operator-manifest.json] [--control-root <cohort run>]`.

## Results

### gcode pair, treatment `3ebdc9d` (2026-09-09, run `full-offline-3ebdc9d-20260909`)

Treatment: atomic Oracle binding, frozen records, billing-frame tolerance,
observed values in `ledger.event` receipts. Profile: typed adapters, Judger
prefetch; no final scope, no typed Judger packet.

| Candidate | Control cost / wall | Treatment cost / wall | Verdict control → treatment |
| --- | ---: | ---: | --- |
| `ELuvDeE` | $0.263 / 416 s | $0.102 / 245 s | unverified → partial |
| `uXCbo9H` | $0.553 / 737 s | $0.171 / 494 s | refuted → partial |
| **Total** | **$0.817 / 1153 s** | **$0.273 / 739 s** | **cost ×2.99, latency ×1.56** |

The control's `refuted` on `uXCbo9H` was a false refutation (a trailing
newline in `/app/out.txt` read as "not exactly the flag"). Control latency
on `ELuvDeE` varied 416–1192 s across two runs of the same input; two
candidates cannot establish a ratio.

### 8-candidate cohort, treatment `82bef17` (run `full-offline-82bef17-tb8-20260909`)

Four task families × one positive and one negative candidate
(`terminal-bench-general-inputs-20260908-v1`). Profile adds the final-only
certificate scope and the typed Judger packet. Treatment accounting was
re-run read-only with the corrected scratch-cwd rules (`--reaccount`).

| Candidate (label) | Control verdict / cost / wall | Treatment verdict / cost / wall | cost × | latency × |
| --- | --- | --- | ---: | ---: |
| sqlite `4JQh54h` (1) | verified / $1.211 / 2955 s (11 redos) | partial / $0.239 / 1217 s (3 redos) | 5.06 | 2.43 |
| sqlite `hUXYSvy` (0) | refuted / $0.425 / 898 s | refuted / $0.152 / 925 s | 2.79 | 0.97 |
| chess `xGt6Pve` (1) | unverified / $0.653 / 1224 s | unverified / $0.236 / 1575 s | 2.77 | 0.78 |
| chess `Da8xED4` (0) | refuted¹ / $0.848 / 1474 s | unverified / $0.287 / 1671 s | 2.96 | 0.88 |
| gcode `ELuvDeE` (1) | refuted² / $0.525 / 862 s | partial / $0.037 / 195 s | 14.13 | 4.42 |
| gcode `HFh2qJM` (0) | refuted / $0.276 / 707 s | refuted / $0.036 / 126 s | 7.66 | 5.63 |
| video `KFN5iFo` (1) | compile failed³ / $0.194 / 593 s | partial / $0.072 / 300 s | — | — |
| video `tJdCJ8R` (0) | unverified / $0.421 / 1321 s | compile failed⁴ / $0.033 / 125 s | — | — |
| **Six decided pairs** | **$3.938 / 8119 s** | **$0.987 / 5708 s** | **3.99** | **1.42** |
| All eight jobs | $4.553 / 10033 s | $1.093 / 6133 s | 4.17 | 1.64 |

¹ False refutation: the candidate issued the standalone submit command at
seq 79; the control's executor counted the verifier's own later audit
events as "work after submission" (the stale-ledger defect) and the Judger
accepted it on the third round.
² False refutation of a correct answer: the control tested "at least one
bash tool call" as `payload.tool == "bash"` on a record whose tool is
`exec` (codex exec-wrapper dialect).
³ Compiler/Critic oscillation over the modality of "reasoning text";
the control raises on an empty final contract.
⁴ The Critic demanded a runner-protocol append that the Compiler's own
gate excludes; the mechanical append guard consumed every repair. Fixed in
`01293a8` (a declined append is reviewed instead of deadlocked).

Best-of-N within each family (rank: verified > supported > partial >
unverified > refuted): the treatment selects the positive candidate in the
sqlite and gcode families, ties in chess, and cannot decide video (one
candidate lost to the deadlock above). The control's two extra refutations
are both spurious.

Where the treatment's remaining time goes: on hard semantic obligations
(chess winning-move set, SQLite recovery maximality) the executor runs
200–330 s per attempt and the Judger 80–110 s, and fidelity redos
(scope narrower than the Oracle's promised exhaustive derivation) repeat
that two to three times; the contract compile took up to four Critic
rounds (209 s on chess).

### 8-candidate cohort, treatment `01293a8` cold (run `full-offline-01293a8-tb8-20260909`)

Treatment-only reruns of the same sealed inputs, paired with the
`82bef17` cohort's control jobs. Adds the record shape for the Oracle, the
pre-execution sufficiency audit, the concurrent final-contract compile,
eight obligation slots, and the declined-append review.

| Candidate (label) | Control verdict / cost / wall | Treatment verdict / cost / wall | cost × | latency × |
| --- | --- | --- | ---: | ---: |
| sqlite `4JQh54h` (1) | verified / $1.211 / 2955 s | unverified / $0.121 / 420 s | 10.04 | 7.04 |
| sqlite `hUXYSvy` (0) | refuted / $0.425 / 898 s | refuted / $0.070 / 410 s | 6.07 | 2.19 |
| chess `xGt6Pve` (1) | unverified / $0.653 / 1224 s | unverified / $0.035 / 145 s | 18.81 | 8.45 |
| chess `Da8xED4` (0) | refuted¹ / $0.848 / 1474 s | partial / $0.102 / 437 s | 8.32 | 3.37 |
| gcode `ELuvDeE` (1) | refuted² / $0.525 / 862 s | partial / $0.030 / 106 s | 17.36 | 8.15 |
| gcode `HFh2qJM` (0) | refuted / $0.276 / 707 s | refuted / $0.050 / 286 s | 5.51 | 2.47 |
| video `KFN5iFo` (1) | compile failed³ / $0.194 / 593 s | unverified / $0.063 / 283 s | 3.06 | 2.10 |
| video `tJdCJ8R` (0) | unverified / $0.421 / 1321 s | refuted / $0.061 / 262 s | 6.93 | 5.04 |
| **All eight** | **$4.553 / 10033 s** | **$0.531 / 2349 s** | **8.57** | **4.27** |

Best-of-N within each family: the treatment selects the positive candidate
in sqlite, gcode and video and the negative one in chess (the negative's
semantic obligation ended `unobtainable`, which reads out as `partial`, while
the positive's methods were rejected as insufficient and read out as
`unverified`); the control selects correctly in sqlite and chess (the latter
through the spurious refutation above) and fails in gcode and video.

The pre-execution sufficiency audit stopped the executor before the two
methods the Judger would have rejected after 200–330 s each on chess, and
the fresh video pair no longer deadlocks on a runner-protocol append.

### Warm batches, treatment `25f20be` (run `full-offline-warm-25f20be-tb8-20260909`)

Both candidates of one task family scored in one process with the
process-local contract and method caches on (`GV_CONTRACT_CACHE=1`,
`GV_METHOD_CACHE=1`) over the same treatment profile; controls are the
`82bef17` cohort's control jobs. The chess family ran from the earlier
`01293a8` warm root.

| Candidate (label) | Treatment verdict / cost / wall | Contract cache | cost × | latency × |
| --- | --- | --- | ---: | ---: |
| video `KFN5iFo` (1) | unverified / $0.106 / 427 s | miss, published | 1.83 | 1.39 |
| video `tJdCJ8R` (0) | refuted / $0.060 / 159 s | hit | 6.99 | 8.33 |
| gcode `ELuvDeE` (1) | unverified / $0.038 / 202 s | miss, published | 13.71 | 4.26 |
| gcode `HFh2qJM` (0) | refuted / $0.053 / 212 s | hit | 5.26 | 3.33 |
| sqlite `4JQh54h` (1) | unverified / $0.077 / 422 s | miss, published | 15.77 | 7.00 |
| sqlite `hUXYSvy` (0) | unverified / $0.105 / 532 s | hit (+1 method reuse) | 4.06 | 1.69 |
| **Six candidates** | **$0.439 / 1954 s** (control $3.052 / 7336 s) | | **6.96** | **3.75** |
| chess `xGt6Pve` (1), `01293a8` root | unverified / $0.373 / 3598 s | miss, published | 1.75 | 0.34 |
| chess `Da8xED4` (0), `01293a8` root | unverified / $0.166 / 927 s | hit | 5.11 | 1.59 |

The second candidate of every family skipped the Compiler/Critic rounds
through a contract-cache hit. Accepted-method reuse applied once: the
`method_cache_bypass` telemetry added in `25f20be` shows that most settled
methods end `failed` or `unobtainable` (no direction to reuse) or are agent
methods without a reusable typed signature. The chess positive candidate's
first warm run hit six Judger redo rounds on its semantic obligation,
including one 1357 s Judger call; that is the redo budget doing its work on
an obligation neither arm can decide, not a cache effect.

Two profile gaps found while reading these runs are fixed after `25f20be`:
the `01293a8` cold run still paid for a precompile thread failure
(`TypeError`, inline fallback after 40–70 s on two candidates), and the warm
profile never used the concurrent precompile because it was disabled
whenever the contract cache was on. Commit `ee97910` composes the two: the
thread yields to a published or in-flight key and otherwise the kernel's own
miss adopts and publishes its result.

### 8-candidate cohort, treatment `ee97910` cold (run `full-offline-ee97910-tb8-20260909`)

Same protocol as the `01293a8` cold run (treatment-only, paired with the
`82bef17` cohort's control jobs), with the precompile thread failure fixed
and the precompile composed with the contract cache. The concurrent
compile was adopted on every candidate.

| Candidate (label) | Control verdict / cost / wall | Treatment verdict / cost / wall | cost × | latency × |
| --- | --- | --- | ---: | ---: |
| sqlite `4JQh54h` (1) | verified / $1.211 / 2955 s | unverified / $0.112 / 710 s | 10.80 | 4.16 |
| sqlite `hUXYSvy` (0) | refuted / $0.425 / 898 s | unverified / $0.114 / 394 s | 3.74 | 2.28 |
| chess `xGt6Pve` (1) | unverified / $0.653 / 1224 s | unverified / $0.035 / 134 s | 18.64 | 9.16 |
| chess `Da8xED4` (0) | refuted¹ / $0.848 / 1474 s | unverified / $0.114 / 844 s | 7.41 | 1.75 |
| gcode `ELuvDeE` (1) | refuted² / $0.525 / 862 s | unverified / $0.036 / 112 s | 14.72 | 7.68 |
| gcode `HFh2qJM` (0) | refuted / $0.276 / 707 s | refuted / $0.051 / 144 s | 5.37 | 4.90 |
| video `KFN5iFo` (1) | compile failed³ / $0.194 / 593 s | partial / $0.071 / 314 s | 2.75 | 1.89 |
| video `tJdCJ8R` (0) | unverified / $0.421 / 1321 s | refuted / $0.057 / 213 s | 7.41 | 6.21 |
| **All eight** | **$4.553 / 10033 s** | **$0.590 / 2865 s** | **7.72** | **3.50** |

Across the two cold runs of the same cohort the treatment's cost ratio is
stable (8.57× and 7.72×) while its latency ratio moves with two executor
sessions: the chess negative's semantic obligation ran 220 s (`unobtainable`)
in one run and 706 s (`failed`, nineteen Stockfish commands) in the other,
and the sqlite positive's recovery obligation ran two ~300 s methods here
against one in the earlier run. Both are honest executor outcomes on
obligations the accepted method could not decide, not verifier waste: the
treatment's critical path is now extract ∥ compile → Oracle → pre-audit →
executor → Judger with no idle stage, and the executor and Judger seats hold
about 80% of the remaining wall on these candidates. The treatment never
refutes a positive candidate in either run; the control does so twice.

The sqlite positive is the one candidate the control verifies and the
treatment does not: the control compiled it into twenty per-row obligations
plus one recovery obligation whose method was accepted after two redos
(146 calls); the treatment's authority-only contract has three
obligations, and its Judger rejected both executed recovery methods as
covering one hard-coded leaf page rather than every recoverable row, so the
obligation ended `failed`. That is the Judger holding the same coverage
standard against a weaker executor artifact, and it costs the positive its
rank only when the negative also ends undecided.

## Mechanisms and where their savings come from

| Mechanism | Removes | Doc |
| --- | --- | --- |
| Frozen records | Judger "stale scan" redo loops caused by the verifier's own appends | `frozen-record.md` |
| Billing-frame tolerance | Paid Judger retries after a well-formed event with `type` not first | commit `f0d3406` |
| Atomic Oracle binding | Partial adapter bindings that silently became agent executions | `record-wire-contracts.md` |
| Record shape for the Oracle | Typed queries bound to a decoder profile the record does not contain | commit `86b7e9d` |
| Final-only certificate scope | Narration-claim audits that cannot change the verdict | commit `23d8bf1` |
| Typed Judger packet | Toolful Judger sessions for deterministic adapter receipts | `adapters.md` |
| Sufficiency pre-audit | Executor runs on methods the Judger then rejects as insufficient | `sufficiency-preaudit.md` |
| Concurrent final-contract compile | Extraction time on the critical path | `final-contract-precompile.md` |
| Prior-attempt briefing for the executor | A replacement method's executor re-making the artifact the Judger just rejected | `executor-briefing.md` |
| Process-local reuse (warm) | Repeated contract compilation and Oracle enumeration across candidates of one task | `contract-cache-proposal.md` |
