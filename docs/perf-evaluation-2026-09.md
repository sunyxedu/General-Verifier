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

### 8-candidate cohort, treatment HEAD cold and warm (planned)

Adds the record shape for the Oracle, the pre-execution sufficiency audit,
the concurrent final-contract compile, eight obligation slots, and (warm
only) contract and accepted-method reuse across the two candidates of each
family.

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
| Process-local reuse (warm) | Repeated contract compilation and Oracle enumeration across candidates of one task | `contract-cache-proposal.md` |
