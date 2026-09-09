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

### 8-candidate cohort, treatment `82bef17` (running)

Four task families × one positive and one negative candidate
(`terminal-bench-general-inputs-20260908-v1`). Profile adds the final-only
certificate scope and the typed Judger packet. Results are appended here
when the run completes.

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
