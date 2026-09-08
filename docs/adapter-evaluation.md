# Typed adapter evaluation

## Current conclusion

The implementation is experimental. Neither 10x end-to-end improvement nor
zero result loss has been established. Do not treat adapter unit-test speedups
as whole-pipeline performance, or matching final verdicts as identical internal
verification results.

## Paired offline pilot

Source revision: `eccefb9`; pipeline fingerprint: `35e435a6ae5535bf`.
Both arms use the same source, existing Terminal-Bench offline scorer, sealed
trajectories and captured worlds, GPT-5.6-Luna, medium effort, standard service,
one candidate worker, four parallel obligations and four parallel claims.
Only `GV_TYPED_ADAPTERS=0/1` differs. Browser observer configuration is unset.
The repetition is `general-ab-r1`, with separate control/treatment scoring roots.

These are ordinary negative-result pilot traces, one candidate per task. They
do not exercise cross-candidate reuse or positive-result preservation. No
benchmark gold is supplied to verification seats. Compiler sampling is not
frozen, so the measurements include upstream sampling variance, not just the
causal effect of adapter execution.

| Trace | Final verdict, off / on | Recorded calls, off / on | Wall seconds, off / on |
| --- | --- | --- | --- |
| `gcode-to-text__HFh2qJM` | refuted / refuted | 27 / 17 | 1049.24 / 1146.76 |
| `write-compressor__pVzF4HY` | refuted / refuted | 63 / 35 + unrecorded timeout | 1785.91 / 2257.36 |

For the completed gcode pair:

| Measure | Adapters off | Adapters on |
| --- | ---: | ---: |
| Uncached input tokens | 827,626 | 479,707 |
| Cached input tokens | 5,932,800 | 5,214,336 |
| Output tokens | 93,227 | 62,940 |
| Cumulative model-call seconds | 2284.54 | 1602.92 |

Calls fell 37.0%, uncached input 42.0%, and output 32.5%, but wall time rose
9.3%. Cumulative call duration is not wall latency: concurrent calls overlap,
and one slow call can dominate the critical path. Tokens are cost proxies;
these runs do not provide a monetary bill.

The compressor control recorded 1,771,431 uncached input tokens, 11,143,680
cached input tokens and 212,082 output tokens. Treatment recorded 757,316,
4,150,272 and 89,406 respectively, but excludes the timed-out call described
below. Its wall time rose 26.4%; no complete token-based saving is claimed.

Both gcode runs compiled four obligations, but not the same four obligations.
The control combined output presence and semantic correctness; treatment split
them. The record requirements also differed in scope (each response versus at
least one command). Final verdict and ranking score matched, but obligation
histograms and semantic granularity did not. This is not evidence of full
result equivalence.

The compressor control had eight total obligations, five attached to the final
claim; treatment had seven, all attached to the final claim. Their internal
outcomes also differed. Both tasks retained the same final ranking score
(`0.3332`), but one candidate per task cannot validate Best-of-N ordering.

| Role calls | Gcode off | Gcode on | Compressor off | Compressor on |
| --- | ---: | ---: | ---: | ---: |
| Extract | 1 | 1 | 1 | 1 |
| Obligation Compiler | 3 | 1 | 5 | 2 |
| Critic | 3 | 1 | 5 | 2 |
| Oracle | 6 | 6 | 16 | 12 |
| Discharge | 8 | 6 | 21 | 13 + unrecorded timeout |
| Judger | 6 | 2 | 15 | 5 |

Only two typed executions succeeded, both `filesystem.exists`; five fast-path
attempts fell back, and there were zero method-cache hits. The observed call
reduction includes changed Compiler/Critic and Judger paths and cannot be
attributed entirely to two adapter executions.

Raw artifacts are under `experiments/terminal_bench/results/` with stems
`ab_eccefb9_{control,treatment}_{g2,g6}`. Each stem has a score `.jsonl`, usage
`.jsonl.usage.jsonl`, `.log`, and `.tsv`. The scorer records successful CLI
usage; failed, timed-out, or retried calls may be absent from the usage ledger.
Any such failures must be reported separately, not silently assigned zero cost.

The compressor treatment had a 1800-second CLI timeout followed by an automatic
retry. The timed-out wrapper left a live Codex child and code-mode child; both
were identified inside this run's state and manually terminated, without
interrupting the retry. The failed attempt is absent from the usage ledger,
so that arm's recorded calls/tokens are a lower bound, not a complete cost.
The wall measurement still includes the wait. This infrastructure incident
also prevents interpreting its latency difference as a clean adapter effect.
All recorded usage rows have return code zero, and event JSON was valid with
no explicit role/schema failure event. The hidden timeout nevertheless happened;
absence of failure events is not proof of complete telemetry.

## Findings from broader historical comparison

Five earlier trajectories had no exact cross-candidate method-cache hits.
Same-task Compiler/Critic calls changed wording, controlling clauses and even
obligation decomposition. Loosening semantic matching would be unsafe.

In the historical baseline, discharge accounted for 31.9% of cumulative
model-call time. Even eliminating that entire role leaves a 1.47x upper bound
on improvement in that particular sum, not 10x. This is not a bound on wall
latency, which needs a dependency-aware critical-path measurement.

The earlier implementation comparison also included verdict regressions and
substantially higher usage, across different source fingerprints. It is useful
as a regression warning, not a clean estimate of the adapter treatment effect.

## Correctness fixes identified during the pilot

- File presence observations must expose the same kind and world-mode fields
  for present and absent targets. Missing kind evidence caused a real Judger
  redo, followed by additional agent evidence-delivery repair.
- Accepted-method reuse must retain the independently audited outcome
  direction. The existing protocol permits one-direction certificates;
  `decisive` does not imply sufficiency in both directions. A changed direction
  must go back through the full game, not inherit the previous audit.
- CLI timeouts must clean up the invocation's subprocess group before retrying;
  killing only the wrapper leaves overlapping work behind. Validate cleanup
  using harmless wrapper/child fixtures, not paid model calls.

All four pilot processes completed before source changes resumed. Subsequent
fixes and their regression tests are not retroactively included in these
measurements; the corrected revision needs its own future performance run.

## Post-fix verification

Corrected package fingerprint: `91143828ee29f394`.

Local regression command:

```sh
TMPDIR=/private/tmp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q --ignore=tests/test_score_llm_verifier.py
```

Result: **597 passed, 3 skipped, 4 subtests passed**. The skipped cases are
opt-in real-browser tests; running them separately with `GV_TEST_CHROMIUM`
pointing to the installed Chrome executable produced **3 passed**. Tests use
synthetic benign documents and records, not task-specific adapter logic.

Coverage includes live/captured wiring, actual browser observations, bounded
ledger scopes, absence gating, same-direction reuse, opposite-direction fresh
Judger rejection, concurrent candidate execution, complete file-kind evidence,
and POSIX same-group process cleanup on normal exit, timeout and interruption.
Process-group cleanup is not an OS sandbox and does not contain descendants
that deliberately create another process group/session.

The excluded legacy scoring test module cannot collect because its separate
`llm_verifier` dependency is unavailable; it was not changed or silently skipped
in source. Full-suite success is not claimed. Failed-attempt usage telemetry
also remains incomplete and must be addressed before making complete cost claims.

## Subsequent work (2026-09-08)

The historical figures above are unchanged. The branch now additionally has:

- terminal usage rows for successful, failed, timed-out and retried CLI attempts;
  unknown usage remains unknown, and does not count as zero cost;
- an experimental, default-off [authority-only contract cache](contract-cache-proposal.md),
  retaining separate Compiler and Critic stages;
- separate, independently audited positive and negative method-cache slots,
  with fresh adapter execution on every candidate; and
- a [paired evaluation driver](../experiments/terminal_bench/TYPED_ADAPTER_EVAL.md)
  that charges cold fills and compares complete wall time, with exact contract
  text retained to expose changed requirements.

These changes still need new real pipeline measurements. The successful
single-role Luna protocol smoke is not a performance benchmark. A six-candidate
warm canary is a regression screen, not an acceptance result; the larger gate
contains 20 distinct candidates across four ordinary tasks.

### Follow-up to the frozen canary

The live canary uses frozen source `fad9290`; later branch changes do not alter
its control or treatment. The [first-control diagnostic](canary-fad9290-diagnostics.md)
records real method-identity and provenance repairs, not missing artifacts.

Subsequent implementation addresses three general seams:

- The executor receives the method's declared strength and bridging assumptions,
  which the Judger already audited. Neither becomes additional task authority.
- Offline toolchain/cache roots are bound by operator construction, not role
  variables. The executor may write its shared tools; the Judger receives
  read-only toolchain access, without the external shared package cache. This
  uses Codex's [specific-path permission overrides](https://learn.chatgpt.com/docs/permissions),
  including a read-only override inside the otherwise writable cold workdir.
- A fresh pure-adapter result in a previously unaudited direction can go
  directly to a new independent Judger. A complete audit may seed that
  direction; otherwise the full game runs and all attempts count toward cost.

These are not measured 10x improvements. In particular, a substantive fidelity
failure still needs repair; normalizing metadata cannot make an incorrect
instrument faithful. Skipping other obligations after a cached refutation was
not added: it could change exported outcomes and advisory tie-breaking.

### Optimization direction

OpenAI's [latency guide](https://developers.openai.com/api/docs/guides/latency-optimization)
emphasizes fewer generated tokens, fewer requests, independent parallel work,
and using conventional computation where it suffices. It cautions that cutting
input length alone usually gives a much smaller latency gain. Applied here,
this supports measuring eliminated repeated role calls and the critical path,
not presenting prompt shortening or accumulated parallel model seconds as a
10x end-to-end improvement. The guide's suggestion to combine dependent model
steps is deliberately not applied across this project's independent auditors.
