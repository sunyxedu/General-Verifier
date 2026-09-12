# Explicit observed-EOF ledger instrument

`ledger.tool-call-string@3` is an executor instrument, not a new pipeline
stage. Obligation Compiler, Critic, Oracle Compiler and independent Judger
retain their existing responsibilities. The Oracle must explicitly choose the
instrument; no task name, benchmark identity or tool-name alias selects it.

## Why another version

Versions 1 and 2 correctly read only an explicit integer `through_seq` prefix.
In the full BindingGame diagnostic, audit records were appended between Oracle
selection and execution. The Judger rejected two negative certificates whose
prefix did not cover the obligation's requested full record scope. Pretending
that a prefix is the complete ledger would weaken verification.

Version 3 instead allows the Oracle to state an ordinary complete-file reading
procedure explicitly, without changing the old versions:

- `scope="prefix"` requires an integer `through_seq`.
- `scope="observed-eof"` forbids `through_seq`. The executor takes one bounded,
  stable, complete JSONL snapshot, then evaluates the predicate on those bytes.

The exact string predicates and explicit record filters retain version 2's
meaning. No retained suffix is silently excluded as an "audit-only" suffix.
Malformed records, unsafe files, unstable reads and exceeded limits are not
absence evidence: they use the existing fallback path.

## What observed EOF does not prove

The receipt identifies the observed bytes, resolved sequence/count and
observation time. It does **not** assert that the ledger is closed, final or
incapable of growing. A complete observation at time *t* is not proof of a
claim about every future time.

The independent Judger must still determine whether the observation's time,
scope, provenance and evidence strength establish the actual obligation.
An online claim requiring a closed epoch can therefore still be rejected.
There is no automatic scope repair, basis/strength upgrade, finality claim or
accepted-method reuse. The original redo, re-enumeration and failure paths
remain available.

Local mechanical tests cannot establish result preservation or performance.
Those require fresh Oracle/executor/Judger comparisons and, ultimately, full
online/offline evaluation against the pinned original-main baseline.

The reader opens each parent component using directory-relative no-follow
operations and rebinds the original path after reading. It rejects lexical
`..` components before normalization, unsafe links/files, path replacement and
in-read identity changes. The receipt records Unix and process-local monotonic
observation intervals; it does not claim clocks are comparable across processes.

Local verification: 48 dedicated tests cover all 13 legal selector/predicate
prefix combinations, EOF identity and limits, malformed suffixes, path races,
absence gates, and fresh Oracle/Judger calls with reuse disabled. The full
suite passed with `TMPDIR=/private/tmp`: 1,594 passed, 5 skipped and 4 subtests
passed, excluding the existing optional `test_score_llm_verifier.py` suite.
The canonical temporary directory avoids an unrelated report-build test's
`/var` versus `/private/var` path-alias assertion; no report code was changed.

## First paid BindingGame diagnostic: no demonstrated improvement

Frozen source `392f2abb71dfd4e7cc2176589840044c25fb017b` was tested in four
serial fresh processes on two sealed synthetic record obligations. Both arms
used GPT-5.6-Luna, medium/standard, with compact inspection, prefetch and
contract/method reuse disabled. The original Oracle, executor, independent
Judger and bounded repair remained in place. This excludes extraction,
Obligation Compiler and Critic: it is not full E2E or an original-main A/B.

| Case | Control process wall / calls | Typed process wall / calls | Final result, both arms |
|---|---:|---:|---|
| Tool identity irrelevant, positive | 175.25 s / 3 | 129.00 s / 3 | holds / record / decisive |
| Exact tool identity required, negative | 188.78 s / 3 | 313.38 s / 5 | refuted / record / decisive |

All four input seals, source fingerprints, artifact addresses, runtime profiles
and usage/span hashes passed postflight. Both signature pairs matched their
explicit synthetic expected outcomes. **Neither typed final binding used v3.**

- Positive: Oracle selected v3 but supplied `expected` with a non-equality
  predicate, which the unchanged argument validator rejects. The original
  agent executor handled the fallback. Its faster run cannot be attributed
  to the typed instrument.
- Negative: v3 executed correctly and produced a stable complete-file
  observation, explicitly without future-closure proof. Judger rejected its
  scope. The original repair path then used a fresh Oracle, agent executor
  and Judger; that agent result became the final binding.

The negative control agent called a one-time ledger read complete/immutable;
the repaired agent called two equal reads a sealed boundary. Neither artifact
provided independent writer-closure or immutable-epoch evidence. Judger accepted
those descriptions while rejecting v3's explicit limitation. This is a scope
interpretation and auditing inconsistency, not evidence that the agent produced
a stronger snapshot. It also means matching final signatures alone does not
establish equal verification quality. No result, rejection or gate was relabeled.

Across both cases, typed used 442.38 s versus control 364.03 s (**21.5% slower**),
8 versus 6 billed calls, and higher totals in every main token bucket:
uncached input 172,252 versus 133,618; cached input 591,872 versus 416,256;
output 16,895 versus 13,451. Rejected and fallback attempts are included.
This does not pass the performance or no-loss target. It does not authorize
inventing ledger closure, excluding audit suffixes, or relaxing independent
judgment to improve adoption.

Retained run: `experiments/terminal_bench/results/gv-observed-eof-gate-runs.tCfWIL`.
The coordinator's prospective signature gate and its separate
`observed_eof_adapter_bound=false` results remain unchanged.
