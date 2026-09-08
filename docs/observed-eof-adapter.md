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
There has not yet been a paid full-BindingGame test of version 3.
