# Common cross-version accounting, without replacing verifier stages

The original-main backend records successful CLI usage but can omit failed
attempts. Comparing those records directly against the current all-attempt
ledger would give the two versions different accounting boundaries.

`experiments/terminal_bench/observe_codex_attempts.py` is a standalone
evaluation observer. It does not replace either backend implementation,
prompts, schemas, model, tools, permissions, retry budgets or response parsing.
Obligation Compiler, Critic, Oracle Compiler, executor and independent Judger
keep their existing responsibilities. There is no merged compiler or new
verification stage.

## Interfaces and limits

`observe_codex_attempts(module, ledger_path, seam=..., context_by_cwd=...)`
temporarily wraps the existing `CodexBackend.complete` and one process seam:

- `legacy-subprocess-run`: a proxy for the original module's own `subprocess`
  binding, without patching the shared standard-library module.
- `run-codex-process`: the current module's existing process helper.

Both delegate to the original functions. Each logical call and CLI attempt
gets start/terminal records with thread-local attribution. A process returning
success and the backend subsequently rejecting its output are separate events;
the latter does not erase the process's charges. Timeouts, retries and missing
usage remain visible. Only metadata, command/stdout hashes and native token
counts are retained, not prompts, command bodies, stdout or exception messages.

The operator supplies a fixed, small context mapping. The ledger must be fresh
and outside every candidate state. Existing usage capture must still be enabled
to make the original backend request native CLI JSON; the observer does not
change command arguments. Keep the old backend's legacy ledger separate and
do not add its duplicate successful bills to the common observer's totals.

`analyze_codex_observer.analyze_file` reads the ledger without modifying it.
It requires the pinned expected contexts/profile and the independent
`observation.summary()` captured **after** the observer context exits. This
summary counts events even when writing them fails: otherwise losing all four
events of one call could leave a deceptively complete-looking ledger.

The accounting gate rejects missing or duplicate lifecycle records, incomplete
usage, wrong runtime model/effort/tier, unbound execution directories, unhealthy
sessions, reused thread IDs and inconsistent counters. Expected execution cwd
defaults to the call's state; any authority-only scratch directory must be
explicitly operator-bound, never inferred from the ledger itself.

Cache-write counts can be absent in native telemetry. Their unknown count is
retained alongside the known input/output buckets, but the full accounting
gate remains incomplete. This deliberately does not turn an optional,
potentially priced bucket into zero. Reasoning tokens are already part of
output and must not be charged again. This module applies no monetary rates.

## What this does not establish

Closed observer lifecycles establish only the observed accounting boundary,
not that every required verifier stage or candidate ran. The outer evaluation
must independently pin source and overlay hashes, seal inputs, record all
process starts/exits and full end-to-end wall time, protect the ledger/summary,
and compare exact contracts, verdicts and evidence strength.

The observer also does not equalize the original and current isolation policies
or process-launcher cleanup. A common isolation overlay and the two-source
runner are separate, still-required work before an acceptance-eligible
original-main comparison. Any such overlay must be explicitly identified;
an instrumented baseline must not be described as an unmodified original run.
These local accounting tests are not evidence of result preservation or 10×.

Local verification: 104 observer/analyzer tests passed, including the real
current backend with a fake process and native usage capture. The full local
regression passed with `TMPDIR=/private/tmp`: 1,719 passed, 5 skipped and
4 subtests passed, excluding the existing optional
`tests/test_score_llm_verifier.py` suite. No model calls were made by these tests.
