# Local CyberGym evaluation

The follow-up to PR #2 runs on the local workstation. macmini supplies only
historical metadata and the already audited runtime image bytes. Historical
cohorts do not contribute scores to this evaluation.

## Progress

| Stage | Required result | Current result |
| --- | --- | --- |
| First five, fixed order | 5/5 | Two infrastructure cohorts retired; fresh local rerun pending |
| Fixed 30 | At least 26/30 (86.67%) | Not started |
| Full pinned catalog | Complete all 1,507 tasks | Not started |

The first five are the first five entries of `FIXED30` in the runner. The
same fixed task list and official vulnerable/fixed exact-byte scoring rule
apply throughout. Investigate implementation and infrastructure bugs before
changing prompts; any prompt change must apply generally. A new implementation
uses a fresh cohort with its own frozen source commit. Tuning against the
first-five or fixed-30 set will be disclosed when reporting their results.

## Setup verification, 2026-09-13

- Local disk: approximately 544 GiB available at initial inspection.
- Imported runtime identities must match the three SHA-256 image IDs in
  `experiments/cybergym/run.py`; transfer and exact-ID verification completed.
- Official runtime checkout is pinned to
  `c6fe2027d39471375920b92cf1025e23a99ffda5`.
- Fix regression with the native helper test enabled: 4,545 passed, 30 skipped,
  4 subtests passed (123.13 s). The unchanged
  optional `llm_verifier` module was excluded. Of the skips, 25 require absent
  repository-local operator helpers, three require Chromium, one requires
  explicit paid-smoke opt-in, and one lacks the platform's xattr API.
- Runner/config/search-guard regression: 387 passed.
- Prompt changes: none. The first fix makes official-runtime and private auth
  paths configurable so the original runner can start on this workstation.

## First cohort infrastructure audit

`round-01` started at commit `38f28b9`. All three initial solvers ran locally;
the first registered and evaluated a candidate and entered GV finalization.
No final official score completed. A real Discharger command failed before
execution with ENOENT; its retained failure identified a missing temporary
`codex-linux-sandbox` helper. The cohort was stopped and retained for audit,
and none of its results count toward the accuracy gates.

A zero-model local Responses fixture reproduced unified-execution ENOENT
when concurrent pinned CLI startups shared a macOS bind-backed Codex home.
The same fixture succeeded with tmpfs. Keeping the home bind and mounting only
`/codex-home/tmp` on tmpfs also succeeded. The production fix uses that narrow
mount, preserving private auth and the existing shell permissions. Its native
regression captures the actual runner mount, executes the tool after concurrent
janitor startups, and verifies that a synthetic private auth marker cannot be
read. CLI exit code alone is insufficient: the failing control's CLI exited
zero while its tool failed. The backend process-cleanup test fixture also now
publishes its PID marker atomically, avoiding an empty-marker race during
intentional interruption; its cleanup assertions remain unchanged.

## Docker Desktop upgrade interruption

`round-02` at commit `95dc732` progressed through obligations into Critic.
It produced no official final score. At 13:12:55 UTC, the Docker Desktop
installer requested `/app/quit` while replacing version 4.70.0 with 4.90.0.
The disappearing Docker socket interrupted all three active workers and the
two subsequent image pulls. This is an environment interruption, not five
candidate failures. All three stopped sidecar states were copied successfully
for postmortem audit; the cohort does not count toward accuracy.

The native concurrency/private-auth test passed again on the updated engine
(1 passed, 9.58 s), as did real CLI configuration and offline toolchain probes,
without model calls. The next cohort uses a fresh directory and records its
Docker version and engine session separately from its frozen source commit.

Private raw outputs live under `experiments/cybergym/results/local-20260913/`,
which is ignored by Git. Each actual cohort has `manifest.json`,
`controller.json`, `attempts/*/status.json`, `results.json`, and monitoring
snapshots. Monitoring tracks phase intervals, role calls, transport/evaluator
failures, token usage, and stalled-progress warnings; a warning alone is not
a task deadline or a correctness verdict. Only official completed scores
count toward the gates above.
