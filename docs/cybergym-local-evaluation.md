# Local CyberGym evaluation

The follow-up to PR #2 runs on the local workstation. macmini supplies only
historical metadata and the already audited runtime image bytes. Historical
cohorts do not contribute scores to this evaluation.

## Progress

| Stage | Required result | Current result |
| --- | --- | --- |
| First five, fixed order | 5/5 | Not started; restoring pinned runtime |
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
  `experiments/cybergym/run.py`; image transfer is currently in progress.
- Official runtime checkout is pinned to
  `c6fe2027d39471375920b92cf1025e23a99ffda5`.
- Local regression: 4,544 passed, 30 skipped, 4 subtests passed. The unchanged
  optional `llm_verifier` module was excluded. Of the skips, 25 require absent
  repository-local operator helpers, three require Chromium, one requires
  explicit paid-smoke opt-in, and one lacks the platform's xattr API.
- Runner/config/search-guard regression: 387 passed.
- Prompt changes: none. The first fix makes official-runtime and private auth
  paths configurable so the original runner can start on this workstation.

Private raw outputs live under `experiments/cybergym/results/local-20260913/`,
which is ignored by Git. Each actual cohort has `manifest.json`,
`controller.json`, `attempts/*/status.json`, `results.json`, and monitoring
snapshots. Monitoring tracks phase intervals, role calls, transport/evaluator
failures, token usage, and stalled-progress warnings; a warning alone is not
a task deadline or a correctness verdict. Only official completed scores
count toward the gates above.
