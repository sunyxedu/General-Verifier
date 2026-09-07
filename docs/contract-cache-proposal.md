# Stable Final Contract Cache

Status: implemented experimentally; `GV_CONTRACT_CACHE=1` opts in. Default is
off until paired multi-candidate evaluation passes. No 10x or no-loss claim yet.

## Scope

Reduce repeated Compiler and Critic calls for same-task candidates. The pipeline remains:

```text
Obligation Compiler -> Critic -> obligations
obligation -> Method cache -> Oracle -> adapter/executor -> Judger
```

The Obligation Compiler still defines what must hold; the Oracle still chooses how to establish it.
The stages are not merged. No 10x end-to-end gain is promised: savings depend on workload composition.

## Problem

Although task authority controls final contracts, solver paraphrases and candidate-local paths reach
the Compiler and Critic. Calls can change text, clauses, absence rules, or decomposition and miss keys.

Fuzzy matching is forbidden: similar text can hide different quantifiers, conjunctions, evidence
floors, modalities, or absence semantics. Do not use embeddings, edit distance, or automatic merging.

## Prerequisite: authority-only compilation

Reuse is unsafe until cacheable final compilation is candidate-independent. The Compiler and Critic
may see only:

- the exact authoritative task statement;
- exact pre-task baseline authority needed to interpret the contract; and
- stable logical addressing and verification-boundary metadata.

They must not see solver final text, delivered-world bytes, ledger events, solver graph, candidate
artifacts, or candidate-local absolute paths. If `Read`, `Grep`, and `Glob` remain, enforce a
realpath-checked, read-only allowlist limited to task and baseline authority; symlink escapes fail
closed. Inlining bounded authority material with no filesystem tools is also valid.

`authority_view.py` reads bounded, stable, no-link task/baseline files and
inlines a canonical logical view. Solver text, memory hits, world paths and
ledger content are withheld from both seats and every repair round. Baseline
absence is explicit; its presence does not expose delivered files.

The built-in SDK uses an empty hard tool palette and a fresh scratch cwd.
Codex uses the same model settings, a fresh scratch cwd, root-deny/read-only
permissions, disabled tool/connector/skill surfaces, and an allowlisted
single-turn JSON event stream. Unexpected tool activity rejects the generation.
An exact capability-disabled startup notice is allowed; arbitrary errors are
not. Unknown backends and custom role subclasses bypass this path. Local and
UTC calendar identity are rechecked after waits and fills.

Permission profiles and non-shell tool controls are separate boundaries; see
[OpenAI permissions](https://learn.chatgpt.com/docs/permissions) and the
[configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
The installed CLI is checked as well: 0.153.4 rejects the documented legacy
`tools.view_image` under strict config, so the implementation uses its supported
`features.view_image=false`. Unknown configuration fails closed.

## Implemented cache

The bounded, process-local, single-flight `CompiledFinalContractCache` stores only the complete
obligation payload after the Critic passes it, never candidate verdicts or evidence. Do not cache
failures, rejected or malformed payloads, exceptions, or interrupted calls.

On a hit, deep-copy the exact payload and construct obligations for the current claim. Only contextual
`claim_id` changes; semantic fields and identities do not.

### Eligibility

Require a final row, empty premise scope, present and stably readable task authority, an active
authority-only view, and stable bounded key inputs. Non-final claims remain premise-specific. Without
authoritative task context, final wording is semantic, so cross-candidate reuse is disabled. Uncertain
identity or authority change is a miss.

### Minimum exact key

Include:

- exact task text and normalized pre-task baseline identity;
- Compiler and Critic prompts, schemas, tools, names, and turn limits;
- backend type, model, effort, service tier, and permission mode;
- Critic rounds, pipeline/cache schema version, and stable memory input; and
- world mode, scheme, captured roots, and logical artifact addresses if still semantic inputs.

Exclude claim id, solver final text, host roots, absolute paths, delivered bytes, ledger contents, and
candidate metadata. Exclude a free-form input only after removing it from the semantic prompt.

### Validation and lifecycle

- Validate the complete schema and publish only a Critic-passed payload.
- Re-read and re-key authority after the cold fill before publication.
- Wake single-flight waiters on success, failure, and exception.
- Bound entries with LRU eviction; return detached deep copies.
- Keep entries in trusted process memory; disk reuse needs separate authentication.
- Log hits, misses, bypass reasons, source digest, and single-flight waits.
- A losing concurrent fill returns the already-stored approved contract and
  logs a hit, not a publication of a payload it never stored.

## Accepted-method direction safety

Oracle and Judger allow verdict-direction certificates: one realized direction may decide the full
obligation while the opposite should fail or be unobtainable. `decisive` therefore does not imply
bidirectional sufficiency.

Reuse an accepted method only in a direction independently accepted by the Judger. The minimum
fail-closed rule stores `source_outcome` and bypasses on a different fresh outcome. A future structured
`valid_outcomes` signature may explicitly authorize both directions; missing or legacy values default
to the source outcome. Adapter purity alone never proves bidirectional sufficiency.

## Evaluation

Use deterministic fixtures first: key separation, authority mutation, single-flight concurrency, copy
isolation, failure recovery, and outcome-direction mismatch. Assert that each candidate still executes
its adapter and that non-reusable methods still receive fresh Oracle and Judger audits.

Then run paired control/treatment evaluation over multiple candidates for each of multiple identical
tasks. One candidate cannot exercise reuse, and one task cannot establish generality. Hold trajectories,
authority, model settings, and scoring inputs fixed; isolate result roots. Compare per-candidate
verdicts, obligation outcomes, rank direction, role calls, tokens, and wall time. Any result regression
blocks rollout regardless of aggregate speedup.
