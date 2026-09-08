# Experimental append-only Compiler repair

Default: **off**. `GV_APPEND_ONLY_DELTAS=1` enables the versioned
`obligations-array-delta-v1` protocol in the shared online/offline Kernel.
Only exact `APPEND OBLIGATION:` Critic instructions select it. `ADD:`,
removal, weakening and ordinary full rewrites retain their existing path.

## What changes

The same Obligation Compiler returns only new tail entries using its existing
`{"obligations": [...]}` envelope. The Kernel deep-copies the old entries and
appends the returned entries. The same independent Critic then reviews the
**complete** resulting contract and full game history. Model, tools, schema
retry limits, repair limits and Critic limits are unchanged. Oracle, executor
and Judger are untouched. This is a repair representation, not a new stage,
merged compiler, skipped audit or task-specific interpretation rule.

An empty delta or exact echo of the entire old prefix consumes a repair
attempt and is rejected mechanically. No fuzzy matching guesses whether a
new entry was intended to modify an old one: the old entry cannot be replaced,
and the Critic must judge the combined contract. A standing attack still
fails closed when the existing budget is exhausted.

## Identity and measurement

Kernel and BindingGame freeze the option at construction; inconsistent
snapshots are refused. Both final-contract and accepted-method cache keys
include the option/protocol, as do offline configuration and replay identity.
The adapter-only cold diagnostic explicitly keeps this option off in all
three processes, so it does not mix two treatment changes.

The motivation is a real cold preparation in frozen revision `14a4fa1`:
one full-list regeneration changed an existing source clause during an
append-only repair and was rejected before Critic review. Avoiding that kind
of regeneration can save output and retries. It does not establish any fixed
speedup, zero result loss, or the requested 10x end-to-end improvement.
Paid append-delta A/B validation is still pending; do not enable by default
based on local fixture tests.

Validation: 24 focused local tests, including a four-attempt Codex protocol
stub through preparation and independent receipt consumption; full regression
**1134 passed, 4 skipped, 4 subtests passed**, with the pre-existing optional
`test_score_llm_verifier.py` dependency excluded. These are correctness and
identity tests, not real-model quality or performance measurements.
