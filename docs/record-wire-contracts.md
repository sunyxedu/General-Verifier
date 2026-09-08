# Record adapter wire-contract corrections

This is a general protocol clarification, not a task-specific rule. The five
stages, role schemas, model settings, evidence floors, repair limits and
independent Judger are unchanged. No incorrect pointer is auto-corrected and
no invalid digest is accepted.

## Architecture boundary

The original responsibilities remain separate: Obligation Compiler defines
what must hold, Critic reviews that contract, Oracle Compiler chooses a method,
the executor produces evidence, and Judger independently audits it. Typed
adapters are executor implementations, not alternative compilers or judges.
Neither a successful adapter run nor a matching artifact digest can settle a
binding without the required audit. Evidence floors, provenance, absence rules,
and production repair budgets are not optimization knobs.

Current record probes disable accepted-method reuse and exclude upstream
compilation explicitly; they do not pretend those stages cost zero. Existing
cross-candidate cache policies are not broadened or enabled by this correction.
Any proposal to skip a fresh role call through reuse needs a separate, explicit
architectural decision and evaluation, not an implicit claim of equivalence.

Local guards exercise both online and offline BindingGames, positive and
negative record results, repeated Oracle/Judger calls, and Judger rejection on
either sufficiency or fidelity. Rejected receipts cannot become accepted
bindings or cache entries. These are orchestration tests with fixture verdicts,
not evidence of model-level no-loss or complete online/offline task performance.

## Failed optimization diagnostic

The adaptive gcode followup at frozen `981a97339236f1ba0fe23c639b42a4df9c1a47d5`
enabled the new general preference for equally sufficient typed methods.
It reused the preceding typed-enabled diagnostic as a baseline; this was not
a randomized independent pair and never a full-task acceptance test.

| Same precompiled record obligation | Previous typed-enabled | Preference followup |
| --- | ---: | ---: |
| Process wall seconds | 186.242 | 724.184 |
| Attempted model calls | 3 | 9 |
| Uncached input tokens | 127,068 | 249,258 |
| Cached input tokens | 364,032 | 1,292,288 |
| Output tokens | 7,217 | 29,441 |
| Successful adapter executions | 0 | 0 |

All attempts and costs were retained. Both final binding results were
holds/record/decisive with fresh yes/yes/n/a Judger audits and unchanged input
pins. The followup was a substantial performance regression, not progress
toward 10×. It exposed two separate mechanical contract misunderstandings.

1. The Oracle selected `ledger.tool-call-string@2` but passed
   `value_pointer=/payload/tool`. The adapter roots pointers at
   `event.payload`; the procedure instead referred to its direct `/tool`
   field. The missing nested field correctly triggered fallback. The
   catalog now spells out the pointer root and the meaning of a real nested
   `payload` key. Argument semantics and fallback behavior are unchanged.
2. The first Judger rejected a correctly addressed artifact. Local
   recomputation with `core.model.content_hash` gave its actual identifier,
   `ae31d5f42f0ea3a2a11c9b7417731f4da15af986624e5a7d0131b0aae0b6cee9`.
   The Judger's reported alternative,
   `996d988f51abe54ef808409ae88ddbc3076f64d3d2db1acaf5ddad084236970b`,
   exactly matches hashing the compact serialization **plus an extra newline**.
   Native diagnostics do not retain commands, so this identifies the byte
   variant, not the exact command used. Three subsequent executor/Judger redo
   rounds consumed further work before the result was accepted.

The Codex backend now supplies the exact existing content-addressing
calculation to Judger as a transport note. This does not pre-approve or inline
an artifact: Judger still opens actual evidence and decides sufficiency and
faithfulness. The note explicitly says a real mismatch remains a defect and
matching bytes do not prove semantic correctness. Its run-scoped identity is
included in accepted-method cache keys; cache eligibility/policy is unchanged.
The note is backend protocol text, not a domain example added to the shared
semantic axioms or role schemas.

This follows official guidance to preserve output/evidence constraints and
remove ambiguity. No published performance range is imported as a project
result. [GPT-5.6-family prompting guidance](https://developers.openai.com/api/docs/guides/prompt-guidance-gpt-5p6).

## Regression coverage and acceptance boundary

Local tests compare the documented calculation with actual ArtifactStore
addresses for Unicode, nested values, whitespace, booleans and numbers. They
reject the extra-byte variant and check that only Judger receives the note,
model/role schemas remain unchanged, and changed notes cannot share method
cache identity. Pointer tests cover both adapter versions, direct and nested
keys, and refusal to silently reinterpret a wrapper-looking invalid pointer.

These tests establish the mechanical contract, not model no-loss or a latency
gain. New paid probes must still show correct adapter exposure, independent
Judger acceptance, complete attempted-call bills, and no outcome regression;
negative controls and full online/offline evaluation remain required.

Raw followup: `experiments/terminal_bench/results/gv-record-preference-runs.Tq9602/`.
Its complete final `binding-game` artifact retains the full Oracle MethodIR,
all attempts and audits. Old archives, results and operator inputs were not
rewritten to hide the regression.

## Corrected positive followup

Frozen `da154d6b11b3c48447342bd7ddc4d282768b4d4d` completed the same
precompiled gcode record obligation with one actual `ledger.tool-call-string@2`
execution, zero fallbacks, one Oracle call and one fresh Judger call.
The result remained holds/record/decisive with yes/yes/n/a audit and unchanged
source and case pins. The emitted receipt was 3,175 bytes; the final binding
artifact was 5,187 bytes. Both are actual files, not token estimates.

| Single-obligation diagnostic | Earlier untyped control | Corrected typed followup |
| --- | ---: | ---: |
| Process wall seconds | 265.916 | 74.505 |
| Attempted model calls | 3 | 2 |
| Uncached input tokens | 90,560 | 49,480 |
| Cached input tokens | 585,984 | 104,192 |
| Output tokens | 11,537 | 3,039 |

This is a historical-control adaptive followup, not a fresh randomized pair.
Its observed wall ratio is 3.57×; the three billable token categories are
reported separately rather than converted into an unsupported monetary cost.
It does not establish 10× overall savings or full-task result preservation.
The Oracle selected a smaller positive-witness prefix; the fresh Judger
accepted that prefix only because one authentic event suffices for this
existential claim. This does not authorize narrower scope for a negative or
universal claim.

Raw corrected runs: `experiments/terminal_bench/results/gv-record-wire-runs.xxClEU/`.
The matrix also prospectively specifies a synthetic negative pair: only the
copied video feed's tool identity is renamed, preserving other payloads and
the original archive. That counterfactual checks tool-identity handling; it
is not an authentic Terminal Bench trajectory or a whole-task score.

## Negative pair: equivalence gate failed

Both prospective arms completed with unchanged source/case pins and complete
attempted-call accounting. Nevertheless, the final results **disagreed**:

| Synthetic renamed-identity input | Untyped control | Corrected typed |
| --- | ---: | ---: |
| Outcome | refuted | holds |
| Process wall seconds | 488.612 | 98.930 |
| Attempted model calls | 6 | 2 |
| Uncached input tokens | 234,148 | 72,249 |
| Cached input tokens | 983,552 | 122,112 |
| Output tokens | 18,382 | 4,057 |
| Successful adapter executions | 0 | 1 |

The apparent speed reduction is **not an accepted optimization result**.
The control's first method treated any nonblank command-bearing record as
sufficient; its Judger rejected the missing tool-identity relation. The normal
repair then checked explicit identity and returned refuted with an admissible
absence witness. All six calls, including the rejected method, are counted.

The corrected typed Oracle selected the renamed identity and a nonblank-string
predicate. The adapter faithfully returned holds for that narrower predicate.
Its fresh Judger accepted the unstated equivalence between that observation
and the required action. Thus mechanical receipt validity and a separate model
audit did not establish model-level result preservation. No task-name routing,
identity alias table, or acceptance override is introduced to fix this case.

There is an important gold-label limit: the operator expected refuted under
exact retained tool identity, but the natural task can also be read as referring
to the underlying execution behavior. That behavior is not established by an
opaque tool name plus command text. The pair establishes a reproducible input
with **observed A/B disagreement**, not a proven false positive under every
possible task reading. Benchmark reward provides no gold for this obligation.
Future controls must distinguish an explicitly exact identity predicate from
an identity-independent content predicate before attributing the disagreement
to a specific optimization change. Neither this single bundle comparison nor
fixture-only tests isolate causality or justify rollout.

After the architecture guards, the local regression command completed with
1,311 passed, 4 skipped, and 4 subtests passed. It retained the existing exclusion
of `tests/test_score_llm_verifier.py` for its unavailable optional dependency.

## Explicit synthetic semantics: parity still fails

The subsequent frozen `1b9b7a7` matrix separates exact literal identity from an
explicitly identity-independent nonblank-string predicate. These are synthetic
authorities and inert records, not Terminal-Bench task rewards. Compact
inspection was fixed off, with no method cache, throughout all eight jobs.

| Predicate / input label | Expected | Control | Typed | Successful typed adapter executions |
| --- | --- | --- | --- | ---: |
| Exact identity / required label | holds | holds | holds | 1 |
| Exact identity / opaque label | refuted | refuted | refuted | 0 |
| Any identity / required label | holds | holds | holds | 1 |
| Any identity / opaque label | holds | holds | failed | 0 |

Every non-failed result was record/decisive. `failed` is an unresolved binding,
not a refutation and not a correct negative. In that arm, two invalid logical
subject declarations caused agent fallback. Judger then rejected both methods
for omitting newly appended verifier metadata, despite observing an authentic
positive existential witness. The ordinary budget exhausted after six calls;
all costs remain counted. Other typed arms also incurred format fallbacks.
In particular, the negative arm had **no successful adapter exposure**, so its
correct final refutation does not validate the adapter's negative path.

Raw runs are `experiments/terminal_bench/results/gv-semantic-profile-runs.xTR0C0/`;
inputs are `gv-record-semantic-gate.th8ZCe/input/` under the same results root.
The semantic jobs overlapped, so their wall times are not a performance A/B.
This gate fails result equivalence. Mechanical argument validation and semantic
sufficiency need separate investigation; neither automatic acceptance nor
task-specific identity rules are justified. The main stage interfaces, ordering,
and independent Judger requirements remain unchanged by this diagnostic.

An independent read-only audit recomputed every input seal, case/feed hash,
frozen-source fingerprint and all 16 usage/span file hashes without mismatch.
Across the four semantic cases, typed used 19 calls versus control's 12;
uncached/cached/output tokens were respectively 415,277/1,403,648/50,946 versus
267,568/845,824/26,985. This is a cost regression in every token category.

After the default-off full offline profile plumbing, local regression completed
with 1,384 passed, 4 skipped and 4 subtests passed, retaining the same optional
`test_score_llm_verifier.py` exclusion. Unit coverage is not a replacement for
the failed paid semantic gate.
