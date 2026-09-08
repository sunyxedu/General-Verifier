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
