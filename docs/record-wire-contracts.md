# Record adapter wire-contract corrections

This is a general protocol clarification, not a task-specific rule. The five
stages, role schemas, model settings, evidence floors, repair limits and
independent Judger are unchanged. No incorrect pointer is auto-corrected and
no invalid digest is accepted.

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
