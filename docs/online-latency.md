# Online latency: what the verifier spends and how it saves

An online run pays for verification on the solver's critical path: every
`commit` blocks until the claim has been dealt with. Measured with Sonnet 5
as both solver and verifier on Terminal-Bench 2.1 tasks, the unoptimized
pipeline spent 90–95 % of a trial's wall time inside `commit`: compile
(~140 s) → critic (150–320 s, up to four rounds) serially, then per
obligation an oracle (35–80 s), an agent executor (20–160 s) and an agent
judger (20–90 s), with intermediate claims compiling to 6–13 obligations.

The mechanisms below are all task-independent. Each is a rule about how
evidence is obtained or scheduled, never about a task's content, and each
is on by default for a live world only; offline scoring keeps the full
sequential pipeline.

## Admission screening (`confident`)

Before any instrument is spent on a non-final claim, a no-tools **screener**
grades it from the retained record alone: the events the claim cites (in
full), an index of every recorded tool call, and the most recent results.
`high` is reserved for a claim that plainly restates a retained observation
— including a *transparent computation*, an operation whose complete text
and output are on the record — is current with respect to everything
recorded after it, and needs no inference or semantic judgment. A `high`
claim is admitted **confident**: a flag orthogonal to the five statuses
(the claim stays `unverified`, nothing was checked), disclosed on the
certificate, cleared by revision, and never taken at face value by an
explicit `verify` or by the certificate. A final is never screened.

Ledger events: `claim_screened`. Knob: `GV_SCREENER=0`.

## Methods in executable form

The oracle may supply a method as a `script`. The executor runs it verbatim
(bash, or the interpreter named by its `#!` line) with the verifier's own
virtualenv removed from the environment, retains exit status and complete
output, and a no-tools **script reader** signs sufficiency from the script
text and reads the outcome off the output. Fidelity holds by construction,
so there is no redo loop. Every gate of the agent path still applies:
nothing below decisive refutes, absence refutes only a presence
requirement, a delivered authority is circular, a run that did not finish
is a failed attempt. A refutation on this path is **countersigned** by the
independent judger with tools before it settles; a disputed refutation is a
failed attempt with the dispute on record. Methods without a script take
the agent path unchanged.

Knob: `GV_SCRIPT_TIMEOUT` (seconds, default 300).

## Final-contract precompile

The final's contract is compiled from the authoritative task statement with
solver premises withheld, so nothing about it waits on the solver's
wording. The server compiles it (compiler + critic) in a background thread
at startup, enumerates method batches for the compiler's draft while the
critic is still reviewing, and serves the reviewed contract and its batches
to the final's commit when the task statement is unchanged. A final that
arrives before the review ends settles the draft speculatively and
reconciles against the reviewed contract afterwards.

Ledger events: `contract_precompiled`, `contract_cache_hit`,
`methods_enumerated` with `cached: true`. Knob: `GV_FINAL_PRECOMPILE=0`.

## Speculative discharge and compile

Obligation nodes are content-addressed, so the compiler's obligations can
be settled while the critic reviews them: whatever the review keeps is
already settled, added obligations are settled afterwards, dropped ones
lose their edge to the claim, and the verdict folds over the reviewed
decomposition only. The compiler likewise starts alongside the screener;
on a `high` screen its call is dropped.

Ledger events: `speculative_compile`, `speculative_discharge`. Knob:
`GV_SPECULATIVE_DISCHARGE=0`.

## Critic budget for non-final claims

Whole-task acceptance is adjudicated at the final, which keeps the full
adversarial game. A non-final decomposition defect can mislead the solver
about an intermediate step but never the certificate, so online the
compiler's decomposition stands for non-finals by default; a positive
`GV_CRITIC_ROUNDS_NONFINAL` seats an advisory review of that many rounds
whose last repair stands.

## Smaller economies

- The oracle is told that enumeration is design, not execution, with an
  explicit tool budget behind a turn cap; a claim's checks share one
  evidence boundary, so an input one check found absent is reported to the
  oracle of its siblings.
- Eight obligations settle concurrently per claim (`GV_PARALLEL_OBLIGATIONS`).
- A commit naming an unknown reference returns the record's live label
  index, since the solver cannot see ledger sequence numbers.
- The solver protocol states what deserves a claim: load-bearing
  conclusions, one per claim, with the evidence left on the record.

## Measuring

Every Agent SDK seat call leaves a receipt in `role_calls.jsonl` next to
the state. `experiments/terminal_bench/latency_breakdown.py <job dir>`
reports, per committed claim, how long the solver waited and which seats
the wait went to; `--summary` gives one row per trial with the reward and
the agent's wall time.
