# Prior-attempt briefing for the executor

## Problem

When the Judger (or the pre-execution sufficiency audit) rejects a method,
the Oracle sees the rejection and proposes a replacement. The seat that then
implements the replacement did not: it received only the new procedure. On
the sqlite recovery obligation the second executor re-made the artifact the
first had just been rejected for (a checker hard-coding one 4096-byte page),
300 s each time, and the fresh Judger rejected it for the same gap.

## Mechanism

Before a fresh method's execution, `BindingGame` hands the executor the
obligation's earlier attempts as untrusted diagnostics (`verifier/engine/game.py`,
`prior_attempt_briefing`): for each of the last three attempts, a bounded
excerpt of its procedure, whether it was rejected before or after execution,
and the recorded rejection reason. The text travels in the same slot as the
redo loop's Judger feedback, under the same rule in the executor prompt: it
cannot amend the bound procedure, evidence source, decision rule, promised
artifact or required outcome, and it is never task or world evidence. A redo
of a briefed attempt keeps the briefing ahead of its redo diagnostics. The
`executor_briefed` event records the attempt number and how many prior
attempts were briefed.

## What it does not change

The Oracle's repair, the sufficiency audit and the fresh Judger run exactly
as before; the briefing adds information to the executor and removes none of
the audits. A first attempt receives `(none)`, as before.
