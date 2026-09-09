# Pre-execution sufficiency audit

## Problem

The binding game runs `Oracle → executor → independent Judger`. The Judger
answers two questions: was the method sufficient for the obligation as
written, and was it executed faithfully. The first question depends only on
the method and the obligation, yet it was asked only after the executor had
run. In the retained gcode pairs the executor spent 209 s, 298 s and 475 s on
methods the Judger then rejected with `sufficient=no`; the Oracle's one
repair pass followed and produced the right answer in 14 s.

## Mechanism

`GV_SUFFICIENCY_PREAUDIT=1` adds a tool-less, single-turn `sufficiency`
seat (`SUFFICIENCY_AGENT`) that the binding game calls for each
agent-executed method immediately before the executor starts. It receives
the obligation with its basis, severity, clause and absence rule, the
authoritative task context, the evidence boundary, and the proposed method
(procedure, promised artifact, basis, strength, bridges). It never sees an
execution or an artifact and cannot settle anything.

- `sufficient=no` is handled exactly like a post-execution `sufficient=no`:
  the method is rejected with `sufficiency pre-audit: <gap>` and the Oracle's
  bounded repair pass receives the gap.
- `sufficient=yes` changes nothing: the executor runs and the fresh Judger
  audits sufficiency and fidelity as before.
- A seat failure (transport, schema) is recorded in a `sufficiency_audited`
  event and execution proceeds.
- Typed adapter methods are not pre-audited; they execute in milliseconds and
  the Judger audits them afterwards.

The seat is part of the scoring configuration identity and is off by
default. The v1 full-offline treatment profile enables it.

## What it does not change

The Compiler, Critic, Oracle, executor and Judger keep their prompts,
schemas, tools, retry budgets and verdict rules. The seat adds one cheap
call per agent-executed method and removes the executor's work only when a
structural gap is named before execution.
