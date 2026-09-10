# The compiler's exclusion ledger

## Problem

The runner-interface quotient is a rule both the Compiler and the Critic
apply: a clause that only tells the candidate how to talk to its runner
(make a tool call, end with a standalone completion command, do not work
after submitting) generates no obligation, because varying the candidate's
compliance changes no task-owned property and no acceptance consequence.
Applied implicitly, the rule was inconsistent across runs: in one cold run
the Critic demanded two such clauses back into a contract, the executor's
exact-command check then failed against the recording's tool dialect, and a
correct candidate was refuted. The control arm retains these clauses in most
of its contracts and both of its false refutations come from them.

## Mechanism

The Compiler's output schema gains an optional `excluded` list: every source
clause that reads as an instruction to the candidate but generated no
obligation under the quotient or the governing-modality rule, with the
controlling reason (`verifier/llm/agents.py`). The Critic receives the
ledger next to the contract and reviews each ruling: it may demand an
excluded clause back only by naming the non-circular task-owned property
and the source-grounded acceptance consequence that survive the quotient
attack; otherwise the ruling stands and is not re-litigated in later rounds.
Append-only repairs carry the ledger forward. The `obligations_compiled`
event retains the ledger, so a contract's exclusions are auditable after the
fact.

## What it does not change

The quotient itself, the Critic's fail-closed rule, and the kernel's
handling of obligations are unchanged; the ledger is never verified and
never becomes an obligation. A Compiler that emits no ledger is read as an
empty one.
