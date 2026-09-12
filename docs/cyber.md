# Cybersecurity profile

`--cyber` selects cybersecurity roles for the existing General Verifier
pipeline. It uses the same claim graph, obligation nodes, binding game,
evidence rules, status fold, and certificate implementation as ordinary GV.

```text
Solver claims and recorded effects
    → Cyber Obligation Compiler → Cyber Critic
    → Cyber Oracle Compiler → Cyber Discharger → Cyber Judger
    → ordinary GV certificate
```

All five roles are agents. Each receives its own complete prompt and tool
palette. The compiler derives necessary conditions from the controlling task;
the critic checks whether those conditions omit a requirement or introduce an
unrequested stronger one. The oracle binds an available verification method,
the discharger produces evidence, and the judger independently checks the
method's sufficiency and the execution's fidelity.

The roles apply to vulnerability reproduction, security analysis, remediation,
and other cybersecurity tasks. Their prompts contain no benchmark task IDs,
project-specific functions, payload recipes, or expected answers.

## Start the MCP server

```sh
python -m verifier.mcp_server --state-dir .gv --cyber
```

The candidate workflow requires an authoritative task at `.gv/task.md`; create
it before starting the server. Ordinary claims can also be used without a
separate task statement, with their own wording as the available contract.
The ordinary `commit`, `verify`, `certificate`, and `status` verbs remain the
verification interface. Ordinary final claims are compiled against the entire
task independently of the solver's supporting claims.

The profile also works with those ordinary verbs alone, including multi-file
security fixes and workflows without an evaluator. Candidate tools are an
optional single-file evidence and delivery interface.

The cyber tools additionally record checkpoints, freeze candidate files, and
run a host-configured evaluator. The hosting process configures the project
root, evaluator, and delivery destination. Evaluators and destinations are not
chosen by instructions embedded in a candidate or its output.

For a Codex-backed verifier, select its runtime explicitly, for example:

```sh
python -m verifier.mcp_server --state-dir .gv --cyber \
  --seats codex --model gpt-5.6-luna --effort max --service-tier fast \
  --project-root /absolute/project --cyber-final-destination answer.bin
```

`--cyber-evaluator-command-json` accepts a host-supplied argv array with exactly
one standalone `"{candidate}"` argument, such as
`["/absolute/evaluator", "{candidate}"]`. The executable is pinned. Declare
additional program or data dependencies with repeated `--cyber-evaluator-pin`
arguments; this includes an entrypoint script when invoking an interpreter.
An environment label records the host's description; the evaluator command's
hosting workflow supplies any required container or sandbox isolation.

Evaluator timeouts are unset by default. The cyber Codex verifier also has no
per-role wall-clock timeout. This profile does not add a task deadline or a
resume/restart policy for the solver.

## Candidates and observations

A registered candidate is one immutable file with an exact SHA-256 digest and
size, limited to 16 MiB by the optional candidate tool. It may contain an input,
program, report, or patch. Registration records
identity; it does not establish correctness. Source files can subsequently
change without changing the registered bytes.

An evaluator run records the exact candidate, evaluator configuration, and
captured output. The recorded tool result becomes a normal Effect claim.
An Effect proves that an observation was recorded. A positive exit status,
crash message, or output saying "verified" does not establish the task's
security condition. The normal oracle/discharger/judger path must connect the
observation to the candidate, real target, entry point, environment, and
condition required by the task.

Source comments, candidate contents, evaluator output, and solver-authored
claims are evidence rather than verification instructions. Solver-generated
artifacts cannot act as the authority deciding their own correctness. Task
requirements are not weakened by a positive observation or a verified
supporting claim.

Use checkpoints for working theories and next actions. Candidate ranking,
control-count heuristics, deadlines, and separate relevance verdicts do not
authorize final delivery.

## Certificate and delivery

`cyber_finalize_candidate` creates a host-bound final claim and submits it to
the ordinary GV pipeline. Its certificate explicitly has candidate scope:

- The complete task's artifact and security semantics must be checked.
- The exact registered candidate substitutes only for the configured canonical
  file whose physical delivery is pending.
- Other required outputs and task postconditions remain ordinary checks.
- The candidate's digest, size, destination, task, and verification profile are
  bound to the certificate and the obligation context.

Candidate scope does not claim that canonical delivery has already happened.
It does not apply to public `commit(kind="final")` calls, even when their text
imitates a candidate binding. Supporting claims cannot eliminate the final's
independent task requirements.

Only an authenticated, matching certificate with verdict `verified` permits
materialization. `supported`, `partial`, `refuted`, and `unverified` leave
delivery pending. The finalizer copies the certified bytes and mechanically
checks the actual destination's identity, digest, and size before recording a
delivery receipt. The receipt records an exact write, not an official benchmark
score or a new security verdict.

The finalizer requires an existing parent directory inside the configured
project root and refuses symlinks and existing output files. Keep proposed
candidates at separate paths and reserve a fresh canonical destination.

Restart and replay retain the host-bound context and the ordinary append-only
evidence history. A different profile or task cannot silently inherit a
previous candidate certificate. Verification context participates in obligation
identity; ordinary GV obligation identities remain compatible.

Repeating a successful finalization checks the retained certificate, current
registered bytes, and actual destination and returns the original receipt
without repeating verification or writing another file. A stale final or new host observations require fresh
adjudication. If receipt recording fails, the finalizer removes only its own
unchanged output; it preserves any substituted file.

## Validation

The service tests use deterministic backends through the actual GV pipeline.
They verify the role sequence, independent final contract, normal Effect
claims, context isolation and replay, rejection of mismatched certificates,
and exact materialization only after a verified candidate certificate. These
checks validate the integration; they do not establish benchmark accuracy.
