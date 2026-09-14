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

The five seats retain their complete agent prompts and tool contracts.
The discharger can execute an exact typed operation from the ordinary adapter
registry, with the agent executor as fallback. Independent Judger review is
still required; adapter success is not a task verdict.
The compiler derives necessary conditions from the controlling task;
the critic checks whether those conditions omit a requirement or introduce an
unrequested stronger one. The oracle binds an available verification method,
the discharger produces evidence, and the judger independently checks the
method's sufficiency and the execution's fidelity.

The roles apply to vulnerability reproduction, security analysis, remediation,
and other cybersecurity tasks. Their prompts contain no benchmark task IDs,
project-specific functions, payload recipes, or expected answers.

## Security claims and methods

A cyber claim remains an ordinary `assertion` or `final`. Its text identifies
the subject and security property, relevant actor capabilities/trust boundary,
preconditions, target/build/configuration/entry point, time, input domain,
quantifier and controlling authority. Include only material dimensions;
unspecified facts remain unknown. Identity, recorded observations and world
security conclusions are separate propositions. Checkpoints hold hypotheses
and exploration, rather than expanding the proof graph with every search.

The obligation compiler defines **what must hold**, preserving the task's
scope and modality. The oracle compiler defines **how to check it** from the
authority, capability inventory and versioned adapter catalog, not from an
observed answer. Compiler and pre-execution-auditor seats do not receive raw
candidate execution capsules; discharger and judger do, scoped to the actual
claim's dependency closure or host-bound subject. Oracle repair feedback is a
mechanical projection of the declared method, structural rejection categories
and retry/availability metadata, not raw outcomes, witnesses or audit prose.
Execution diagnostics remain available to the executor and independent Judger;
they never redefine acceptance predicates or expected values.

Whole-task coverage belongs to completion decomposition/certification.
Intermediate claims retain their residual scope; each Oracle, Discharger and
Judger handles only its current obligation, not another complete task audit.
The host explicitly marks ordinary claims separately from candidate phases.
Registering a candidate does not make every source or analysis assertion owe
a candidate/destination binding. A claim that actually asserts such a binding
still has to establish it; public finals still cover the complete task.

Reusable method families cover identity/content comparisons, authenticated
retained executions, isolated dynamic witnesses, pinned static reasoning, and
remediation/coverage checks. These are selection guidance, not mandatory extra
tests. One admissible witness can establish an existential reproduction;
finite successful tests alone cannot establish unbounded safety. Controls,
repeated trials and broader impact are required only when the inference or
task needs them.

Typed host operations have their own public-evidence boundary, independent of
agent shell permissions. They read the claim's authenticated frozen ledger,
declared public project subjects and permitted retained evidence, never private
evaluator/controller state or credentials. Descriptor-bound ancestor walks
prevent a concurrent parent symlink replacement from redirecting a file read.
The optional sufficiency pre-audit role is also cyber-specialized, but remains
disabled unless the operator enables the ordinary GV pre-audit flag.

The cyber Codex seat policy is shared by the ordinary MCP service, offline
scoring and the benchmark sidecar. Planner shells default-deny the state and
project trees, with read access only to captured task/baseline authority and
method descriptors. Execution/audit seats can read the declared public project,
but write only owned scratch; shell network access and external search are
disabled. Host-provided custom backends are responsible for their own boundary.
Optional compact evidence transport preserves the exact frozen record even
when a runtime opts out; it never falls back to a later live ledger.

Native CLI policy telemetry validates JSON incrementally, omitting oversized
string bodies while retaining event structure. Large command output therefore
does not hide a subsequent search identifier or invalidate an otherwise parsed
event. The structural frame remains bounded; malformed JSON, invalid UTF-8,
oversized keys/frames and failed readers still make policy evidence incomplete.
Raw CLI output follows the existing private capture path, and no queries or
results are added to telemetry. This guard detects observed activity; it is not
a proof that every internal tool registry is search-free.

The Linux benchmark sidecar maps each execution/audit call's `/tmp` to a
host-created private directory before starting the native CLI sandbox. This
supports unmodified programs with fixed temporary paths without exposing the
sidecar's shared `/tmp`. Namespace setup fails closed; planner shells receive
no additional grant, and `/proc` remains unavailable to seat commands. CLI final
messages and image uploads stay in separate host-private capture storage.
Discharger temporary files are retained in its work directory for the ordinary
bounded snapshot before cleanup; a transport manifest identifies the mapping
and omitted aliases/special files. Retention authenticates no result or build.
Other hosting workflows must explicitly select and support this Linux setup.

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

In cyber MCP mode, heavy requests return a durable `job_id` and may initially
return `pending=true`. Poll `verification_job(job_id)` or reconnect with the
same request to recover the same adjudication; disconnecting a waiter does not
cancel the worker. A retained result is historical, not a fresh correctness or
official-score assertion. `verification_retry(job_id)` explicitly retries an
unfinished terminal attempt after rechecking its scope. There is no retry-count
deadline or automatic replacement solver thread. Operator configuration,
wire protocols and host capabilities participate in state identity.

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

Agent snapshots retain a text view, which can change invalid UTF-8 bytes and
line endings. Selected private regular files whose byte identity would be lost,
and files containing NUL bytes, also receive exact content-addressed `raw_files`
references before executor cleanup. This additional pass inherits the selected
file boundary with independent raw-body budgets of 20 MiB per file and
40 MiB total. The inline text budgets stay at 65,536 characters per file and
2,000,000 total; oversized or unstorable raw bodies are marked incomplete.
Raw references are locators for independent inspection, not correctness
verdicts or automatic image attachments. Raster retention keeps its separate
existing byte and attachment rules.
The Discharger receives the inline, raw-body, and raster limits before
execution, using the same constants as the snapshot implementation. This
general budget disclosure changes no retention limit or evidence rule.

Successful (`holds`) executor results use an empty counterexample witness
(normalized to null) and `witness_kind=none`. The Judger inspects the retained
outputs and provenance for positive support; the counterexample-kind audit
applies to refutations. This schema distinction changes no evidence requirement
or correctness verdict.

Use checkpoints for working theories and next actions. Candidate ranking,
control-count heuristics, deadlines, and separate relevance verdicts do not
authorize final delivery.

## Certificate and delivery

`cyber_finalize_candidate` uses two phases of the same ordinary GV pipeline:
candidate authorization, then actual whole-task completion. It does not seat
a second independent verifier or replace the original kernel.

The host first creates an independently checked authorization **assertion**:

- The complete task's artifact and security semantics must be checked.
- Its current subject address is the real immutable blob, not the future
  canonical destination. Only the host's explicit single-file existence,
  regular-file and exact-byte delivery postconditions are pending.
- Other required outputs and task postconditions remain ordinary checks.
- The candidate's digest, size, destination, task, and verification profile are
  bound to the certificate and the obligation context.

This assertion is registered by the host, never inferred from solver text.
It admits no assumed solver premises or residualized task requirements.
Its scoped certificate uses an independent `root`, not a task-completion
`final`, and does not claim canonical delivery has happened.
It does not apply to public `commit(kind="final")` calls, even when their text
imitates a candidate binding. Supporting claims cannot eliminate independent
task requirements.

Materialization requires an authenticated, current candidate authorization.
A full `verified` GV certificate is sufficient. If every obligation holds but
an advisory SHOULD's weaker evidence makes the full verdict `supported`, the
candidate policy can retain a separately hash-linked MUST acceptance record:
every MUST must have direct decisive world evidence, no assumptions or bridges,
and every SHOULD must still hold at its declared floor. The full GV certificate
and advisory evidence remain unchanged and retained. Missing/refuted checks,
weaker MUST evidence and stale bindings cannot authorize delivery.
This is a mechanical delivery policy over the same GV evidence, not another
agent verifier or an evaluator-positive shortcut.
Before publishing bytes, the host durably records the planned output inode,
parent identity, hash and delivery transaction. It then copies the certified
bytes and checks actual delivery. Next it submits an ordinary, independent
whole-task **final** through Compiler, Critic, Oracle, Discharger and Judger.
This final checks the complete original task against the actual delivered
state. Only its `verified` certificate together with current byte delivery
permits a completion receipt. Candidate MUST acceptance cannot upgrade a
`supported` or failed whole-task final. No receipt asserts an official score.

The finalizer requires an existing parent directory inside the configured
project root and refuses symlinks and existing output files. Keep proposed
candidates at separate paths and reserve a fresh canonical destination.

Restart and replay retain the host-bound context and the ordinary append-only
evidence history. A different profile or task cannot silently inherit a
previous candidate certificate. Verification context participates in obligation
identity; ordinary GV obligation identities remain compatible.

Repeating a successful finalization checks the retained certificate, current
registered bytes, and actual destination and returns the original receipt
without repeating verification or writing another file. A new delivery
transaction cannot reuse an old completion, even if the OS reuses an inode.
A stale final or new host observations require fresh adjudication. Failed
completion or receipt recording rolls back only the host's own unchanged
output; substituted files are preserved. Replay retains failed historical
finals without requiring their rolled-back destinations to still exist.

## Validation

The service tests use deterministic backends through the actual GV pipeline.
They verify the role sequence, independent final contract, normal Effect
claims, context isolation and replay, rejection of mismatched certificates,
and exact materialization only after authorization, whole-task completion
gating, and real process-crash recovery around canonical publication. These
checks validate the integration; they do not establish benchmark accuracy.

The pinned CyberGym agent image extends the same Codex base with generic
Clang/compiler runtimes, CMake, Ninja, LLVM, LLD and GDB. This is an environment
capability change, not a model upgrade or a payload recipe. The launcher records
an offline, networkless toolchain metadata probe. The agent remains arm64 while
the official runtime is amd64; that mismatch is declared, and tool presence
does not establish build success, runtime fidelity or an official result.
An incomplete or malformed official fixed result is an infrastructure error,
not a model failure or success; a completed vulnerable negative needs no fixed
execution.

## Read-only record receipt replay

When an agent fallback executes a `record.program@1` method bound to a fixed
ledger prefix, the framework supplies the ordinary Judger with a bounded replay
of the shipped pure VM against that same claim-scoped record boundary. The
summary carries the actual projection identity and typed event-reference
validation, allowing comparison with a manually reconstructed receipt's own
normalizer and self-reported validation flag. This does
not certify the obligation, reject aggregate predicates, execute historical
commands, or replay third-party adapters. Observed-EOF methods keep the existing
path because a new observation could select a different boundary.
