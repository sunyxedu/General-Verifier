# Typed-adapter Terminal-Bench gate

This gate answers two separate questions without changing the verifier's
architecture. It records them as separate comparisons:

1. `adapter-isolation`: with the authority contract cache enabled in both
   arms, does changing only the executor preserve the exact contract outcome?
2. `full-e2e`: does `(typed adapters=1, contract cache=1)` beat the unchanged
   `(0, 0)` pipeline by 10x, and what paired output and contract drift does it
   actually exhibit?

The machine-readable candidate set is
`typed_adapter_eval_manifest.json`. It contains no reward or historical
verdict. Those labels are joined only after every verifier seat has stopped.

## Candidate set

The acceptance tier has 20 distinct archived candidates, five for each of four
ordinary, non-security tasks. The archive is already local; no solver or paid
verifier run is needed to establish the inventory.

| Task | Archive job | reward=1 / reward=0 | historical Luna `verified` |
| --- | --- | ---: | ---: |
| `gcode-to-text` | `lav-official-g2-k5` | 2 / 3 | 0 |
| `write-compressor` | `lav-official-g6-k5` | 3 / 2 | 1 |
| `extract-moves-from-video` | `lav-official-g2-k5` | 1 / 4 | 0 |
| `sqlite-db-truncate` | `rec-codex-terra-medium-tb21-strat20-k5-r1` | 4 / 1 | 4 |

Thus the full gate is balanced at ten positive and ten negative rewards and
contains five historical `verified` results. The three-task, 15-candidate core
and six-candidate canary are early-stop screens only; neither may establish the
10x/no-loss claim. Security, exploit reproduction, vulnerability repair,
sanitizer bypass, password recovery, cracking, and cryptanalysis tasks are
outside this suite.

The canary has only one historically positive and one historically negative
candidate per task. It can catch a regression cheaply, but it cannot measure
both-direction warm reuse or support a general cost/quality claim.

## Retain contract text; freeze it only for adapter isolation

An ordinary independent adapter-only A/B run is confounded: the obligation
compiler and critic sample different decompositions, so fewer calls in one arm
need not come from adapters. A valid adapter-isolation experiment needs a
sealed obligation-contract replay seam with these properties:

- The task contract is built from `task.md`, logical task-address metadata,
  and compiler/critic identities only. It never receives reward, historical
  verdicts/certificates, official verifier output, final answer bytes, world
  bytes, or the solver transcript.
- Candidate-specific claim contracts preserve the original support Compiler's
  semantic inputs: sealed statements, complete graph context, premise
  ids/statuses, task context, world mapping, and authenticated effect call/result
  referent context. The fixed snapshot is after extraction and before any
  verification. Solver records needed to identify a proposition are not
  verifier adjudication outcomes; deleting them would change the compiler's
  task. Prior Judger results, evidence and certificates remain forbidden.
- The bundle binds the task hash, claim ids/text hashes, ordered obligations
  including clause/modality/evidence floor, compiler and critic prompts,
  model/effort, and schema. Both arms must report the same bundle hash.
- The oracle compiler is intentionally **not** frozen. Adapter selection is
  the treatment being tested; both arms start from the same obligations and
  independently bind methods.
- A replayed contract never carries a verdict, method, evidence, or candidate
  artifact. Candidate evidence is freshly obtained in each arm.

This snapshot policy is **diagnostic only**. Production may compile a support
claim after its premises settle; compiling against an unadjudicated snapshot
is not asserted to be equivalent to that interleaved scheduling. The ordinary
online/offline stages, their responsibilities, full support audit and
verification standard remain unchanged. Isolation can diagnose adapter effects
but cannot replace the independent full-e2e zero-loss/10x gate.

Cold contract-build calls are not free setup. Record their attempts in a
process-private usage file and add the identical measured build cost and wall
time to each arm. Warm and batch runs build once per task and include that cost
in the task/batch total. If the shared compiler floor alone exceeds 10% of the
control, the 10x target correctly fails by Amdahl's law.

Every current result retains these fields verbatim, in compiler order:

- `contract_audit_status` — `available` only when the compiler event and its
  certificate alignment are both valid; missing/empty/malformed audit data
  remains diagnostic and does not change the underlying score;
- `final_obligation_contract` — id, exact obligation text, exact source
  clause, modality, evidence floor, absence rule, and premise scope;
- `final_obligation_outcomes_ordered` — the aligned evaluated flag, status,
  basis, strength, and reason.

No normalized task-scope digest is used. Two differently worded LLM contracts
can change a quantifier while producing the same digest scheme or status
histogram; the analyzer flags any exact contract drift and includes both raw
arms for review. Matching verdict/rank/histogram with contract drift is
reported as observed agreement with unproven contract equivalence, never as a
zero-loss result.

A future adapter-isolation run additionally needs the shared-bundle fields:

- `obligation_contract_bundle_sha256`
- `task_contract_sha256`
- `compiler_fingerprint`
- `final_obligation_outcomes_sha256` (ordered obligation id/status pairs)

`analyze_typed_adapter_eval.py` fails closed if any shared-bundle field is
absent or differs between adapter-isolation arms. The runner keeps that mode
unavailable until a real sealed replay seam exists.

## Three latency measurements

Every phase uses GPT-5.6-Luna, medium effort, standard tier, offline mode,
browser observer disabled, and the same source commit.

### Cold

Run one fresh process for one candidate and one arm, with a fresh scoring root,
fresh process cache, and `--no-shared-toolchain`. Use `--workers 1`. Measure
from contract-build start through `score.py` exit. Alternate arm order by a
precommitted hash of the trial id. The primary cold metric is the sum over all
unique candidates; treatment must also avoid a wall regression on every
candidate.

### Warm

Run one fresh process per task and arm with `--workers 1` and all five distinct
candidates. The lexical first trial named by `warm_seed_trial` is the seed;
the other four are held out. Nothing from the cold run is reusable. Report the
whole invocation wall time including contract build, seed, misses, fallbacks,
failed calls, and all held-out candidates. A hit-only suffix is diagnostic,
never an acceptance metric. The cache must rerun the adapter and produce a
fresh evidence artifact; reusing a candidate verdict or evidence fails the
quality gate.

### Batch

Use fresh roots again. Run one process per archive job with `--workers 4`, in a
fixed sequential job order for both arms. The metric is first process start to
last process finish, not accumulated parallel model seconds. All candidates,
contract builds, fallbacks, failures, and retries are included.

Candidates repeat across cold/warm/batch only because these are different
state regimes. They are never pooled as 60 independent quality samples, and a
candidate is never repeated inside one phase to manufacture an exact hit.
Every phase also pins `GV_MAX_CONCURRENT_SEATS=12`; ambient process settings
cannot silently change model-call concurrency between arms. The base `PATH`
is pinned to `/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin`; the verifier
prepends its own run-private toolchain directory when a warm instrument needs
one. Offline seats receive read-only access to the manifest-owned public
runtime trees `/opt/homebrew/bin`, `/opt/homebrew/Cellar`, and
`/opt/homebrew/opt`; no task or candidate field can extend that allowlist.

## Run ledger and complete cost

Wrap each `score.py` process and append one
`gv-terminal-bench-eval-run/v1` row to a run JSONL. The required shape is in
the analyzer's module docstring. Each result and usage path must be private to
that process. `wall_includes_contract_build=true` is an assertion made by the
wrapper from its actual timing boundary, not added by hand later.

The Codex usage ledger must have one terminal row for every real CLI attempt.
The accepted additive attempt schema includes `call_id`, unique `attempt_id`,
1-based `attempt_index`, `started`, `attempt_status`, `usage_status`, and
`usage_complete`. Provider retries share a call id; an outer AgentRole retry
gets a new call id. Failed calls and timeouts stay in the ledger.

Cost is complete only when the wrapper's telemetry audit says complete and
every started attempt has `usage_status=complete`, `usage_complete=true`, and
numeric uncached-input, cached-input, and output token counts. A null or
missing count makes `cost_gate=incomplete`; the analyzer does not sum known
rows and call the result a saving.

Without a pinned billing record, the 10x token/call proxy is deliberately a
Pareto gate: treatment/control must be at most 0.1 separately for started
attempts, uncached input, cached input, and output. Exact billed cost may be
reported in addition, but never reconstructed from an uncertain price table.

## Acceptance and limits

For all 20 candidates and for each phase:

- both arms must use the same source fingerprint and immutable candidate seal;
- adapter-isolation requires the same contract bundle and ordered obligation
  outcomes; full-E2E retains both raw contracts and flags exact drift;
- verdict, verdict strength, and final-obligation status histogram must match;
- there may be no unscoreable result and no reward-direction/ranking loss;
- cost must be complete and all four cost ratios must be at most 0.1;
- cold, warm, and batch end-to-end wall ratios must each be at most 0.1, with
  no individual cold regression.

A full-E2E contract drift makes the zero-loss conclusion `inconclusive`, even
when every observed verdict and rank matches and the 10x resource gates pass.
Only an exact-contract comparison can isolate executor result preservation;
that is why adapter-isolation remains a separate, currently unavailable gate.

Run the analyzer only after all scoring processes finish:

```bash
.venv/bin/python experiments/terminal_bench/analyze_typed_adapter_eval.py \
      --manifest experiments/terminal_bench/typed_adapter_eval_manifest.json \
  --runs /absolute/path/to/eval-runs.jsonl \
  --tier acceptance \
  --comparison full-e2e \
  --output /absolute/path/to/eval-report.json
```

The driver is plan-only unless the paid confirmation string is present. Point
it at a frozen `git archive` extraction and at the existing absolute Harbor
jobs root; the verifier fingerprint is checked before and after every process.
`warm` keeps all selected candidates for one task in one `score.py` process,
which is required for process-local caches. `all` runs all three regimes into
one fresh output root.

Do not place the source archive or run output under `/private/tmp`, `/tmp`, or
the process temp directory: the native `:minimal` policy can read siblings in
those host temp trees. Use two independent directories under `~/Documents`.
The runner rejects a paid output root under the known temp trees.

The offline seat policy must deny the host filesystem root and grant reads only
to the current sealed state plus a minimal trusted runtime (and exact writable
instrument/toolchain roots). A merely read-only host is not isolated: it would
let a seat inspect Harbor reward files, wrapper outputs, or sibling candidates.
The source manifest's omission of rewards is not a substitute for this runtime
boundary.

```bash
.venv/bin/python experiments/terminal_bench/run_typed_adapter_eval.py \
  --manifest experiments/terminal_bench/typed_adapter_eval_manifest.json \
  --source-root /absolute/path/to/frozen-source \
  --source-commit FULL_40_HEX_COMMIT \
  --jobs-root /absolute/path/to/experiments/terminal_bench/results/jobs \
  --comparison full-e2e --tier canary --phase warm --arm both

# Add only after inspecting the plan:
#   --execute --confirm-paid GPT-5.6-Luna
```

The runner still refuses adapter-isolation execution. The new
`verifier.contract_replay` input-cloning/snapshot helpers,
`verifier.engine.contract_bundle` pinned-contract boundary and
`shared_setup.py` accounting helpers are foundations, not a working paid-run
orchestrator. They are not imported by ordinary scoring. Execution stays
disabled until compile-only preparation, fresh replay, private bundle pinning,
and complete shared-cost accounting are integrated and independently tested.
No completed arm's compiled ledger may become a shared source snapshot.
Each support contract needs its own fresh copy of that input ledger: leaving
earlier Compiler/Critic events in a shared ledger would change what later
seats can read. The shared final authority cache may retain its separate,
operator-attested Critic receipt; it does not authorize sharing support
verification state.

The loader binds a provider to one kernel/store/log and the entire immutable
claim structure. Explicit activation rechecks the fresh snapshot; runtime
lookups allow normal evidence/status growth but reject structural drift.
An operator-only diagnostic marker makes ordinary `offline.score` refuse the
directory before any model call, without deleting prior evidence. Re-running
extraction alone would not remove old artifacts, so returning to normal
scoring requires a new directory copied from sealed immutable inputs only.

Claim-to-call lineage is a required accounting gate: a valid
bundle and a complete but unrelated setup usage file cannot establish the
cost of preparing that bundle. Every contract needs its actual Compiler and
Critic span/receipt linkage, and warm final-cache reuse must link to the
original shared, charged pass instead of inventing a fresh compilation.
Generic token addition alone is not proof of this correspondence.
The standalone validator now requires exact bidirectional Compiler/Critic
claim coverage and pins `setup_id`. Multi-candidate warm/batch setups remain
explicitly incomplete until source-receipt lineage is implemented; matching
claim IDs alone is not a replacement for the producer's real pass receipts.

`prepare_contract_entry.py` now provides one diagnostic-only **direct** claim
preparation. It creates a new post-extraction state, invokes the existing
Compiler/Critic path once (retaining its normal internal retry budget), and
materializes without evidence work. Each claim gets a separate fresh ledger;
the final uses a new local authority cache, never an ambient cache hit.
Operator receipts and usage remain outside the role-readable state. The
producer checks exact rounds, successful schema/model-call status chains,
every provider attempt, source/config/whole-graph identity, and stable ledger
bytes. Missing usage or any failed boundary produces an inconclusive receipt
and no candidate entry; the created files and incurred attempts are retained.

This output is still an **untrusted candidate**, not a replay authorization.
Its payload binding explicitly relies on the pinned Kernel's terminal-pass
control flow, not a cryptographic signature of a returned role result. An
independent receipt consumer, parent-owned extraction/all-claims lifecycle,
warm source-receipt lineage and fresh replay coordinator remain required
before execution can be enabled. One claim's receipt explicitly excludes
extraction and replay costs. Local process-stub counters test accounting
semantics only: they are not paid benchmark data or evidence for 10x gains.
No ordinary online/offline stage, prompt, scheduler or validation gate is
changed by this experiment module.

A zero observed verdict/rank mismatch on 20 candidates has a 95%
rule-of-three-style upper bound of about 13.9% on an unconstrained disagreement
rate. That bound describes observed outputs only; it neither resolves contract
drift nor proves universal zero loss.
