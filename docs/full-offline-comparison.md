# Full offline comparison: architecture and measurement boundary

The optimization must preserve the original division of responsibility:

`Extractor → Obligation Compiler → Critic → Oracle Compiler → executor → independent Judger`

An adapter is an executor selected by the Oracle Compiler. It does not define
the obligations, replace the Critic, or decide whether evidence satisfies an
obligation. Evidence prefetch changes delivery of bounded, verifier-owned
evidence, not the Judger's authority or acceptance standard. Unsupported
adapter inputs and unavailable prefetch retain the ordinary path.

## Cold comparison profile

`experiments/terminal_bench/run_full_offline_worker.py` runs the selected frozen
source's own `verifier.offline.score` once in a fresh process. It does not replay
compiled contracts, inject a replacement kernel, or score old derived state.

| Setting | Control | Treatment |
| --- | --- | --- |
| Source | Original main `be3581a9a5c61973fb2d98ed6c88625d95cda8dc` | Explicitly pinned optimization commit |
| Model / effort / tier | GPT-5.6-Luna / medium / standard | Same |
| Typed adapters | Off | On |
| Judger prefetch | Off | On; actual attachments audited separately |
| Typed-adapter Judger packet | Off | On since `71d0a49`; restricted no-tools Judger rows audited separately |
| Certificate scope | Full slice | `final` since `71d0a49`: only the final claim is settled; narration claims are exported open |
| Contract cache / accepted-method reuse | Off / none | Off / none under fresh-prefetch profile |
| Extraction retry bound | 3 | 3 |
| Critic and backend retry bounds | Native defaults | Native defaults |
| Parallel claims / obligations / seat ceiling | 4 / 4 / 12 | 4 / 4 / 12 |

Both arms receive the same separately pinned offline isolation policy and
content-free attempt observer. This is an isolation-normalized original-source
comparison, **not an untouched original runtime**. The policy overlay changes
only the backend's offline CLI policy arguments; the observer preserves native
launching, parsing, retries, return values and exceptions.

Since `82bef17` the observer accounting also recognises the verifier-owned
`gv-authority-*` temporary scratch in which the candidate-blind final contract
is compiled; those compiler and Critic attempts are counted separately.

Runs are built without hand-written plans by
`experiments/terminal_bench/build_full_offline_comparison_plan.py`, which
freezes both commits with `git archive`, copies a strict package set or a
general evaluation package's model inputs, pins the support copies and writes
balanced worker/comparison plans.

The worker checks source and input seals, retains the certificate, requires
observed compiler/Critic/Oracle/Judger role coverage, rejects cache-hit events,
and inventories every treatment attempt against the common observer. Missing
cost buckets or terminal records make accounting incomplete. Failures are
retained; they are not removed from the denominator.

## What this does not establish

A complete worker result is not a quality or 10× pass. The outer comparison
is implemented by `run_full_offline_comparison.py`: it measures
process-start-to-exit latency, authenticates source/input/support files,
compares compiled contracts and per-obligation outcomes, and retains all
attempted jobs. Contract drift is not normalized away. An enabled prefetch flag
is not evidence that a packet was actually used; sampled acceptance requires an
actual verifier-owned packet attachment.

The two prepared gcode candidate inputs contain historical reconstructed `/app`
snapshots, not original live task worlds. They can support a same-input relative
comparison, but their historical Terminal Bench rewards are not ground truth
for the reconstructed snapshots. They also do not establish generality across
tasks or online-mode performance.

The 10× cost and end-to-end latency target, without result loss, remains
unproven. No architecture change is authorized to make that target easier.
