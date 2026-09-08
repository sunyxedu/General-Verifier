# Compact inspection experiment

Status: explicit opt-in, off by default. No result-preservation, monetary-cost,
or 10× claim. This changes neither the verification architecture nor the native
Codex base instructions.

The corrected gcode binding-only diagnostic at `da154d6` spent 74.202 seconds
inside its two sequential Codex calls out of 74.505 seconds of process wall.
Its Judger returned 214,089 command-output bytes while the adapter receipt was
3,175 bytes. The content-free diagnostics do not identify those output files
or prove which content was reread; they do establish that a compact receipt
alone did not produce compact tool output. Removing all work outside those
two calls would improve this observed wall by only about 1.004×.

The experiment supplies an evidence-inspection preference only to executor
and Judger: when complete local computation suffices, return the exact facts,
counts, source identities and integrity checks needed for the audit instead
of unnecessarily displaying entire sources. Complete required input coverage,
access to additional contents, method scope, promised artifacts, absence rules,
and independent semantic judgment remain mandatory. A digest or truncated view
does not prove uninspected content. No tools, permissions, role schemas, models,
effort, tiers or production retry budgets change. Record obligations still
cannot be decided by replaying historical actions.

Enable explicitly with `CodexBackend(compact_inspection=True)` or the diagnostic
probe's `--compact-inspection 1`. The probe default is `0`, and there is no
ambient environment switch or default production rollout. Both live and offline
backend paths support the option; paid online performance is not established.
The actual note is captured at backend construction, included in runtime/cache
identity, and recorded on every attempted-call bill. The disabled identity is
unchanged. The probe rejects a backend whose actual profile disagrees with its
declared experiment setting.

The ordinary offline scoring path also accepts an explicit
`compact_inspection=False` argument, exposed by Terminal-Bench `score.py` as
`--compact-inspection 0|1`. It is forwarded to both extraction and kernel
backends; only the existing executor/Judger preference changes their prompts.
Authority-only backend clones preserve the captured runtime instructions.
The scoring config always records the boolean and includes it in its hash, so
an old config with no field is not silently relabeled as a new measured run.
An injected backend whose actual profile differs from the declared setting is
rejected. This plumbing does not enable contract or accepted-method caches,
merge stages, alter budgets, or establish end-to-end speedup.

Official documentation permits replacing Codex's built-in instructions, but
this experiment does **not** use that broader mechanism. It follows the narrower
guidance to define efficient tool-loop criteria while preserving evidence,
permission and output requirements. Published performance ranges are not project
measurements. [Codex configuration](https://learn.chatgpt.com/docs/config-file/config-reference),
[GPT-5.6-family prompting guidance](https://developers.openai.com/api/docs/guides/prompt-guidance-gpt-5p6).

## Evaluation

Compare fresh default/opt-in runs at the same frozen source revision, input
seal, case hash, Luna medium/standard settings, typed-adapter configuration,
and no accepted-method cache. Count every attempted model call and failure.
Check actual adapter exposure, final outcome/basis/strength, fresh Judger audits,
all billable token categories, process wall, and native command-output bytes.
Lower output bytes alone are not lower monetary cost or result preservation.

Keep the separate exact-identity/identity-independent semantic gate at
`--compact-inspection 0` first, so that it diagnoses the earlier adapter parity
question without adding another treatment. No single-obligation experiment
stands in for full online/offline, multi-candidate, end-to-end acceptance.

### First frozen profile pair: no measured gain

At `1b9b7a7aaa2cbddbae2286dfaaaa05031a1a96cb`, runs are retained under
`experiments/terminal_bench/results/gv-semantic-profile-runs.xTR0C0/`.
The gcode pair ran sequentially, default then compact, after all eight separate
synthetic semantic jobs terminated. Both used the same precompiled record
obligation, sealed input, typed adapters enabled, no method cache, and Luna
medium/standard. All attempt bills and span sidecars were complete, with file
hashes matching their result records.

| Metric | Default | Compact |
| --- | ---: | ---: |
| Process wall seconds | 139.330 | 307.125 |
| Model attempts | 2 | 5 |
| Uncached input tokens | 85,828 | 113,361 |
| Cached input tokens | 181,248 | 575,744 |
| Output tokens | 5,125 | 11,401 |
| Actual adapter executions | 1 | 0 |
| Final result | holds / record / decisive | holds / record / decisive |

Compact was 2.20× slower in this observation and used more of every priced token
category; no monetary tariff was assumed. This is not an isolated causal
estimate: the unchanged Oracle prompt generated different methods. Default
used `ledger.tool-call-string@2`; compact selected `ledger.event` with an invalid
subject reference, causing agent fallback. Its first execution omitted the
promised exact event, which the independent Judger rejected; normal redo then
delivered the event and passed. Those fallback/redo costs remain included.
The preference therefore remains experimental and off by default. It has not
established result preservation beyond this individual obligation, much less
10× end-to-end improvement.
