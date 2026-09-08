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
