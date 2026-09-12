# Content-free native execution diagnostics

The existing usage ledger measures a complete Codex invocation, not the size
of one model request. To investigate expensive Oracle/Judger runs, attempted
usage records additionally summarize the native JSONL event stream. This is
observability only: no role prompt, model, tool palette, permission, stage,
retry budget, result or token-accounting decision changes.

The stream interface is documented in
[OpenAI's non-interactive mode guide](https://learn.chatgpt.com/docs/non-interactive-mode#make-output-machine-readable).
Our offline invocations use `--ephemeral`, so old session transcripts cannot
be assumed available. Previously completed attempts cannot be retroactively
given these diagnostics.

The `execution_diagnostic` field records event/item counts and observed byte
sizes, not original contents. It deliberately excludes command strings,
messages, tool output, paths and reasoning text. Completed command outputs
are counted once per item ID, so lifecycle updates do not multiply a large
output. Maps use a fixed vocabulary; unknown kinds are counted, not retained.

Parsing has explicit stream, line, event and identifier limits. Malformed,
partial, unknown or over-budget streams cannot look like complete diagnostic
measurements. Diagnostic completeness is separate from usage completeness:
a failure to summarize tool activity must not erase a bill, make missing
usage free, or change a verification verdict.

Interpretation limits:

- Item counts are not internal model-call counts.
- Observed command-output bytes are not total model-input tokens.
- A small returned receipt does not prove Judger avoided reading the ledger.
- These content-free statistics cannot identify the files a model reread.
- Missing or partial diagnostics are not zero tool work.

This supports the next profiling experiment; it is not itself evidence of a
cost or latency reduction. The existing five verification stages and the
independent Judger remain unchanged.

## First measured case: gcode record obligation

Frozen revision `3500185a94972219d5a278b562286f5d85bf788d`, fingerprint
`81e39174cc1569aff385fdec6eaa2dc13f9cd79fd4d28830cef9c00b03de4900`.
The paired record probe used GPT-5.6-Luna medium/standard, typed first and
control second, fresh processes/states, and no method cache. It reused the
archived Compiler obligation `05061657b81e9e416940b7c1e84e677044582a3b9ec1d8c07f5e84c098c87171`
without changing its `refutable_by_absence=false` field. The source contains
68 source-preserving calls and 68 results; no task/world code was replayed
as part of this preparation.

| One-obligation diagnostic | Control | Typed enabled |
| --- | ---: | ---: |
| Complete process wall seconds | 265.916 | 186.242 |
| Attempted model calls | 3 | 3 |
| Uncached input tokens | 90,560 | 127,068 |
| Cached input tokens | 585,984 | 364,032 |
| Output tokens | 11,537 | 7,217 |
| Actual adapter executions | 0 | 0 |

Both returned holds/record/decisive with fresh yes/yes/n/a Judger audits,
empty bridges/premises, complete bills, identical case pins and unchanged
input/code seals. **There was no adapter exposure.** The faster typed-enabled
wall is not a measured adapter execution benefit. Uncached input actually
increased by about 40%; the token components do not establish lower monetary
cost. A single stochastic pair is not general no-loss or 10× evidence.

All six native diagnostics were complete:

| Arm / role | Command items | Returned command-output bytes |
| --- | ---: | ---: |
| Control / Oracle | 0 | 0 |
| Control / executor | 6 | 215,345 |
| Control / Judger | 5 | 150,002 |
| Typed enabled / Oracle | 2 | 15,254 |
| Typed enabled / executor | 5 | 200,516 |
| Typed enabled / Judger | 4 | 239,085 |

The catalog was offered but the Oracle chose a custom full-ledger method.
The new measurements establish substantial returned data and tool activity;
they do not identify exact reread paths or prove why the model chose that
method. This changes the next experiment: test a general preference for an
equally sufficient registered method before custom execution. No task names,
adapter names, forced binding, smaller verification scope or skipped Judger
are introduced. Required evidence/listings remain mandatory. The change is
limited to Oracle's existing selection instructions, not its output schema.

This follows the official emphasis on explicit tool-routing criteria while
preserving evidence, permission and output constraints, rather than importing
the guide's reported gains into this project.
[GPT-5.6-family prompting guidance](https://developers.openai.com/api/docs/guides/prompt-guidance-gpt-5p6).

Input/provenance: `results/gv-record-probe-gcode-input.D627R0/` under
`experiments/terminal_bench/`; case SHA-256
`94de0ce289710371fbee49679ca9042ab260d01064f745fe933b0a4dda836ce1`, seal
`73d8c7fa563db7c8abfa18d4846142f13456a526affa492c41a491e833ee4837`.
Runs and complete usage records: `results/gv-record-gcode-runs.Q5TA9y/`.
The unused partial `gv-record-probe-gcode-input.BC4HhO` copy was not evaluated.

The separate idea of replacing Codex's approximately 17,730-character general
coding-assistant instruction template is only a read-only feasibility finding
from the local model cache. No runtime instruction override was implemented,
and character counts are not a measured token/cost gain.
