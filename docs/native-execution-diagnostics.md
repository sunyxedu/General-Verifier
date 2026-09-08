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
