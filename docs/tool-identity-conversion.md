# Source tool identity correction

This change is confined to the Terminal-Bench replay converter and two trace
readers. It does not change Obligation Compiler, Critic, Oracle Compiler,
execution, independent Judger, obligation schemas, scheduling, or scoring.

The old replay converter accepted only source `bash` calls, renamed them
`exec`, and wrapped their command in an invented `tools.exec_command` string.
Consequently, an exact ledger query for `bash` could contradict the original
source record. This was one cause of the failed
[cold diagnostic](cold-14a4fa1-diagnostic.md).

## Correction and limits

- Preserve exact source tool names (including case), arguments as JSON,
  session and agent name. No task-name rules or tool alias inference.
- Preserve explicit source call IDs when supplied; otherwise derive stable
  IDs from source position. Repeated conversions are byte-identical.
- Keep missing results pending, instead of inventing empty observations.
- Reject malformed or ambiguous call/result groups before writing a feed.
  The current profile accepts at most one call and one result per step;
  it does not guess pairings for unlabelled multi-call steps.
- Keep unknown tools in the record. Refuse world reconstruction for tools
  the replay executor cannot run; a retained record is not a replay capability.
- Keep legacy trace rendering/replay and support structured `bash` input.

The returned conversion report is diagnostic only. It records counts, profile,
and hashes, excludes top-level gold/metrics, and adds no verifier authority.
Only tool identity and arguments are preserved exactly: non-tool messages and
complete result metadata are **not** represented by this feed. The existing
result output-text projection and reconstructed-world evidence limits remain.
The existing string adapter's existence test also does not prove a command
string is nonempty.

Existing archived feeds and seals are not migrated. Newly converted inputs
must be sealed and evaluated afresh. In particular, old paid results cannot
be relabelled as measurements of this correction.

## Local source audit (2026-09-08)

Read-only input: the original repository's
`experiments/vendor/llm-as-a-verifier/data/terminal_bench_2.1_trajs/mini-swe-agent_deepseek-v4-flash`.
For each accepted source, checked exact ordered tool/argument equality, unique
IDs, call/result counts, deterministic second conversion, parsed commands, and
unchanged source bytes. No command execution, Docker, or paid model calls.

| Task | Accepted / inspected | Calls and results preserved |
| --- | ---: | ---: |
| extract-moves-from-video | 5 / 5 | 755 each |
| gcode-to-text | 2 / 5 | 128 each |
| write-compressor | 3 / 5 | 63 each |

The other five traces contain unlabelled multi-call/multi-result steps and are
explicitly rejected. This is incomplete input-format coverage, not a no-loss
success. The old converter reused the first result for every call in those
steps; restoring broader coverage requires establishing the source pairing
contract, not silently assigning observations.

For `extract-moves-from-video__nFHKtnx`, all 147 calls and results survive.
Source file SHA-256:
`894950daab2230ed4e6daec61504bbce76c92c747ad07f55fd258d95adc3c5c2`.
The new feed is 320,941 bytes; no cost or latency improvement is claimed from
its size. This correctness correction has not yet had a fresh paid A/B run.
The separate final-response/task-artifact binding issue remains unresolved;
an obligation-schema or inter-stage interface change requires discussion first.

Local regression: **1,184 passed, 4 skipped, 4 subtests passed**. The same
pre-existing optional-dependency test module (`test_score_llm_verifier.py`)
was excluded as in the preceding baseline; no new exclusion was added.
