# Lossless Judger evidence prefetch

Experimental, explicitly enabled, and **off by default**. No paid speedup,
full-E2E result preservation, or 10× result is established for this profile.

The preceding eight Judger-only diagnostics each used one successful CLI
attempt, but collectively consumed 535,566 input tokens and 431.1 role seconds
for artifacts of roughly 3 KB. Aggregate native diagnostics show 25 command
items; they do not reveal exact internal model-loop counts or command contents.
Repeated fixed context is a plausible bottleneck, not an established causal
breakdown. This experiment targets mechanical evidence-reading round trips,
following [OpenAI latency guidance](https://developers.openai.com/api/docs/guides/latency-optimization).
Its separate suggestion to combine sequential model roles is **not** adopted.

## Contract

The existing BindingGame constructs a packet from its own ArtifactStore and
EventLog, never from a model-supplied arbitrary path. It transports complete
UTF-8 artifact JSON, verifies the parsed-value content address, and records
the original path, raw hash and byte count. For record methods, a small complete
ledger can also be included. Total raw content is capped at 16 KiB and rendered
JSON at 64 KiB. No individual file is summarized, filtered or truncated.

If only the ledger exceeds the cap, the complete artifact can still be supplied;
the packet explicitly marks the ledger as absent and requiring original-path
inspection. Such a packet neither reads nor freezes the omitted ledger.
Invalid artifact identity, unsafe paths/links, unavailable platform primitives,
malformed JSON/JSONL, or unstable reads use the original inspection path.

Only the independent Judger receives the private, verifier-created transport
object. Its conditional instruction permits inspecting those exact raw bytes
instead of issuing a redundant file-open call. It still audits method sufficiency,
execution fidelity, provenance, scope, absence and evidence strength itself.
Transport hashes are not semantic approval. All tools, permissions, schemas,
models, effort settings and budgets remain unchanged; the two compilers remain
separate. The native Codex base instructions are not replaced.

Every transported file is checked again before returning a packet and after
Judger completion. A changed file, including a ledger append, invalidates that
verdict. Its cost remains recorded; only the existing error budget can permit
a fresh, non-prefetched retry. No new source epoch or historical time boundary
is imposed. Online/parallel invalidations may increase rather than reduce cost.

## Configuration and evaluation

Enable with `CodexBackend(judger_prefetch=True)`, offline
`score(..., seats="codex", judger_prefetch=True)`, or Terminal-Bench
`score.py --seats codex --judger-prefetch 1`. There is no ambient enable flag.
Config identity records the boolean and enabled protocol/caps. Captured runtime
notes are part of backend identity and preserved by authority-only backend
clones. Every attempt bill records the actual profile, whether bytes were
attached, and the packet hash. Prefetch mode disables accepted-method reuse,
retaining fresh Oracle and Judger calls.

The isolated direction probe accepts `--judger-prefetch 0|1`. Compare both modes
at one frozen source, with identical methods, evidence and Luna medium/standard
settings. Verify positive acceptance, negative rejection, all attempt costs,
actual packet exposure, input seals and artifact hashes. This probe excludes
the other pipeline stages and cannot stand in for full online/offline or
multi-candidate Terminal-Bench acceptance.
