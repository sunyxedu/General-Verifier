# Typed execution adapters

Adapters replace one execution step with a versioned, mechanically checked
operation. The obligation compiler still establishes the requirement; the
Oracle compiler selects a method and its inputs; the Judger checks sufficiency
and fidelity. An unsupported operation, incomplete observation, or invalid
input goes through the existing agent executor.

## Generality contract

An adapter implements an operation on declared evidence. It never receives a
benchmark score, selects behavior from a task name, or contains task answers.
Addresses, expected values, selectors, and record scopes are method arguments.
Their relevance to the task remains the Judger's responsibility. Catalogs
describe available operations; the model prompts remain domain independent.

| Adapter | Mechanical observation | Limitation |
| --- | --- | --- |
| `filesystem.exists@1` | Address presence and artifact kind | Absence requires an explicit presence obligation |
| `filesystem.sha256@1` | File bytes compared with a declared digest | The digest's authority is audited independently |
| `document.json-value@1` | Exact, type-sensitive JSON Pointer comparison | Strict UTF-8 JSON only |
| `document.json-predicate@1` | Bounded structural equality, array membership/length, and array-item shape/type predicates | Closed algebra over strict UTF-8 JSON only |
| `document.json-predicate@2` | The same bounded predicates, with object-member presence independent of optional type constraints | Explicit JSON Pointer required, including `""` for the root |
| `evidence.program@1` | A bounded pure program checks relations across declared world files | Holds-only; any false check, unsupported syntax, or limit falls back |
| `evidence.program@2` | The same pure program over a stable complete vector of private regular-file snapshots | A valid false `require` fails only the method; it is never a refutation |
| `ledger.event@1` | One retained event's type and payload subset | Establishes record facts only |
| `ledger.tool-call-string@1` | Exact string selection within an explicit ledger interval | Cannot interpret or execute command text |
| `ledger.tool-call-string@2` | Explicit existence, equality, nonempty or nonblank string predicate | Record facts only; no command-language interpretation |
| `ledger.sequence@1` | A bounded ordered sequence of exact source-event predicates and relations | Holds-only; every no-match falls back |
| `ledger.sequence@2` | The same positive receipt, with a stable complete no-match recorded as method failure | Never converts no-match into an obligation verdict |
| `browser.html@1` | Dialog, console and DOM observations within a finite time window | Optional pinned backend; standalone documents only |

`ledger.tool-call-string` requires `selector`, `value_pointer`, and
`through_seq`; optional arguments are `expected`, `tool_names`, `after_seq`,
and `payload_equals` (exact predicates indexed by payload JSON Pointers).
The scope is `(after_seq, through_seq]`. Its subject declaration is
`{"ref":"ledger:events","provenance":"ledger","role":"subject"}`.
A matching tool call with an unreadable value invalidates the fast path.
Failure to find an existential match is absence, not a positive contradiction.

Version 2 additionally requires `predicate`: `exists`, `equals`, `nonempty`,
or `nonblank`. `equals` requires a string `expected`; the other predicates
forbid that field. `exists` requires `selector=any`. `nonempty` means length
greater than zero; `nonblank` means Python's Unicode-aware `str.strip()` leaves
at least one character. Neither proves that a string is a valid command or
that the command succeeded. These distinctions remain for the Oracle and
independent Judger to assess against the obligation.

Version 2 retains one witness for a successful existential query, or one
counterexample for a failed universal query. It still scans the full declared
interval and records its counts and digests; a witness is not presented as a
complete listing. Version 1 keeps its original semantics and receipt format.
Both versions require fresh execution and an independent Judger; neither is
eligible for accepted-method reuse.

`ledger.sequence` scans a fixed source-event protocol and may apply the frozen
`record.program@1` decoder to inert command envelopes. Version 1 remains
unchanged. Version 2 returns `failed` only when a stable, bounded observed-EOF
scan proves that its declared positive sequence has no witness. It never
returns `refuted` for that condition, and prefix no-match, malformed ledgers,
races, ambiguity, or any resource cap still use the agent fallback. The normal
single Oracle repair remains available after failure; every positive result
still receives a fresh independent Judger.

An Oracle backend receives the compact identity index only when its effective
tool boundary is hard read-only. The default Agent backend checks the actual
Oracle palette against its read-only allowlist. Codex and unknown or custom
backends receive the established safe inline baseline (the old basis slice for
world obligations and the full catalog for record obligations). Codex stays
inline because its macOS execution boundary may add filesystem provenance
metadata while reading an otherwise unchanged descriptor; retrying that safe
host-side mutation costs more than the shorter initial prompt saves.

Within the index, a fixed 4 KiB budget inlines as many complete small contracts
as fit, ordered mechanically by canonical byte length and digest. Every other
available entry points to an exact, read-only, content-addressed descriptor
under verifier state; the Oracle cannot bind from a summary. Record-basis pages
mechanically reconstruct the full catalog byte for byte. A world-basis root
publishes only eligible world pages and marks record-only identities as below
its basis floor. Publication, integrity, size, or registry-drift failure falls
back to a freshly computed safe inline baseline. A page-integrity race discards
that payload and uses the one existing schema retry with the inline baseline;
normal initial and repair enumeration add no calls. Execution, cache identity,
and typed Judger replay always bind the full registry, never this display view,
and an unusable typed binding still follows the ordinary custom executor path.

`filesystem.exists` reports `address`, `exists`, `actual_kind`,
`expected_kind`, and `world_mode` for every substantive result. A missing
target has `actual_kind="absent"`; an existing target also carries its file
identity. This operation does not inspect file contents.

`document.json-predicate` accepts one world address and 1--16 conjoined
predicate objects. `valid` checks the whole document. `equals` and
`array.contains` take an RFC 6901 `pointer` and an arbitrary JSON `expected`
value. `array.length` adds `operator=eq|gte|lte` and a non-negative integer
`expected`. `array.items-object` declares `required` property names,
`properties` mapped to closed JSON type names, and whether additional
properties are allowed. It scans every array item and retains bounded failure
details. Empty arrays therefore satisfy an item-shape predicate vacuously;
combine it with `array.length` when non-emptiness is required. Missing members
or values fail by absence and can refute only an obligation explicitly marked
`refutable_by_absence`; wrong types, extra fields, and invalid JSON are positive
world-state contradictions. The adapter never evaluates JSONPath, regexes,
schemas, expressions, or code. It is eligible for the restricted typed packet,
which still invokes a fresh independent Judger, but not accepted-method verdict
reuse. Its receipt retains the complete parsed document when its canonical JSON
is at most 16,384 characters; larger documents carry the canonical digest,
size, and an explicit incomplete marker instead.

Version 1 requires every `required` member to also have a type declaration in
`properties`. Version 2 separates those constraints: `required` means presence
only, while `properties` constrains a member's type when that member appears.
With `additional_properties=false`, the allowed names are
`required ∪ properties`. Their union is capped at 64 names. Both versions
require an explicit RFC 6901 `pointer`; use `pointer:""` for the document root.
Neither version silently repairs or defaults a missing pointer.

For example, `/input/cmd` and `/input/keystrokes` are different explicit
record dialects. A raw program containing a nested tool invocation needs the
agent executor; this adapter does not infer what the program executes.

## Pure evidence programs

`evidence.program@1` is a task-independent executor for relations that the
fixed adapters cannot express. The Oracle supplies a small Python-shaped
program and an ordered list of declared world files. The verifier parses and
interprets that language itself; it never invokes Python execution. Imports,
attributes, ambient paths, commands, processes, network, time, randomness,
exceptions, mutation, `eval`, and `exec` do not exist. The complete AST is
validated before execution, including unreachable branches and unused
functions.

The language provides bounded byte, strict UTF-8, strict JSON, indexing,
slicing, loops, pure functions, checked arithmetic, endian decoding, SHA-256,
and type-sensitive comparisons. Every declared runtime file must be freshly
read. Every successful `require` must dynamically depend on a declared subject;
hard-coded truth, labels, and observations cannot manufacture that dependency.
All resource caps are published in the adapter catalog.

Version 1 is deliberately holds-only. A false requirement, malformed input,
unsupported construct, incomplete read, or exhausted cap makes it inapplicable
and routes to the unchanged agent executor.

Version 2 changes only a fully validated false requirement. It opens every
declared runtime input as a distinct, private regular file without following
path symlinks, reads every file to EOF under per-file and aggregate byte caps,
and keeps the files open while the pure program runs. Before and after the
program it rebinds every path and checks the file and world-view identities.
Every `require` must have a bounded third `observed` argument covering the
subject slots used by its predicate, and every declared runtime slot must have
been read before a requirement may fail. Under those conditions the first
false `require` returns method `failed`. This means only that the exact MethodIR
did not establish the obligation; it is never `refuted`, never an absence
witness, and proves neither directory completeness nor future closure. Any
invalid input, unsafe alias, race, malformed program, or resource boundary
still falls back to the ordinary executor.

A successful receipt from either version is still only evidence: the fresh
independent Judger checks whether the Oracle's program is sufficient for the
obligation and faithful to the task. Restricted typed transport additionally
requires a complete, untruncated observation for every requirement, exact
declared/consumed input binding, and fresh replay before and after the Judger
call. Phase 1 deliberately leaves `evidence.program@2` outside that restricted
transport allowlist, so its positive receipts use the ordinary fresh toolful
Judger.

## Optional browser observer

The browser adapter accepts an address, `wait_ms` (0–5000), and exact assertions:

```json
{
  "address": "/site/document.html",
  "wait_ms": 100,
  "assertions": [
    {"kind": "selector.count", "selector": "main", "expected": 1},
    {"kind": "dialog.messages", "expected": []}
  ]
}
```

Other kinds are `selector.texts` (ordered text-content list) and
`console.messages` (ordered message list, optionally filtered by level).
The evidence records navigation through page freeze after `load + wait_ms`.
It makes no statement about subsequent behavior or other browser environments.
The backend records its browser digest, version, fixed display/locale/timezone
profile, dialog policy, and observed input digest.

Enable it only with verifier-owned configuration:

```bash
export GV_BROWSER_OBSERVER="/absolute/repo/scripts/gv_browser_observer.py"
export GV_BROWSER_OBSERVER_SHA256="<SHA-256 of that observer>"
export GV_BROWSER_EXECUTABLE="/absolute/path/to/chromium"
export GV_BROWSER_EXECUTABLE_SHA256="<SHA-256 of that browser binary>"
```

All four values are required. They cannot be supplied through MethodIR. No
runtime is downloaded or discovered automatically. The bundled POSIX observer
uses a fresh profile and blocks requests beyond its single document. Any
blocked resource request makes the adapter inapplicable, because the observed
page would have an incomplete environment. This request policy is not an OS
security sandbox; run untrusted active content only inside an independently
isolated verifier deployment.

## Reuse and experiments

Each adapter explicitly declares whether it supports accepted-method reuse.
The current opt-ins are the three pure world-file adapters. Reuse also requires
an exact obligation, task authority, backend/catalog identity, decisive method,
delivered subject inputs, and a previously successful Judger audit. Execution
always runs again on the current candidate. The accepted signature applies
only to its recorded `source_outcome`. The cache keeps at most one separately
approved signature per direction. The first opposite-direction result receives
a fresh, independent Judger audit of that exact MethodIR and new evidence;
only `sufficient=yes`, `faithful=yes`, and `degradation=n/a` may publish it.
Rejection, infrastructure failure, or changed task authority falls back to the
full game. Later candidates may use that direction's signature only for the
exact same MethodIR, with fresh evidence. A decisive one-direction
certificate never implies the other direction. Browser and ledger methods
retain a fresh Judger audit on every binding.

For telemetry, `adapter_discharge.accepted_method_reuse_attempt` records only
an attempted cache execution; `method_cache_hit` records actual audit reuse.
`method_cache_direction_audit` records a new audit, not a free cache hit;
the profiler counts it separately while retaining all model-attempt costs.
Existing published-claim invalidation semantics are unchanged: adapter freshness
applies when a new binding executes, not as a background watch on live files.

`GV_TYPED_ADAPTERS=0` selects the original agent execution path and disables
accepted-method reuse. `1` enables adapters. Both modes use the same compilers
and Judger. The setting and optional observer identity are included in offline
configuration fingerprints, so resume cannot mix the two configurations.

For performance comparisons, run paired candidates at the same code revision,
model, effort, service tier, and parallelism, with separate scoring roots.
Measure verdict changes, calls, uncached input/output tokens, and wall time.
Synthetic speedups do not establish a whole-pipeline speedup on real tasks.

The missing-direction audit removes one redundant Oracle call and adapter
execution when it succeeds. It does not eliminate the first audit, optimize
unrelated obligations, or establish identical outputs to independently sampled
Oracle/Judger runs. Its total cost includes failed fast audits and fallbacks.
