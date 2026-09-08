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
| `ledger.event@1` | One retained event's type and payload subset | Establishes record facts only |
| `ledger.tool-call-string@1` | Exact string selection within an explicit ledger interval | Cannot interpret or execute command text |
| `ledger.tool-call-string@2` | Explicit existence, equality, nonempty or nonblank string predicate | Record facts only; no command-language interpretation |
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

For example, `/input/cmd` and `/input/keystrokes` are different explicit
record dialects. A raw program containing a nested tool invocation needs the
agent executor; this adapter does not infer what the program executes.

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
