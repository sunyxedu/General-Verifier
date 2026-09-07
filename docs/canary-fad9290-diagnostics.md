# Canary `fad9290`: first video control diagnostic

This is an in-progress diagnostic snapshot, not a final evaluation result. It
covers source revision `fad9290` and the first control run ending in
`1d8e1857`. No historical gold data was consulted. The figures below must not
be used to claim a cost or latency ratio.

## Runtime snapshot

At roughly 9 minutes 33 seconds, 36 model calls had completed successfully and
all had complete usage records. They accounted for 1,809 model-seconds,
3.871M input tokens (833k uncached and 3.038M cached), and 75.4k output tokens.
The run did not yet have a terminal result.

Discharge and Judger dominated this snapshot: six Discharge calls consumed
773 seconds and five Judger calls consumed 504 seconds. Together they were
about 71% of accumulated model time and 79% of input tokens. Compiler and
Oracle calls were not the observed long pole. Parallel obligations reduced
wall-clock time, but serial `Discharge -> Judger -> redo/repair` chains created
the tail.

All completed calls exited successfully with complete usage. The captured
artifacts contained no certificate, TLS, package-install, fetch, permission, or
backend error signal. This is evidence only for this snapshot; it is not a
general statement about every network path in the runtime.

## Three-step method binding chain

One file-presence obligation required an initial execution and two redos. The
three evidence artifacts remained present and readable, contained only small
text scripts and outputs, and were not image-dependent. Each audit found the
reported positive lookup sufficient; the failures concerned method identity or
input provenance rather than missing, truncated, or inaccessible evidence.

1. Initial execution: the agent used a relative-path shell lookup rather than
   the exact verifier captured-world lookup promised by the method. It also
   labelled the world manifest as `task_given`. The audit returned
   `sufficient=yes, faithful=no`.
2. Redo 1: the observed procedure now matched the promised lookup, and the
   manifest was correctly labelled `verifier`. The delivered subject file was
   still incorrectly labelled `task_given`, so the audit again returned
   `sufficient=yes, faithful=no`.
3. Redo 2: the subject was labelled `delivered`, while the manifest remained
   `verifier` and the task specification `task_given`. The audit returned
   `sufficient=yes, faithful=yes`, and the method bound successfully.

The repeated `faithful=no` label therefore represented two different defects.
The first redo made real progress; identical top-level verdict labels alone are
not a safe basis for stopping a redo chain.

## Treatment comparison to make

The immediate comparison is whether the existing verifier-owned
`filesystem.exists` adapter naturally removes this agent execution and Judger
redo chain while preserving the outcome. A deterministic adapter can emit its
own method identity and provenance when those facts are known from the sealed
world mapping.

The harness must not infer that every path in a captured tree is `delivered`.
A captured tree may also contain genuine task-given inputs, particularly when
no baseline exists. Any future mechanical provenance filling or validation
must use explicit source evidence rather than path membership alone.
