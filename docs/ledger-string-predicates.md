# Smaller mechanical record proofs

`ledger.tool-call-string@2` extends the existing typed execution registry, not
the verification architecture. Compiler, Critic, Oracle, executor and Judger
remain distinct. Oracle must explicitly bind the predicate and record scope;
the result still receives a fresh independent Judger audit. No task-name
dispatch, model change, verdict reuse or accepted-method reuse is introduced.

See [adapter semantics](adapters.md) for the exact predicate contract. Version
1 remains registered and unchanged. Both versions share the same bounded
ledger scanner: wrong types, missing in-scope fields, malformed records and
out-of-range watermarks cannot silently become successful checks.

## Why this targets measured work

In the previous cold diagnostic, control's 12 agent-discharge calls consumed
about 3.32 million input tokens. One record-check chain alone consumed roughly
172,000 discharge input tokens plus 308,000 Judger input tokens. Repeated model
inspection of a ledger is much more expensive than local JSON serialization.
These are cumulative input counts, not a monetary bill or critical-path sum.

The new executor computes explicit string predicates and retains a small
proof. A successful existential query needs one matching witness, not dozens
of matching values. A failed universal query needs one counterexample. The
full declared ledger interval is still scanned and its counts/digests retained;
the receipt does not claim its single witness lists all candidate values.

This applies the general recommendation to reduce repeated generated/read
content and use conventional computation where appropriate. It does **not**
apply suggestions to merge model stages or lower model capability.
[Official latency guidance](https://developers.openai.com/api/docs/guides/latency-optimization).

## Local evidence

Using the identity-corrected `extract-moves-from-video__nFHKtnx` source, both
versions scanned all 294 retained events / 147 string candidates:

| Equivalent existence query | Version 1 | Version 2 |
| --- | ---: | ---: |
| Outcome | holds | holds |
| Values included in receipt | 64 | 1 |
| Receipt file bytes | 28,763 | 3,053 |

The new receipt is **9.42× smaller**. This is a byte-size comparison, not a
9.42× model-cost or end-to-end latency result. `nonblank` on the same source
also returned holds with a 3,059-byte receipt. Local execution took about
3–4 milliseconds in this single run; no speedup is inferred from that noise.

Separately, 96 version-1 cases were compared directly with tracked revision
`7d400d6`: all resulting canonical evidence values and fallback reasons were
identical. The cases crossed selectors, exact expected values, absence
permissions, tool scopes and session predicates. Further unit tests cover
Unicode whitespace, first/last/all/any, invalid inputs, watermarks, exact
session/agent scope and fresh Judger rejection/fallback.

## Paid probe boundary

`probe_record_game.py` runs one operator-specified, precompiled **record**
obligation through the existing BindingGame. Each arm has a fresh sealed-input
copy, Oracle call, executor and independent Judger, without a method cache.
It records attempted-call usage and elapsed time even when a chain fails.
Only the router is toggled between the two arms; the Oracle is not forced to
select version 2. A run without actual adapter selection is not exposure to
the optimization.

This diagnostic deliberately excludes extraction, Compiler/Critic, other
obligations, supporting claims and candidate ranking. Their costs are unknown
in the probe, not zero. It cannot establish full-task no-loss, online
performance or the overall 10× goal. It exists to reject an ineffective
optimization before paying for a complete benchmark comparison.

Input preparation and local receipt measurements are retained under
`experiments/terminal_bench/results/gv-record-probe-input.diqidV/`.
The original trajectory and archived states were not changed. The corrected
source seal is
`7fa0e82294b0e521aa40ca9d188202fe9559591e31a1d283a0a2f6fdc842d991`.
