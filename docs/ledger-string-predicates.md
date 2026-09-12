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

## Luna diagnostic result (2026-09-08)

Frozen source commit `34114c02ea4b9cb2bc34601a7c1c18e12437af46`, fingerprint
`b5703c6cf727671f88b5d349b0dc63f63c88ec89926ec95f0d98d8bc32463ce6`.
The fixed order was control then typed, each in a fresh process and cloned
sealed state. Model: GPT-5.6-Luna, medium effort, standard tier. No method
cache, append delta, outer retry, or forced adapter selection was used.

| Measured one-obligation chain | Control | Typed | Control / typed |
| --- | ---: | ---: | ---: |
| Complete child-process wall seconds | 144.343 | 88.247 | 1.636× |
| Attempted model calls | 3 | 2 | 1.50× |
| Uncached input tokens | 108,748 | 38,055 | 2.858× |
| Cached input tokens | 433,664 | 133,120 | 3.258× |
| Output tokens | 5,798 | 3,364 | 1.724× |

Both accounting audits completed with no findings; both source/input pins
remained unchanged. The typed Oracle selected `ledger.tool-call-string@2`
with `any`, exact `bash`, `/input/keystrokes`, `nonblank`, and watermark 294.
It executed once with no fallback. Both chains returned `holds`, record
basis, decisive strength, no bridges/premises, and a fresh Judger's
yes/yes/n/a audit. This is agreement on **one positive record obligation**,
not an equality claim about their different evidence bytes or whole tasks.

The typed execution receipt was 3,549 bytes; its final bound artifact was
5,954 bytes (control final artifact: 4,512 bytes). These are distinct layers.
The earlier 9.42× size result compared v1 and v2 on identical local queries;
it is **not** the paid agent-versus-adapter artifact ratio or model speedup.

The discharge model call disappeared, but Oracle grew from 9.80 to 27.04
seconds and Judger only fell from 70.67 to 61.05 seconds. Typed Oracle/Judger
input totals were 59,091 / 112,084 tokens. This identifies remaining work,
not evidence for skipping either stage. Raw telemetry cannot establish
exactly which permitted files Judger reread.

Cost figures are observed token components, not a verified monetary bill.
The single fixed-order pair has sampling/cache/order confounds and does not
establish a general 10× reduction. Upstream compilation, other obligations,
support claims, ranking and online latency remain excluded, not free.

Raw records: `experiments/terminal_bench/results/gv-record-probe-runs.xz0WIj/`
(`pair-process.json`, each arm's `probe-result.json`, usage and span ledgers).
The external process timer includes child startup and accounting; the probe's
internal walls (144.230 / 88.121 seconds) end before accounting.

Additional local online/offline parity tests append a late counterexample:
an unchanged watermark preserves the exact artifact, while an extended
watermark produces a fresh refutation. Record method reuse remains disabled.
Regression after these tests: 1,251 passed, 4 skipped, 4 subtests passed;
the same pre-existing optional `test_score_llm_verifier.py` module was excluded.

## Architecture boundary audit

Relative to `7d400d6`, `34114c0` changes production code only in
`verifier/execution/adapters.py`. The game, kernel, server, offline pipeline,
configuration and backend are unchanged. Version 2 still requires an Oracle
MethodIR, mechanical structural/basis checks and a fresh Judger. Its own
reuse flag is false and the existing game cache additionally requires world
basis, so record checks cannot enter that shortcut.

Separate pre-existing behavior must not be hidden by that statement: the
shipped typed-adapter path also injects an accepted-method cache. Eligible
world-method warm hits can reuse prior Oracle/Judger work in an already
audited direction while executing fresh evidence. This predates this change;
the current diagnostic explicitly disabled it. If preserving the original
architecture means every candidate must call both roles anew, future
multi-candidate evaluation needs a separately disabled method cache. There
is currently no environment switch that disables only that cache while
keeping adapters. This interpretation needs user confirmation before a
policy change; the current work does not silently change the default.
