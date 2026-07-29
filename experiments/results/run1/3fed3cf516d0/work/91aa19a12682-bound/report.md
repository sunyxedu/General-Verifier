# Check: 30 | n^5-n for 1≤n≤300, decomposed via 2/3/5 kernel checks

## Method
Standalone Lean 4 (v4.32.2) file, no Mathlib. Three independent `decide`-based
theorems verify 2, 3, 5 each divide n^5-n for all n in [1,300], plus two
meta-level counterexamples showing "6 | m" does not imply "5 | m" (and
similarly "10 | m" does not imply "3 | m") — formalizing why the claim's
obligation (proving 2,3,5 separately, not just 6 or 10) is non-vacuous.

## Instrument
lean 4.32.2 (arm64-apple-darwin24.6.0), invoked as `lean check.lean`.

## Result
Compiled with exit code 0, no errors. `#print axioms` shows dvd2/dvd3/dvd5
depend only on `propext` (no `sorryAx`), confirming the `decide` proofs are
genuine kernel-checked computations, not stubs. The two counterexample
`example`s (¬∀m,6∣m→5∣m and ¬∀m,10∣m→3∣m) also type-checked with no errors.

## Verdict
holds — the bound method's decomposition (2, 3, 5 independently, plus the
"6/10 insufficient" meta-check) was successfully executed and all parts
verified true in the Lean kernel.

## Artifacts
- check.lean — the Lean source (this file's sibling)
- lean_output.txt — full stdout/stderr from `lean check.lean` (exit 0)
