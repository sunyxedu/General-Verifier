#!/usr/bin/env python3
"""
Generate an SMT-LIB2 script for z3 that checks:
  claim: for all n in {1,...,1000}, 24 | (n^3 - n)

Two independent encodings are emitted into the same file:

  (A) Bounded existential search for a counterexample using a free
      integer variable n constrained to 1<=n<=1000:
          exists n. 1<=n<=1000 /\ (n^3-n) mod 24 != 0
      UNSAT  => claim holds for the whole range.
      SAT    => claim is refuted; the model gives a witness n.

  (B) Explicit enumeration: one push/check-sat/pop block per concrete
      n in 1..1000, each asserting the ground fact
          (n^3-n) mod 24 != 0
      so that every single n in the range is individually queried to
      z3 (satisfying the "cover every n, not just samples" obligation),
      with an (echo ...) marker printed before each query so the raw
      z3 output is directly auditable per-n.

NOTE ON THE BOUND METHOD TEXT: the task's "绑定的方法" instructs
encoding "assert(n^3-n)%12==0", i.e. modulus 12. That does not match
the claim actually under test in this exam item, which is stated as
"24 整除 n^3-n". Modulus 12 belongs to a different (sibling) claim in
this benchmark family. To faithfully test the claim as stated, this
script uses modulus 24. See report.md for the flagged discrepancy and
a cross-check against modulus 12.
"""

MOD = 24
N_MIN, N_MAX = 1, 1000

lines = []
lines.append("(set-option :produce-models true)")
lines.append("")
lines.append("; --- Part A: bounded existential negation (ForAll-equivalent) ---")
lines.append("(declare-const n Int)")
lines.append(f"(assert (>= n {N_MIN}))")
lines.append(f"(assert (<= n {N_MAX}))")
lines.append(f"(assert (not (= (mod (- (* n n n) n) {MOD}) 0)))")
lines.append("(echo \"PART_A_RESULT:\")")
lines.append("(check-sat)")
lines.append("(get-model)")
lines.append("")
lines.append("; --- Part B: explicit per-n ground enumeration, n=1..1000 ---")
for n in range(N_MIN, N_MAX + 1):
    val = n ** 3 - n
    lines.append(f"(push)")
    lines.append(f"(echo \"PART_B n={n}:\")")
    lines.append(f"(assert (not (= (mod {val} {MOD}) 0)))")
    lines.append(f"(check-sat)")
    lines.append(f"(pop)")

with open("check.smt2", "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"wrote check.smt2 with {len(lines)} lines, modulus={MOD}, range={N_MIN}..{N_MAX}")
