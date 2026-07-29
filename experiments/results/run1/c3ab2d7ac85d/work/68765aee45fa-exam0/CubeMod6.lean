-- Task: verify the claim "for all 1 ≤ n ≤ 1000, 6 ∣ n^3 + n" using the bound method:
-- prove the general lemma `∀ n : Nat, n ^ 3 % 6 = n % 6` in Lean4 core (no Mathlib),
-- via Nat.pow_mod plus a decidable bounded quantifier over the 6 residue classes,
-- and then check whether the claim follows as a special case.

-- Step 1: the general, non-sampling lemma (true, and proved for ALL naturals, not by
-- extrapolating from a few sampled n): reduce n^3 mod 6 to (n mod 6)^3 mod 6, then
-- exhaustively decide the finite bounded statement over the 6 residues 0..5.
theorem bounded_check : ∀ m : Nat, m < 6 → m ^ 3 % 6 = m := by
  decide

theorem cube_mod_six (n : Nat) : n ^ 3 % 6 = n % 6 := by
  have hpow : n ^ 3 % 6 = (n % 6) ^ 3 % 6 := by
    rw [Nat.pow_mod]
  have hlt : n % 6 < 6 := Nat.mod_lt n (by decide)
  rw [hpow, bounded_check (n % 6) hlt]

#print axioms cube_mod_six  -- should depend only on core axioms (propext), no sorryAx

-- Step 2: `cube_mod_six` says n^3 ≡ n (mod 6), i.e. 6 ∣ (n^3 - n), the product of three
-- consecutive integers (n-1)*n*(n+1). It does NOT give 6 ∣ (n^3 + n) = n*(n^2+1).
-- Attempting the natural derivation from cube_mod_six to the claim fails to type-check,
-- because it is not a valid derivation:
-- theorem attempted_claim (n : Nat) : 6 ∣ (n ^ 3 + n) := by
--   have h1 : n ^ 3 % 6 = n % 6 := cube_mod_six n
--   have h2 : (n ^ 3 + n) % 6 = 0 := by
--     conv_lhs => rw [Nat.add_mod, h1, ← Nat.add_mod]  -- goal left as (n % 6 + n % 6) % 6 = 0, false in general
--   exact Nat.dvd_of_mod_eq_zero h2

-- Directly checking the claim finds a counterexample already at n = 1, well inside [1,1000]:
example : (1 : Nat) ^ 3 + 1 = 2 := by decide
example : ¬ (6 ∣ ((1 : Nat) ^ 3 + 1)) := by decide

-- Formal refutation of the bound claim, restricted to the required range 1 ≤ n ≤ 1000:
theorem claim_refuted : ¬ (∀ n : Nat, 1 ≤ n → n ≤ 1000 → 6 ∣ (n ^ 3 + n)) := by
  intro h
  have := h 1 (by decide) (by decide)
  exact absurd this (by decide)

#print axioms claim_refuted  -- should depend only on core axioms, no sorryAx

-- Exhaustive count of failures over the exact required range 1..1000, computed inside Lean
-- (no external tool), to show the counterexample is not an isolated edge case:
def failCount : Nat := ((List.range 1000).map (· + 1)).filter (fun n => ¬ decide (6 ∣ (n ^ 3 + n))) |>.length

#eval failCount              -- number of n in [1,1000] for which 6 ∤ n^3 + n
#eval ((List.range 1000).map (· + 1)).filter (fun n => ¬ decide (6 ∣ (n ^ 3 + n))) |>.take 10  -- first counterexamples

#eval "PROOF_CHECK_DONE"
