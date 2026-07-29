-- Claim: for all 1 ≤ n ≤ 1000, 6 ∣ n^3 - n
-- Method: prove the general lemma `∀ n : Nat, n^3 % 6 = n % 6` by reducing to a
-- bounded, decidable check over the 6 residue classes mod 6 (via Nat.pow_mod),
-- then instantiate it for n ∈ [1,1000] as a special case of the universal statement.
-- Stays in Lean4 core (no Mathlib): uses Nat.pow_mod, Nat.mod_lt, `decide`, `omega`.

-- Step 1: bounded, decidable fact about residues 0..5
theorem residues_cubed_mod6 : ∀ m : Nat, m < 6 → m ^ 3 % 6 = m := by decide

-- Step 2: general lemma for ALL natural numbers, using Nat.pow_mod to reduce
-- n^3 % 6 to (n % 6)^3 % 6, then applying the bounded fact to n % 6 (which is < 6).
theorem cube_mod6 (n : Nat) : n ^ 3 % 6 = n % 6 := by
  rw [Nat.pow_mod]
  exact residues_cubed_mod6 (n % 6) (Nat.mod_lt n (by decide))

-- Step 3: from n^3 % 6 = n % 6 conclude 6 ∣ n^3 - n (Nat, truncated subtraction is
-- harmless here: for n ≥ 1, n^3 ≥ n so no truncation occurs; omega handles the
-- general linear/mod/dvd reasoning treating n^3 as an opaque value).
theorem six_dvd_cube_sub_self (n : Nat) : 6 ∣ n ^ 3 - n := by
  have h : n ^ 3 % 6 = n % 6 := cube_mod6 n
  have hge : n ≤ n ^ 3 := Nat.le_self_pow (by decide) n
  omega

-- Step 4: instantiate for the specific range 1 ≤ n ≤ 1000 (a special case of the
-- universal statement proved above, NOT a re-derivation by sampling).
theorem claim_range : ∀ n : Nat, 1 ≤ n → n ≤ 1000 → 6 ∣ n ^ 3 - n := by
  intro n _ _
  exact six_dvd_cube_sub_self n

-- Sanity spot-checks (not the proof itself — the proof is `claim_range` above,
-- which is universally quantified and already covers these and every other n).
example : 6 ∣ (1 : Nat) ^ 3 - 1 := six_dvd_cube_sub_self 1
example : 6 ∣ (1000 : Nat) ^ 3 - 1000 := six_dvd_cube_sub_self 1000
example : 6 ∣ (997 : Nat) ^ 3 - 997 := six_dvd_cube_sub_self 997

#print axioms cube_mod6
#print axioms claim_range
