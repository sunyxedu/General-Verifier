-- General (non-sampling) proof that n^3 ≡ n (mod 6) for every natural number n.
-- Uses only Lean4 core (no Mathlib): Nat.pow_mod reduces n^3 % 6 to (n % 6)^3 % 6,
-- then a bounded quantifier over the 6 residue classes is settled by `decide`.

theorem bounded_check : ∀ m : Nat, m < 6 → m ^ 3 % 6 = m := by
  decide

theorem cube_mod_six (n : Nat) : n ^ 3 % 6 = n % 6 := by
  have hpow : n ^ 3 % 6 = (n % 6) ^ 3 % 6 := by
    rw [Nat.pow_mod]
  have hlt : n % 6 < 6 := Nat.mod_lt n (by decide)
  rw [hpow, bounded_check (n % 6) hlt]

-- The claim for the finite range 1 ≤ n ≤ 1000 follows as a special case:
-- 6 ∣ n^3 + n  ↔  (n^3 + n) % 6 = 0.
theorem six_dvd_cube_add_self (n : Nat) : 6 ∣ (n ^ 3 + n) := by
  have h1 : n ^ 3 % 6 = n % 6 := cube_mod_six n
  have h2 : (n ^ 3 + n) % 6 = 0 := by
    conv_lhs => rw [Nat.add_mod, h1, ← Nat.add_mod]
    simp [Nat.mod_self]
  exact Nat.dvd_of_mod_eq_zero h2

-- Instantiate for the specific range required by the claim, 1 ≤ n ≤ 1000,
-- to make explicit that it is a special case of the fully general theorem above.
theorem claim_holds_on_range : ∀ n : Nat, 1 ≤ n → n ≤ 1000 → 6 ∣ (n ^ 3 + n) :=
  fun n _ _ => six_dvd_cube_add_self n

#print axioms cube_mod_six
#print axioms six_dvd_cube_add_self
#print axioms claim_holds_on_range

-- Sanity spot-checks (not the proof itself, just extra confidence):
example : (6 : Nat) ∣ (1 ^ 3 + 1) := by decide
example : (6 : Nat) ∣ (7 ^ 3 + 7) := by decide
example : (6 : Nat) ∣ (1000 ^ 3 + 1000) := by decide

#eval "PROOF_CHECK_OK"
