-- Obligation: for the claim "for all 1 ≤ n ≤ 1000, 4 ∣ n^3 - n", it must in particular hold
-- that 2 ∣ n^3 - n (n^3 - n is even), and the proof of this sub-obligation must cover both
-- the odd and even residue classes of n, not just spot-check a handful of samples.
--
-- Bound method: pure Lean4 core, no Mathlib. Prove the general lemma
-- `∀ n : Nat, n ^ 3 % 2 = n % 2` via `Nat.pow_mod` + `Nat.mod_lt` reducing to a finite,
-- machine-decided check over the two residue classes m = 0 (even) and m = 1 (odd), then
-- derive `2 ∣ (n ^ 3 - n)`.

-- Step 1: exhaustive, machine-checked base case over the 2 residue classes 0 (even) and
-- 1 (odd) — `decide` evaluates both m = 0 and m = 1 as concrete computations.
theorem parity_check : ∀ m : Nat, m < 2 → (m ^ 3) % 2 = m % 2 := by
  decide

-- Step 2: reduce n^3 % 2 to (n % 2)^3 % 2 via Nat.pow_mod, then close with parity_check.
-- This covers ALL n : Nat (not just n in [1,1000]) by reduction to its residue mod 2.
theorem cube_mod_two (n : Nat) : n ^ 3 % 2 = n % 2 := by
  have hpow : n ^ 3 % 2 = (n % 2) ^ 3 % 2 := by
    rw [Nat.pow_mod]
  have hlt : n % 2 < 2 := Nat.mod_lt n (by decide)
  rw [hpow, parity_check (n % 2) hlt, Nat.mod_mod]

#print axioms cube_mod_two  -- expect: only 'propext' (or no axioms), no sorryAx

-- Step 3: n^3 % 2 = n % 2 means n^3 and n agree mod 2, i.e. their (truncated Nat) difference
-- is even. `omega` treats `n ^ 3` as an opaque atom and closes this purely from the mod fact,
-- so this still genuinely covers both parities of n rather than sampling specific values.
theorem two_dvd_cube_sub_self (n : Nat) : 2 ∣ (n ^ 3 - n) := by
  have h := cube_mod_two n
  omega

#print axioms two_dvd_cube_sub_self  -- expect: only 'propext' (or no axioms), no sorryAx

-- Sanity: exhibit the odd/even split explicitly and check the sub-obligation holds on the
-- exact required range 1..1000 as an additional, redundant machine check.
def failCount : Nat :=
  ((List.range 1000).map (· + 1)).filter (fun n => ¬ decide (2 ∣ (n ^ 3 - n))) |>.length

#eval failCount  -- number of n in [1,1000] for which 2 ∤ n^3 - n (expect 0)

#eval "PROOF_CHECK_DONE"
