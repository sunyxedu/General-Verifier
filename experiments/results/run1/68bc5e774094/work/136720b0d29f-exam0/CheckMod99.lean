-- Claim under test: for all 1 ≤ n ≤ 200, 99 ∣ (10^n - 1)
-- The bound method's stated theorem `nine_dvd` proves 9 ∣ (10^n - 1),
-- which is a DIFFERENT (weaker) statement than the claim (modulus 99, not 9).
-- Part A executes the bound method literally as specified (modulus 9).
-- Part B tests the actual claim as stated (modulus 99) to see whether the
-- bound method's substitution of 9 for 99 was hiding a refutation.
-- Pure Lean 4 core library, no Mathlib.

-- === Part A: bound method as literally written (modulus 9) ===
theorem pow10_mod9 : ∀ n : Nat, 10 ^ n % 9 = 1 := by
  intro n
  induction n with
  | zero => rfl
  | succ k ih => rw [Nat.pow_succ, Nat.mul_mod, ih]

theorem nine_dvd : ∀ n : Nat, 1 ≤ n → 9 ∣ (10 ^ n - 1) := by
  intro n _
  have h1 : 10 ^ n % 9 = 1 := pow10_mod9 n
  have h2 : 9 * (10 ^ n / 9) + 10 ^ n % 9 = 10 ^ n := Nat.div_add_mod (10 ^ n) 9
  rw [h1] at h2
  exact ⟨10 ^ n / 9, by omega⟩

example : (9:Nat) ∣ (10 ^ 1 - 1) := nine_dvd 1 (by decide)
example : (9:Nat) ∣ (10 ^ 200 - 1) := nine_dvd 200 (by decide)

-- === Part B: the actual claim as stated (modulus 99) ===
-- Endpoint n = 1: 10^1 - 1 = 9. Check whether 99 ∣ 9.
example : ¬ (99:Nat) ∣ (10 ^ 1 - 1) := by decide

-- Endpoint n = 200 (this one does hold, since 200 is even and 10^2 ≡ 1 mod 99,
-- but a single true endpoint cannot rescue a universally-quantified claim
-- already refuted at n = 1).
example : (99:Nat) ∣ (10 ^ 200 - 1) := by decide
