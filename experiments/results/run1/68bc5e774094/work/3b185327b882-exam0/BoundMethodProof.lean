-- Reproduction of the "bound method" instrument described in the exam ticket.
-- It proves 9 | (10^n - 1) for all n, NOT 11 | (10^n - 1) as the claim requires.

theorem pow10_mod9 : ∀ n : Nat, 10 ^ n % 9 = 1 := by
  intro n
  induction n with
  | zero => rfl
  | succ k ih =>
    rw [Nat.pow_succ, Nat.mul_mod, ih]

theorem nine_dvd_pow10_sub_one : ∀ n : Nat, 1 ≤ n → n ≤ 200 → 9 ∣ (10 ^ n - 1) := by
  intro n _ _
  have h1 : 10 ^ n % 9 = 1 := pow10_mod9 n
  have h2 : 9 * (10 ^ n / 9) + 10 ^ n % 9 = 10 ^ n := Nat.div_add_mod (10 ^ n) 9
  rw [h1] at h2
  exact ⟨10 ^ n / 9, by omega⟩

#check @nine_dvd_pow10_sub_one
