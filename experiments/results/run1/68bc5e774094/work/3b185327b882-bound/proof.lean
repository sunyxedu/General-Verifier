-- Bound method for claim A: 对所有 1≤n≤200，9 整除 10^n-1
-- Pure Lean 4 core library, no Mathlib.
-- Step 1: general lemma by induction on n: 10^n % 9 = 1 (holds for ALL n, unbounded).
theorem pow10_mod9 : ∀ n : Nat, 10 ^ n % 9 = 1 := by
  intro n
  induction n with
  | zero => rfl
  | succ k ih =>
    rw [Nat.pow_succ, Nat.mul_mod, ih]

-- Step 2: derive divisibility on the required range [1,200].
theorem nine_dvd_pow10_sub_one : ∀ n : Nat, 1 ≤ n → n ≤ 200 → 9 ∣ (10 ^ n - 1) := by
  intro n _ _
  have h1 : 10 ^ n % 9 = 1 := pow10_mod9 n
  have h2 : 9 * (10 ^ n / 9) + 10 ^ n % 9 = 10 ^ n := Nat.div_add_mod (10 ^ n) 9
  rw [h1] at h2
  exact ⟨10 ^ n / 9, by omega⟩

#check @pow10_mod9
#check @nine_dvd_pow10_sub_one

-- Sanity check: explicit endpoint witnesses n=1 and n=200 also type-check directly.
example : 9 ∣ (10 ^ 1 - 1) := nine_dvd_pow10_sub_one 1 (by decide) (by decide)
example : 9 ∣ (10 ^ 200 - 1) := nine_dvd_pow10_sub_one 200 (by decide) (by decide)

#print axioms nine_dvd_pow10_sub_one
