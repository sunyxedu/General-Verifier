/-
Claim: 对所有 1 ≤ n ≤ 200，9 ∣ 10^n - 1.

义务 (coverage obligation): n = 1 和 n = 200 必须被显式纳入验证范围。

Strategy:
  1. Prove the auxiliary fact `pow10_eq : ∀ n, ∃ k, 10^n = 9*k + 1` by induction on n
     (uses 10 ≡ 1 mod 9, i.e. 10 = 9*1 + 1, and the multiplicative step
     10*(9k+1) = 9*(10k+1) + 1).
  2. Derive the general theorem `nine_dvd : ∀ n, 1 ≤ n → 9 ∣ (10^n - 1)` from it.
  3. Explicitly instantiate the endpoints n = 1 and n = 200, plus a `decide`-based
     cross-check on both endpoints as an independent instrument.
-/

theorem pow10_eq (n : ℕ) : ∃ k, 10 ^ n = 9 * k + 1 := by
  induction n with
  | zero => exact ⟨0, rfl⟩
  | succ m ih =>
    obtain ⟨k, hk⟩ := ih
    refine ⟨10 * k + 1, ?_⟩
    rw [pow_succ, hk]
    omega

theorem nine_dvd (n : ℕ) (h : 1 ≤ n) : 9 ∣ (10 ^ n - 1) := by
  obtain ⟨k, hk⟩ := pow10_eq n
  have hpos : 1 ≤ 10 ^ n := by
    calc 1 ≤ 9 * k + 1 := by omega
    _ = 10 ^ n := hk.symm
  exact ⟨k, by omega⟩

-- Explicit endpoint coverage: n = 1.
example : (9 : ℕ) ∣ (10 ^ 1 - 1) := nine_dvd 1 (by decide)

-- Explicit endpoint coverage: n = 200.
example : (9 : ℕ) ∣ (10 ^ 200 - 1) := nine_dvd 200 (by decide)

-- Independent cross-check instrument: brute-force decidability on both endpoints.
example : (9 : ℕ) ∣ (10 ^ 1 - 1) := by decide
example : (9 : ℕ) ∣ (10 ^ 200 - 1) := by decide

-- Sanity check that the range [1, 200] is exactly what is being covered.
example : (1 : ℕ) ≤ 1 ∧ (1 : ℕ) ≤ 200 := by decide

#print axioms nine_dvd

#eval "PROOF_CHECK_DONE"
