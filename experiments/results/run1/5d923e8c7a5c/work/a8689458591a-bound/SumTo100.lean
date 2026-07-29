import Mathlib

/-
Claim: 1 到 100 的所有整数之和等于 5050.
义务 (coverage obligation): 求和范围必须恰好覆盖 1..100 的全部整数
  —— 恰好 100 项、首项为 1、末项为 100、闭区间 (inclusive)。

We encode the range as `s := Finset.Icc 1 100` (ℕ closed interval), first prove the
coverage obligation as an explicit conjunction (card = 100, min' = 1, max' = 100),
and only then prove the sum equals 5050.
-/

open Finset

def s : Finset ℕ := Finset.Icc 1 100

theorem s_nonempty : s.Nonempty := by
  refine ⟨1, ?_⟩
  simp [s]

-- Coverage obligation: exactly 100 terms, first term 1, last term 100, closed interval.
theorem coverage :
    s.card = 100 ∧ s.min' s_nonempty = 1 ∧ s.max' s_nonempty = 100 := by
  refine ⟨?_, ?_, ?_⟩
  · simp [s, Nat.card_Icc]
  · apply le_antisymm
    · exact Finset.min'_le s 1 (by simp [s])
    · exact Finset.le_min' s s_nonempty 1 (fun x hx => (Finset.mem_Icc.mp (by simpa [s] using hx)).1)
  · apply le_antisymm
    · exact Finset.max'_le s s_nonempty 100 (fun x hx => (Finset.mem_Icc.mp (by simpa [s] using hx)).2)
    · exact Finset.le_max' s 100 (by simp [s])

-- Main claim: the sum over the covering range equals 5050, via the Gauss formula.
theorem sum_eq_5050 : ∑ i ∈ s, i = 5050 := by
  have hins : Finset.range 101 = insert 0 s := by
    ext x
    simp only [Finset.mem_range, Finset.mem_insert, s, Finset.mem_Icc]
    omega
  have h0 : (0 : ℕ) ∉ s := by simp [s]
  have hsum : ∑ i ∈ Finset.range 101, i = ∑ i ∈ s, i := by
    rw [hins, Finset.sum_insert h0, zero_add]
  have h101 : ∑ i ∈ Finset.range 101, i = 5050 := by
    rw [Finset.sum_range_id]
  rw [← hsum, h101]

-- Cross-check the sum by full computation as well, per the method's alternative instrument.
theorem sum_eq_5050_native : ∑ i ∈ s, i = 5050 := by
  native_decide

-- The full statement demanded by the bound method: coverage obligation ∧ the sum claim.
theorem claim_holds :
    s.card = 100 ∧ s.min' s_nonempty = 1 ∧ s.max' s_nonempty = 100 ∧
    ∑ i ∈ s, i = 5050 :=
  ⟨coverage.1, coverage.2.1, coverage.2.2, sum_eq_5050⟩

#print axioms coverage
#print axioms sum_eq_5050
#print axioms claim_holds

#eval "PROOF_CHECK_DONE"
