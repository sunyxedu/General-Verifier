-- Concrete counterexample to the claim "∀ 1≤n≤300, 60 ∣ n^5 - n": n = 2 gives n^5-n = 30, and 60 ∤ 30.
example : ¬ (60 ∣ (2^5 - 2 : Nat)) := by decide

-- Also record the true statement the bound method actually encodes (divisor 30, not 60):
set_option maxRecDepth 2000 in
theorem bound_method_actual_claim : ∀ n : Fin 300, 30 ∣ ((n.val+1)^5 - (n.val+1)) := by decide
