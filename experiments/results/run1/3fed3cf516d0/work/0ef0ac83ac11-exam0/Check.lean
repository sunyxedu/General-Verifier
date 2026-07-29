-- Claim under test: ∀ 1 ≤ n ≤ 300, 60 ∣ n^5 - n
-- Bound method's actual Lean proposition (as given in the task) uses divisor 30, NOT 60:
--   ∀ n : Fin 300, 30 ∣ ((n.val+1)^5 - (n.val+1))
-- This is a mismatch: the bound method does not encode the claim being checked.
-- We first test the claim exactly as stated (divisor 60).

set_option maxRecDepth 4000 in
theorem claim_60 : ∀ n : Fin 300, 60 ∣ ((n.val+1)^5 - (n.val+1)) := by decide

