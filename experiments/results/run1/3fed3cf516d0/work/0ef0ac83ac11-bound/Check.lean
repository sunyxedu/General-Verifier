-- claim: for all 1 <= n <= 300, 30 | n^5 - n
-- obligation: cover all 300 integers exhaustively via kernel-checked `decide`, not sampling.
set_option maxRecDepth 2000 in
theorem thirty_dvd_pow5_sub_self : ∀ n : Fin 300, 30 ∣ ((n.val + 1)^5 - (n.val + 1)) := by
  decide

#print axioms thirty_dvd_pow5_sub_self
