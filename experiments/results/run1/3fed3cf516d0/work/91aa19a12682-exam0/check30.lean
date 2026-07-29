-- Bound method as specified in the task: verify 30 | n^5 - n via three
-- independent kernel `decide` checks (2, 3, 5), plus two meta-lemmas showing
-- that "6 | m" or "10 | m" alone do not imply the missing third prime factor.
set_option maxRecDepth 4000

theorem dvd2 : ∀ n : Fin 300, 2 ∣ ((n.val + 1) ^ 5 - (n.val + 1)) := by decide
theorem dvd3 : ∀ n : Fin 300, 3 ∣ ((n.val + 1) ^ 5 - (n.val + 1)) := by decide
theorem dvd5 : ∀ n : Fin 300, 5 ∣ ((n.val + 1) ^ 5 - (n.val + 1)) := by decide

example : ¬ ∀ m, 6 ∣ m → 5 ∣ m := ⟨6, by decide⟩
example : ¬ ∀ m, 10 ∣ m → 3 ∣ m := ⟨10, by decide⟩
