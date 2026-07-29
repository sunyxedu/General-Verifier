example : ¬ ∀ m, 6 ∣ m → 5 ∣ m := by
  intro h
  exact absurd (h 6 (by decide)) (by decide)

example : ¬ ∀ m, 10 ∣ m → 3 ∣ m := by
  intro h
  exact absurd (h 10 (by decide)) (by decide)
