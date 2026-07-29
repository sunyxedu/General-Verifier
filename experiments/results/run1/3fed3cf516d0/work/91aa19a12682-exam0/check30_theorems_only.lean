set_option maxRecDepth 4000

theorem dvd2 : ∀ n : Fin 300, 2 ∣ ((n.val + 1) ^ 5 - (n.val + 1)) := by decide
theorem dvd3 : ∀ n : Fin 300, 3 ∣ ((n.val + 1) ^ 5 - (n.val + 1)) := by decide
theorem dvd5 : ∀ n : Fin 300, 5 ∣ ((n.val + 1) ^ 5 - (n.val + 1)) := by decide
