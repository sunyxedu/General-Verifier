set_option maxRecDepth 4000

theorem dvd2 : ∀ n : Fin 300, 2 ∣ ((n.val + 1)^5 - (n.val + 1)) := by decide
theorem dvd3 : ∀ n : Fin 300, 3 ∣ ((n.val + 1)^5 - (n.val + 1)) := by decide
theorem dvd5 : ∀ n : Fin 300, 5 ∣ ((n.val + 1)^5 - (n.val + 1)) := by decide

example : ¬ ∀ m : Nat, 6 ∣ m → 5 ∣ m := by
  intro h
  have := h 6 (by decide)
  contradiction

example : ¬ ∀ m : Nat, 10 ∣ m → 3 ∣ m := by
  intro h
  have := h 10 (by decide)
  contradiction

#print axioms dvd2
#print axioms dvd3
#print axioms dvd5
