def sumTo (n : Nat) : Nat := (List.range (n+1)).foldl (·+·) 0

#eval sumTo 100

theorem sumTo100_eq_5050 : sumTo 100 = 5050 := by decide
theorem sumTo100_eq_5051 : sumTo 100 = 5051 := by decide
