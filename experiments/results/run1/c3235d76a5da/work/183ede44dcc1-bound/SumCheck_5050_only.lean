def sumTo (n : Nat) : Nat := (List.range (n+1)).foldl (·+·) 0

#eval sumTo 100

theorem sumTo_100_eq_5050 : sumTo 100 = 5050 := by decide
