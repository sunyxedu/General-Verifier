-- s realizes the closed interval [1,100]: (List.range 100).map (·+1) = [1,2,...,100]
def s : List Nat := (List.range 100).map (· + 1)

-- Coverage obligation: exactly 100 terms, first term 1, last term 100 (closed interval)
theorem coverage : s.length = 100 ∧ s.head? = some 1 ∧ s.getLast? = some 100 := by decide

-- Claim under test: sum of 1..100 equals 5100 (this is FALSE, decide will refute it below)
theorem sum_1_to_100_eq_5100 : s.sum = 5100 := by decide

-- Witness: what the sum actually is
#eval s.sum

-- Confirming the correct value compiles (Gauss: n(n+1)/2 = 100*101/2 = 5050)
theorem sum_1_to_100_eq_5050 : s.sum = 5050 := by decide

#print axioms coverage
#print axioms sum_1_to_100_eq_5050
