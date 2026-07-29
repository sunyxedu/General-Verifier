-- Bounded trial-division primality predicate, no Mathlib dependency.
def isPrimeUpTo (n bound : Nat) : Bool :=
  (List.range' 2 (bound - 1)).all (fun d => n % d ≠ 0)

theorem isPrimeUpTo_8191_90 : isPrimeUpTo 8191 90 = true := by decide

#eval isPrimeUpTo 8191 90
#eval (List.range' 2 89)
#eval (List.range' 2 89).length
