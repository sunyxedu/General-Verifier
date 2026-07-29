set_option maxRecDepth 4000

-- ok(n): the claimed factorization n*(n-1)*(n+1)*(n^2+1) must equal n^5-n
-- (algebraic identity check), AND n^5-n must be divisible by 30, both by
-- kernel-checked `decide`, per the bound method text.
def ok (n : Nat) : Bool :=
  decide (n * (n - 1) * (n + 1) * (n ^ 2 + 1) = n ^ 5 - n) && ((n ^ 5 - n) % 30 == 0)

theorem all_ok : ((List.range 300).map (· + 1)).all ok = true := by decide

#print axioms all_ok
