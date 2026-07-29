-- Verification for exam0 obligation:
-- claim: for all 1<=n<=300, 120 | n^5 - n
-- obligation: the proof must give a correct algebraic factorization of n^5-n,
--   e.g. n^5 - n = n*(n-1)*(n+1)*(n^2+1), and must use it correctly in the
--   downstream divisibility argument.
-- bound method's own encoded check (as specified in the task) uses 60, not
-- 120 -- both are checked below, plus the claim's literal 120 statement.

-- (a) pointwise `decide` checks that the factorization identity has no
--     algebraic error, for several concrete n
example : (1:Nat)^5 - 1 = 1*(1-1)*(1+1)*(1^2+1) := by decide
example : (2:Nat)^5 - 2 = 2*(2-1)*(2+1)*(2^2+1) := by decide
example : (3:Nat)^5 - 3 = 3*(3-1)*(3+1)*(3^2+1) := by decide
example : (7:Nat)^5 - 7 = 7*(7-1)*(7+1)*(7^2+1) := by decide
example : (10:Nat)^5 - 10 = 10*(10-1)*(10+1)*(10^2+1) := by decide
example : (300:Nat)^5 - 300 = 300*(300-1)*(300+1)*(300^2+1) := by decide

-- (a') the factorization identity checked for every n in [1,300] via a
--      computable Bool predicate, confirmed by native_decide
def factCheck : Bool := (List.range 300).all (fun i => let n := i + 1;
  n^5 - n == n*(n-1)*(n+1)*(n^2+1))

#eval factCheck

example : factCheck = true := by native_decide

-- (b) the bound method's literal encoding: '∀ 1<=n<=300, 60 | n^5-n'
def check : Bool := (List.range 300).all (fun i => let n := i + 1;
  (n^5 - n) % 60 == 0)

#eval check

-- also the claim as literally stated in the task: '∀ 1<=n<=300, 120 | n^5-n'
def check120 : Bool := (List.range 300).all (fun i => let n := i + 1;
  (n^5 - n) % 120 == 0)

#eval check120

-- locate the first index (0-based, n = i+1) where 60 | n^5-n fails
def firstFail60 : Option Nat := (List.range 300).find? (fun i => let n := i + 1;
  (n^5 - n) % 60 != 0)

#eval firstFail60

-- locate the first index where 120 | n^5-n fails
def firstFail120 : Option Nat := (List.range 300).find? (fun i => let n := i + 1;
  (n^5 - n) % 120 != 0)

#eval firstFail120

-- explicit witness value at n = 2
#eval (2:Nat)^5 - 2
#eval ((2:Nat)^5 - 2) % 60
#eval ((2:Nat)^5 - 2) % 120

-- native_decide-checked, compiler-trusted refutations (these should succeed
-- as proofs of `= false`, confirming the checks fail, not `= true`)
example : check = false := by native_decide
example : check120 = false := by native_decide
