-- Claim: for all 1 ≤ n ≤ 200, 11 ∣ 10^n - 1
-- Bare Lean 4 (no Mathlib). We use `decide` on the decidable proposition
-- `11 ∣ 10^n - 1` (Nat subtraction, Nat divisibility) for each concrete n,
-- and let the kernel check every reduction step.

-- Enumerate all n in [1,200] for which the claim FAILS, via kernel-checked `decide`.
#eval (List.range 200).map (fun i => i + 1) |>.filter (fun n => ¬ decide (11 ∣ (10^n - 1)))

-- Cross-check with concrete `decide`-based proofs for the first few odd/even n.
example : ¬ (11 ∣ (10^1 - 1)) := by decide   -- n=1: 10^1-1 = 9, not divisible by 11
example : (11 ∣ (10^2 - 1)) := by decide     -- n=2: 10^2-1 = 99 = 11*9
example : ¬ (11 ∣ (10^3 - 1)) := by decide   -- n=3: 999, not divisible by 11
example : (11 ∣ (10^4 - 1)) := by decide     -- n=4: 9999 = 11*909

-- Full witness: n=1 is a counterexample to the universal claim.
theorem counterexample_n1 : ¬ (11 ∣ (10^1 - 1)) := by decide

-- Sanity: exactly the odd n in [1,200] should fail (11 | 10^n - 1 holds iff n even).
def failing : List Nat := (List.range 200).map (fun i => i + 1) |>.filter (fun n => ¬ decide (11 ∣ (10^n - 1)))
def oddsUpTo200 : List Nat := (List.range 200).map (fun i => i + 1) |>.filter (fun n => n % 2 = 1)

#eval failing.length          -- expect 100
#eval oddsUpTo200.length      -- expect 100
#eval decide (failing = oddsUpTo200)  -- expect true

-- Report first and last few failing n for readability.
#eval failing.take 5
#eval failing.reverse.take 5
