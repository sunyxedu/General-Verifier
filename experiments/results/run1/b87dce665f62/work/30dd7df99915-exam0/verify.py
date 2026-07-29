"""
Verify claim: C(10,3) == 720

Method: symbolic verification with sympy.
- Use sympy's built-in binomial(n,k) function.
- Independently construct the combination formula by hand:
      C(n,k) = n! / (k! * (n-k)!)
  using sympy.factorial, NOT the permutation formula P(n,k) = n!/(n-k)!.
- Substitute n=10, k=3 into both expressions, simplify, and assert equality.
- Compare the resulting value against the claimed value 720.
"""

import sys
from sympy import symbols, factorial, binomial, simplify, Eq, Rational

n, k = symbols('n k', positive=True, integer=True)

# Correct combination formula (definition), built by hand
comb_formula = factorial(n) / (factorial(k) * factorial(n - k))

# Sympy's built-in binomial coefficient function
comb_builtin = binomial(n, k)

# For contrast/reference only: permutation formula (must NOT be used for the combination claim)
perm_formula = factorial(n) / factorial(n - k)

n_val, k_val = 10, 3
claimed_value = 720

comb_formula_val = simplify(comb_formula.subs({n: n_val, k: k_val}))
comb_builtin_val = simplify(comb_builtin.subs({n: n_val, k: k_val}))
perm_formula_val = simplify(perm_formula.subs({n: n_val, k: k_val}))

print(f"n = {n_val}, k = {k_val}")
print(f"Hand-derived combination formula C(n,k) = n!/(k!(n-k)!) evaluated: {comb_formula_val}")
print(f"Sympy built-in binomial(n,k) evaluated: {comb_builtin_val}")
print(f"(For reference only) permutation formula P(n,k) = n!/(n-k)! evaluated: {perm_formula_val}")
print(f"Claimed value in the claim under test: {claimed_value}")

# Assert the two independent derivations of the combination number agree with each other
definitions_agree = bool(Eq(comb_formula_val, comb_builtin_val))
print(f"\nAssertion: hand-derived formula == sympy built-in binomial -> {definitions_agree}")
assert definitions_agree, "Combination formula and sympy binomial() disagree - internal inconsistency"

# Now check whether that mutually-agreed combination value equals the claimed value (720)
matches_claim = bool(Eq(comb_formula_val, claimed_value))
print(f"Assertion: combination value ({comb_formula_val}) == claimed value ({claimed_value}) -> {matches_claim}")

# Sanity check: is the claimed value actually the permutation-formula result instead?
matches_permutation = bool(Eq(perm_formula_val, claimed_value))
print(f"Sanity check: claimed value ({claimed_value}) == permutation P(10,3) result ({perm_formula_val}) -> {matches_permutation}")

print("\n=== FINAL RESULT ===")
if matches_claim:
    print(f"CLAIM HOLDS: C({n_val},{k_val}) = {comb_formula_val} = {claimed_value} as claimed.")
    sys.exit(0)
else:
    print(f"CLAIM REFUTED: C({n_val},{k_val}) = {comb_formula_val}, which does NOT equal the claimed value {claimed_value}.")
    if matches_permutation:
        print(f"Note: {claimed_value} matches the PERMUTATION formula P({n_val},{k_val}) = {perm_formula_val} instead, "
              f"suggesting the claim likely confused C(n,k) with P(n,k).")
    sys.exit(1)
