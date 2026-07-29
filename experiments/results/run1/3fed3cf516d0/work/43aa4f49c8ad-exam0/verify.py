"""
Claim: for all 1<=n<=300, 30 divides n^5+n.
Bound method: use sympy to
  1) factor n^5-n symbolically and check it equals n*(n-1)*(n+1)*(n**2+1)
  2) verify divisibility by 2,3,5 of that factorization via residue-class
     exhaustive check mod 30 (equivalent to full check over all integers
     since n^5-n mod 30 only depends on n mod 30)
  3) cross-check directly by brute force over 1<=n<=300 for n^5+n (the
     stated claim uses +n; note n^5+n and n^5-n differ by 2n, and since
     30 | 2n is NOT generally true, we must check the ACTUAL claim n^5+n,
     not silently substitute n^5-n)
"""
import sympy
from sympy import symbols, factor, gcd, Poly

n = symbols('n')

print("=== Step 1: symbolic factorization ===")
expr_minus = n**5 - n
expr_plus = n**5 + n

fact_minus = factor(expr_minus)
fact_plus = factor(expr_plus)
print(f"factor(n^5 - n) = {fact_minus}")
print(f"factor(n^5 + n) = {fact_plus}")

expected_minus = n*(n-1)*(n+1)*(n**2+1)
diff = sympy.expand(fact_minus - expected_minus)
print(f"factor(n^5-n) - n(n-1)(n+1)(n^2+1) expanded = {diff}  (should be 0)")

print()
print("=== Step 2: residue-class check for n^5-n mod 30 (all r in 0..29) ===")
all_div_minus = True
bad_minus = []
for r in range(30):
    val = r**5 - r
    if val % 30 != 0:
        all_div_minus = False
        bad_minus.append((r, val, val % 30))
print(f"n^5-n divisible by 30 for all residues mod 30: {all_div_minus}")
if bad_minus:
    print(f"counterexamples (residue, value, value mod 30): {bad_minus}")

print()
print("=== Step 3: residue-class check for n^5+n mod 30 (all r in 0..29) ===")
all_div_plus = True
bad_plus = []
for r in range(30):
    val = r**5 + r
    if val % 30 != 0:
        all_div_plus = False
        bad_plus.append((r, val, val % 30))
print(f"n^5+n divisible by 30 for all residues mod 30: {all_div_plus}")
if bad_plus:
    print(f"first few counterexamples (residue, value, value mod 30): {bad_plus[:10]}")
    print(f"total bad residues out of 30: {len(bad_plus)}")

print()
print("=== Step 4: direct brute force for the ACTUAL claim, n^5+n, over 1<=n<=300 ===")
fail_list = []
for k in range(1, 301):
    v = k**5 + k
    if v % 30 != 0:
        fail_list.append((k, v, v % 30))

total = 300
n_fail = len(fail_list)
print(f"Total n checked: {total}")
print(f"Number of n in [1,300] where 30 does NOT divide n^5+n: {n_fail}")
if fail_list:
    print(f"First 10 counterexamples (n, n^5+n, (n^5+n) mod 30): {fail_list[:10]}")
    print(f"Smallest counterexample: n = {fail_list[0][0]}, n^5+n = {fail_list[0][1]}, mod 30 = {fail_list[0][2]}")

print()
print("=== Step 5: sympy symbolic gcd/Poly cross-check on n^5+n mod small primes 2,3,5 ===")
for p in [2, 3, 5]:
    bad_p = [r for r in range(p) if (r**5 + r) % p != 0]
    print(f"mod {p}: residues 0..{p-1} where p does not divide r^5+r: {bad_p}")

print()
print("=== CONCLUSION ===")
if n_fail == 0:
    print("CLAIM HOLDS: 30 | n^5+n for all 1<=n<=300")
else:
    print("CLAIM REFUTED: found counterexamples where 30 does NOT divide n^5+n")
    print(f"Smallest witness: n={fail_list[0][0]}, n^5+n={fail_list[0][1]}, "
          f"(n^5+n) mod 30 = {fail_list[0][2]}")
