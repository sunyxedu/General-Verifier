"""
Claim: for all 1 <= n <= 300, 30 | n^5 - n.
Bound method: use sympy to
  1) symbolically factor n**5 - n and confirm it equals n*(n-1)*(n+1)*(n**2+1)
  2) verify mod-2, mod-3, mod-5 divisibility of the factorization symbolically
     (i.e. that n*(n-1)*(n+1)*(n**2+1) is always divisible by 2, 3, 5)
  3) do a full residue-class sweep mod 30 (equivalent to checking all n mod 30)
  4) do a direct brute-force check over 1..300 as the concrete instance range
     required by the claim.
"""
import sympy
from sympy import symbols, factor, Poly, gcd, GF, expand, Integer

n = symbols('n')

report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)

log("=== Step 1: symbolic factorization ===")
expr = n**5 - n
factored = factor(expr)
log(f"n**5 - n factored by sympy = {factored}")

candidate = n*(n-1)*(n+1)*(n**2+1)
candidate_expanded = expand(candidate)
expr_expanded = expand(expr)
match = sympy.simplify(candidate_expanded - expr_expanded) == 0
log(f"n*(n-1)*(n+1)*(n**2+1) expanded = {candidate_expanded}")
log(f"n**5 - n expanded            = {expr_expanded}")
log(f"Factorization matches expected form n(n-1)(n+1)(n^2+1): {match}")

step1_ok = match

log("")
log("=== Step 2: symbolic mod-2/3/5 divisibility checks over all residues ===")
# For each modulus m in {2,3,5}, check every residue class r in [0, m-1]
# that n(n-1)(n+1)(n^2+1) mod m == 0 when n = r (mod m), using explicit
# substitution over a full residue system (this is a complete case analysis
# since polynomial value mod m only depends on n mod m).
mod_results = {}
for m in (2, 3, 5):
    all_zero = True
    per_residue = []
    for r in range(m):
        val = candidate.subs(n, r)
        val_mod = sympy.Mod(val, m)
        per_residue.append((r, int(val), int(val_mod)))
        if val_mod != 0:
            all_zero = False
    mod_results[m] = (all_zero, per_residue)
    log(f"modulus {m}: residues checked = {per_residue}, all divisible = {all_zero}")

step2_ok = all(mod_results[m][0] for m in (2, 3, 5))

log("")
log("=== Step 3: full residue sweep mod 30 (n = 0..29) ===")
residue_fail = []
for r in range(30):
    val = expand(expr.subs(n, r))
    if int(val) % 30 != 0:
        residue_fail.append((r, int(val)))
step3_ok = len(residue_fail) == 0
log(f"Residues 0..29 checked. Failures: {residue_fail if residue_fail else 'none'}")
log(f"All 30 residue classes give n^5-n divisible by 30: {step3_ok}")

log("")
log("=== Step 4: brute-force direct check for n in 1..300 ===")
brute_fail = []
for k in range(1, 301):
    val = k**5 - k
    if val % 30 != 0:
        brute_fail.append((k, val))
step4_ok = len(brute_fail) == 0
log(f"Checked n=1..300. Failures: {brute_fail if brute_fail else 'none'}")
log(f"All n in [1,300] satisfy 30 | n^5-n: {step4_ok}")

log("")
log("=== Overall ===")
overall = step1_ok and step2_ok and step3_ok and step4_ok
log(f"step1_ok={step1_ok} step2_ok={step2_ok} step3_ok={step3_ok} step4_ok={step4_ok}")
log(f"OVERALL_RESULT={'HOLDS' if overall else 'REFUTED'}")

with open('verify_output.txt', 'w') as f:
    f.write('\n'.join(report_lines) + '\n')
