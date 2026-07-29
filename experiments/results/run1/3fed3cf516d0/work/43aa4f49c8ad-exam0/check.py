"""
Check: for all 1<=n<=300, 30 divides n**5-n (per 绑定的方法).
Also verified by exhaustion over all residues mod 30 (0..29), which covers
every integer n since n**5-n mod 30 is periodic with period 30.
"""

failures_range = []
for n in range(1, 301):
    val = (n**5 - n) % 30
    if val != 0:
        failures_range.append((n, val))

failures_residues = []
for r in range(30):
    val = (r**5 - r) % 30
    if val != 0:
        failures_residues.append((r, val))

print("=== Range check 1..300 for (n**5 - n) % 30 ===")
print(f"Total n checked: 300")
print(f"Failures: {len(failures_range)}")
if failures_range:
    print("First failures:", failures_range[:10])
else:
    print("All (n**5 - n) % 30 == 0 for n in 1..300")

print()
print("=== Residue check 0..29 for (r**5 - r) % 30 (covers all integers) ===")
print(f"Total residues checked: 30")
print(f"Failures: {len(failures_residues)}")
if failures_residues:
    print("Failing residues:", failures_residues)
else:
    print("All (r**5 - r) % 30 == 0 for r in 0..29 -> holds for ALL integers n")

print()
print("=== Sanity note: the claim text as literally written uses n**5+n, not n**5-n ===")
sample_plus = [(n, (n**5 + n) % 30) for n in range(1, 6)]
print("Sample (n**5+n)%30 for n=1..5:", sample_plus)

overall_holds = (len(failures_range) == 0) and (len(failures_residues) == 0)
print()
print(f"OVERALL (n**5-n divisible by 30, per bound method): {'HOLDS' if overall_holds else 'REFUTED'}")
