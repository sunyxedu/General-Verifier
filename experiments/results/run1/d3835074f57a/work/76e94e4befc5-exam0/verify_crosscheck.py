#!/usr/bin/env python3
"""
Independent cross-check: parse z3_output.txt's Part B results and
compare, point by point for every n in 1..1000, against a direct
arithmetic computation of (n**3 - n) % 24. This is not itself the
proof (z3 is); it is an audit that the z3 run's per-n verdicts were
parsed and reported faithfully, with no n skipped or misread.
"""

with open("z3_output.txt") as f:
    txt = f.read()

blocks = txt.split("PART_B n=")[1:]
assert len(blocks) == 1000, f"expected 1000 Part B blocks, got {len(blocks)}"

mismatches = []
sat_ns = []
for b in blocks:
    header, rest = b.split(":", 1)
    n = int(header)
    result = rest.strip().splitlines()[0].strip()
    direct_is_counterexample = (n**3 - n) % 24 != 0
    z3_is_counterexample = (result == "sat")
    if direct_is_counterexample != z3_is_counterexample:
        mismatches.append((n, result, direct_is_counterexample))
    if z3_is_counterexample:
        sat_ns.append(n)

print(f"checked n=1..1000, blocks parsed: {len(blocks)}")
print(f"mismatches between z3 and direct arithmetic: {len(mismatches)}")
if mismatches:
    print("MISMATCH DETAILS:", mismatches)
print(f"counterexample count (z3, mod 24): {len(sat_ns)}")
print(f"first counterexample n: {sat_ns[0]}")
print(f"n=2 check: 2**3-2 = {2**3-2}, mod 24 = {(2**3-2)%24}")
