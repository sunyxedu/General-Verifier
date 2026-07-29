"""
Generate a Lean 4 file with `decide`-checked examples for n = 1..200, for BOTH
  (a) the statement in 义务/方法: 11 | 10^n - 1
  (b) the statement in the original claim text: 11 | 10^n + 1
This lets Lean's kernel cross-check, per n, whether divisibility holds or not,
for both readings, since the task text has a mismatch between claim (+1) and
义务/方法 (-1). We don't want to silently pick one; we check both and report.
For each n, the *correct* direction (dvd or ~dvd) is computed here in Python
first, and the Lean example asserts that direction. If our Python arithmetic
were wrong, `decide` would fail to compile (kernel checked), so a clean
compile is a real cross-check, not just trust in the Python computation.
"""

N = 200
lines = []
lines.append("-- Auto-generated. Each example is checked by Lean's kernel via `decide`.")
lines.append("")

fails_minus1 = []  # n where 11 does NOT divide 10^n - 1   (义务/方法 statement)
fails_plus1 = []   # n where 11 does NOT divide 10^n + 1   (claim statement)

for n in range(1, N + 1):
    minus_val = 10**n - 1
    plus_val = 10**n + 1
    minus_ok = (minus_val % 11 == 0)
    plus_ok = (plus_val % 11 == 0)
    if not minus_ok:
        fails_minus1.append(n)
    if not plus_ok:
        fails_plus1.append(n)

    if minus_ok:
        lines.append(f"example : (11 ∣ (10^{n} - 1)) := by decide")
    else:
        lines.append(f"example : ¬ (11 ∣ (10^{n} - 1)) := by decide")

    if plus_ok:
        lines.append(f"example : (11 ∣ (10^{n} + 1)) := by decide")
    else:
        lines.append(f"example : ¬ (11 ∣ (10^{n} + 1)) := by decide")

lines.append("")
lines.append("#eval (List.range 200).map (· + 1) |>.filter (fun n => ¬ decide (11 ∣ 10^n - 1))")
lines.append("#eval (List.range 200).map (· + 1) |>.filter (fun n => ¬ decide (11 ∣ 10^n + 1))")

with open("Check.lean", "w") as f:
    f.write("\n".join(lines) + "\n")

print("fails for 11 | 10^n - 1 (n where it FAILS, i.e. 11 does not divide):", fails_minus1[:10], "... total", len(fails_minus1))
print("fails for 11 | 10^n + 1 (n where it FAILS, i.e. 11 does not divide):", fails_plus1[:10], "... total", len(fails_plus1))
