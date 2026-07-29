"""
Verify claim: for all 1 <= n <= 1000, 12 | (n^3 - n).

Encoding: bounded enumeration. For each concrete n in {1,...,1000} we
emit an SMT-LIB2 (push ... (check-sat) ... pop) block asserting the
negation "(n^3 - n) mod 12 != 0" for that specific n. If the claim holds,
every one of the 1000 checks must return unsat (no counterexample at that
n). Any 'sat' result identifies a concrete counterexample n via the model.

The generated .smt2 script is fed to the z3 CLI binary (instrument:
z3 4.15.4, /opt/homebrew/bin/z3) and its full stdout is captured verbatim.
"""
import subprocess
import sys

SMT2_PATH = "bound_check.smt2"
OUTPUT_PATH = "bound_check_z3_output.txt"
Z3_BIN = "/opt/homebrew/bin/z3"

N_MIN, N_MAX = 1, 1000

lines = []
lines.append("(set-option :produce-models true)")
lines.append("(declare-const n Int)")
for k in range(N_MIN, N_MAX + 1):
    lines.append(f"(push)")
    lines.append(f"(assert (= n {k}))")
    lines.append(f"(assert (not (= (mod (- (* n n n) n) 12) 0)))")
    lines.append(f"(echo \"n={k}\")")
    lines.append(f"(check-sat)")
    lines.append(f"(pop)")

with open(SMT2_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")

proc = subprocess.run([Z3_BIN, SMT2_PATH], capture_output=True, text=True, timeout=300)

with open(OUTPUT_PATH, "w") as f:
    f.write("=== stdout ===\n")
    f.write(proc.stdout)
    f.write("\n=== stderr ===\n")
    f.write(proc.stderr)
    f.write(f"\n=== returncode ===\n{proc.returncode}\n")

out_lines = proc.stdout.splitlines()
results = {}
current_n = None
for line in out_lines:
    line = line.strip()
    if line.startswith("n="):
        current_n = int(line[2:])
    elif line in ("sat", "unsat", "unknown"):
        results[current_n] = line

missing = [k for k in range(N_MIN, N_MAX + 1) if k not in results]
sat_ns = [k for k, v in results.items() if v == "sat"]
unknown_ns = [k for k, v in results.items() if v == "unknown"]

print(f"returncode: {proc.returncode}")
print(f"total n checked: {len(results)} (expected {N_MAX - N_MIN + 1})")
print(f"missing n (no result parsed): {missing}")
print(f"sat (counterexample found) n: {sat_ns}")
print(f"unknown n: {unknown_ns}")

if proc.returncode != 0:
    print("FAILED: z3 process returned nonzero exit code")
    sys.exit(2)
if missing or unknown_ns:
    print("FAILED: incomplete coverage of range 1..1000")
    sys.exit(2)
if sat_ns:
    print(f"REFUTED: counterexamples found at n = {sat_ns}")
    sys.exit(1)
print("HOLDS: all 1000 checks returned unsat, no counterexample in 1..1000")
sys.exit(0)
