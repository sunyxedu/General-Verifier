"""
Verify claim: for all 1<=n<=200, 11 | (10^n - 1).
Also cross-check the bound method's actual target: 9 | (10^n - 1).
"""

def check_divisor(d, lo=1, hi=200):
    failures = []
    for n in range(lo, hi + 1):
        val = 10 ** n - 1
        if val % d != 0:
            failures.append((n, val % d))
    return failures


def main():
    print("=== Claim under test: for all 1<=n<=200, 11 | (10^n - 1) ===")
    fails_11 = check_divisor(11)
    if not fails_11:
        print("PASS: 11 divides 10^n - 1 for all n in [1,200]")
    else:
        print(f"FAIL: {len(fails_11)} counterexamples found out of 200")
        print("First 5 counterexamples (n, (10^n-1) mod 11):", fails_11[:5])

    print()
    print("=== Sanity check on the bound method's actual target: 9 | (10^n - 1) ===")
    fails_9 = check_divisor(9)
    if not fails_9:
        print("PASS: 9 divides 10^n - 1 for all n in [1,200] (this is TRUE, but proves nothing about 11)")
    else:
        print(f"FAIL: {len(fails_9)} counterexamples found for divisor 9")

    print()
    print("=== Modular pattern check: 10^n mod 11 ===")
    for n in range(1, 11):
        print(f"n={n}: 10^n mod 11 = {pow(10, n, 11)}, (10^n - 1) mod 11 = {(10**n - 1) % 11}")

    print()
    print("=== Verdict ===")
    if fails_11:
        n0, r0 = fails_11[0]
        print(f"Claim is REFUTED. Smallest counterexample: n={n0}, 10^{n0}-1 = {10**n0 - 1}, "
              f"which mod 11 = {r0} (not divisible by 11).")
        print("Additionally, the bound method proves 9 | (10^n - 1), which is a TRUE but DIFFERENT "
              "statement (wrong modulus: 9 instead of 11). It does not establish the claim about 11.")
    else:
        print("Claim HOLDS for all tested n.")


if __name__ == "__main__":
    main()
