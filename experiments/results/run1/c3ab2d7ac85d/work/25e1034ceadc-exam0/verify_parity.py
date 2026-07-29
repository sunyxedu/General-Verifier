#!/usr/bin/env python3
"""
Check: for all 1 <= n <= 1000 (and a larger redundant range),
2 divides n^3 - n, with explicit grouping by parity of n.
"""

def check(N):
    even_remainders = []
    odd_remainders = []
    counterexamples = []

    for n in range(1, N + 1):
        val = n**3 - n
        rem = val % 2
        if n % 2 == 0:
            even_remainders.append(rem)
        else:
            odd_remainders.append(rem)
        if rem != 0:
            counterexamples.append((n, val, rem))

    max_even_rem = max(even_remainders) if even_remainders else None
    max_odd_rem = max(odd_remainders) if odd_remainders else None

    print(f"Range checked: n = 1..{N}")
    print(f"Even-n group: count={len(even_remainders)}, max remainder={max_even_rem}")
    print(f"Odd-n group:  count={len(odd_remainders)}, max remainder={max_odd_rem}")
    print(f"Counterexamples found: {len(counterexamples)}")
    if counterexamples:
        print("List of counterexamples (n, n^3-n, remainder):")
        for ce in counterexamples:
            print(f"  {ce}")
    else:
        print("No counterexamples.")
    print()
    return len(counterexamples) == 0


if __name__ == "__main__":
    ok1 = check(1000)
    print("=" * 60)
    ok2 = check(1_000_000)  # redundant larger-range check
    print("=" * 60)
    overall_ok = ok1 and ok2
    print(f"OVERALL RESULT: {'HOLDS' if overall_ok else 'REFUTED'}")
