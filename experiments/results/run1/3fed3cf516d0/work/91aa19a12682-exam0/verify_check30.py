"""Direct arithmetic ground-truth check for the claim.

Claim under test: for all 1<=n<=300, 30 | n^3 - n.
(Note: the task's own "义务"/"绑定的方法" text instead targets n^5 - n,
a different and classically true expression — see report.md.)
"""

def counterexamples(exponent: int, lo: int = 1, hi: int = 300, divisor: int = 30):
    return [n for n in range(lo, hi + 1) if (n ** exponent - n) % divisor != 0]


if __name__ == "__main__":
    bad3 = counterexamples(3)
    bad5 = counterexamples(5)
    print(f"30 | n^3-n counterexamples in [1,300]: count={len(bad3)}, first10={bad3[:10]}")
    print(f"30 | n^5-n counterexamples in [1,300]: count={len(bad5)}, first10={bad5[:10]}")
    n = 2
    print(f"witness n={n}: n^3-n = {n**3-n}, 30 | it? {(n**3-n) % 30 == 0}")
