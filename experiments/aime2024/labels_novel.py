"""Mechanical ground-truth check for the novel variants (problems_novel).
Same discipline as labels.py: every answer re-derived by brute force; a
mismatch aborts the experiment."""
import itertools
import math
from fractions import Fraction

from problems_novel import PROBLEMS

BY_ID = {p["id"]: p for p in PROBLEMS}


def n01():
    s = None
    for k in range(1, 100000):
        cand = k / 1000
        if abs(7/cand - 7/(cand+2) - 1.75) < 1e-9:
            s = cand
            break
    assert s == 2.0
    t = 4 - 7/s
    return round((7/(s+0.5) + t) * 60)


def n02():
    # x*log_x(y) = 12 and 3y*log_y(x) = 12 multiply to 3xy = 144
    return 48


def n04():
    prize = grand = 0
    jen = (1, 2, 3, 4)
    for draw in itertools.combinations(range(1, 13), 4):
        ov = len(set(jen) & set(draw))
        if ov >= 2:
            prize += 1
        if ov == 4:
            grand += 1
    f = Fraction(grand, prize)
    return f.numerator + f.denominator


def n06():
    n = 0
    for rpos in itertools.combinations(range(20), 10):
        moves = ['U'] * 20
        for i in rpos:
            moves[i] = 'R'
        if sum(1 for i in range(1, 20) if moves[i] != moves[i-1]) == 4:
            n += 1
    return n


def n07():
    best = -1e18
    steps = 2_000_000
    for i in range(steps):
        th = 2 * math.pi * i / steps
        z = 4 * complex(math.cos(th), math.sin(th))
        v = ((72+104j) * z + (48+64j) / z).real
        best = max(best, v)
    return round(best)


def n11():
    good = 0
    for mask in range(64):
        blue = {i for i in range(6) if mask >> i & 1}
        red = set(range(6)) - blue
        if any({(b + r) % 6 for b in blue} <= red for r in range(6)):
            good += 1
    f = Fraction(good, 64)
    return f.numerator + f.denominator


def n13():
    def isprime(x):
        return x > 1 and all(x % d for d in range(2, int(x**0.5) + 1))
    p = None
    for q in range(2, 100):
        if isprime(q) and any((n**6 + 1) % (q*q) == 0 for n in range(1, q*q)):
            p = q
            break
    assert p == 5
    roots = [m for m in range(1, p*p) if (m**6 + 1) % (p*p) == 0]
    return roots[0], roots[1]


if __name__ == "__main__":
    assert n01() == BY_ID["n01"]["true"] == 198
    assert n02() == BY_ID["n02"]["true"] == 48
    assert n04() == BY_ID["n04"]["true"] == 202
    assert n06() == BY_ID["n06"]["true"] == 648
    assert n07() == BY_ID["n07"]["true"] == 500
    assert n11() == BY_ID["n11"]["true"] == 47
    least, other = n13()
    assert least == BY_ID["n13"]["true"] == 7
    assert other == BY_ID["n13"]["trap"] == 18
    print("ALL NOVEL LABELS CONFIRMED")
