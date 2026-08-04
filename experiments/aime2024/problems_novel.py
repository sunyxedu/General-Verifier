"""Novel variants of the seven AIME 2024 I problems: same shapes, new
constants, so answers cannot be recalled from training data — the judge
must actually reason, and the verifier must actually check. Every answer
below is derived by brute force in labels_novel.py before any run.

The p13 variant collapses in difficulty (the least prime becomes 5), but
its minimality trap (18 is a genuine root of m^6 = -1 mod 25, just not the
least) still tests exactly the failure mode it was built for.
"""

PROBLEMS = [
    {
        "id": "n01",
        "statement": (
            "Every morning Maya goes for a 7-kilometer-long walk and stops at "
            "a coffee shop afterwards. When she walks at a constant speed of "
            "s kilometers per hour, the walk takes her 4 hours, including t "
            "minutes spent in the coffee shop. When she walks s+2 kilometers "
            "per hour, the walk takes her 2 hours and 15 minutes, including "
            "t minutes spent in the coffee shop. Suppose Maya walks at s+1/2 "
            "kilometers per hour. Find the number of minutes the walk takes "
            "her, including the t minutes spent in the coffee shop."
        ),
        "true": 198,
        "trap": 168,   # forgetting the coffee-shop minutes
    },
    {
        "id": "n02",
        "statement": (
            "There exist real numbers x and y, both greater than 1, such "
            "that log_x(y^x) = log_y(x^(3y)) = 12. Find xy."
        ),
        "true": 48,
        "trap": 144,   # dropping the factor 3 (3xy=144 -> xy=48)
    },
    {
        "id": "n04",
        "statement": (
            "Jen enters a lottery by picking 4 distinct numbers from "
            "S = {1, 2, ..., 12}. Four numbers are randomly chosen from S. "
            "She wins a prize if at least two of her numbers were among the "
            "randomly chosen numbers, and wins the grand prize if all four "
            "of her numbers were the randomly chosen numbers. The "
            "probability of her winning the grand prize given that she won "
            "a prize is m/n where m and n are relatively prime positive "
            "integers. Find m+n."
        ),
        "true": 202,   # 1/201
        "trap": 496,   # unconditioned 1/495 -> 1+495
    },
    {
        "id": "n06",
        "statement": (
            "Consider the paths of length 20 that follow the lines from the "
            "lower left corner to the upper right corner on a 10x10 grid. "
            "Find the number of such paths that change direction exactly "
            "four times."
        ),
        "true": 648,
        "trap": 324,   # counting only one of the two starting directions
    },
    {
        "id": "n07",
        "statement": (
            "Find the largest possible real part of (72+104i)z + (48+64i)/z "
            "where z is a complex number with |z| = 4."
        ),
        "true": 500,
        "trap": 400,   # |sin coefficient| instead of the amplitude
    },
    {
        "id": "n11",
        "statement": (
            "Each vertex of a regular hexagon is independently colored "
            "either red or blue with equal probability. The probability that "
            "the hexagon can then be rotated so that all of the blue "
            "vertices end up at positions where there were originally red "
            "vertices is m/n, where m and n are relatively prime positive "
            "integers. Find m+n."
        ),
        "true": 47,    # 15/32
        "trap": 45,    # a near-miss numerator slip (13/32)
    },
    {
        "id": "n13",
        "statement": (
            "Let p be the least prime number for which there exists a "
            "positive integer n such that n^6 + 1 is divisible by p^2. Find "
            "the least positive integer m such that m^6 + 1 is divisible by "
            "p^2."
        ),
        "true": 7,
        "trap": 18,    # a genuine root of m^6 = -1 mod 25, but not the least
    },
]
