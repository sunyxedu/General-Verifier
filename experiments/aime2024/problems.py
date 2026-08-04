"""Six AIME 2024 I problems with mechanically confirmed labels (labels.py).

`true` is the official answer, re-derived by brute force before any run.
`trap` is a HIGH-QUALITY wrong answer: the value a plausible mistake
produces, not a random number — e.g. p13's trap 134 genuinely satisfies
17² | m⁴+1 (it is one of the four roots mod 289) but is not the LEAST such
m, so it punishes any verifier that checks divisibility and skips
minimality.
"""

PROBLEMS = [
    {
        "id": "p01",
        "statement": (
            "Every morning Aya goes for a 9-kilometer-long walk and stops at a "
            "coffee shop afterwards. When she walks at a constant speed of s "
            "kilometers per hour, the walk takes her 4 hours, including t "
            "minutes spent in the coffee shop. When she walks s+2 kilometers "
            "per hour, the walk takes her 2 hours and 24 minutes, including t "
            "minutes spent in the coffee shop. Suppose Aya walks at s+1/2 "
            "kilometers per hour. Find the number of minutes the walk takes "
            "her, including the t minutes spent in the coffee shop."
        ),
        "true": 204,
        "trap": 180,   # forgetting the coffee-shop minutes
    },
    {
        "id": "p02",
        "statement": (
            "There exist real numbers x and y, both greater than 1, such that "
            "log_x(y^x) = log_y(x^(4y)) = 10. Find xy."
        ),
        "true": 25,
        "trap": 100,   # dropping the factor 4 (4xy=100 -> xy=25)
    },
    {
        "id": "p04",
        "statement": (
            "Jen enters a lottery by picking 4 distinct numbers from "
            "S = {1, 2, ..., 10}. Four numbers are randomly chosen from S. She "
            "wins a prize if at least two of her numbers were among the "
            "randomly chosen numbers, and wins the grand prize if all four of "
            "her numbers were the randomly chosen numbers. The probability of "
            "her winning the grand prize given that she won a prize is m/n "
            "where m and n are relatively prime positive integers. Find m+n."
        ),
        "true": 116,
        "trap": 211,   # unconditioned 1/210 -> 1+210
    },
    {
        "id": "p06",
        "statement": (
            "Consider the paths of length 16 that follow the lines from the "
            "lower left corner to the upper right corner on an 8x8 grid. Find "
            "the number of such paths that change direction exactly four "
            "times."
        ),
        "true": 294,
        "trap": 147,   # counting only one of the two starting directions
    },
    {
        "id": "p07",
        "statement": (
            "Find the largest possible real part of (75+117i)z + (96+144i)/z "
            "where z is a complex number with |z| = 4."
        ),
        "true": 540,
        "trap": 432,   # |sin coefficient| instead of the amplitude
    },
    {
        "id": "p11",
        "statement": (
            "Each vertex of a regular octagon is independently colored either "
            "red or blue with equal probability. The probability that the "
            "octagon can then be rotated so that all of the blue vertices end "
            "up at positions where there were originally red vertices is m/n, "
            "where m and n are relatively prime positive integers. Find m+n."
        ),
        "true": 371,   # 115/256
        "trap": 369,   # a near-miss numerator slip (113/256)
    },
    {
        "id": "p13",
        "statement": (
            "Let p be the least prime number for which there exists a positive "
            "integer n such that n^4 + 1 is divisible by p^2. Find the least "
            "positive integer m such that m^4 + 1 is divisible by p^2."
        ),
        "true": 110,
        "trap": 134,   # a genuine root of m^4 = -1 mod 289, but not the least
    },
]


def claim_text(problem, answer) -> str:
    return (f"AIME 2024 I problem: {problem['statement']} "
            f"The answer is {answer}.")
