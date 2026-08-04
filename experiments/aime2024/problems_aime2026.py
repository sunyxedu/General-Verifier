"""Seven non-geometry problems from AIME 2026 I (held 2026-02-05, after
the seat models' training cutoff — memorization is off the table for both
arms). Statements are restated in our own words; the mathematical content
is unchanged. Every official answer is re-derived by brute force in
labels_aime2026.py, and the official answer key (Areteem/AoPS) agrees.

Traps are computed mistake-values, not noise: p07's 240 is the count if
you misread "π⁶ = id" as "order exactly 6"; p09's 41 is the answer if you
forget the conditioning; p13's 463 counts the complement; p08's 458 is
the count with the free exponent forgotten (mod 1000); p15's 166 double
counts; p06's 121 is a factor-2 slip in Vieta's sum; p02's 124 doubles a
self-symmetric count.
"""

PROBLEMS = [
    {
        "id": "q02",
        "statement": (
            "Find the number of positive integers with the following three "
            "properties: the base-10 representation is a palindrome (it "
            "reads the same left to right as right to left), none of its "
            "digits is zero, and its digits add up to 13. For example, "
            "42124 has all three properties."
        ),
        "true": 62,
        "trap": 124,
    },
    {
        "id": "q06",
        "statement": (
            "The product of all positive real numbers x that satisfy the "
            "equation  (x^(log_2026(x)))^(1/20) = 26x  is an integer P "
            "(here log_2026 denotes the logarithm with base 2026, and the "
            "exponent 1/20 denotes the twentieth root). Find the number of "
            "positive integer divisors of P."
        ),
        "true": 441,
        "trap": 121,
    },
    {
        "id": "q07",
        "statement": (
            "Find the number of functions pi mapping the set "
            "A = {1, 2, 3, 4, 5, 6} onto itself such that for every a in A, "
            "applying pi six times gives back a: "
            "pi(pi(pi(pi(pi(pi(a)))))) = a."
        ),
        "true": 396,
        "trap": 240,
    },
    {
        "id": "q08",
        "statement": (
            "Let N be the number of positive integer divisors of 17017^17 "
            "that leave a remainder of 5 upon division by 12. Find the "
            "remainder when N is divided by 1000."
        ),
        "true": 244,
        "trap": 458,
    },
    {
        "id": "q09",
        "statement": (
            "Joanne has a blank fair six-sided die and six stickers labeled "
            "with the integers 1 through 6, one label each. She rolls the "
            "die and places the sticker labeled 1 on the face that lands on "
            "top. She rolls again and places the sticker labeled 2 on the "
            "face now on top, and continues in order until sticker 6 is "
            "placed after the sixth roll. If the face landing on top "
            "already carries a sticker, the new sticker is placed over it, "
            "covering it. Let p be the conditional probability that at the "
            "end exactly one face of the die is still blank, given that all "
            "three even-numbered stickers are visible. Then p = m/n in "
            "lowest terms with m and n positive integers. Find m+n."
        ),
        "true": 29,
        "trap": 41,
    },
    {
        "id": "q13",
        "statement": (
            "For each nonnegative integer r less than 502, define S_r to be "
            "the sum of all binomial coefficients C(10000, k) with "
            "0 <= k <= 10000 and k congruent to r modulo 502. Find how many "
            "of the 502 numbers S_0, S_1, ..., S_501 are multiples of the "
            "prime 503."
        ),
        "true": 39,
        "trap": 463,
    },
    {
        "id": "q15",
        "statement": (
            "In a 10x10 grid of unit cells, define an a x b cell loop "
            "(where a and b are integers with 2 <= a, b <= 10) to be the "
            "set of 2a+2b-4 cells forming the boundary ring of an a x b "
            "rectangle of cells — that is, all cells of the rectangle "
            "except its (a-2) x (b-2) interior (the interior may be "
            "empty). Find the number of ways to partition the 10x10 grid "
            "into exactly 5 cell loops so that every cell belongs to "
            "exactly one loop."
        ),
        "true": 83,
        "trap": 166,
    },
]
