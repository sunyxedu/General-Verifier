import math

N = 8191
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
          47, 53, 59, 61, 67, 71, 73, 79, 83, 89]

# 1. Verify the upper bound floor(sqrt(8191)) == 90 claimed in the obligation.
sqrt_exact = math.sqrt(N)
floor_sqrt = math.isqrt(N)
print(f"N = {N}")
print(f"math.sqrt({N}) = {sqrt_exact!r}")
print(f"math.isqrt({N}) (floor(sqrt(N))) = {floor_sqrt}")
bound_claim_ok = (floor_sqrt == 90)
print(f"Claimed bound floor(sqrt({N})) == 90 ? {bound_claim_ok}")
# Sanity check: 90^2 <= N < 91^2
print(f"90**2 = {90**2}, 91**2 = {91**2}, 90**2 <= N < 91**2 ? {90**2 <= N < 91**2}")
print()

# 2. Check the given prime list actually covers all primes <= floor(sqrt(N)).
def sieve_primes_upto(n):
    is_p = [True] * (n + 1)
    is_p[0:2] = [False, False]
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]

expected_primes = sieve_primes_upto(floor_sqrt)
print(f"Primes <= floor(sqrt(N))={floor_sqrt} via sieve: {expected_primes}")
print(f"Given obligation prime list          : {primes}")
list_matches_sieve = (expected_primes == primes)
print(f"Given list matches full prime list up to bound ? {list_matches_sieve}")
print()

# 3. Trial-divide N by every prime in the given list, print remainder and divisibility.
print("Trial division of 8191 by each candidate prime:")
print(f"{'p':>4} | {'8191 % p':>10} | divides?")
print("-" * 32)
any_divides = False
results = []
for p in primes:
    rem = N % p
    divides = (rem == 0)
    any_divides = any_divides or divides
    results.append((p, rem, divides))
    print(f"{p:>4} | {rem:>10} | {divides}")

print()
all_nondivide = not any_divides
print(f"All 24 primes fail to divide {N} (i.e. no divisor found) ? {all_nondivide}")

# 4. Final verdict combining bound correctness, list completeness, and trial division result.
is_prime_confirmed = bound_claim_ok and list_matches_sieve and all_nondivide
print()
print(f"floor(sqrt(N)) == 90 correct: {bound_claim_ok}")
print(f"Prime list complete (matches sieve up to bound): {list_matches_sieve}")
print(f"No candidate prime divides N: {all_nondivide}")
print(f"=> 8191 is prime, obligation fully verified: {is_prime_confirmed}")
