bad = []
for n in range(1, 301):
    v = n**5 - n
    if v % 60 != 0:
        bad.append((n, v, v % 60))

print("total n in 1..300 with 60 not dividing n^5-n:", len(bad))
print("all such n:", [n for n, _, _ in bad])
print("first witness:", bad[0])
