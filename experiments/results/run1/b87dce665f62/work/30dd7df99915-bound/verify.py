import sympy as sp

n, k = sp.symbols('n k', positive=True, integer=True)

# 组合数的正确定义/公式：C(n,k) = n! / (k! * (n-k)!)
comb_formula = sp.factorial(n) / (sp.factorial(k) * sp.factorial(n - k))

# sympy 内置的组合数函数
comb_builtin = sp.binomial(n, k)

# 代入 n=10, k=3
n_val, k_val = 10, 3
comb_formula_at_10_3 = comb_formula.subs({n: n_val, k: k_val})
comb_builtin_at_10_3 = comb_builtin.subs({n: n_val, k: k_val})

simplified_formula = sp.simplify(comb_formula_at_10_3)
simplified_builtin = sp.simplify(comb_builtin_at_10_3)

diff = sp.simplify(simplified_formula - simplified_builtin)

print("组合数公式 C(n,k) = n!/(k!*(n-k)!) 代入 n=10,k=3:", comb_formula_at_10_3, "-> 化简后:", simplified_formula)
print("sympy 内置 binomial(n,k) 代入 n=10,k=3:", comb_builtin_at_10_3, "-> 化简后:", simplified_builtin)
print("两者之差 (应为0):", diff)

assert diff == 0, f"公式与内置函数结果不一致: formula={simplified_formula}, builtin={simplified_builtin}"
assert simplified_formula == 120, f"手写公式计算结果不是120,而是{simplified_formula}"
assert simplified_builtin == 120, f"内置函数计算结果不是120,而是{simplified_builtin}"

# 反例检查：确认没有误用排列数公式 P(n,k) = n!/(n-k)!
perm_formula = sp.factorial(n) / sp.factorial(n - k)
perm_at_10_3 = sp.simplify(perm_formula.subs({n: n_val, k: k_val}))
print("对照：若误用排列数公式 P(n,k)=n!/(n-k)! 代入 n=10,k=3 将得到:", perm_at_10_3, "(应与120不同，用于反证未误用排列公式)")
assert perm_at_10_3 != 120 or perm_at_10_3 == comb_formula_at_10_3 * sp.factorial(k_val), "排列数与组合数关系校验"

print("\n结论: C(10,3) = 120 成立，且论证使用的是组合数公式 n!/(k!(n-k)!) 而非排列数公式 n!/(n-k)!。")
print("VERIFICATION_PASSED")
