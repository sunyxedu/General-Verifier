# 检查报告

## Claim
对所有 1≤n≤300，60 整除 n^5-n。

## 发现 1：给定的绑定方法与 claim 不一致
任务描述中的"绑定的方法"给出的 Lean 命题是：
```
∀ n : Fin 300, 30 ∣ ((n.val+1)^5 - (n.val+1))
```
divisor 是 **30**，而 claim 陈述的是 **60**。二者不是同一个命题：n^5-n 对所有整数恒被 30 整除（Fermat 小定理对 p=2,3,5 的推论），但并不恒被 60 整除。因此即便按原文所述方法跑通 `by decide`，也只是证明了一个比 claim 弱的、不同的命题，并不能验证 claim 本身。

## 发现 2：claim 本身按字面（divisor=60）不成立
直接用 Lean 对字面 claim 编码并 `by decide`：
```
theorem claim_60 : ∀ n : Fin 300, 60 ∣ ((n.val+1)^5 - (n.val+1)) := by decide
```
结果（见 `lean_output.txt`）：
```
Check.lean:8:71: error: Tactic `decide` proved that the proposition
  ∀ (n : Fin 300), 60 ∣ (↑n + 1) ^ 5 - (↑n + 1)
is false
```
即 Lean 内核直接给出该全称命题为假的判定证明（对 300 个具体实例逐一规约后发现存在反例），而非采样。

## 反例
n = 2：2^5 - 2 = 30，60 ∤ 30。
已用 Lean 独立确认（见 `counterexample_output.txt`）：
```
example : ¬ (60 ∣ (2^5 - 2 : Nat)) := by decide   -- 编译通过，exit 0
```
Python 穷举 1..300 复核：300 个整数中有 75 个使 60 ∤ (n^5-n)（如 n=2,6,10,14,...），进一步说明反例并非孤立情况。

作为对照，绑定方法实际编码的命题（divisor=30）确实成立：
```
theorem bound_method_actual_claim : ∀ n : Fin 300, 30 ∣ ((n.val+1)^5 - (n.val+1)) := by decide
```
该定理编译通过（同样记录在 `counterexample_output.txt`），证实 30 |与 60 的区分是关键，而非笔误可忽略的细节。

## 结论
outcome: refuted
witness: n = 2（2^5-2=30，60 不整除 30）；另有 74 个反例（n=6,10,14,18,...300 区间内共 75 个），均可由 Python 穷举复核。
附带发现：任务给出的"绑定方法"命题本身编码的是 divisor=30 而非 claim 所需的 60，属于方法与义务不匹配，不能用于验证该 claim。
