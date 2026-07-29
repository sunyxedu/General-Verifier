# 检查报告：11 | 10^n - 1 (1≤n≤200)

## 结论：refuted

## 反例
n = 1: 10^1 - 1 = 9，9 mod 11 = 9 ≠ 0，故 11 ∤ 9。
（事实上所有奇数 n ∈ [1,200] 都是反例，共 100 个；偶数 n 才成立，因为 10 ≡ -1 (mod 11)，10^n ≡ (-1)^n，故 10^n-1 ≡ 0 (mod 11) 当且仅当 n 为偶数。）

## 方法与声明不匹配
声明要求证明 `11 | 10^n - 1`，但"绑定的方法"描述并实际编译的引理是
`∀ n, 10^n % 9 = 1` 以及 `9 | 10^n - 1`（模数是 9，不是 11）。
`9 | 10^n - 1` 本身是真命题（因为 10 ≡ 1 mod 9），已用 Lean 4 核心库重新编译验证通过
（见 BoundMethodProof.lean / BoundMethodProof_output.txt，exit_code=0，无 sorry）。
但这个真命题与题目声明的模数（11）完全不同，不能作为声明的证据。
"仪器"验证的是错误的目标（错误的除数），因此该仪器对本声明不构成有效证明，
无论其编译是否通过。

## 数值复核
用 Python 对 n=1..200 直接计算 (10^n - 1) mod 11，逐一核验，
100/200 个 n 不满足整除关系，首个反例 n=1。见 verify_check.py / verify_check_output.txt。

## Artifact
- verify_check.py — 数值枚举验证脚本（对声明本身：11 | 10^n-1，及对绑定方法真实目标：9 | 10^n-1）
- verify_check_output.txt — 上述脚本的完整输出
- BoundMethodProof.lean — 复现"绑定方法"描述的 Lean 证明（目标是 9 | 10^n-1）
- BoundMethodProof_output.txt — 该 Lean 文件的编译输出（exit_code=0）
