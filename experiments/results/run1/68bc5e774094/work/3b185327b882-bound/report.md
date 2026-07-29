# 检查报告：对所有 1≤n≤200，9 整除 10^n-1（绑定方法执行）

## 结论：holds

## 方法执行情况
按声明绑定的方法，在本工作目录用纯 Lean 4 核心库（无 Mathlib）独立编译
`proof.lean`：
1. `pow10_mod9 : ∀ n : Nat, 10 ^ n % 9 = 1` —— 对 n 归纳，succ 情形用
   `Nat.pow_succ` + `Nat.mul_mod` 展开，覆盖全称量词的无界版本（对所有自然数
   n 成立，非仅 [1,200]）。
2. `nine_dvd_pow10_sub_one : ∀ n : Nat, 1 ≤ n → n ≤ 200 → 9 ∣ (10 ^ n - 1)` ——
   由 (1) 结合 `Nat.div_add_mod` 与 `omega` 推出，严格对应声明的量词范围
   [1,200]。

编译命令：`/opt/homebrew/bin/lean proof.lean`，退出码 0，无 sorry、无警告
（见 lean_output.txt）。`#print axioms nine_dvd_pow10_sub_one` 显示仅依赖
`propext`、`Quot.sound` 两条 Lean 核心公理，未出现 `sorryAx`，证明未被
`sorry` 占位。

## 义务核验
- **量词范围义务**：证明的是无界全称命题 `∀ n, 10^n % 9 = 1`，严格蕴含
  [1,200] 子区间，非存在性证据冒充全称证据。满足。
- **端点覆盖义务**：proof.lean 中显式给出 `n=1` 与 `n=200` 两个
  example 实例，均类型检查通过（见 lean_output.txt 无报错）。满足。
- **整除定义一致性义务**：Lean 证明用标准 `Dvd` 结构（`⟨10^n/9, ...⟩`
  给出商），对应存在整数 k 使 `10^n-1=9k`，非近似判定。满足。
- **大数精度义务**：Lean 的 `Nat` 是任意精度整数，不存在 64 位溢出问题；
  独立交叉检查用 Python 任意精度整数复算，10^200-1 为 200 位十进制数，
  未做截断（见 verify_crosscheck_output.txt）。满足。
- **通用引理证明义务**：同余在乘方下保持的引理即
  `pow10_mod9` 本身（本质上是 10≡1(mod 9) 蕴含 10^n≡1(mod 9) 的归纳证明），
  已显式给出，非未加说明跳步。满足。
- **反例排除义务**：Python 对 n=1..200 逐一计算 `(10**n-1) % 9`，
  0 个反例（见 verify_crosscheck_output.txt）。满足。
- **可机检域的可执行验证义务**：本目录同时留存 Lean 形式证明（含编译
  日志）与 Python 可复现数值枚举脚本（含真实运行输出），双重可机检产物。
  满足。

## 与之前 exam0（模数 11 变体）的区别
上一个 `-exam0` 目录检查的是量词变体"对所有 1≤n≤200，11 整除 10^n-1"，
该变体被正确判定为 refuted（首个反例 n=1）。本次 `-bound` 检查的是原始
声明本身（模数 9），绑定方法的目标模数与声明模数一致，未出现 exam0
中发现的"仪器验证错误目标"问题。经独立编译与数值交叉检查，两条证据
一致支持 holds。

## Artifact
- proof.lean —— 按绑定方法描述编写的 Lean 4 核心库证明（本次独立复现编译）
- lean_output.txt —— `lean proof.lean` 完整输出（含 #check、#print axioms、退出码）
- verify_crosscheck.py —— Python 任意精度数值交叉检查脚本
- verify_crosscheck_output.txt —— 该脚本完整运行输出
