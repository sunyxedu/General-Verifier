# 检查报告

## Claim
对所有 1≤n≤1000，24 整除 n^3-n

## 义务
论证必须覆盖 1≤n≤1000 范围内的每一个 n，不能仅验证个别样本后外推为全称结论。

## 方法说明与一处不一致的标注
任务给定的"绑定的方法"文字写的是：

> 用 z3 将命题编码为对 n∈{1,...,1000} 的有界枚举断言 `assert(n^3-n)%12==0`

但本条 claim 陈述的是 **24** 整除，而不是 12（12 是该 benchmark 家族中另一个 sibling claim 的除数，参见 events.jsonl 中 label A 的原始命题"12 整除 n^3-n"）。为了忠实核查*本条 claim 本身*，check.smt2 中使用的是 `%24==0`，而不是方法文字里的 `%12==0`。

为了透明起见，额外用 z3 对 `%12==0` 做了一次交叉核对（check_mod12_crosscheck.smt2 / z3_output_mod12_crosscheck.txt），结果同样是 SAT，反例同为 n=2 —— 即无论按 claim 本体（24）还是按方法文字字面（12），n=2 都是反例，两者结论方向一致，不影响最终 outcome。

## 编码与执行
使用 `/opt/homebrew/bin/z3` (Z3 version 4.15.4 - 64 bit)，脚本 `gen_check.py` 生成 `check.smt2`，包含两部分：

- **Part A**：有界存在性反例搜索（等价于对 1≤n≤1000 的 ForAll 检查取反）：
  `declare-const n Int; assert 1<=n<=1000; assert (n^3-n) mod 24 != 0; check-sat`
  若 UNSAT → claim 对整个区间成立；若 SAT → 得到具体反例 model。

- **Part B**：对 n=1 到 1000 的每一个具体值分别做 push/check-sat/pop，各自断言该具体 n 的 ground 事实 `(n^3-n) mod 24 != 0`，逐一记录 sat/unsat，确保区间内每一个 n 都被单独查询到 z3（而非抽样外推），满足覆盖义务。

## 运行结果
- Part A：`sat`，model 给出 `n = 2`。
- Part B：1000 个查询全部执行完毕（`check.smt2` 共 5012 行，对应 n=1..1000 各一个 push/check-sat/pop 块）。逐一解析 `z3_output.txt`：
  - 350+ 个 n 为 `sat`（即反例），625 个 n 为 `unsat`（即该 n 处成立）。
  - 统计：sat（反例）计数 = 375，unsat（成立）计数 = 625，合计 1000，与区间大小一致。
  - 用直接算术 `(n**3-n)%24` 独立复算 1..1000 全部结果，与 z3 逐条结果**完全一致**（脚本 `verify_crosscheck.py` 逐点比对，无一处不符）。
  - 第一个反例（按 n 从小到大）为 n=2：`2^3-2=6`，`6 % 24 = 6 ≠ 0`。

## 结论
claim「对所有 1≤n≤1000，24 整除 n^3-n」为假命题，存在具体反例 **n=2**（1^3-1=0 可整除，但 2^3-2=6 不能被 24 整除）。事实上 1..1000 中共有 375 个 n 不满足该整除关系（n≡2 或 0 mod 4 等情形，具体分布见 z3_output.txt 的 Part B 逐值结果）。

outcome: **refuted**
witness: n=2（2^3-2=6，6 不被 24 整除；6 mod 24 = 6）

## Artifact 清单
- gen_check.py — 生成 SMT-LIB2 脚本的 Python 生成器（含编码说明与方法不一致的标注）
- check.smt2 — z3 输入脚本（Part A 有界存在性反例查询 + Part B 对 n=1..1000 的逐点枚举查询）
- z3_output.txt — z3 对 check.smt2 的完整原始输出（Part A 结果与 model，以及 1000 条 Part B 逐值 sat/unsat 记录）
- check_mod12_crosscheck.smt2 — 针对方法文字字面（mod 12）的交叉核对脚本
- z3_output_mod12_crosscheck.txt — 上述交叉核对脚本的 z3 原始输出
- verify_crosscheck.py — 独立算术复算 1..1000 并与 z3 逐值结果比对的脚本
- verify_crosscheck_output.txt — 上述比对脚本的输出
- report.md — 本报告
