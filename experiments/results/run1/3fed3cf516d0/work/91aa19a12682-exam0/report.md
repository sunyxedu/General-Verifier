# 检查报告：30 | n^3-n（1≤n≤300）

## 结论：refuted

claim 字面表述为「对所有 1≤n≤300，30 整除 n^3-n」。直接算术反例：

- n = 2：2^3 - 2 = 6，30 ∤ 6。
- 遍历 1..300，共有 **120** 个 n 使得 30 ∤ (n^3-n)（`verify_check30.py` / `verify_check30_output.txt`）。

即 claim 本身为假，不需要依赖任何 Lean 检查即可判定。

## 附带发现：claim 与"绑定的方法"目标不一致

任务给出的「义务」与「绑定的方法」文本里，实际写的分解目标是 **n^5-n**（
`(n.val+1)^5-(n.val+1)`），而不是 claim 里的 n^3-n。这是两个不同的数学命题：

- 30 | n^5-n 对所有整数 n 恒成立（这是费马小定理的经典推论：n^5≡n (mod 2,3,5)）。
- 30 | n^3-n **不**恒成立——n^3-n = (n-1)n(n+1) 只保证被 6 整除，不保证被 5 整除。

按绑定方法字面执行（`check30.lean`）：

- `dvd2`/`dvd3`/`dvd5`（针对 n^5-n 的三条独立 `decide` 判定）单独编译 **通过**，exit 0
  （见 `check30_theorems_only.lean` / `lean_output_theorems_only.txt`）。
- 但方法里给出的两条反驳捷径，逐字照抄后 **编译失败**：
  `example : ¬ ∀ m, 6∣m → 5∣m := ⟨6, by decide⟩` 触发
  `Invalid ⟨...⟩ notation: The expected type ... is not an inductive type`
  （见 `lean_output.txt`，exit 1）。原因是 `¬P` 展开为 `P → False`，是箭头类型，
  匿名构造子 `⟨...⟩` 不适用于此，需要改写成 `by intro h; exact absurd (h 6 (by decide)) (by decide)`
  这样的形式才能通过（见 `check30_examples_fixed.lean` / `lean_output_examples_fixed.txt`，exit 0）。
  也就是说方法文本里声称"已实测可编译通过，exit 0"这一点，对逐字给出的代码并不成立。

即便这两条反驳捷径被修正到可编译，`check30.lean` 全文验证的对象始终是 **n^5-n**，
和 claim 陈述的 **n^3-n** 是两个不同的表达式。哪怕整份 Lean 文件顺利编译，
也只能证明「30 | n^5-n」，不能证明 claim 所说的「30 | n^3-n」——后者是假的。

## 判定依据

- outcome: **refuted**
- witness: n=2，2^3-2=6，30∤6（1..300 范围内共 120 个反例，见 `verify_check30_output.txt`）
- 附带问题（不影响最终判定，但记录在案）：
  1. 义务/绑定方法文本把 n^3-n 误写/替换成了 n^5-n，与 claim 不一致；
  2. 绑定方法给出的两条反驳捷径代码逐字执行编译失败，与方法文本"已实测可编译通过"的描述不符。

## Artifact 清单

- `check30.lean` — 逐字按绑定方法写出的完整 Lean 文件（含 dvd2/dvd3/dvd5 + 两条反驳捷径）
- `lean_output.txt` — 对 `check30.lean` 执行 `lean check30.lean` 的完整输出（exit 1）
- `check30_theorems_only.lean` — 仅 dvd2/dvd3/dvd5 三条（隔离测试）
- `lean_output_theorems_only.txt` — 对应输出（exit 0）
- `check30_examples_fixed.lean` — 修正写法后的两条反驳捷径（诊断用）
- `lean_output_examples_fixed.txt` — 对应输出（exit 0）
- `verify_check30.py` — 直接算术核查 claim 本身（n^3-n）与方法文本所述（n^5-n）的脚本
- `verify_check30_output.txt` — 该脚本的运行输出
- `report.md` — 本报告
