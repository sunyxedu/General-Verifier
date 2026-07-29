# 成立性实验：false-pass 率（验证器 vs LLM 直判）

基准：12 条可机检数学 claim（6 真 + 6 微妙假），见 `../false_pass.py`。
两臂同模型（claude-sonnet-5）：直判臂 = 无工具单发结构化判断；验证器臂 =
完整内核（义务编译 agent → 绑定博弈：oracle 枚举/区分度考试/蕴含审计 → agent
discharge）。指标：false-pass（假当真，唯一致命项）、false-alarm（真当假）、
abstain（验证器专有：unverified 是诚实的"没验上"，不折算成通过）。

## 结果

| 臂 | n | false-pass | false-alarm | abstain | 正确 |
|---|---|---|---|---|---|
| 直判 (run1) | 12 | 0 | 0 | — | 12 |
| 验证器 (run1, 初版机制) | 12 | **0** | 0 | 6 | 5（+1 infra error）|
| 直判 (run2) | 12 | 0 | 0 | — | 12 |
| 验证器 (run2, 终版机制) | 12 | **0** | **0** | **0** | **12** |

run2 验证器裁决明细：6 假全部 refuted（Lean/python 反例入证）；6 真全部
verified，其中 4 条 **proven**（Lean 形式化 / sympy 符号复算通过蕴含审计）、
1 条 demonstrated、1 条 corroborated——等级如实反映证据种类，从不谎报。

两点解读：
1. 这套基准对 sonnet-5 直判太容易（12/12），区分度在别处：验证器的每个
   verdict 都带可复核证据链（检查脚本、运行输出、审计记录，内容寻址冻结），
   且在验不动时弃权而非猜测——run1 的 6 次弃权全部发生在真 claim 上，
   假 claim 一条都没放过（fail-closed 方向从未破）。
2. run1 → run2 的差距全部来自实验驱动的三个机制修复（见下），
   不是提示词调参。

## 实验暴露并修复的三个机制缺口

1. **考题模式移植**（`fix(game): transport obligation pattern onto exam variants`）
   义务是 claim 的函数不是常量："3|n⁵−n" 编译自原 claim 后是算术事实，
   对任何变体都成立，用它直接考变体会冤枉所有正确方法。区分度考试现在把
   义务作为模式移植到变体上检查同构实例。

2. **考题质检 + 打回重出**（`feat(game): exam-question audit …` / `… regenerate …`）
   变体被某方法放行时，先由独立 agent 复核变体是否真的以违反该义务的方式
   为假；质检坐实"不假/偏题"才剔题（事件 `variant_disqualified`），并把理由
   反馈给生成器重出一轮；质检不明确则留题拒方法——失败方向永远 fail-closed。
   空考卷不放行任何方法。

3. **义务必须是世界性条件**（`fix(agents): obligations must be world-checkable…`）
   answer-blind 的正确推论：编译器看不到论证，因此"证明必须给出某个变形"
   这类面向论证的义务是范畴错误（论证的中间步骤各自作为 claim 入图受验）。
   收紧契约后义务从 ~7 条塌缩到 1-2 条中肯的真值条件，真 claim 的弃权全部
   转化为 verified，且多数以 proven 等级绑定形式化仪器。

附带的工程教训（都已修复入库）：强度优先直取 Lean 会当场拉 mathlib——快照
必须机械排除工具链目录并设总量上限（磁盘曾被 7.3G workdir + 1.1G artifact
打满）；CLI 把 429 报成不透明的 "error result: success"——SDK 后端需要
`api_error_status` 分类 + 指数退避（run2 曾整轮被限流打崩）。

复现：
```bash
.venv/bin/python experiments/false_pass.py --out experiments/results/run3 \
    --model claude-sonnet-5 --concurrency 2
```
每条 claim 的完整事件流、artifact 与证书素材在 `runN/<slug>/`（不入库），
裁决记录在 `runN/<slug>.json`，汇总在 `runN/summary.json`。
