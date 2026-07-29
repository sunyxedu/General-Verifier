# 验证驱动搜索协议（solver 侧，建议性）

把这段文本放进 solver 的 CLAUDE.md（或系统提示附录）。它只约定**机制**，
不含任何领域知识；soundness 不依赖它——不守协议的 solver 只是拿不到
certificate，不会拿到错的 certificate。

## 提交纪律

- 每当你得出一个后续推理要**依赖**的结论，立即 `gv.commit` 它：
  - `kind`: `assertion`（中间结论）| `final`（最终答案）
  - `label`: 简短稳定（`A1`、`lemma-even`…），后续 `deps` 用 label 引用
  - `deps`: 这个结论直接依赖的所有已提交 claim；引用工具执行结果用
    `effect:<seq>`（hook 采集的事件序号）
  - 结论修正时用 `revises=<旧 label>` 重新提交：旧结论及其下游会自动失效
- claim 的陈述必须**自含且可检验**：写"对所有 1≤n≤100，6 整除 n³−n"，
  不写"上面那个性质成立"。
- 你**不需要**自己证明它——验证是 verifier 的事，什么时候验证由触发规则
  决定（一个 claim 第一次被依赖时）。

## 并行假设（AND-OR 搜索）

- 互斥假设用 subagent 并行探索；每个 subagent 提交 claim 时带
  `branch=<假设名或 agent id>`。
- 随时可调 `gv.branches` 查看：某分支出现 refuted 后整个下游锥即为死分支
  （**硬杀**：立即停止在该分支上花预算）；verified 只作软排序参考。
- 一条 claim 被打 refuted 时，它是**学到的排除项**：把反例（witness）
  读出来，避免后续假设踩同一个坑。
- 不同分支重推出相同结论会被内容寻址去重：第二次不再花验证成本。

## 收尾

- 提交 `final` 后调 `gv.certificate(final_label=...)`。
- verdict 只有三种诚实取值：`verified`（带 overall_grade）、`unverified`
  （residue 未解决，含原因）、`refuted`（切片上有反例）。把 verdict 与
  certificate_id 原样带给用户/CI；不要转述成"通过/失败"以外的话术。
- `unverified` 不是失败，是"没验上"：可以补充依据后 revise 重交。
