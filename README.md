# General Verifier — Test-time Verification

通用验证器 meta-harness：验证 solver（如 Claude Code）在推理过程中产出的 claim，
不靠"再问一遍模型"，靠**义务编译 → 方法绑定博弈 → 机械执行 → 双重审计**的证据链。
两个立项问题：(1) 只靠推理的 verifier 验不了需要与环境交互的 claim；
(2) verifier 用了你没打算让它用的验法，"通过"不等于"对"。

设计文档：`Test-time Verification 技术报告.pdf`（v0.3.1，HTML 源 `report/report.html`）。
**硬约束**：`verifier/` 内所有 prompt 模板零领域知识——领域内容只在运行时涌现
（环境探测、模型潜在知识、记忆检索），正确性全部由机械闸门把守
（总原则：生成只影响效率，不影响正确性）。

## 架构（全部已实现）

```
solver (原装 Claude Code) ──MCP stdio──> sidecar (verifier/sidecar.py)
     └─ hooks ──> hook_feed.jsonl ──tail──> 统一事件流 events.jsonl (append-only)
                                                  │ builder (增量构图)
                                                  v
                claim store (SQLite, 内容寻址, 状态机, 分支/修订/污染守卫)
                                                  │ 单一触发规则:
                                                  │ "unverified claim 获得第一个依赖者时验证它"
                                                  v
        kernel ── 义务编译(agent) ─ 覆盖审计(验漏) ─ 绑定博弈 ─ 结算
                     绑定博弈 (verifier/game.py, 强度优先):
                     oracle 枚举(agent 实测环境) → 机械按证据强度排序
                     → 闸一 区分度考试(错误变体必须全部被拒)
                     → discharge(agent 真执行, artifact 落盘快照)
                     → 闸二 蕴含审计(独立 agent 开 artifact 复核)
                     → 不可达/被拒 → 降级到次强方法 → 穷尽则 failed
                                                  v
                certificate: 依赖闭包切片 + 证据链 + 冻结
                verdict ∈ {verified(带最弱等级), unverified(带原因), refuted(带反例)}
```

- **SDK 双轨**：要动手的角色（探测/执行/审计）= Claude Agent SDK agent
  （`verifier/agents.py`，官方 `output_format` 结构化输出 + `tools` 硬工具面板）；
  纯单发 LLM call = OpenAI SDK（`verifier/roles.py`，provider 无关）。
- **等级诚实**：沙箱机械执行 `demonstrated`；agent 自报告压 `corroborated`；
  蕴含审计给等级、证据类型给上限（formal→proven / execute→demonstrated /
  text→corroborated），取弱者；整体 = 切片最弱链。
- **搜索支持 (M3)**：分支即 claim 身份的一部分；refuted 硬杀下游锥（事件
  `branch_pruned`，建议性——soundness 靠污染守卫）；跨分支同内容结论去重复用证据。
- **记忆 (M4)**：义务模式 / 攻击模式 / 仪器注册表三库，机械检索（bigram Jaccard），
  只进 prompt 槽位——命中让系统更快，永远不能让错的验成对的。

## 成立性实验（experiments/results/README.md）

12 条真/假数学 claim，验证器臂 vs 同模型直判臂：验证器 **false-pass 0 /
false-alarm 0 / 弃权 0，12/12**——6 假全部 refuted 带反例，6 真全部 verified
（4 条 proven：Lean 形式化 / sympy 复算过蕴含审计）。区分度考试、考题质检、
打回重出、世界性义务契约这三个机制修复都由该实验驱动出来。

## 用法

```bash
# 测试（120 离线；live 验收 GV_LIVE=1 会花真 token）
.venv/bin/python -m pytest
GV_LIVE=1 .venv/bin/python -m pytest tests/test_agents_live.py tests/test_sidecar_live.py

# sidecar：solver 侧 .mcp.json 挂载
# {"mcpServers": {"gv": {"type": "stdio", "command": ".venv/bin/python",
#   "args": ["-m", "verifier.sidecar", "--state-dir", ".gv", "--memory-dir", ".gv-mem"]}}}
# solver 协议文本（放进 CLAUDE.md，建议性）: protocol/solver-protocol.md
# hooks 采集工具事件: hooks/pre_tool_use.py, hooks/post_tool_use.py

# CI 出证闸（消费端执法：verdict 非 verified 则退出码 1）
.venv/bin/python -m verifier.cli certify --events trace.jsonl --state-dir .gv-ci
.venv/bin/python -m verifier.cli replay --events trace.jsonl   # LLM-free 状态回放

# 成立性实验（false-pass 率 vs LLM 直判）
.venv/bin/python experiments/false_pass.py --out experiments/results/run1
```

## 模块地图

| 模块 | 报告组件 | 内容 |
|---|---|---|
| `verifier/model.py` | §14 | 内容寻址、Claim/Obligation/Evidence、等级 |
| `verifier/events.py` | 5 | 统一事件流（append-only JSONL、seq 校验） |
| `verifier/store.py` | 10/12/13 | claim store、单一触发规则、切片、分支/死支 |
| `verifier/builder.py` + `tail.py` | 4/6/3 | 增量构图；撕裂行容忍的在线 tail |
| `verifier/roles.py` | 15 | 单发角色（OpenAI SDK）、schema 闸 |
| `verifier/agents.py` | 15 | agent 角色（Claude Agent SDK）、五座位 |
| `verifier/instruments.py` | 25/26 | python 沙箱（fail-closed）、查账仪器、artifact 库 |
| `verifier/discharge.py` | 20 | 沙箱/agent 两种 discharge 座位 |
| `verifier/game.py` | 17-19/21 | 强度优先绑定博弈、两道闸、降级 |
| `verifier/coverage.py` | 22 | 验漏：claim 级变体考义务集、补义务 |
| `verifier/kernel.py` | 14/20/21 | 触发入口、污染守卫、结算、去重、剪枝 |
| `verifier/certificate.py` | 13/23/33 | 结算残余、冻结、诚实 verdict |
| `verifier/sidecar.py` + `cli.py` | §11 | MCP 服务、hook 摄入、CI 闸 |
| `verifier/memory.py` | §13 | 三库（advisory only） |
