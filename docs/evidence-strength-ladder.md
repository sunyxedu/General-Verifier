# 证据强度阶梯（graded evidence ladder）落地方案

状态：已实现（`strength-ladder` 分支）。实现期间新增两条精化：①空 repair 的边界宣告不算 executor 审计 gap（纯失败不升档）；②非 decisive 的反面信号（被转换的 refutation）记 contrary_signal——聚合封顶 failed（不得 partial），且该 tier 无 unobtainable 天然锁住更弱档（活着的反向指标之上不许弱证据说 holds）。驱动场景：

> claim =「有个 bug 是 xxx」，obligation 与 claim 同义（world floor），
> 可用方法三种：① system test ② unit test ③ 读相关代码片段给结论。

## 0. 验收行为（实现完成的判据）

| 执行路径 | 今天 | 方案后 |
|---|---|---|
| system test 跑通，观察到违反 | verified | verified（不变） |
| system test 结构性不可用 → unit test holds | **partial**（unit test 根本不许列） | **supported@bounded**，证书带桥接假设 |
| unit test 也不可用 → 纪律化读码 holds | partial | **supported@analytic**，证书带 witness 轨迹 |
| system test 跑通，行为正确 | refuted | refuted（不变） |
| unit test 挂了 / 读码认为没 bug | —（不许列） | **不落 refuted**：机械转 failed + witness 入 repair 反馈；contrary_signal 封顶 unverified 并锁住更弱档 |

分解层零改动：obligation ≡ claim 本就合法（critic 攻击充分性/必要性，同义分解平凡满足；contract-locus 规则禁止添加 claim 没断言的要求）。

## 1. 数据模型（model.py / store.py / discharge.py）

- 新增第二轴，与 basis 正交：`STRENGTH_DECISIVE > STRENGTH_BOUNDED > STRENGTH_ANALYTIC`（model.py 常量 + `weakest_strength()` fold 辅助）。
  - decisive：对 obligation 原文的决定程序（今天 sufficiency 的标准，双向可结算）。
  - bounded：有界仪器，判定严格更窄的命题，**必须**附桥接假设清单使 {仪器 verdict + 桥} ⇒ obligation。只准 holds 方向。
  - analytic：纪律化读码（见 §2 准入三件套）。只准 holds 方向。
- 流转路径与 basis 完全同构：oracle method 申报 → 绑定 `_settle` 进 `DischargeResult.strength` → kernel `_discharge` 写进 evidence payload + `store.set_verified(oid, basis=…, strength=…)` → claim fold → certificate。
- store.py：claims 表加 `strength TEXT` 列，迁移与 basis 同款（`PRAGMA table_info` + `ALTER TABLE`）。`_row`/`graph_snapshot`/`obligation_children` 带出。
- 缺省朝弱收：绑定路径 `method.get("strength")` 缺失按 **analytic** 收（对称于 basis 缺失收 record）。LedgerInstrument/effect 与 graph 继承按其来源 strength；ledger 摘要核对本身是 decisive。
- **不设静态 strength floor**：与 `required_basis` 不同，强度的"地板"是动态的——由降级审计决定这次允许落到哪一档，不在编译期声明。compiler/critic 不感知 strength。

## 2. Oracle 阶梯（prompts ORACLES_SYSTEM/TEMPLATE + agents schema）

schema（mini-DSL 全键必填）：

```python
"methods": [{"procedure": str, "artifact": str, "level": str,
             "basis": ("world", "record"),
             "strength": ("decisive", "bounded", "analytic"),
             "bridges": [str]}]   # decisive 传 []
```

prompt 规则（改写 ORACLES_SYSTEM 的排序段，新增 STRENGTH LADDER 段）：

1. decisive 档维持现行 DECISION PROCEDURE 标准原文不动。
2. "insufficient 的方法一律不许出现"改为：**作为 decisive 不充分**的方法，可以把缺口逐条转写成显式桥接假设后以 bounded 档列出；缺口无法有限枚举成可检命题的，仍然禁止出现。
3. 桥接假设纪律：每条桥 = 关于本任务实例的、可独立检查的具体命题（"路径 P 下 f(I) 可达"、"上游 sanitizer 不拦截 I"、"harness stub 与真实调用上下文一致"）。"我的推理正确"、复述 obligation 本身，都不是桥。桥集必须闭合缺口：{仪器结论 + 全部桥} 必须逻辑上推出 obligation。
4. analytic 准入三件套：(a) 证据钉死——引用片段 + 对 captured 树的文件 hash；(b) 具体 witness——触发输入 + 逐步手工执行的值轨迹（可复查的符号执行），不是"看起来不对"；(c) 桥申报——"读到的代码即构建所跑代码"类。三缺一即不许列。
5. 排序：跨档由引擎机械排（见 §3），oracle 只需**档内**维持 simplest-sufficient-first；bounded/analytic 是后备不是替代——仅当 decisive 档存在现实的结构性不可用风险时才列。
6. 空档即签字：不列 decisive 档 = 签字声明"该 obligation 原则上不存在决定程序"，judger 会对照降级记录审计这个声明；站不住 ⇒ 弱档方法被拒。
7. 非 decisive 方法的充分性合同是单向的：只能建立，不能反驳。

## 3. 执行引擎机械规则（game.py）

1. **排序**：basis floor 过滤后，执行队列按 `(tier, oracle 档内序)` 稳定排序；repair 批次并入后对未执行尾部重排。
2. **解锁门（与 binding-game 聚合修复共用同一条规则）**：
   tier 记为结构性不可用 ⟺ 该 tier ≥1 次 **executor 审计的** unobtainable 且 0 个已结算 verdict。failed 尝试在案但不挡门（judger 的降级审计兜底：认为某个 failed 的 decisive 其实可完成 ⇒ unjustified 拒绝）。空 tier 直接过门，但"空档"写入降级记录。
   同一条规则同时作 obligation 终端聚合：`unavailable_only` = (有 executor 审计 gap 或（有边界宣告且无 failed）) 且无降级争议且无 contrary_signal——chess 案（构造失败 + 诚实 unobtainable 混合）由此落 partial 而非 unverified。两个不升档的例外：空 repair 的"再无新方法"边界宣告不算审计 gap（否则纯失败批会被 oracle 一句话洗成结构性缺口）；contrary_signal 在场时封顶 failed。RoleError 类协议失败照旧 failed，不入此规则。
3. **降级记录**：非 decisive 方法送审时附 `[{tier, procedure(截断), outcome, reason}]`（全部更强档尝试）+ `omitted_tiers`。由 attempts 列表机械构建。
4. **holds-only 硬闸**：非 decisive 方法 discharge 出 refuted → 机械转 `OUTCOME_FAILED`，witness 全文保留并进 `structural_feedback`（提示 repair：有界仪器出了反面信号，找 decisive 反驳法）。kernel `_discharge` 现有 basis backstop 旁加同款 strength backstop（覆盖 direct-discharge 路径）。
5. `_attempt_diagnostic` 带 strength/bridges，repair 反馈可见档位全景。

## 4. Judger 三审（prompts JUDGER_SYSTEM/TEMPLATE + schema）

schema 加一键：`"degradation": ("justified", "unjustified", "n/a")`。

- **审一 sufficient——按申报档位审**：decisive 按现行标准；bounded 审"仪器对其所测命题是否 sound + 桥集是否闭合缺口 + 每条桥是否真可独立检查"；analytic 审读码合同是否满足三件套。假 decisive（有缺口却报 decisive）⇒ `sufficient=no`，reason 说出诚实档位与缺的桥。
- **审二 faithful——照旧**，analytic 的逐步轨迹复核归入 faithful：每一步是否真的跟着钉死的代码走、witness 是否具体、hash 是否对得上 captured 树。
- **审三 degradation（新）**：对照降级记录审"上面每一档真的不可用吗"——unobtainable 理由是否属实（真无 API key？ledger/world 里查）、失败的 decisive 是否其实可完成、空档声明是否成立（明明存在"跑 captured 树里的 repro 脚本"这种决定程序）。`unjustified` ⇒ 该方法按 insufficient 路径拒绝（不结算、进 structural_feedback）；decisive 方法恒 `n/a`。
- reason 强制前缀追加一段：`STRENGTH AUDIT: declared=<tier>; honest=<tier+why>; bridges_complete=yes|no|n/a; degradation=<…+why>`。

## 5. 结算 / 证书 / 排序（kernel / certificate / offline / experiments）

- claim fold（kernel `_settle`）：与 bases 并行收集 strengths，全 holds 时 claim strength = 最弱档；`store.set_verified(claim_id, basis=…, strength=…)`。**不新增 status**——状态机零改动，supported 是读出层：
- certificate.py：`verdict = "supported"` 当 final status=verified 且其 strength ≠ decisive；新增 `verdict_strength` 字段；entries 逐节点带 `strength`；bridges 已在 evidence payload 里，自动上证书。
- dedup / `_inherit` / `settled_statement` 继承路径同时传递 strength（防止 supported 级 verdict 经继承洗成满级）。
- offline.py：`VERDICT_RANK = {verified 4, supported 3, partial 2, unverified 1, refuted 0}`；features 加 `verdict_strength`；fingerprint 自动 bump。
- experiments/common/scoring.py：scalar 改六段带（/6）；selection.py `weighted_vote` 插 supported 权重。
- 测量分层：P(reward=1 | supported@bounded) 与 @analytic 分开算；analytic 的命运（降权/独立降档）由数据定。

## 6. 不动的边界

1. basis 轴原样：record 方法照旧不许碰 world floor、all-record final 照旧弃权、洗白 backstop 原文不动——solver 自述"我看过代码"永远是 record，不因分档受惠。
2. refuted 只认 decisive（§3.4 机械保证）——同时是 FCMkLWN 型假反驳的结构性防线：假反驳面不因新档扩大。
3. compiler / critic / 分解层一行不动。
4. 各 fail-closed 路径（RoleError、append guard、judger `_infra`）原样。

## 7. 验收剧本

bug 场景 fixture 全分支（见 §0 表）+ 回归：
- chess-best-move__Q7bxNyn：完备性 conjunct 预期 unverified → partial（聚合规则）或 supported@bounded（引擎有界证据 + 桥申报）；
- fix-git__g2uoWWb：不受影响（refuted 通道未放宽）；
- 全部现有测试改造点：oracle/judger schema 新键、judger fixture 加 degradation、direct-discharge 缺省 strength、rank 插档。
- A/B：同批 TB/FM trace 新旧 config 重打，硬门槛 false certificate=0，supported 分层 precision 为 go/no-go 后验。

## 8. 主要风险

supported 的全部价值压在 judger 对桥闭合性与降级属实性的严格度上——所以 refuted 不放宽、precision 分层后验、unjustified 即拒绝三条缺一不可。次要风险：解锁门与聚合规则放宽 failed 的拦截，靠三重兜底——降级审计、边界宣告不算审计 gap、contrary_signal 封顶；若 A/B 仍出现 failed 掩护下的假 supported，把门收紧回"failed 即锁"并单独重跑该批。
