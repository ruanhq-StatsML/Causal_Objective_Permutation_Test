# FSDS 场景一页纸（中文）— 数据消化与经济效益 justification

每场景一页，侧重 **数据怎么消化**、**钱怎么 justify**。模拟数字标 **（sim）**。复现：`python3 run_fsds_business_impact_pack.py`。

---

## 场景 1 · 跨境销售 Copilot（联邦：法务 / 财务 / 产品）

**数据消化**

| 环节 | 内容 |
|------|------|
| **切分** | 垂直块：条款 embedding（法务）、margin 特征（财务）、个性化（产品）；各 silo **本地持 X_k**。 |
| **REF/LIVE** | 批次 **W**；每块算 MMD²、域 AUC(**X**,**W**)、ESS、VIMP top。 |
| **上行** | 每轮 **O(\|F_k\|)** 包络，不传 **(n_ref+n_live)×p** 全矩阵。 |
| **服务端** | 合并排序 → drift 类型 → 块级 OFS 或 **仅收紧法务链**。 |
| **多 agent** | 角色段做 budget；**法务块 drift ≠ 关停** 财务/产品 agent。 |

**经济效益 justification**

| 类型 | 说法 | 依据 |
|------|------|------|
| **直接 OPEX** | 少全局重辩、stable 角色少 token | demo ~**39%** $/debate、~**20%** 轮次 |
| **增量 (sim)** | 可控上线 vs 一刀切停 copilot | 叙事 **~$125k/季度** |
| **审计** | 块级 **impact receipt** | `federated_legal_agent_blocks.json` |

**进一步 justify**：用真实暂停市场的季度 GMV 替换 sim；A/B 法务收窄 vs 全站下线。

---

## 场景 2 · 银行 / 保险联邦监控

**数据消化**

| 环节 | 内容 |
|------|------|
| **约束** | 法人 **不能并表 X**；可共享 **Y**、批次 **W**、聚合统计。 |
| **本地** | 各节点在 **F_k** 上 batch FSDS；可选 DRPerm / PO-risk。 |
| **协议** | 上行 **3\|F_k\|+3** 标量/轮（相对全矩阵的 Eff）。 |
| **服务端** | p 值合并 + **在线 FDR**；keep / retire 候选 / 复核告警。 |
| **验证** | 注入 shift → bench **top-1 块命中**。 |

**经济效益 justification**

| 类型 | 说法 | 依据 |
|------|------|------|
| **直接（基础设施）** | 少 egress/存储 | **48×–1410×** Eff；Spambase **1.45 KiB** vs **~2 MiB**/窗 |
| **增量** | 禁并表处仍能监控 → 原 $0 项目可立项 | 合规可行 |
| **风险** | concept 型不自动删特征 | drift 分型逻辑 |

**进一步 justify**：云账单×窗数/年；并表法务评审 FTE 节省。

---

## 场景 3 · 集团 Uplift / 投流（两批次）

**数据消化**

| 环节 | 内容 |
|------|------|
| **契约** | REF 训 **τ̂**；LIVE **冻结**打分；**W** 批次、**T** 处理。 |
| **闸门** | MMD²、ESS(**ê(W\|X)**)、域 AUC、SRM。 |
| **L1** | AUUC、LOCO–AUUC、分位 pairwise；**AUUC_ovlp vs global** 分 mix / 排序。 |
| **L2** | 分 slice ATE；PO-risk 门控概念动作。 |
| **输出** | business_rules + economics + receipts；RELEARN 进 **CAPEX 桶**。 |

**经济效益 justification**

| 类型 | 说法 | 依据 |
|------|------|------|
| **直接 OPEX (sim)** | 差分位 cap → 少烧 treatment | 均值 **~$818/窗**；四场景合计 **~$3.3k** |
| **CAPEX 纪律** | RELEARN **不进** OPEX net | tickets **~$10k** 单列 |
| **增量** | promo mix  alone 不触发每周重训 | 先 OPEX 阶梯 |

**进一步 justify**：填入真实 **margin**、**单次 treat 成本**、LIVE 样本量。

---

## 场景 4 · Agent 平台（多角色辩论 + LangGraph）

**数据消化**

| 环节 | 内容 |
|------|------|
| **段** | 辩论角色 utterance **或** 图 checkpoint 向量。 |
| **批次** | REF/LIVE 轨迹特征（工具、长度、混部）。 |
| **预算** | shift×quality×need → **L**、tier、checks。 |
| **宏观** | 轮次 centroid **dispersion** → **辩论早停**。 |
| **微观** | 只重跑 **drift 节点**（如 tool_call）。 |
| **收据** | closed_loop **impact_receipt**。 |

**经济效益 justification**

| 类型 | 说法 | 依据 |
|------|------|------|
| **直接 OPEX** | SoT 关键路径 | **~34%** $/ep（**~$0.012/ep**） |
| **直接 OPEX** | 多 agent + 早停 | **~39%** $/debate + **~20%** 轮次 |
| **规模示意** | 50 万 ep/月 | SoT **~$72k/年** + 辩论 **~$16k/年** |

**进一步 justify**：平台按「带 receipt 的 $/千任务」定价；生产 shadow A/B。

---

## 场景 5 · BPO / 人机协同质检

**数据消化**

| 环节 | 内容 |
|------|------|
| **单元** | Handoff embedding（agent↔人 / 人↔人）。 |
| **REF/LIVE** | handoff 特征分布 shift。 |
| **定位** | FSDS 标 **哪些段** drift — 非全量复核。 |
| **联邦** | 职场/租户 = 块；只上传统计、不跨域原文。 |
| **收据** | 与 agent 闭环同构，支撑抽检政策。 |

**经济效益 justification**

| 类型 | 说法 | 依据 |
|------|------|------|
| **直接 OPEX** | 督导人时 ∝ 复核条数 | **只审 shift 段** |
| **公式** | (全量率 − FSDS 率) × **$/条** × 量 | 用 BPO 合同填 |
| **增量** | 监管认可的抽检 vs 拍脑袋 spot check | statistic + rule_id |

**进一步 justify**：一个月复核日志 → shift 覆盖 vs 随机抽样的缺陷捕获率。

---

## 场景 6 · 特征仓 / 在线 OFS（联邦块）

**数据消化**

| 环节 | 内容 |
|------|------|
| **流** | 特征组/块随时间入仓（OFS 不可撤销）。 |
| **本地 FSDS** | 每块 REF/LIVE → 分数 + **drift_type**。 |
| **策略** | 协变量/复合 → **candidate retire**；纯概念 → **retrain 驱动**，不自动删列。 |
| **服务端** | 新块加入时增量 LOGO；合并排序。 |
| **联动** | ESS 与 uplift overlap 闸门共用。 |

**经济效益 justification**

| 类型 | 说法 | 依据 |
|------|------|------|
| **直接** | 退坏组 → 下游训练/推理/存储省 | 各站 pipeline 单价 |
| **直接** | 相对集中式 FSDS 少传矩阵 | 同场景 2 Eff |
| **增量** | 多团队合规仓里 **安全退役** 更快 | 特征上线周期 |

**进一步 justify**：退役候选数 × 单特征 pipeline 成本；注入实验 FDR/命中率。

---

*PNG：`artifacts/fsds_scenario_slides/slide_*_{en|zh}.png` · 生成：`python3 plot_fsds_scenario_slides_deck.py`*
