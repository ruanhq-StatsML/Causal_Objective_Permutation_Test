# FSDS 用在哪？X / Y 是什么？哪几步有正向收益？

本文 **逐项 justify**：每个落地场景的 **对象定义**、**代码步骤**、**哪一步产生可核对正向收益**（算力 OPEX、治疗/曝光 OPEX、成功率、通信成本、战略收入保全），以及 **哪一步只做门禁、不直接进账**。

复现数字：`cd Python && python3 run_fsds_business_impact_pack.py`。

---

## 0. 符号（全 repo 统一）

| 符号 | 含义 | 常见误读 |
|------|------|----------|
| **X** | 喂给 **监控/归因** 的协变量（用户特征、agent trace 标量、联邦 block 列） | 不是 treatment |
| **Y** | **结果**（转化、任务 success、违约/DPD） | 用于 AUUC、DRPerm、MC 评估 |
| **T** | **处理臂**（营销 treatment、提额 arm） | **不是** 批次标签 |
| **W** | **批次**：REF 窗 vs LIVE 窗（FSDS 要测的 shift） | $W=0$ REF，$W=1$ LIVE |
| **Z** | Agent **段/角色 embedding**（SoT 预算用 Z，batch RF 多在 X 上） | 与 X 分工见 Track B |

**正向收益** 在本文分四类：**① OPEX 变量成本↓** **② 结果质量↑（success/转化）** **③  infra/通信↓** **④ 战略（避免错误关停）**。门禁步骤（G0/G2）的收益是 **避免负向决策**，通常 **不单独记美元**。

---

## Track A — 两批 Uplift（营销 / 信贷履约主路径）

### 用在哪

- 同一 uplift 模型 **REF 窗训练、LIVE 窗冻结打分**，监控 **人/流量从 REF 切到 LIVE** 后是否还能按原排序投 treatment。
- 生产：**campaign 两窗**、**信贷 REF/LIVE 放款窗**（见 `fsds_treasury_credit_risk_po.tex`）、任何「模型不立刻重训、但要决定 cap/hold/relearn」的场景。

### X、Y、T、W（具体）

| 对象 | Hillstrom bench | 合成 DGP | 信贷映射（config 换单位） |
|------|-----------------|----------|---------------------------|
| **X** | `block_0…` 等客户协变量 | 注入 shift 的特征子集 |  bureau + 渠道 + 宏观标签 |
| **Y** | 是否转化 `{0,1}` | 同上 | 90+ DPD / 违约 `{0,1}` |
| **T** | 邮件 treatment vs control | 同上 | 提额/offer vs holdout |
| **W** | REF pool vs LIVE pool（行归属哪一窗） | 同上 | 旧政策窗 vs 新政策窗 |

代码：`run_uplift_subset_benchmark.py` → `run_uplift_subset_localization`（`loco_auuc/subset_localization.py`）。

### 逐步：函数 → 产出 → 是否正向收益

| 步 | 输入 | 函数 | 产出 | 正向收益？ | Justify |
|----|------|------|------|------------|---------|
| **G0** | LIVE 上 **T** | `check_srm` | SRM 是否异常 | **间接** | 坏随机化下任何 cap 都不可信；**避免负向误操作** |
| **G1** | **X** REF vs LIVE | `_rbf_mmd2` | MMD²（Hillstrom **0.00162**） | **间接** | 解释「有没有人来变了」；单独 MMD 不触发 cap 美元 |
| **G2** | **X**, **W** | GBM `ê(W|X)` + `propensity_overlap_ess` | ESS_ovlp（Hillstrom **0.990**） | **间接** | 支持集可比才允许读 AUUC；过低 → **REFRESH_REF** 避免错 cap |
| **G3** | **X,Y** + batch **W** | `DRPerm` | PO p（Hillstrom **0.727 不拒**；synthetic_concept **0.030 拒**） | **间接 + 分流** | 区分 mix shift vs **Y|X concept drift**；决定 RELEARN **资格**，不直接等于 OPEX |
| **L1a** | REF 训练，LOCO 删组 | `loco_auuc_monitor` | 哪组特征驱动 LIVE 上 AUUC 变化 | **间接** | 定位「该 hold 哪块特征」；为 L1d/R 提供证据 |
| **L1b** | **X** 分组 | `logo_mmd_groups` | LOGO-MMD drop | **间接** | 协变量 shift 定位到哪几个 block |
| **L1c** | LIVE **τ̂(X)** | `auuc_live_by_domain_quintile` | 各 quintile AUUC | **间接** | 找到 **哪个人群排序坏了** |
| **L1d** | quintile 对 | `pairwise_auuc_compare` | ΔAUUC（Q3 vs Q4 **−0.126**） | **间接→R** | 明确 **worst = domain_quintile_3** |
| **L1e** | LIVE **ê** | `auuc_live_on_overlap_support` | AUUC_ovlp vs global | **间接→R** | 窄支持上排序仍好 → **narrow targeting** 规则 |
| **L2** | LIVE slice | `pairwise_slice_uplift_compare` | Δτ̂_obs | **条件** | **仅 PO 拒** 时 REALLOCATE 用 ATE 切片；否则 MONITOR（防误调） |
| **R** | 上列进 `loc` | `business_rules_from_localization` + `attach_uplift_rule_economics` | REALLOCATE / RELEARN + **impact_receipt** | **① 直接 OPEX** | Hillstrom：cap Q3，**sim 省 $546/窗**；四场景 OPEX net **$3271**；RELEARN **$2500×4 CAPEX 不计入 OPEX** |

**正向收益集中在一处：R 步 REALLOCATE + `econ_bucket=opex`**。前面 G/L 步的价值 = **用对的统计量触发对的规则**，避免「一报警就全量重训」或「在坏支持上乱 cap」。

**与 naive baseline 的决策差（Hillstrom）：** MMD² 小 + PO 不拒 → **先 cap quintile（OPEX）**；不是立刻重训 τ̂。synthetic_concept PO 拒 → **仍先 cap**，同时 **RELEARN ticket 有资格**（试窗 A/B 见 formulation tex）。

---

## Track B — 单 Agent Skeleton-of-Thought（SoT）

### 用在哪

- **一条 episode** 多 branch（plan / retrieve / reason / …）：在 **LIVE trace 相对 REF 漂移** 时，决定 **topology 是否并行**、**各 branch token/tier/check**。
- 生产：copilot、RAG 多跳、plan-execute、ReAct（`fsds_sot/applications.py` 各 `build_*_trace`）。

### X、Y、T、W、Z

| 对象 | 内容 |
|------|------|
| **X** | `trace_features`：branch embedding 均值 **concat** 工具次数、失败、延迟等标量（`SegmentTrace.trace_features`） |
| **Y** | episode **success** `{0,1}`（DGP / MC；**不**进入 batch 分类 RF） |
| **W** | REF trace 批 vs LIVE trace 批（`covariate_attribution(X_old, X_new)`） |
| **Z** | shape `(B, d)` 各 branch skeleton embedding → **Object 2 预算** |
| **T** | （本 track **无** uplift式 treatment 臂；动作是 **预算/拓扑**，不是 T） |

代码：`FSDSSoT.fit_plan`（`fsds_sot/pipeline.py`），demo：`boss_delivery_six_points.json`。

### 逐步与正向收益

| 步 | 函数 | 产出 | 正向收益？ | 实测 |
|----|------|------|------------|------|
| 1 | `covariate_attribution(X_ref, X_live)` | MMD², domain AUC, VIMP, overlap | **间接** | 解释 shift；指导 **哪条 branch 该加预算** |
| 2 | `evaluate_decomposability(Z)` | 是否并行 cluster | **间接** | 拓扑门：错并行会 **浪费 token** |
| 3 | `allocate_branch_budgets(κ, s_b, q_b, need)` | 每 branch `L_b`, tier, checks | **① 直接** | 相对 uniform：**非关键 branch** 降 L（boss 例：plan **397** vs uniform **520** tokens） |
| 4 | `estimate_economics` vs uniform | `SoTEconomicsReport` | **①+②** | **$0.0356→$0.0236/ep（−33.6%）**；**~$0.012/ep save**；公式：`net_gain = Δsuccess×v + (baseline_var−fsds_var) − overhead` |

**正向收益 justify：** 第 3–4 步把 **可测量的 token/check 列表** 改短；不是抽象「更智能」。若 critical path 已顶满 `C_lat`，span 可能仍平，**省在 non-critical branches**（boss pack 已说明）。

---

## Track C — 多 Agent 辩论（3 role）

### 用在哪

- **pro / con / judge** 多轮辩论；LIVE 对 **con 角色 embedding** 注入 drift。
- 生产：multi-agent 评审、风控对抗辩论、合同 red-team。

### X、Y、W、Z

| 对象 | 内容 |
|------|------|
| **X** | 与 B 相同，由 `build_multi_agent_trace` 得 `trace_features` |
| **Z** | `(3, d)` 三角色 utterance embedding |
| **Y** | MC **success**（Bernoulli） |
| **W** | REF vs LIVE trace 批 |

### 逐步与正向收益

| 步 | 函数 | 正向收益 | 实测（`multi_agent_fsds_economics.json`） |
|----|------|----------|-------------------------------------------|
| 1 | `fit_agent_plan(Z, X_ref, X_live)` | **①** 每 role token/tier | FSDS **$0.01252** vs uniform **$0.02064/debate（−39.3%）** |
| 2 | MC 50 ep | **②** success | **0.66→0.68（+2pp）** |
| 3 | `debate_early_stop_round` | **①** 少轮次 | **~20%** 轮次成本（`multi_agent_debate_early_stop.json`） |

**justify：** 1=微观预算；3=宏观停轮；可叠加（receipt 分记）。**X/W 管「何时改预算」；Z 管「改哪一 role」。**

---

## Track D — LangGraph checkpoint

### 用在哪

- 每个 **graph 节点** checkpoint 一条 embedding；LIVE 在 **tool_call** 节点注入 drift。
- 生产：LangGraph / Temporal workflow **只重跑漂移节点**（`demo_fsds_langgraph_hook.py`）。

### X、Y、W、Z

| 对象 | 内容 |
|------|------|
| **X** | 5 节点 checkpoint 的 `trace_features` 堆成 REF/LIVE 批 |
| **Z** | 节点 state embedding |
| **Y** | 节点/episode success 标量（demo 里 `node_success`） |
| **W** | REF vs LIVE 窗 |

### 正向收益

| 步 | 产出 | 收益 |
|----|------|------|
| `fit_agent_plan` + `suggest_intervention` | `impact_receipt` | **~$0.01535/ep** plan 侧节省（JSON）；**增量 = 条件边跳过稳定节点**，非全图重跑（CAPEX） |

---

## Track E — 联邦 vertical block

### 用在哪

- 多 silo（legal / finance / product）**各持 X_k**，服务器 **不见行级 X**。
- 生产：跨境、事业部合规、联合建模 **只上传 block 统计**。

### X、Y、W

| 对象 | 内容 |
|------|------|
| **X_k** | 仅 k 方拥有的列（demo：0:5, 5:10, 10:15） |
| **Y** | 可选全局 outcome（legal demo 算 PO p） |
| **W** | REF vs LIVE 索引 |

### 逐步与正向收益

| 步 | 函数 | 正向收益 | 实测 |
|----|------|----------|------|
| 1 | 本地 `covariate_attribution(X_k,ref, X_k,live)` | **间接** | legal block MMD² **0.041** vs finance **0.0037** → **定向** action |
| 2 | 上传 `LocalAttributionPayload` | **③ 通信** | Spambase bench **1410×** uplink 标量 vs 中心化矩阵 |
| 3 | Server FDR merge | **④ 战略** | 避免 **全局 copilot 关停**；sim **$125k/q**（**非** token 算出来的 OPEX） |

**justify：** ③ 是可测 **字节/费**；④ 要 **合同+财务** 签才进银行 narrative。

---

## Track F — 信贷履约（同一 Track A 代码，换 X/Y 语义）

| 对象 | 信贷 |
|------|------|
| **X** | 借款人特征 + 渠道 |
| **Y** | 违约/DPD |
| **T** | 提额/marketing offer |
| **W** | 旧/新 origination 窗 |

**有正向收益的步：** 仍只有 **R / REALLOCATE（OPEX retained）**；G3 PO 拒 → **RELEARN 资格** 不是即时美元。详见 `docs/latex/fsds_treasury_credit_risk_po.tex`。

---

## 汇总：哪几步「直接」产生正向收益

| Track | 直接 OPEX / 通信 $ | 直接质量↑ | 仅门禁 / 定位 |
|-------|-------------------|-----------|----------------|
| **A Uplift** | **R: REALLOCATE** | — | G0–G3, L1a–L2 |
| **B SoT** | **步 3–4 预算** | success（MC） | 步 1–2 |
| **C Multi-agent** | **fit_agent_plan + early-stop** | +2pp success | attribution |
| **D LangGraph** | **节点级 skip** | — | attribution |
| **E Federated** | **uplink Eff** | — | 本地 MMD；战略 sim |

---

## 闭环与收据（finance 只认这一步之后）

无论哪条 track，**可进 treasury 对账的美元** 应挂在：

- `suggest_intervention(..., attach_impact_receipt=True)`（agent）
- `impact_receipts[]`（uplift / federated）

字段含义见 `docs/FSDS_BANK_CASH_RECONCILIATION_ZH.md`。

---

## 英文 formulation（LaTeX，推荐 PDF）

**完整版：** `docs/latex/fsds_xy_benefit_formulation_en.tex`（目录 + 全 track 逐步 justify + 信贷 + 收据）

```bash
cd docs/latex && pdflatex fsds_xy_benefit_formulation_en.tex
```

**紧凑版：** `docs/latex/fsds_application_xy_benefit_ledger.tex`
