# FSDS 里 embedding 逻辑是什么？（具体要求）

Repo 里 **不是**「用一个 embedding 做所有事」，而是 **两层向量、两种用途**。搞清这个，ReAct / multi-agent 才接得对。

---

## 1. 两层对象：`Z`（段内）和 `X`（批间）

| 符号 | shape | 谁用 | 干什么 |
|------|--------|------|--------|
| **`Z`（segment embeddings）** | `(B, d)` | `fit_plan` → `allocate_branch_budgets` | **本 episode 内** B 个段/角色/ hop/节点；算 **shift_score**、token/tier/check |
| **`X`（trace_features）** | `(p,)` 每条 trace，堆成 `(n, p)` 两批 | `covariate_attribution(X_ref, X_live)` | **REF 窗 vs LIVE 窗** 批间漂移；MMD²、domain AUC、VIMP |

代码入口：`SegmentTrace`（`applications.py`）→ `fit_agent_plan(..., X_old, X_new, segment.embeddings)`（`pipeline.py` 的 `fit_plan`）。

**要求（生产）：**

1. **每个 episode 必须产出一份 `Z` 和一份 `trace_features`（拼进 `X` 批）。**
2. **`X_ref` / `X_live` 是 many episodes 的 trace 矩阵，不是单条的 `Z`。**
3. **`Y`（success/转化）不喂给 batch RF**；只用于 MC/业务评估（与 uplift 里 DRPerm 的 `Y` 不同 track）。

---

## 2. `Z` 怎么来：段 embedding + 行归一化

所有 pattern 共用：

```python
# applications.py — _normalize_rows
Z_row = Z_row / ||Z_row||_2   # 每个 segment/role/hop 一行，L2 归一化
```

**语义：** 用 **方向**（主题/语义）比绝对模长；段与段之间可比 shift。

### ReAct（3 行 `Z`）

| 行 | 含义 | 典型生产来源 |
|----|------|----------------|
| 0 | thought | 该步 planner 最后一层 hidden mean，或 summary embedding |
| 1 | action | tool name + args 的 embedding |
| 2 | observation | API/检索返回文本的 embedding |

`build_react_trace` → `Z.shape == (3, d)`，`quality` 可来自 `tool_ok` 等。

### Multi-agent（3 行 `Z`）

| 行 | 角色 | LIVE 漂移 demo |
|----|------|----------------|
| 0 | pro | 通常稳定 |
| 1 | **con** | **故意加噪** → shift_score 往往次高 |
| 2 | judge | 常是 critical path（need 大） |

`build_multi_agent_trace(role_names, utterance_embs)`：**每角色一条 utterance embedding**。

### 其他 pattern（同一套 `Z`）

- **RAG：** 每 hop 一行（chunk 聚合向量）
- **LangGraph：** 每个 checkpoint 节点一行
- **Self-consistency：** 每条 chain 样本一行
- **Plan-execute：** 每个 plan step 一行

---

## 3. `trace_features`（拼成批间 `X`）怎么构造

**原则：** 把 **整条 episode 的 `Z` 压成一条固定长度向量**，再 concat **少量标量**，供 REF/LIVE 分类器用。

**ReAct 示例（代码）：**

```text
trace_features = concat( mean(Z, axis=0),   # d 维
                         [n_segments, mean(q), min(q)] )   # 3 标量
→ p = d + 3
```

**Multi-agent 示例：**

```text
trace_features = concat( mean(Z, axis=0), [n_roles, std(Z), 0] )
```

**要求（生产）：**

| # | 要求 |
|---|------|
| E1 | **同一 pattern 下 `p` 固定**；REF/LIVE 窗内 schema 不变 |
| E2 | **mean(Z)** 建议与 sidecar 用的 encoder **同一模型、同一 pooling** |
| E3 | 标量建议含：**段数、工具失败率、延迟、min(quality)** 等（DGP 里在 `trace_features` 尾部） |
| E4 | LIVE 窗写入前 **不要** 用 LIVE 统计去归一化 REF（避免泄漏）；两窗独立堆批 |

批间监控：`covariate_attribution(X_ref, X_live)` → MMD²、overlap、RF 预测 **W∈{REF,LIVE}** 的 AUC。

---

## 4. 本 episode 内：`Z` → shift_score → 预算（Object 2）

在 **不重新 fit 批间模型** 时也可只跑 `fit_plan_branches(Z)`：

```text
ref = mean(Z, axis=0)
shift_score_b = ||Z_b - ref||_2 / max_b ||Z_b - ref||_2   # 归一化到 [0,1]
```

然后（`budget.py`）：

```text
L_b = clip( κ · s_b · q_b · need_b , L_min , C_lat )
tier_b = f(shift_b, q_b)   # 高 shift + 低 quality → 更大 tier/check
```

- **`q_b`：** 段质量（tool_ok、role_quality、node_success）
- **`need_b`：** 业务关键度（judge、observation 常更大）

**ReAct 实用点：** observation 行 drift 大 → **只加 obs 的 check/tier**，thought/action 的 `L_b` 可降。

**Multi-agent 实用点：** con 漂移 → **con 改 small tier、减 L**；judge 保持 `L_min` 地板（demo `min_tokens_by_branch`）。

这与 **批间 `X` 的结论一致但粒度更细**：批间说「LIVE 变了」，段间说「变在哪一段/角色」。

---

## 5. Multi-agent 协同：两套 embedding 形状（别混）

| 用途 | 张量 | 逻辑 |
|------|------|------|
| **FSDS 预算** | `Z` shape `(3, d)` **最后一轮**（或当前轮）各 role | `fit_agent_plan` → 表 R10b 那种 per-role `L`/tier |
| **Early-stop** | `round_role_embs` shape **`(R, B, d)`** | 每轮对 role 取 centroid，算 **轮间 dispersion**；≤ target 停辩 |

Early-stop **不用**替换 `Z`；它是 **macro 时间维** 上的协同信号。Stacked = micro（`Z` 预算）+ macro（`R,B,d` 停轮）。

**要求：**

- 辩论日志：**每轮、每 role 至少一条 embedding**（与 FSDS 同 encoder 维数 `d`）。
- 预算 sidecar：用 **拟执行下一轮前** 的 role `Z` + 滚动 `X_ref/X_live`。

---

## 6. 与 uplift track 的 `X` 对比（避免概念串台）

| | Agent FSDS | Uplift 两批 |
|--|------------|-------------|
| **X** | `trace_features`（agent 轨迹） | 客户协变量 block |
| **Y** | success（MC/日志） | 转化/违约 |
| **W** | REF/LIVE **trace 窗** | REF/LIVE **campaign 窗** |
| **T** | 无（动作为 budget） | treatment arm |

Uplift 的 **ranking** 在 **τ̂(X_customer)**；agent 的 **ranking** 在 **segment shift + need**，不是同一向量。

---

## 7. 生产接入 checklist（embedding）

1. **Encoder：** 固定 `d`（demo 24/32；生产可用 e5-small 等）；REF/LIVE **同一模型**。
2. **每 episode：** 写 `Z`（按 pattern 行序命名 segment）、`trace_features`、`success`。
3. **每窗：** 滚动 REF/LIVE 矩阵 `X_ref`, `X_live`（行=episode）。
4. **Sidecar：** `fit_agent_plan(segment, X_ref, X_live)` → budgets + `impact_receipt`。
5. **辩论：** 另写 `(R,B,d)` 给 `debate_early_stop_round`。
6. **验收：** 人工 inject 漂移段（如只改 obs/con）→ 对应行 `shift_score` 最大；批间 MMD 在 LIVE 窗升高。

---

## 8. 代码锚点

| 步骤 | 文件 / 函数 |
|------|-------------|
| 建 `Z`, trace | `fsds_sot/applications.py` — `build_react_trace`, `build_multi_agent_trace`, … |
| 批间归因 | `fsds_sot/attribution.py` — `covariate_attribution` |
| 段间预算 | `fsds_sot/pipeline.py` — `fit_plan` / `fit_plan_branches` |
| 协同停辩 | `debate_inter_round_dispersion`, `debate_early_stop_round` |
| MC 对比 | `demo_fsds_agent_collaboration_suite.py` |

英文 spec：`docs/latex/fsds_embedding_logic_en.tex`。
