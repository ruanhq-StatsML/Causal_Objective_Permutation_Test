# FSDS — 哪些场景「今晚落地」有直接经济效益

前提：`docs/latex/fsds_concrete_pipeline_justification.tex` 已把 **X/Y/W/T、逐步算法、JSON 实测增量** 写死。本文只回答：**不要白活 — 今晚接什么能立刻反映在账单或治疗 spend 上**。

机器可读清单：`artifacts/fsds_tonight_landing_checklist.json`（`python3 export_fsds_tonight_landing.py` 生成）。

---

## 一张表：直接 vs 间接

| 层级 | 场景 | 今晚能省钱？ | 钱从哪来 | 典型量级（demo 单位） | 证据 |
|------|------|-------------|----------|----------------------|------|
| **T0** | Agent SoT + `impact_receipt` | **是** | LLM token / tier / check | ~**$0.012/ep**，~34% $/ep | `boss_delivery_six_points.json` |
| **T0** | 多 agent 辩论 FSDS 路由 | **是** | 同上 | ~**$0.0081/debate**，~39% | `multi_agent_fsds_economics.json` |
| **T0** | 辩论 early-stop | **是** | 少跑轮次 | ~**$0.0088/debate**，~20% 轮次 | `multi_agent_debate_early_stop.json` |
| **T0** | LangGraph checkpoint | **是** | 只重跑漂移节点 | 与 SoT 同构 receipt | `langgraph_fsds_hook.json` |
| **T1** | Uplift 两批 **REALLOCATE cap** | **是（需真实 config）** | 治疗/触达 spend 封顶 | bench 均 ~**$818/窗**（sim）；4 场景合计 OPEX **$3271** | `uplift_two_layer_benchmark.json` |
| **T2** | 联邦 legal block | **间接** | 避免整站关停 → 季度收入解锁（sim **$125k/q**） | 今晚可接 **收据**，不是今晚算力 OPEX | `federated_legal_agent_blocks.json` |

**今晚不算直接 OPEX 的：** 默认路径上的 **RELEARN 全量重训**（CAPEX 工单，bench 里 **$2500×4** 单独 bucket，不进 OPEX net）。

---

## T0 — 今晚 2–4 小时（已有 API 账单或 trace）

共性：不改训练栈；**REF/LIVE 滚动窗** + sidecar 出 **`impact_receipt`**，FinOps 可对账。

### 1) 单 agent SoT

- **接法：** 每 episode 写 `trace_features`；sidecar `fit_agent_plan` → `suggest_intervention(..., attach_impact_receipt=True)`（见 `fsds_sot/closed_loop.py`）。
- **验收：**
  ```bash
  cd Python && python3 demo_fsds_multi_agent_economics.py  # 同 economics 栈
  ```
- **24h 指标：** `sum(estimated_net_gain_usd)`；`usd_per_episode` 前后对比。
- **填规模：** 编辑 `config/fsds_economics_assumptions.json` → `traffic.*.agent_episodes_per_month`；年化见 executive summary。

### 2) 多 agent 辩论 + early-stop（建议一起上）

- **接法：** 存每轮 role embedding；`fit_agent_plan` + `debate_early_stop_round`（见 `FSDS_DEBATE_EARLY_STOP.md`）。
- **验收：**
  ```bash
  cd Python && python3 demo_fsds_multi_agent_economics.py
  cd Python && python3 demo_fsds_multi_agent_debate_early_stop.py
  ```
- **24h 指标：** 辩论平均 cost ↓；early-stop 触发率；receipt 覆盖率。

### 3) LangGraph

- **接法：** checkpoint `{node_id, embedding, W}` → `build_langgraph_node_trace` → 读 `intervention.impact_receipt` 做 conditional edge（`FSDS_LANGGRAPH_INTEGRATION.md`）。
- **验收：** `python3 demo_fsds_langgraph_hook.py`

---

## T1 — 今晚第一窗 uplift（要真实商业参数）

1. **改 config（必须）：** `margin_usd_per_conversion`、`treatment_cost_usd`、`treat_rate` → `config/fsds_economics_assumptions.json`。
2. **导出 REF/LIVE：** 行级 `X, Y, T` + 窗标签（与 bench 相同契约，见 concrete pipeline Track A）。
3. **跑：**
   ```bash
   cd Python && python3 run_uplift_subset_benchmark.py --n-perm 32
   # 或接你们自己的 CSV 路径（同 loco 管线）
   ```
4. **只执行 OPEX 规则：** `business_rules` 里 `econ_bucket=opex` 的 cap；**RELEARN** 只开 CAPEX 工单，不当作今晚 ROI。
5. **24h 指标：** 已应用 rule_id 数；真实 treat volume × cap 带来的 spend delta（receipt 对齐 finance）。

---

## T2 — 今晚可接监控，经济账在「不停全站」

- **验收：** `python3 demo_federated_legal_agent_blocks.py`
- **今晚价值：** block 级 **impact_receipt** + 定向暂停；季度 unlock 用 `strategic_sim.revenue_unlock_usd_quarter` 做情景，不是算力条。

---

## 一条命令刷新全部证据 + 清单

```bash
cd Python
python3 run_fsds_business_impact_pack.py   # 含 synthesize executive summary
python3 export_fsds_tonight_landing.py
```

---

## 建议今晚执行顺序（最短路径到美元）

1. **T0-2 + T0-3**（辩论路由 + early-stop）— 若已有 multi-agent 流量，账单最快下降。  
2. **T0-1**（SoT）— 单 agent / copilot 同样接 receipt。  
3. **T1** — 若有营销两批数据，改 config 后跑一窗 REALLOCATE。  
4. **T2** — 并行走收据，不挡 1–3。

英文 mirror：`docs/FSDS_TONIGHT_LANDING_ECONOMICS_EN.md`。
