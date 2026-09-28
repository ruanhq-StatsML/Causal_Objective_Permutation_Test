# 老板交付 — FSDS Agent SoT 六点说明

> 生成时间 (UTC): 2026-09-28T09:30:34.819467+00:00  
> 复现: `cd Python && python3 demo_fsds_boss_six_points.py`  
> 完整 JSON: `artifacts/boss_delivery_six_points.json`

---

## 第一点：要解决什么、交付哪两个对象

- **问题**：Agent live 流量相对 reference **分布漂移**（retrieve/act embedding、tool 失败率），均匀 SoT **每枝同 L、同 tier、固定 2 次 check** → 算力浪费在稳定枝、该加预算的枝反而不够。
- **对象 1（拓扑）**：分解度门控 → 建议并行簇数 \(K\)、merge/顺序（弱耦合才并行）。
- **对象 2（预算）**：**Critical-path 执行器** → 每枝 \(L_b\)、model tier、check 次数，由 **shift × quality × need** 驱动。
- **对象 0（监测）**：batch 上 RF/MMD 归因，解释「为什么今天要多花/少花」。

---

## 第二点：Critical-path 预算怎么算（为什么延迟有时几乎不变）

**公式（与代码一致）**：

\\[
L_b = \\mathrm{clip}\\big(\\kappa \\cdot s_b \\cdot q_b \\cdot need_b,\\, L_{\\min},\\, C_{\\mathrm{lat}}\\big)
\\]

- \(s_b\)：该 skeleton 枝 embedding 相对本 episode 的 shift 分数（归一化）。
- \(q_b\)：PRM / tool 质量 proxy。
- \(need_b\)：业务先验（retrieve/act = 1.0，plan = 0.35 等）。
- **Critical path（并行 SoT）**：\(T_\\infty = \\max_b L_b + t_{\\mathrm{skel}} + t_{\\mathrm{merge}}\) — **只由最长枝决定**，不是 token 总和。

**单 episode 示例（5 枝 agent）**：

| 指标 | Uniform | FSDS |
|------|---------|------|
| Span (max L) | 520 | 520 |
| Total tokens (Σ L) | 2600 | 2306 |

**要点**：若 drift 把某一枝推到 **\(C_{\\mathrm{lat}}=520\)**，span 与 uniform 几乎一样；**省钱主要在「非关键枝缩短 + tier 分流」**，不是 myth 的「每条枝都变短」。

---

## 第三点：成本与 total tokens（汇报用两个数）

**推荐配置**：`kappa=0.9`，`entropy_lambda=0.05`，retrieve/act **L_floor=320**（与 Pareto 一致）。

| 指标 | Uniform | FSDS (eval 均值) | 降幅 |
|------|---------|------------------|------|
| **$/episode** | $0.0356 | $0.0236 | **≈ 33.6%** |
| **Σ tokens** | 2600 | 2299 | **≈ 11.6%** |
| **Checks** | 10 | 5 | **≈ 50%** |

- **单 episode 示例** Σ tokens：2600 → 2306（**11.3%**）；entropy 正则后 eval 均值 token 降幅通常 **~10–12%**，**$/ep 仍可达 ~33%**（tier 下调 + checks 减半）。
- 机制：低 \(s\\times q\) 枝缩短；高 need 枝 floor；\\(w=(1-\\lambda)\\mathrm{softmax}+\\lambda/B\\) 避免 budget 坍缩。

| Branch | Uniform L | FSDS L |
|--------|-----------|--------|
| plan | 520 | 397 (medium) |
| retrieve | 520 | 520 (medium) |
| reason | 520 | 520 (small) |
| act | 520 | 520 (large) |
| verify | 520 | 349 (small) |

---

## 第四点：Checks 减半 — 机制与数字

- **Uniform**：每枝 **2** 次 verify → 5 枝 **10** 次/episode（eval 均值 **10.0**）。
- **FSDS**：\(\mathrm{checks}_b = \mathrm{round}(1 + 2\cdot s_b(1-q_b))\in[1,5]\) — shift 高且 quality 低才加 check。
- Eval 均值 **5.0** 次 → **约减半**；示例 episode **10 → 5**。

**业务含义**：验证算力跟着 **concept-risk** 走，不是每枝固定 2 次；稳定 plan/verify 枝可降到 1。

---

## 第五点：Pareto POC（成本 vs 成功率 + KPI 非劣 guard）

- **Guard**：相对 uniform，**success 下降 ≤ 3%** 的配置才进推荐集（`kpi_pass`）。
- **扫描**：kappa × retrieve/act **L_floor**（保证高 need 枝不被削过头）。
- **Uniform 基线**：success **0.700**，cost **$0.0356**/ep。
- **推荐配置**：`fsds_k0.85_floor0` — success **0.700** (Δ **+0.000**)，cost ↓ **33.7%**，tokens ↓ **8.7%**，checks ↓ **50.0%**，KPI **PASS**。
- **Pareto 前沿标签数**：5；**KPI 通过配置数**：17。

（细节见 JSON `six_points.5_pareto_poc.points`。）

---

## 第六点：交付物、闭环与下一步

| 交付 | 路径 |
|------|------|
| 六点 JSON | `artifacts/boss_delivery_six_points.json` |
| Agent 应用 + 闭环 LaTeX | `FSDS_agent_reasoning_applications.tex` |
| 完整 writeup | `FSDS_SoT_complete_writeup.tex` |
| Pareto / 增量 demo | `demo_fsds_pareto_poc.py`, `demo_fsds_boss_six_points.py` |
| PR | #12 |

**下一步（上线 POC）**：① 用真实工单解决率/tool_ok 替换 simulator；② canary 仅当 KPI guard pass 才 adopt 预算；③ 高分解 SoT / ReAct 场景单独报 span 降幅。

**已知限制**：当前 success 来自透明 simulator；span 在「单枝顶满 cap」时不下降 — 汇报时需 **同时报 span 与 Σ tokens**。

---

## 10:00 汇报一句话

**在 KPI 非劣前提下，FSDS-SoT 将 agentic 合成 batch 的 $/episode 降低约 34%，checks 减半，success 与 uniform 持平；延迟由 critical path 决定，需同时报 span 与 Σ tokens。**

*Last refresh: timer iteration — see `generated_at_utc` in JSON.*
