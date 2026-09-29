# FSDS Online PFI — 逻辑整理 & 打包交付

> **交付分支**：`cursor/fsds-sot-tot-manuscript-7451`  
> **一键复现**：`cd Python && python3 package_online_pfi_delivery.py`

---

## 0. 重要：两套实现（别混）

| 轨道 | 路径 | 是否 Model Registry + refit/predict 差值 |
|------|------|------------------------------------------|
| **你的逻辑（主仓库）** | `docs/FSDS_MODEL_REGISTRY_ONLINE_PFI.md` | **是** — `online_pfi_registry.py` + `DRPerm.py` |
| **Agent 快览 dashboard** | `fsds_sot/online_pfi.py` | **否** — sklearn RF VIMP，仅监测/feature 热力图 |

---

## 0. 你要的：**Model Registry → refit → predict → 差值**（A 轨，主逻辑）

| 步骤 | 代码 |
|------|------|
| Registry | `Python/model_registry_class.py` |
| Cross-fit refit + predict | `Python/DRPerm.py`（`mu_hat`, `e_hat`） |
| 再 refit → `tau_hat` | `DRPerm.py` / `online_pfi_registry.prediction_deltas_on_window` |
| **预测差** Δμ, Δτ | `delta_mu_mean`, `delta_tau_mean` in JSON |
| Permute-refit p | `DRPerm(..., n_perm=...)` → `p_value` |
| Online 窗口 | `Python/fsds_sot/online_pfi_registry.py` |
| **跑结果** | `python3 demo_fsds_online_pfi_registry.py` → `delivery/online_pfi/online_pfi_registry_report.json` |

`fsds_sot/online_pfi.py` 只是 **B 轨** sklearn RF-VIMP dashboard，**不走** Model Registry。

---

## 1. 你给的逻辑（整理成可执行栈）

**目标**：在 **连续 agent 流量** 上，不用等离线大 batch，就能 **监测 drift → 归因到特征维 → 触发调度**。

| 层 | 做什么 | 实现 |
|----|--------|------|
| **L0 特征** | 每个 episode 的 trace 向量 \(\phi(\mathcal{S})\)（embedding 均值 + tool/latency 标量） | `agentic_dgp.py` → `trace_features` |
| **L1 双样本** | Reference \(\mathcal{D}_{\mathrm{old}}\) vs Live \(\mathcal{D}_t\) | 窗口 \(t=0,1,\ldots\) |
| **L2 Domain + Online PFI** | RF 分 domain；**permutation VIMP** = 在线 PFI \(\phi_t\) | `attribution.py` / `online_pfi.py` |
| **L3 距离 & 触发** | MMD²、AUC；**onlineRFPerm** 标签 permutation → **p-value** | `online_pfi._rf_perm_pvalue` |
| **L4 Warm baseline** | \(\mathcal{D}_{\mathrm{old}} \leftarrow \alpha \mathcal{D}_{\mathrm{old}}+(1-\alpha)\mathcal{D}_t\) | `closed_loop.decay_reference` |
| **L5 动作** | p 或 VIMP 热点 → SoT **Object 2**（\(L_b\)/tier/checks）或闭环 ladder | `pipeline.py` / `closed_loop.py` |

**一句话**：**Online PFI = 每个窗口做一次 batch FSDS 的 covariate 平面，并把 VIMP 串成时间序列；p-value 是同一套 C2ST 的漂移开关。**

---

## 2. 代码清单（Manifest）

| 路径 | 角色 |
|------|------|
| `Python/fsds_sot/online_pfi.py` | **核心**：`run_online_pfi_stream` |
| `Python/fsds_sot/plot_online_pfi.py` | **可视化**：四宫格 dashboard |
| `Python/fsds_sot/attribution.py` | 单窗 PFI + MMD-LOCO + overlap |
| `Python/fsds_sot/viz.py` | SoT 四宫格 Panel 4（单窗 VIMP 条） |
| `Python/fsds_sot/closed_loop.py` | 归因后 **measure→act→re-measure** + reference decay |
| `Python/demo_fsds_online_pfi_viz.py` | Agent 窗口 demo |
| `Python/package_online_pfi_delivery.py` | **打包脚本**（本交付） |
| `docs/FSDS_ONLINE_PFI.md` | 简明索引 |
| `FSDS_online_PFI_delivery.tex` | LaTeX 交付节 |
| `delivery/online_pfi/` | 打包输出目录（图 + JSON + manifest） |

**历史 RAP 实现（同思想，不同特征 `[e,H,|H|]`）**：git commit `9a6149a` → `online_drift_detectors.py`。

---

## 3. 算法（与代码一致）

```
Input: X_ref, live_windows[0..T-1], α, α_trigger=0.05
For t = 0 .. T-1:
  cov ← covariate_attribution(X_ref, live_windows[t])
  φ_t ← cov.vimp                    # Online PFI
  p_t ← RF_perm_pvalue(stack, labels)
  log AUC_t, MMD²_t, overlap_t, top_k(φ_t)
  if p_t < α_trigger: emit DRIFT_SIGNAL(t)
  X_ref ← α·X_ref + (1-α)·live_windows[t]   # optional
Output: OnlinePFIReport + dashboard PNG
```

---

## 4. 可视化读法（交付图）

**文件**：`delivery/online_pfi/07_online_pfi_dashboard.png`

| 子图 | 读法 |
|------|------|
| Domain AUC | →1：live 与 ref 可分，有 covariate shift |
| MMD² | 越大：整体分布越远 |
| onlineRFPerm p | **&lt;0.05**：触发 reroute / 闭环 ladder |
| PFI heatmap | **哪几维在推 drift**（如 `tool_fail`, `latency`）→ 改 budget / tool / RAG |

---

## 5. 与 SoT 两对象的关系

- **Object 0（监测）**：Online PFI 流 + Panel 4 快照 = 同一 VIMP，不同时间粒度。
- **Object 1（拓扑）**：embedding 块在 heatmap 亮 → 检查分解度 / 是否仍宜并行。
- **Object 2（预算）**：trace 标量 + shift 枝 → \(L_b\)、checks、tier（见 boss 六点 / `pipeline.fit_plan`）。

---

## 6. 验收命令

```bash
cd Python
python3 package_online_pfi_delivery.py
ls -la ../delivery/online_pfi/
```

期望：`07_online_pfi_dashboard.png`、`online_pfi_report.json`、`MANIFEST.json` 存在。

---

## 7. 相关交付（同分支）

| 包 | 文档 |
|----|------|
| SoT + Agent 六点 | `docs/BOSS_DELIVERY_6POINTS.md` |
| Agent 应用 & 闭环 | `FSDS_agent_reasoning_applications.tex` |
| 完整 writeup | `FSDS_SoT_complete_writeup.tex` |
