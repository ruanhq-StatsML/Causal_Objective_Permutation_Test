# Model Registry + refit → predict → 差值（你的逻辑）

## 先说清楚：两套软件

| 轨道 | 路径 | 干什么 |
|------|------|--------|
| **A. 你的逻辑（本仓库主实现）** | 见下表 | **Model Registry** → cross-fit **refit** → **predict** `mu_hat`/`e_hat`/`tau_hat` → 算 **live−ref 预测差** → 可选 **DRPerm permute-refit p** |
| **B. Agent 轻量 dashboard** | `Python/fsds_sot/online_pfi.py` | sklearn RF domain + **permutation_importance**（快、给 SoT 四宫格用）**没有**走 Model Registry |

之前交付包里的 heatmap 是 **B**；你要看的 **refit/predict/差值** 在 **A**。

---

## A 轨：软件在哪（本地 = clone 后的 `Python/`）

| 组件 | 文件 | 作用 |
|------|------|------|
| **Model Registry 工厂** | `Python/model_registry_class.py` | `ModelRegistry`, `default_model_registry()` → `{fit, predict}` 字典 |
| **Cross-fit refit + predict** | `Python/DRPerm.py` | 每 fold：`model_registry[model_e/m].fit` → `predict` → `mu_hat`, `e_hat` |
| **Permute-then-refit 检验** | `Python/DRPerm.py` | 对 pseudo-outcome **再 refit** `tau`，permutation **再 refit** → `p_value` |
| **R-risk / LOCO** | `Python/R_risk_loco.py`, `Python/RRPerm.py` | 同一 Registry，R-learner + LOCO VIMP |
| **Cross-fit 工具** | `Python/utils.py` | `make_folds`, `cross_nuisance_fit`（与 DRPerm 同思路） |
| **R 版** | `R/model_registry.R`, `R/DRPerm.R` | 与 Python 对齐的 Registry + DRPerm |
| **Online 窗口封装（新）** | `Python/fsds_sot/online_pfi_registry.py` | 每个 live 窗：`prediction_deltas_on_window` + `DRPerm` |
| **Demo** | `Python/demo_fsds_online_pfi_registry.py` | 输出 `delivery/online_pfi/online_pfi_registry_report.json` |

---

## 你的逻辑（公式级）

对每个窗口，合并 \(\mathcal{D}_{\mathrm{ref}}\cup\mathcal{D}_{\mathrm{live}}\)，\(W=0/1\)：

1. **Registry refit（cross-fit）**  
   \(\hat\mu(X)\), \(\hat e(X)\) ← `model_m`, `model_e` 的 `fit`/`predict`

2. **Pseudo-outcome**  
   \(\tilde Y = (Y-\hat\mu)(W-\hat e)\)

3. **再 refit → predict tau**  
   `fit(X, pseudo)` → `tau_hat(X)`（PO-risk 里的 prediction value）

4. **差值（你要看的）**  
   - \(\Delta_\mu = \overline{\hat\mu}_{\mathrm{live}} - \overline{\hat\mu}_{\mathrm{ref}}\)  
   - \(\Delta_\tau = \overline{\hat\tau}_{\mathrm{live}} - \overline{\hat\tau}_{\mathrm{ref}}\)  
   - \(\overline{\tilde Y}_{\mathrm{live}} - \overline{\tilde Y}_{\mathrm{ref}}\)

5. **可选：DRPerm**  
   permute \(W\) → **再 refit** → 比较 PO-risk statistic → `p_value`

代码入口：`fsds_sot/online_pfi_registry.prediction_deltas_on_window`

---

## 本地怎么跑（Model Registry 轨）

```bash
cd Python
python3 demo_fsds_online_pfi_registry.py
# → delivery/online_pfi/online_pfi_registry_report.json
```

依赖：numpy, scikit-learn, xgboost（与 `model_registry_class.py` 一致）。

---

## 和 `delivery/online_pfi/` 的关系

- `07_online_pfi_dashboard.png`：**B 轨**可视化（RF VIMP 热力图）  
- `online_pfi_registry_report.json`：**A 轨** refit/predict/差值 + DRPerm（跑上面 demo 生成）

两条可以并存：B 看 **哪维 feature**；A 看 **meta-learner 预测值是否拉开**（concept 平面）。
