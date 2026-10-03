# TARNet / Dragon-Net + feature-selection sensitivity

## Architectures (``loco_auuc/neural_uplift.py``)

| Model | Heads | Loss |
|-------|-------|------|
| **TARNet** | Shared $\Phi(x)$, $\hat Y(0), \hat Y(1)$ | Factual BCE on treated/control heads |
| **Dragon-Net** | + propensity $\hat e(x)$ | + BCE on $e$ + **targeted regularization** (Shi et al.) |

Both expose ``predict_tau(X) = \sigma(\hat Y(1)) - \sigma(\hat Y(0))`` for binary $Y$.

## Feature-selection sensitivity

1. **Group LOCO** — remove block, **retrain on REF**, AUUC on REF eval & LIVE → ``drop_ref``, ``drop_live``.
2. **Spearman**(`drop_ref`, `drop_live`) — stability under batch shift.
3. **domain_auc_X** — FSDS-style separability of ref vs live on $X$.
4. **domain_auc_tau_ranking** — ref vs live separability on **$\hat\tau$ alone** (ranking calibration drift).
5. **Retention curve** — drop groups in ascending LOCO importance; plot **AUUC_live vs #features**.

## Visualization

```bash
cd Python && python3 demo_loco_auuc_neural_viz.py
```

Outputs: ``artifacts/loco_auuc_TARNet_dashboard.png``, ``..._DragonNet_...``, JSON compare.

Panels: uplift curves (REF vs LIVE), LOCO bars, retention sensitivity.

## Benchmark

Full matrix includes ``tarnet``, ``dragonnet``, Registry X-learners:

```bash
python3 run_loco_auuc_benchmark.py
```

## FSDS monitoring efficiency: neural vs meta-learner

**FSDS 在这里指同一套 L0 窗口诊断**（MMD / overlap / Online PFI 只看 **X**；排序质量看 **AUUC**；归因看 **LOCO-AUUC**）。换 DragonNet/TARNet 还是 Registry X-learner，**监控流程不变**；变的是 **每次 LOCO 循环要完整 retrain 的代价** 和 **AUUC 信号是否更稳**。

### 计算效率（LOCO 成本）

Group LOCO 对每个 block \(g\)：在 REF 上 drop \(g\) → **retrain** → 在 REF eval 与 LIVE 上重算 AUUC。实现里 baseline 与每个 group 各训练两次（ref eval / live 各一次），共 **\(2(1+G)\)** 次 fit（见 ``loco.py``）。

| 因素 | Meta-learner (Registry / sklearn) | TARNet / Dragon-Net |
|------|-----------------------------------|---------------------|
| 单次 fit | 毫秒–秒级（RF / 线性） | 秒级（× epoch × batch） |
| LOCO 周期 | 随 \(G\) 线性，常可 hourly | 同样 \(2(1+G)\) 次 **深度** retrain |
| 特征维 \(p\) 大 | 仍贵但可 group/modality LOCO | 通常更贵；必须 **block LOCO**，勿 per-feature |

**Hillstrom 实测**（5 维、2 blocks、``artifacts/loco_auuc_efficiency_hillstrom.json``）：

| Learner | AUUC\_live | LOCO 整窗 wall (s) | 单次 fit (s) |
|---------|------------|-------------------|--------------|
| Registry Logistic-X | **0.049** | **0.60** | 0.01 |
| DragonNet (fast) | 0.047 | 6.17 | 0.79 |
| sklearn X-learner | 0.015 | 2.91 | 0.42 |
| TARNet (fast) | −0.001 | 7.77 | 0.64 |

结论：**tabular 小特征** 上 Registry Logistic-X 的 **AUUC\_live 与 DragonNet 同量级**，但 LOCO 监控约 **10× 更快**。TARNet 在此数据上 **排序监控信号弱**（AUUC\_live≈0），不等于 DragonNet 一定更好——要选 **主 measurement AUUC\_live + gap** 与成本一起权衡。

### 统计效率（主 measurement = AUUC）

| 读数 | 含义 | Neural vs Meta |
|------|------|----------------|
| **AUUC\_ref / AUUC\_live** | 排序在 REF 校准 vs LIVE 部署 | 高维非线性时 DragonNet 可能 ↑；低维 tabular 常 **meta 不输** |
| **auuc\_gap** | REF−LIVE；配合 MMD(X) 区分 covariate vs concept | 与 learner **无关** 的 FSDS 规则；gap 大 + MMD 小 → concept |
| **LOCO drop\_ref / drop\_live** | 哪组特征支撑 targeting | 必须 **retrain**；frozen SHAP 不能替代 |
| **Spearman(drop\_ref, drop\_live)** | 归因排序跨窗是否稳定 | 理想 ≈1；flip → 概念/策略漂移 |
| **domain\_auc\_τ** (sensitivity) | \(\hat\tau\) 分布在 REF vs LIVE 是否可分 | 与 AUUC gap 互补（校准漂移） |

Neural 的 **targeted regularization**（Dragon-Net）有时改善 **AUUC\_ref**，但在 **随机 REF/LIVE 切分** 的 Hillstrom 上会出现 **ref 负、live 正** 的 gap——这是 **窗口噪声**，不是 FSDS 失效；应用 **固定 REF 窗 + 滚动 LIVE**。

### 推荐分工（production）

1. **生产打分**：可用 DragonNet/TARNet（若离线 uplift 验证更好）。
2. **FSDS L0 监控**：用 **同一 learner 族** 最一致；若 LOCO SLA 紧，用 **Registry Logistic/RF X** 做 **parallel monitor**，仅当 AUUC\_live / LOCO 轮廓与生产模型 **分歧** 时再触发全量 neural LOCO 或 relearn。
3. **overlap\_ess 低**：无论哪种 learner，**先 stratify REF**，再信 LOCO。

JSON 对比：``artifacts/loco_auuc_neural_compare.json``（合成块漂移 + domain\_auc\_τ）；效率表：``artifacts/loco_auuc_efficiency_hillstrom.json``。
