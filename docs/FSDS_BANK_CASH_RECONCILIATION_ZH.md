# FSDS → 银行账户：怎么对到「存款变多」

不是 PPT 上的「节省 %」，而是 **运营 checking 户** 在某一结算周期里，相对「若不接 FSDS」**期末余额多了多少**。代码：`Python/fsds_sot/bank_cash_bridge.py` + `python3 reconcile_fsds_bank_cash.py` → `artifacts/fsds_bank_cash_bridge.json`。

---

## 1. 一个公式（Treasury 能用的）

设同一运营银行账户，周期 \(t\)：

\[
\underbrace{B_t - B_t^{\mathrm{cf}}}_{\text{多出来的存款}}
\approx
\underbrace{\Delta R_t}_{\text{客户入账}}
+
\underbrace{\Delta V_t}_{\text{少付供应商/API}}
-
\underbrace{H_t}_{\text{监控等额外支出}}
-
\underbrace{C_t^{\mathrm{relearn}}}_{\text{批准后的重训现金}}
\]

| 符号 | 含义 | FSDS 从哪来 |
|------|------|-------------|
| \(B_t^{\mathrm{cf}}\) | 反事实余额（不接策略时的预测） | 历史 invoice + 流量外推 |
| \(\Delta V_t\) | **retained**：该付的 API/营销借记变少 | T0 算力、T1 cap |
| \(\Delta R_t\) | **cash_in**：Stripe/客户多收 | agent 成功率↑、uplift 边际 margin |
| \(H_t\) | sidecar 云资源账单 | `DEFAULT_FSDS_OVERHEAD_USD` 量级 |
| \(C_t\) | RELEARN 工单 | `econ_bucket=capex_ticket`，**不进** OPEX net |

**重要：** 省 API 费 **不会** 凭空从银行外打一笔钱进来；表现是 **同一张运营户，对 OpenAI/广告商的借记变小**，或 **预充值户少补一笔**。

---

## 2. 三条业务线 → 谁从户头扣钱 / 谁进账

### T0 — Agent / 辩论 / LangGraph（LLM API）

| 步骤 | 生产事实 | 银行侧对手方 | 何时动账 |
|------|----------|--------------|----------|
| 1 | Scheduler 按 `intervention.budgets` 降 tier/token 或 early-stop 少轮 | **LLM 云厂商**（OpenAI/Azure/Anthropic…） | 月结 invoice 或 **预充值扣减**（T+7 常见） |
| 2 | `impact_receipt.kpi_before_after.usd_per_episode` = `[baseline, fsds]` | 同上 | 对账：`episodes × (baseline−fsds)` ≈ invoice Δ |
| 3 | 成功率提升 `expected_value_usd[1]−[0]` | **客户 / Stripe** | 通常 **晚于** API 账期（T+14~30） |

**Receipt → 分录（GL 示例）：**

- `6100` API COGS：**direction=retained**，amount = \(\sum_e (\text{cost}_0-\text{cost}_1)\)
- `4100` 收入：**direction=cash_in**，amount = \(\sum_e (EV_1-EV_0)\)
- `6310` 监控：**direction=cash_out**，amount ≈ `fsds_overhead × episodes`

Demo 单位（config）：SoT ~**$0.012/ep** retained；辩论 FSDS ~**$0.0081/debate**；early-stop ~**$0.0088/debate**。

**FinOps 勾对（今晚就能做）：**

1. 导出厂商账单 CSV：`date, vendor, amount_usd, project=agent`
2. 定 baseline = 接 FSDS **前** 同流量周/API  spend
3. `retained = baseline − actual`（见 `match_vendor_invoice()`）
4. 与 `sum(impact_receipt.estimated_net_gain_usd)` 比，误差 >5% 则查未打 tag 的流量

### T1 — Uplift REALLOCATE cap（营销/治疗 spend）

| Receipt 字段 | 银行含义 | GL |
|--------------|----------|-----|
| `estimated_saved_spend_usd` | 少触达 → **少付** 广告/履约商 | `6200` **retained** |
| `estimated_incremental_margin_usd` | 更好 targeting 的 **客户净收款**（需财务确认） | `4100` **cash_in** |
| `econ_bucket=capex_ticket` | 重训 GPU/标注 **现金流出** | `1700` **cash_out**（批准日） |

公式（与 `attach_uplift_rule_economics` 一致）：

\[
\text{saved} \approx n_{\mathrm{cap}} \times \text{treat\_rate} \times \text{treatment\_cost\_usd}
\]

**必须改真实数：** `config/fsds_economics_assumptions.json` 里 `margin_usd_per_conversion`、`treatment_cost_usd`、`treat_rate` — 否则只是 sim，**不能**对银行。

动账通常 **比 API 快**（广告/短信 T+1~3）。

### T2 — 联邦 block（战略）

`estimated_net_gain_usd` 在 demo 里是 **「若整站关停则收不到的款」** → 记 **`4900` 情景 contra**，**不是** T0 那种 invoice 级 retained。要上银行预测，需 **法务+财务** 签「定向暂停 vs 全站关」的收入保全假设。

---

## 3. 预充值 vs 月结（很多人在这里对不上）

| 付款模式 | FSDS 节省在账上长什么样 |
|----------|---------------------------|
| **月结 invoice** | 下个月运营户 **借记变小** → 期末余额 ↑ |
| **预充值 credits** | 余额表 prepaid 消耗变慢；**直到少买一笔充值**，checking 才明显少流出 |
| **Stripe 扣款** | 按成功事件入账；agent 成功率↑ 直接体现在 **cash_in** |

Treasury 规则：**retained 与 cash_in 分开列**，不要混成一个「ROI 数字」去对银行。

---

## 4. `impact_receipt` → ledger 行（机器可读）

每条 receipt 可展开为多行 `BankLedgerRow`：

```json
{
  "gl_account": "6100",
  "direction": "retained",
  "amount_usd": 120.0,
  "counterparty": "LLM_provider",
  "settlement_lag_days": 7,
  "receipt_id": "...",
  "rule_ids": ["agent_reallocate", "segment_budget_branch_2"]
}
```

生成：

```bash
cd Python
python3 reconcile_fsds_bank_cash.py --episodes 50000
# 有真实发票时：
python3 reconcile_fsds_bank_cash.py --invoice-csv ../data/openai_invoices.csv --baseline-api-usd 42000
```

---

## 5. 30 天「入账验收」清单（不白活）

| 天 | 动作 | 银行/ERP 证据 |
|----|------|----------------|
| D0 | T0 接 receipt 日志 + 策略生效 | — |
| D1–7 | 流量稳定 | Cloud invoice 或 prepaid 消耗曲线 |
| D7 | FinOps | `baseline − actual` = retained，贴 `rule_id` |
| D14 | 若看成功率 | Stripe 净入账 vs 反事实 |
| D30 | 关账 | \(\Delta B\) 与 `fsds_bank_cash_bridge.json` 汇总行一致（± agreed tolerance） |

**不算进存款的（别忽悠自己）：**

- bench 里 `simulated: true` 且未换 config 的 uplift 数字
- RELEARN **ticket 未批准** 前的「 avoided retrain 叙事」
- 联邦 **$125k/q** 未签约的情景收入

---

## 6. 与 repo 公式对齐（agent 单条 episode）

`fsds_sot/economics.py`：

```text
net_gain = (p − p0) × v_succ + (baseline_var − fsds_var) − overhead
           └─ cash_in ─┘   └────── retained (API) ──────┘   └ cash_out ┘
```

`impact_receipt.estimated_net_gain_usd` 应 **拆成** 上三项再对银行；不要只用一个总数对 invoice。

---

英文：`docs/FSDS_BANK_CASH_RECONCILIATION_EN.md`（若存在）或见 JSON `notes_en`。
