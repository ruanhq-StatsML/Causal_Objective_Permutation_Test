"""
Map FSDS impact_receipt fields → bank-operating-account cash movements (USD).

This is not accrual P&L: each ledger row is tagged with *when cash typically
leaves or enters* the operating checking account and which counterparty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

CashDirection = Literal["cash_in", "cash_out", "retained"]  # retained = avoided outflow


@dataclass
class BankLedgerRow:
    """One explainable line for treasury / FinOps reconciliation."""

    gl_account: str
    gl_label: str
    direction: CashDirection
    amount_usd: float
    counterparty: str
    settlement_lag_days: int
    receipt_id: str
    rule_ids: List[str]
    pattern: str
    mechanism_zh: str
    mechanism_en: str
    simulated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BankCashBridgeSummary:
    """Roll-up: effect on operating-account ending balance vs counterfactual."""

    period_label: str
    cash_in_usd: float
    cash_out_avoided_usd: float  # retained: less debit than baseline
    cash_out_extra_usd: float  # FSDS overhead, CAPEX tickets
    net_retained_in_checking_usd: float
    ledger: List[BankLedgerRow] = field(default_factory=list)
    notes_zh: List[str] = field(default_factory=list)
    notes_en: List[str] = field(default_factory=list)
    generated_at_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ledger"] = [r.to_dict() if isinstance(r, BankLedgerRow) else r for r in self.ledger]
        if not d.get("generated_at_utc"):
            d["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        return d


# Minimal GL map (override in ERP import)
GL = {
    "api_cogs": ("6100", "Cloud / LLM API COGS"),
    "marketing_treat": ("6200", "Marketing & treatment fulfillment"),
    "product_revenue": ("4100", "Product / subscription cash receipts"),
    "fsds_monitor_opex": ("6310", "FSDS monitor sidecar infra"),
    "capex_ml": ("1700", "Capitalized ML retrain (RELEARN)"),
    "strategic_avoided_loss": ("4900", "Contra: avoided campaign shutdown (risk)"),
}


def _rid(receipt: Dict[str, Any]) -> str:
    return str(receipt.get("episode_id") or receipt.get("receipt_id") or "unknown")


def agent_receipt_to_ledger(
    receipt: Dict[str, Any],
    *,
    episodes_in_settlement_period: int = 1,
    simulated: bool = False,
) -> List[BankLedgerRow]:
    """
    T0 agent / LangGraph / debate receipts.

    Bank effect:
      - Lower API/check debits → *retained* cash (same checking balance higher vs baseline).
      - Success uplift → *cash_in* when customers pay (lags episodes).
      - FSDS overhead → small *cash_out* to infra vendor.
    """
    kpi = receipt.get("kpi_before_after") or {}
    rule_ids = list(receipt.get("rule_ids") or [])
    pattern = str(receipt.get("pattern", "agent"))
    rows: List[BankLedgerRow] = []

    usd_ep = kpi.get("usd_per_episode")
    if isinstance(usd_ep, (list, tuple)) and len(usd_ep) >= 2:
        per_ep_save = float(usd_ep[0]) - float(usd_ep[1])
    else:
        per_ep_save = 0.0

    compute_saved = max(0.0, per_ep_save * episodes_in_settlement_period)
    if compute_saved > 0:
        acct, label = GL["api_cogs"]
        rows.append(
            BankLedgerRow(
                gl_account=acct,
                gl_label=label,
                direction="retained",
                amount_usd=round(compute_saved, 6),
                counterparty="LLM_provider (OpenAI/Azure/Anthropic/Bedrock)",
                settlement_lag_days=7,
                receipt_id=_rid(receipt),
                rule_ids=rule_ids,
                pattern=pattern,
                mechanism_zh="同一运营户：次月 API 发票借记金额低于反事实 baseline",
                mechanism_en="Operating checking: next API invoice debit lower than baseline",
                simulated=simulated,
            )
        )

    ev = kpi.get("expected_value_usd")
    if isinstance(ev, (list, tuple)) and len(ev) >= 2:
        revenue_uplift = float(ev[1]) - float(ev[0])
        if revenue_uplift > 0:
            acct, label = GL["product_revenue"]
            rows.append(
                BankLedgerRow(
                    gl_account=acct,
                    gl_label=label,
                    direction="cash_in",
                    amount_usd=round(revenue_uplift * episodes_in_settlement_period, 6),
                    counterparty="Customer / Stripe / merchant acquirer",
                    settlement_lag_days=14,
                    receipt_id=_rid(receipt),
                    rule_ids=rule_ids,
                    pattern=pattern,
                    mechanism_zh="成功率提升 → 客户付款入账（通常晚于 API 账期）",
                    mechanism_en="Higher success → customer settlements (lags API billing)",
                    simulated=simulated,
                )
            )

    net = float(receipt.get("estimated_net_gain_usd", 0.0))
    tagged = sum(r.amount_usd for r in rows if r.direction in ("retained", "cash_in"))
    overhead = max(0.0, tagged - net) if tagged > net else 0.0005 * episodes_in_settlement_period
    if overhead > 0:
        acct, label = GL["fsds_monitor_opex"]
        rows.append(
            BankLedgerRow(
                gl_account=acct,
                gl_label=label,
                direction="cash_out",
                amount_usd=round(overhead, 6),
                counterparty="Sidecar host (K8s / observability)",
                settlement_lag_days=30,
                receipt_id=_rid(receipt),
                rule_ids=rule_ids + ["fsds_overhead"],
                pattern=pattern,
                mechanism_zh="监控 sidecar 固定小额 OPEX，冲减部分算力节省",
                mechanism_en="Monitor sidecar OPEX offsets part of compute savings",
                simulated=simulated,
            )
        )
    return rows


def uplift_receipt_to_ledger(
    receipt: Dict[str, Any],
    *,
    simulated: bool = True,
) -> List[BankLedgerRow]:
    """T1 uplift cap: marketing/treatment outflow avoided + optional margin inflow."""
    kpi = receipt.get("kpi_before_after") or {}
    rule_ids = list(receipt.get("rule_ids") or [])
    rows: List[BankLedgerRow] = []

    saved = float(kpi.get("estimated_saved_spend_usd") or 0.0)
    if saved > 0:
        acct, label = GL["marketing_treat"]
        rows.append(
            BankLedgerRow(
                gl_account=acct,
                gl_label=label,
                direction="retained",
                amount_usd=round(saved, 2),
                counterparty="Ad / promo / fulfillment vendor",
                settlement_lag_days=3,
                receipt_id=_rid(receipt),
                rule_ids=rule_ids,
                pattern="uplift_two_batch",
                mechanism_zh="封顶后少触达/少发货 → 运营户少一笔对供应商的借记",
                mechanism_en="Cap → fewer treatments billed by vendor",
                simulated=simulated,
            )
        )

    margin = float(kpi.get("estimated_incremental_margin_usd") or 0.0)
    if margin > 0:
        acct, label = GL["product_revenue"]
        rows.append(
            BankLedgerRow(
                gl_account=acct,
                gl_label=label,
                direction="cash_in",
                amount_usd=round(margin, 2),
                counterparty="Customer (incremental conversion margin)",
                settlement_lag_days=30,
                receipt_id=_rid(receipt),
                rule_ids=rule_ids,
                pattern="uplift_two_batch",
                mechanism_zh="减少低质量转化带来的净收款改善（需与财务确认是否同一户）",
                mechanism_en="Net customer receipts from better targeting (confirm with finance)",
                simulated=simulated,
            )
        )

    if str(receipt.get("action", "")).upper() == "RELEARN":
        acct, label = GL["capex_ml"]
        rows.append(
            BankLedgerRow(
                gl_account=acct,
                gl_label=label,
                direction="cash_out",
                amount_usd=2500.0,
                counterparty="GPU cloud / labeling vendor",
                settlement_lag_days=45,
                receipt_id=_rid(receipt),
                rule_ids=rule_ids + ["capex_relearn"],
                pattern="uplift_two_batch",
                mechanism_zh="重训工单批准后才从运营或 Capex 户支出 — 不是 cap 的即时省钱",
                mechanism_en="RELEARN cash out only when ticket approved — not instant OPEX save",
                simulated=simulated,
            )
        )
    return rows


def federated_receipt_to_ledger(
    receipt: Dict[str, Any],
    *,
    simulated: bool = True,
) -> List[BankLedgerRow]:
    """T2: strategic — avoided shutdown → revenue that would have been lost."""
    net = float(receipt.get("estimated_net_gain_usd", 0.0))
    if net <= 0:
        return []
    acct, label = GL["strategic_avoided_loss"]
    return [
        BankLedgerRow(
            gl_account=acct,
            gl_label=label,
            direction="cash_in",
            amount_usd=round(net, 2),
            counterparty="Customer (campaign would have stayed live)",
            settlement_lag_days=60,
            receipt_id=_rid(receipt),
            rule_ids=list(receipt.get("rule_ids") or []),
            pattern=str(receipt.get("pattern", "federated")),
            mechanism_zh="若否则整站关停：本可继续产生的客户入账（情景，需法务+财务签）",
            mechanism_en="Revenue preserved vs full shutdown (scenario; legal+finance sign-off)",
            simulated=simulated,
        )
    ]


def summarize_ledger(
    rows: List[BankLedgerRow],
    *,
    period_label: str = "settlement_period",
) -> BankCashBridgeSummary:
    cash_in = sum(r.amount_usd for r in rows if r.direction == "cash_in")
    retained = sum(r.amount_usd for r in rows if r.direction == "retained")
    cash_out = sum(r.amount_usd for r in rows if r.direction == "cash_out")
    net = cash_in + retained - cash_out
    return BankCashBridgeSummary(
        period_label=period_label,
        cash_in_usd=round(cash_in, 2),
        cash_out_avoided_usd=round(retained, 2),
        cash_out_extra_usd=round(cash_out, 2),
        net_retained_in_checking_usd=round(net, 2),
        ledger=rows,
        notes_zh=[
            "银行存款上升 = 客户入账增加 + 对供应商/API 的借记减少 − 监控/重训等额外支出。",
            "API 预充值：节省可能体现为「少补一笔充值」而非当月对账单可见，需对 prepaid 子科目。",
            "receipt 的 estimated_net_gain_usd 必须与发票/Stripe 导出逐条勾对，勾上才算入账。",
        ],
        notes_en=[
            "Higher checking balance = more customer deposits + fewer vendor/API debits − overhead/CAPEX.",
            "Prepaid API credits: savings may skip a top-up rather than show on monthly invoice.",
            "Treat estimated_net_gain_usd as hypothesis until matched to invoice/Stripe exports.",
        ],
    )


def match_vendor_invoice(
    invoice_rows: List[Dict[str, Any]],
    *,
    baseline_usd: float,
    tag_key: str = "project",
    tag_value: str = "agent",
) -> Dict[str, Any]:
    """
    Simple FinOps check: sum tagged invoice lines vs baseline.

    invoice_rows: [{amount_usd, date, vendor, project?, ...}]
    """
    actual = sum(float(r.get("amount_usd", 0)) for r in invoice_rows if r.get(tag_key) == tag_value)
    delta_retained = max(0.0, baseline_usd - actual)
    return {
        "tag": {tag_key: tag_value},
        "baseline_usd": baseline_usd,
        "actual_usd": round(actual, 2),
        "retained_in_checking_usd": round(delta_retained, 2),
        "matched": delta_retained > 0,
    }
