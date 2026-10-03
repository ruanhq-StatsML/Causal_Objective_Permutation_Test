"""
Monitor impact receipts — shared audit JSON for agent closed-loop and uplift rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .economics import SoTEconomicsReport
from .pipeline import SoTAttributionReport, SoTPlan
from .closed_loop import LadderRung


@dataclass
class ImpactReceipt:
    episode_id: str
    pattern: str
    ref_window: str
    live_window: str
    shift_summary: Dict[str, Any]
    action: str
    kpi_before_after: Dict[str, Any]
    estimated_net_gain_usd: float
    rule_ids: List[str] = field(default_factory=list)
    generated_at_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d.get("generated_at_utc"):
            d["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        return d


def build_agent_impact_receipt(
    *,
    intervention: Dict[str, Any],
    report: SoTAttributionReport,
    econ: Optional[SoTEconomicsReport] = None,
    pattern: str = "multi_agent_debate",
    ref_window: str = "ref_batch",
    live_window: str = "live_batch",
    episode_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Attachable receipt for ops / finance from agent FSDS plan."""
    e = econ
    kpi: Dict[str, Any] = {}
    net = 0.0
    if e is not None:
        kpi = {
            "expected_value_usd": [e.baseline_expected_value_usd, e.expected_value_usd],
            "usd_per_episode": [e.baseline_total_cost_usd, e.fsds_total_cost_usd],
            "cost_savings_pct": e.cost_savings_pct,
            "latency_reduction_pct": e.latency_reduction_pct,
        }
        net = float(e.net_economic_gain_usd)

    rung = intervention.get("rung", "REALLOCATE")
    rule_ids = [f"agent_{rung.lower()}", intervention.get("type", "reallocate")]
    if intervention.get("type") == "reallocate" and "budgets" in intervention:
        top = intervention.get("top_shift_branch", 0)
        rule_ids.append(f"segment_budget_branch_{top}")

    receipt = ImpactReceipt(
        episode_id=episode_id or str(uuid4())[:12],
        pattern=pattern,
        ref_window=ref_window,
        live_window=live_window,
        shift_summary={
            "mmd2": float(report.mmd2),
            "domain_auc": float(report.domain_auc),
            "overlap_ok": bool(report.overlap_ok),
            "overlap_ess": float(report.overlap_ess),
        },
        action=str(intervention.get("type", rung)),
        kpi_before_after=kpi,
        estimated_net_gain_usd=net,
        rule_ids=rule_ids,
    )
    return receipt.to_dict()


def build_uplift_rule_receipt(
    rule_dict: Dict[str, Any],
    loc: Dict[str, Any],
    *,
    ref_window: str = "ref_uplift_window",
    live_window: str = "live_uplift_window",
) -> Dict[str, Any]:
    """One receipt per uplift business rule (cap / hold / relearn ticket)."""
    econ = rule_dict.get("evidence", {}).get("economics", {})
    diag = loc.get("diagnosis", {})
    receipt = ImpactReceipt(
        episode_id=str(uuid4())[:12],
        pattern="uplift_two_batch",
        ref_window=ref_window,
        live_window=live_window,
        shift_summary={
            "mmd2_x": loc.get("mmd2_x"),
            "auuc_live_global": loc.get("auuc_live_global"),
            "po_pvalue": loc.get("po_risk_pvalue"),
            "diagnosis": diag.get("label"),
        },
        action=str(rule_dict.get("action", "MONITOR")),
        kpi_before_after={
            "auuc_live": loc.get("auuc_live_global"),
            "estimated_saved_spend_usd": econ.get("estimated_saved_spend_usd"),
            "estimated_incremental_margin_usd": econ.get("estimated_incremental_margin_usd"),
        },
        estimated_net_gain_usd=float(econ.get("estimated_net_impact_usd", 0.0)),
        rule_ids=[f"uplift_p{rule_dict.get('priority', 5)}_{rule_dict.get('action', 'MONITOR')}"],
    )
    return receipt.to_dict()


def build_federated_block_receipt(
    block_dict: Dict[str, Any],
    global_mmd2: float,
    *,
    action_hint: str = "",
    scenario: str = "federated_vertical_blocks",
    ref_window: str = "ref_federated",
    live_window: str = "live_federated",
) -> Dict[str, Any]:
    """Per-silo receipt for enterprise compliance / GTM."""
    bid = str(block_dict.get("block_id", "block"))
    drift = str(block_dict.get("drift_type", "unknown"))
    net = 0.0
    if drift in ("feature_drift", "compound"):
        net = 15_000.0  # sim: avoided wrongful global shutdown
    elif drift == "stable":
        net = 500.0  # sim: uplink + monitor only

    receipt = ImpactReceipt(
        episode_id=f"fed_{bid}",
        pattern=scenario,
        ref_window=ref_window,
        live_window=live_window,
        shift_summary={
            "block_mmd2": block_dict.get("mmd2"),
            "block_domain_auc": block_dict.get("domain_auc"),
            "overlap_ess": block_dict.get("overlap_ess"),
            "drift_type": drift,
            "global_mmd2": global_mmd2,
        },
        action=action_hint[:120] if action_hint else f"monitor_{bid}",
        kpi_before_after={"po_risk_pvalue": block_dict.get("po_risk_pvalue")},
        estimated_net_gain_usd=float(net),
        rule_ids=[f"fed_block_{bid}", f"drift_{drift}"],
    )
    out = receipt.to_dict()
    out["simulated"] = True
    return out


def merge_receipt_into_intervention(
    intervention: Dict[str, Any],
    receipt: Dict[str, Any],
) -> Dict[str, Any]:
    out = dict(intervention)
    out["impact_receipt"] = receipt
    return out
