"""Summarize simulated uplift rule economics (OPEX vs CAPEX tickets)."""

from __future__ import annotations

from typing import Any, Dict, List


def summarize_rule_economics(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    opex_net = 0.0
    saved = 0.0
    capex_tickets = 0.0
    n_opex = n_capex = 0
    for br in rules:
        econ = (br.get("evidence") or {}).get("economics") or {}
        bucket = econ.get("econ_bucket", "neutral")
        if bucket == "opex":
            opex_net += float(econ.get("estimated_net_impact_usd", 0.0))
            saved += float(econ.get("estimated_saved_spend_usd", 0.0))
            n_opex += 1
        elif bucket == "capex_ticket":
            capex_tickets += float(econ.get("estimated_capex_ticket_usd", 0.0))
            n_capex += 1
    return {
        "simulated": True,
        "estimated_opex_net_usd": round(opex_net, 2),
        "estimated_saved_spend_usd_total": round(saved, 2),
        "estimated_capex_tickets_usd": round(capex_tickets, 2),
        "estimated_net_impact_usd_total": round(opex_net, 2),
        "n_rules": len(rules),
        "n_opex_rules": n_opex,
        "n_capex_tickets": n_capex,
    }
