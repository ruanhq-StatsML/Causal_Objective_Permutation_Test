"""Load config/fsds_economics_assumptions.json for rollup and uplift sim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "config" / "fsds_economics_assumptions.json"


def load_assumptions(path: Path | None = None) -> Dict[str, Any]:
    p = path or DEFAULT_PATH
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def uplift_economics_kwargs(cfg: Dict[str, Any] | None = None) -> Dict[str, float]:
    c = cfg or load_assumptions()
    u = c.get("uplift") or {}
    return {
        "margin_usd_per_conversion": float(u.get("margin_usd_per_conversion", 48.0)),
        "treatment_cost_usd": float(u.get("treatment_cost_usd", 6.5)),
        "treat_rate": float(u.get("treat_rate", 0.35)),
    }


def federated_egress_savings_usd_per_year(cfg: Dict[str, Any] | None = None) -> float:
    """Illustrative: tenants × windows × (centralized − federated) egress cost."""
    c = cfg or load_assumptions()
    f = c.get("federated_infra") or {}
    price = float(f.get("egress_usd_per_gb", 0.05))
    windows = float(f.get("monitor_windows_per_year", 52))
    tenants = float(f.get("tenants", 100))
    mib = float(f.get("centralized_mib_per_window_per_tenant", 2.0))
    kib = float(f.get("federated_kib_per_window_per_tenant", 1.45))
    gb_saved = tenants * windows * max(0.0, (mib - kib / 1024.0)) / 1024.0
    return round(gb_saved * price, 2)
