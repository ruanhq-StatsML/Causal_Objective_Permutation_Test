#!/usr/bin/env python3
"""
Build bank-cash bridge from checked-in demo receipts + optional prod invoice CSV.

Usage:
  cd Python && python3 reconcile_fsds_bank_cash.py
  cd Python && python3 reconcile_fsds_bank_cash.py --invoice-csv ../data/openai_invoices.csv

CSV columns (minimal): date, vendor, amount_usd, project
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

from fsds_sot.bank_cash_bridge import (  # noqa: E402
    agent_receipt_to_ledger,
    federated_receipt_to_ledger,
    match_vendor_invoice,
    summarize_ledger,
    uplift_receipt_to_ledger,
)


def _load_json(name: str) -> Dict[str, Any]:
    p = ART / name
    return json.loads(p.read_text()) if p.is_file() else {}


def _collect_receipts() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    ma = _load_json("multi_agent_fsds_economics.json")
    if ma.get("impact_receipt"):
        out.append(ma["impact_receipt"])

    early = _load_json("multi_agent_debate_early_stop.json")
    if early.get("impact_receipt"):
        out.append(early["impact_receipt"])

    lg = _load_json("langgraph_fsds_hook.json")
    if lg.get("impact_receipt"):
        out.append(lg["impact_receipt"])

    uplift = _load_json("uplift_two_layer_benchmark.json")
    for run in uplift.get("runs") or []:
        for rec in run.get("impact_receipts") or []:
            out.append(rec)

    fed = _load_json("federated_legal_agent_blocks.json")
    for rec in fed.get("impact_receipts") or []:
        out.append(rec)

    return out


def _read_invoice_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "date": row.get("date"),
                    "vendor": row.get("vendor"),
                    "amount_usd": float(row.get("amount_usd", 0) or 0),
                    "project": row.get("project", ""),
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--invoice-csv", type=Path, default=None)
    ap.add_argument("--baseline-api-usd", type=float, default=None, help="Counterfactual API spend for period")
    ap.add_argument("--episodes", type=int, default=10_000, help="Volume for scaling per-ep receipt demo")
    args = ap.parse_args()

    ledger_rows = []
    for rec in _collect_receipts():
        pat = str(rec.get("pattern", ""))
        if pat == "uplift_two_batch":
            ledger_rows.extend(uplift_receipt_to_ledger(rec, simulated=True))
        elif pat.startswith("federated") or "fed_" in str(rec.get("episode_id", "")):
            ledger_rows.extend(federated_receipt_to_ledger(rec, simulated=True))
        else:
            ledger_rows.extend(
                agent_receipt_to_ledger(rec, episodes_in_settlement_period=args.episodes, simulated=True)
            )

    summary = summarize_ledger(ledger_rows, period_label=f"demo_scaled_{args.episodes}_episodes")
    payload: Dict[str, Any] = {
        "summary": summary.to_dict(),
        "formula_checking_balance_zh": (
            "期末运营户余额 − 反事实余额 ≈ Σ(cash_in) + Σ(retained 少付供应商) − Σ(cash_out 监控/Capex)"
        ),
        "formula_checking_balance_en": (
            "Ending checking − counterfactual ≈ Σ cash_in + Σ retained − Σ extra cash_out"
        ),
        "receipt_count": len(_collect_receipts()),
    }

    if args.invoice_csv and args.invoice_csv.is_file():
        inv = _read_invoice_csv(args.invoice_csv)
        baseline = args.baseline_api_usd
        if baseline is None:
            baseline = sum(r["amount_usd"] for r in inv) * 1.35  # placeholder if no baseline supplied
        payload["invoice_match"] = match_vendor_invoice(inv, baseline_usd=baseline)
        payload["invoice_match"]["note"] = "Replace baseline with pre-FSDS same-period spend for real reconcile"

    out = ART / "fsds_bank_cash_bridge.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out.relative_to(ROOT)}")
    s = payload["summary"]
    print(
        f"net_retained_in_checking_usd (scaled demo): ${s['net_retained_in_checking_usd']:,.2f} "
        f"(in ${s['cash_in_usd']:,.2f} in, ${s['cash_out_avoided_usd']:,.2f} avoided out, "
        f"${s['cash_out_extra_usd']:,.2f} extra out)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
